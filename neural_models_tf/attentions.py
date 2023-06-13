
import numpy as np
import tensorflow as tf

from pyaromatics.stay_organized.utils import str2val
from anthe_official.neural_models_tf.tensor_chain.dense import TCDense


class ScaledDotProductAttention(tf.keras.layers.Layer):
    def __init__(self, d_h):
        super().__init__()
        self.d_h = d_h
        self.softmax = tf.nn.softmax

    def call(self, query, key, value, mask=None):
        matmul_q_and_transposed_k = tf.matmul(query, key, transpose_b=True)
        scale = tf.sqrt(tf.cast(self.d_h, dtype=tf.float32))
        scaled_attention_score = matmul_q_and_transposed_k / scale
        if mask is not None:
            scaled_attention_score += (mask * -1e9)

        attention_weight = self.softmax(scaled_attention_score)

        return tf.matmul(attention_weight, value), attention_weight


def gatingmech(x, y, z, wq=None, wk=None, wv=None):
    x = x * tf.nn.sigmoid(wq(y))
    x = wv(x)
    z = wk(z)
    return x, y, z


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, attention_head_count, d_model, comments=''):
        super(MultiHeadAttention, self).__init__()

        # model hyper parameter variables
        self.comments = comments
        self.attention_head_count = attention_head_count

        if d_model % attention_head_count != 0:
            raise ValueError(
                "d_model({}) % attention_head_count({}) is not zero.d_model must be multiple of attention_head_count.".format(
                    d_model, attention_head_count
                )
            )

        self.d_h = d_model // attention_head_count

        if 'nopreatt' in comments:
            w_query = lambda x: x
            w_key = lambda x: x
            w_value = lambda x: x

        elif 'sharedqkv' in comments:
            dense = tf.keras.layers.Dense(d_model)
            w_query, w_key, w_value = [dense] * 3

        elif 'tclayer' in comments or 'tcpreatt' in comments:
            tcr = str2val(comments, 'tcpreatt', float, default=.2)
            tcr = str2val(comments, 'tclayer', float, default=tcr)
            tclength = str2val(comments, 'tclength', int, default=3)
            tclength = str2val(comments, 'tclayerlength', int, default=tclength)

            w_query = TCDense(d_model, tc_length=tclength, ratio=tcr)
            w_key = TCDense(d_model, tc_length=tclength, ratio=tcr)
            w_value = TCDense(d_model, tc_length=tclength, ratio=tcr)

        else:
            w_query = tf.keras.layers.Dense(d_model)
            w_key = tf.keras.layers.Dense(d_model)
            w_value = tf.keras.layers.Dense(d_model)

        self.w_query = w_query
        self.w_key = w_key
        self.w_value = w_value

        self.scaled_dot_product = ScaledDotProductAttention(self.d_h)
        self.ff = tf.keras.layers.Dense(d_model)

        self.d_model = d_model

        qkv_order = str2val(comments, 'gateattention', str, default='kqv')
        assert all([k in qkv_order for k in 'qkv'])

        order = np.argsort(list(qkv_order))
        self.mixer = lambda x: [x[i] for i in order]

        mixed = self.mixer(list('abc'))
        unorder = np.argsort(list(mixed))
        self.unmixer = lambda x: [x[i] for i in unorder]

    def call(self, inputs):
        query, key, value, mask = inputs
        batch_size = tf.shape(query)[0]

        if 'gateattention' in self.comments:
            key, query, value = self.mixer([key, query, value])
            value, key, query = gatingmech(value, key, query, wq=self.w_query, wk=self.w_key, wv=self.w_value)
            key, query, value = self.unmixer([key, query, value])

        else:
            query = self.w_query(query)
            key = self.w_key(key)
            value = self.w_value(value)

        query = self.split_head(query, batch_size)
        key = self.split_head(key, batch_size)
        value = self.split_head(value, batch_size)

        output, attention = self.scaled_dot_product(query, key, value, mask)
        output = self.concat_head(output, batch_size)
        output = self.ff(output)

        return output, attention

    def split_head(self, tensor, batch_size):
        # inputs tensor: (batch_size, seq_len, d_model)
        return tf.transpose(
            tf.reshape(
                tensor,
                (batch_size, -1, self.attention_head_count, self.d_h)
                # tensor: (batch_size, seq_len_splited, attention_head_count, d_h)
            ),
            [0, 2, 1, 3]
            # tensor: (batch_size, attention_head_count, seq_len_splited, d_h)
        )

    def concat_head(self, tensor, batch_size):
        return tf.reshape(
            tf.transpose(tensor, [0, 2, 1, 3]),
            (batch_size, -1, self.attention_head_count * self.d_h)
        )
