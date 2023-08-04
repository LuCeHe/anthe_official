import os

import tensorflow as tf

from pyaromatics.stay_organized.utils import str2val
from anthe_official.neural_models_tf.attentions import MultiHeadAttention
from anthe_official.neural_models_tf.helper_layers import ProjectionLayer
from anthe_official.neural_models_tf.tensor_chain.dense import TCDense
from anthe_official.neural_models_tf.special_embedding import select_embedding_type

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# class Transformer(tf.keras.Model):
class Transformer:
    def __init__(self,
                 inputs_vocab_size,
                 target_vocab_size,
                 encoder_count,
                 decoder_count,
                 attention_head_count,
                 d_model,
                 d_point_wise_ff,
                 dropout_prob, comments=''):
        super(Transformer, self).__init__()

        # model hyper parameter variables
        self.inputs_vocab_size = inputs_vocab_size
        self.target_vocab_size = target_vocab_size
        self.encoder_count = encoder_count
        self.decoder_count = decoder_count
        self.attention_head_count = attention_head_count
        self.d_model = d_model
        self.d_point_wise_ff = d_point_wise_ff
        self.dropout_prob = dropout_prob
        self.comments = comments

        self.encoder_embedding_dropout = tf.keras.layers.Dropout(self.dropout_prob)
        self.decoder_embedding_dropout = tf.keras.layers.Dropout(self.dropout_prob)

        select_embedding_type(self, self.comments, self.inputs_vocab_size, self.target_vocab_size, self.d_model)

        if 'sameemb' in self.comments:
            self.decoder_embedding_layer = self.encoder_embedding_layer

        self.encoder_layers = [
            EncoderLayer(
                self.attention_head_count, self.d_model, self.d_point_wise_ff, self.dropout_prob, comments=self.comments)
            for _ in range(self.encoder_count)
        ]

        self.decoder_layers = [
            DecoderLayer(
                self.attention_head_count, self.d_model, self.d_point_wise_ff, self.dropout_prob, comments=self.comments)
            for _ in range(self.decoder_count)
        ]

        if 'projectoutput' in self.comments:
            self.decoder_embedding_layer.embedding.build((1,))
            embm = tf.transpose(self.decoder_embedding_layer.embedding.embeddings)
            self.output_projection = ProjectionLayer()
            self.output_projection.project_matrix = embm
            # self.output_projection = lambda x: self.decoder_embedding_layer(x, mode='projection')

        elif 'mpolastlayer' in self.comments:
            tcr = str2val(self.comments, 'mpolastlayer', float, default=.2)
            tclength = str2val(self.comments, 'tclength', int, default=3)
            tclength = str2val(self.comments, 'mpolastlayerlength', int, default=tclength)
            self.output_projection = TCDense(self.target_vocab_size, length=tclength, ratio=tcr)

        else:
            self.output_projection = tf.keras.layers.Dense(self.target_vocab_size)
            
            
    def get_config(self):

        config = {
            'inputs_vocab_size' : self.inputs_vocab_size,
            'target_vocab_size' : self.target_vocab_size,
            'encoder_count' : self.encoder_count,
            'decoder_count' : self.decoder_count,
            'attention_head_count' : self.attention_head_count,
            'd_model' : self.d_model,
            'd_point_wise_ff' : self.d_point_wise_ff,
            'dropout_prob' : self.dropout_prob,
            'comments' : self.comments,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def __call__(self, inputs, training=False):
        source, target, inputs_padding_mask, look_ahead_mask, target_padding_mask = inputs

        source = self.encoder_embedding_layer(source)
        encoder_tensor = self.encoder_embedding_dropout(source)
        target = self.decoder_embedding_layer(target)
        decoder_tensor = self.decoder_embedding_dropout(target)

        for i in range(self.encoder_count):
            encoder_tensor, _ = self.encoder_layers[i](encoder_tensor, inputs_padding_mask)

        for i in range(self.decoder_count):
            decoder_tensor, _, _ = self.decoder_layers[i](decoder_tensor, encoder_tensor, look_ahead_mask,
                                                          target_padding_mask)

        return self.output_projection(decoder_tensor)


Anthe = lambda *args, **kwargs: Transformer(
    *args, **kwargs,
    comments='geglu_gateattention_hsoftpos:2_tcffn:.005_tcpreatt:.07_tclength:2'
)


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, attention_head_count, d_model, d_point_wise_ff, dropout_prob, comments=''):
        super(EncoderLayer, self).__init__()
        # model hyper parameter variables
        self.attention_head_count = attention_head_count
        self.d_model = d_model
        self.d_point_wise_ff = d_point_wise_ff
        self.dropout_prob = dropout_prob
        self.comments = comments

        self.attention = MultiHeadAttention(self.attention_head_count, self.d_model, self.comments)

        self.dropout_1 = tf.keras.layers.Dropout(self.dropout_prob)
        self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        if 'geglu' in self.comments:
            self.position_wise_feed_forward_layer = GEGLU(
                self.d_point_wise_ff, self.d_model, self.comments
            )
        else:
            self.position_wise_feed_forward_layer = PositionWiseFeedForwardLayer(
                self.d_point_wise_ff, self.d_model, self.comments
            )
        self.dropout_2 = tf.keras.layers.Dropout(self.dropout_prob)
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
    def get_config(self):

        config = {
            'attention_head_count' : self.attention_head_count,
            'd_model' : self.d_model,
            'd_point_wise_ff' : self.d_point_wise_ff,
            'dropout_prob' : self.dropout_prob,
            'comments' : self.comments,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs, mask):
        output, attention = self.attention([inputs, inputs, inputs, mask])

        output = self.dropout_1(output)
        output = self.layer_norm_1(tf.add(inputs, output))  # residual network
        output_temp = output

        output = self.position_wise_feed_forward_layer(output)
        output = self.dropout_2(output)
        output = self.layer_norm_2(tf.add(output_temp, output))  # correct

        return output, attention


AntheEncoderBlock = lambda attention_head_count, d_model, d_point_wise_ff, dropout_prob: \
    EncoderLayer(
        attention_head_count, d_model, d_point_wise_ff, dropout_prob,
        comments='geglu_gateattention_hsoftpos:2_tcffn:.005_tcpreatt:.07_tclength:2'
    )


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, attention_head_count, d_model, d_point_wise_ff, dropout_prob, comments=''):
        super(DecoderLayer, self).__init__()
        
        # model hyper parameter variables
        self.attention_head_count = attention_head_count
        self.d_model = d_model
        self.d_point_wise_ff = d_point_wise_ff
        self.dropout_prob = dropout_prob
        self.comments = comments

        self.attention = MultiHeadAttention(self.attention_head_count, self.d_model, self.comments)

        self.dropout_1 = tf.keras.layers.Dropout(self.dropout_prob)
        self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.conditioned_attention = MultiHeadAttention(self.attention_head_count, self.d_model, self.comments)

        self.dropout_2 = tf.keras.layers.Dropout(self.dropout_prob)
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        if 'geglu' in self.comments:
            self.position_wise_feed_forward_layer = GEGLU(
                self.d_point_wise_ff, self.d_model, self.comments
            )
        else:
            self.position_wise_feed_forward_layer = PositionWiseFeedForwardLayer(
                self.d_point_wise_ff, self.d_model, self.comments
            )
        self.dropout_3 = tf.keras.layers.Dropout(self.dropout_prob)
        self.layer_norm_3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def get_config(self):

        config = {
            'attention_head_count' : self.attention_head_count,
            'd_model' : self.d_model,
            'd_point_wise_ff' : self.d_point_wise_ff,
            'dropout_prob' : self.dropout_prob,
            'comments' : self.comments,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
        

    def call(self, decoder_inputs, encoder_output, look_ahead_mask, padding_mask):
        output, attention_1 = self.attention([decoder_inputs, decoder_inputs, decoder_inputs, look_ahead_mask])
        output = self.dropout_1(output)
        query = self.layer_norm_1(tf.add(decoder_inputs, output))  # residual network
        output, attention_2 = self.conditioned_attention([query, encoder_output, encoder_output, padding_mask])
        output = self.dropout_2(output)
        encoder_decoder_attention_output = self.layer_norm_2(tf.add(output, query))

        output = self.position_wise_feed_forward_layer(encoder_decoder_attention_output)
        output = self.dropout_3(output)
        output = self.layer_norm_3(tf.add(encoder_decoder_attention_output, output))  # residual network

        return output, attention_1, attention_2


AntheDecoderBlock = lambda attention_head_count, d_model, d_point_wise_ff, dropout_prob: \
    DecoderLayer(
        attention_head_count, d_model, d_point_wise_ff, dropout_prob,
        comments='geglu_gateattention_hsoftpos:2_tcffn:.005_tcpreatt:.07_tclength:2'
    )


class PositionWiseFeedForwardLayer(tf.keras.layers.Layer):
    def __init__(self, d_point_wise_ff, d_model, comments=''):
        super(PositionWiseFeedForwardLayer, self).__init__()
        
        # model hyper parameter variables
        self.d_model = d_model
        self.d_point_wise_ff = d_point_wise_ff
        self.comments = comments

        if 'noffn' in self.comments:
            self.w_1 = lambda x: x
            self.w_2 = lambda x: x

        elif 'tclayer' in self.comments or 'tcffn' in self.comments:
            tcr = str2val(self.comments, 'tcffn', float, default=.2)
            tcr = str2val(self.comments, 'tclayer', float, default=tcr)
            tclength = str2val(self.comments, 'tclength', int, default=3)
            tclength = str2val(self.comments, 'tclayerlength', int, default=tclength)

            self.w_1 = TCDense(self.d_point_wise_ff, length=tclength, ratio=tcr)
            self.w_2 = TCDense(self.d_model, length=tclength, ratio=tcr)
        else:
            self.w_1 = tf.keras.layers.Dense(self.d_point_wise_ff)
            self.w_2 = tf.keras.layers.Dense(self.d_model)

    def get_config(self):

        config = {
            'd_model' : self.d_model,
            'd_point_wise_ff' : self.d_point_wise_ff,
            'comments' : self.comments,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
            
    def call(self, inputs):
        inputs = self.w_1(inputs)
        inputs = tf.nn.relu(inputs)
        return self.w_2(inputs)


class GEGLU(tf.keras.layers.Layer):
    def __init__(self, d_point_wise_ff, d_model, comments=''):
        super().__init__()
        # https://arxiv.org/pdf/2002.05202.pdf
        
        # model hyper parameter variables
        self.d_model = d_model
        self.d_point_wise_ff = d_point_wise_ff
        self.comments = comments

        d_point_wise_ff = 2 * self.d_point_wise_ff // 3

        if 'noffn' in self.comments:
            self.w_1 = lambda x: x
            self.w_3 = lambda x: x
            self.w_2 = lambda x: x

        elif 'tclayer' in self.comments or 'tcffn' in self.comments:
            tcr = str2val(self.comments, 'tcffn', float, default=.2)
            tcr = str2val(self.comments, 'tclayer', float, default=tcr)
            tclength = str2val(self.comments, 'tclength', int, default=3)
            tclength = str2val(self.comments, 'tclayerlength', int, default=tclength)

            self.w_1 = TCDense(d_point_wise_ff, tc_length=tclength, ratio=tcr)
            self.w_3 = TCDense(d_point_wise_ff, tc_length=tclength, ratio=tcr)
            self.w_2 = TCDense(self.d_model, tc_length=tclength, ratio=tcr)
        else:
            self.w_1 = tf.keras.layers.Dense(d_point_wise_ff)
            self.w_3 = tf.keras.layers.Dense(d_point_wise_ff)
            self.w_2 = tf.keras.layers.Dense(self.d_model)

        swish = lambda x: x * tf.nn.sigmoid(x)
        self.activation = swish

    def get_config(self):

        config = {
            'd_model' : self.d_model,
            'd_point_wise_ff' : self.d_point_wise_ff,
            'comments' : self.comments,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
    def call(self, inputs):
        x1 = self.w_1(inputs)
        x3 = self.w_3(inputs)
        x2 = self.activation(x1) * x3
        return self.w_2(x2)


def build_model(
        inputs_timesteps,
        target_timesteps,
        inputs_vocab_size,
        target_vocab_size,
        encoder_count,
        decoder_count,
        attention_head_count,
        d_model,
        d_point_wise_ff,
        dropout_prob, comments=''
):
    transformer = Transformer(
        inputs_vocab_size=inputs_vocab_size,
        target_vocab_size=target_vocab_size,
        encoder_count=encoder_count,
        decoder_count=decoder_count,
        attention_head_count=attention_head_count,
        d_model=d_model,
        d_point_wise_ff=d_point_wise_ff,
        dropout_prob=dropout_prob,
        comments=comments
    )

    inputs_layer = tf.keras.layers.Input((None,))
    target_layer = tf.keras.layers.Input((None,))
    inputs_padding_mask = tf.keras.layers.Input((1, 1, None,))
    look_ahead_mask = tf.keras.layers.Input((1, None, None,))
    target_padding_mask = tf.keras.layers.Input((1, 1, None,))

    output = transformer([inputs_layer, target_layer, inputs_padding_mask, look_ahead_mask, target_padding_mask])
    # output = transformer(inputs_layer, target_layer, inputs_padding_mask, look_ahead_mask, target_padding_mask)

    model = tf.keras.models.Model(
        [inputs_layer, target_layer, inputs_padding_mask, look_ahead_mask, target_padding_mask],
        output
    )

    model.summary()
    return model
