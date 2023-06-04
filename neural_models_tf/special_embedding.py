import tensorflow as tf
from pyaromatics.stay_organized.utils import str2val
from anthe_official.neural_models_tf.tensor_chain.convolutions import TCConv1D

from anthe_official.neural_models_tf.tensor_chain.embedding import TCEmbedding


def positional_encoding(max_len, d_model):
    depth = d_model / 2

    positions = tf.expand_dims(tf.range(max_len), 1)  # (seq, 1)
    depths = tf.expand_dims(tf.range(depth) / depth, 0)  # (1, depth)

    depths = tf.cast(depths, dtype=tf.float32)
    positions = tf.cast(positions, dtype=tf.float32)

    angle_rates = 1 / (10000 ** depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)

    pos_encoding = tf.concat([tf.sin(angle_rads), tf.cos(angle_rads)], axis=-1)

    return tf.cast(pos_encoding, dtype=tf.float32)


def angle(pos, index, d_model):
    pos = tf.cast(pos, tf.float32)
    return pos / tf.pow(10000., tf.cast((index - index % 2) / d_model, tf.float32))


class EmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model, **kwargs):
        # model hyper parameter variables
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)

    def call(self, sequences, **kwargs):
        max_sequence_len = tf.shape(sequences)[1]

        output = self.embedding(sequences)
        output = output * tf.sqrt(tf.cast(self.d_model, dtype=tf.float32))
        pos = positional_encoding(max_sequence_len, self.d_model)

        output = output + pos

        return output


class SoftPOS(tf.keras.layers.Layer):
    def __init__(self, add_units, n_subpos=3, repeat_subpos=2, initializer='orthogonal', **kwargs):
        super().__init__(**kwargs)

        self.add_units = add_units
        self.n_subpos = n_subpos
        self.repeat_subpos = repeat_subpos
        self.initializer = initializer

    def build(self, input_shape):
        if self.n_subpos > 0:
            self.spos = []
            for i in range(self.repeat_subpos):
                spos = self.add_weight(
                    f"spos_{i}", shape=[self.n_subpos, self.add_units], dtype="float32",
                    initializer=self.initializer
                )
                self.spos.append(spos)

        self.built = True

    def call(self, inputs):
        x = inputs
        emb = x
        if self.n_subpos > 0:
            for i, spos in enumerate(self.spos):
                spos_select = tf.nn.softmax(emb[..., i * self.n_subpos:(i + 1) * self.n_subpos])
                _spos = spos_select @ spos
                x = tf.concat([x, _spos], axis=-1, name='concat')

        return x

    def get_config(self):
        config = {
            'add_units': self.add_units, 'n_subpos': self.n_subpos, 'repeat_subpos': self.repeat_subpos,
            'initializer': tf.keras.initializers.serialize(tf.keras.initializers.get(self.embeddings_initializer)),
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class HSoftPOS(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embed_dim, n_layers=2, tcembr=None, tcconvr=None, tclength=2, **kwargs):
        super().__init__(**kwargs)

        assert tcembr is None or isinstance(tcembr, float)
        assert tcconvr is None or isinstance(tcconvr, float)

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.tcembr = tcembr
        self.tcconvr = tcconvr
        self.tclength = tclength

        local_d = int(embed_dim / 2 / n_layers)
        embd_d = embed_dim - local_d * (2 * n_layers - 1)

        if tcembr is None:
            self.emb = EmbeddingLayer(vocab_size, embd_d)
        else:
            self.emb = TCEmbedding(vocab_size, embd_d, ratio=tcembr, tc_length=tclength)

        if tcconvr is None:
            conv1d = tf.keras.layers.Conv1D
        else:
            conv1d = lambda *args, **kwargs: TCConv1D(*args, **kwargs, ratio=tcconvr, tc_length=tclength)

        self.spos, self.convs = [], []
        for i in range(n_layers):
            self.spos.append(SoftPOS(local_d, n_subpos=local_d, repeat_subpos=1))
            if i < n_layers - 1: self.convs.append(conv1d(local_d, 3, padding='causal', dilation_rate=2 ** i))

    def call(self, inputs):
        x = self.emb(inputs)

        xs = [x]
        for conv in self.convs:
            x = conv(x)
            xs.append(x)

        ys = []
        for x, spos in zip(xs, self.spos):
            y = spos(x)
            ys.append(y)

        x = tf.concat(ys, axis=-1)

        return x

    def get_config(self):

        config = {
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'n_layers': self.n_layers,
            'tcembr': self.tcembr,
            'tcconvr': self.tcconvr,
            'tclength': self.tclength,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


def select_embedding_type(self, comments, inputs_vocab_size, target_vocab_size, d_model):
    if not 'tcemb' in comments:
        emb = lambda vocab, embd: EmbeddingLayer(vocab, embd)
    else:
        tclength = str2val(comments, 'tclength', int, default=3)
        tcr = str2val(comments, 'tcemb', float, default=.2)
        emb = lambda vocab, embd: TCEmbedding(vocab, embd, ratio=tcr, tc_length=tclength)

    if 'hsoftpos' in comments:
        n = str2val(comments, 'hsoftpos', output_type=int, default=3)

        local_d = int(d_model / 2 / n)
        embd_d = d_model - local_d * (2 * n - 1)

        if not 'tcemb' in comments:
            eemb = EmbeddingLayer(inputs_vocab_size, embd_d)
            demb = EmbeddingLayer(target_vocab_size, embd_d)
        else:
            tcr = str2val(comments, 'tcemb', float, default=.2)

            tclength = str2val(comments, 'tclength', int, default=3)
            eemb = TCEmbedding(inputs_vocab_size, embd_d, ratio=tcr, tc_length=tclength)
            demb = TCEmbedding(target_vocab_size, embd_d, ratio=tcr, tc_length=tclength)

        if 'tcconv' in comments:
            tcr = str2val(comments, 'tcconv', float, default=.2)
            tclength = str2val(comments, 'tclength', int, default=3)

            conv1d = lambda *args, **kwargs: TCConv1D(*args, **kwargs, ratio=tcr, tc_length=tclength)
        else:
            conv1d = tf.keras.layers.Conv1D

        espos, econvs = [], []
        for i in range(n):
            espos.append(SoftPOS(local_d, n_subpos=local_d, repeat_subpos=1))
            if i < n - 1: econvs.append(conv1d(local_d, 3, padding='causal', dilation_rate=2 ** i))

        dspos, dconvs = [], []
        for i in range(n):
            dspos.append(SoftPOS(local_d, n_subpos=local_d, repeat_subpos=1))
            if i < n - 1: dconvs.append(conv1d(local_d, 3, padding='causal', dilation_rate=2 ** i))

        if 'projectoutput' in comments:
            projector = conv1d(embd_d, 1, padding='causal')

        def code(input, emb, convs, sposs, mode):
            if mode == 'embedding':
                x = emb(input)
                xs = [x]
                for conv in convs:
                    x = conv(x)
                    xs.append(x)

                ys = []
                for x, spos in zip(xs, sposs):
                    y = spos(x)
                    ys.append(y)

                x = tf.concat(ys, axis=-1)

            elif mode == 'projection':
                input = projector(input)
                x = emb(input, mode='projection')

            return x

        self.encoder_embedding_layer = lambda x, mode='embedding': code(x, eemb, econvs, espos, mode)
        self.decoder_embedding_layer = lambda x, mode='embedding': code(x, demb, dconvs, dspos, mode)

    elif 'layerhspos' in comments:
        print('nice!')
        n = str2val(comments, 'layerhspos', output_type=int, default=3)

        tclength = str2val(comments, 'tclength', int, default=2)
        tcembr, tcconvr = None, None
        if 'tcemb' in comments:
            tcembr = str2val(comments, 'tcemb', float, default=.2)

        if 'tcconv' in comments:
            tcconvr = str2val(comments, 'tcconv', float, default=.2)

        self.encoder_embedding_layer = HSoftPOS(
            inputs_vocab_size, d_model, n_layers=n, tcembr=tcembr, tcconvr=tcconvr, tclength=tclength
        )
        self.decoder_embedding_layer = HSoftPOS(
            target_vocab_size, d_model, n_layers=n, tcembr=tcembr, tcconvr=tcconvr, tclength=tclength
        )
    else:
        self.encoder_embedding_layer = emb(inputs_vocab_size, d_model)
        self.decoder_embedding_layer = emb(inputs_vocab_size, d_model)
