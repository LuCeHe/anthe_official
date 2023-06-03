import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyaromatics.stay_organized.utils import str2val
from anthe_official.neural_models_pt.attentions import MultiHeadAttention
from anthe_official.neural_models_pt.helper_layers import ProjectionLayer
from anthe_official.neural_models_pt.tensor_chain.dense import TCDense
from anthe_official.neural_models_pt.special_embedding import select_embedding_type

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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
        self.encoder_count = encoder_count
        self.decoder_count = decoder_count
        self.attention_head_count = attention_head_count
        self.d_model = d_model
        self.d_point_wise_ff = d_point_wise_ff
        self.dropout_prob = dropout_prob
        self.comments = comments

        self.encoder_embedding_dropout = nn.Dropout(dropout_prob)
        self.decoder_embedding_dropout = nn.Dropout(dropout_prob)

        select_embedding_type(self, comments, inputs_vocab_size, target_vocab_size, d_model)

        if 'sameemb' in comments:
            self.decoder_embedding_layer = self.encoder_embedding_layer

        self.encoder_layers = [
            EncoderLayer(
                attention_head_count, d_model, d_point_wise_ff, dropout_prob, comments=comments)
            for _ in range(encoder_count)
        ]

        self.decoder_layers = [
            DecoderLayer(
                attention_head_count, d_model, d_point_wise_ff, dropout_prob, comments=comments)
            for _ in range(decoder_count)
        ]

        if 'projectoutput' in self.comments:
            embm = self.decoder_embedding_layer.embedding.weight.data.transpose(0, 1)
            self.output_projection = ProjectionLayer()
            self.output_projection.project_matrix.weight = nn.Parameter(embm)

        elif 'mpolastlayer' in self.comments:
            tcr = str2val(comments, 'mpolastlayer', float, default=.2)
            tclength = str2val(comments, 'tclength', int, default=3)
            tclength = str2val(comments, 'mpolastlayerlength', int, default=tclength)
            self.output_projection = TCDense(target_vocab_size, length=tclength, ratio=tcr)

        else:
            self.output_projection = nn.Linear(d_model, target_vocab_size)

    def __call__(self, inputs):
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


class EncoderLayer(nn.Module):
    def __init__(self, attention_head_count, d_model, d_point_wise_ff, dropout_prob, comments=''):
        super(EncoderLayer, self).__init__()
        self.comments = comments
        # model hyper parameter variables
        self.attention_head_count = attention_head_count
        self.d_model = d_model
        self.d_point_wise_ff = d_point_wise_ff
        self.dropout_prob = dropout_prob

        self.attention = MultiHeadAttention(attention_head_count, d_model, comments)

        self.dropout_1 = nn.Dropout(dropout_prob)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)

        if 'geglu' in comments:
            self.position_wise_feed_forward_layer = GEGLU(d_point_wise_ff, d_model, comments)
        else:
            self.position_wise_feed_forward_layer = PositionWiseFeedForwardLayer(d_point_wise_ff, d_model, comments)

        self.dropout_2 = nn.Dropout(dropout_prob)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, inputs, mask):
        output, attention = self.attention([inputs, inputs, inputs, mask])
        output = self.dropout_1(output)
        output = self.layer_norm_1(inputs + output)  # residual network
        output_temp = output

        output = self.position_wise_feed_forward_layer(output)
        output = self.dropout_2(output)
        output = self.layer_norm_2(output_temp + output)  # residual network

        return output, attention


AntheEncoderBlock = lambda attention_head_count, d_model, d_point_wise_ff, dropout_prob: \
    EncoderLayer(
        attention_head_count, d_model, d_point_wise_ff, dropout_prob,
        comments='geglu_gateattention_hsoftpos:2_tcffn:.005_tcpreatt:.07_tclength:2'
    )


class DecoderLayer(nn.Module):
    def __init__(self, attention_head_count, d_model, d_point_wise_ff, dropout_prob, comments=''):
        super(DecoderLayer, self).__init__()
        self.comments = comments
        # model hyper parameter variables
        self.attention_head_count = attention_head_count
        self.d_model = d_model
        self.d_point_wise_ff = d_point_wise_ff
        self.dropout_prob = dropout_prob

        self.attention = MultiHeadAttention(attention_head_count, d_model, comments)

        self.dropout_1 = nn.Dropout(dropout_prob)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)

        self.conditioned_attention = MultiHeadAttention(attention_head_count, d_model, comments)

        self.dropout_2 = nn.Dropout(dropout_prob)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)

        if 'geglu' in comments:
            self.position_wise_feed_forward_layer = GEGLU(d_point_wise_ff, d_model, comments)
        else:
            self.position_wise_feed_forward_layer = PositionWiseFeedForwardLayer(d_point_wise_ff, d_model, comments)
        self.dropout_3 = nn.Dropout(dropout_prob)
        self.layer_norm_3 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, decoder_inputs, encoder_output, look_ahead_mask, padding_mask):
        output, attention_1 = self.attention(decoder_inputs, decoder_inputs, decoder_inputs, look_ahead_mask)
        output = self.dropout_1(output)
        query = self.layer_norm_1(decoder_inputs + output)  # residual network
        output, attention_2 = self.conditioned_attention(query, encoder_output, encoder_output, padding_mask)
        output = self.dropout_2(output)
        encoder_decoder_attention_output = self.layer_norm_2(query + output)

        output = self.position_wise_feed_forward_layer(encoder_decoder_attention_output)
        output = self.dropout_3(output)
        output = self.layer_norm_3(encoder_decoder_attention_output + output)  # residual network

        return output, attention_1, attention_2


AntheDecoderBlock = lambda attention_head_count, d_model, d_point_wise_ff, dropout_prob: \
    EncoderLayer(
        attention_head_count, d_model, d_point_wise_ff, dropout_prob,
        comments='geglu_gateattention_hsoftpos:2_tcffn:.005_tcpreatt:.07_tclength:2'
    )


class PositionWiseFeedForwardLayer(nn.Module):
    def __init__(self, d_point_wise_ff, d_model, comments=''):
        super(PositionWiseFeedForwardLayer, self).__init__()

        if 'noffn' in comments:
            self.w_1 = lambda x: x
            self.w_2 = lambda x: x

        elif 'tclayer' in comments or 'tcffn' in comments:
            tcr = str2val(comments, 'tcffn', float, default=.2)
            tcr = str2val(comments, 'tclayer', float, default=tcr)
            tclength = str2val(comments, 'tclength', int, default=3)
            tclength = str2val(comments, 'tclayerlength', int, default=tclength)

            self.w_1 = TCDense(d_point_wise_ff, length=tclength, ratio=tcr)
            self.w_2 = TCDense(d_model, length=tclength, ratio=tcr)
        else:
            self.w_1 = nn.Linear(d_model, d_point_wise_ff)
            self.w_2 = nn.Linear(d_point_wise_ff, d_model)

    def forward(self, inputs):
        inputs = self.w_1(inputs)
        inputs = nn.functional.relu(inputs)
        return self.w_2(inputs)


class GEGLU(nn.Module):
    def __init__(self, d_point_wise_ff, d_model, comments=''):
        super(GEGLU, self).__init__()
        # https://arxiv.org/pdf/2002.05202.pdf

        d_point_wise_ff = 2 * d_point_wise_ff // 3

        if 'noffn' in comments:
            self.w_1 = lambda x: x
            self.w_3 = lambda x: x
            self.w_2 = lambda x: x

        elif 'tclayer' in comments or 'tcffn' in comments:
            tcr = str2val(comments, 'tcffn', float, default=.2)
            tcr = str2val(comments, 'tclayer', float, default=tcr)
            tclength = str2val(comments, 'tclength', int, default=3)
            tclength = str2val(comments, 'tclayerlength', int, default=tclength)

            self.w_1 = TCDense(d_point_wise_ff, length=tclength, ratio=tcr)
            self.w_3 = TCDense(d_point_wise_ff, length=tclength, ratio=tcr)
            self.w_2 = TCDense(d_model, length=tclength, ratio=tcr)
        else:
            self.w_1 = nn.Linear(d_model, d_point_wise_ff)
            self.w_3 = nn.Linear(d_model, d_point_wise_ff)
            self.w_2 = nn.Linear(d_point_wise_ff, d_model)

        self.activation = nn.SiLU()

    def forward(self, inputs):
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

    inputs_layer = nn.Identity()
    target_layer = nn.Identity()
    inputs_padding_mask = nn.Identity()
    look_ahead_mask = nn.Identity()
    target_padding_mask = nn.Identity()

    output = transformer([inputs_layer, target_layer, inputs_padding_mask, look_ahead_mask, target_padding_mask])

    model = nn.ModuleDict({
        'inputs_layer': inputs_layer,
        'target_layer': target_layer,
        'inputs_padding_mask': inputs_padding_mask,
        'look_ahead_mask': look_ahead_mask,
        'target_padding_mask': target_padding_mask,
        'transformer': transformer,
        'output': output
    })

    return model
