import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pyaromatics.stay_organized.utils import str2val
from anthe_official.neural_models_pt.tensor_chain.dense import TCDense


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_h):
        super(ScaledDotProductAttention, self).__init__()
        self.d_h = d_h

    def forward(self, query, key, value, mask=None):
        matmul_q_and_transposed_k = torch.matmul(query, key.transpose(-2, -1))
        scale = torch.sqrt(torch.tensor(self.d_h, dtype=torch.float32))
        scaled_attention_score = matmul_q_and_transposed_k / scale
        if mask is not None:
            scaled_attention_score += (mask * -1e9)

        attention_weight = F.softmax(scaled_attention_score, dim=-1)

        return torch.matmul(attention_weight, value), attention_weight


def gatingmech(x, y, z, wq=None, wk=None, wv=None):
    # print(y)
    x = x * torch.sigmoid(wq(y))
    x = wv(x)
    z = wk(z)
    return x, y, z


class MultiHeadAttention(nn.Module):
    def __init__(self, attention_head_count, d_model, comments='', axis=-1):
        super(MultiHeadAttention, self).__init__()

        self.comments = comments
        self.attention_head_count = attention_head_count
        self.axis = axis

        if d_model % attention_head_count != 0:
            raise ValueError(
                "d_model({}) % attention_head_count({}) is not zero. d_model must be multiple of attention_head_count.".format(
                    d_model, attention_head_count))

        self.d_h = d_model // attention_head_count

        if 'nopreatt' in comments:
            self.w_query = lambda x: x
            self.w_key = lambda x: x
            self.w_value = lambda x: x

        elif 'sharedqkv' in comments:
            w = nn.Linear(d_model, d_model)
            self.w_query = w
            self.w_key = w
            self.w_value = w

        elif 'tclayer' in comments or 'tcpreatt' in comments:
            tcr = str2val(comments, 'tcpreatt', float, default=.2)
            tcr = str2val(comments, 'tclayer', float, default=tcr)
            tclength = str2val(comments, 'tclength', int, default=3)
            tclength = str2val(comments, 'tclayerlength', int, default=tclength)

            self.w_query = TCDense(d_model, d_model, tc_length=tclength, ratio=tcr)
            self.w_key = TCDense(d_model, d_model, tc_length=tclength, ratio=tcr)
            self.w_value = TCDense(d_model, d_model, tc_length=tclength, ratio=tcr)

        else:
            self.w_query = nn.Linear(d_model, d_model)
            self.w_key = nn.Linear(d_model, d_model)
            self.w_value = nn.Linear(d_model, d_model)

        self.scaled_dot_product = ScaledDotProductAttention(self.d_h)
        self.ff = nn.Linear(d_model, d_model)

        self.d_model = d_model

        qkv_order = str2val(comments, 'gateattention', str, default='kqv')
        assert all([k in qkv_order for k in 'qkv'])

        order = np.argsort(list(qkv_order))
        self.mixer = lambda x: [x[i] for i in order]

        mixed = self.mixer(list('abc'))
        unorder = np.argsort(list(mixed))
        self.unmixer = lambda x: [x[i] for i in unorder]

    def forward(self, query, key, value, mask=None, layer_past=None):
        batch_size = query.size(0)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if self.axis == 1:
            query = torch.transpose(query, 1, 2)
            key = torch.transpose(key, 1, 2)
            value = torch.transpose(value, 1, 2)

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

        if self.axis == 1:
            output = torch.transpose(output, 1, 2)
            attention = torch.transpose(attention, 1, 2)

        return output, attention

    def split_head(self, tensor, batch_size):
        # inputs tensor: (batch_size, seq_len, d_model)
        return tensor.view(batch_size, -1, self.attention_head_count, self.d_h).transpose(1, 2)

    def concat_head(self, tensor, batch_size):
        return tensor.transpose(1, 2).contiguous().view(batch_size, -1, self.attention_head_count * self.d_h)
