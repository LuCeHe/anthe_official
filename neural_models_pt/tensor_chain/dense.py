import torch
import torch.nn as nn

from anthe_official.neural_models_pt.tensor_chain.utils import get_tc_kernel


class TCDense(nn.Module):
    def __init__(self, in_features, out_features, tc_length=3, bond=None, ratio=None, axis=-1, use_bias=True):
        super(TCDense, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.tc_length = tc_length
        self.bond = bond
        self.ratio = ratio
        self.axis = axis
        self.use_bias = use_bias

        self.weight = None
        self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        self.weight = get_tc_kernel(
            self.in_features, self.out_features,
            length=self.tc_length, bond=self.bond, ratio=self.ratio
        )
        kernels, es_string = get_tc_kernel(
            self.in_features, self.out_features,
            length=self.tc_length, bond=self.bond, ratio=self.ratio, return_tensors=True
        )
        self.es_string = es_string
        self.weights = torch.nn.ParameterList(kernels)

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(self.out_features), requires_grad=True)
        else:
            self.bias = 0

    def forward(self, input):
        x = input
        if self.axis == 1:
            x = torch.transpose(x, 1, 2)

        weight = torch.einsum(self.es_string, *self.weights)
        weight = torch.reshape(weight, (self.in_features, self.out_features))

        x = torch.matmul(x, weight) + self.bias

        if self.axis == 1:
            x = torch.transpose(x, 1, 2)

        return x
