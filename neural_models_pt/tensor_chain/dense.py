import torch
import torch.nn as nn

from anthe_official.neural_models_pt.tensor_chain.utils import get_tc_kernel


class TCDense(nn.Module):
    def __init__(self, in_features, out_features, tc_length=3, bond=None, ratio=None):
        super(TCDense, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.tc_length = tc_length
        self.bond = bond
        self.ratio = ratio

        self.weight = None
        self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        self.weight = nn.Parameter(get_tc_kernel(self.in_features, self.out_features,
                                                 length=self.tc_length, bond=self.bond, ratio=self.ratio),
                                   requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(self.out_features), requires_grad=True)

    def forward(self, input):
        return torch.matmul(input, self.weight) + self.bias
