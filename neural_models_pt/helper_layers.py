import torch
import torch.nn as nn

class ProjectionLayer(nn.Module):
    def __init__(self):
        super(ProjectionLayer, self).__init__()
        self.project_matrix = nn.Parameter(torch.Tensor())

    def forward(self, inputs):
        output = torch.matmul(inputs, self.project_matrix)

        return output
