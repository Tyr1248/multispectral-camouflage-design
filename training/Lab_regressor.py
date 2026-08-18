import torch.nn as nn
from torch.nn.utils import spectral_norm
class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearBlock, self).__init__()
        self.fc = spectral_norm(nn.Linear(in_features, out_features))
        self.bn = nn.BatchNorm1d(out_features)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_block_1 = LinearBlock(4, 256)
        self.linear_blocks = nn.Sequential(
            *[LinearBlock(256, 256) for _ in range(7)]
        )
        self.final_fc = spectral_norm(nn.Linear(256, 3))

    def forward(self, x):
        x = self.linear_block_1(x)
        x = self.linear_blocks(x)
        x = self.final_fc(x)
        return x