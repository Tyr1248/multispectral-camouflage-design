import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from Lab_regressor import NeuralNetwork


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu', a=0.2)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


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


class ResidualLinearBlock(nn.Module):
    def __init__(self, features):
        super(ResidualLinearBlock, self).__init__()
        self.block = nn.Sequential(
            spectral_norm(nn.Linear(features, features)),
            nn.BatchNorm1d(features),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(features, features)),
            nn.BatchNorm1d(features),
        )
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return self.activation(out)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.z_upraising = LinearBlock(2, 128)
        self.lab_upraising = LinearBlock(3, 128)
        self.feature_fusion = nn.Sequential(
            *[ResidualLinearBlock(256) for _ in range(3)]
        )
        self.thickness_regressor = nn.Sequential(
            LinearBlock(256, 128),
            spectral_norm(nn.Linear(128, 4)),
            nn.Sigmoid()
        )

    def forward(self, z, lab):
        z_upraised = self.z_upraising(z)
        lab_upraised = self.lab_upraising(lab)
        combined = torch.cat([z_upraised, lab_upraised], dim=1)
        features = self.feature_fusion(combined)
        thickness = self.thickness_regressor(features)
        return thickness


class DistributionEvaluator(nn.Module):
    def __init__(self):
        super(DistributionEvaluator, self).__init__()
        self.layers = nn.Sequential(
            LinearBlock(4, 256),
            *[ResidualLinearBlock(256) for _ in range(1)],
            LinearBlock(256, 128),
            nn.Dropout(0.9),
            spectral_norm(nn.Linear(128, 1))
        )

    def forward(self, x):
        return self.layers(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.lab_regressor = NeuralNetwork()  # 预训练的Lab regressor
        # 注意：需要确保Lab regressor支持4维厚度输入
        self.lab_regressor.load_state_dict(torch.load('models/lab_regressor.pth'))
        for param in self.lab_regressor.parameters():  # 冻结预训练模型参数
            param.requires_grad = False
        self.evaluator = DistributionEvaluator()

    def forward(self, thickness, real_lab=None):
        if real_lab == None:
            pred_lab = self.lab_regressor(thickness)  # 预测Lab颜色
            return self.evaluator(thickness), pred_lab
        else:
            return self.evaluator(thickness), real_lab