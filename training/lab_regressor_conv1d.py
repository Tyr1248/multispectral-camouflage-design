import torch
import torch.nn as nn

class Conv1DModel(nn.Module):
    def __init__(self, input_channels=1, output_dim=3):
        super(Conv1DModel, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=2, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(2)  # (B, 512, 2)
        self.head = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        # x: (B, 1, 4) ← already in (batch, channels, length) format
        x = self.conv_net(x)      # (B, 512, L'), L' >=4 due to padding
        x = self.pool(x)          # (B, 512, 2)
        x = x.flatten(1)          # (B, 1024)
        x = self.head(x)          # (B, 3)
        return x