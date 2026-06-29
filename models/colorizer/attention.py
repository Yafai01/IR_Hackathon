import torch
import torch.nn as nn


class AttentionGate(nn.Module):

    def __init__(self, gate_channels, skip_channels, inter_channels):

        super().__init__()

        self.Wg = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, 1),
            nn.BatchNorm2d(inter_channels)
        )

        self.Wx = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, 1),
            nn.BatchNorm2d(inter_channels)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip):

        g = self.Wg(gate)

        x = self.Wx(skip)

        attention = self.relu(g + x)

        attention = self.psi(attention)

        return skip * attention