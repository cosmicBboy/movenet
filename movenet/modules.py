from typing import List

import torch
import torch.nn as nn
from torch.nn import functional as F


class ConvTranspose2d(nn.Module):
    """Upsampler for conditioning features."""
    pass


class CausalConv1d(nn.Module):
    def __init__(self, input_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(
            input_channels,
            out_channels,
            kernel_size=2,
            stride=1,
            padding=1,
            bias=False,
        )

    def forward(self, x):
        # remove last value for causal convolution
        return self.conv(x)[:, :, :-1]


class DilatedCausalConv1d(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=2,
            stride=1,
            dilation=dilation,
            padding=0,
            bias=False,
        )

    def forward(self, x):
        return self.conv(x)


class GatedResidualConv1d(nn.Module):
    def __init__(self, residual_channels, skip_channels, dilation):
        super().__init__()
        self.dilated_conv = DilatedCausalConv1d(
            residual_channels, dilation=dilation
        )
        self.conv_residual = nn.Conv1d(residual_channels, residual_channels, 1)
        self.conv_skip = nn.Conv1d(residual_channels, skip_channels, 1)

    def forward(self, x, skip_size):
        dilated = self.dilated_conv(x)
        # pixel-cnn gating
        gated = torch.tanh(dilated) * torch.sigmoid(dilated)

        # residual network
        residual = self.conv_residual(gated)
        residual += x[:, :, -residual.size(2):]

        # skip connection
        skip = self.conv_skip(gated)
        skip = skip[:, :, -skip_size:]
        return residual, skip


class ResidualConvStack(nn.Module):
    def __init__(
        self, layer_size, stack_size, residual_channels, skip_channels
    ):
        super().__init__()
        self.layer_size = layer_size
        self.stack_size = stack_size
        self.conv_layers = nn.ModuleList([
            GatedResidualConv1d(
                residual_channels,
                skip_channels,
                dilation,
            ) for dilation in self.dilations
        ])

    @property
    def dilations(self) -> List[int]:
        """Get dilations for each layer and stack."""
        return [
            2 ** x for _ in range(self.stack_size)
            for x in range(self.layer_size)
        ]

    def forward(self, x, skip_size):
        output = x
        skip_connections = []
        for conv in self.conv_layers:
            output, skip = conv(output, skip_size)
            skip_connections.append(skip)
        return torch.stack(skip_connections)


class DenseConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 1)
        self.conv2 = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        x = self.conv1(F.relu(x))
        x = self.conv2(F.relu(x))
        return F.softmax(x, dim=1)
