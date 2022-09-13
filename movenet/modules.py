from typing import List, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from movenet.types import AudioTensor, VideoTensor


class ConvTranspose2d(nn.Module):
    """Upsampler for conditioning features."""
    pass


class CausalConv1d(nn.Module):
    def __init__(self, input_channels, out_channels, kernel_size=2, bias=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            input_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1),
            bias=bias,
        )

    def forward(self, x):
        # remove last values for causal convolution
        return self.conv(x)[:, :, :-(self.kernel_size - 1)]


class DilatedCausalConv1d(nn.Module):
    def __init__(self, channels, dilation=1, kernel_size=2, bias=False):
        super().__init__()
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=1,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        return self.conv(x)


class GatedResidualConv1d(nn.Module):
    def __init__(self, residual_channels, skip_channels, dilation):
        super().__init__()
        self.conv_filter = DilatedCausalConv1d(
            residual_channels, dilation=dilation
        )
        self.conv_gate = DilatedCausalConv1d(
            residual_channels, dilation=dilation
        )
        self.context_conv_filter = nn.Conv1d(
            residual_channels, residual_channels, 1, padding=0, dilation=1
        )
        self.context_conv_gate = nn.Conv1d(
            residual_channels, residual_channels, 1, padding=0, dilation=1
        )
        self.conv_residual = nn.Conv1d(residual_channels, residual_channels, 1)
        self.conv_skip = nn.Conv1d(residual_channels, skip_channels, 1)

    def forward(
        self,
        input: AudioTensor,
        context: Optional[VideoTensor],
        skip_size,
    ):
        f, g = self.conv_filter(input), self.conv_gate(input)

        if context is not None:
            f += self.context_conv_filter(context)
            g += self.context_conv_gate(context)

        # pixel-cnn gating
        gated = torch.tanh(f) * torch.sigmoid(g)

        # residual network
        residual = self.conv_residual(gated)
        residual += input[:, :, -residual.size(2):]

        # skip connection: skip_size determines the output size of the network
        # given the model's receptive field and the input audio size. The latter
        # must always be larger than the former...
        # TODO: skip_size should really be called output_size for clarity
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

    def forward(
        self,
        input: AudioTensor,
        context: Optional[VideoTensor],
        skip_size: int,
    ):
        output = input
        skip_connections = []
        for gated_residual_conv in self.conv_layers:
            output, skip = gated_residual_conv(output, context, skip_size)
            skip_connections.append(skip)
        return torch.stack(skip_connections)


class DenseConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 1)

    def forward(self, x):
        x = self.conv1(F.leaky_relu(x))
        x = self.conv2(F.leaky_relu(x))
        return x
