import functools

import numpy as np
import torch
import torch.nn as nn
from torchtyping import TensorType

from movenet.modules import (
    CausalConv1d,
    ResidualConvStack,
    DenseConv,
)


class WaveNet(nn.Module):
    """WaveNet implementation that supports local and global conditioning.

    The main components of the wavenet model are:
    - causal convolutions: ensure that x_{t+1} is conditioned only on previous
      observations (x_1, ... x_t)
    - dilated convolutions: skip input values to increase receptive field.
    - quantize audio signal with range [-1, 1] using mu-law encoding to make
      model output a softmax distribution with 256 distinct values.
    - gated activation units to learn a convolutional filter that modulates the
      audio signal by learning which signal to amplify and which to weaken.
    - residual and parameterized skip connections to speed up convergence
    - global conditioning based on other static data source.
    - local conditioning based on dynamic data source, using a transposed
      CNN for learned upsampling. For example, this would upsample video frames
      to be at the same sample frequency as the audio signal.
    """

    # TODO: implementation plan
    # 1. create backbone of the architecture for processing audio signal
    # 2. add local conditioning for video signal
    # 3. add global conditioning for on dance-style category

    def __init__(self, layer_size, stack_size, input_channels, residual_channels):
        super().__init__()
        
        # attributes
        self.layer_size = layer_size
        self.stack_size = stack_size
        self.input_channels = input_channels
        self.residual_channels = residual_channels

        # modules
        self.causal_conv = CausalConv1d(input_channels, residual_channels)
        self.residual_conv_stack = ResidualConvStack(
            layer_size, stack_size, residual_channels, input_channels,
        )
        self.dense_conv = DenseConv(input_channels)

    @property
    @functools.lru_cache(maxsize=None)
    def receptive_fields(self):
        return sum(self.residual_conv_stack.dilations)

    def compute_output_size(self, x):
        output_size = int(x.size(2)) - self.receptive_fields
        if output_size < 1:
            raise ValueError(
                "input time steps must be larger than the number of receptive "
                f"fields. Number of input timesteps = {x.size(2)}, "
                f"receptive fields = {self.receptive_fields}"
            )
        return output_size

    def forward(
        self,
        audio: TensorType["batch", "channels", "frames"],
        video: TensorType["batch", "channels", "frames"] = None,
        global_features: TensorType = None,
    ):
        output = self.causal_conv(audio)
        skip_connections = self.residual_conv_stack(
            output, self.compute_output_size(audio)
        )
        output = self.dense_conv(torch.sum(skip_connections, dim=0))
        return output
