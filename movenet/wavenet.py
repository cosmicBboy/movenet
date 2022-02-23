"""Wavenet Module.

Reference implementation:
https://github.com/golbin/WaveNet/blob/master/wavenet/networks.py
"""

import functools
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType

from movenet.modules import (
    CausalConv1d,
    ResidualConvStack,
    DenseConv,
)

AudioTensor = TensorType["batch", "channels", "frames"]
VideoTensor = TensorType["batch", "frames", "height", "width", "channels"]

MAX_AUDIO_FRAMES = 400000
MAX_VIDEO_FRAMES = 400
VIDEO_KERNEL_SIZE = (1, 128, 128)

UPSAMPLE_STRIDE = 10


def upsample_kernel_size_solver(
    in_size, out_size, stride=1, padding=0, output_padding=0, dilation=1,
):
    """
    Returns kernel size needed to upsample a tensor of some input size to
    a desired output size.

    The implementation solves for kernel size in the equation described the
    the "Shape" section of the pytorch docs:
    https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html
    """
    x = out_size - 1 - output_padding - (in_size - 1) * stride + 2 * padding
    x = int(x / dilation + 1)
    return (x, )


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
    # 1. ✅ create backbone of the architecture for processing audio signal
    # 2. ✅ add local conditioning for video signal
    #    - 🔬 first convolve the video frames to reduce dimensionality of the
    #      video signal before upsampling via transposed convolutions
    # 3. 🚧 add global conditioning for on dance-style category

    def __init__(
        self,
        layer_size: int,
        stack_size: int,
        input_channels: int,  # audio input channels
        residual_channels: int,
        video_in_channels: int = 1,
    ):
        super().__init__()
        
        # attributes
        self.layer_size = layer_size
        self.stack_size = stack_size
        self.input_channels = input_channels
        self.residual_channels = residual_channels

        # modules
        self.video_conv = nn.Conv3d(
            in_channels=video_in_channels,
            out_channels=residual_channels,
            kernel_size=VIDEO_KERNEL_SIZE,
        )

        # compute a sequence of kernel sizes for iteratively upsampling the
        # video frames
        upsample_sizes = np.geomspace(
            MAX_VIDEO_FRAMES,
            MAX_AUDIO_FRAMES,
            num=int(np.log10(MAX_AUDIO_FRAMES / MAX_VIDEO_FRAMES) + 1)
        ).astype(int)

        self.video_transpose = nn.Sequential(*[
            nn.ConvTranspose1d(
                in_channels=residual_channels,
                out_channels=residual_channels,
                kernel_size=upsample_kernel_size_solver(
                    in_size, out_size, stride=UPSAMPLE_STRIDE
                ),
                stride=UPSAMPLE_STRIDE,
            )
            for in_size, out_size in zip(upsample_sizes[:-1], upsample_sizes[1:])
        ])
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

    def upsample_video(self, video):
        video = video.permute(0, 4, 1, 2, 3)
        # convolve video signal to be of dim (batch x channels x frames)
        video_encoding = self.video_conv(video).squeeze(-1).squeeze(-1)
        # upsample video to match the number of frames in the audio sample.
        video_transposed = self.video_transpose(video_encoding)
        assert video_transposed.shape[-1] == MAX_AUDIO_FRAMES
        return video_transposed

    def forward(
        self,
        audio: AudioTensor,
        video: VideoTensor,
        global_features: TensorType = None,
        generate: bool = False,
        n_samples: Optional[int] = None,
    ):
        if generate:
            return self.generate(audio, video, global_features, n_samples)
        video = self.upsample_video(video)
        audio = self.causal_conv(audio)
        # TODO: apply causal_conv to video signal as well?
        assert video.size() == audio.size(), (
            "expected video and audio tensors to have equal sizes, found "
            f"{video.size()}, {audio.size()}"
        )
        skip_connections = self.residual_conv_stack(
            input=audio,
            context=video,
            skip_size=self.compute_output_size(audio)
        )
        output = self.dense_conv(torch.sum(skip_connections, dim=0))
        return output

    def generate(
        self,
        audio: AudioTensor,
        video: VideoTensor,
        global_features: TensorType = None,
        n_samples: Optional[int] = None,
    ):
        video = self.upsample_video(video)

        shape = audio.shape if n_samples is None else (
            audio.shape[0], audio.shape[1], n_samples
        )

        generated_audio = torch.zeros(
            shape, dtype=audio.dtype, device=audio.device
        )
        generated_audio[:, :, :self.receptive_fields] = audio[
            :, :, :self.receptive_fields
        ]

        # start generating audio at the point where there is sufficient data
        # in the model's receptive field.
        for i in range(self.receptive_fields, generated_audio.shape[-1] - 1):
            start, end = i - self.receptive_fields, i + 1
            local_audio = generated_audio[:, :, start: end]
            local_video = video[:, :, start: end]

            local_audio = self.causal_conv(local_audio)
            skip_connections = self.residual_conv_stack(
                input=local_audio,
                context=local_video,
                skip_size=self.compute_output_size(audio)
            )
            output = self.dense_conv(
                torch.sum(skip_connections, dim=0)
            )[:, :, -1:]
            generated_audio[:, :, [i + 1]] = torch.zeros_like(output).scatter_(
                1, output.argmax(1, keepdims=True), 1
            )
            del local_audio
            del local_video
            del skip_connections
            del output

        return generated_audio
