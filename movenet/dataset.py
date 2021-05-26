"""Create dataset for modeling."""

from dataclasses import dataclass
from typing import Any, Dict

try:
    from typing import TypedDict
except:
    from typing_extensions import TypedDict

import torch
import torchaudio
import torchvision.io
from torchtyping import TensorType


Info = TypedDict("info", video_fps=float, audio_fps=float)


@dataclass
class Example:
    video: TensorType[float]
    audio: TensorType[float]
    info: Info


def load_video(video_file: str):
    # video, audio, info = torchvision.io.read_video(video_file, pts_unit="sec")
    # timestamps, _ = torchvision.io.read_video_timestamps(
    #     video_file, pts_unit="sec"
    # )
    # import ipdb; ipdb.set_trace()
    return Example(*torchvision.io.read_video(video_file, pts_unit="pts"))


def quantize_audio():
    pass


def create_dataset():
    pass


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("video_file", type=str)
    args = parser.parse_args()

    load_video(args.video_file)
