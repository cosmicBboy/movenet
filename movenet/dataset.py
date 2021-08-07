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
    return Example(*torchvision.io.read_video(video_file, pts_unit="pts"))


class KineticsDataset(torch.utils.data.Dataset):

    def __init__(self, filepath, train=True):
        self.filepath = filepath
        self.train = train

    def __len__(self):
        pass

    def __getitem__(self):
        pass


def get_dataloader():
    pass


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("filepath", type=str)
    args = parser.parse_args()

    dataset = KineticsDataset(args.filepath)
    import ipdb; ipdb.set_trace()
