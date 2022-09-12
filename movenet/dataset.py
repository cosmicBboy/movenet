"""Create dataset for modeling."""

import logging
from collections import Counter
from functools import partial
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from typing import List, NamedTuple

try:
    from typing import TypedDict
except:
    from typing_extensions import TypedDict

import pandas as pd
import torch
import torchvision.io
from pytorchvideo.transforms.functional import uniform_temporal_subsample
from torchvision.transforms.functional import rgb_to_grayscale
from torchaudio.functional import mu_law_encoding, resample
from torchtyping import TensorType

from movenet.wavenet import (
    MAX_AUDIO_FRAMES, MAX_VIDEO_FRAMES, VIDEO_KERNEL_SIZE,
)

logger = logging.getLogger(__file__)


AudioTensor = TensorType["batch", "channels", "frames"]
VideoTensor = TensorType["batch", "frames", "height", "width", "channels"]


Info = TypedDict(
    "info",
    video_fps=float,
    audio_fps=float,
    video_orig_dim=float,
    audio_orig_dim=float,
)


class RawMetadata(NamedTuple):
    context: str
    filepath: str


class Example(NamedTuple):
    context: str
    filepath: str
    video: TensorType[float]
    audio: TensorType[float]
    info: Info


class KineticsDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        filepath: str,
        input_channels: int,
        train: bool = True,
        use_video: bool = True,
        normalize_audio: bool = True,
    ):
        self.filepath = Path(filepath)
        self.train = train
        self.input_channels = input_channels
        self.use_video = use_video
        self.normalize_audio = normalize_audio

        # here we use the class label in the kinetics dataset as global
        # context
        self.contexts = [x.name for x in self.root_path.glob("*")]
        logger.info(f"dataset train={train} with contexts: {self.contexts}")
        logger.info(f"use video data: {use_video}")

        index = []
        for context in self.contexts:
            for fp in (self.root_path / context).glob("*.mp4"):
                if "_raw" in fp.stem or fp.stem.startswith("."):
                    logging.debug(f"skipping file {fp}")
                    continue
                logging.debug(f"adding {fp} from context {context}")
                index.append(RawMetadata(context, str(fp)))
        self.index = pd.Series(index)

        self.class_balance = {
            k: v / len(index)
            for k, v in Counter(x.context for x in index).items()
        }
        logger.info(
            f"dataset index contains {len(self.index)} data points with class "
            f"balance: {self.class_balance}"
        )

    @property
    def root_path(self):
        return self.filepath / ("train" if self.train else "valid")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        return Example(
            self.index[item].context,
            self.index[item].filepath,
            *read_video(
                self.index[item].filepath,
                self.input_channels,
                self.use_video,
                self.normalize_audio,
            ),
        )


def read_video(filepath, input_channels, use_video: bool, normalize_audio: bool):
    video, audio, info = torchvision.io.read_video(filepath, pts_unit="sec")
    # permute to: frames x channels x height x width
    info.update({
        "video_orig_dim": video.shape[0],
        "audio_orig_dim": audio.shape[1],
    })
    if video.shape[0] == 0:
        return None, None, info

    video = resize_video(video.permute(0, 3, 1, 2)) if use_video else None
    audio = one_hot_encode_audio(
        resample_audio(audio), input_channels, normalize_audio,
    )
    return video, audio, info


def get_dataloader(
    filepath,
    input_channels: int,
    batch_size: int = 64,
    train: bool = True,
    rank: int = 0,
    world_size: int = 0,
    use_video: bool = True,
    normalize_audio: bool = True,
    **kwargs,
):
    dataset = KineticsDataset(
        filepath,
        input_channels,
        train=train,
        use_video=use_video,
        normalize_audio=normalize_audio,
    )
    sampler = None
    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            rank=rank,
            num_replicas=1 if not world_size else world_size,
            shuffle=True
        )
        kwargs.pop("shuffle")
        logger.info(f"DataLoader kwargs: {kwargs}")
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=partial(make_batch, use_video),
        sampler=sampler,
        **kwargs
    )


class Batch:
    def __init__(self, audio, video, contexts, filepaths, info):
        self.audio = audio
        self.video = video
        self.contexts = contexts
        self.filepaths = filepaths
        self.info = info

    def pin_memory(self):
        self.audio = self.audio.pin_memory()
        if self.video is not None:
            self.video = self.video.pin_memory()
        return self

    def __iter__(self):
        yield from (
            self.audio, self.video, self.contexts, self.filepaths, self.info
        )


def make_batch(use_video: bool, examples: List[Example]):
    logger.debug(f"-----\nprocessing data: {[x.filepath for x in examples]}")

    audio, video, contexts, filepaths, info = [], [], [], [], []

    for example in examples:
        if example.video is None and example.audio is None:
            continue
        video.append(example.video)
        audio.append(example.audio)
        contexts.append(example.context)
        filepaths.append(example.filepath)
        info.append(example.info)

    if not video:
        raise ValueError(
            f"Cannot process empty batch for instances {examples}."
        )

    return Batch(
        torch.stack(audio),
        torch.stack(video) if use_video else None,
        contexts,
        filepaths,
        info,
    )


def resample_audio(
    audio: TensorType["channels", "frames"], freq=MAX_AUDIO_FRAMES
) -> TensorType["channels", "frames"]:
    x = audio.mean(dim=0)
    resampled_audio = resample(x, x.shape[0], freq).reshape(1, -1)
    if resampled_audio.shape[1] > freq:
        resampled_audio = resampled_audio[:, :freq]
    return resampled_audio


def _normalize_audio(audio: TensorType["channels", "frames"]):
    """Normalize audio to be between -1 and 1."""
    if audio.sum() == 0:
        # TODO: need to better handle cases where all values are 0
        return audio

    min_val, max_val = audio.min(), audio.max()
    audio = (audio - min_val) / (max_val - min_val)
    # normalize with the formula: audio * (upper - lower) + lower
    # upper = 1, lower = -1, this simplifies to the expression below
    return audio * 2 - 1


def one_hot_encode_audio(audio, input_channels: int, normalize_audio: bool):
    # need to figure out a more principled way of combining two audio
    # (left/right) channels into one
    # https://stackoverflow.com/questions/37313320/how-to-convert-two-channel-audio-into-one-channel-audio
    if normalize_audio:
        audio = _normalize_audio(audio)
    quantized = mu_law_encoding(audio, input_channels)
    one_hot_enc = (
        torch.zeros(input_channels, quantized.size(1))
        .scatter_(0, quantized, 1)
    )
    return one_hot_enc


def resize_video(
    video: TensorType["frames", "channels", "height", "width"],
    num_samples: int = MAX_VIDEO_FRAMES,
) -> List[TensorType["frames", "channels", "height", "width"]]:
    video_resized = torch.zeros(
        video.shape[0],
        1,  # downsampling from RGB to Grayscale
        *VIDEO_KERNEL_SIZE[1:]
    )
    for i, frame in enumerate(video):
        video_resized[i] = torchvision.transforms.functional.resize(
            rgb_to_grayscale(frame), size=VIDEO_KERNEL_SIZE[1:]
        )
    sampled_video = uniform_temporal_subsample(
        video_resized, num_samples=num_samples, temporal_dim=0
    )
    if sampled_video.shape[0] > num_samples:
        sampled_video = sampled_video[:num_samples]
    return sampled_video.permute(0, 2, 3, 1)


if __name__ == "__main__":
    import time
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("filepath", type=str)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(1000)

    dataloader = get_dataloader(
        args.filepath,
        input_channels=16,
        batch_size=8,
        shuffle=True,
        num_workers=args.num_workers,
    )
    n_batches = len(dataloader)
    print(f"iterating through {n_batches} batches")
    writer = SummaryWriter()
    start = time.time()
    for i, (audio, video, contexts, filepaths) in enumerate(dataloader, 1):
        writer.add_scalar("n_steps", i, i)
        writer.add_scalar("percent_progress", i / n_batches, i)
        print(f"[batch {i}/{n_batches}]")
    print("done iterating through dataset")

    print("now doing some random stuff")
    import numpy as np

    offset = np.random.uniform(0, 5, 1)[0]
    for x in range(1, 10):
        y = -np.log(x) + offset + (np.sin(x) * 0.1)
        writer.add_scalar('y=-log(x) + c + 0.1sin(x)', y, x)
        writer.add_scalar('fake_metric', -y, x)

    # print GPUs, params and random tensors
    print('-' * 50)
    print(f'GPUS: There are {torch.cuda.device_count()} GPUs on this machine')
    print('-' * 50)
    print('i can run any ML library like numpy, pytorch lightning, sklearn pytorch, keras, tensorflow')
    print('torch:', torch.rand(1), 'numpy', np.random.rand(1))

    # write some artifacts
    f = open("weights.pt", "a")
    f.write("fake weights")
    f.close()

    with open("time.txt", "a") as f:
        f.write(f"time taken: {time.time() - start}")
