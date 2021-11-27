"""Create dataset for modeling."""

import logging
from collections import Counter
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from typing import List, NamedTuple

try:
    from typing import TypedDict
except:
    from typing_extensions import TypedDict

import torch
import torchaudio
import torchaudio.functional
import torchvision.io
from pytorchvideo.transforms.functional import uniform_temporal_subsample
from torchvision.transforms.functional import rgb_to_grayscale
from torchaudio.functional import mu_law_encoding
from torchtyping import TensorType

logger = logging.getLogger(__file__)


AudioTensor = TensorType["batch", "channels", "frames"]
VideoTensor = TensorType["batch", "frames", "height", "width", "channels"]

MAX_AUDIO_FRAMES = 400000
MAX_VIDEO_FRAMES = 300
VIDEO_KERNEL_SIZE = (1, 10, 10)


Info = TypedDict("info", video_fps=float, audio_fps=float)


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

    def __init__(self, filepath: str, train=True):
        self.filepath = Path(filepath)
        self.train = train

        # here we use the class label in the kinetics dataset as global
        # context
        self.contexts = [x.name for x in self.root_path.glob("*")]
        logger.info(f"dataset train={train} with contexts: {self.contexts}")

        self.index = []
        for context in self.contexts:
            for fp in (self.root_path / context).glob("*.mp4"):
                if "_raw" in fp.stem or fp.stem.startswith("."):
                    logging.debug(f"skipping file {fp}")
                    continue
                logging.debug(f"adding {fp} from context {context}")
                self.index.append(RawMetadata(context, str(fp)))

        self.class_balance = {
            k: v / len(self.index)
            for k, v in Counter(x.context for x in self.index).items()
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
            *torchvision.io.read_video(self.index[item].filepath)
        )


def get_dataloader(
    filepath,
    input_channels: int,
    batch_size: int = 64,
    train: bool = True,
    rank: int = 0,
    world_size: int = 0,
    **kwargs,
):
    dataset = KineticsDataset(filepath, train=train)
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
        collate_fn=partial(make_batch, input_channels),
        sampler=sampler,
        **kwargs
    )


class Batch:
    def __init__(self, audio, video, contexts, filepaths):
        self.audio = audio
        self.video = video
        self.contexts = contexts
        self.filepaths = filepaths

    def pin_memory(self):
        self.audio = self.audio.pin_memory()
        self.video = self.video.pin_memory()
        return self

    def __iter__(self):
        yield from (self.audio, self.video, self.contexts, self.filepaths)


def make_batch(input_channels: int, examples: List[Example]):
    logger.debug(f"-----\nprocessing data: {[x.filepath for x in examples]}")

    example = examples[0]
    audio = one_hot_encode_audio(resample_audio(example.audio), input_channels)
    video = resize_video(example.video.permute(0, 3, 1, 2))
    contexts = [example.context]
    filepaths = [example.filepath]

    if len(examples[1:]):
        for example in examples[1:]:
            try:
                # permute to: frames x channels x height x width
                video = torch.stack([
                    video, resize_video(example.video.permute(0, 3, 1, 2))]
                )
                audio = torch.stack([
                    audio,
                    audio.append(
                        one_hot_encode_audio(
                            resample_audio(example.audio), input_channels
                        )
                    )
                ])
                contexts.append(example.context)
                filepaths.append(example.filepath)
            except Exception as e:
                print(
                    f"ERROR: {e} filepath - {example.filepath}, "
                    f"audio - {example.audio}, "
                    f"video - {example.video}, "
                    f"context - {example.context}"
                )
    else:
        audio = torch.stack([audio])
        video = torch.stack([video])

    return Batch(audio, video, contexts, filepaths)


def resample_audio(
    audio: TensorType["channels", "frames"], freq=MAX_AUDIO_FRAMES
) -> TensorType["channels", "frames"]:
    x = audio.mean(dim=0)
    resampled_audio = torchaudio.functional.resample(
        x, x.shape[0], freq
    ).reshape(1, -1)
    if resampled_audio.shape[1] > freq:
        resampled_audio = resampled_audio[:, :freq]
    return resampled_audio


def normalize_audio(audio: TensorType["channels", "frames"]):
    """Normalize audio to be between -1 and 1."""
    if audio.sum() == 0:
        # TODO: need to better handle cases where all values are 0
        return audio
    mean_centered = audio - audio.mean()
    audio_output = mean_centered / mean_centered.abs().max()
    assert audio_output.min() >= -1, "audio minimum can't be less than -1"
    assert audio_output.max() <= 1, "audio maximum can't be more than 1"
    return audio_output


def one_hot_encode_audio(audio, input_channels):
    # need to figure out a more principled way of combining two audio
    # (left/right) channels into one
    # https://stackoverflow.com/questions/37313320/how-to-convert-two-channel-audio-into-one-channel-audio
    audio = normalize_audio(audio)
    quantized = mu_law_encoding(audio, input_channels)
    one_hot_enc = (
        torch.zeros(input_channels, quantized.size(1))
        .scatter_(0, quantized, 1)
    )
    assert (one_hot_enc.sum(dim=0) == 1).all(), "one hot encoding error"
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
