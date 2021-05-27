"""Train the movenet model."""

import logging
from dataclasses import dataclass, asdict, field
from dataclasses_json import dataclass_json, config
from pathlib import Path
from typing import List, Optional

import torch
import torch.optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchaudio.functional import mu_law_encoding, mu_law_decoding
from torchtyping import TensorType

from movenet import dataset
from movenet.wavenet import WaveNet


logger = logging.getLogger(__file__)


@dataclass_json
@dataclass
class ModelConfig:
    layer_size: int = 2
    stack_size: int = 2
    input_channels: int = 256
    residual_channels: int = 512


@dataclass_json
@dataclass
class TrainingConfig:
    model_config: ModelConfig = ModelConfig()
    checkpoint_every: int = 25
    optimizer: str = "Adam"
    learning_rate: float = 0.0002
    weight_decay: float = 0.0
    n_training_steps: int = 100
    pretrained_model_path: Optional[Path] = field(
        default=None,
        metadata=config(
            encoder=lambda x: x if x is None else str(x),
            decoder=lambda x: x if x is None else Path(x),
        ),
    )
    model_output_path: Path = field(
        default="models", metadata=config(encoder=str, decoder=Path),
    )
    tensorboard_dir: Path = field(
        default="tensorboard_logs", metadata=config(encoder=str, decoder=Path),
    )


def one_hot_encode(audio, input_channels):
    # need to figure out a more principled way of combining two audio
    # (left/right) channels into one
    # https://stackoverflow.com/questions/37313320/how-to-convert-two-channel-audio-into-one-channel-audio
    combined_channels = audio.mean(dim=0, keepdims=True)
    quantized = mu_law_encoding(combined_channels, input_channels)
    one_hot_enc = (
        torch.zeros(input_channels, quantized.size(1))
        .scatter_(0, quantized, 1)
    )
    assert (one_hot_enc.sum(dim=0) == 1).all(), "one hot encoding error"
    return one_hot_enc


def zero_padded_stack(tensors: List[TensorType["channels", "frames"]]):
    max_frames = max(x.size(1) for x in tensors)
    output = []
    for x in tensors:
        padded_x = torch.zeros(x.size(0), max_frames)
        padded_x[:, :x.size(1)] = x
        output.append(padded_x)
    return torch.stack(output)


def make_batch(examples: List[dataset.Example], config: TrainingConfig):
    audio = zero_padded_stack(
        [
            one_hot_encode(x.audio, config.model_config.input_channels)
            for x in examples
        ]
    )
    return audio


def train_model(config: TrainingConfig, batch_fps: List[str]):
    if config.pretrained_model_path:
        model = torch.load(config.pretrained_model_path)
    else:
        model = WaveNet(**asdict(config.model_config))

    optimizer = getattr(torch.optim, config.optimizer)(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    # training loop
    writer = SummaryWriter(config.tensorboard_dir)
    raw_data = [dataset.load_video(fp) for fp in batch_fps]
    audio = make_batch(raw_data, config)
    for i in range(1, config.n_training_steps + 1):
        output = model(audio)
        target = audio[:, :, model.receptive_fields:].argmax(1)
        loss = F.cross_entropy(output, target)
        optimizer.zero_grad()
        loss.backward()

        grad_norm = 0.
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        loss = loss.data.item()
        optimizer.step()

        logger.info(
            f"[step {i}] loss={loss:0.08f}, grad_norm={grad_norm:0.08f}"
        )
        writer.add_scalar("loss/train", loss, i)
        writer.add_scalar("grad_norm", grad_norm, i)

        if  i % config.checkpoint_every == 0:
            logger.info(f"creating checkpoint at step {i}")
            fp = args.model_output_path / "checkpoints" / str(i)
            fp.mkdir(parents=True)
            torch.save(model, fp / "model.pth")
            torch.save(
                mu_law_decoding(output, config.model_config.input_channels),
                fp / "output_sample.pth"
            )

    return model


if __name__ == "__main__":
    import argparse
    import json
    import time
    from pathlib import Path
    from datetime import datetime

    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(levelname)s] %(asctime)s:: %(message)s",
    )

    MAX_RETRIES = 10

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--learning_rate", type=float, default=0.0003)
    parser.add_argument("--n_training_steps", type=int, default=3)
    parser.add_argument("--checkpoint_every", type=int, default=25)
    parser.add_argument("--input_channels", type=int, default=16)
    parser.add_argument("--residual_channels", type=int, default=16)
    parser.add_argument("--layer_size", type=int, default=3)
    parser.add_argument("--stack_size", type=int, default=3)
    parser.add_argument(
        "--pretrained_model_path",
        type=lambda x: x if x is None else Path(x),
        default=None,
    )
    parser.add_argument(
        "--model_output_path",
        type=Path,
        default=Path("models") / datetime.now().strftime("%Y%m%d%H%M%S"),
    )
    parser.add_argument(
        "--training_logs_path", type=Path, default=Path("training_logs"),
    )
    args = parser.parse_args()

    logger.info(f"starting training run")
    (args.model_output_path / "checkpoints").mkdir(exist_ok=True, parents=True)

    config = TrainingConfig(
        model_config=ModelConfig(
            input_channels=args.input_channels,
            residual_channels=args.residual_channels,
            layer_size=args.layer_size,
            stack_size=args.stack_size,
        ),
        checkpoint_every=args.checkpoint_every,
        learning_rate=args.learning_rate,
        n_training_steps=args.n_training_steps,
        pretrained_model_path=args.pretrained_model_path,
        model_output_path=args.model_output_path,
        tensorboard_dir=args.training_logs_path,
    )
    with (args.model_output_path / "config.json").open("w") as f:
        f.write(config.to_json())

    training_data_path = Path(args.dataset) / "train" / "breakdancing"
    batch_fps = [
        str(file_name) for file_name in training_data_path.glob("*.mp4")
    ]
    logger.info(f"config: {config}")
    logger.info(f"files: {batch_fps}")

    # make sure video files can be loaded
    for _ in range(MAX_RETRIES):
        try:
            dataset.load_video(batch_fps[0])
            logger.info(f"video loading success")
            break
        except:
            time.sleep(1)
            pass

    model = train_model(config, batch_fps)
    torch.save(model, args.model_output_path / "model.pth")
