"""Train the movenet model."""

import logging
import shutil
from dataclasses import dataclass, asdict, field
from dataclasses_json import dataclass_json, config
from pathlib import Path
from typing import List, Optional

import torch
import torch.optim
import torch.nn.functional as F
import torchaudio
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchaudio.functional import mu_law_encoding, mu_law_decoding
from torchtyping import TensorType

from movenet import dataset
from movenet.wavenet import (
    WaveNet,
    AudioTensor,
    VideoTensor,
    VIDEO_KERNEL_SIZE,
    MAX_AUDIO_FRAMES,
    MAX_VIDEO_FRAMES,
)


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
    batch_size: int = 3
    checkpoint_every: int = 25
    optimizer: str = "Adam"
    learning_rate: float = 0.0002
    weight_decay: float = 0.0
    n_epochs: int = 100
    n_steps_per_epoch: Optional[int] = None
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


def training_step(model, optimizer, audio, video):
    if torch.cuda.is_available():
        audio, video = audio.to("cuda"), video.to("cuda")
    output = model(audio, video)
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
    return loss, grad_norm


@torch.no_grad()
def validation_step(model, audio, video):
    if torch.cuda.is_available():
        audio, video = audio.to("cuda"), video.to("cuda")
    output = model(audio, video)
    target = audio[:, :, model.receptive_fields:].argmax(1)
    loss = F.cross_entropy(output, target).item()
    return loss, output


def train_model(config: TrainingConfig, dataset_fp: str):

    dataloader = dataset.get_dataloader(
        dataset_fp,
        input_channels=config.model_config.input_channels,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )

    valid_dataloader = dataset.get_dataloader(
        dataset_fp,
        input_channels=config.model_config.input_channels,
        batch_size=config.batch_size,
        train=False,
        shuffle=False,
        num_workers=0,
    )

    # sample one batch to save for inspecting predictions, make sure it's
    # consistent over the epochs
    sample_batch_number = torch.randint(len(valid_dataloader), (1, )).item()

    if config.pretrained_model_path:
        model = torch.load(config.pretrained_model_path)
    else:
        model = WaveNet(**asdict(config.model_config))

    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = getattr(torch.optim, config.optimizer)(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    # training loop
    writer = SummaryWriter(config.tensorboard_dir)
    for epoch in range(1, config.n_epochs + 1):

        model.train()
        train_loss = 0.0
        for step, (audio, video, contexts, _) in enumerate(dataloader, 1):
            loss, grad_norm = training_step(model, optimizer, audio, video)
            train_loss += loss

            progress = step / len(dataloader)
            mean_loss = loss / config.batch_size
            logger.info(
                f"[epoch {epoch} | step {step}] "
                f"batch_progress={progress}, "
                f"minibatch_loss={loss:0.08f}, "
                f"minibatch_grad_norm={grad_norm:0.08f}"
            )
            writer.add_scalar("minibatch/progress/train", progress)
            writer.add_scalar("minibatch/loss/train", mean_loss, epoch * step)
            writer.add_scalar("minibatch/grad_norm", grad_norm, epoch * step)

            if config.n_steps_per_epoch and step > config.n_steps_per_epoch:
                break

        val_loss = 0.0
        sample_output = None
        sample_fps = None
        for step, (audio, video, contexts, fps) in enumerate(valid_dataloader):
            loss, output = validation_step(model, audio, video)
            val_loss += loss
            if step == sample_batch_number:
                sample_output = output
                sample_fps = fps

        train_loss /= len(dataloader.dataset)
        val_loss /= len(valid_dataloader.dataset)
        logger.info(
            f"[epoch {epoch}] "
            f"train_loss={train_loss:0.08f}, "
            f"val_loss={loss:0.08f}"
        )
        writer.add_scalar("epochs", epoch, epoch)
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)

        if epoch % config.checkpoint_every == 0:
            logger.info(f"creating checkpoint at epoch {epoch}")
            fp = args.model_output_path / "checkpoints" / str(epoch)
            fp.mkdir(parents=True)
            output_samples = mu_law_decoding(
                sample_output.argmax(1), config.model_config.input_channels
            ).to("cpu")
            torch.save(model, fp / "model.pth")
            torch.save(output_samples, fp / "output_samples.pth")
            for i, (sample_fp, sample) in enumerate(
                zip(sample_fps, output_samples)
            ):
                sample_fp = Path(sample_fp)
                # save original video files
                shutil.copyfile(
                    sample_fp, fp / f"original_video_{i}_{sample_fp.stem}.mp4"
                )
                # save generated mp3 file
                torchaudio.save(
                    str(fp / f"generated_autio_{i}_{sample_fp.stem}.mp3"),
                    # mp3 requires 2 channels (left, right)
                    torch.stack([sample, sample]),
                    sample_rate=sample.shape[0],
                )

    return model


if __name__ == "__main__":
    import argparse
    import os
    import subprocess
    from pathlib import Path
    from datetime import datetime


    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s:: %(message)s",
    )

    try:
        subprocess.call(["ls", "/artifacts"])
    except Exception as e:
        logging.info(f"skipping artifacts printing: {e}")

    try:
        logging.info("Downloading artifacts")
        logging.info(os.environ.keys())
        logging.info(str(os.getenv("FOOBAR")))
        logging.info(str(os.getenv("USERNAME")))
        logging.info(str(os.getenv("API_KEY")))
        logging.info(str(os.getenv("EXPERIMENT_ID")))
        subprocess.call([
            "grid",
            "login",
            "--username",
            str(os.getenv("USERNAME")),
            "--key",
            str(os.getenv("API_KEY")),
        ])
        subprocess.call([
            "grid",
            "artifacts",
            str(os.getenv("EXPERIMENT_ID")),
            "--download_dir",
            "/artifacts"
        ])
    except Exception as e:
        print(f"download artifacts failed: {e}")

    MAX_RETRIES = 10

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=0.0003)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--n_steps_per_epoch", type=int, default=None)
    parser.add_argument("--checkpoint_every", type=int, default=1)
    parser.add_argument("--input_channels", type=int, default=16)
    parser.add_argument("--residual_channels", type=int, default=16)
    parser.add_argument("--layer_size", type=int, default=3)
    parser.add_argument("--stack_size", type=int, default=3)
    parser.add_argument(
        "--pretrained_model_path",
        type=lambda x: None if x is None or x == "" else Path(x),
        default=None,
    )
    parser.add_argument(
        "--pretrained_run_exp_name",
        type=lambda x: None if x is None or x == "" else x,
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
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
        learning_rate=args.learning_rate,
        n_epochs=args.n_epochs,
        n_steps_per_epoch=args.n_steps_per_epoch,
        pretrained_model_path=args.pretrained_model_path,
        model_output_path=args.model_output_path,
        tensorboard_dir=args.training_logs_path,
    )
    with (args.model_output_path / "config.json").open("w") as f:
        f.write(config.to_json())

    logger.info(f"config: {config}")
    model = train_model(config, args.dataset)
    torch.save(model, args.model_output_path / "model.pth")
