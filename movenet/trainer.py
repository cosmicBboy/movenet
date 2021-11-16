"""Train the movenet model."""

import json
import logging
import os
import shutil
from dataclasses import dataclass, asdict, field
from dataclasses_json import dataclass_json, config
from pathlib import Path
from typing import Optional

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel
import torch.optim
import torchaudio
import wandb
from torch import distributed as dist
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


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s:: %(message)s",
    )


def wandb_setup():
    wandb.login()
    wandb.init(project="dance2music", entity="nielsbantilan")


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
    dist_backend: str = None
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


def training_step(model, optimizer, audio, video, receptive_fields):
    output = model(audio, video)
    target = audio[:, :, receptive_fields:].argmax(1)
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
def validation_step(model, audio, video, receptive_fields):
    # TODO: make sure this is only using video to generate audio
    # - need to figure out how to auto-regressively feed in the audio output to
    #   generate full audio sequence
    output = model(audio, video)
    target = audio[:, :, receptive_fields:].argmax(1)
    loss = F.cross_entropy(output, target).item()
    return loss, output


def train_model(
    config: TrainingConfig,
    dataset_fp: str,
    rank: int = 0,
    world_size: int = 0,
) -> nn.Module:

    dataloader = dataset.get_dataloader(
        dataset_fp,
        input_channels=config.model_config.input_channels,
        batch_size=config.batch_size,
        rank=rank,
        world_size=world_size,
        shuffle=True,
        num_workers=0,
    )

    valid_dataloader = dataset.get_dataloader(
        dataset_fp,
        input_channels=config.model_config.input_channels,
        batch_size=config.batch_size,
        train=False,
        rank=rank,
        world_size=world_size,
        shuffle=False,
        num_workers=0,
    )

    # sample one batch to save for inspecting predictions, make sure it's
    # consistent over the epochs
    sample_batch_number = (
        torch.randint(len(valid_dataloader), (1, )).item() + 1
    )

    logger.info("defining model")
    model = WaveNet(**asdict(config.model_config))
    if config.pretrained_model_path:
        model_state = torch.load(config.pretrained_model_path)
        if issubclass(type(model_state), nn.Module):
            model = model_state
        else:
            model.load_state_dict(
               torch.load(
                    config.pretrained_model_path,
                    map_location={"cuda:0": f"cuda:{rank}"}
                )
            )

    logger.info(f"model: {model}")

    receptive_fields = model.receptive_fields

    logger.info(f"CUDA AVAILABLE: {torch.cuda.is_available()}")
    distributed = False
    if torch.cuda.is_available():
        if world_size > 1:
            distributed = True
            model = nn.parallel.DistributedDataParallel(
                model.cuda(rank),
                device_ids=[rank],
                find_unused_parameters=True,
            )
        else:
            model = model.cuda(rank)

    optimizer = getattr(torch.optim, config.optimizer)(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    # training loop
    if rank == 0:
        writer = SummaryWriter(config.tensorboard_dir)

    for epoch in range(config.n_epochs):

        model.train()
        train_loss = 0.0
        logger.info(f"starting training loop for epoch {epoch}")
        for step, (audio, video, contexts, _) in enumerate(dataloader, 1):
            if torch.cuda.is_available():
                audio, video = audio.to(rank), video.to(rank)
            loss, grad_norm = training_step(
                model, optimizer, audio, video, receptive_fields
            )
            train_loss += loss

            prog = step / len(dataloader)
            mean_loss = loss / config.batch_size
            total = epoch * len(dataloader) + step
            if rank == 0:
                logger.info(
                    f"[epoch {epoch} | step {step}] "
                    f"batch_progress={prog}, "
                    f"minibatch_loss={loss:0.08f}, "
                    f"minibatch_grad_norm={grad_norm:0.08f}"
                )
                writer.add_scalar("minibatch/progress/train", prog, total)
                writer.add_scalar("minibatch/loss/train", mean_loss, total)
                writer.add_scalar("minibatch/grad_norm", grad_norm, total)

                wandb.log(
                    {"minibatch/progress/train": prog, "train_step": total}
                )
                wandb.log(
                    {"minibatch/loss/train": mean_loss, "train_step": total}
                )
                wandb.log(
                    {"minibatch/grad_norm": grad_norm, "train_step": total}
                )

            if config.n_steps_per_epoch and step > config.n_steps_per_epoch:
                break

        val_loss = 0.0
        sample_output = None
        sample_fps = None
        logger.info(f"starting validation loop for epoch {epoch}")
        for step, (audio, video, contexts, fps) in enumerate(
            valid_dataloader, 1
        ):
            if torch.cuda.is_available():
                audio, video = audio.to(rank), video.to(rank)
            _val_loss, output = validation_step(
                model, audio, video, receptive_fields
            )
            # wait for all processes to complete the validation step
            if distributed:
                dist.barrier()

            val_loss += _val_loss
            mean_val_loss = val_loss / config.batch_size
            prog = step / len(valid_dataloader)
            total = epoch * len(valid_dataloader) + step
            if rank == 0:
                logger.info(
                    f"[epoch {epoch} | step {step}] "
                    f"batch_progress={prog}, "
                    f"minibatch_loss={val_loss:0.08f}"
                )
                writer.add_scalar("minibatch/progress/val", prog, total)
                writer.add_scalar("minibatch/loss/val", mean_val_loss, total)

                wandb.log({"minibatch/progress/val": prog, "val_step": total})
                wandb.log(
                    {"minibatch/loss/val": mean_val_loss, "val_step": total}
                )
            if rank == 0 and step == sample_batch_number:
                sample_output = output
                if torch.cuda.is_available():
                    sample_output = sample_output.to(rank)
                sample_fps = fps

        if rank == 0:
            logger.info(f"computing training and validation loss")
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

            wandb.log({"loss/train": train_loss, "epoch": epoch})
            wandb.log({"loss/val": val_loss, "epoch": epoch})

        fp = config.model_output_path / "checkpoints" / str(epoch)
        checkpoint_path = fp / "model.pth"
        if epoch % config.checkpoint_every == 0 and rank == 0:
            # TODO: find out where training halts
            logger.info(f"creating checkpoint at epoch {epoch}")

            logger.info(f"checkpoint path: {fp}")
            fp.mkdir(parents=True)
            torch.save(model.state_dict(), checkpoint_path)

            logger.info("decoding output samples")
            output_samples = mu_law_decoding(
                sample_output.argmax(1), config.model_config.input_channels
            ).to("cpu")

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

        if distributed:
            dist.barrier()
        # wait for rank 0 to finish writing checkpoint
        logger.info(f"ending training loop for epoch {epoch}")
        if isinstance(model, nn.parallel.DistributedDataParallel):
            logger.info("loading distributed model from checkpoints")
            model.load_state_dict(
                torch.load(
                    checkpoint_path,
                    map_location={"cuda:0": f"cuda:{rank}"}
                )
            )

    return model



def dist_train_model(
    rank: int,
    world_size: int,
    config: TrainingConfig,
    dataset_fp: str,
    model_output_path: Path,
):
    configure_logging()

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8888"
    dist.init_process_group(
        config.dist_backend, rank=rank, world_size=world_size
    )
    if rank == 0:
        wandb_setup()
        wandb.config.update(json.loads(config.to_json()))

    model = train_model(config, dataset_fp, rank=rank, world_size=world_size)
    if rank == 0:
        wandb.finish()
        torch.save(model.state_dict(), model_output_path / "model.pth")

    dist.destroy_process_group()


if __name__ == "__main__":
    import argparse
    import os
    import subprocess
    from pathlib import Path
    from datetime import datetime


    configure_logging()

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
    parser.add_argument("--dist_backend", type=str, default="nccl")
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
    # temporary hack until grid environment issue is solved
    parser.add_argument("--grid_user_name", type=str, default="")
    parser.add_argument("--grid_api_key", type=str, default="")
    parser.add_argument("--wandb_api_key", type=str, default="")
    args = parser.parse_args()

    os.environ["WANDB_API_KEY"] = args.wandb_api_key

    if args.pretrained_model_path and args.pretrained_run_exp_name:
        try:
            logging.info("Downloading artifacts")
            subprocess.call([
                "grid",
                "login",
                "--username",
                args.grid_user_name,
                "--key",
                args.grid_api_key,
            ])
            subprocess.call([
                "grid",
                "artifacts",
                args.pretrained_run_exp_name,
                "--download_dir",
                "/artifacts"
            ])
        except Exception as e:
            print(f"download artifacts failed: {e}")

    logger.info(f"starting training run")
    logger.info(f"CUDA AVAILABLE: {torch.cuda.is_available()}")
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
        dist_backend=args.dist_backend,
        pretrained_model_path=args.pretrained_model_path,
        model_output_path=args.model_output_path,
        tensorboard_dir=args.training_logs_path,
    )
    with (args.model_output_path / "config.json").open("w") as f:
        f.write(config.to_json())

    logger.info(f"config: {config}")

    world_size = torch.cuda.device_count()
    if world_size > 1:
        mp.spawn(
            dist_train_model,
            args=(world_size, config, args.dataset, args.model_output_path),
            nprocs=world_size,
            join=True
        )
    else:
        # initialize wandb
        wandb_setup()
        model = train_model(config, args.dataset)
        torch.save(model, args.model_output_path / "model.pth")
