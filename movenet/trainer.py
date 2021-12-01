"""Train the movenet model."""

import json
import gc
import logging
import os
import shutil
from dataclasses import dataclass, asdict, field
from dataclasses_json import dataclass_json, config
from pathlib import Path
from typing import Optional

import librosa
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel
import torch.optim
import torchaudio
import torchvision.io
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

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

CLIP_GRAD = 10.0


logger = logging.getLogger(__file__)


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s:: %(message)s",
    )


def wandb_setup():
    wandb.login()
    wandb.init(
        project="dance2music",
        entity="nielsbantilan",
        name=os.getenv("GRID_EXPERIMENT_NAME", None)
    )


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
    optimizer: str = "AdamW"
    scheduler: str = "OneCycleLR"
    learning_rate: float = 0.0002
    accumulation_steps: int = 1
    num_workers: int = 0
    pin_memory: bool = False

    # found through learning rate range experiment:
    # https://wandb.ai/nielsbantilan/dance2music/runs/3a4sfxev?workspace=user-nielsbantilan
    max_learning_rate: float = 0.00003

    weight_decay: float = 0.0
    n_epochs: int = 100
    n_steps_per_epoch: Optional[int] = None
    dist_backend: str = None
    dist_port: str = "8888"
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


def training_step(
    config: TrainingConfig,
    step,
    model,
    optimizer,
    audio,
    video,
    rank=0,
    scaler=None,
    last_step=False,
):
    with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
        if torch.cuda.is_available():
            audio, video = audio.cuda(rank), video.cuda(rank)
        output = model(audio, video)
        target = audio[:, :, model.receptive_fields:].argmax(1)
        loss = F.cross_entropy(output, target)
        loss /= config.accumulation_steps

    if scaler:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
    else:
        loss.backward()

    nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)

    grad_norm = 0.
    for param in model.parameters():
        if param.grad is not None:
            grad_norm += param.grad.detach().norm(2).cpu().item() ** 2
    grad_norm = grad_norm ** 0.5

    if step % config.accumulation_steps == 0 or last_step:
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return loss, grad_norm


@torch.no_grad()
def validation_step(model, audio, video, rank=0):
    # TODO: make sure this is only using video to generate audio
    # - need to figure out how to auto-regressively feed in the audio output to
    #   generate full audio sequence
    with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
        if torch.cuda.is_available():
            audio, video = audio.cuda(rank), video.cuda(rank)
        output = model(audio, video)
        target = audio[:, :, model.receptive_fields:].argmax(1)
        loss = F.cross_entropy(output, target).detach().item()

    return loss.detach().cpu().item(), output


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
        pin_memory=config.pin_memory,
        num_workers=config.num_workers,
    )

    valid_dataloader = dataset.get_dataloader(
        dataset_fp,
        input_channels=config.model_config.input_channels,
        batch_size=config.batch_size,
        train=False,
        rank=rank,
        world_size=world_size,
        shuffle=False,
        pin_memory=config.pin_memory,
        num_workers=config.num_workers,
    )

    # sample one batch to save for inspecting predictions, make sure it's
    # consistent over the epochs
    sample_batch_number = (
        torch.randint(len(valid_dataloader), (1, )).detach() + 1
    )

    logger.info(f"CUDA AVAILABLE: {torch.cuda.is_available()}")
    logger.info("Defining model")

    distributed = False
    model = WaveNet(**asdict(config.model_config))

    if torch.cuda.is_available():
        if world_size > 1:
            distributed = True
            logger.info("Training with nn.parallel.DistributedDataParallel")
            model = nn.parallel.DistributedDataParallel(
                model.cuda(rank),
                device_ids=[rank],
                find_unused_parameters=True,
            )
            model.receptive_fields = model.module.receptive_fields
        else:
            logger.info("Training on gpu")
            model = model.cuda(rank)

    if config.pretrained_model_path:
        pretrained_model = torch.load(
            config.pretrained_model_path,
            map_location=(
                "cpu"
                if not (torch.cuda.is_available() or distributed)
                else {"cuda:0": f"cuda:{rank}"}
            ),
        )
        if issubclass(type(pretrained_model), nn.Module):
            logger.info("pretrained model is an nn.Module")
            state_dict = pretrained_model.state_dict()
            
            model.load_state_dict(state_dict)
        else:
            logger.info(f"Pretrained model is a state dict.")
            state_dict = pretrained_model

        if distributed:
            state_dict = {
                f"module.{k}" if not k.startswith("module") else k: v
                for k, v in state_dict.items()
            }

        model.load_state_dict(state_dict)

    if rank == 0:
        logger.info(f"Model: {model}")

    optimizer = getattr(torch.optim, config.optimizer)(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = getattr(torch.optim.lr_scheduler, config.scheduler)(
        optimizer,
        max_lr=config.max_learning_rate,
        epochs=config.n_epochs,
        steps_per_epoch=len(dataloader),
        three_phase=True,
    )

    # training loop
    if rank == 0:
        writer = SummaryWriter(config.tensorboard_dir)

    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    for epoch in range(config.n_epochs):

        if distributed:
            dataloader.sampler.set_epoch(epoch)

        model.train()
        train_loss = 0.0
        logger.info(f"starting training loop for epoch {epoch}")
        for step, (audio, video, contexts, fps, _) in enumerate(dataloader, 1):
            last_step = step == len(dataloader)
            loss, grad_norm = training_step(
                config, step, model, optimizer, audio, video, rank, scaler,
                last_step=last_step
            )
            loss = loss.detach().cpu().item()
            train_loss += loss

            prog = step / len(dataloader)
            mean_loss = loss / step
            total = epoch * len(dataloader) + step
            if (
                rank == 0
                and (step % config.accumulation_steps == 0 or last_step)
            ):
                batch_lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"[epoch {epoch} | step {step}] "
                    f"batch_progress={prog}, "
                    f"minibatch_loss={loss:0.08f}, "
                    f"minibatch_grad_norm={grad_norm:0.08f}, "
                    f"minibatch_filepaths={fps}"
                )
                writer.add_scalar("minibatch/progress/train", prog, total)
                writer.add_scalar("minibatch/loss/train", mean_loss, total)
                writer.add_scalar("minibatch/grad_norm", grad_norm, total)
                writer.add_scalar("minibatch/learning_rate", batch_lr, total)

                wandb.log({
                    "minibatch/progress/train": prog,
                    "minibatch/loss/train": mean_loss,
                    "minibatch/grad_norm": grad_norm,
                    "minibatch/learning_rate": batch_lr,
                    "train_step": total,
                })

                scheduler.step()

            if config.n_steps_per_epoch and step > config.n_steps_per_epoch:
                break

        val_loss = 0.0
        sample_output = None
        sample_fps = None
        logger.info(f"starting validation loop for epoch {epoch}")

        for step, (audio, video, contexts, fps, info) in enumerate(
            valid_dataloader, 1
        ):
            _val_loss, output = validation_step(model, audio, video, rank)
            # wait for all processes to complete the validation step
            if distributed:
                dist.barrier()

            val_loss += _val_loss
            mean_val_loss = val_loss / step
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
                wandb.log({
                    "minibatch/progress/val": prog, "val_step": total,
                    "minibatch/loss/val": mean_val_loss,
                })

            if rank == 0 and step == sample_batch_number:
                sample_output = output
                sample_fps = fps
                sample_info = info

        if rank == 0:
            learning_rate = optimizer.param_groups[0]["lr"]
            logger.info(f"computing training and validation loss")
            train_loss /= len(dataloader.dataset)
            val_loss /= len(valid_dataloader.dataset)
            logger.info(
                f"[epoch {epoch}] "
                f"train_loss={train_loss:0.08f}, "
                f"val_loss={loss:0.08f}, "
                f"learning_rate={learning_rate:0.08f}"
            )
            writer.add_scalar("epochs", epoch, epoch)
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            writer.add_scalar("learning_rate", learning_rate, epoch)

            wandb.log(
                {
                    "loss/train": train_loss,
                    "loss/val": val_loss,
                    "learning_rate": learning_rate,
                    "epoch": epoch,
                },
                commit=not (epoch % config.checkpoint_every == 0)
            )

        fp = config.model_output_path / "checkpoints" / str(epoch)
        checkpoint_path = fp / "model.pth"
        if epoch % config.checkpoint_every == 0 and rank == 0:
            # TODO: find out where training halts
            logger.info(f"creating checkpoint at epoch {epoch}")

            logger.info(f"checkpoint path: {fp}")
            fp.mkdir(parents=True)
            torch.save(model.state_dict(), checkpoint_path)

            module = model if not distributed else model.module.cuda(rank)
            torch.save(
                module.state_dict(), config.model_output_path / "model.pth"
            )

            logger.info("decoding output samples")
            output_samples = mu_law_decoding(
                sample_output.argmax(1), config.model_config.input_channels
            ).to("cpu")

            torch.save(output_samples, fp / "output_samples.pth")

            checkpoint_samples = wandb.Table(
                columns=["epoch", "id", "video", "orig_audio", "synth_audio"]
            )

            for i, (sample_fp, info, sample) in enumerate(
                zip(sample_fps, sample_info, output_samples)
            ):
                sample_fp = Path(sample_fp)
                # save original video files
                video_fp = str(fp / f"original_video_{i}_{sample_fp.stem}.mp4")
                _, orig_audio, _ = torchvision.io.read_video(
                    str(sample_fp)
                )
                shutil.copyfile(sample_fp, video_fp)
                # upsample audio to the original dimensionality
                resampled = torch.from_numpy(
                    librosa.resample(
                        sample.numpy(),
                        sample.shape[0],
                        info["audio_orig_dim"],
                    )
                )
                resampled = torch.stack([resampled, resampled])
                # save generated mp3 file
                synth_audio_fp = str(fp / f"synth_audio_{sample_fp.stem}.wav")
                orig_audio_fp = str(fp / f"orig_audio_{sample_fp.stem}.wav")
                torchaudio.save(
                    synth_audio_fp,
                    resampled,
                    sample_rate=info["audio_fps"],
                    format="wav",
                )
                torchaudio.save(
                    orig_audio_fp,
                    orig_audio,
                    sample_rate=info["audio_fps"],
                    format="wav",
                )
                # log on wandb
                caption = f"instance id: {sample_fp.stem}"
                checkpoint_samples.add_data(
                    epoch,
                    sample_fp.stem,
                    wandb.Video(video_fp, caption=caption),
                    wandb.Audio(orig_audio_fp, caption=caption),
                    wandb.Audio(synth_audio_fp, caption=caption),
                )

            wandb.run.log(
                {"checkpoint_samples": checkpoint_samples}, commit=True,
            )

        if distributed:
            dist.barrier()
        # wait for rank 0 to finish writing checkpoint
        logger.info(f"ending training loop for epoch {epoch}")
        if isinstance(model, nn.parallel.DistributedDataParallel):
            logger.info("loading distributed model from checkpoints")
            model.load_state_dict(
                torch.load(
                    checkpoint_path, map_location={"cuda:0": f"cuda:{rank}"}
                )
            )

    return model



def dist_train_model(
    rank: int,
    world_size: int,
    config: TrainingConfig,
    dataset_fp: str,
):
    configure_logging()

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = config.dist_port
    dist.init_process_group(
        config.dist_backend, rank=rank, world_size=world_size
    )
    if rank == 0:
        wandb_setup()
        wandb.config.update(json.loads(config.to_json()))

    model = train_model(config, dataset_fp, rank=rank, world_size=world_size)
    if rank == 0:
        wandb.finish()
        torch.save(
            model.module.cuda(rank).state_dict(),
            config.model_output_path / "model.pth"
        )

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
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", type=lambda x: bool(int(x)), default=False)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--n_steps_per_epoch", type=int, default=None)
    parser.add_argument("--checkpoint_every", type=int, default=1)
    parser.add_argument("--input_channels", type=int, default=16)
    parser.add_argument("--residual_channels", type=int, default=16)
    parser.add_argument("--layer_size", type=int, default=3)
    parser.add_argument("--stack_size", type=int, default=3)
    parser.add_argument("--dist_backend", type=str, default="nccl")
    parser.add_argument("--dist_port", type=str, default="8888")
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
        accumulation_steps=args.accumulation_steps,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        n_epochs=args.n_epochs,
        n_steps_per_epoch=args.n_steps_per_epoch,
        dist_backend=args.dist_backend,
        dist_port=args.dist_port,
        pretrained_model_path=(
            args.pretrained_model_path
            if args.pretrained_model_path and args.pretrained_run_exp_name
            else None
        ),
        model_output_path=args.model_output_path,
        tensorboard_dir=args.training_logs_path,
    )
    with (args.model_output_path / "config.json").open("w") as f:
        f.write(config.to_json())

    logger.info(f"config: {config}")

    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    logging.info(
        "Effective batch size: "
        f"{config.batch_size * world_size * config.accumulation_steps}"
    )
    if world_size > 1:
        logging.info(
            f"Launching distributed training job with world_size: {world_size}"
        )
        mp.spawn(
            dist_train_model,
            args=(world_size, config, args.dataset),
            nprocs=world_size,
            join=True
        )
    else:
        # initialize wandb
        wandb_setup()
        model = train_model(config, args.dataset)
        torch.save(model.state_dict(), args.model_output_path / "model.pth")
