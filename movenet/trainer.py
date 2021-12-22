"""Train the movenet model."""

import json
import gc
import math
import logging
import os
import shutil
from dataclasses import dataclass, asdict, field
from dataclasses_json import dataclass_json, config
from pathlib import Path
from time import time
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

GRID_EXPERIMENT_NAME = os.getenv("GRID_EXPERIMENT_NAME", None)
WANDB_PROJECT = "dance2music-local"
if GRID_EXPERIMENT_NAME:
    WANDB_PROJECT = "dance2music"

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
        project=WANDB_PROJECT,
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

    # sample generation
    generate_n_samples: Optional[int] = None

    # found through learning rate range experiment:
    # https://wandb.ai/nielsbantilan/dance2music/runs/3a4sfxev?workspace=user-nielsbantilan
    max_learning_rate: float = 0.00003
    lr_pct_start: float = 0.45

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
    scheduler,
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
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    return loss.detach().cpu().item(), grad_norm


@torch.no_grad()
def validation_step(model, audio, video, rank=0, n_samples=None):
    generated_loss = 0.0
    generated_output = None
    with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
        if torch.cuda.is_available():
            audio, video = audio.cuda(rank), video.cuda(rank)
        output = model(audio, video)
        target = audio[:, :, model.receptive_fields:].argmax(1)
        loss = F.cross_entropy(output, target).detach().item()

        if n_samples:
            start = time()
            generated_output = model(
                audio, video, generate=True, n_samples=n_samples
            )
            generated_loss = F.cross_entropy(
                generated_output, target[:, :n_samples]
            ).detach().item()
            logger.info(f"sample generation took {time() - start} seconds")

    return loss, output, generated_loss, generated_output


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

    n_updates = math.ceil(len(dataloader) / config.accumulation_steps)

    optimizer = getattr(torch.optim, config.optimizer)(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = getattr(torch.optim.lr_scheduler, config.scheduler)(
        optimizer,
        max_lr=config.max_learning_rate,
        epochs=config.n_epochs,
        steps_per_epoch=n_updates,
        pct_start=config.lr_pct_start,
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
        batch_train_loss = []
        logger.info(f"starting training loop for epoch {epoch}")
        total_n_steps = config.n_steps_per_epoch or len(dataloader)
        for step, (audio, video, contexts, fps, _) in enumerate(dataloader, 1):
            last_step = step == total_n_steps
            loss, grad_norm = training_step(
                config, step, model, optimizer, scheduler,
                audio, video, rank, scaler, last_step=last_step
            )
            train_loss += loss
            batch_train_loss.append(loss)

            prog = step / total_n_steps
            total = epoch * total_n_steps + step
            if (
                rank == 0
                and (step % config.accumulation_steps == 0 or last_step)
            ):
                batch_train_loss = sum(batch_train_loss) / len(batch_train_loss)
                batch_lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"[epoch {epoch} | step {step}] "
                    f"batch_progress={prog}, "
                    f"minibatch_loss={batch_train_loss:0.08f}, "
                    f"minibatch_grad_norm={grad_norm:0.08f}, "
                    f"minibatch_filepaths={fps}"
                )
                writer.add_scalar("minibatch/progress/train", prog, total)
                writer.add_scalar("minibatch/loss/train", batch_train_loss, total)
                writer.add_scalar("minibatch/grad_norm", grad_norm, total)
                writer.add_scalar("minibatch/learning_rate", batch_lr, total)

                wandb.log({
                    "minibatch/progress/train": prog,
                    "minibatch/loss/train": batch_train_loss,
                    "minibatch/grad_norm": grad_norm,
                    "minibatch/learning_rate": batch_lr,
                    "train_step": total,
                })
                batch_train_loss = []

            if config.n_steps_per_epoch and step > config.n_steps_per_epoch:
                break

        val_loss = 0.0
        val_gen_loss = 0.0
        sample_output = None
        sample_generated_output = None
        sample_fps = None
        logger.info(f"starting validation loop for epoch {epoch}")
        logger.info(f"sample_batch_number: {sample_batch_number}")

        model.eval()
        for step, (audio, video, contexts, fps, info) in enumerate(
            valid_dataloader, 1
        ):
            n_samples = None
            if step == sample_batch_number:
                n_samples = (
                    config.generate_n_samples
                    or audio.shape[-1] - model.receptive_fields
                )
                logger.info(f"generating {n_samples} samples for step {step}")
            _val_loss, output, _gen_loss, generated_output = validation_step(
                model, audio, video, rank, n_samples=n_samples,
            )
            # wait for all processes to complete the validation step
            if distributed:
                dist.barrier()

            val_loss += _val_loss
            val_gen_loss += _gen_loss
            prog = step / len(valid_dataloader)
            total = epoch * len(valid_dataloader) + step
            if rank == 0:
                logger.info(
                    f"[epoch {epoch} | step {step}] "
                    f"batch_progress={prog}, "
                    f"minibatch_loss={_val_loss:0.08f}"
                )
                writer.add_scalar("minibatch/progress/val", prog, total)
                writer.add_scalar("minibatch/loss/val", _val_loss, total)
                writer.add_scalar("minibatch/loss/gen/val", _gen_loss, total)
                wandb.log({
                    "minibatch/progress/val": prog,
                    "minibatch/loss/val": _val_loss,
                    "minibatch/loss/gen/val": _gen_loss,
                    "val_step": total,
                })

            if rank == 0 and step == sample_batch_number:
                sample_synth_output = output
                sample_generated_output = generated_output
                if torch.cuda.is_available():
                    sample_synth_output = sample_synth_output.to(rank)
                    sample_generated_output = sample_generated_output.to(rank)
                sample_fps = fps
                sample_info = info

            if distributed:
                dist.barrier()

        if rank == 0:
            learning_rate = optimizer.param_groups[0]["lr"]
            logger.info(f"computing training and validation loss")
            train_loss /= n_updates
            val_loss /= len(valid_dataloader.dataset)
            val_gen_loss /= len(valid_dataloader.dataset)
            logger.info(
                f"[epoch {epoch}] "
                f"train_loss={train_loss:0.08f}, "
                f"val_loss={val_loss:0.08f}, "
                f"val_gen_loss={val_gen_loss:0.08f}, "
                f"learning_rate={learning_rate:0.08f}"
            )
            writer.add_scalar("epochs", epoch, epoch)
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            writer.add_scalar("loss/gen/val", val_gen_loss, epoch)
            writer.add_scalar("learning_rate", learning_rate, epoch)

            wandb.log(
                {
                    "loss/train": train_loss,
                    "loss/val": val_loss,
                    "loss/gen/val": val_gen_loss,
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
            synth_output_samples = mu_law_decoding(
                sample_synth_output.argmax(1),
                config.model_config.input_channels
            ).to("cpu")

            gen_output_samples = mu_law_decoding(
                sample_generated_output.argmax(1),
                config.model_config.input_channels
            ).to("cpu")

            torch.save(synth_output_samples, fp / "synth_output_samples.pth")
            torch.save(gen_output_samples, fp / "gen_output_samples.pth")

            checkpoint_samples = wandb.Table(
                columns=[
                    "epoch",
                    "id",
                    "video",
                    "orig_audio",
                    "synth_audio",  # given full information on audio signal
                    "generated_audio",  # generated from scratch
                ]
            )

            for i, (sample_fp, info, synth_sample, gen_sample) in enumerate(
                zip(
                    sample_fps,
                    sample_info,
                    synth_output_samples,
                    gen_output_samples,
                )
            ):
                sample_fp = Path(sample_fp)
                # save original video files
                video_fp = str(fp / f"original_video_{i}_{sample_fp.stem}.mp4")
                _, orig_audio, _ = torchvision.io.read_video(
                    str(sample_fp)
                )
                shutil.copyfile(sample_fp, video_fp)
                # upsample audio to the original dimensionality
                synth_sample = torch.from_numpy(
                    librosa.resample(
                        synth_sample.numpy(),
                        synth_sample.shape[0],
                        info["audio_orig_dim"],
                    )
                )
                if config.generate_n_samples is None:
                    gen_sample = torch.from_numpy(
                        librosa.resample(
                            gen_sample.numpy(),
                            gen_sample.shape[0],
                            info["audio_orig_dim"],
                        )
                    )
                synth_sample = torch.stack([synth_sample, synth_sample])
                gen_sample = torch.stack([gen_sample, gen_sample])
                # save generated mp3 file
                synth_audio_fp = fp / f"synth_audio_{sample_fp.stem}.wav"
                gen_audio_fp = fp / f"gen_audio_{sample_fp.stem}.wav"
                orig_audio_fp = fp / f"orig_audio_{sample_fp.stem}.wav"
                for ext in ["wav", "mp3"]:
                    torchaudio.save(
                        str(synth_audio_fp.with_suffix(f".{ext}")),
                        synth_sample,
                        sample_rate=info["audio_fps"],
                        format=ext,
                    )
                    torchaudio.save(
                        str(gen_audio_fp.with_suffix(f".{ext}")),
                        gen_sample,
                        sample_rate=info["audio_fps"],
                        format=ext,
                    )
                    torchaudio.save(
                        str(orig_audio_fp.with_suffix(f".{ext}")),
                        orig_audio,
                        sample_rate=info["audio_fps"],
                        format=ext,
                    )
                # log on wandb
                caption = f"instance id: {sample_fp.stem}"
                checkpoint_samples.add_data(
                    epoch,
                    sample_fp.stem,
                    wandb.Video(video_fp, caption=caption),
                    wandb.Audio(str(orig_audio_fp), caption=caption),
                    wandb.Audio(str(synth_audio_fp), caption=caption),
                    wandb.Audio(str(gen_audio_fp), caption=caption),
                )

            wandb.run.log(
                {"checkpoint_samples": checkpoint_samples}, commit=True,
            )
            logger.info(f"finished created checkpoint for epoch {epoch}")

        if distributed:
            dist.barrier()
        # wait for rank 0 to finish writing checkpoint
        logger.info(f"ending training loop for epoch {epoch}")

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
    parser.add_argument(
        "--pin_memory", type=lambda x: bool(int(x)), default=False,
    )
    parser.add_argument(
        "--generate_n_samples",
        type=lambda x: x if x is None else int(x),
        default=None
    )
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
        generate_n_samples=args.generate_n_samples,
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
