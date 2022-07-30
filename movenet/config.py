import argparse
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dataclasses_json import dataclass_json, config


@dataclass_json
@dataclass
class ModelConfig:
    layer_size: int = 2
    stack_size: int = 2
    input_channels: int = 256
    residual_channels: int = 16
    skip_channels: int = 16


@dataclass_json
@dataclass
class TrainingConfig:
    # model hyperparameters
    model_config: ModelConfig = ModelConfig()

    # training parameters
    batch_size: int = 3
    checkpoint_every: int = 25
    optimizer: str = "AdamW"
    scheduler: str = "OneCycleLR"
    learning_rate: float = 0.0002
    accumulation_steps: int = 1
    num_workers: int = 0
    val_num_workers: int = 0
    pin_memory: bool = False
    weight_decay: float = 0.0
    n_epochs: int = 100
    n_steps_per_epoch: Optional[int] = None
    use_video: bool = True

    # sample generation
    generate_n_samples: Optional[int] = None

    # found through learning rate range experiment:
    # https://wandb.ai/nielsbantilan/dance2music/runs/3a4sfxev?workspace=user-nielsbantilan
    max_learning_rate: float = 0.003
    lr_pct_start: float = 0.45

    # distributed compute
    dist_backend: str = None
    dist_port: str = "8888"

    # model IO
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

    # logging
    tensorboard_dir: Path = field(
        default="tensorboard_logs", metadata=config(encoder=str, decoder=Path),
    )


def config_from_args(args) -> TrainingConfig:
    return TrainingConfig(
        model_config=ModelConfig(
            input_channels=args.input_channels,
            residual_channels=args.residual_channels,
            skip_channels=args.skip_channels,
            layer_size=args.layer_size,
            stack_size=args.stack_size,
        ),
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
        learning_rate=args.learning_rate,
        max_learning_rate=args.max_learning_rate,
        generate_n_samples=args.generate_n_samples,
        accumulation_steps=args.accumulation_steps,
        num_workers=args.num_workers,
        val_num_workers=args.val_num_workers,
        pin_memory=args.pin_memory,
        weight_decay=args.weight_decay,
        n_epochs=args.n_epochs,
        n_steps_per_epoch=args.n_steps_per_epoch,
        use_video=args.use_video,
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


def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=0.003)
    parser.add_argument("--max_learning_rate", type=float, default=0.003)
    parser.add_argument("--weight_decay", type=float, default=0.000)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--val_num_workers", type=int, default=1)
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
    parser.add_argument(
        "--use_video",
        type=lambda x: bool(int(x)),
        default=True,
    )
    parser.add_argument("--checkpoint_every", type=int, default=1)
    parser.add_argument("--input_channels", type=int, default=16)
    parser.add_argument("--residual_channels", type=int, default=16)
    parser.add_argument("--skip_channels", type=int, default=8)
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
    parser.add_argument("--logger", default=None, type=str, choices=["wandb"])
    parser.add_argument("--log_samples", default=False, action="store",)
    parser.add_argument("--wandb_api_key", type=str, default="")
    return parser
