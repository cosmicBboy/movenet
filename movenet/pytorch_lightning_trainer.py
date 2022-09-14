import math
from dataclasses import asdict
from typing import Optional

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
import torch.nn.functional as F

from movenet.callbacks import LogSamplesCallback
from movenet.config import TrainingConfig, config_from_args, arg_parser
from movenet.dataset import get_dataloader
from movenet.wavenet import WaveNet

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor


# NOTE: consider rename: move2music
class Dance2Music(LightningModule):

    def __init__(self, dataset_fp: str, config: TrainingConfig):
        super().__init__()
        self.learning_rate = config.learning_rate
        self.dataset_fp = dataset_fp
        self.config = config
        self.model = WaveNet(**asdict(config.model_config))

    def forward(self, audio, video, **kwargs):
        return self.model(audio, video, **kwargs)

    def training_step(self, batch, batch_idx):
        audio, video, contexts, fps, info = batch

        # need to do this manually since batch contains non-tensors
        dtype = getattr(torch, f"float{self.precision}")
        audio = audio.type(dtype).to(self.device)

        if self.config.use_video:
            video = video.type(dtype).to(self.device)

        output = self(audio, video)

        target = audio[:, :, self.model.receptive_fields:].argmax(1)
        loss = F.cross_entropy(output, target)
        acc = (output.argmax(1) == target).float().mean()
        self.log("train_loss", loss, batch_size=self.config.batch_size)
        self.log("train_acc", acc, batch_size=self.config.batch_size)

        generated_output = None
        if (
            self.config.log_samples_every is not None
            and (self.current_epoch + 1) % self.config.log_samples_every == 0
        ):
            generated_output = (
                self.model.generate(
                    audio,
                    video,
                    n_samples=self.config.generate_n_samples,
                    temperature=self.config.generate_temperature,
                )
            )

        return {
            "loss": loss,
            "output": output.detach(),
            "generated_output": generated_output.detach(),
        }

    def validation_step(self, batch, batch_idx):
        audio, video, contexts, fps, info = batch

        # need to do this manually since batch contains non-tensors
        dtype = getattr(torch, f"float{self.precision}")
        audio = audio.type(dtype).to(self.device)

        if self.config.use_video:
            video = video.type(dtype).to(self.device)

        output = self(audio, video)

        target = audio[:, :, self.model.receptive_fields:].argmax(1)
        loss = F.cross_entropy(output, target)
        acc = (output.argmax(1) == target).float().mean()
        self.log("val_loss", loss, batch_size=self.config.batch_size)
        self.log("val_acc", acc, batch_size=self.config.batch_size)

        generated_output = None
        if (
            self.config.log_samples_every is not None
            and (self.current_epoch + 1) % self.config.log_samples_every == 0
        ):
            generated_output = (
                self.model.generate(
                    audio,
                    video,
                    n_samples=self.config.generate_n_samples,
                    temperature=self.config.generate_temperature,
                )
            )

        return {
            "output": output.detach(),
            "generated_output": generated_output.detach(),
        }

    def train_dataloader(self):
        return get_dataloader(
            self.dataset_fp,
            input_channels=self.config.model_config.input_channels,
            batch_size=self.config.batch_size,
            train=True,
            shuffle=True,
            pin_memory=self.config.pin_memory,
            num_workers=self.config.num_workers,
            use_video=self.config.use_video,
        )

    def val_dataloader(self):
        return get_dataloader(
            self.dataset_fp,
            input_channels=self.config.model_config.input_channels,
            batch_size=self.config.batch_size,
            train=False,
            pin_memory=self.config.pin_memory,
            num_workers=self.config.val_num_workers,
            use_video=self.config.use_video,
        )

    def configure_optimizers(self):

        def optimizer_kwargs(optimizer: str):
            kws = {
                "lr": self.learning_rate,
                "weight_decay": self.config.weight_decay,
            }
            opts = {
                "Adam": kws,
                "AdamW": kws,
                "SGD": {**kws, "momentum": self.config.momentum},
                "RMSprop": {**kws, "momentum": self.config.momentum},
            }
            if optimizer not in opts:
                raise ValueError(
                    f"optimizer {optimizer} not recognized. "
                    f"Must be one of {opts.keys()}"
                )
            return opts[optimizer]

        def scheduler_kwargs(scheduler: str):
            dataloader = self.train_dataloader()
            n_updates_per_epoch = math.ceil(
                len(dataloader) / self.config.accumulation_steps
            )
            sched = {
                "OneCycleLR": {
                    "max_lr": self.config.max_learning_rate,
                    "epochs": self.config.n_epochs,
                    "steps_per_epoch": n_updates_per_epoch,
                    "pct_start": self.config.lr_pct_start,
                    "three_phase": True,
                },
                "CyclicLR": {
                    "base_lr": self.config.base_learning_rate,
                    "max_lr": self.config.max_learning_rate,
                    "step_size_up": self.config.scheduler_step_size_up,
                    "step_size_down": self.config.scheduler_step_size_down,
                    "mode": self.config.scheduler_cyclic_mode,
                    "gamma": self.config.scheduler_cyclic_gamma,
                    "cycle_momentum": self.config.scheduler_cycle_momentum,
                },
                "StepLR": {
                    "step_size": self.config.scheduler_step_size,
                    "gamma": self.config.scheduler_step_gamma,
                },
                "MultiStepLR": {
                    "milestones": self.config.scheduler_milestones,
                    "gamma": self.config.scheduler_step_gamma,
                },
            }
            if scheduler not in sched:
                raise ValueError(
                    f"scheduler {scheduler} not recognized. "
                    f"Must be one of {sched.keys()}"
                )
            return sched[scheduler]

        optimizer = getattr(torch.optim, self.config.optimizer)(
            self.model.parameters(),
            **optimizer_kwargs(self.config.optimizer)
        )
        print(f"using optimizer: {optimizer}")
        optimizers = {"optimizer": optimizer}

        if self.config.scheduler is not None:
            optimizers["lr_scheduler"] = getattr(
                torch.optim.lr_scheduler, self.config.scheduler
            )(optimizer, **scheduler_kwargs(self.config.scheduler))
            print(f"using scheduler: {optimizers['lr_scheduler']}")

        return optimizers

def train_model(
    dataset: str,
    config: TrainingConfig,
    logger_name: Optional[Literal["wandb"]] = None,
    log_video: bool = False,
    wandb_project: Optional[str] = None,
):
    model = Dance2Music(dataset, config)

    logger = None
    callbacks = []
    if logger_name == "wandb":
        logger = WandbLogger(
            project=wandb_project,
            log_model="all",
        )
        logger.experiment.config.update(asdict(config))
        logger.experiment.config.update({"dataset": dataset})
        callbacks.append(LearningRateMonitor(logging_interval="step"))
        if config.log_samples_every:
            callbacks.append(
                LogSamplesCallback(
                    log_every_n_epochs=config.log_samples_every,
                    log_video=log_video,
                )
            )

    print(f"Using logger: {logger}")

    trainer = Trainer(
        max_epochs=config.n_epochs,
        default_root_dir=config.model_output_path,
        gradient_clip_val=config.gradient_clipping,
        logger=logger,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        track_grad_norm=2,
    )
    if logger:
        logger.watch(model, log="all", log_freq=1)

    trainer.fit(model=model)


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s: %(levelname)s: %(name)s: %(message)s"
    )

    parser = arg_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    train_model(
        args.dataset,
        config,
        logger_name=args.logger,
        log_video=args.log_video,
        wandb_project=args.wandb_project,
    )
