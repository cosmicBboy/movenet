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


# NOTE: consider rename: move2music
class Dance2Music(LightningModule):

    def __init__(self, dataset_fp: str, config: TrainingConfig):
        super().__init__()
        self.learning_rate = config.learning_rate
        self.dataset_fp = dataset_fp
        self.config = config
        self.model = WaveNet(**asdict(config.model_config))

    def configure_optimizers(self):
        return getattr(torch.optim, self.config.optimizer)(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def forward(self, audio, video, generate=False, n_samples=None):
        return self.model(audio, video, generate=generate, n_samples=n_samples)

    def training_step(self, batch, batch_idx):
        audio, video, contexts, fps, info = batch
        # need to do this manually since batch contains non-tensors
        audio = audio.to(self.device)
        video = video.to(self.device) if self.config.use_video else video
        output = self(audio, video)

        target = audio[:, :, self.model.receptive_fields:].argmax(1)
        loss = F.cross_entropy(output, target)
        self.log("train_loss", loss, batch_size=self.config.batch_size)
        return {
            "loss": loss,
            "output": output.detach(),
        }

    def validation_step(self, batch, batch_idx):
        audio, video, contexts, fps, info = batch
        # need to do this manually since batch contains non-tensors
        audio = audio.to(self.device)
        video = video.to(self.device) if self.config.use_video else video
        output = self(audio, video)

        target = audio[:, :, self.model.receptive_fields:].argmax(1)
        loss = F.cross_entropy(output, target)
        self.log("val_loss", loss, batch_size=self.config.batch_size)

        return {"output": output.detach()}

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


def train_model(
    dataset: str,
    config: TrainingConfig,
    logger_name: Optional[Literal["wandb"]] = None,
    log_samples: bool = False,
    wandb_project: Optional[str] = None,
):
    model = Dance2Music(dataset, config)

    callbacks = None
    if log_samples:
        callbacks = [LogSamplesCallback(log_every_n_epochs=10)]

    logger = None
    if logger_name == "wandb":
        logger = WandbLogger(
            project=wandb_project,
            log_model="all",
        )

    print(f"Using logger: {logger}")

    trainer = Trainer(
        max_epochs=config.n_epochs,
        default_root_dir=config.model_output_path,
        logger=logger,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        callbacks=callbacks,
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
        log_samples=args.log_samples,
    )
