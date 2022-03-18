from dataclasses import asdict

import torch
import torch.nn.functional as F

from movenet.callbacks import LogSamplesCallback
from movenet.config import TrainingConfig, config_from_args, arg_parser
from movenet.dataset import get_dataloader
from movenet.wavenet import WaveNet

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger


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
        audio, video = audio.to(self.device), video.to(self.device)
        output = self(audio, video)

        target = audio[:, :, self.model.receptive_fields:].argmax(1)
        loss = F.cross_entropy(output, target)
        self.log("train_loss", loss)
        return {
            "loss": loss,
            "output": output,
        }

    def validation_step(self, batch, batch_idx):
        audio, video, contexts, fps, info = batch
        # need to do this manually since batch contains non-tensors
        audio, video = audio.to(self.device), video.to(self.device)
        output = self(audio, video)

        target = audio[:, :, self.model.receptive_fields:].argmax(1)
        loss = F.cross_entropy(output, target)
        self.log("val_loss", loss)

        generated_output = None
        if batch_idx == 0:
            generated_output = self(
                audio,
                video,
                generate=True,
                n_samples=self.config.generate_n_samples,
            )

        return {
            "output": output,
            "generated_output": generated_output,
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
        )

    def val_dataloader(self):
        return get_dataloader(
            self.dataset_fp,
            input_channels=self.config.model_config.input_channels,
            batch_size=self.config.batch_size,
            train=False,
            pin_memory=self.config.pin_memory,
            num_workers=self.config.val_num_workers,
        )


def train_model(dataset: str, config: TrainingConfig, wandb_project: str):
    model = Dance2Music(dataset, config)
    wandb_logger = WandbLogger(
        project=wandb_project,
        log_model="all",
    )
    trainer = Trainer(
        max_epochs=config.n_epochs,
        default_root_dir=config.model_output_path,
        logger=wandb_logger,
        log_every_n_steps=1,
        num_sanity_val_steps=1,
        callbacks=[LogSamplesCallback()],
    )
    wandb_logger.watch(model, log="all", log_freq=1)
    trainer.fit(model=model)


if __name__ == "__main__":
    parser = arg_parser()
    args = parser.parse_args()
    config = config_from_args(args)

    train_model(args.dataset, config, "dance2music-pl-testing")
