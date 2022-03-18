from dataclasses import asdict

import torch
import torch.nn.functional as F

from movenet.callbacks import LogSamplesCallback
from movenet.config import TrainingConfig, config_from_args, arg_parser
from movenet.dataset import get_dataloader
from movenet.wavenet import WaveNet

from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger


class Dance2Music(LightningModule):

    def __init__(self, dataset_fp: str, config: TrainingConfig):
        super().__init__()
        self.dataset_fp = dataset_fp
        self.config = config
        self.model = WaveNet(**asdict(config.model_config))

    def configure_optimizers(self):
        return getattr(torch.optim, self.config.optimizer)(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def forward(self, audio, video, generate=False, n_samples=None):
        return self.model(audio, video, generate=generate, n_samples=n_samples)

    def training_step(self, batch, batch_idx):
        audio, video, contexts, fps, info = batch
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


if __name__ == "__main__":
    from pytorch_lightning import Trainer

    parser = arg_parser()
    args = parser.parse_args()
    config = config_from_args(args)

    model = Dance2Music(args.dataset, config)
    wandb_logger = WandbLogger(
        project="dance2music-pl-testing",
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
