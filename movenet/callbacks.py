from pathlib import Path

import librosa
import torch
import torchvision.io
import wandb
from torchaudio.functional import mu_law_decoding, resample
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import Callback

from movenet.wavenet import MAX_AUDIO_FRAMES


COLUMNS = [
    "split",
    "epoch",
    "batch_idx",
    "fp",
    "origin_audio",
    "pred_audio",
    "gen_audio",
]
 
class LogSamplesCallback(Callback):

    def __init__(
        self,
        log_every_n_epochs: int = 10,
        log_video: bool = True,
        temperature: float = 1.0,
    ):
        self.log_every_n_epochs = log_every_n_epochs
        self.log_video = log_video
        self.temperature = temperature

        self.columns = COLUMNS
        if self.log_video:
            self.columns += ["video"]

    def on_train_batch_end(
        self, trainer: Trainer, pl_module, outputs, batch, batch_idx
    ):
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return

        self.log_samples(
            "train", trainer, pl_module, outputs, batch, batch_idx
        )

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return

        self.log_samples(
            "validation", trainer, pl_module, outputs, batch, batch_idx
        )
    
    def log_samples(
        self, split, trainer, pl_module, outputs, batch, batch_idx
    ):
        config = pl_module.config
        _, _, _, fps, infos = batch

        pred_outputs = mu_law_decoding(
            outputs["output"].cpu().argmax(1),
            config.model_config.input_channels
        )

        gen_outputs = [None] * len(fps)
        if outputs.get("generated_output", None) is not None:
            gen_outputs = mu_law_decoding(
                outputs["generated_output"].cpu().argmax(1),
                config.model_config.input_channels
            )

        data = []
        for fp, info, pred_output, gen_output in zip(
            fps, infos, pred_outputs, gen_outputs
        ):
            _, orig_audio, vid_info = torchvision.io.read_video(
                fp, pts_unit="sec"
            )

            pred_audio = torch.from_numpy(
                librosa.resample(
                    pred_output.numpy(),
                    orig_sr=MAX_AUDIO_FRAMES / 10,  # 16,000
                    target_sr=info["audio_fps"],  # 48,000
                )
            )

            if gen_output is not None:
                gen_audio = gen_output
                if pl_module.config.generate_n_samples:
                    # scale number of samples in downsampled data space
                    # (to MAX_AUDIO_FRAMES) to number of sample in the original
                    # audio
                    target_dim = (
                        pl_module.config.generate_n_samples
                        / MAX_AUDIO_FRAMES
                        * orig_audio.shape[1]
                    )
                else:
                    # upsample to original audio dims
                    target_dim = orig_audio.shape[1]
                gen_audio = torch.from_numpy(
                    librosa.resample(
                        gen_output.numpy(),
                        orig_sr=gen_output.shape[0],
                        target_sr=target_dim,
                    )
                )
            else:
                gen_audio = torch.zeros_like(pred_audio)

            row = [
                split,
                trainer.current_epoch,
                batch_idx,
                fp,
                wandb.Audio(orig_audio[0], sample_rate=vid_info["audio_fps"]),
                wandb.Audio(pred_audio, sample_rate=vid_info["audio_fps"]),
                wandb.Audio(gen_audio, sample_rate=vid_info["audio_fps"]),
            ]
            if self.log_video:
                row += [wandb.Video(fp)]

            data.append(row)

        pl_module.logger.log_table(
            key='sample_output', columns=self.columns, data=data
        )
