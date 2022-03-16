from pathlib import Path

import librosa
import torch
import torchvision.io
import wandb
from torchaudio.functional import mu_law_decoding
from pytorch_lightning.callbacks import Callback

COLUMNS = [
    "epoch",
    "batch_idx",
    "dataloader_idx",
    "fp",
    "video",
    "origin_audio",
    "pred_audio",
    "gen_audio",
]
 
class LogSamplesCallback(Callback):
    
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if batch_idx > 0:
            return

        config = pl_module.config
        _, _, _, fps, infos = batch

        pred_outputs = mu_law_decoding(
            outputs["output"].argmax(1),
            config.model_config.input_channels
        )

        gen_outputs = [None] * len(fps)
        if outputs["generated_output"] is not None:
            gen_outputs = mu_law_decoding(
                outputs["generated_output"].argmax(1),
                config.model_config.input_channels
            )

        data = []
        for fp, info, pred_output, gen_output in zip(
            fps, infos, pred_outputs, gen_outputs
        ):
            _, orig_audio, vid_info = torchvision.io.read_video(fp)

            pred_audio = torch.from_numpy(
                librosa.resample(
                    pred_output.numpy(),
                    pred_output.shape[0],
                    vid_info["audio_fps"],
                )
            )

            if gen_output is not None:
                gen_audio = gen_output
                gen_sample_rate = len(gen_output)
                if not pl_module.config.generate_n_samples:
                    gen_sample_rate = vid_info["audio_fps"]
                    gen_audio = torch.from_numpy(
                        librosa.resample(
                            gen_output.numpy(),
                            gen_output.shape[0],
                            gen_sample_rate,
                        )
                    )
            else:
                gen_audio = torch.zeros_like(pred_audio)

            data.append([
                trainer.current_epoch,
                batch_idx,
                dataloader_idx,
                fp,
                wandb.Video(fp),
                wandb.Audio(orig_audio[0], sample_rate=vid_info["audio_fps"]),
                wandb.Audio(pred_audio, sample_rate=vid_info["audio_fps"]),
                wandb.Audio(gen_audio, sample_rate=gen_sample_rate),
            ])

        pl_module.logger.log_table(
            key='sample_output', columns=COLUMNS, data=data
        )
