"""Train the movenet model."""

from dataclasses import dataclass, asdict
from typing import List

import torch
import torch.optim
import torch.nn.functional as F
from torchaudio.functional import mu_law_encoding, mu_law_decoding
from torchtyping import TensorType

from movenet import dataset
from movenet.wavenet import WaveNet


@dataclass
class ModelConfig:
    layer_size: int = 2
    stack_size: int = 2
    input_channels: int = 256
    residual_channels: int = 512


@dataclass
class TrainingConfig:
    model_config: ModelConfig = ModelConfig()
    optimizer: str = "Adam"
    learning_rate: float = 0.0002
    weight_decay: float = 0.0
    n_training_steps: int = 100


def one_hot_encode(audio, input_channels):
    # need to figure out a more principled way of combining two audio
    # (left/right) channels into one
    # https://stackoverflow.com/questions/37313320/how-to-convert-two-channel-audio-into-one-channel-audio
    combined_channels = audio.mean(dim=0, keepdims=True)
    quantized = mu_law_encoding(combined_channels, input_channels)
    one_hot_enc = (
        torch.zeros(input_channels, quantized.size(1))
        .scatter_(0, quantized, 1)
    )
    assert (one_hot_enc.sum(dim=0) == 1).all(), "one hot encoding error"
    return one_hot_enc


def zero_padded_stack(tensors: List[TensorType["channels", "frames"]]):
    max_frames = max(x.size(1) for x in tensors)
    output = []
    for x in tensors:
        padded_x = torch.zeros(x.size(0), max_frames)
        padded_x[:, :x.size(1)] = x
        output.append(padded_x)
    return torch.stack(output)


def make_batch(examples: List[dataset.Example], config: TrainingConfig):
    audio = zero_padded_stack(
        [
            one_hot_encode(x.audio, config.model_config.input_channels)
            for x in examples
        ]
    )
    return audio


def train_model(config: TrainingConfig, batch_fps: List[str]):
    model = WaveNet(**asdict(config.model_config))
    optimizer = getattr(torch.optim, config.optimizer)(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    # training loop
    raw_data = [dataset.load_video(fp) for fp in batch_fps]
    audio = make_batch(raw_data, config)
    for i in range(1, config.n_training_steps + 1):
        output = model(audio)
        target = audio[:, :, model.receptive_fields:].argmax(1)
        loss = F.cross_entropy(output, target)
        optimizer.zero_grad()
        loss.backward()

        grad_norm = 0.
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2

        grad_norm = grad_norm ** 0.5
        optimizer.step()

        print(f"[step {i}] loss={loss.data.item():0.08f}, {grad_norm=:0.08f}")
    return model


if __name__ == "__main__":
    import json
    from pathlib import Path
    from datetime import datetime

    data_root = Path("datasets") / "kinetics" / "train" / "breakdancing"
    model_root = Path("models")
    config = TrainingConfig(
        model_config=ModelConfig(
            input_channels=16,
            residual_channels=16,
            layer_size=3,
            stack_size=3,
        ),
        learning_rate=0.0003,
        n_training_steps=3,
    )
    batch_fps = [
        str(data_root / file_name)
        for file_name in [
            "zkyRFux7BWc.mp4",
            "eB4wwvnXwrI.mp4",
            "MEguK5_ding.mp4",
        ]
    ]
    model = train_model(config, batch_fps)
    model_path = model_root / datetime.now().strftime("%Y%m%d%H%M%S")
    model_path.mkdir(exist_ok=True, parents=True)
    with (model_path / "config.json").open("w") as f:
        json.dump(asdict(config), f, indent=4)
    torch.save(model, model_path / "model.pth")
