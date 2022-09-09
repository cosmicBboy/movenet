import numpy as np
import pandas as pd
import torch
import torch.optim as opt
import torch.nn.functional as F

from torchaudio.functional import mu_law_encoding, mu_law_decoding

from movenet.wavenet import WaveNet

import pytest


INPUT_TIME_LENGTH = 400
SAMPLE_RATE_FRAC = 0.1
INPUT_CHANNELS = 256
BATCH_SIZE = 64


@pytest.fixture
def sine_wave():
    quantized = mu_law_encoding(
        torch.from_numpy(
            np.sin(np.arange(0, INPUT_TIME_LENGTH, SAMPLE_RATE_FRAC))
        ),
        quantization_channels=INPUT_CHANNELS,
    ).unsqueeze(0)

    one_hot = (
        torch.zeros(INPUT_CHANNELS, quantized.size(1))
        .scatter_(0, quantized, 1)
    ).unsqueeze(0)

    out = {
        "quantized": torch.concat([quantized for _ in range(4)], dim=0),
        "one_hot": torch.concat([one_hot for _ in range(4)], dim=0),
    }
    return out


def test_wavenet_model(sine_wave):
    wavenet = WaveNet(
        layer_size=10,
        stack_size=3,
        input_channels=INPUT_CHANNELS,
        residual_channels=64,
        skip_channels=64,
    )
    print(f"receptive fields: {wavenet.receptive_fields}")

    optim = opt.Adam(wavenet.parameters(), lr=0.0003)

    target = sine_wave["quantized"][:, wavenet.receptive_fields:]
    losses = []
    accs = []
    for step in range(1000):
        optim.zero_grad()
        output = wavenet(sine_wave["one_hot"])

        assert output.shape[-1] == target.shape[-1]
        loss = F.cross_entropy(output, target)
        acc = (output.argmax(1) == target).numpy().astype(int).mean()
        print(f"step={step:04} loss={loss:0.08f} acc={acc:0.08f}")
    
        loss.backward()
        losses.append(loss.data.item())
        accs.append(float(acc))
        optim.step()

    decoded_target = pd.DataFrame(
        mu_law_decoding(target, INPUT_CHANNELS).numpy().transpose()
    )
    decoded_output = pd.DataFrame(
        mu_law_decoding(output.argmax(1), INPUT_CHANNELS).numpy().transpose()
    )

    with torch.no_grad():
        upper = wavenet.receptive_fields + 200
        generated_audio = wavenet.eval().generate(
            sine_wave["one_hot"][..., :wavenet.receptive_fields],
            n_samples=upper,
            temperature=0.0,
        )

    generated_output = pd.DataFrame(
        mu_law_decoding(
            generated_audio[..., wavenet.receptive_fields: upper].argmax(1),
            INPUT_CHANNELS
        )
        .numpy().transpose()
    )

    comparison = pd.concat(
        [
            decoded_output[0].rename("pred"),
            decoded_target[0].rename("target"),
            generated_output[0].rename("generated"),
        ],
        axis=1
    )


    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows=4)
    axes = axes.ravel()
    comparison[["target", "pred"]].head(200).plot(ax=axes[0])
    comparison[["target", "generated"]].dropna().tail(200).plot(ax=axes[1])
    pd.Series(losses).plot(ax=axes[2])
    pd.Series(accs).plot(ax=axes[3])

    fig.savefig("./plot.png")
