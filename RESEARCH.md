# Research Log

This is a log of different approaches, techniques, and other thoughts relating
to the project as it evolves.

**Note:** _For the best viewing experience, we recommend installing the_
_[github+mermaid browser extension](https://github.com/BackMarket/github-mermaid-extension)_
_to render all of the diagrams_

## References

[wave_net]: https://arxiv.org/abs//1609.03499
[wave_net_autoencoders]: https://arxiv.org/abs/1901.08810

- [Wavenet: A Generative Model for Raw Audio][wave_net]
- [Unsupervised speech representation learning using WaveNet autoencoders][wave_net_autoencoders]


## Problem Statement

**03/11/2021**

The goal of the `dance2wave` is to create an instrument from human body
movement - to translate that motion into music.

```mermaid
graph LR;
    vid_stream([Video Stream])-->vid_enc[Video Encoder];
    vid_enc-->vid_repr([Video Representation]);
    vid_repr-->audio_dec[Audio Decoder];
    audio_dec-->audio_repr[Audio Representation];
    audio_repr-->audio_wav[Audio Waveform];
```

The prior research enumerated in the [resources](./README.md#resources) section
should provide ample inspiration for this project, most of which are concerned
with the inverse problem: translating music into some representation of human
movement. In principle, we can use similar architectures to map a sequence of
images (i.e. video) into a raw audio waveform.

## WaveNet: A Strong Baseline Candidate

**03/20/2021**

The objective of this project is to produce high quality music given a video
input of a human dancing. In order to achieve this, the *audio decoder* must
have enough capacity to generate qualitatively good samples.

A strong baseline model for this task might be the [WaveNet][wave_net] model,
which has been demonstrated to generate high-quality raw audio samples. In fact,
when trained on music, the WaveNet model was able to produce what the authors
report generating samples that "were often harmonic and aesthetically pleasing,
even when produced by unconditional models". What makes WaveNet a good candidate
for this project is that we can condition the output on both static and
sequential inputs.

Reproducing the WaveNet model in the context of conditioning on video input
will be the first phase of this project.

### Potential Issue: Deterministic Outputs

We can anticipate issues some issues with the baseline model.

A potential issue with the WaveNet model is that the outputs are deterministic
with respect to the conditioning inputs. This issue may be solved by using
variational methods as described in [Chorowski et al. 2019][wave_net_autoencoders],
where they use variational autoencoders (VAE) to capture latent representations
of speech waveforms in an autoencoder setting. While this paper used VAEs to solve
the problem of learning to better disentangle different high-level semantics of
the input, e.g. acoustic content and speaker information, we can use these
methods in this project to also produce non-deterministic outputs as a function
of video input. The hope is that that same exact movement might produce
qualitatively different sounds.

VAE methods are a potential direction to go once the baseline has been
established.
