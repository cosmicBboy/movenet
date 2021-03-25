# dance2wave

Generate raw audio from dance videos


## Background

`dance2wave` is a research project for generating music from dance. The idea
is to turn the human body into an instrument, converting sequences of images
into raw audio waveforms.

## Dataset

This project uses the [Kinetics](https://deepmind.com/research/open-source/kinetics)
dataset to train the dance-to-audio generation model because it conveniently
comes with a [downloader](https://github.com/Showmax/kinetics-downloader) that
supports downloading video and audio.


## Environment

This repo uses [miniconda](https://docs.conda.io/en/latest/miniconda.html)
as a virtual environment.

```
conda create -n dance2wave python=3.8
conda activate dance2wave
```

Install [youtube-dl](https://github.com/ytdl-org/youtube-dl) depending on your
system.

## Resources

- [Dance Video Datasets for Artificial Intelligence](https://markgituma.medium.com/dance-video-datasets-for-artificial-intelligence-6c0a77f2b929):
  A medium post pointing to several relevant datasets.
- [WaveNet](https://arxiv.org/abs//1609.03499): DeepMind model that generates raw audio waveforms.
- [Unsupervised speech representation learning using WaveNet autoencoders](https://arxiv.org/abs/1901.08810)
- [Dance Revolution: Long-term Dance Generation With Music Via Curriculum Learning](https://arxiv.org/pdf/2006.06119v6.pdf)
- [Dancing to Music](https://arxiv.org/pdf/1911.02001v1.pdf)
  - [Repo](https://github.com/NVlabs/Dancing2Music)
  - [Repo Fork](https://github.com/cosmicBboy/Dancing2Music)
  - [Author Website](https://vllab.ucmerced.edu/hylee/)
  - [Script for data and models](https://vllab.ucmerced.edu/hylee/Dancing2Music/script.txt)
- [Music-oriented Dance Video Synthesis with Pose Perceptual Loss](https://arxiv.org/pdf/1912.06606v1.pdf)
- [Feel the Music: Automatically Generating A Dance For An Input Song](https://arxiv.org/pdf/2006.11905v2.pdf)
- [Everybody Dance Now](https://arxiv.org/pdf/1808.07371v2.pdf)
- [Learning to Dance: A graph convolutional adversarial network to generate realistic dance motions from audio](https://arxiv.org/pdf/2011.12999v2.pdf)
- [Weakly-Supervised Deep Recurrent Neural Networks for Basic Dance Step Generation](https://arxiv.org/pdf/1807.01126v3.pdf)
- [MagnaTagATune Dataset](https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset)
- [OpenAI Jukebox](https://openai.com/blog/jukebox/)
   - [paper](https://arxiv.org/abs/2005.00341)
   - [code](https://github.com/openai/jukebox/)
- [OpenAI Musenet](https://openai.com/blog/musenet/)


## Download the Datasets

### Kinetics Dataset


Clone the downloader
```
git clone https://github.com/Showmax/kinetics-downloader
```

Download the dataset
```
cd kinetics-downloader
python download.py --categories "dancing" --num-workers <NUM_WORKERS> -v
cd .. && cp -R kinetics-downloader/dataset datasets/kinetics
```

### Dancing to Music Dataset

Download dataset

```
./scripts/dancing-to-music-dataset.sh
```

Download models

```
./scripts/dancing-to-music-models.sh
```
