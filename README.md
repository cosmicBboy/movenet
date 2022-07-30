# movenet

Generate raw audio from dance videos


## Background

`movenet` is a research project for generating music from dance. The idea
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
conda create -n movenet python=3.9
conda activate movenet
conda env update -n movenet -f env.yml
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

### Creating Kinetics Dataset


Clone the downloader

```bash
git clone https://github.com/hai-labs/kinetics-downloader
```

If you want to reconstitute a fresh dataset, download it with:
```bash
cd kinetics-downloader
python download.py --categories "dancing" --num-workers <NUM_WORKERS> -v
cd ..
cp -R kinetics-downloader/dataset datasets/kinetics
```

### Downloading the Data

You can also download the datasets from [google drive](https://drive.google.com/drive/folders/1JDt4QapusLD9AH7ZxeW88gaFmg1Hvix8?usp=sharing). For example you can dump the
`kinetics_debug` directory into `dataset/kinetics_debug`.

### Running the Trainer

```bash
python movenet/pytorch_lightning_trainer.py --dataset datasets/kinetics_debug --n_epochs 1
```

### Running an Experiment on `gridai`

The `experiments` directory contains makefiles for running jobs over various
exprimental setups.

```bash
source env/gridai
make -f experiments/<makefile> <target>
```
