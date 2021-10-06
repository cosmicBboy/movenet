# base image you want to use
# make sure to use a CUDA image if running on GPUs
# FROM nvidia/cuda:XX.X-cudnnX-devel-ubuntuXX.XX
FROM python:3.9.6-slim

# these two lines are mandatory
WORKDIR /gridai/project
COPY . .

# any RUN commands you'd like to run
# use this to install dependencies
RUN apt-get update -y && \
    apt-get install curl wget libsndfile-dev -y

RUN pip install lightning-grid wandb

RUN pip install av \
    dataclasses==0.6 \
    dataclasses-json==0.5.2 \
    joblib \
    librosa==0.8.1 \
    numpy==1.20.3 \
    torch==1.9.0 \
    torchvision==0.10.0 \
    torchaudio==0.9.0 \
    torchtyping==0.1.2 \
    typeguard \
    typing-extensions==3.7.4.3 \
    pytorchvideo==0.1.2 \
    dask \
    ipdb==0.13.7 \
    opencv-python \
    opencv-python-headless==4.5.3.56 \
    tensorflow \
    tensorboard \
    tqdm

ARG WANDB_API_KEY

ENV WANDB_ENTITY=nielsbantilan
ENV WANDB_API_KEY=$WANDB_API_KEY