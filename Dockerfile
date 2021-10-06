# base image you want to use
# make sure to use a CUDA image if running on GPUs
# FROM nvidia/cuda:XX.X-cudnnX-devel-ubuntuXX.XX
FROM python:3.9.6-slim

# these two lines are mandatory
WORKDIR /gridai/project
COPY ./movenet ./movenet
COPY ./setup.py .
COPY ./requirements.txt .
COPY ./dev-requirements.txt .

# any RUN commands you'd like to run
# use this to install dependencies
RUN apt-get update -y && \
    apt-get install curl wget libsndfile-dev -y

RUN pip install lightning-grid wandb

RUN pip install -e .

ENV WANDB_ENTITY=nielsbantilan
