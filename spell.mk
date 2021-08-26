# makefile for running experiments on spell infrastructure

.PHONY: train-debug
train-debug:
	spell run \
		--project dance2music \
		--machine-type cpu-big \
		--mount uploads/kinetics-debug:/mnt/dataset \
		--pip-req requirements.txt \
		--apt libsndfile-dev \
		"python movenet/trainer.py \
			--dataset /mnt/dataset \
			--n_epochs 3 \
			--learning_rate 0.0003 \
			--input_channels 64 \
			--residual_channels 64 \
			--layer_size 3 \
			--stack_size 3 \
			--checkpoint_every 1"
