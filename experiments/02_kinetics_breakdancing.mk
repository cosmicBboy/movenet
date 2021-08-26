GRID_DATASTORE_NAME=kinetics-breakdancing
GRID_DATASTORE_VERSION=2
GRID_DATASTORE_MOUNT_DIR=/opt/datastore

TRAIN_DEBUG_OPTS?=--dataset /opt/datastore \
	--n_epochs 3 \
	--learning_rate 0.0003 \
	--input_channels 64 \
	--residual_channels 64 \
	--layer_size 3 \
	--stack_size 3 \
	--checkpoint_every 5

.PHONY: train
train:
	envsubst < config/gridai-config-gpu.yml > /tmp/gridai-config-gpu.yml && \
	grid run --config /tmp/gridai-config-gpu.yml movenet/trainer.py ${TRAIN_DEBUG_OPTS}
