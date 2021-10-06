TRAIN_DEBUG_OPTS?=--dataset /opt/datastore \
	--n_epochs 10 \
	--batch_size 2 \
	--learning_rate 0.0003 \
	--input_channels 64 \
	--residual_channels 64 \
	--layer_size 3 \
	--stack_size 3 \
	--checkpoint_every 1


PRETRAINED_MODEL_PATH?=
PRETRAINED_RUN_EXP_NAME?=
N_EPOCHS?=1


.PHONY: train-debug
train-debug:
	GRID_DATASTORE_NAME=kinetics-debug \
	GRID_DATASTORE_VERSION=5 \
	GRID_DATASTORE_MOUNT_DIR=/opt/datastore \
	GRID_ARTIFACTS_RUNS_OR_EXPERIMENTS=${PRETRAINED_RUN_EXP_NAME} \
	envsubst < config/gridai-config.yml > /tmp/gridai-config.yml && \
	grid run -d Dockerfile \
		--instance_type t2.2xlarge \
		--cpus 7  \
		--scratch_size 512 \
		--memory 60G \
		--framework torch \
		--datastore_name ${GRID_DATASTORE_NAME} \
		--datastore_version ${GRID_DATASTORE_VERSION} \
		--datastore_mount_dir ${GRID_DATASTORE_MOUNT_DIR} \
		movenet/trainer.py ${TRAIN_DEBUG_OPTS}
