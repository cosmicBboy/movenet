GRID_OPTS?=GRID_DATASTORE_NAME=kinetics-debug \
	GRID_DATASTORE_MOUNT_DIR=/kinetics_debug \
	GRID_DATASTORE_VERSION=3

TRAIN_DEBUG_OPTS?=--dataset /kinetics_debug \
	--n_training_steps 500 \
	--learning_rate 0.0003 \
	--input_channels 64 \
	--residual_channels 64 \
	--layer_size 3 \
	--stack_size 3 \
	--checkpoint_every 25


train-debug:
	${GRID_OPTS} scripts/run-grid-experiment.sh ${TRAIN_DEBUG_OPTS}

train-debug-continue:
	${GRID_OPTS} GRID_ARTIFACTS_RUNS_OR_EXPERIMENTS=nostalgic-impala-272 \
		scripts/run-grid-experiment.sh ${TRAIN_DEBUG_OPTS} \
		--pretrained_model_path /artifacts/nostalgic-impala-272/models/20210612210715/model.pth \
		--pretrained_run_exp_name nostalgic-impala-272
