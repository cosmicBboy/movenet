GRID_OPTS?=GRID_DATASTORE_NAME=kinetics-debug \
	GRID_DATASTORE_VERSION=5 \
	GRID_DATASTORE_MOUNT_DIR=/opt/datastore

TRAIN_DEBUG_OPTS?=--dataset /opt/datastore \
	--n_training_steps 500 \
	--learning_rate 0.0003 \
	--input_channels 64 \
	--residual_channels 64 \
	--layer_size 3 \
	--stack_size 3 \
	--checkpoint_every 25

.PHONY: train-debug
train-debug:
	${GRID_OPTS} scripts/run-grid-experiment.sh ${TRAIN_DEBUG_OPTS}

.PHONY: train-debug-continue
train-debug-continue:
	${GRID_OPTS} GRID_ARTIFACTS_RUNS_OR_EXPERIMENTS=nostalgic-impala-272-exp0 \
		scripts/run-grid-experiment.sh ${TRAIN_DEBUG_OPTS} \
		--pretrained_model_path /artifacts/nostalgic-impala-272-exp0/models/20210612210715/model.pth \
		--pretrained_run_exp_name nostalgic-impala-272-exp0

.PHONY: train-debug-continue-2
train-debug-continue-2:
	${GRID_OPTS} GRID_ARTIFACTS_RUNS_OR_EXPERIMENTS=upbeat-iguana-246-exp0 \
		scripts/run-grid-experiment.sh ${TRAIN_DEBUG_OPTS} \
		--model_output_path models \
		--pretrained_model_path /artifacts/upbeat-iguana-246-exp0/models/20210614141239/model.pth \
		--pretrained_run_exp_name upbeat-iguana-246-exp0

.PHONY: train-debug-continue-3
train-debug-continue-3:
	${GRID_OPTS} GRID_ARTIFACTS_RUNS_OR_EXPERIMENTS=cornflower-rooster-462-exp0 \
		scripts/run-grid-experiment.sh ${TRAIN_DEBUG_OPTS} \
		--model_output_path models \
		--pretrained_model_path /artifacts/cornflower-rooster-462-exp0/models/model.pth \
		--pretrained_run_exp_name cornflower-rooster-462-exp0
