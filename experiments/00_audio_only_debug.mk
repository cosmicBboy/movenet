GRID_OPTS?=GRID_DATASTORE_NAME=kinetics-debug \
	GRID_DATASTORE_MOUNT_DIR=/kinetics_debug \
	GRID_DATASTORE_VERSION=4

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
	${GRID_OPTS} GRID_ARTIFACTS_RUNS_OR_EXPERIMENTS=merciful-frog-774-exp0 \
		scripts/run-grid-experiment.sh ${TRAIN_DEBUG_OPTS} \
		--pretrained_model_path /artifacts/merciful-frog-774-exp0/models/20210527105745/model.pth \
		--pretrained_run_exp_name merciful-frog-774-exp0

train-debug-continue-2:
	${GRID_OPTS} GRID_ARTIFACTS_RUNS_OR_EXPERIMENTS=cherry-perch-119-exp0 \
		scripts/run-grid-experiment.sh ${TRAIN_DEBUG_OPTS} \
		--pretrained_model_path /artifacts/cherry-perch-119-exp0/models/20210527151106/model.pth \
		--pretrained_run_exp_name cherry-perch-119-exp0

session-debug:
	grid session create \
		--g_instance_type t2.2xlarge \
		--g_datastore_name kinetics-debug \
		--g_datastore_version 3 \
		--g_datastore_mount_dir /kinetics_debug
