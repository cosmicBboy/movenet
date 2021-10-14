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
	grid run --config /tmp/gridai-config.yml --ignore_warnings \
		movenet/trainer.py ${TRAIN_DEBUG_OPTS} \
			--grid_user_name "${GRID_USERNAME}" \
			--grid_api_key "${GRID_API_KEY}"


.PHONY: train-breakdancing
train-breakdancing:
	GRID_DATASTORE_NAME=kinetics-breakdancing \
	GRID_DATASTORE_VERSION=2 \
	GRID_DATASTORE_MOUNT_DIR=/opt/datastore \
	envsubst < config/gridai-config-gpu.yml > /tmp/gridai-config-gpu.yml && \
	grid run --config /tmp/gridai-config-gpu.yml --ignore_warnings movenet/trainer.py ${TRAIN_DEBUG_OPTS}


.PHONY: train
train:
	GRID_DATASTORE_NAME=kinetics-all \
	GRID_DATASTORE_VERSION=1 \
	GRID_DATASTORE_MOUNT_DIR=/opt/datastore \
	GRID_ARTIFACTS_RUNS_OR_EXPERIMENTS=${PRETRAINED_RUN_EXP_NAME} \
	envsubst < config/gridai-config-gpu.yml > /tmp/gridai-config-gpu.yml && \
	grid run --config /tmp/gridai-config-gpu.yml --ignore_warnings \
		movenet/trainer.py \
			--dataset /opt/datastore \
			--n_epochs ${N_EPOCHS} \
			--batch_size 2 \
			--learning_rate 0.0003 \
			--input_channels 64 \
			--residual_channels 64 \
			--layer_size 3 \
			--stack_size 3 \
			--checkpoint_every 1 \
			--model_output_path models \
			--pretrained_model_path "${PRETRAINED_MODEL_PATH}" \
			--pretrained_run_exp_name "${PRETRAINED_RUN_EXP_NAME}" \
			--grid_user_name "${GRID_USERNAME}" \
			--grid_api_key "${GRID_API_KEY}" \
			--wandb_api_key ${WANDB_API_KEY}


.PHONY: train-spot
train-spot:
	GRID_DATASTORE_NAME=kinetics-all \
	GRID_DATASTORE_VERSION=1 \
	GRID_DATASTORE_MOUNT_DIR=/opt/datastore \
	GRID_ARTIFACTS_RUNS_OR_EXPERIMENTS=${PRETRAINED_RUN_EXP_NAME} \
	envsubst < config/gridai-config-gpu.yml > /tmp/gridai-config-gpu.yml && \
	grid run --config /tmp/gridai-config-gpu.yml --ignore_warnings \
		movenet/trainer.py \
			--dataset /opt/datastore \
			--n_epochs ${N_EPOCHS} \
			--batch_size 2 \
			--learning_rate 0.0003 \
			--input_channels 64 \
			--residual_channels 64 \
			--layer_size 3 \
			--stack_size 3 \
			--checkpoint_every 1 \
			--model_output_path models \
			--pretrained_model_path "${PRETRAINED_MODEL_PATH}" \
			--pretrained_run_exp_name "${PRETRAINED_RUN_EXP_NAME}" \
			--grid_user_name "${GRID_USERNAME}" \
			--grid_key "${GRID_API_KEY}"
