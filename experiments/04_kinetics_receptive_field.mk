# Experiments involving increased receptive field
N_DEBUG_EPOCHS?=3

TRAIN_DEBUG_OPTS?=--dataset /opt/datastore \
	--n_epochs ${N_DEBUG_EPOCHS} \
	--batch_size 1 \
	--learning_rate 0.00003 \
	--input_channels 128 \
	--residual_channels 16 \
	--layer_size 14 \
	--stack_size 1 \
	--checkpoint_every 1 \
	--accumulation_steps 3 \
	--generate_n_samples 100000

DATASET_DEBUG_OPTS?=--datastore_name kinetics-debug \
	--datastore_version 5 \
	--datastore_mount_dir /opt/datastore

INFRA_DEBUG_OPTS?=--scratch_size 512 \
	--memory 100


PRETRAINED_RUN_EXP_NAME?=


.PHONY: train-debug
train-debug:
	grid run --dockerfile Dockerfile \
		--instance_type t2.2xlarge \
		--cpus 7  \
		--ignore_warnings \
		${INFRA_DEBUG_OPTS} \
		${DATASET_DEBUG_OPTS} \
		movenet/trainer.py ${TRAIN_DEBUG_OPTS} \
			--model_output_path models \
			--wandb_api_key=${WANDB_API_KEY}


.PHONY: train-debug-gpu
train-debug-gpu:
	grid run --dockerfile Dockerfile-gpu \
		--instance_type p3.2xlarge \
		--cpus 7  \
		--gpus 1 \
		--ignore_warnings \
		${INFRA_DEBUG_OPTS} \
		${DATASET_DEBUG_OPTS} \
		movenet/trainer.py ${TRAIN_DEBUG_OPTS} \
			--model_output_path models \
			--wandb_api_key=${WANDB_API_KEY}


N_EPOCHS?=3

TRAIN_OPTS?=--dataset /opt/datastore \
	--n_epochs ${N_EPOCHS} \
	--batch_size 2 \
	--learning_rate 0.00003 \
	--pin_memory 1 \
	--num_workers 4 \
	--input_channels 128 \
	--residual_channels 16 \
	--layer_size 14 \
	--stack_size 1 \
	--checkpoint_every 1 \
	--accumulation_steps 10 \
	--generate_n_samples 100000

INFRA_OPTS?=--scratch_size 512 \
	--memory 100

DATASET_OPTS?=--datastore_name kinetics-breakdancing \
	--datastore_version 2 \
	--datastore_mount_dir /opt/datastore \

.PHONY: train-gpu
train-gpu:
	grid run --dockerfile Dockerfile-gpu \
		--instance_type p3.8xlarge \
		--cpus 30  \
		--gpus 4 \
		--ignore_warnings \
		${INFRA_OPTS} \
		${DATASET_OPTS} \
		movenet/trainer.py ${TRAIN_OPTS} \
			--model_output_path models \
			--pretrained_model_path "/artifacts/${PRETRAINED_RUN_EXP_NAME}/models/model.pth" \
			--pretrained_run_exp_name "${PRETRAINED_RUN_EXP_NAME}" \
			--grid_user_name="${GRID_USERNAME}" \
			--grid_api_key="${GRID_API_KEY}" \
			--wandb_api_key=${WANDB_API_KEY}


.PHONY: train-gpu-spot
train-gpu-spot:
	grid run --dockerfile Dockerfile-gpu \
		--instance_type p3.8xlarge \
		--cpus 30  \
		--gpus 4 \
		--ignore_warnings \
		--use_spot \
		--auto_resume \
		${INFRA_OPTS} \
		${DATASET_OPTS} \
		movenet/trainer.py ${TRAIN_OPTS} \
			--model_output_path models \
			--pretrained_model_path "/artifacts/${PRETRAINED_RUN_EXP_NAME}/models/model.pth" \
			--pretrained_run_exp_name "${PRETRAINED_RUN_EXP_NAME}" \
			--grid_user_name="${GRID_USERNAME}" \
			--grid_api_key="${GRID_API_KEY}" \
			--wandb_api_key=${WANDB_API_KEY}
