N_DEBUG_EPOCHS?=1

TRAIN_DEBUG_OPTS?=--dataset /opt/datastore \
	--n_epochs ${N_DEBUG_EPOCHS} \
	--batch_size 2 \
	--learning_rate 0.0003 \
	--input_channels 64 \
	--residual_channels 64 \
	--layer_size 3 \
	--stack_size 3 \
	--checkpoint_every 1

DATASET_DEBUG_OPTS?=--datastore_name kinetics-debug \
	--datastore_version 5 \
	--datastore_mount_dir /opt/datastore \

DATASET_GPU_DEBUG_OPTS?=--datastore_name kinetics-breakdancing \
	--datastore_version 2 \
	--datastore_mount_dir /opt/datastore \

INFRA_DEBUG_OPTS?=--scratch_size 512 \
	--memory 60G \
	--framework torch \


PRETRAINED_MODEL_PATH?=
PRETRAINED_RUN_EXP_NAME?=


.PHONY: train-debug
train-debug:
	grid run --dockerfile Dockerfile \
		--instance_type t2.2xlarge \
		--cpus 7  \
		${INFRA_DEBUG_OPTS} \
		${DATASET_DEBUG_OPTS} \
		movenet/trainer.py ${TRAIN_DEBUG_OPTS} \
			--model_output_path models \
			--wandb_api_key=${WANDB_API_KEY}


.PHONY: train-debug-gpu
train-debug-gpu:
	grid run --dockerfile Dockerfile-gpu \
		--instance_type p3.8xlarge \
		--cpus 30  \
		--gpus 4 \
		${INFRA_INFRA_DEBUG_OPTS} \
		${DATASET_GPU_DEBUG_OPTS} \
		movenet/trainer.py ${TRAIN_DEBUG_OPTS} \
			--model_output_path models \
			--wandb_api_key=${WANDB_API_KEY}


N_EPOCHS?=1

TRAIN_OPTS?=--dataset /opt/datastore \
	--n_epochs ${N_EPOCHS} \
	--batch_size 2 \
	--learning_rate 0.0003 \
	--input_channels 64 \
	--residual_channels 64 \
	--layer_size 3 \
	--stack_size 3 \
	--checkpoint_every 1

INFRA_OPTS?=--scratch_size 512 \
	--memory 60G \
	--framework torch \

# DATASET_OPTS?=--datastore_name kinetics-all \
# 	--datastore_version 1 \
# 	--datastore_mount_dir /opt/datastore \

DATASET_OPTS?=--datastore_name kinetics-breakdancing \
	--datastore_version 2 \
	--datastore_mount_dir /opt/datastore \

.PHONY: train-gpu
train-gpu:
	grid run --dockerfile Dockerfile-gpu \
		--instance_type p3.8xlarge \
		--cpus 30  \
		--gpus 4 \
		${INFRA_OPTS} \
		${DATASET_OPTS} \
		movenet/trainer.py ${TRAIN_OPTS} \
			--model_output_path models \
			--pretrained_model_path "/artifacts/${PRETRAINED_RUN_EXP_NAME}/models/model.pth" \
			--pretrained_run_exp_name "${PRETRAINED_RUN_EXP_NAME}" \
			--grid_user_name="${GRID_USERNAME}" \
			--grid_api_key="${GRID_API_KEY}" \
			--wandb_api_key=${WANDB_API_KEY}
