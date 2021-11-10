TRAIN_DEBUG_OPTS?=--dataset /opt/datastore \
	--n_epochs 3 \
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
N_EPOCHS?=1


.PHONY: train-debug
train-debug:
	grid run --dockerfile Dockerfile \
		--instance_type t2.2xlarge \
		--cpus 7  \
		${INFRA_DEBUG_OPTS} \
		${DATASET_DEBUG_OPTS} \
		movenet/trainer.py ${TRAIN_DEBUG_OPTS} \
			--wandb_api_key=${WANDB_API_KEY}


.PHONY: train-debug-gpu
train-debug-gpu:
	grid run --dockerfile Dockerfile-gpu \
		--instance_type p3.2xlarge \
		--cpus 3  \
		--gpus 1 \
		${INFRA_DEBUG_OPTS} \
		${DATASET_GPU_DEBUG_OPTS} \
		movenet/trainer.py ${TRAIN_DEBUG_OPTS} \
			--wandb_api_key=${WANDB_API_KEY}
