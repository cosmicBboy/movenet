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
	envsubst < Dockerfile > /tmp/movenet/Dockerfile && \
	grid run --dockerfile /tmp/movenet/Dockerfile \
		--instance_type t2.2xlarge \
		--cpus 7  \
		--scratch_size 512 \
		--memory 60G \
		--framework torch \
		--datastore_name kinetics-debug \
		--datastore_version 5 \
		--datastore_mount_dir /opt/datastore \
		movenet/trainer.py ${TRAIN_DEBUG_OPTS}
