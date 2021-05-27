kinetics-debug.tar.gz:
	tar -C datasets/kinetics -cvzf kinetics-debug.tar.gz \
		train/breakdancing/zkyRFux7BWc.mp4 \
		train/breakdancing/eB4wwvnXwrI.mp4 \
		train/breakdancing/MEguK5_ding.mp4 \
		valid/breakdancing/_OGG4vXzHSA.mp4 \
		valid/breakdancing/3ob3NvTp-YA.mp4 \
		valid/breakdancing/K-81lIy6PoI.mp4

datasets/kinetics_debug:
	@mkdir -p datasets/kinetics_debug/train/breakdancing
	mkdir -p datasets/kinetics_debug/valid/breakdancing
	cp datasets/kinetics/train/breakdancing/zkyRFux7BWc.mp4 datasets/kinetics_debug/train/breakdancing
	cp datasets/kinetics/train/breakdancing/eB4wwvnXwrI.mp4 datasets/kinetics_debug/train/breakdancing
	cp datasets/kinetics/train/breakdancing/MEguK5_ding.mp4 datasets/kinetics_debug/train/breakdancing
	cp datasets/kinetics/valid/breakdancing/_OGG4vXzHSA.mp4 datasets/kinetics_debug/valid/breakdancing
	cp datasets/kinetics/valid/breakdancing/3ob3NvTp-YA.mp4 datasets/kinetics_debug/valid/breakdancing
	cp datasets/kinetics/valid/breakdancing/K-81lIy6PoI.mp4 datasets/kinetics_debug/valid/breakdancing

kinetics-breakdancing.tar.gz:
	tar -C datasets/kinetics -cvzf kinetics-breakdancing.tar.gz \
		train/breakdancing valid/breakdancing

kinetics.tar.gz:
	tar -C datasets/kinetics -cvzf kinetics.tar.gz train valid

.kinetics:
	mkdir -p .kinetics/train .kinetics/valid
	cp -R datasets/kinetics/train .kinetics/
	cp -R datasets/kinetics/valid .kinetics/

.PHONY: create-kinetics-debug
create-kinetics-debug: datasets/kinetics_debug
	grid datastores create --source datasets/kinetics_debug --name kinetics-debug

.PHONY: create-kinetics-breakdancing
create-kinetics-breakdancing: kinetics-breakdancing.tar.gz
	grid datastores create --source kinetics-breakdancing.tar.gz --name kinetics-breakdancing

.PHONY: create-kinetics
create-kinetics: .kinetics
	grid datastores create --source .kinetics --name kinetics-all

clean:
	rm -rf \
		kinetics-debug.tar.gz \
		kinetics-breakdancing.tar.gz \
		kinetics.tar.gz \
		.kinetics

GRID_OPTS?=--g_datastore_name kinetics-debug \
	--g_datastore_mount_dir /kinetics_debug \
	--g_datastore_version 3 \
	--g_instance_type t2.2xlarge \
	--g_cpus 7 \
	--g_memory 32G

TRAIN_DEBUG_OPTS?=--dataset /kinetics_debug \
	--n_training_steps 500 \
	--learning_rate 0.0003 \
	--input_channels 64 \
	--residual_channels 64 \
	--layer_size 3 \
	--stack_size 3

train-debug:
	grid train ${GRID_OPTS} movenet/trainer.py ${TRAIN_DEBUG_OPTS}

train-debug-continue:
	grid train ${GRID_OPTS} --g_config gridai-config.yml gridai_test.py

session-debug:
	grid session create \
		--g_instance_type t2.2xlarge \
		--g_datastore_name kinetics-debug \
		--g_datastore_version 3 \
		--g_datastore_mount_dir /kinetics_debug
