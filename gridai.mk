kinetics-debug.tar.gz:
	tar -C datasets/kinetics -cvzf kinetics-debug.tar.gz \
		train/breakdancing/zkyRFux7BWc.mp4 \
		train/breakdancing/eB4wwvnXwrI.mp4 \
		train/breakdancing/MEguK5_ding.mp4 \
		valid/breakdancing/_OGG4vXzHSA.mp4 \
		valid/breakdancing/3ob3NvTp-YA.mp4 \
		valid/breakdancing/K-81lIy6PoI.mp4

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
create-kinetics-debug: kinetics-debug.tar.gz
	grid datastores create --source kinetics-debug.tar.gz --name kinetics-debug

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

train-debug:
	grid train \
		--g_datastore_name kinetics-debug \
		--g_datastore_mount_dir /kinetics_debug \
		--g_datastore_version 2 \
		--g_instance_type t2.medium \
		movenet/trainer.py \
		--dataset /kinetics_debug \
		--n_training_steps 10
