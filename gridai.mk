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

datasets/kinetics_breakdancing:
	@mkdir -p datasets/kinetics_breakdancing/train/breakdancing
	mkdir -p datasets/kinetics_breakdancing/valid/breakdancing
	cp -R datasets/kinetics/train/breakdancing datasets/kinetics_breakdancing/train/breakdancing
	cp -R datasets/kinetics/valid/breakdancing datasets/kinetics_breakdancing/valid/breakdancing

kinetics-breakdancing.tar.gz:
	tar -C datasets/kinetics -cvzf kinetics-breakdancing.tar.gz \
		train/breakdancing valid/breakdancing

kinetics.tar.gz:
	tar --exclude '**/*.part' \
		--exclude '**/*.ytdl' \
		--exclude '**/*_raw*' \
		--exclude '**/*.mkv' \
		-C datasets/kinetics -cvzf kinetics.tar.gz train valid

.PHONY: create-kinetics-debug
create-kinetics-debug: datasets/kinetics_debug
	grid datastore create --source datasets/kinetics_debug --name kinetics-debug-b

.PHONY: create-kinetics-breakdancing
create-kinetics-breakdancing: kinetics-breakdancing.tar.gz
	grid datastore create --source kinetics-breakdancing.tar.gz --name kinetics-breakdancing

.PHONY: create-kinetics
create-kinetics: datasets/kinetics
	grid datastore create --source datasets/kinetics --name kinetics --compression

# loop through entire kinetics dataset
.PHONY: test-kinetics-dataloader
test-kinetics-dataloader:
	envsubst < config/test-kinetics-dataloader.yml > /tmp/test-kinetics-dataloader.yml && \
	grid run \
		--config /tmp/test-kinetics-dataloader.yml \
		--ignore_warnings \
		movenet/dataset.py /opt/datastore

.PHONY: testing
testing:
	envsubst < config/test-kinetics-dataloader.yml > /tmp/test-kinetics-dataloader.yml && \
	grid run \
		--config /tmp/test-kinetics-dataloader.yml \
		--ignore_warnings \
		testing.py

clean:
	rm -rf \
		kinetics-debug.tar.gz \
		kinetics-breakdancing.tar.gz \
		kinetics.tar.gz \
		.kinetics

env/gridai:
	@echo "export GRID_USERNAME=<USERNAME>" > env/gridai
	@echo "export GRID_API_KEY=<API_KEY>" >> env/gridai
	@echo "export GRID_PROVIDER_CREDENTIALS=<PROVIDER_CREDENTIALS>" >> env/gridai
