# !/bin/sh
# Execute a grid training substituting env vars in the gridai-config.yml file.

set -e

source env/gridai
envsubst < gridai-config.yml > /tmp/gridai-config.yml
grid run \
    --config /tmp/gridai-config.yml \
    --framework torch \
    --datastore_name $GRID_DATASTORE_NAME \
    --datastore_version $GRID_DATASTORE_VERSION \
    --datastore_mount_dir $GRID_DATASTORE_MOUNT_DIR \
    movenet/trainer.py $@
