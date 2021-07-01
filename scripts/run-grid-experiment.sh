# !/bin/sh
# Execute a grid training substituting env vars in the gridai-config.yml file.

set -e

envsubst < gridai-config.yml >> /tmp/gridai-config.yml && \
grid run --config /tmp/gridai-config.yml movenet/trainer.py $@
