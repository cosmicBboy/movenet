# !/bin/sh
# Execute a grid training substituting env vars in the gridai-config.yml file.

set -e

envsubst < gridai-config.yml >> /tmp/gridai-config.yml && \
grid train --g_config /tmp/gridai-config.yml movenet/trainer.py $@
