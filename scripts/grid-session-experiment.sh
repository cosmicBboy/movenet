# !/bin/bash
#
# Run an experiment

python movenet/trainer.py \
--dataset /datastores/kinetics-breakdancing \
--n_epochs 4 \
--batch_size 3 \
--learning_rate 0.00003 \
--pin_memory 1 \
--num_workers 6 \
--input_channels 128 \
--residual_channels 32 \
--layer_size 2 \
--stack_size 2 \
--checkpoint_every 1 \
--accumulation_steps 3 \
--dist_port 8889 \
--wandb_api_key $WANDB_API_KEY
