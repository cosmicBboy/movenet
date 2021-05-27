# Experiments

This document keeps track of experiment runs during the course of this project.

- **Debugging Experiment**: Is the training loss going down?
  - **Run Name**: `merciful-frog-774`
  - **Date**: 05/26/2021
  - **Reproduce**: [![Grid](https://img.shields.io/badge/rid_AI-run-78FF96.svg?labelColor=black&logo=data:image/svg%2bxml;base64,PHN2ZyB3aWR0aD0iNDgiIGhlaWdodD0iNDgiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PHBhdGggZD0iTTEgMTR2MjBhMTQgMTQgMCAwMDE0IDE0aDlWMzYuOEgxMi42VjExaDIyLjV2N2gxMS4yVjE0QTE0IDE0IDAgMDAzMi40IDBIMTVBMTQgMTQgMCAwMDEgMTR6IiBmaWxsPSIjZmZmIi8+PHBhdGggZD0iTTM1LjIgNDhoMTEuMlYyNS41SDIzLjl2MTEuM2gxMS4zVjQ4eiIgZmlsbD0iI2ZmZiIvPjwvc3ZnPg==)](https://platform.grid.ai/#/runs?script=https://github.com/cosmicBboy/movenet/blob/fdce62ba4744df79d7380e51041a79227ddf5031/movenet/trainer.py&cloud=grid&instance=t2.2xlarge&accelerators=7&disk_size=200&framework=lightning&script_args=grid%20train%20--g_datastore_name%20kinetics-debug%20--g_datastore_mount_dir%20%2Fkinetics_debug%20--g_datastore_version%203%20--g_instance_type%20t2.2xlarge%20--g_cpus%207%20--g_memory%2032G%20movenet%2Ftrainer.py%20--dataset%20%2Fkinetics_debug%20--n_training_steps%20500%20--learning_rate%200.0003%20--input_channels%2064%20--residual_channels%2064%20--layer_size%203%20--stack_size%203)
