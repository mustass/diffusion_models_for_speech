#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J Generate
#BSUB -n 1
#BSUB -W 24:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err
echo
module load python3/3.9.11
source path_to_your_venv

wandb login your_wandb_key
echo "Running script..."
python3 ./scripts/inference.py inference.run_name=pretrained_model inference.device=gpu datamodule=nst_da_cond
