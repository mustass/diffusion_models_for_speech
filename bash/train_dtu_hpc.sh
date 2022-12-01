#!/bin/sh
#BSUB -q gpuv100
#BSUB -R "select[gpu32gb]"
#BSUB -gpu "num=1"
#BSUB -J DiffWave
#BSUB -n 1
#BSUB -W 24:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err
echo
module load python3/3.9.11
source ../venv/bin/activate

wandb login a9a49618b7c9a34f36b1f55dfc6e9175e7962060
echo "Running script..."
python3 ./scripts/train.py datamodule=nst_da_cond trainer.gpus=1 trainer.accelerator=gpu 
