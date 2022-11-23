#!/bin/sh
#BSUB -q gpuv100
#BSUB -R "select[gpu32gb]"
#BSUB -gpu "num=1"
#BSUB -J DiffWave
#BSUB -n 1
#BSUB -W 10:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err
echo
module load python3/3.9.11
#module load cudnn/v8.3.2.44-prod-cuda-11.5
source ../venv/bin/activate

wandb login a9a49618b7c9a34f36b1f55dfc6e9175e7962060
echo "Running script..."
python3 ./scripts/train.py datamodule.params.audio_len=75000 datamodule.params.batch_size=8 trainer.gpus=1 trainer.accelerator=gpu 
