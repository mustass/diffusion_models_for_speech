#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J GenPreTJ
#BSUB -n 1
#BSUB -W 24:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err
echo
module load python3/3.9.11
source /zhome/ed/0/170279/Github/deep-learning/venv/bin/activate

wandb login 6fee9cf3b83c356e644334bb6ec3ebb0169f2de3
echo "Running script..."
python3 ./scripts/inference.py datamodule=tjspeech_cond inference=pretrained inference.audio_path=/dtu/blackhole/19/s176453/diffusion_for_speech/synthesized_audio/experiment1/pretrained/tj datamodule.params.batch_size=1
