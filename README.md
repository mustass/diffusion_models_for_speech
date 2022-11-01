# Diffusion models for speech synthesis

## Background

DiffWave approach [https://arxiv.org/abs/2009.09761]. 

## Data 
[https://keithito.com/LJ-Speech-Dataset/]

## Repo structure
This repo takes advantage of two frameworks: (1) Hydra for configs management and (2) Pytorch Lightning for improving our lives when colaborating and running experiments on different hardware. 

The particular approach of this repo is heavily inspired by [https://youtu.be/w10WrRA-6uI].

## Getting started 

1. Create a virtual environment the way you are used to (conda, venv, pyenv, whatever). 
2. Run the bash script from the root folder:

```{bash}
./get_started.sh
```

This will:
1. Install the package in editable mode with all requirements
2. Download the dataset and extract it

You can further install the requirements for developing the package:
```{bash}
pip install -r reqiorements-dev.txt
```

## Contribution guide

This repo has protection on the ``main`` branch. Therefore any contribution has to go through a Pull Request. 

Make sure to run ``make`` in the root directory and push changes before creating a Pull Request. 
