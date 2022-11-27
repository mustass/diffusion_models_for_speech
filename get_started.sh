#!/bin/bash
echo "#### Installing the package in editable mode... ####"
python3 -m venv $PATH_TO_VENV
source $PATH_TO_VENV/bin/activate
pip install -e .
python3 scripts/download_danish_dataset.py