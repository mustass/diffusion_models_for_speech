#!/bin/bash
echo "#### Installing the package in editable mode... ####"
python3 -m venv ../venv
source ../venv/bin/activate
pip install -e .
python download_data.py