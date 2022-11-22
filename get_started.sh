#!/bin/bash
echo "#### Installing the package in editable mode... ####"
python3 -m venv ../venv
source ../venv/bin/activate
pip install -e .
echo "#### Checking and creating data folder if it does not exist ####"
mkdir -p ./data/tj/external
mkdir -p ./data/tj/raw


FILE_TAR=./data/tj/external/LJSpeech-1.1.tar.bz2
FILE_metadata=./data/tj/raw/LJSpeech-1.1/metadata.csv

if [ -f "$FILE_TAR" ]; then
   echo "#### Data is already there ####"
else
   echo "### Downloading data ####"
   curl -o ./data/tj/external/LJSpeech-1.1.tar.bz2 'https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2'
fi

if [ -f "$FILE_metadata" ]; then
   echo "#### Data is already extracted ####"
else
   echo "#### Extracting data ####"
   tar -jxf ./data/tj/external/LJSpeech-1.1.tar.bz2 --directory ./data/tj/raw
fi
echo "#### You're all set! ####"