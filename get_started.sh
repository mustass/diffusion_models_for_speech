#!/bin/bash
echo "#### Installing the package in editable mode... ####"
python3 -m venv $PATH_TO_VENV
source $PATH_TO_VENV/bin/activate
pip install -e .
echo "#### Checking and creating data folder if it does not exist ####"
mkdir -p $DATA_PATH_PREFIX/data/tj/external
mkdir -p $DATA_PATH_PREFIX/data/tj/raw


FILE_EN_TAR=$DATA_PATH_PREFIX/data/tj/external/LJSpeech-1.1.tar.bz2
FILE_EN_metadata=$DATA_PATH_PREFIX/data/tj/raw/LJSpeech-1.1/metadata.csv


if [ -f "$FILE_EN_TAR" ]; then
   echo "#### LJSpeech Dataset is already there ####"
else
   echo "### Downloading LJSpeech dataset ####"
   curl -o $DATA_PATH_PREFIX/data/tj/external/LJSpeech-1.1.tar.bz2 'https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2'
fi

if [ -f "$FILE_EN_metadata" ]; then
   echo "#### LJSpeech Dataset is already extracted ####"
else
   echo "#### Extracting data ####"
   tar -jxf $DATA_PATH_PREFIX/data/tj/external/LJSpeech-1.1.tar.bz2 --directory $DATA_PATH_PREFIX/data/tj/raw
fi

echo "#### Downloading the Danish Dataset"
python3 download_danish_dataset.py