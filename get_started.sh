#!/bin/bash
echo "#### Installing the package in editable mode... ####"
python3 -m venv ../venv
source ../venv/bin/activate
pip install -e .
echo "#### Checking and creating data folder if it does not exist ####"
mkdir -p /work3/s210527/dl22/data/tj/external
mkdir -p /work3/s210527/dl22/data/tj/raw


FILE_TAR=/work3/s210527/dl22/data/tj/external/LJSpeech-1.1.tar.bz2
FILE_metadata=/work3/s210527/dl22/data/tj/raw/LJSpeech-1.1/metadata.csv

if [ -f "$FILE_TAR" ]; then
   echo "#### Data is already there ####"
else
   echo "### Downloading data ####"
   curl -o /work3/s210527/dl22/data/tj/external/LJSpeech-1.1.tar.bz2 'https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2'
fi

if [ -f "$FILE_metadata" ]; then
   echo "#### Data is already extracted ####"
else
   echo "#### Extracting data ####"
   tar -jxf /work3/s210527/dl22/data/tj/external/LJSpeech-1.1.tar.bz2 --directory /work3/s210527/dl22/data/tj/raw
fi
echo "#### You're all set! ####"
