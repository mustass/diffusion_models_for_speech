#!/bin/bash
echo "#### Installing the package in editable mode... ####"
<<<<<<< HEAD
#pip install -e .
echo "#### Checking and creating data folder if it does not exist ####"
mkdir -p ./data/tj/external
mkdir -p ./data/tj/raw
mkdir -p ./data/nst_danish/external
mkdir -p ./data/nst_danish/raw


FILE_EN_TAR=./data/tj/external/LJSpeech-1.1.tar.bz2
FILE_EN_metadata=./data/tj/raw/LJSpeech-1.1/metadata.csv
=======
python3 -m venv ../venv
source ../venv/bin/activate
pip install -e .
echo "#### Checking and creating data folder if it does not exist ####"
mkdir -p /work3/s210527/dl22/data/tj/external
mkdir -p /work3/s210527/dl22/data/tj/raw


FILE_TAR=/work3/s210527/dl22/data/tj/external/LJSpeech-1.1.tar.bz2
FILE_metadata=/work3/s210527/dl22/data/tj/raw/LJSpeech-1.1/metadata.csv
>>>>>>> 4da874f8c2d91d54d0d3ca8b8e8ee83c73f17a9c

if [ -f "$FILE_EN_TAR" ]; then

   echo "#### LJSpeech Dataset is already there ####"
else
<<<<<<< HEAD
   echo "### Downloading LJSpeech dataset ####"
   curl -o ./data/tj/external/LJSpeech-1.1.tar.bz2 'https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2'
=======
   echo "### Downloading data ####"
   curl -o /work3/s210527/dl22/data/tj/external/LJSpeech-1.1.tar.bz2 'https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2'
>>>>>>> 4da874f8c2d91d54d0d3ca8b8e8ee83c73f17a9c
fi

if [ -f "$FILE_EN_metadata" ]; then
   echo "#### LJSpeech Dataset is already extracted ####"
else
   echo "#### Extracting data ####"
   tar -jxf /work3/s210527/dl22/data/tj/external/LJSpeech-1.1.tar.bz2 --directory /work3/s210527/dl22/data/tj/raw
fi
<<<<<<< HEAD

declare -a FILE_DK_TAR=(da.16kHz.0611.tar.gz da.16kHz.0565-1.tar.gz)

FILE_DK_metadata=./data/tj/raw/nst_danish/metadata.csv
for ((idx=0; idx<${#FILE_DK_TAR[@]}; ++idx)); 
   do
      path=./data/tj/raw/nst_danish/external/${FILE_DK_TAR[$idx]}
      echo "$path"
      if [ -f "$path" ]; then
         echo "#### Danish Dataset is already there ####"
      else
         echo "### Downloading Danish dataset part $(($idx + 1))/3####"
         echo " $idx ${FILE_DK_TAR[$idx]}"
       #  echo "https://www.nb.no/sbfil/talegjenkjenning/16kHz/${FILE_DK_TAR[$idx]}"
        # curl -C - -o "./data/nst_danish/external/${FILE_DK_TAR[$idx]}" "https://www.nb.no/sbfil/talegjenkjenning/16kHz/${FILE_DK_TAR[$idx]}"
      fi
   done
=======
>>>>>>> 4da874f8c2d91d54d0d3ca8b8e8ee83c73f17a9c
echo "#### You're all set! ####"
