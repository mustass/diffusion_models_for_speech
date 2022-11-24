import os
import tarfile
from pathlib import Path

import wget

print("#### Checking and creating data folder if it does not exist ####")
data_path_prefix = os.getenv("DATA_PATH_PREFIX")

en_path = Path(data_path_prefix) / "data/tj/external"
en_raw_path = Path(data_path_prefix) / "data/tj/raw"
dk_path = Path(data_path_prefix) / "data/nst_danish/external"
dk_raw_path = Path(data_path_prefix) / "data/nst_danish/raw"

en_path.mkdir(parents=True, exist_ok=True)
en_raw_path.mkdir(parents=True, exist_ok=True)
dk_path.mkdir(parents=True, exist_ok=True)
dk_raw_path.mkdir(parents=True, exist_ok=True)

en_tar = en_path / "LJSpeech-1.1.tar.bz2"
en_meta = en_raw_path / "LJSpeech-1.1/metadata.csv"

if en_tar.is_file():
    print("#### LJSpeech Dataset is already there ####")
else:
    print("#### Downloading LJSpeech dataset ####")
    en_url = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
    wget.download(en_url, out=en_path)

if en_meta.is_file():
    print("#### LJSpeech Dataset is already extracted ####")
else:
    print("#### Extracting LJSpeech data ####")
    en_file = tarfile.open(en_tar)
    en_file.extractall(en_raw_path)
    en_file.close()

dk_tars = ["da.16kHz.0611.tar.gz", "da.16kHz.0565-1.tar.gz", "da.16kHz.0565-2.tar.gz"]
dk_extracted_data = ["da_0611_test", "Stasjon01", "Stasjon04"]
for i, tar in enumerate(dk_tars):
    tar_path = dk_path / tar
    if tar_path.is_file():
        print("#### Danish Dataset " + str(i + 1) + "/3 is already there ####")
    else:
        print("#### Downloading Danish Dataset part " + str(i + 1) + "/3 ####")
        dk_url = "https://www.nb.no/sbfil/talegjenkjenning/16kHz/" + tar
        wget.download(dk_url, str(tar_path))

    dk_extracted_path = dk_raw_path / dk_extracted_data[i]
    if dk_extracted_path.is_dir():
        print("#### Danish Dataset part " + str(i + 1) + "/3 is already extracted ####")
    else:
        print("#### Extracting Danish Dataset " + str(i + 1) + "/3 ####")
        dk_file = tarfile.open(tar_path)
        dk_file.extractall(dk_raw_path)
        dk_file.close()
