from pathlib import Path
import wget

print("#### Checking and creating data folder if it does not exist ####")
en_path = Path.cwd() / 'data/tj/external'
en_raw_path = Path.cwd() / 'data/tj/raw'
dk_path = Path.cwd() / 'data/nst_danish/external'
dk_raw_path = Path.cwd() / 'data/nst_danish/raw'
en_path.mkdir(parents=True, exist_ok=True)
en_raw_path.mkdir(parents=True, exist_ok=True)
dk_path.mkdir(parents=True, exist_ok=True)
dk_raw_path.mkdir(parents=True, exist_ok=True)

en_tar = en_path / 'LJSpeech-1.1.tar.bz2'
if en_tar.is_file():
    print("#### LJSpeech Dataset is already there ####")
else:
    print("#### Downloading LJSpeech dataset ####")
    en_url = 'https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2'
    wget.download(en_url,out = en_path)

dk_tars = ['da.16kHz.0611.tar.gz', 'da.16kHz.0565-1.tar.gz']

for i, tar in enumerate(dk_tars):
    tar_path = dk_path / tar
    if tar_path.is_file():
        print("#### Danish Dataset " + str(i+1) + "/3 is already there ####")
    else:
        print("#### Downloading Danish Dataset part " + str(i+1) + "/3 ####")
        dk_url = 'https://www.nb.no/sbfil/talegjenkjenning/16kHz/' + tar
        wget.download(dk_url, str(tar_path))

