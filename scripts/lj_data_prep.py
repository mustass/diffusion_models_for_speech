import requests
from pathlib import Path
import os

def check_and_create_data_subfolders(
    root="./data/tj/", subfolders=["raw", "interim", "processed", "external"]
):
    for folder in subfolders:
        if not os.path.exists(root + folder):
            os.makedirs(root + folder)


def download_file(url, local_filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                f.write(chunk)
    return local_filename


def download_tj_dataset():
    check_and_create_data_subfolders()

    target = (Path.cwd() / 'data' / 'tj' / 'external'/ 'LJSpeech-1.1.tar.bz2')

    file_exists = target.exists()


    if not file_exists:
        output = download_file('https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2', str(target))
        print(output)

def main():
    download_tj_dataset()

if __name__ == "__main__":
    main()