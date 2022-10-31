import requests
from pathlib import Path

def check_and_create_data_subfolders(
    root="/data/tj/", subfolders=["raw", "interim", "processed", "external"]
):
    for folder in subfolders:
        if not os.path.exists(root + folder):
            os.makedirs(root + folder)


def download_tj_dataset():
    check_and_create_data_subfolders()

    target = (Path.cwd() / 'data' / 'tj' / 'external'/ 'LJSpeech-1.1.tar.bz2')

    file_exists = target.exists()


    if not file_exists:
        URL = 'https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2'
        response = requests.get(URL)
        open(str(target), "wb").write(response.content)


def main():
    download_tj_dataset()

if __name__ == "__main__":
    main()