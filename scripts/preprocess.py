import os
import warnings
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from diffspeak.utils.technical_utils import load_obj
from diffspeak.utils.utils import save_useful_info

warnings.filterwarnings("ignore")


def preprocess(cfg: DictConfig) -> None:
    """
    Preprocess the .wav files to create spectrograms for conditional
    audio synthesis.
    """

    data_path_prefix = Path(os.getenv("DATA_PATH_PREFIX"))
    for dataset_name in cfg.datamodule.params.datasets:
        root_dir = data_path_prefix / 'data' / dataset_name
        transformer = load_obj(cfg.datamodule.preprocessing.transformer)(cfg, root_dir=root_dir)
        transformer.create_spectrograms()
        audiolengainer = load_obj("diffspeak.datasets.utils.AudioLengthsToCSV")(cfg, root_dir=root_dir)
        audiolengainer.create_audio_lengths()


@hydra.main(config_path="../configs", config_name="config")
def run_preprocessing(cfg: DictConfig) -> None:
    os.makedirs("logs", exist_ok=True)
    print(OmegaConf.to_yaml(cfg))
    if cfg.general.log_code:
        save_useful_info(os.path.basename(__file__))
    preprocess(cfg)


if __name__ == "__main__":
    run_preprocessing()
