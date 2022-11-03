import os
import warnings
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from diffspeak.utils.technical_utils import load_obj
from diffspeak.utils.utils import save_useful_info, set_seed
from diffspeak.datasets import Transformer

warnings.filterwarnings("ignore")


def preprocess(cfg: DictConfig) -> None:
    """
    Preprocess the .wav files to create spectrograms for conditional
    audio synthesis.
    """

    params = cfg.preprocessing

    transofrmer = Transformer(params)
    transofrmer.create_spectrograms()


@hydra.main(config_path="../configs", config_name="preprocessing_config")
def run_preprocessing(cfg: DictConfig) -> None:
    os.makedirs("logs", exist_ok=True)
    print(OmegaConf.to_yaml(cfg))
    if cfg.general.log_code:
        save_useful_info(os.path.basename(__file__))
    preprocess(cfg)


if __name__ == "__main__":
    run_preprocessing()
