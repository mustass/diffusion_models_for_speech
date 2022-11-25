# Inspired by
# https://github.com/lmnt-com/diffwave/blob/master/src/diffwave/dataset.py
# ==============================================================================

from glob import glob
from pathlib import Path

import pandas as pd
import torch
import torchaudio as T
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
import os


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        self.data_path_prefix = Path(os.getenv("DATA_PATH_PREFIX"))
        # self.audio_path = self.data_path_prefix / "raw"
        # self.spectrograms_path = self.data_path_prefix / "spectrograms"

        if self.cfg.datamodule.params.remove_shorts:
            assert (
                self.dataset_root / "audio_lenghts.csv"
            ).exists(), "The metadata file audio_lenghts.csv does not exist. Run the preprocessing before proceeding"

            self.read_audiolens()
        self.filenames = self.filenames.apply(
            lambda l: Path(get_original_cwd() / Path(l))
        )

    def __len__(self):
        return len(self.filenames)

    def read_audiolens(self):
        audio_lengths = pd.read_csv(self.dataset_root / "audio_lenghts.csv")
        assert len(audio_lengths) == len(self.filenames)
        self.filenames = audio_lengths[
            audio_lengths["length"] >= self.cfg.datamodule.params.audio_len
        ]["path"].reset_index(drop=True)
    
    def _get_filenames(self):
        if self.cfg.datamodule.params.remove_shorts == True:
            #TODO: This could be moved to config_utils with some considerations
            assert (
                self.dataset_root / "audio_lenghts.csv"
            ).exists(), "The metadata file audio_lenghts.csv does not exist. Run the preprocessing before proceeding"
            self.read_audiolens()
        else :
            filenames_list = glob(f"{self.dataset_root}/**/*.wav", recursive=True)



class ConditionalDataset(AudioDataset):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def __getitem__(self, idx):
        audio_filename = self.filenames.loc[idx]
        spec_filename = f"{self.spectrograms_path / Path(audio_filename).name}.spec.pt"
        signal, _ = T.load(audio_filename)
        spectrogram = torch.load(spec_filename)
        return {"audio": signal[0], "spectrogram": spectrogram.T}


class UnconditionalDataset(AudioDataset):
    def __init__(self, cfg):
        super().__init__(cfg)

    def __getitem__(self, idx):
        audio_filename = self.filenames.loc[idx]
        signal, _ = T.load(audio_filename)
        return {"audio": signal[0], "spectrogram": None}


def lj_speech_from_path(cfg):
    if cfg.datamodule.params.unconditional:
        dataset = UnconditionalDataset(cfg)
    else:  # with spectrograms
        dataset = ConditionalDataset(cfg)
    return dataset
