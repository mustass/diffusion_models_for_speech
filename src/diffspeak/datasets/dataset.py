# Inspired by
# https://github.com/lmnt-com/diffwave/blob/master/src/diffwave/dataset.py
# ==============================================================================
import os
import pandas as pd

from pathlib import Path
from omegaconf import DictConfig

import torch
import torchaudio as T

from hydra.utils import get_original_cwd


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.df = pd.read_csv(str(Path(get_original_cwd()) / "data" / "annotations.csv"))
        self.df = self.df[self.df['split'] == 0]
        if self.cfg.datamodule.params.remove_shorts:
            #TODO: conditional case?
            self.df = self.df[self.df['audio_len'] >= self.cfg.datamodule.params.audio_len]
        self.df.reset_index()

    def __len__(self):
        return len(self.df)


class ConditionalDataset(AudioDataset):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def __getitem__(self, idx):
        audio_filename = self.df.iloc[idx]['audio_path']
        spec_filename = self.df.iloc[idx]['spectrogram_path']
        audio = T.load(audio_filename)[0][0]
        spectrogram = torch.load(spec_filename).T
        return {"audio": audio, "spectrogram": spectrogram}


class UnconditionalDataset(AudioDataset):
    def __init__(self, cfg):
        super().__init__(cfg)

    def __getitem__(self, idx):
        audio_filename = self.df.iloc[idx]['audio_path']
        audio = T.load(audio_filename)[0][0]
        return {"audio": audio, "spectrogram": None}


def lj_speech_from_path(cfg):
    if cfg.datamodule.params.unconditional:
        dataset = UnconditionalDataset(cfg)
    else:  # with spectrograms
        dataset = ConditionalDataset(cfg)
    return dataset
