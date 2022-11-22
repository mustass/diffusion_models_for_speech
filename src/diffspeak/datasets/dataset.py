# Inspired by
# https://github.com/lmnt-com/diffwave/blob/master/src/diffwave/dataset.py
# ==============================================================================

import random
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from hydra.utils import get_original_cwd
from omegaconf import DictConfig


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.filenames = []

        self.dataset_root = Path(get_original_cwd()).joinpath(self.cfg.datamodule.path)
        self.spectrograms_path = self.dataset_root / "spectrograms"
        self.filenames = glob(f"{self.dataset_root}/**/*.wav", recursive=True)

    def __len__(self):
        return len(self.filenames)


class ConditionalDataset(AudioDataset):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def __getitem__(self, idx):
        audio_filename = self.filenames[idx]
        spec_filename = f"{self.spectrograms_path / Path(audio_filename).name}.spec.pt"
        signal, _ = torchaudio.load(audio_filename)
        spectrogram = torch.load(spec_filename)
        return {"audio": signal[0], "spectrogram": spectrogram.T}


class UnconditionalDataset(AudioDataset):
    def __init__(self, cfg):
        super().__init__(cfg)

    def __getitem__(self, idx):
        audio_filename = self.filenames[idx]
        signal, _ = torchaudio.load(audio_filename)
        return {"audio": signal[0], "spectrogram": None}


def lj_speech_from_path(cfg):
    if cfg.datamodule.params.unconditional:
        dataset = UnconditionalDataset(cfg)
    else:  # with spectrograms
        dataset = ConditionalDataset(cfg)
    return dataset


