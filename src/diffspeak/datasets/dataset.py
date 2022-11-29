# Inspired by
# https://github.com/lmnt-com/diffwave/blob/master/src/diffwave/dataset.py
# ==============================================================================
from pathlib import Path

import pandas as pd
import torch
import torchaudio as T
from hydra.utils import get_original_cwd
from omegaconf import DictConfig


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: DictConfig, inference: bool = False):
        super().__init__()
        self.cfg = cfg
        self.df = pd.read_csv(
            str(Path(cfg.datamodule.path_to_metadata) / "annotations.csv")
        )

        self.df = self.df[self.df["split"] == int(inference)]
        self.df = self.df[
            self.df["language"].map(lambda l: l in self.cfg.datamodule.params.datasets)
        ]
        if self.cfg.datamodule.params.remove_shorts:
            # TODO: conditional case?
            self.df = self.df[
                self.df["audio_len"] >= self.cfg.datamodule.params.audio_len
            ]
        self.df.reset_index()

    def __len__(self):
        return len(self.df)


class ConditionalDataset(AudioDataset):
    def __init__(self, cfg: DictConfig, inference=False):
        super().__init__(cfg, inference)

    def __getitem__(self, idx):
        audio_filename = self.df.iloc[idx]["audio_path"]
        spec_filename = self.df.iloc[idx]["spectrogram_path"]
        audio = T.load(audio_filename)[0][0]
        spectrogram = torch.load(spec_filename).T
        return {"audio": audio, "spectrogram": spectrogram, "filename": audio_filename}


class UnconditionalDataset(AudioDataset):
    def __init__(self, cfg, inference: bool = False):
        super().__init__(cfg, inference)

    def __getitem__(self, idx):
        audio_filename = self.df.iloc[idx]["audio_path"]
        audio = T.load(audio_filename)[0][0]
        return {"audio": audio, "spectrogram": None, "filename": audio_filename}


def lj_speech_from_path(cfg, inference=False):
    if cfg.datamodule.params.unconditional:
        dataset = UnconditionalDataset(cfg, inference)
    else:  # with spectrograms
        dataset = ConditionalDataset(cfg, inference)
    return dataset
