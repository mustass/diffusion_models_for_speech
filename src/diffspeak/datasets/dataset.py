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


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.filenames = []

        self.dataset_root = Path(get_original_cwd()).joinpath(self.cfg.datamodule.path)
        self.spectrograms_path = self.dataset_root / "spectrograms"
        self.filenames = pd.Series(
            glob(f"{self.dataset_root}/**/*.wav", recursive=True)
        )
        if self.cfg.datamodule.params.remove_shorts:
            assert (
                self.cfg.datamodule.params.collator
                == "diffspeak.datasets.collator.Collator"
            ), "Handling too short audio in the collator is not necessary when remove_shorts = True"
            assert (
                self.dataset_root / "audio_lenghts.csv"
            ).exists(), "The metadata file audio_lenghts.csv does not exist. Run the preprocessing before proceeding"

            self.remove_shorts()
        self.filenames = self.filenames.apply(
            lambda l: Path(get_original_cwd() / Path(l))
        )

    def __len__(self):
        return len(self.filenames)

    def remove_shorts(self):
        audio_lengths = pd.read_csv(self.dataset_root / "audio_lenghts.csv")
        assert len(audio_lengths) == len(self.filenames)
        self.filenames = audio_lengths[
            audio_lengths["length"] >= self.cfg.datamodule.params.audio_len
        ]["path"].reset_index(drop=True)


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
