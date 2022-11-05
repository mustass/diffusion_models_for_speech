# Inspired by:
# https://github.com/lmnt-com/diffwave/blob/master/src/diffwave/preprocess.py
# ==============================================================================

from concurrent.futures import ProcessPoolExecutor
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torchaudio as T
import torchaudio.transforms as TT
from hydra.utils import get_original_cwd
from tqdm import tqdm


class Spectrogrammer:
    def __init__(self, cfg):
        self.cfg = cfg.datamodule
        self.dataset_root = Path(get_original_cwd()).joinpath(self.cfg.path)

        Path(self.dataset_root / "spectrograms").mkdir(parents=True, exist_ok=True)

    def transform(self, filename):
        audio, sr = T.load(filename)
        audio = torch.clamp(audio[0], -1.0, 1.0)

        if self.cfg.preprocessing.sample_rate != sr:
            raise ValueError(f"Invalid sample rate {sr}.")

        mel_args = {
            "sample_rate": sr,
            "win_length": self.cfg.preprocessing.hop_samples * 4,
            "hop_length": self.cfg.preprocessing.hop_samples,
            "n_fft": self.cfg.preprocessing.n_fft,
            "f_min": 20.0,
            "f_max": sr / 2.0,
            "n_mels": self.cfg.preprocessing.n_mels,
            "power": 1.0,
            "normalized": True,
        }

        mel_spec_transform = TT.MelSpectrogram(**mel_args)

        with torch.no_grad():
            spectrogram = mel_spec_transform(audio)
            spectrogram = 20 * torch.log10(torch.clamp(spectrogram, min=1e-5)) - 20
            spectrogram = torch.clamp((spectrogram + 100) / 100, 0.0, 1.0)
            np.save(
                f"{self.dataset_root}/spectrograms/{Path(filename).name}.spec.npy",
                spectrogram.cpu().numpy(),
            )

    def create_spectrograms(self):
        filenames = glob(f"{self.dataset_root}/**/*.wav", recursive=True)
        if self.cfg.preprocessing.subset_frac < 1:
            filenames = filenames[
                0 : int(len(filenames) * self.cfg.preprocessing.subset_frac)
            ]
        with ProcessPoolExecutor() as executor:
            list(
                tqdm(
                    executor.map(self.transform, filenames),
                    desc="Preprocessing",
                    total=len(filenames),
                )
            )
