# Inspired by:
# https://github.com/lmnt-com/diffwave/blob/master/src/diffwave/preprocess.py
# ==============================================================================

import os
from glob import glob
from pathlib import Path

import pandas as pd
import scipy
import torch
import torchaudio as T
import torchaudio.transforms as TT
from tqdm import tqdm


class Preprocessor:
    def __init__(self, cfg):
        self.cfg = cfg.datamodule
        self.data_path_prefix = Path(os.getenv("DATA_PATH_PREFIX"))
        self.audio_paths = self.get_audio_paths()
        self.annotations = []

    def get_audio_paths(self):
        audio_paths = []
        for dataset in self.cfg.params.datasets:
            audio_paths += glob(
                str(self.data_path_prefix / "data" / dataset / "**/*.wav"),
                recursive=True,
            )
        return audio_paths

    def get_language_from_audio_path(self, audio_path):
        return audio_path.relative_to(self.data_path_prefix).parts[1]

    def get_spec_path_from_audio_path(self, audio_path):
        return (
            self.data_path_prefix
            / Path(*audio_path.relative_to(self.data_path_prefix).parts[:2])
            / "spectrograms"
            / Path(*audio_path.relative_to(self.data_path_prefix).parts[3:-1])
            / Path(audio_path.stem).with_suffix(".pt")
        )

    def get_processed_audio_path(self, audio_path):
        return (
            self.data_path_prefix
            / Path(*audio_path.relative_to(self.data_path_prefix).parts[:2])
            / "processed"
            / Path(*audio_path.relative_to(self.data_path_prefix).parts[3:])
        )

    def transform(self, spec_path, processed_wav_path, audio, sr):
        audio = torch.clamp(audio[0], -1.0, 1.0)

        if self.cfg.preprocessing.sample_rate != sr:
            audio = T.functional.resample(
                audio, orig_freq=sr, new_freq=self.cfg.preprocessing.sample_rate
            )
            sr = self.cfg.preprocessing.sample_rate

        processed_wav_path.parent.mkdir(parents=True, exist_ok=True)
        T.save(processed_wav_path, audio.unsqueeze(dim=1).T, sr)

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
            Path(spec_path).parent.mkdir(exist_ok=True, parents=True)
            torch.save(
                spectrogram.cpu(), spec_path,
            )

    def preprocess_audio_file(self, audio_path):
        audio_path = Path(audio_path)
        try:
            audio, sr = T.load(audio_path)
        except Exception as e:
            print(e)
            return

        spec_path = self.get_spec_path_from_audio_path(audio_path)
        processed_wav_path = self.get_processed_audio_path(audio_path)
        self.transform(spec_path, processed_wav_path, audio, sr)
        language = self.get_language_from_audio_path(audio_path)
        self.annotations.append(
            {
                "audio_path": processed_wav_path,
                "spectrogram_path": spec_path,
                "audio_len": audio.shape[1],
                "language": language,
            }
        )

    def preprocess_audio_files(self):
        filenames = self.audio_paths
        if self.cfg.preprocessing.subset_frac < 1:
            filenames = filenames[
                0 : int(len(filenames) * self.cfg.preprocessing.subset_frac)
            ]

        for filename in tqdm(filenames):
            self.preprocess_audio_file(filename)

        df = pd.DataFrame(self.annotations)
        df = self.split(df, self.cfg.datamodule.params.split_for_conditional_inference)
        df.to_csv(self.data_path_prefix / "data" / "annotations.csv")

    @staticmethod
    def split(df, split_sizes=[0.99, 0.01]):
        assert sum(split_sizes.copy()) == 1, "Split sizes must add up to 1."
        x = scipy.stats.multinomial(1, split_sizes)
        sample = x.rvs(len(df))
        df["split"] = sample.argmax(axis=1)
        return df
