import random

import torch
import torch.nn.functional as F


class Collator:
    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def collate(self, minibatch):
        for record in minibatch:
            subsample(self.cfg, record)
        return self.assamble(minibatch)

    def assamble(self, minibatch):
        audio = torch.stack(
            [record["audio"] for record in minibatch if "audio" in record]
        )

        if self.cfg.datamodule.params.unconditional:
            return {
                "audio": audio,
                "spectrogram": None,
            }
        spectrogram = torch.stack(
            [record["spectrogram"] for record in minibatch if "spectrogram" in record]
        )
        return {
            "audio": audio,
            "spectrogram": spectrogram,
            "filename": [
                record["filename"] for record in minibatch if "filename" in record
            ]
            if "filename" in minibatch[0].keys()
            else None,
        }


class InferenceCollator(Collator):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

    def collate(self, minibatch):
        max_len = max([record["spectrogram"].shape[0] for record in minibatch])
        for record in minibatch:
            zero_pad(self.cfg, record, max_len)
            record["spectrogram"] = record["spectrogram"].T
        return self.assamble(minibatch)


class ZeroPadCollator(Collator):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

    def collate(self, minibatch):
        for record in minibatch:
            zero_pad(self.cfg, record)
            subsample(self.cfg, record)
        return self.assamble(minibatch)


def zero_pad(cfg, record, inference_len=None):
    if cfg.datamodule.params.unconditional:
        if len(record["audio"]) < cfg.datamodule.params.audio_len:
            pad_size_audio = max(
                0, cfg.datamodule.params.audio_len - len(record["audio"])
            )
            record["audio"] = F.pad(record["audio"], (0, pad_size_audio), "constant", 0)
    else:
        required_len = (
            cfg.datamodule.params.crop_mel_frames
            if inference_len is None
            else inference_len
        )

        if len(record["spectrogram"]) < required_len:
            pad_size_spectrogram = max(0, required_len - len(record["spectrogram"]))
            record["spectrogram"] = F.pad(
                record["spectrogram"], (0, 0, 0, pad_size_spectrogram), "constant", 0
            )
        len_audio = required_len * cfg.datamodule.preprocessing.hop_samples
        delta_len_audio = len_audio - len(record["audio"])
        if delta_len_audio > 0:
            record["audio"] = F.pad(
                record["audio"], (0, delta_len_audio), "constant", 0
            )
        else:
            record["audio"] = record["audio"][0:len_audio]


def subsample(cfg, record):
    if cfg.datamodule.params.unconditional:
        start = random.randint(
            0, max(record["audio"].shape[-1] - cfg.datamodule.params.audio_len, 0)
        )
        end = start + cfg.datamodule.params.audio_len
        record["audio"] = torch.squeeze(record["audio"][start:end])
        record["audio"] = F.pad(
            record["audio"],
            (0, (end - start) - len(record["audio"])),
            mode="constant",
            value=0,
        )

    else:
        samples_per_frame = cfg.datamodule.preprocessing.hop_samples
        start = random.randint(
            0,
            max(
                0,
                record["spectrogram"].shape[0] - cfg.datamodule.params.crop_mel_frames,
            ),
        )
        end = start + cfg.datamodule.params.crop_mel_frames

        record["spectrogram"] = record["spectrogram"][start:end].T

        start *= samples_per_frame
        end *= samples_per_frame
        record["audio"] = torch.squeeze(record["audio"][start:end])
        record["audio"] = F.pad(
            record["audio"],
            (0, (end - start) - len(record["audio"])),
            mode="constant",
            value=0,
        )
