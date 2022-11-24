import random

import torch
import torch.nn.functional as F


class DeleteShorts:
    def __init__(self, cfg):
        self.cfg = cfg

    def collate(self, minibatch):

        for record in minibatch:
            self.delete_shorts(record)
        return self.assamble(minibatch)

    def delete_shorts(self, record):
        samples_per_frame = self.cfg.datamodule.preprocessing.hop_samples
        if self.cfg.datamodule.params.unconditional:
            # Filter out records that aren't long enough.
            if len(record["audio"]) < self.cfg.datamodule.params.audio_len:
                del record["spectrogram"]
                del record["audio"]
                return -1

            start = random.randint(
                0, record["audio"].shape[-1] - self.cfg.datamodule.params.audio_len
            )
            end = start + self.cfg.datamodule.params.audio_len
            record["audio"] = torch.squeeze(record["audio"][start:end])
            record["audio"] = F.pad(
                record["audio"],
                (0, (end - start) - len(record["audio"])),
                mode="constant",
                value=0,
            )
            return 0

        else:
            # Filter out records that aren't long enough.
            if len(record["spectrogram"]) < self.cfg.datamodule.params.crop_mel_frames:
                del record["spectrogram"]
                del record["audio"]
                return -1

            start = random.randint(
                0,
                record["spectrogram"].shape[0]
                - self.cfg.datamodule.params.crop_mel_frames,
            )
            end = start + self.cfg.datamodule.params.crop_mel_frames
            record["spectrogram"] = record["spectrogram"][start:end].T

            start *= samples_per_frame
            end *= samples_per_frame
            record["audio"] = torch.squeeze(
                record["audio"][start:end]
            )  # Depends on the shape here
            record["audio"] = F.pad(
                record["audio"],
                (0, (end - start) - len(record["audio"])),
                mode="constant",
                value=0,
            )
            return 0

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
        }


class ZeroPad:
    def __init__(self, cfg):
        self.cfg = cfg

    def collate(self, minibatch):
        for record in minibatch:
            self.zero_pad(record)
        return self.assamble(minibatch)

    def zero_pad(self, record):
        if self.cfg.datamodule.params.unconditional:
            len_audio = self.cfg.datamodule.params.audio_len

            start_audio = random.randint(
                0, max(0, record["audio"].shape[0] - len_audio)
            )
            end_audio = start_audio + len_audio

            pad_size_audio = max(0, len_audio - record["audio"].shape[0])

            record["audio"] = F.pad(record["audio"], (0, pad_size_audio), "constant", 0)
            record["audio"] = record["audio"][start_audio:end_audio]
        else:
            samples_per_frame = self.cfg.datamodule.preprocessing.hop_samples

            len_spectrogram = self.cfg.datamodule.params.crop_mel_frames

            start_spectrogram = random.randint(
                0, max(0, record["spectrogram"].shape[0] - len_spectrogram)
            )
            end_spectrogram = (
                start_spectrogram + self.cfg.datamodule.params.crop_mel_frames
            )

            pad_size_spectrogram = max(
                0, len_spectrogram - record["spectrogram"].shape[0]
            )

            record["spectrogram"] = F.pad(
                record["spectrogram"], (0, pad_size_spectrogram), "constant", 0
            )
            record["spectrogram"] = record["spectrogram"][
                start_spectrogram:end_spectrogram
            ].T

            len_audio = len_spectrogram * samples_per_frame

            start_audio = int(start_spectrogram / samples_per_frame)
            end_audio = start_audio + len_audio

            pad_size_audio = max(0, len_audio - record["audio"].shape[0])

            record["audio"] = F.pad(record["audio"], (0, pad_size_audio), "constant", 0)
            record["audio"] = record["audio"][start_audio:end_audio]

        return 0

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
        }
