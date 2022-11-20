import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from diffspeak.utils.technical_utils import load_obj


class LitDiffWaveModel(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        self.model = load_obj(cfg.model.class_name)(cfg=cfg)
        self.loss = load_obj(cfg.loss.class_name)()

        self.metrics = torch.nn.ModuleDict(
            {
                self.cfg.metric.metric.metric_name: load_obj(
                    self.cfg.metric.metric.class_name
                )(**cfg.metric.metric.params)
            }
        )

    def forward(self, x, mask, *args, **kwargs):
        print(f"Getting {x} for lightning module forward.")
        return self.model(x)

    def configure_optimizers(self):
        optimizer = load_obj(self.cfg.optimizer.class_name)(
            self.model.parameters(), **self.cfg.optimizer.params
        )

        scheduler = load_obj(self.cfg.scheduler.class_name)(
            optimizer, **self.cfg.scheduler.params
        )

        return (
            [optimizer],
            [
                {
                    "scheduler": scheduler,
                    "interval": self.cfg.scheduler.step,
                    "monitor": self.cfg.scheduler.monitor,
                }
            ],
        )

    def training_step(self, batch, batch_idx):
        audio = batch["audio"]
        spectrogram = batch["spectrogram"]

        N, T = audio.shape

        t = torch.randint(0, len(self.model.noise_schedule), [N])
        noise_scale = self.model.noise_level[t].unsqueeze(1).to(audio)
        noise_scale_sqrt = noise_scale**0.5
        noise = torch.randn_like(audio)
        noisy_audio = noise_scale_sqrt * audio + (1.0 - noise_scale) ** 0.5 * noise

        predicted = self.model(noisy_audio, t, spectrogram)
        loss = self.loss(noise, predicted.squeeze(1))
        self.log(
            "train_Loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        for metric in self.metrics:
            score = self.metrics[metric](
                noise, predicted.squeeze(1)
            )  # lol, this probably makes no sense. But I do not know what metrics make sense right now.
            self.log(
                f"train_{metric}",
                score,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        return loss

    def validation_step(self, batch, batch_idx):
        audio = batch["audio"]
        spectrogram = batch["spectrogram"]

        N, T = audio.shape

        t = torch.randint(0, len(self.model.noise_schedule), [N])
        noise_scale = self.model.noise_level[t].unsqueeze(1).to(audio)
        noise_scale_sqrt = noise_scale**0.5
        noise = torch.randn_like(audio)
        noisy_audio = noise_scale_sqrt * audio + (1.0 - noise_scale) ** 0.5 * noise

        predicted = self.model(noisy_audio, t, spectrogram)
        loss = self.loss(noise, predicted.squeeze(1))

        self.log(
            "val_Loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        for metric in self.metrics:
            score = self.metrics[metric](noise, predicted.squeeze(1))
            self.log(
                f"val_{metric}",
                score,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

    def test_step(self, batch, batch_idx):
        audio = batch["audio"]
        spectrogram = batch["spectrogram"]

        N, T = audio.shape

        t = torch.randint(0, len(self.model.noise_schedule), [N])
        noise_scale = self.model.noise_level[t].unsqueeze(1).to(audio)
        noise_scale_sqrt = noise_scale**0.5
        noise = torch.randn_like(audio)
        noisy_audio = noise_scale_sqrt * audio + (1.0 - noise_scale) ** 0.5 * noise

        predicted = self.model(noisy_audio, t, spectrogram)
        loss = self.loss(noise, predicted.squeeze(1))

        self.log(
            "test_Loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        for metric in self.metrics:
            score = self.metrics[metric](noise, predicted.squeeze(1))
            self.log(
                f"test_{metric}",
                score,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
