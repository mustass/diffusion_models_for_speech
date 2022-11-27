from typing import Any

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from tqdm import tqdm

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

    def forward(self, x, *args, **kwargs):
        spectrogram = x["spectrogram"]
        # lang = x["lang"] TODO in the future

        beta = self.model.noise_schedule
        alpha = torch.ones_like(beta) - beta
        alpha_cum = torch.cumprod(alpha, 0)

        T = torch.tensor(list(range(len(alpha_cum))))

        if (
            self.cfg.model.params.inference_noise_schedule is not None
            and not self.cfg.model.params.inference_noise_schedule
            == self.model.noise_schedule
        ):
            T, beta, alpha, alpha_cum = self.adjust_Ts(alpha_cum)

        if not self.cfg.model.params.unconditional:
            if (
                len(spectrogram.shape) == 2
            ):  # Expand rank 2 tensors by adding a batch dimension.
                spectrogram = spectrogram.unsqueeze(0)
            audio = torch.randn(
                spectrogram.shape[0],
                self.cfg.model.params.hop_samples * spectrogram.shape[-1],
            )
        else:
            audio = torch.randn(1, self.cfg.datamodule.params.audio_len)

        for n in (pbar := tqdm(range(len(alpha) - 1, -1, -1))) :
            pbar.set_description(f"Denoising step {n}")
            c1 = 1 / alpha[n] ** 0.5
            c2 = beta[n] / (1 - alpha_cum[n]) ** 0.5
            audio = c1 * (
                audio
                - c2 * self.model(audio, torch.tensor([T[n]]), spectrogram).squeeze(1)
            )
            if n > 0:
                noise = torch.randn_like(audio)
                sigma = (
                    (1.0 - alpha_cum[n - 1]) / (1.0 - alpha_cum[n]) * beta[n]
                ) ** 0.5
                audio += sigma * noise
            audio = torch.clamp(audio, -1.0, 1.0)

        return audio

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
        noise_scale_sqrt = noise_scale ** 0.5
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
        noise_scale_sqrt = noise_scale ** 0.5
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
        noise_scale_sqrt = noise_scale ** 0.5
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

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0,
    ):
        return self(batch)

    def adjust_Ts(self, alpha_train_cum):
        beta = torch.tensor(self.cfg.model.params.inference_noise_schedule)
        alpha = torch.ones_like(beta) - beta
        alpha_cum = torch.cumprod(alpha, 0)

        T = []
        for s in range(len(self.cfg.model.params.inference_noise_schedule)):
            for t in range(len(self.model.noise_schedule) - 1):
                if alpha_train_cum[t + 1] <= alpha_cum[s] <= alpha_train_cum[t]:
                    twiddle = (alpha_train_cum[t] ** 0.5 - alpha_cum[s] ** 0.5) / (
                        alpha_train_cum[t] ** 0.5 - alpha_train_cum[t + 1] ** 0.5
                    )
                    T.append(t + twiddle)
                    break
        return torch.tensor(T), beta, alpha, alpha_cum
