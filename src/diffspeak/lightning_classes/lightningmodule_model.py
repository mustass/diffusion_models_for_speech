from typing import Dict, Union

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from diffspeak.utils.technical_utils import load_obj


class LitSaModel(pl.LightningModule):
    def __init__(self, cfg: DictConfig, steps_total):
        super(LitSaModel, self).__init__()
        self.cfg = cfg
        self.steps_total = steps_total
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
        return self.model(x, mask)

    def configure_optimizers(self):
        optimizer = load_obj(self.cfg.optimizer.class_name)(
            self.model.parameters(), **self.cfg.optimizer.params
        )
        if (
            self.cfg.scheduler.class_name
            == "diffspeak.schedulers.linear_schedule_with_warmup.LinearScheduleWithWarmupConfig"
        ):
            scheduler = load_obj(self.cfg.scheduler.class_name)(
                optimizer,
                num_training_steps=self.steps_total,
                **self.cfg.scheduler.params,
            )
        else:
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
        label = batch["label"]
        dat = batch["input_ids"]
        mask = batch["attention_mask"]
        logits = self.model(dat, mask)
        loss = self.loss(logits, label)
        self.log(
            "train_Loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        for metric in self.metrics:
            score = self.metrics[metric](logits, label)
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
        label = batch["label"]
        dat = batch["input_ids"]
        mask = batch["attention_mask"]
        logits = self.model(dat, mask)
        loss = self.loss(logits, label)
        self.log(
            "val_Loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        for metric in self.metrics:
            score = self.metrics[metric](logits, label)
            self.log(
                f"val_{metric}",
                score,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

    def test_step(self, batch, batch_idx):
        label = batch["label"]
        dat = batch["input_ids"]
        mask = batch["attention_mask"]
        logits = self.model(dat, mask)
        loss = self.loss(logits, label)
        self.log(
            "test_Loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        for metric in self.metrics:
            score = self.metrics[metric](logits, label)
            self.log(
                f"test_{metric}",
                score,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
