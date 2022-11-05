from typing import Optional

from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from diffspeak.datasets import Collator, lj_speech_from_path


class LJSpeechDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.config = cfg

    def setup(self, stage: Optional[str] = None):
        # called on every GPU
        self.dataset = lj_speech_from_path(self.config)
        self.splits = random_split(self.dataset, self.config.datamodule.params.split)

        self.train = self.splits[0]
        self.val = self.splits[1]
        self.test = self.splits[2]

        self.collator = Collator(self.config)

    def train_dataloader(self):
        return DataLoader(
            self.train,
            collate_fn=self.collator.collate,
            batch_size=self.config.datamodule.params.batch_size,
            num_workers=self.config.datamodule.params.num_workers,
            pin_memory=self.config.datamodule.params.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            collate_fn=self.collator.collate,
            batch_size=self.config.datamodule.params.batch_size,
            num_workers=self.config.datamodule.params.num_workers,
            pin_memory=self.config.datamodule.params.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            collate_fn=self.collator.collate,
            batch_size=self.config.datamodule.params.batch_size,
            num_workers=self.config.datamodule.params.num_workers,
            pin_memory=self.config.datamodule.params.pin_memory,
        )
