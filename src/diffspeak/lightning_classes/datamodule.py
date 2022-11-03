# This is hackidy-hacky:
import os
from typing import Optional

from datasets import load_dataset
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from diffspeak.datasets.preprocessing_utils import transform, ensemble
from diffspeak.datasets.dataset import from_path

class LJSpeechDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.config = cfg

    def setup(self, stage: Optional[str] = None):
        # called on every GPU

        self.dataset = 





        datasets = self.load_datasets(
            f'{self.wd}/data/SA_amazon_data/processed/{self.config.datamodule["name"]}/'
        )
        self.train = datasets["train"]
        self.val = datasets["val"]
        self.test = datasets["test"]
        self.t_total = (
            len(self.train)
            // self.config.datamodule["batch_size"]
            * self.config.trainer.max_epochs
        )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.config.datamodule["batch_size"],
            num_workers=self.config.datamodule["num_workers"],
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.config.datamodule["batch_size"],
            num_workers=self.config.datamodule["num_workers"],
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.config.datamodule["batch_size"],
            num_workers=self.config.datamodule["num_workers"],
        )

    def load_dataset(self, folder_path):
        try:
            dataset = from_path(self.config.datamodule)
            return dataset
        except Exception as ex:
            if type(ex) == FileNotFoundError:
                raise FileNotFoundError(
                    f"The datasets could not be found in {folder_path}"
                )
