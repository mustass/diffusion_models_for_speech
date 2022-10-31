from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from diffspeak.datasets.amazon_utils import ensemble, clean_data
from hydra.utils import get_original_cwd
from typing import Optional
from omegaconf import DictConfig
from datasets import load_dataset

# This is hackidy-hacky:
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
# Hackidy hack over ¯\_(ツ)_/¯


class AmazonDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.config = cfg
        self.wd = get_original_cwd()
        self.prepare_data()

    def prepare_data(self):
        ensemble(self.config.datamodule, self.wd)
        clean_data(self.config.datamodule, self.wd)

    def setup(self, stage: Optional[str] = None):
        # called on every GPU
        datasets = self.load_datasets(
            f'{self.wd}/data/SA_amazon_data/processed/{self.config.datamodule["name"]}/'
        )
        self.train = datasets["train"]
        self.val = datasets["val"]
        self.test = datasets["test"]
        self.t_total = len(self.train) // self.config.datamodule["batch_size"] * self.config.trainer.max_epochs

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.config.datamodule["batch_size"],
            num_workers=self.config.datamodule["num_workers"],
            shuffle=True
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

    def load_datasets(self, folder_path):
        try:
            datasets = load_dataset(
                "parquet",
                data_files={
                    "train": f"{folder_path}train.parquet",
                    "val": f"{folder_path}val.parquet",
                    "test": f"{folder_path}test.parquet",
                },
            ).with_format("torch")
            return datasets
        except Exception as ex:
            if type(ex) == FileNotFoundError:
                raise FileNotFoundError(
                    f"The datasets could not be found in {folder_path}"
                )
