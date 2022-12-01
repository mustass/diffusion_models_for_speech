from omegaconf import DictConfig, OmegaConf
import argparse
import hydra
from pathlib import Path
import pandas as pd
import numpy as np
import os
import pathlib
import glob
import torchaudio
import torch.nn.functional as F


def process_annotated_csv(cfg: DictConfig) -> None:
    """
    Load annotated.csv and extract dataset and filenames
    of generated files
    """

    annotations_path = Path(cfg.evaluate.annotations_path)
    ann_data = pd.read_csv(annotations_path)
    ann_data = ann_data[['audio_path', 'split']]
    ann_data = ann_data[ann_data['split'] == 1]
    ann_data['dataset'] = ann_data['audio_path'].str.split('/',expand=True)[7]
    ann_data['audio_files'] = ann_data['audio_path'].apply(Path)
    ann_data['audio_files'] = ann_data['audio_files'].apply(lambda x : x.name if '.' in x.suffix  else np.nan)
    return ann_data

def calculate_metrics(ann_dataset_specific, dataset_path, cfg: DictConfig) -> None:
    """
    Calculate stoi and snr
    """

    gen_files_path = glob.glob(str(dataset_path) +  '/*.wav')
    stoi = []
    snr = []

    for gen_path in gen_files_path:

        gen_file = pathlib.PurePath(gen_path).name.split("_")[1]
    
        matched_row  = ann_dataset_specific[ann_dataset_specific['audio_files'] == gen_file]
        input_file_path = matched_row['audio_path']

        target, sr_target = torchaudio.load(input_file_path)
        pred, sr_pred = torchaudio.load(gen_path)

        assert sr_target == sr_pred, "Target and Predicted signal sampled at different sampling frequencies"

        if pred.shape[1] > target.shape[1]:
            target =  F.pad(target, (0, pred.shape[1] - target.shape[1]), "constant", 0)
        else:
            pred = F.pad(pred,(0, target.shape[1] - pred.shape[1]), "constant", 0)

        stoi.append(float(stoi(pred,target)))
        snr.append(float(snr(pred,target)))
    return stoi, snr
    

def create_csv(ann_data, cfg: DictConfig) -> None:
    """
    Add metrics to overall csv file
    """

    for dataset_path in os.walk(cfg.evaluate.generated_path):
        dataset = pathlib.Purepath(dataset_path[0].name)
        ann_dataset_specific = ann_data[ann_data['dataset'] == dataset]
        stoi, snr = calculate_metrics(ann_dataset_specific, cfg)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate STOI and SNR evaluation metrics")
    parser.add_argument(
        "--epoch", help="Epoch", type=str, default='0'
    )
    parser.add_argument(
        "--meta_data", help="meta_data_folder", type=str, default="./data/annotations.csv"
    )
    
    args = parser.parse_args()

    hydra.initialize(config_path="../configs")

    cfg = hydra.compose(config_name="config_evaluate")

    cfg["evaluate"]["epoch"] = args.epoch
    cfg.evaluate.annotations_path = args.meta_data
    print(cfg.evaluate.epoch)
    ann_data = process_annotated_csv(cfg)
    create_csv(ann_data, cfg)
