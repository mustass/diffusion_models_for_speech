import argparse
import glob
import os
import pathlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch.nn.functional as F
import torchaudio
from torchmetrics import SignalNoiseRatio
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility


def process_annotated_csv(annotations_path) -> None:
    """
    Load annotated.csv and extract dataset and filenames
    of generated files
    """

    annotations_path = Path(annotations_path)

    ann_data = pd.read_csv(annotations_path)
    ann_data = ann_data[["audio_path", "split"]]
    ann_data = ann_data[ann_data["split"] == 1]
    ann_data["dataset"] = ann_data["audio_path"].str.split("/", expand=True)[7]
    ann_data["audio_files"] = ann_data["audio_path"].apply(Path)
    ann_data["audio_files"] = ann_data["audio_files"].apply(
        lambda x: x.name if "." in x.suffix else np.nan
    )
    return ann_data


def calculate_metrics(ann_dataset_specific, dataset_path, sr) -> None:
    """
    Calculate stoi and snr
    """

    gen_files_path = glob.glob(str(dataset_path) + "/*.wav")
    snr = SignalNoiseRatio()
    stoi = ShortTimeObjectiveIntelligibility(sr, False)

    stoi_total = []
    snr_total = []
    files = []

    for gen_path in gen_files_path:

        gen_file = pathlib.PurePath(gen_path).name.split("_")[1]

        matched_row = ann_dataset_specific[
            ann_dataset_specific["audio_files"] == gen_file
        ]
        input_file_path = matched_row["audio_path"].iloc[0]

        target, sr_target = torchaudio.load(input_file_path)
        pred, sr_pred = torchaudio.load(gen_path)

        assert (
            sr_target == sr_pred
        ), "Target and Predicted signal sampled at different sampling frequencies"

        if pred.shape[1] > target.shape[1]:
            target = F.pad(target, (0, pred.shape[1] - target.shape[1]), "constant", 0)
        else:
            pred = F.pad(pred, (0, target.shape[1] - pred.shape[1]), "constant", 0)

        stoi_total.append(float(stoi(pred, target)))
        snr_total.append(float(snr(pred, target)))
        files.append(gen_file)
    return stoi_total, snr_total, files


def create_csv(ann_data, save_path, generated_path, experiment, model) -> None:
    """
    Add metrics to overall csv file
    """
    filename = "exp_" + experiment + "_metrics.csv"
    filepath = save_path / filename
    try:
        df_data = pd.read_csv(filepath, index_col=0)

    except FileNotFoundError:
        df_data = pd.DataFrame()

    for subdir, dirs, files in os.walk(generated_path):
        for dataset in dirs:
            dataset_path = Path(subdir) / Path(dataset)
            ann_dataset_specific = ann_data[ann_data["dataset"] == dataset]
            stoi, snr, files = calculate_metrics(
                ann_dataset_specific, dataset_path, args.sr
            )
            new_metrics = {
                "stoi": stoi,
                "snr": snr,
                "audio_file": files,
                "model": [model] * len(stoi),
                "dataset": [dataset] * len(stoi),
            }
            df_data = df_data.append(pd.DataFrame(new_metrics))

    filepath.parent.mkdir(parents=True, exist_ok=True)
    df_data.to_csv(filepath)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Calculate STOI and SNR evaluation metrics"
    )

    parser.add_argument(
        "--synthesis_dir",
        help="synthesized data folder",
        type=str,
        default="/dtu/blackhole/19/s176453/diffusion_for_speech/synthesized_audio",
    )
    parser.add_argument("--experiment", help="Experiment", type=str, default="1")
    parser.add_argument("--model", help="Model", type=str, default="epoch0")
    parser.add_argument(
        "--meta_data",
        help="meta_data_folder",
        type=str,
        default="./data/annotations.csv",
    )
    parser.add_argument("--sr", help="Sampling Rate", type=int, default=22050)
    parser.add_argument(
        "--save_dir",
        help="Saved Metrics folder",
        type=str,
        default="/dtu/blackhole/19/s176453/diffusion_for_speech/evaluate",
    )

    args = parser.parse_args()

    ann_data = process_annotated_csv(args.meta_data)
    generated_path = Path(args.synthesis_dir) / (
        "experiment" + args.experiment + "/" + args.model
    )
    save_path = Path(args.save_dir)
    create_csv(ann_data, save_path, generated_path, args.experiment, args.model)
