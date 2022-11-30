from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
import torchaudio
from pathlib import Path
import glob
import pathlib
import pandas as pd
import numpy as np
import torch.nn.functional as F


data_path = Path('/dtu/blackhole/19/s176453/diffusion_for_speech')
gen_path = data_path / 'synthesized_audio/pretrained_model @Sandor, here'
gen_files = glob.glob(str(gen_path) +  '/*.wav')

stoi_tj = []
stoi_nst_danish = []
snr_tj = []
snr_nb_danish = []

sr = 16000
stoi = ShortTimeObjectiveIntelligibility(sr, False)


annotations = Path('./data/annotations.csv')
ann = pd.read_csv(annotations)

ann_data = pd.read_csv(annotations)
ann_data = ann_data[['audio_path', 'split']]
ann_data['dataset'] = ann_data['audio_path'].str.split('/',expand=True)[7]
ann_data['audio_files'] = ann_data['audio_path'].apply(Path)
ann_data['audio_files'] = ann_data['audio_files'].apply(lambda x : x.name if '.' in x.suffix  else np.nan)

for gen_file_path in gen_files:
    
    gen_file = pathlib.PurePath(gen_file_path).name.split("_")[1]
    pred, sr = torchaudio.load(gen_file_path)
    
    matched_row  = ann_data[ann_data['audio_files'] == gen_file].iloc[0]
    input_file_path = matched_row['audio_path']
    dataset = matched_row['dataset']

    target, sr_target = torchaudio.load(input_file_path)
    pred, sr_pred = torchaudio.load(gen_file_path)

    assert sr_target == sr_pred, "Target and Predicted signal sampled at different sampling frequencies"


    if pred.shape[1] > target.shape[1]:
        target =  F.pad(target, (0, pred.shape[1] - target.shape[1]), "constant", 0)
    else:
        pred = F.pad(pred,(0, target.shape[1] - pred.shape[1]), "constant", 0)

    if dataset == 'tj':
        stoi_tj.append(float(stoi(pred,target)))
    elif dataset == 'nst_danish':
        stoi_nst_danish.append(float(stoi(pred,target)))