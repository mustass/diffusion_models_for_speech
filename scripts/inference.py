import argparse
import glob
from hydra import initialize, compose
import torch
import torchaudio as T
import yaml
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from tqdm import tqdm
from diffspeak.utils.utils import set_seed
from diffspeak.utils.technical_utils import load_obj



def synthesize_audio(cfg: DictConfig) -> None:
    """
    Run pytorch-lightning model inference
    Args:
        cfg: hydra config
    Returns:
        None
    """
    device = 'cpu'# torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_path = Path(cfg.inference.audio_path)
    save_path.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.training.seed) 
    
    
    model_names = glob.glob(f'outputs/{cfg.inference.run_name}/saved_models/*') # TODO later we pick the best
    print(f'### Found these models: {model_names}, will use {model_names[1]}')
    dataloader = load_obj(cfg.datamodule.datamodule_name)(cfg=cfg).setup(inference=True).test_dataloader() if not cfg.datamodule.params.unconditional else None

    print(f"### Loaded the dataloader: {dataloader}")
    
    lit_model = load_obj(cfg.training.lightning_module_name).load_from_checkpoint(checkpoint_path=model_names[1], cfg=cfg)
    lit_model.to(device)
    lit_model.eval()
    print(f"### Loaded the model")
    print(f'### Starting the synthesis, unconditional is {cfg.datamodule.params.unconditional}')

    with torch.no_grad():
        if cfg.datamodule.params.unconditional:
            for i in tqdm(range(cfg.inference.n_audiofiles)):
                input = {'spectrogram':None, 'lang':None}
                audio = lit_model(input)
                path = save_path / f"audio_{i}.wav" 
                T.save(path, audio, cfg.datamodule.preprocessing.sample_rate)
        else:
            for i, batch in tqdm(enumerate(dataloader)):
                audio = lit_model(batch)
                filename = Path(batch['filename']).stem
                path = save_path / f"synthesized_{filename}.wav"
                T.save(path, audio, cfg.datamodule.preprocessing.sample_rate)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Synthesize Audio with DiffWave')
    parser.add_argument('--run_name', help='folder_name', type=str, default='2022-11-24_18-59-52')
    
    args = parser.parse_args()
    
    initialize(config_path='../configs')
    
    inference_cfg = compose(config_name='config')
    
    inference_cfg['inference']['run_name'] = args.run_name
    
    print(inference_cfg.inference.run_name)
    
    path = f'outputs/{inference_cfg.inference.run_name}/.hydra/config.yaml'
    
    with open(path) as cfg:
        cfg_yaml = yaml.safe_load(cfg)
    cfg_yaml['inference'] = inference_cfg['inference']
    
    cfg = OmegaConf.create(cfg_yaml)
    
    synthesize_audio(cfg)