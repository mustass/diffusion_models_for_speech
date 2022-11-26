import argparse
import glob
from hydra import experimental
import torch
import torchaudio as T
import yaml
from omegaconf import DictConfig, OmegaConf
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
    set_seed(cfg.training.seed) 
    model_name = glob.glob(f'outputs/{cfg.inference.run_name}/saved_models/*')[0] # TODO later we pick the best
    
    dataloader = load_obj(cfg.datamodule.datamodule_name).test_dataloader() if not cfg.datamodule.params.unconditional else None

    device = cfg.data.device
    
    lit_model = load_obj(cfg.training.lightning_module_name).load_from_checkpoint(checkpoint_path=model_name, cfg=cfg).to(device)
    lit_model.to(device)
    lit_model.eval()
    with torch.no_grad():
        if cfg.datamodule.params.unconditional:
            for i in range(cfg.inference.n_audiofiles):
                input = {'spectrogram':None, 'lang':None}
                audio = lit_model(input)
                path = f"audio_{i}.wav" 
                T.save(path, audio, cfg.datamodule.preprocessing.sample_rate)
        else:
            for i, batch in enumerate(dataloader):
                audio = lit_model(batch)
                path = f"audio_{i}.wav" #TODO could be cool if the dataloader actually returned the filename so we could save the sytnthesized with the same name. otherwise one could just have shuffle=False and then we know the order.
                T.save(path, audio, cfg.datamodule.preprocessing.sample_rate)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Synthesize Audio with DiffWave')
    parser.add_argument('--run_name', help='folder_name', type=str, default='2022-11-24_18-59-52')
    
    args = parser.parse_args()
    
    experimental.initialize(config_dir='conf', strict=True)
    
    inference_cfg = experimental.compose(config_file='config.yaml')
    
    inference_cfg['inference']['run_name'] = args.run_name
    
    inference_cfg['inference']['mode'] = args.mode
    
    print(inference_cfg.inference.run_name)
    
    path = f'outputs/{inference_cfg.inference.run_name}/.hydra/config.yaml'
    
    with open(path) as cfg:
        cfg_yaml = yaml.safe_load(cfg)
    cfg_yaml['inference'] = inference_cfg['inference']
    
    cfg = OmegaConf.create(cfg_yaml)
    
    synthesize_audio(cfg)