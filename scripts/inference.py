import argparse
import glob
from pathlib import Path

import torch
import torchaudio as T
import yaml
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from diffspeak.utils.technical_utils import load_obj
from diffspeak.utils.utils import set_seed


def synthesize_audio(cfg: DictConfig) -> None:
    """
    Run pytorch-lightning model inference
    Args:
        cfg: hydra config
    Returns:
        None
    """
    device = "cpu"  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_path = Path(cfg.inference.audio_path)
    save_path.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.training.seed)

    cfg.model.params.hop_samples = 256 # SUPER DIRTY

    if not args.run_name == 'pretrained_model':
        model_names = glob.glob(
            f"outputs/{cfg.inference.run_name}/saved_models/*"
        )  # TODO later we pick the best
        print(f"### Found these models: {model_names}, will use {model_names[1]}")
        model_name = model_names[1]
    else:
        model_name = cfg.inference.model_path

    dataloader = None
    
    if not cfg.datamodule.params.unconditional:
        dataloader = load_obj(cfg.datamodule.datamodule_name)(cfg=cfg)
        dataloader.setup(inference=True)
        dataloader= dataloader.test_dataloader()
    
    print(f"### Loaded the dataloader: {dataloader}")

    lit_model = load_obj(cfg.training.lightning_module_name).load_from_checkpoint(
        checkpoint_path=model_name, cfg=cfg
    )
    lit_model.to(device)
    lit_model.eval()
    print(f"### Loaded the model")
    print(
        f"### Starting the synthesis, unconditional is {cfg.datamodule.params.unconditional}"
    )

    with torch.no_grad():
        if cfg.datamodule.params.unconditional:
            for i in tqdm(range(cfg.inference.n_audiofiles)):
                input = {"spectrogram": None, "lang": None}
                audio = lit_model(input)
                path = save_path / f"audio_{i}.wav"
                T.save(path, audio, cfg.datamodule.preprocessing.sample_rate)
        else:
            for i, batch in tqdm(enumerate(dataloader)):
                print(batch.keys())
                audio = lit_model(batch)
                filename = 'sound' #Path(batch["filename"]).stem # This does not work because Collator does not pass it on. 
                path = save_path / f"synthesized_{filename}.wav"
                T.save(path, audio, cfg.datamodule.preprocessing.sample_rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthesize Audio with DiffWave")
    parser.add_argument(
        "--run_name", help="folder_name", type=str, default="2022-11-24_18-59-52"
    )
    parser.add_argument(
        "--meta_data", help="meta_data_folder", type=str, default="./data"
    )

    args = parser.parse_args()

    hydra.initialize(config_path="../configs")

    cfg = hydra.compose(config_name="config_pretrained")

    cfg["inference"]["run_name"] = args.run_name
    cfg.datamodule.path_to_metadata = args.meta_data

    print(cfg.inference.run_name)

    if not args.run_name == 'pretrained_model':
        path = f"outputs/{cfg.inference.run_name}/.hydra/config.yaml"

        with open(path) as cfg_load:
            cfg_yaml = yaml.safe_load(cfg_load)
        cfg_yaml["inference"] = cfg["inference"]
        cfg = OmegaConf.create(cfg_yaml)

    synthesize_audio(cfg)
