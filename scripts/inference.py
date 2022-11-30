import argparse
import glob
import os
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
import torchaudio as T
import yaml
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
    set_seed(cfg.training.seed)

    save_path = Path(cfg.inference.audio_path)
    save_path.mkdir(parents=True, exist_ok=True)

    dataloader = load_dataloader(cfg)
    lit_model = load_model(cfg)

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
            run_name = os.path.basename(os.getcwd())
            loggers = []
            if cfg.logging.log:
                for logger in cfg.logging.loggers:
                    if "experiment_name" in logger.params.keys():
                        logger.params["experiment_name"] = run_name
                    loggers.append(load_obj(logger.class_name)(**logger.params))
            trainer = pl.Trainer(
                logger=loggers,
                callbacks=[],
                **cfg.trainer,
            )
            trainer.predict(lit_model, dataloaders=[dataloader])


def load_dataloader(cfg):
    dataloader = None

    if not cfg.datamodule.params.unconditional:
        dataloader = load_obj(cfg.datamodule.datamodule_name)(cfg=cfg)
        dataloader.setup(inference=True)
        dataloader = dataloader.inference_dataloader()

    print(f"### Loaded the dataloader: {dataloader}")
    return dataloader


def load_model(cfg):
    if not cfg.inference.run_name == "pretrained_model":
        model_names = glob.glob(
            f"outputs/{cfg.inference.run_name}/saved_models/*"
        )  # TODO later we pick the best
        print(f"### Found these models: {model_names}, will use {model_names[1]}")
        model_name = model_names[1]
    else:
        model_name = cfg.inference.model_path
    lit_model = load_obj(cfg.training.lightning_module_name).load_from_checkpoint(
        checkpoint_path=model_name, cfg=cfg
    )
    print(f"### Loaded the model")

    return lit_model


@hydra.main(config_path="../configs", config_name="config_synthesis")
def main():
    # If we want to use a custom trained / finetuned model
    # Then copy all of the contents of the config file of that training session
    # And append the inference related contents of this current config file to that
    if not cfg.inference.run_name == "pretrained_model":
        path = f"outputs/{cfg.inference.run_name}/.hydra/config.yaml"
        with open(path) as cfg_load:
            cfg_yaml = yaml.safe_load(cfg_load)
        cfg_yaml["inference"] = cfg["inference"]
        cfg_yaml["datamodule"] = cfg["datamodule"]

        cfg = OmegaConf.create(cfg_yaml)

    # SUPER DIRTY
    cfg.model.params.hop_samples = 256  

    print(cfg.inference.run_name)

    if cfg.inference.device == "gpu":
        cfg.trainer.accelerator = "gpu"
        cfg.trainer.gpus = 1

    synthesize_audio(cfg)

if __name__ == "__main__":
   main()
