import os
from pathlib import Path

from hydra.utils import get_original_cwd


def sanity_check(cfg):
    """
    Check config for training.
    Do not use this when preprocessing!
    """

    assert not (
        cfg.datamodule.params.remove_shorts is False
        and cfg.datamodule.params.collator == "diffspeak.datasets.collator.Collator"
    ), "The default Collator can not handle too short audio.\nSet remove_shorts = True or use another Collator!"

    assert not (
        cfg.datamodule.params.remove_shorts is True
        and cfg.datamodule.params.collator
        == "diffspeak.datasets.collator.ZeroPadCollator"
    ), "Handling too short audio in the collator is not necessary when remove_shorts = True"
    assert (Path(cfg.datamodule.path_to_metadata) / "annotations.csv").exists()
