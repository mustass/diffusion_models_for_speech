def sanity_check(cfg):
    assert not (
        cfg.datamodule.params.remove_shorts is False
        and cfg.datamodule.params.collator == "diffspeak.datasets.collator.Collator"
    ), "The default Collator can not handle too short audio.\nSet remove_shorts = True or use another Collator!"

    assert not (
        cfg.datamodule.params.remove_shorts is True
        and cfg.datamodule.params.collator == "diffspeak.datasets.collator.ZeroPadCollator"
    ), "Handling too short audio in the collator is not necessary when remove_shorts = True"
