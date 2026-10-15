"""Dataloader registry for the V3 native96 training path."""


def _build_lerobot_native96(cfg, dataset_py: str, split: str, **kwargs):
    from rhinovla.dataloader.lerobot_native96 import build_dataloader as _build

    return _build(cfg=cfg, dataset_py=dataset_py, split=split, **kwargs)


DATASET_BUILDERS = {
    "lerobot_native96": _build_lerobot_native96,
}


def build_dataloader(cfg, dataset_py: str = "lerobot_native96", split: str = "train", **kwargs):
    if dataset_py in DATASET_BUILDERS:
        return DATASET_BUILDERS[dataset_py](cfg=cfg, dataset_py=dataset_py, split=split, **kwargs)
    raise ValueError(f"Unsupported dataset_py: {dataset_py}")
