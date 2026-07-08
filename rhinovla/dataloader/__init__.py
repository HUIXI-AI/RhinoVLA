"""Dataloader registry for the native72 training path."""


def _build_lerobot_native72(cfg, dataset_py: str, split: str, **kwargs):
    from rhinovla.dataloader.lerobot_native72 import build_dataloader as _build

    return _build(cfg=cfg, dataset_py=dataset_py, split=split, **kwargs)


DATASET_BUILDERS = {
    "lerobot_native72": _build_lerobot_native72,
}


def build_dataloader(cfg, dataset_py: str = "lerobot_native72", split: str = "train", **kwargs):
    if dataset_py in DATASET_BUILDERS:
        return DATASET_BUILDERS[dataset_py](cfg=cfg, dataset_py=dataset_py, split=split, **kwargs)
    raise ValueError(f"Unsupported dataset_py: {dataset_py}")
