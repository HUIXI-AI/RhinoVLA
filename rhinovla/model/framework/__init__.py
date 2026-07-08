from rhinovla.model.framework.RhinoVLA import RhinoVLA


def build_framework(cfg):
    """Build the single supported RhinoVLA framework."""
    if not hasattr(cfg.framework, "name"):
        cfg.framework.name = "RhinoVLA"
    if cfg.framework.name != "RhinoVLA":
        raise NotImplementedError(f"Only framework.name=RhinoVLA is supported, got {cfg.framework.name!r}")
    return RhinoVLA(cfg)


__all__ = ["RhinoVLA", "build_framework"]
