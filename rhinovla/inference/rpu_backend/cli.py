"""Command line entry point for single-frame RPU backend inference."""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image

from rhinovla.inference.rpu_backend.executor import RhinoVLARPUBackendExecutor


def parse_slots(spec: str | None) -> Optional[list[int]]:
    if not spec:
        return None
    out: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            out.extend(range(int(start), int(end) + 1))
        else:
            out.append(int(part))
    return out


def parse_csv(spec: str | None) -> Optional[list[str]]:
    if spec is None:
        return None
    out = [x.strip() for x in str(spec).split(",") if x.strip()]
    return out or None


def merge_rpu_env(
    env_file: str | Path | None,
    overrides: Sequence[str] | None,
    *,
    train_config: str | Path | None = None,
    checkpoint: str | Path | None = None,
) -> dict[str, str]:
    env: dict[str, str] = {}
    if env_file:
        data = tomllib.loads(Path(env_file).read_text(encoding="utf-8"))
        raw_env = data.get("env", {})
        if not isinstance(raw_env, dict):
            raise ValueError(f"{env_file} [env] must be a table")
        env.update({str(k): str(v) for k, v in raw_env.items()})
    for item in overrides or []:
        if "=" not in item:
            raise ValueError("--rpu-env must be KEY=VALUE")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("--rpu-env must be KEY=VALUE")
        env[key] = value
    if train_config is not None:
        env.setdefault("RHINOVLA_CONFIG", str(train_config))
    if checkpoint is not None:
        env.setdefault("RHINOVLA_CKPT", str(checkpoint))
    return env


def parse_state(value: str) -> np.ndarray:
    stripped = value.strip()
    if stripped.startswith("["):
        raw = json.loads(value)
    else:
        path = Path(value)
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
        else:
            raw = [float(x.strip()) for x in value.split(",") if x.strip()]
    return np.asarray(raw, dtype=np.float32).reshape(-1)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-artifact", required=True, help="RPU prepare artifact .pt")
    parser.add_argument("--config", "--train-config", dest="train_config", required=True, help="training config YAML")
    parser.add_argument("--checkpoint", required=True, help="trained RhinoVLA checkpoint")
    parser.add_argument("--norm-stats", required=True, help="matching norm.json or checkpoint norm sidecar")
    parser.add_argument("--mapping", default=None, help="native72 mapping YAML for native-dim norm stats")
    parser.add_argument("--mapping-dataset-id", default=None)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--image", action="append", required=True, help="image path, repeat in view order")
    parser.add_argument("--state", required=True, help="raw state JSON/list string or path to JSON array")
    parser.add_argument("--output", default=None, help="output JSON path")
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--action-hz", type=float, default=30.0)
    parser.add_argument("--active-slots", default=None)
    parser.add_argument("--view-roles", default="top_head,hand_left,hand_right")
    parser.add_argument("--view-modalities", default="rgb,rgb,rgb")
    parser.add_argument("--rpu-rhino-repo", default=None)
    parser.add_argument("--rpu-env-file", default=None)
    parser.add_argument("--rpu-env", action="append", default=None, help="RPU runtime env override, KEY=VALUE")
    parser.add_argument("--rpu-artifact-strict", action="store_true")
    parser.add_argument("--noise-seed", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        runtime_env = merge_rpu_env(
            args.rpu_env_file,
            args.rpu_env,
            train_config=args.train_config,
            checkpoint=args.checkpoint,
        )
    except ValueError as exc:
        parser.error(str(exc))

    images = [Image.open(path).convert("RGB") for path in args.image]
    executor = RhinoVLARPUBackendExecutor(
        prepare_artifact=args.prepare_artifact,
        train_config=args.train_config,
        checkpoint=args.checkpoint,
        norm_stats_path=args.norm_stats,
        mapping_path=args.mapping,
        mapping_dataset_id=args.mapping_dataset_id,
        instruction=args.instruction,
        num_steps=args.num_steps,
        action_hz=args.action_hz,
        active_slots=parse_slots(args.active_slots),
        view_roles=parse_csv(args.view_roles),
        view_modalities=parse_csv(args.view_modalities),
        rhino_repo=args.rpu_rhino_repo,
        runtime_env=runtime_env,
        artifact_strict=bool(args.rpu_artifact_strict),
        noise_seed=int(args.noise_seed),
    )
    try:
        result = executor.infer(images, parse_state(args.state))
    finally:
        executor.close()

    payload = {
        "actions_raw": result.actions_raw.tolist(),
        "actions_norm": result.actions_norm.tolist(),
        "action_hz": result.action_hz,
        "latency_ms": result.latency_ms,
        "extra": result.extra,
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        sys.stdout.write(text + "\n")


if __name__ == "__main__":
    main()
