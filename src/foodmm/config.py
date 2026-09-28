"""Load YAML configs and apply dotted `--set key=value` overrides."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def parse_value(raw: str) -> Any:
    """Parse an override value with YAML rules; strings like '1e-4' become floats."""
    value = yaml.safe_load(raw)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def set_by_path(cfg: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = cfg
    for key in keys[:-1]:
        if not isinstance(node.get(key), dict):
            node[key] = {}
        node = node[key]
    node[keys[-1]] = value


def load_config(path: str | Path = DEFAULT_CONFIG, overrides: list[str] | None = None) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got {item!r}")
        key, raw = item.split("=", 1)
        set_by_path(cfg, key.strip(), parse_value(raw))
    return cfg


def save_config(cfg: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def work_paths(cfg: dict) -> dict[str, Path]:
    work = Path(cfg["paths"]["work_dir"])
    data = work / "data"
    return {
        "work_dir": work,
        "data_dir": data,
        "manifest": data / "manifest.csv",
        "classes": data / "classes.json",
        "stats": data / "stats.json",
        "runs": work / "runs",
        "results": work / "results",
    }


def add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML config file")
    parser.add_argument("--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE",
                        help="Config overrides, e.g. data.text_mask=strict (must be the last option)")


def config_from_args(args: argparse.Namespace) -> dict:
    return load_config(args.config, args.overrides)
