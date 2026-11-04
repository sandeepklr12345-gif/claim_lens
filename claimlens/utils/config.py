"""Load configs/app.yaml, with OLLAMA_HOST / OLLAMA_MODEL overridable from the environment."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "app.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)
    if os.environ.get("OLLAMA_HOST"):
        cfg["llm"]["host"] = os.environ["OLLAMA_HOST"]
    if os.environ.get("OLLAMA_MODEL"):
        cfg["llm"]["model"] = os.environ["OLLAMA_MODEL"]
    return cfg
