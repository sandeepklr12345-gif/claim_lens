"""Shared helpers for the evaluation scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from claimlens.schemas import Case
from claimlens.utils.config import ROOT

RESULTS = ROOT / "eval" / "results"


def load_cases() -> list[Case]:
    files = sorted((ROOT / "data" / "cases").glob("case_*.json"))
    return [Case.model_validate_json(f.read_text(encoding="utf-8")) for f in files]


def write_json(name: str, obj: Any) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / name
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def read_json(name: str) -> Any:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))
