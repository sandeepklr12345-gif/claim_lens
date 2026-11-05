"""Case state as JSON files, the one storage module (SPEC.md 12.1, 13.5).

The default folder is runs/cases/ (git-ignored); CLAIMLENS_STATE_DIR overrides it. A write goes
to a temporary file and is then renamed, so a reader never sees half a record.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from claimlens.schemas import CaseRecord
from claimlens.utils.config import ROOT

_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def state_dir() -> Path:
    path = Path(os.environ.get("CLAIMLENS_STATE_DIR") or ROOT / "runs" / "cases")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _path(case_id: str) -> Path:
    if not _ID.match(case_id):  # case ids reach here from URLs; never build a path from others
        raise ValueError(f"invalid case id: {case_id!r}")
    return state_dir() / f"{case_id}.json"


def save_case(record: CaseRecord) -> None:
    path = _path(record.case_id)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(record.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(path)


def load_case(case_id: str) -> CaseRecord | None:
    """The stored record, or None if there is none (including for a malformed id)."""
    try:
        path = _path(case_id)
    except ValueError:
        return None
    if not path.is_file():
        return None
    return CaseRecord.model_validate_json(path.read_text(encoding="utf-8"))
