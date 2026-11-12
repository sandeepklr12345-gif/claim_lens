"""Plain functions behind the appeal MCP server (no `mcp` import, so CI can test them)."""

from __future__ import annotations

import re
from typing import Any

from claimlens.appeal.deadlines import compute_deadlines as _compute_deadlines
from claimlens.appeal.templates import TEMPLATES


def list_templates() -> dict[str, Any]:
    return {"templates": [{"name": n, "description": t["description"]}
                          for n, t in TEMPLATES.items()]}


def get_template(name: str) -> dict[str, Any]:
    if name not in TEMPLATES:
        raise ValueError(f"unknown template {name!r}; available: {sorted(TEMPLATES)}")
    text = TEMPLATES[name]["text"]
    return {"name": name, "description": TEMPLATES[name]["description"], "text": text,
            "placeholders": sorted(set(re.findall(r"\[[^\]]{2,60}\]", text)))}


def compute_deadlines(rejection_date: str, grievance_filed_date: str | None = None,
                      insurer_reply_date: str | None = None) -> dict[str, Any]:
    return _compute_deadlines(rejection_date, grievance_filed_date,
                              insurer_reply_date).model_dump(mode="json")
