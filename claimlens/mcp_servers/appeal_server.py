"""MCP server `appeal` (stdio). Run: python -m claimlens.mcp_servers.appeal_server

Thin wrapper over claimlens.appeal.tools. No model is loaded. The deadline rules are unverified.
"""

from __future__ import annotations

import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

from claimlens.appeal import tools

mcp = FastMCP("appeal")


@mcp.tool()
def list_templates() -> dict[str, Any]:
    """List the appeal templates (name and description)."""
    return tools.list_templates()


@mcp.tool()
def get_template(name: str) -> dict[str, Any]:
    """Return one template's text and its [placeholders]."""
    return tools.get_template(name)


@mcp.tool()
def compute_deadlines(rejection_date: str, grievance_filed_date: str | None = None,
                      insurer_reply_date: str | None = None) -> dict[str, Any]:
    """Earliest/latest Ombudsman complaint dates (ISO dates). Rules are NOT verified."""
    return tools.compute_deadlines(rejection_date, grievance_filed_date, insurer_reply_date)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)  # stderr; stdout carries the MCP protocol
    mcp.run()
