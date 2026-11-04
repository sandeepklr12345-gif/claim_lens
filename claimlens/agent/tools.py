"""Tool access for the agent: one interface, two implementations (SPEC.md 7.7).

`McpTools` calls the two MCP servers over stdio. `DirectTools` calls the same core
functions in-process (the SPEC.md 7.7 fallback, also used by tests and the Streamlit UI).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from typing import Any, Protocol

from claimlens import core
from claimlens.schemas import (
    AlignmentResult,
    Attribution,
    Chunk,
    CitedSpan,
    MatchResult,
    RetrievedClause,
)

POLICY_MODULE = "claimlens.mcp_servers.policy_server"
EXPLAINER_MODULE = "claimlens.mcp_servers.explainer_server"


class Tools(Protocol):
    def search_clauses(self, query: str, doc_id: str, top_k: int = 5) -> list[RetrievedClause]: ...
    def get_clause(self, doc_id: str, clause_id: str) -> Chunk: ...
    def score_pair(self, reason: str, clause: str, clause_id: str = "") -> MatchResult: ...
    def attribute_ig(self, reason: str, clause: str, clause_id: str = "") -> Attribution: ...
    def alignment_score(
        self, reason: str, clause: str, cited_spans: list[CitedSpan],
        attribution: Attribution | None = None,
    ) -> AlignmentResult: ...


class DirectTools:
    """In-process calls to claimlens.core (no MCP)."""

    def search_clauses(self, query: str, doc_id: str, top_k: int = 5) -> list[RetrievedClause]:
        return core.search_clauses(query, doc_id, top_k)

    def get_clause(self, doc_id: str, clause_id: str) -> Chunk:
        return core.get_clause(doc_id, clause_id)

    def score_pair(self, reason: str, clause: str, clause_id: str = "") -> MatchResult:
        return core.score_pair(reason, clause, clause_id)

    def attribute_ig(self, reason: str, clause: str, clause_id: str = "") -> Attribution:
        return core.attribute_ig(reason, clause, clause_id)

    def alignment_score(
        self, reason: str, clause: str, cited_spans: list[CitedSpan],
        attribution: Attribution | None = None,
    ) -> AlignmentResult:
        return core.alignment_score(reason, clause, cited_spans, attribution)

    def close(self) -> None:
        pass


class McpTools:
    """Calls both MCP servers over stdio. Sessions live in one background asyncio task."""

    def __init__(self, startup_timeout: float = 600.0) -> None:
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True).start()
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._sessions: dict[str, Any] = {}
        self._stop: asyncio.Event | None = None
        self._task = asyncio.run_coroutine_threadsafe(self._main(), self._loop)
        if not self._ready.wait(startup_timeout) or self._error is not None:
            raise RuntimeError(f"MCP servers failed to start: {self._error!r}")

    async def _main(self) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        self._stop = asyncio.Event()
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        try:
            async with (
                stdio_client(StdioServerParameters(
                    command=sys.executable, args=["-m", POLICY_MODULE], env=env)) as (r1, w1),
                stdio_client(StdioServerParameters(
                    command=sys.executable, args=["-m", EXPLAINER_MODULE], env=env)) as (r2, w2),
                ClientSession(r1, w1) as policy,
                ClientSession(r2, w2) as explainer,
            ):
                await policy.initialize()
                await explainer.initialize()
                self._sessions = {"policy": policy, "explainer": explainer}
                self._ready.set()
                await self._stop.wait()
        except BaseException as e:  # noqa: BLE001 - surface startup failure to __init__
            self._error = e
            self._ready.set()

    def _call(self, server: str, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        fut = asyncio.run_coroutine_threadsafe(
            self._sessions[server].call_tool(tool, args), self._loop)
        res = fut.result(timeout=600)
        text = res.content[0].text  # type: ignore[union-attr]
        if res.isError:
            raise RuntimeError(f"MCP tool {tool} failed: {text}")
        out: dict[str, Any] = json.loads(text)
        return out

    def search_clauses(self, query: str, doc_id: str, top_k: int = 5) -> list[RetrievedClause]:
        out = self._call("policy", "search_clauses",
                         {"query": query, "doc_id": doc_id, "top_k": top_k})
        return [RetrievedClause.model_validate(x) for x in out["results"]]

    def get_clause(self, doc_id: str, clause_id: str) -> Chunk:
        return Chunk.model_validate(
            self._call("policy", "get_clause", {"doc_id": doc_id, "clause_id": clause_id}))

    def score_pair(self, reason: str, clause: str, clause_id: str = "") -> MatchResult:
        return MatchResult.model_validate(self._call(
            "explainer", "score_pair",
            {"reason": reason, "clause": clause, "clause_id": clause_id}))

    def attribute_ig(self, reason: str, clause: str, clause_id: str = "") -> Attribution:
        return Attribution.model_validate(self._call(
            "explainer", "attribute_ig",
            {"reason": reason, "clause": clause, "clause_id": clause_id}))

    def alignment_score(
        self, reason: str, clause: str, cited_spans: list[CitedSpan],
        attribution: Attribution | None = None,
    ) -> AlignmentResult:
        args: dict[str, Any] = {
            "reason": reason, "clause": clause,
            "cited_spans": [s.model_dump(mode="json") for s in cited_spans]}
        if attribution is not None:
            args["attribution"] = attribution.model_dump(mode="json")
        return AlignmentResult.model_validate(self._call("explainer", "alignment_score", args))

    def close(self) -> None:
        if self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
            try:
                self._task.result(timeout=30)
            except Exception:  # noqa: BLE001 - best-effort shutdown
                pass
        self._loop.call_soon_threadsafe(self._loop.stop)
