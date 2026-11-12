"""ClaimLens HTTP API (SPEC.md 13.5). A thin wrapper around the agent and the storage module.

    uvicorn api.main:app --reload

POST /cases                create a case from letter text and a policy id
POST /cases/{case_id}/run  run the agent; blocks until the audit finishes
GET  /cases/{case_id}      read a case and its result
GET  /cases/{case_id}/appeal  the structured appeal draft (R2)
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, status

from claimlens.agent.graph import run_case
from claimlens.agent.tools import DirectTools
from claimlens.llm.client import OllamaClient
from claimlens.rag.parse import list_policies
from claimlens.schemas import (
    AppealDraft,
    Case,
    CaseCreate,
    CaseRecord,
    CaseResult,
    CaseStatus,
    Verdict,
)
from claimlens.utils.config import load_config
from claimlens.utils.storage import load_case, save_case

log = logging.getLogger(__name__)
Runner = Callable[[Case], CaseResult]
app = FastAPI(title="ClaimLens", description="Decision support, not legal advice.")
_claim_lock = threading.Lock()  # makes the created -> running check atomic


@lru_cache(maxsize=1)
def _agent() -> tuple[DirectTools, OllamaClient, dict]:
    return DirectTools(), OllamaClient(), load_config()


def get_runner() -> Runner:
    """The real agent; tests replace this dependency so no LLM or model is needed."""
    tools, llm, cfg = _agent()
    llm.health()  # fail before retrieval if Ollama is down
    return lambda case: run_case(case, tools, llm, cfg)


def get_policies() -> list[str]:
    return list_policies()


def _found(case_id: str) -> CaseRecord:
    record = load_case(case_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no case {case_id}")
    return record


@app.post("/cases", status_code=status.HTTP_201_CREATED)
def create_case(body: CaseCreate,
                policies: Annotated[list[str], Depends(get_policies)]) -> CaseRecord:
    if body.policy_id not in policies:
        raise HTTPException(422, f"unknown policy_id; available: {policies}")
    record = CaseRecord(case_id=uuid.uuid4().hex, policy_id=body.policy_id,
                        letter_text=body.letter_text)
    save_case(record)
    return record


@app.post("/cases/{case_id}/run")
def run(case_id: str, runner: Annotated[Runner, Depends(get_runner)]) -> CaseRecord:
    with _claim_lock:
        record = _found(case_id)
        if record.status == CaseStatus.RUNNING:
            raise HTTPException(status.HTTP_409_CONFLICT, "this case is already running")
        record.status, record.error = CaseStatus.RUNNING, None
        save_case(record)
    case = Case(case_id=case_id, policy_id=record.policy_id, letter_text=record.letter_text,
                reason="", gold_clause_ids=[], gold_verdict=Verdict.INSUFFICIENT_EVIDENCE,
                source="user")
    try:
        record.result = runner(case)
        record.status = CaseStatus.DONE
    except Exception as e:  # noqa: BLE001 - a failed audit is stored, not left as "running"
        log.exception("audit failed for %s", case_id)
        record.status, record.error = CaseStatus.FAILED, f"{type(e).__name__}: {e}"
    save_case(record)
    return record


@app.get("/cases/{case_id}")
def get_case(case_id: str) -> CaseRecord:
    return _found(case_id)


@app.get("/cases/{case_id}/appeal")
def get_appeal(case_id: str) -> AppealDraft:
    """The structured appeal (or clarification request) for a finished case."""
    record = _found(case_id)
    if record.result is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "this case has no result yet; run it first")
    if record.result.appeal is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND,
                            "no appeal for this case (drafted only for NOT_SUPPORTED and "
                            "INSUFFICIENT_EVIDENCE verdicts)")
    return record.result.appeal
