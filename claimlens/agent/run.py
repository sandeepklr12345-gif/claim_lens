"""CLI: python -m claimlens.agent.run --case data/cases/case_001.json [--mcp]"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from claimlens.agent.graph import run_case
from claimlens.agent.tools import DirectTools, McpTools, Tools
from claimlens.llm.client import OllamaClient
from claimlens.schemas import Case, CaseResult
from claimlens.utils.config import ROOT, load_config


def print_result(case: Case, res: CaseResult) -> None:
    if not res.top_clauses:
        print(f"\n{case.case_id}: no clause retrieved -> {res.verdict}")
        return
    best = res.top_clauses[0].chunk
    print(f"\n=== {case.case_id} | policy {case.policy_id} | gold verdict {case.gold_verdict} ===")
    print(f"Reason (extracted): {res.reason}")
    print(f"VERDICT: {res.verdict}   (match prob {res.match.prob:.3f})" if res.match else "")
    print(f"Best clause: {best.chunk_id}  p.{best.page}  {best.section_path}")
    print(f"  {best.text[:300]}...")
    for label, expl, aas in (("INITIAL", res.explanation_initial, res.aas_initial),
                             ("FINAL", res.explanation_final, res.aas_final)):
        if expl is None or aas is None:
            continue
        print(f"\n{label} explanation (AAS {aas:.3f}):\n  {expl.text}")
        for s in expl.cited_spans:
            print(f"    cited: \"{s.text}\"")
    if res.appeal_draft:
        print(f"\nAppeal draft:\n{res.appeal_draft}")
    print("\nDecision support, not legal advice.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, type=Path)
    ap.add_argument("--mcp", action="store_true", help="call tools through the MCP servers")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    case = Case.model_validate_json(args.case.read_text(encoding="utf-8"))
    cfg = load_config()
    tools: Tools = McpTools() if args.mcp else DirectTools()
    try:
        res = run_case(case, tools, OllamaClient(), cfg)
    finally:
        tools.close()  # type: ignore[attr-defined]
    print_result(case, res)
    out = ROOT / "runs" / f"{case.case_id}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res.model_dump(mode="json"), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
