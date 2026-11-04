"""Intake screen and live run progress for the ClaimLens Streamlit app.

Pure helpers (no Streamlit or torch import) so CI can test them. All text is escaped with the
same `esc` as the result view: pasted letters and LLM output end up in these strings.
Every figure in the progress panel is measured or comes from the run; nothing is estimated.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.result_view import DISCLAIMER, esc, verdict_info

STEPS: list[tuple[str, str]] = [
    ("Read the letter", "Pull out the stated reason for rejection."),
    ("Find clauses", "Keyword and meaning search over the policy."),
    ("Match", "Score how well the best clause supports the reason."),
    ("Explain and check", "Write an explanation, then test it against the evidence."),
    ("Verdict", "Supported, not supported, or not enough evidence."),
]
PREPARE = "Prepare"

CSS = """<style>
.stApp{font-family:'IBM Plex Sans',system-ui,sans-serif}
.block-container{max-width:1200px}
.cl-head{display:flex;flex-wrap:wrap;gap:12px;justify-content:space-between;align-items:center;margin-bottom:24px}
.cl-mark{display:flex;align-items:center;gap:10px;font-family:'Source Serif 4',Georgia,serif;font-size:24px;font-weight:600;color:#14201F}
.cl-pill{font-size:13px;padding:6px 14px;border-radius:999px;background:#fff;border:1px solid #D9DFDA;color:#3F4B48}
.cl-hero{font-family:'Source Serif 4',Georgia,serif !important;font-size:44px;line-height:1.1;font-weight:600;color:#14201F;margin:0 0 12px}
.cl-lead{font-size:18px;line-height:1.55;color:#55615E;max-width:760px;margin:0 0 28px}
.cl-sub{display:flex;align-items:center;gap:10px;font-size:15px;font-weight:600;color:#14201F;margin-bottom:10px}
.cl-row{padding:12px 14px;border-radius:12px;border:1px solid #D9DFDA;background:#fff}
.cl-row.on{border-color:#0F766E;background:#E8F3F1}
.cl-id{font-family:'IBM Plex Mono',monospace;font-size:13px;color:#55615E}
.cl-title{font-size:16px;font-weight:500;color:#14201F;line-height:1.35;margin:2px 0}
.cl-meta{font-size:14px;color:#55615E}
.cl-note{font-size:13px;line-height:1.5;color:#55615E;margin-top:8px}
.cl-steps{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;margin-top:8px}
.cl-step b{display:block;font-size:15px;font-weight:600;color:#14201F;margin:2px 0}
.cl-step span{font-size:14px;line-height:1.45;color:#55615E}
.cl-step i{font-family:'IBM Plex Mono',monospace;font-style:normal;font-size:13px;color:#0F766E}
.cl-prog{display:flex;gap:12px;padding:10px 0;border-top:1px solid #E6EAE5}
.cl-dot{flex:none;width:12px;height:12px;margin-top:4px;border-radius:50%;border:2px solid #B9C2BC}
.cl-dot.active{border-color:#0F766E;box-shadow:0 0 0 3px #BFDDD8}
.cl-dot.done{border-color:#0F766E;background:#0F766E}
.cl-dot.stopped{border-color:#B42318;background:#B42318}
.cl-pname{font-size:15px;font-weight:500;color:#14201F}
.cl-pname.pending{color:#55615E;font-weight:400}
.cl-pdetail{font-size:14px;line-height:1.45;color:#3F4B48;overflow-wrap:anywhere}
.cl-ptime{margin-left:auto;font-family:'IBM Plex Mono',monospace;font-size:13px;color:#55615E;white-space:nowrap}
.st-key-intake_card{background:#fff;border:1px solid #D9DFDA;border-radius:14px;padding:24px}
</style>"""

MARK_SVG = (
    '<svg width="26" height="26" viewBox="0 0 26 26" fill="none" stroke="#0F766E" '
    'stroke-width="2.5" stroke-linecap="round" aria-hidden="true">'
    '<circle cx="11" cy="11" r="8"/><path d="M17 17l6 6"/></svg>'
)


def short_title(reason: str, max_words: int = 8) -> str:
    """A short title taken from the case's own reason text."""
    words = reason.split()
    title = " ".join(words[:max_words]).rstrip(" .,;:")
    return title + "..." if len(words) > max_words else title


def header_html() -> str:
    return (f'<div class="cl-head"><div class="cl-mark">{MARK_SVG}ClaimLens</div>'
            f'<div class="cl-pill">{esc(DISCLAIMER)}</div></div>')


def hero_html() -> str:
    return ('<h1 class="cl-hero">Was this claim rejection justified?</h1>'
            '<p class="cl-lead">Pick a sample rejection letter or paste your own, choose the '
            'policy, and ClaimLens finds the clause behind the decision and shows which words '
            'carried it.</p>')


def case_row_html(case_id: str, reason: str, policy_id: str, source: str, selected: bool) -> str:
    chip = '<span class="cl-chip">Synthetic data</span>' if source == "synthetic" else ""
    cls = "cl-row on" if selected else "cl-row"
    return (f'<div class="{cls}"><span class="cl-id">{esc(case_id)}</span> {chip}'
            f'<div class="cl-title">{esc(short_title(reason))}</div>'
            f'<div class="cl-meta">{esc(policy_id)}</div></div>')


def steps_html() -> str:
    cells = "".join(
        f'<div class="cl-step"><i>{i:02d}</i><b>{esc(name)}</b><span>{esc(text)}</span></div>'
        for i, (name, text) in enumerate(STEPS, 1)
    )
    return ('<div class="cl-card"><div class="cl-h">What happens when you run it</div>'
            f'<div class="cl-steps">{cells}</div></div>')


def friendly_error(exc: Exception, host: str) -> str:
    """Plain-language message for a failed run; the detail goes to the log."""
    name = type(exc).__name__
    if name in {"ConnectionError", "ConnectTimeout"}:
        return f"The local model stopped answering at {host}. Check Ollama, then run the audit again."
    if name in {"ReadTimeout", "Timeout"}:
        return "The local model took too long to answer. Run the audit again."
    if isinstance(exc, ValueError) and "JSON" in str(exc):
        return "The local model did not return usable output. Run the audit again."
    return "The audit could not finish. Details are in the app log."


def fmt_seconds(seconds: float) -> str:
    return f"{seconds:.1f} s"


@dataclass
class Row:
    name: str
    detail: str = ""
    status: str = "pending"  # pending | active | done | stopped
    started: float | None = None
    seconds: float | None = None


@dataclass
class RunProgress:
    """Live state of one audit. `handle` takes agent events; `progress_html` draws the state.

    Elapsed time comes from `clock`, so the panel can tick between events.
    """

    policy_id: str
    llm_model: str
    tau_aas: float
    max_rounds: int
    last_seconds: float | None = None
    clock: Callable[[], float] = time.perf_counter
    rows: list[Row] = field(default_factory=lambda: [Row(PREPARE)] + [Row(n) for n, _ in STEPS])
    device: str = ""
    started: float | None = None
    total: float | None = None

    def _start(self, i: int, detail: str) -> None:
        row = self.rows[i]
        row.status, row.detail, row.started = "active", detail, self.clock()

    def _finish(self, i: int, detail: str) -> None:
        row = self.rows[i]
        row.status, row.detail = "done", detail
        if row.started is not None:
            row.seconds = self.clock() - row.started

    def begin(self) -> None:
        self.started = self.clock()
        self._start(0, f"Loading the search and matching models and the {self.policy_id} index. "
                       "The first run takes longest.")

    def stop(self) -> None:
        """Mark the active row as stopped after a failure."""
        for row in self.rows:
            if row.status == "active":
                row.status = "stopped"
                row.seconds = self.clock() - row.started
        self.total = self.elapsed()

    def elapsed(self) -> float:
        if self.total is not None:
            return self.total
        return self.clock() - self.started if self.started is not None else 0.0

    def handle(self, event: str, data: dict[str, Any]) -> None:
        if event == "ready":
            self.device = data["device"]
            self._finish(0, "Models and policy index are ready.")
            self._start(1, f"Asking {self.llm_model} for the stated reason.")
        elif event == "parse":
            self._finish(1, f'Stated reason: "{data["reason"]}"')
            self._start(2, f"Searching {self.policy_id} by keyword and meaning, then re-ranking.")
        elif event == "retrieve":
            self._finish(2, self._retrieved(data["top_clauses"]))
            self._start(3, "Scoring the best clause against the reason.")
        elif event == "match":
            match = data.get("match")
            self._finish(3, f"Match score {match.prob:.2f}." if match else "No clause to score.")
            self._start(4, "Working out which clause words drove the match (Integrated Gradients).")
        elif event == "attribution":
            self.rows[4].detail = (f"Word influence ready for {data['words']} words. "
                                   f"Writing the first explanation with {self.llm_model}.")
        elif event == "aas":
            self.rows[4].detail = self._alignment(data["label"], data["aas"])
        elif event == "explain_reflect":
            self._finish(4, self._explained(data))
            self._start(5, "Applying the verdict thresholds.")
        elif event == "verdict":
            self.rows[5].detail = f"Verdict: {verdict_info(str(data['verdict']))[0]}."
        elif event == "draft":
            self._finish(5, self.rows[5].detail)
            self.total = self.elapsed()

    @staticmethod
    def _retrieved(clauses: list[Any]) -> str:
        if not clauses:
            return "No clause found in this policy."
        best = clauses[0].chunk
        return (f"Kept the top {len(clauses)}. Best: {best.section_path or best.chunk_id}, "
                f"page {best.page}.")

    def _alignment(self, label: str, aas: float) -> str:
        done = 0 if label == "initial" else int(label.rsplit("_", 1)[1])
        name = "First draft" if done == 0 else f"Revision {done}"
        out = f"{name} alignment {aas:.2f}, target {self.tau_aas:.2f}."
        if aas >= self.tau_aas:
            return out + " Target met."
        if done < self.max_rounds:
            return out + f" Asking {self.llm_model} for revision {done + 1} of {self.max_rounds}."
        return out + " No revisions left, keeping the best version."

    @staticmethod
    def _explained(data: dict[str, Any]) -> str:
        if "aas_final" not in data:
            return "Skipped: no clause to explain."
        history = " to ".join(f"{aas:.2f}" for _, aas in data["aas_history"])
        return f"Alignment {history}. Best version kept: {data['aas_final']:.2f}."


def _progress_row(row: Row, clock: Callable[[], float]) -> str:
    seconds = row.seconds
    if row.status == "active" and row.started is not None:
        seconds = clock() - row.started
    stamp = f'<div class="cl-ptime">{fmt_seconds(seconds)}</div>' if seconds is not None else ""
    detail = f'<div class="cl-pdetail">{esc(row.detail)}</div>' if row.detail else ""
    label = f"{row.name} (stopped)" if row.status == "stopped" else row.name
    pending = " pending" if row.status == "pending" else ""
    return (f'<div class="cl-prog"><div class="cl-dot {row.status}"></div><div>'
            f'<div class="cl-pname{pending}">{esc(label)}</div>{detail}</div>{stamp}</div>')


def progress_html(p: RunProgress) -> str:
    finished = p.total is not None
    stopped = any(r.status == "stopped" for r in p.rows)
    title = "Audit stopped" if stopped else "Audit finished" if finished else "Audit running"
    facts = [f"LLM {p.llm_model}, local Ollama", f"Policy {p.policy_id}"]
    if p.device:
        facts.append(f"Matcher on {p.device}")
    if p.last_seconds is not None and not finished:
        facts.append(f"Last run took {fmt_seconds(p.last_seconds)}")
    rows = "".join(_progress_row(r, p.clock) for r in p.rows)
    return ('<div class="cl-card"><div style="display:flex;justify-content:space-between;'
            f'align-items:baseline"><div class="cl-h">{esc(title)}</div>'
            f'<div class="cl-mono" style="font-size:22px;font-weight:500">'
            f'{fmt_seconds(p.elapsed())}</div></div>'
            f'<div class="cl-meta" style="margin-bottom:10px">{esc(" · ".join(facts))}</div>'
            f'{rows}</div>')
