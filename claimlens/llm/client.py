"""Provider-agnostic LLM client (Ollama now; a hosted API later via env vars, SPEC.md 12.1)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Protocol

from claimlens.utils.config import load_config

log = logging.getLogger(__name__)


class LLM(Protocol):
    def generate_json(self, prompt: str, system: str | None = None) -> dict[str, Any]: ...


def parse_json_loose(text: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating code fences or leading/trailing prose."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))


class OllamaClient:
    """Calls Ollama /api/chat with format=json. Retries once on unparseable output."""

    def __init__(self, host: str | None = None, model: str | None = None, seed: int = 42,
                 timeout: float = 300.0) -> None:
        cfg = load_config()["llm"]
        self.host = (host or cfg["host"]).rstrip("/")
        self.model = model or cfg["model"]
        self.seed = seed
        self.timeout = timeout

    def health(self) -> str | None:
        """None if Ollama answers and has the model, else the problem in plain language.

        Checked before a run: `extract_reason` falls back quietly when the LLM fails, so without
        this a stopped server would only show up after retrieval and matching have run.
        """
        import requests  # lazy, as in generate_text

        try:
            r = requests.get(f"{self.host}/api/tags", timeout=3)
            r.raise_for_status()
            names = {str(m.get("name", "")) for m in r.json().get("models", [])}
        except Exception as e:  # noqa: BLE001 - any failure here means the server is unusable
            log.warning("Ollama health check failed: %s", e)
            return f"Ollama is not answering at {self.host}. Start it, then run the audit again."
        if not {self.model, f"{self.model}:latest"} & names:
            return (f"Ollama is running, but the model {self.model} is not installed. "
                    f"Run: ollama pull {self.model}")
        return None

    def generate_text(self, prompt: str, system: str | None = None, json_mode: bool = False) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.2, "seed": self.seed},
        }
        if json_mode:
            body["format"] = "json"
        import requests  # lazy: CI installs no HTTP client (SPEC.md 7.10)

        r = requests.post(f"{self.host}/api/chat", json=body, timeout=self.timeout)
        r.raise_for_status()
        return str(r.json()["message"]["content"])

    def generate_json(self, prompt: str, system: str | None = None) -> dict[str, Any]:
        last: Exception | None = None
        for attempt in range(2):
            text = self.generate_text(prompt, system, json_mode=True)
            try:
                return parse_json_loose(text)
            except json.JSONDecodeError as e:
                last = e
                log.warning("unparseable JSON from LLM (attempt %d): %.120s", attempt + 1, text)
        raise ValueError(f"LLM did not return valid JSON: {last}")


class FakeLLM:
    """Deterministic stand-in for tests: returns queued responses (or a default)."""

    def __init__(self, responses: list[dict[str, Any]] | None = None,
                 default: dict[str, Any] | None = None) -> None:
        self.responses = list(responses or [])
        self.default = default or {}
        self.calls: list[str] = []

    def generate_json(self, prompt: str, system: str | None = None) -> dict[str, Any]:
        self.calls.append(prompt)
        return self.responses.pop(0) if self.responses else dict(self.default)
