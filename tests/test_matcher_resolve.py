"""The matcher finds a local checkpoint and falls back on a fresh clone."""

import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from claimlens.matcher import scorer  # noqa: E402


@pytest.fixture
def config(monkeypatch):
    cfg = {"matcher": {"model": "models/absent", "fallback_model": "org/pretrained"}}
    monkeypatch.setattr(scorer, "load_config", lambda: cfg)
    return cfg


def test_missing_checkpoint_falls_back_and_warns(config, caplog):
    assert scorer.resolve_model() == "org/pretrained"
    assert "not found" in caplog.text


def test_hub_id_passes_through(config):
    assert scorer.resolve_model("org/other") == "org/other"


def test_existing_checkpoint_is_used(config, tmp_path):
    assert scorer.resolve_model(str(tmp_path)) == str(tmp_path)


def test_checkpoint_length_comes_from_its_metadata(tmp_path):
    assert scorer.checkpoint_max_len(str(tmp_path)) == scorer.MAX_LEN
    (tmp_path / scorer.TRAIN_META).write_text(json.dumps({"max_length": 256}), encoding="utf-8")
    assert scorer.checkpoint_max_len(str(tmp_path)) == 256
