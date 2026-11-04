import pytest

from claimlens.llm.client import FakeLLM, parse_json_loose


def test_parse_json_loose_handles_fences_and_prose():
    assert parse_json_loose('{"a": 1}') == {"a": 1}
    assert parse_json_loose('Sure!\n```json\n{"a": 2}\n```') == {"a": 2}
    with pytest.raises(ValueError):
        parse_json_loose("no json here")


def test_fake_llm_queue_and_default():
    llm = FakeLLM([{"x": 1}], default={"y": 2})
    assert llm.generate_json("p") == {"x": 1}
    assert llm.generate_json("q") == {"y": 2}
    assert llm.calls == ["p", "q"]
