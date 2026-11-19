"""API tests with the agent mocked: no LLM, no models, no Ollama (SPEC.md 13.5)."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from api import main  # noqa: E402
from claimlens.schemas import CaseRecord, CaseResult, CaseStatus, Verdict  # noqa: E402
from claimlens.utils.storage import load_case, save_case  # noqa: E402

LETTER = {"letter_text": "Your claim is rejected.", "policy_id": "pol"}


def fake_runner(case):
    return CaseResult(case_id=case.case_id, reason="rejected", top_clauses=[],
                      verdict=Verdict.INSUFFICIENT_EVIDENCE)


def broken_runner(case):
    raise RuntimeError("Ollama is down")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAIMLENS_STATE_DIR", str(tmp_path))
    main.app.dependency_overrides[main.get_policies] = lambda: ["pol"]
    main.app.dependency_overrides[main.get_runner] = lambda: fake_runner
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()


def create(client) -> str:
    response = client.post("/cases", json=LETTER)
    assert response.status_code == 201
    return response.json()["case_id"]


def test_create_then_run_then_read(client):
    case_id = create(client)
    assert client.get(f"/cases/{case_id}").json()["status"] == "created"
    ran = client.post(f"/cases/{case_id}/run")
    assert ran.status_code == 200
    assert ran.json()["status"] == "done"
    body = client.get(f"/cases/{case_id}").json()
    assert body["result"]["verdict"] == "INSUFFICIENT_EVIDENCE"
    assert body["letter_text"] == LETTER["letter_text"]


def test_every_response_carries_the_disclaimer(client):
    case_id = create(client)
    assert "not legal advice" in client.post(f"/cases/{case_id}/run").json()["disclaimer"]


def test_unknown_policy_is_rejected(client):
    response = client.post("/cases", json={**LETTER, "policy_id": "other"})
    assert response.status_code == 422
    assert "pol" in response.json()["detail"]


def test_empty_letter_is_rejected(client):
    assert client.post("/cases", json={**LETTER, "letter_text": ""}).status_code == 422


def test_unknown_case_is_404(client):
    assert client.get("/cases/nope").status_code == 404
    assert client.post("/cases/nope/run").status_code == 404


def test_a_path_traversal_id_is_404_and_reads_nothing(client):
    assert client.get("/cases/..%2F..%2Fsecret").status_code == 404
    assert load_case("../secret") is None


def test_a_failed_audit_is_stored_not_left_running(client):
    main.app.dependency_overrides[main.get_runner] = lambda: broken_runner
    case_id = create(client)
    body = client.post(f"/cases/{case_id}/run").json()
    assert body["status"] == "failed"
    assert "Ollama is down" in body["error"]
    assert client.get(f"/cases/{case_id}").json()["status"] == "failed"


def test_a_failed_case_can_be_run_again(client):
    main.app.dependency_overrides[main.get_runner] = lambda: broken_runner
    case_id = create(client)
    client.post(f"/cases/{case_id}/run")
    main.app.dependency_overrides[main.get_runner] = lambda: fake_runner
    body = client.post(f"/cases/{case_id}/run").json()
    assert (body["status"], body["error"]) == ("done", None)


def test_a_running_case_cannot_start_twice(client):
    case_id = create(client)
    save_case(CaseRecord(case_id=case_id, policy_id="pol", letter_text="x",
                         status=CaseStatus.RUNNING))
    assert client.post(f"/cases/{case_id}/run").status_code == 409


def test_record_round_trips_through_storage(client):
    case_id = create(client)
    client.post(f"/cases/{case_id}/run")
    assert load_case(case_id).result.case_id == case_id
