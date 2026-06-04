"""API endpoint tests for Clinical Note Summarizer.

These run against the StubSummarizer backend (enabled via ALLOW_STUB_INFERENCE
in conftest.py) so no GPU or trained checkpoint is required.

Run with:
    python -m pytest api/tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app

SAMPLE_NOTE = (
    "Diagnosis: Type 2 diabetes mellitus.\n"
    "Patient started on metformin and aspirin.\n"
    "Discharge home with follow-up in two weeks."
)


@pytest.fixture
def client():
    """Create a test client for the FastAPI app (triggers lifespan startup)."""
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    """GET /health returns 200 with status ok and readiness info."""
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_ready"] is True
    assert body["vocab_size"] > 0


def test_summarize_valid_note(client):
    """POST /summarize with a valid note returns 200 and the correct schema."""
    resp = client.post("/summarize", json={"note": SAMPLE_NOTE})
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["medications"], list)
    # The stub recognizes metformin and aspirin in the note.
    names = {m["name"].lower() for m in body["medications"]}
    assert "metformin" in names and "aspirin" in names


def test_summarize_empty_note(client):
    """POST /summarize with an empty/whitespace string returns 422."""
    assert client.post("/summarize", json={"note": ""}).status_code == 422
    assert client.post("/summarize", json={"note": "   "}).status_code == 422


def test_summarize_note_too_long(client):
    """POST /summarize with an oversized note returns 422."""
    resp = client.post("/summarize", json={"note": "x" * 24_001})
    assert resp.status_code == 422


def test_summarize_missing_field(client):
    """POST /summarize without the note field returns 422 (validation error)."""
    assert client.post("/summarize", json={}).status_code == 422


def test_summarize_response_schema(client):
    """Validates the response JSON has all required fields with correct types."""
    body = client.post("/summarize", json={"note": SAMPLE_NOTE}).json()
    assert set(body) == {
        "diagnoses",
        "medications",
        "procedures",
        "discharge_instructions",
        "confidence_flags",
    }
    assert isinstance(body["diagnoses"], list)
    assert isinstance(body["procedures"], list)
    assert isinstance(body["discharge_instructions"], str)
    assert isinstance(body["confidence_flags"], list)
    for med in body["medications"]:
        assert set(med) == {"name", "dose", "freq", "route"}


def test_guardrail_runs_on_response(client):
    """confidence_flags is present and the guardrail flags an unknown drug."""
    body = client.post("/summarize", json={"note": SAMPLE_NOTE}).json()
    assert "confidence_flags" in body
    # A note mentioning a fabricated drug should be flagged. The stub only emits
    # known drugs, so we verify the guardrail directly on a crafted payload.
    from api.guardrail import validate_medications

    out = validate_medications(
        {"medications": [{"name": "Fakeazol", "dose": "", "freq": "", "route": ""}],
         "confidence_flags": []},
        vocab=getattr(app.state, "vocab", set()),
    )
    assert any("Fakeazol" in f for f in out["confidence_flags"])


def test_batch_summarize(client):
    """POST /batch_summarize returns one result per note, isolating bad inputs."""
    resp = client.post(
        "/batch_summarize",
        json={"notes": [SAMPLE_NOTE, "", "y" * 24_001]},
    )
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) == 3
    assert any("empty note" in f for f in results[1]["confidence_flags"])
    assert any("too long" in f for f in results[2]["confidence_flags"])


def test_batch_summarize_empty_list(client):
    """POST /batch_summarize with no notes returns 422."""
    assert client.post("/batch_summarize", json={"notes": []}).status_code == 422
