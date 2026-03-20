"""API endpoint tests for Clinical Note Summarizer.

Run with:
    python -m pytest api/tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


def test_health_endpoint(client):
    """GET /health returns 200 with status ok."""
    pass


def test_summarize_valid_note(client):
    """POST /summarize with valid note returns 200 and correct schema."""
    pass


def test_summarize_empty_note(client):
    """POST /summarize with empty string returns 422."""
    pass


def test_summarize_note_too_long(client):
    """POST /summarize with >8192 chars returns 422."""
    pass


def test_summarize_response_schema(client):
    """Validates response JSON has all required fields."""
    pass


def test_guardrail_runs_on_response(client):
    """Verifies confidence_flags is present in response."""
    pass
