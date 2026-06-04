"""Shared pytest fixtures and test-environment setup.

Importing this module configures environment variables that let the API and
guardrail run without a GPU, a trained checkpoint, or the real (HIPAA-regulated)
DrugBank vocabulary. The sample vocabulary committed under data/ is used instead.
"""

import os
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent
_SAMPLE_VOCAB = _REPO_ROOT / "data" / "drugbank_vocabulary.sample.csv"

# Configure the test environment before the app or guardrail import-time reads.
os.environ.setdefault("DRUGBANK_VOCAB_PATH", str(_SAMPLE_VOCAB))
os.environ.setdefault("ALLOW_STUB_INFERENCE", "1")


@pytest.fixture(scope="session")
def sample_vocab_path() -> str:
    """Absolute path to the committed sample DrugBank vocabulary."""
    return str(_SAMPLE_VOCAB)


@pytest.fixture(scope="session")
def drugbank_vocab(sample_vocab_path) -> set[str]:
    """Loaded, normalized DrugBank vocabulary from the sample CSV."""
    from api.guardrail import load_drugbank_vocabulary

    return load_drugbank_vocabulary(sample_vocab_path)
