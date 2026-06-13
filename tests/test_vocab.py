"""Tests for the RxNorm drug-vocabulary fetcher (scripts/fetch_drug_vocab.py).

The CSV-writing path and its compatibility with the guardrail loader are tested
offline. The live RxNorm fetch is gated behind RUN_VOCAB_NETWORK_TESTS so CI
stays hermetic.

Run with:
    python -m pytest tests/test_vocab.py -v
"""

import os

import pytest

from api.guardrail import load_drugbank_vocabulary, validate_medications
from scripts.fetch_drug_vocab import write_vocab


def _med(name):
    return {"name": name, "dose": "", "freq": "", "route": ""}


def test_written_vocab_loads_in_guardrail(tmp_path):
    """write_vocab output is consumable by the guardrail loader."""
    path = tmp_path / "vocab.csv"
    write_vocab(["Aspirin", "Metformin", "Lipitor"], str(path))

    vocab = load_drugbank_vocabulary(str(path))
    assert {"aspirin", "metformin", "lipitor"} <= vocab

    out = validate_medications(
        {"medications": [_med("Aspirin"), _med("Hepatraxone")], "confidence_flags": []},
        vocab=vocab,
    )
    # Real drug passes; fabricated drug is flagged.
    assert len(out["confidence_flags"]) == 1
    assert "Hepatraxone" in out["confidence_flags"][0]


@pytest.mark.skipif(
    not os.getenv("RUN_VOCAB_NETWORK_TESTS"),
    reason="set RUN_VOCAB_NETWORK_TESTS=1 to hit the live RxNorm endpoint",
)
def test_fetch_names_live():
    from scripts.fetch_drug_vocab import fetch_names

    names = fetch_names()
    assert len(names) > 10_000
    assert any(n.lower() == "aspirin" for n in names)
