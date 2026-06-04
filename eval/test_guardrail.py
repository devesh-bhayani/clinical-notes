"""Unit tests for the DrugBank NER guardrail validation layer.

Run with:
    python -m pytest eval/test_guardrail.py -v
"""

import pytest

from api.guardrail import (
    load_drugbank_vocabulary,
    normalize_name,
    validate_medications,
)


def _med(name, dose="", freq="", route=""):
    return {"name": name, "dose": dose, "freq": freq, "route": route}


def _output(meds, flags=None):
    return {
        "diagnoses": [],
        "medications": meds,
        "procedures": [],
        "discharge_instructions": "",
        "confidence_flags": flags if flags is not None else [],
    }


# --- Vocabulary loading -----------------------------------------------------

def test_vocab_loads_common_names_and_synonyms(drugbank_vocab):
    """Common names and pipe-delimited synonyms are both normalized into the set."""
    assert "aspirin" in drugbank_vocab
    assert "acetylsalicylic acid" in drugbank_vocab  # synonym
    assert "paracetamol" in drugbank_vocab  # synonym of acetaminophen


def test_vocab_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_drugbank_vocabulary("does/not/exist.csv")


def test_normalize_name():
    assert normalize_name("  Aspirin. ") == "aspirin"
    assert normalize_name("METFORMIN") == "metformin"
    assert normalize_name(None) == ""
    assert normalize_name("") == ""


# --- validate_medications ---------------------------------------------------

def test_exact_match_recognized(drugbank_vocab):
    """Known drug name passes validation without adding confidence flags."""
    out = validate_medications(_output([_med("Aspirin")]), vocab=drugbank_vocab)
    assert out["confidence_flags"] == []


def test_case_insensitive_match(drugbank_vocab):
    """Case variants of a known drug name are still recognized."""
    out = validate_medications(_output([_med("METFORMIN")]), vocab=drugbank_vocab)
    assert out["confidence_flags"] == []


def test_synonym_recognized(drugbank_vocab):
    """A synonym (paracetamol) resolves to its drug without flagging."""
    out = validate_medications(_output([_med("Paracetamol")]), vocab=drugbank_vocab)
    assert out["confidence_flags"] == []


def test_minor_misspelling_recognized_via_fuzzy(drugbank_vocab):
    """A near-miss spelling passes via fuzzy match above threshold."""
    out = validate_medications(_output([_med("Ibuprofin")]), vocab=drugbank_vocab)
    assert out["confidence_flags"] == []


def test_unrecognized_drug_flagged(drugbank_vocab):
    """Fabricated drug name triggers a confidence flag."""
    out = validate_medications(_output([_med("Fakeazol")]), vocab=drugbank_vocab)
    assert len(out["confidence_flags"]) == 1
    assert "Fakeazol" in out["confidence_flags"][0]


def test_empty_medications_array(drugbank_vocab):
    """Empty medications list passes validation without error."""
    out = validate_medications(_output([]), vocab=drugbank_vocab)
    assert out["confidence_flags"] == []


def test_multiple_medications_mixed(drugbank_vocab):
    """Mix of valid and invalid drug names returns exactly the invalid flags."""
    out = validate_medications(
        _output([_med("Aspirin"), _med("Zzzdrug"), _med("Warfarin")]),
        vocab=drugbank_vocab,
    )
    assert len(out["confidence_flags"]) == 1
    assert "Zzzdrug" in out["confidence_flags"][0]


def test_does_not_mutate_input(drugbank_vocab):
    """The input dict and its medications are never mutated."""
    original = _output([_med("Zzzdrug")])
    validate_medications(original, vocab=drugbank_vocab)
    assert original["confidence_flags"] == []
    assert original["medications"] == [_med("Zzzdrug")]


def test_preserves_existing_flags(drugbank_vocab):
    """Pre-existing confidence flags are preserved and appended to."""
    out = validate_medications(
        _output([_med("Zzzdrug")], flags=["pre-existing flag"]),
        vocab=drugbank_vocab,
    )
    assert "pre-existing flag" in out["confidence_flags"]
    assert len(out["confidence_flags"]) == 2


def test_missing_medication_name_flagged(drugbank_vocab):
    """A medication entry with a blank name is flagged, not skipped."""
    out = validate_medications(_output([_med("")]), vocab=drugbank_vocab)
    assert len(out["confidence_flags"]) == 1
    assert "missing a name" in out["confidence_flags"][0]


def test_empty_vocab_flags_everything():
    """With no vocabulary, every drug is flagged rather than silently passed."""
    out = validate_medications(_output([_med("Aspirin")]), vocab=set())
    assert len(out["confidence_flags"]) == 1


def test_malformed_input_handled(drugbank_vocab):
    """Non-dict or missing-key input does not raise an exception."""
    # Non-dict top-level input.
    out = validate_medications("not a dict", vocab=drugbank_vocab)
    assert isinstance(out, dict)
    assert any("malformed" in f for f in out["confidence_flags"])

    # Missing medications key.
    out = validate_medications({"diagnoses": []}, vocab=drugbank_vocab)
    assert isinstance(out["confidence_flags"], list)

    # medications is not a list.
    out = validate_medications({"medications": "aspirin"}, vocab=drugbank_vocab)
    assert any("not a list" in f for f in out["confidence_flags"])

    # medication entries are not dicts (string entries).
    out = validate_medications(_output(["Aspirin", "Zzzdrug"]), vocab=drugbank_vocab)
    assert len(out["confidence_flags"]) == 1  # only Zzzdrug flagged
