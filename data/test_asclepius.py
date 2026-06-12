"""Tests for the Asclepius adapter and prose-aware diagnosis extraction.

The network/datasets-backed loader is exercised only when
RUN_ASCLEPIUS_NETWORK_TESTS is set, so CI stays hermetic. The extraction and
split logic is tested on synthetic Asclepius-style notes built with safe section
headers (no PHI-shaped literals) so this test file passes the repo-wide scan.

Run with:
    python -m pytest data/test_asclepius.py -v
"""

import json
import os

import pytest

from data import asclepius
from data.pipeline import extract_diagnoses

# Asclepius-style narrative notes (prose diagnoses, sparse meds). Safe headers.
NOTE_PROSE = """\
Discharge Summary:

Hospital Course:
The patient was admitted with fever and cough and treated supportively.

Discharge Diagnosis:
The patient was discharged with a diagnosis of septic shock, acute kidney \
injury, and COVID-19 infection.

Discharge Medications:
None specified.

Discharge Instructions:
Follow up with the primary care provider in two weeks. Rest at home.
"""

NOTE_WITH_MED = """\
Discharge Summary:

Discharge Diagnosis: Acute appendicitis

Discharge Medications:
1. Aspirin 81 mg PO daily

Discharge Instructions:
Avoid heavy lifting for two weeks.
"""

NOTE_NO_DX = """\
Discharge Summary:

Hospital Course:
The patient recovered and was discharged in stable condition.
"""


# --- Prose diagnosis extraction ---------------------------------------------

def test_extract_diagnoses_prose_list():
    dx = extract_diagnoses(
        "The patient was discharged with a diagnosis of septic shock, "
        "acute kidney injury, and COVID-19 infection."
    )
    assert dx == ["septic shock", "acute kidney injury", "COVID-19 infection"]


def test_extract_diagnoses_terse_single():
    assert extract_diagnoses("Acute appendicitis") == ["Acute appendicitis"]


def test_extract_diagnoses_negative_section():
    assert extract_diagnoses("None") == []
    assert extract_diagnoses("N/A") == []
    assert extract_diagnoses("") == []


def test_extract_diagnoses_keeps_descriptive_prose_whole():
    """A descriptive sentence is kept as one diagnosis, not shredded on commas."""
    dx = extract_diagnoses(
        "Based on the pathological examination, nodular, non-capsulated, "
        "well-circumscribed mass with atypical spindle cells."
    )
    # The "based on" lead is filtered; the remainder stays a single entry (not
    # split into "nodular"/"non-capsulated"/... fragments).
    assert len(dx) <= 1


# --- build_chosen_summaries (offline, injected notes) -----------------------

def test_build_chosen_summaries_offline(tmp_path):
    stats = asclepius.build_chosen_summaries(
        tmp_path,
        notes=iter([NOTE_PROSE, NOTE_WITH_MED, NOTE_NO_DX]),
        min_diagnoses=1,
    )
    assert stats["processed"] == 3
    assert stats["kept"] == 2  # NOTE_NO_DX dropped (no diagnoses)

    rows = [json.loads(l) for l in open(stats["output"], encoding="utf-8")]
    assert len(rows) == 2
    med_row = next(r for r in rows if r["chosen"]["medications"])
    assert med_row["chosen"]["medications"][0]["name"].lower() == "aspirin"
    prose_row = next(r for r in rows if "septic shock" in r["chosen"]["diagnoses"])
    assert prose_row["chosen"]["medications"] == []  # "None specified" -> empty


def test_prepare_orpo_splits(tmp_path):
    asclepius.build_chosen_summaries(
        tmp_path,
        notes=iter([NOTE_PROSE, NOTE_WITH_MED] * 10),
        min_diagnoses=1,
    )
    counts = asclepius.prepare_orpo_splits(tmp_path / "chosen_summaries.jsonl", tmp_path)
    assert counts["train"] + counts["val"] + counts["test"] == 20

    pairs = [json.loads(l) for l in open(tmp_path / "train_orpo.jsonl", encoding="utf-8")]
    assert pairs, "expected at least one training pair"
    for p in pairs:
        assert {"prompt", "chosen", "rejected", "failure_class"} <= set(p)
        assert p["chosen"] != p["rejected"]
        assert isinstance(json.loads(p["chosen"]), dict)


# --- Real dataset loader (opt-in, network) ----------------------------------

@pytest.mark.skipif(
    not os.getenv("RUN_ASCLEPIUS_NETWORK_TESTS"),
    reason="set RUN_ASCLEPIUS_NETWORK_TESTS=1 to exercise the live datasets loader",
)
def test_iter_asclepius_notes_live():
    notes = list(asclepius.iter_asclepius_notes(limit=5))
    assert len(notes) == 5
    assert all(isinstance(n, str) and n for n in notes)
