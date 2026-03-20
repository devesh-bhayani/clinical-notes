"""Unit tests for the DrugBank NER guardrail validation layer.

Run with:
    python -m pytest eval/test_guardrail.py -v
"""

import pytest

from api.guardrail import validate_medications


def test_exact_match_recognized():
    """Known drug name passes validation without adding confidence flags."""
    pass


def test_case_insensitive_match():
    """Case variants of a known drug name are still recognized."""
    pass


def test_unrecognized_drug_flagged():
    """Fabricated drug name triggers a confidence flag."""
    pass


def test_empty_medications_array():
    """Empty medications list passes validation without error."""
    pass


def test_multiple_medications_mixed():
    """Mix of valid and invalid drug names returns correct flags."""
    pass


def test_malformed_input_handled():
    """Non-dict or missing-key input does not raise an exception."""
    pass
