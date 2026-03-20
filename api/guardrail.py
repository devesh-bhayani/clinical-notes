"""DrugBank NER validation layer for Clinical Note Summarizer.

Validates medication entity names in model outputs against the DrugBank
vocabulary. Appends unrecognized names to confidence_flags without mutating
the medications array. Must not raise exceptions on malformed input.

Contract:
    Input:  parsed model output dict
    Output: same dict with confidence_flags populated
    - Must not mutate medications[] entries — only append to confidence_flags
    - Must not raise exceptions on malformed input — catch and flag instead
"""

import csv
import os

from dotenv import load_dotenv

load_dotenv()


def load_drugbank_vocabulary(vocab_path: str | None = None) -> set[str]:
    """Load and normalize drug names from DrugBank CSV.

    Args:
        vocab_path: Path to CSV. Defaults to os.getenv("DRUGBANK_VOCAB_PATH").

    Returns:
        Set of lowercase, normalized drug names.
    """
    raise NotImplementedError


def validate_medications(output: dict, vocab: set[str] | None = None) -> dict:
    """Validate medication names against DrugBank vocabulary.

    For each medication in output["medications"], checks if the name exists
    in the vocabulary. Unrecognized names are appended to confidence_flags.

    Args:
        output: Model output dict conforming to the output schema.
        vocab: Optional pre-loaded vocabulary set. If None, loads from env path.

    Returns:
        The output dict with confidence_flags updated.
    """
    raise NotImplementedError
