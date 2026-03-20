"""Streamlit UI for Clinical Note Summarizer.

Provides an interactive interface for submitting clinical notes
and viewing structured JSON summaries with confidence flags.

Usage:
    streamlit run ui/streamlit_app.py
"""

import json

import httpx
import streamlit as st

API_BASE_URL = "http://localhost:8000"


def render_input_panel() -> str | None:
    """Render the clinical note input text area.

    Returns:
        The submitted note text, or None if not yet submitted.
    """
    raise NotImplementedError


def render_output_panel(response: dict) -> None:
    """Render structured output: diagnoses, medications, procedures, instructions.

    Args:
        response: API response dict conforming to the output schema.
    """
    raise NotImplementedError


def render_confidence_flags(flags: list[str]) -> None:
    """Render confidence flags as highlighted warnings.

    Args:
        flags: List of confidence flag strings from the model output.
    """
    raise NotImplementedError


def call_api(note: str) -> dict:
    """POST clinical note to FastAPI /summarize endpoint.

    Args:
        note: Raw clinical note text.

    Returns:
        API response dict.
    """
    raise NotImplementedError


def main() -> None:
    """Streamlit page layout and interaction flow."""
    st.set_page_config(
        page_title="Clinical Note Summarizer",
        page_icon="🏥",
        layout="wide",
    )
    st.title("Clinical Note Summarizer")
    st.caption("Hallucination-resistant discharge summary generation via ORPO fine-tuning")

    raise NotImplementedError


if __name__ == "__main__":
    main()
