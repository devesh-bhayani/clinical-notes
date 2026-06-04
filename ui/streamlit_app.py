"""Streamlit UI for Clinical Note Summarizer.

Provides an interactive interface for submitting clinical notes and viewing
structured JSON summaries with confidence flags.

Usage:
    streamlit run ui/streamlit_app.py
"""

import json
import os

import httpx
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT = float(os.getenv("API_TIMEOUT", "30"))


def render_input_panel() -> str | None:
    """Render the clinical note input area. Returns the note on submit, else None."""
    with st.form("note_form"):
        note = st.text_area(
            "Clinical note",
            height=320,
            placeholder="Paste a discharge note here...",
        )
        submitted = st.form_submit_button("Summarize", type="primary")
    return note if submitted else None


def render_confidence_flags(flags: list[str]) -> None:
    """Render confidence flags as highlighted warnings."""
    if not flags:
        st.success("No confidence flags raised.")
        return
    st.warning(f"{len(flags)} confidence flag(s) raised:")
    for flag in flags:
        st.markdown(f"- ⚠️ {flag}")


def render_output_panel(response: dict) -> None:
    """Render structured output: diagnoses, medications, procedures, instructions."""
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Diagnoses")
        diagnoses = response.get("diagnoses") or []
        if diagnoses:
            for d in diagnoses:
                st.markdown(f"- {d}")
        else:
            st.caption("None extracted.")

        st.subheader("Procedures")
        procedures = response.get("procedures") or []
        if procedures:
            for p in procedures:
                st.markdown(f"- {p}")
        else:
            st.caption("None extracted.")

    with col2:
        st.subheader("Medications")
        meds = response.get("medications") or []
        if meds:
            st.dataframe(meds, use_container_width=True, hide_index=True)
        else:
            st.caption("None extracted.")

    st.subheader("Discharge Instructions")
    st.write(response.get("discharge_instructions") or "_None provided._")

    st.subheader("Confidence Flags")
    render_confidence_flags(response.get("confidence_flags") or [])

    with st.expander("Raw JSON"):
        st.code(json.dumps(response, indent=2), language="json")


def call_api(note: str) -> dict:
    """POST a clinical note to the FastAPI /summarize endpoint.

    Raises httpx.HTTPStatusError on non-2xx responses.
    """
    resp = httpx.post(
        f"{API_BASE_URL}/summarize",
        json={"note": note},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def _backend_status() -> dict | None:
    try:
        resp = httpx.get(f"{API_BASE_URL}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    """Streamlit page layout and interaction flow."""
    st.set_page_config(
        page_title="Clinical Note Summarizer",
        page_icon="🏥",
        layout="wide",
    )
    st.title("Clinical Note Summarizer")
    st.caption("Hallucination-resistant discharge summary generation via ORPO fine-tuning")

    status = _backend_status()
    with st.sidebar:
        st.header("Backend")
        st.write(f"API: `{API_BASE_URL}`")
        if status is None:
            st.error("API unreachable. Start it with `uvicorn api.main:app --port 8000`.")
        else:
            ready = status.get("model_ready")
            (st.success if ready else st.warning)(
                f"Model ready: {ready} · Vocab: {status.get('vocab_size', 0)} drugs"
            )

    note = render_input_panel()
    if note is None:
        return
    if not note.strip():
        st.error("Please enter a clinical note.")
        return

    with st.spinner("Summarizing..."):
        try:
            response = call_api(note)
        except httpx.HTTPStatusError as exc:
            detail = exc.response.json().get("detail", str(exc))
            st.error(f"API error ({exc.response.status_code}): {detail}")
            return
        except httpx.HTTPError as exc:
            st.error(f"Could not reach the API: {exc}")
            return

    render_output_panel(response)


if __name__ == "__main__":
    main()
