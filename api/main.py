"""FastAPI inference server for Clinical Note Summarizer.

Serves the fine-tuned BioMistral-7B model with DrugBank guardrail validation.
The summarizer backend and DrugBank vocabulary are loaded once at startup via
the lifespan context and held on ``app.state``.

Usage:
    uvicorn api.main:app --reload --port 8000
"""

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from api.guardrail import get_vocabulary, validate_medications
from api.inference import BioMistralSummarizer, StubSummarizer, Summarizer

load_dotenv()

logger = logging.getLogger("clinical_notes.api")

# Reject clearly oversized notes before tokenization. ~4 chars/token, so this
# corresponds to roughly the 6k-token context budget for a note.
MAX_NOTE_CHARS = 24_000


def _truthy(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


# --- Pydantic Models (output schema contract) ---

class SummarizeRequest(BaseModel):
    note: str = Field(..., description="Raw clinical note text")


class MedicationItem(BaseModel):
    name: str
    dose: str
    freq: str
    route: str


class SummarizeResponse(BaseModel):
    diagnoses: list[str]
    medications: list[MedicationItem]
    procedures: list[str]
    discharge_instructions: str
    confidence_flags: list[str]


class BatchSummarizeRequest(BaseModel):
    notes: list[str]


class BatchSummarizeResponse(BaseModel):
    results: list[SummarizeResponse]


# --- Backend selection ---

def _build_summarizer() -> Summarizer | None:
    """Choose an inference backend from the environment.

    Production: a trained checkpoint at CHECKPOINT_DIR loads BioMistral.
    Otherwise the stub is used only if ALLOW_STUB_INFERENCE is explicitly set,
    so a misconfigured production deploy fails loudly instead of serving fakes.
    """
    checkpoint_dir = os.getenv("CHECKPOINT_DIR")
    if checkpoint_dir and os.path.isdir(checkpoint_dir):
        logger.info("Loading BioMistral summarizer from checkpoint: %s", checkpoint_dir)
        return BioMistralSummarizer(checkpoint_dir)

    if _truthy(os.getenv("ALLOW_STUB_INFERENCE")):
        logger.warning(
            "Using StubSummarizer — outputs are NOT model-generated. "
            "Set CHECKPOINT_DIR to a trained adapter for real inference."
        )
        return StubSummarizer()

    logger.error(
        "No checkpoint found and ALLOW_STUB_INFERENCE is not set; "
        "/summarize will return 503 until a model is configured."
    )
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the summarizer and DrugBank vocabulary at startup."""
    app.state.summarizer = _build_summarizer()
    try:
        app.state.vocab = get_vocabulary()
        logger.info("Loaded DrugBank vocabulary (%d entries).", len(app.state.vocab))
    except (FileNotFoundError, ValueError) as exc:
        app.state.vocab = set()
        logger.error("DrugBank vocabulary unavailable: %s", exc)
    yield
    app.state.summarizer = None
    app.state.vocab = None


app = FastAPI(
    title="Clinical Note Summarizer",
    version="0.1.0",
    lifespan=lifespan,
)


# --- Dependencies (overridable in tests) ---

def get_summarizer(request: Request) -> Summarizer:
    summarizer = getattr(request.app.state, "summarizer", None)
    if summarizer is None:
        raise HTTPException(
            status_code=503,
            detail="Inference backend not available. Configure CHECKPOINT_DIR.",
        )
    return summarizer


def get_vocab(request: Request) -> set[str]:
    return getattr(request.app.state, "vocab", set()) or set()


# --- Routes ---

@app.get("/health")
async def health(request: Request):
    """Liveness/readiness check reporting backend availability."""
    summarizer = getattr(request.app.state, "summarizer", None)
    vocab = getattr(request.app.state, "vocab", set()) or set()
    return {
        "status": "ok",
        "model_ready": summarizer is not None,
        "vocab_size": len(vocab),
    }


def _summarize_note(note: str, summarizer: Summarizer, vocab: set[str]) -> dict:
    """Run inference and the guardrail for a single note."""
    raw = summarizer.summarize(note)
    return validate_medications(raw, vocab=vocab)


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(
    request: SummarizeRequest,
    summarizer: Summarizer = Depends(get_summarizer),
    vocab: set[str] = Depends(get_vocab),
):
    """Summarize a clinical note into structured JSON.

    Validates input length, runs inference, applies the DrugBank guardrail,
    and returns the validated output.
    """
    if not request.note.strip():
        raise HTTPException(status_code=422, detail="Note cannot be empty")
    if len(request.note) > MAX_NOTE_CHARS:
        raise HTTPException(
            status_code=422,
            detail=f"Note exceeds the {MAX_NOTE_CHARS}-character limit",
        )
    return _summarize_note(request.note, summarizer, vocab)


@app.post("/batch_summarize", response_model=BatchSummarizeResponse)
async def batch_summarize(
    request: BatchSummarizeRequest,
    summarizer: Summarizer = Depends(get_summarizer),
    vocab: set[str] = Depends(get_vocab),
):
    """Summarize multiple clinical notes in a single request.

    Each note is processed independently through the same pipeline as
    /summarize. Empty or oversized notes yield an error-flagged result rather
    than failing the whole batch.
    """
    if not request.notes:
        raise HTTPException(status_code=422, detail="notes list cannot be empty")

    results = []
    for note in request.notes:
        if not isinstance(note, str) or not note.strip():
            results.append(
                {**_empty_result(), "confidence_flags": ["error: empty note"]}
            )
        elif len(note) > MAX_NOTE_CHARS:
            results.append(
                {**_empty_result(), "confidence_flags": ["error: note too long"]}
            )
        else:
            results.append(_summarize_note(note, summarizer, vocab))
    return {"results": results}


def _empty_result() -> dict:
    return {
        "diagnoses": [],
        "medications": [],
        "procedures": [],
        "discharge_instructions": "",
        "confidence_flags": [],
    }
