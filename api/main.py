"""FastAPI inference server for Clinical Note Summarizer.

Serves the fine-tuned BioMistral-7B model with DrugBank guardrail validation.
Model is loaded once at startup via lifespan context.

Usage:
    uvicorn api.main:app --reload --port 8000
"""

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from api.guardrail import validate_medications

load_dotenv()

MAX_NOTE_TOKENS = 6000


# --- Pydantic Models (output schema contract) ---

class SummarizeRequest(BaseModel):
    note: str


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


# --- App lifecycle and routes ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and DrugBank vocabulary at startup."""
    # TODO: Load model and tokenizer into app.state
    # TODO: Load DrugBank vocabulary into app.state
    yield
    # Cleanup resources on shutdown


app = FastAPI(
    title="Clinical Note Summarizer",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    """Basic health check endpoint."""
    return {"status": "ok"}


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest):
    """Summarize a clinical note into structured JSON.

    Validates input length, runs inference, applies DrugBank guardrail,
    and returns the validated output.
    """
    if not request.note.strip():
        raise HTTPException(status_code=422, detail="Note cannot be empty")
    # Token counting requires tokenizer loaded at startup in app.state
    # For now, approximate: reject if character count suggests >6000 tokens
    if len(request.note) > MAX_NOTE_TOKENS * 4:  # ~4 chars/token approximation until tokenizer available
        raise HTTPException(status_code=422, detail=f"Note exceeds {MAX_NOTE_TOKENS} token limit")

    raise NotImplementedError


class BatchSummarizeRequest(BaseModel):
    notes: list[str]


class BatchSummarizeResponse(BaseModel):
    results: list[SummarizeResponse]


@app.post("/batch_summarize", response_model=BatchSummarizeResponse)
async def batch_summarize(request: BatchSummarizeRequest):
    """Summarize multiple clinical notes in a single request.

    Each note is processed independently through the same pipeline
    as the /summarize endpoint.
    """
    raise NotImplementedError
