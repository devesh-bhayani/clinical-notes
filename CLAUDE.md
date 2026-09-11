# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Run Commands

```bash
# Train
python train/orpo_train.py --config configs/orpo_base.yaml

# Evaluate
python eval/run_eval_suite.py --split test

# Serve API
uvicorn api.main:app --reload --port 8000

# Serve UI
streamlit run ui/streamlit_app.py
```

## Architecture

Two-stage fine-tuning pipeline on BioMistral-7B for clinical note summarization:

- **Quantization**: 4-bit NF4 via `BitsAndBytesConfig` — **CUDA only**. Apple
  Silicon loads bf16 instead (`load_in_4bit: false`); bitsandbytes has no MPS build.
- **Adapters**: LoRA r=32, alpha=64 via PEFT. The Mac config also targets the
  MLP projections (gate/up/down), not just attention.
- **Alignment**: ORPO preference optimization via TRL `ORPOTrainer`
- **Serving**: FastAPI app at `api/main.py`, port 8000

Pinned library versions live in the requirements files, not here. See
**Dependencies** below for the two upper bounds that are load-bearing.

### Key directory layout (expected)
- `train/orpo_train.py` — ORPO trainer entry point, reads `configs/orpo_base.yaml`
- `eval/run_eval_suite.py` — evaluation suite (Drug Entity Error Rate, HHEM, BERTScore)
- `api/main.py` — FastAPI inference server
- `configs/orpo_base.yaml` — hyperparameters and training config
- `/data/drugbank_vocabulary.csv` — DrugBank entity vocabulary for NER evaluation
- `/data/mimic/` — **HIPAA-regulated training data, never commit or log**

## Dependencies

Two requirements files: `requirements.txt` (CUDA) and `requirements-mac.txt`
(Apple Silicon / MPS — same set minus bitsandbytes, plus an explicit torch).

The trainer targets TRL's `processing_class` API (trl >= 0.12); trl 0.9.x will
not work (it used the old `tokenizer=` argument).

**The upper bounds are load-bearing.** Without them a fresh install resolves to
versions that break the pipeline, both silently enough to waste a day:

| bound | what breaks above it |
|---|---|
| `trl<0.29` | 0.29 moved `ORPOConfig`/`ORPOTrainer` into `trl.experimental`, so `from trl import ORPOConfig` raises `ImportError`. 0.24–0.28 keep the top-level export. |
| `transformers<5` | 5.x dropped `warmup_ratio` and `logging_dir` from `TrainingArguments`, and breaks Vectara's HHEM model — which makes `eval/metrics.py` fall back to a lexical proxy that does not measure hallucination. |

`train/orpo_train.py::_adapt_config_kwargs` translates or drops the kwargs 5.x
removed, so the trainer still runs there. It is a guard, not a licence to lift
the pin: the HHEM breakage is unaffected by it. On 4.x the shim is a no-op and
`warmup_ratio` is honored exactly rather than approximated into `warmup_steps`.

Verified working set (Python 3.11, Apple Silicon and CUDA):

```
transformers>=4.46.0,<5     # verified 4.57.6
peft>=0.12.0                # verified 0.20
trl>=0.12.0,<0.29           # verified 0.24
bitsandbytes>=0.43.3        # CUDA only; omitted on Mac
```

### Apple Silicon

```bash
uv venv --python 3.11 .venv
.venv/bin/python -m pip install -r requirements-mac.txt
.venv/bin/python train/orpo_train.py --config configs/orpo_mac.yaml
```

Mac configs: `configs/orpo_mac.yaml` (full run) and `configs/orpo_smoke_mac.yaml`
(20-step validation). Both set `load_in_4bit: false` and `device_map: mps`.
`configs/orpo_smoke.yaml` is CUDA-only and will fail on a Mac.

macOS caps GPU-wired memory near 75% of unified memory; raise it for a long run
with `sudo sysctl iogpu.wired_limit_mb=40960` (48 GB machine).

## Data Rules

- **Never commit or log any file under `/data/mimic/`** — HIPAA-regulated patient data.
- All training data paths must be loaded from `.env`, never hardcoded in source.
- DrugBank vocab lives at `/data/drugbank_vocabulary.csv`.

## Output Schema

Every model output must be valid JSON conforming to:

```json
{
  "diagnoses": [],
  "medications": [{"name": "", "dose": "", "freq": "", "route": ""}],
  "procedures": [],
  "discharge_instructions": "",
  "confidence_flags": []
}
```

## Evaluation Targets

| Metric | Target |
|---|---|
| Drug Entity Error Rate | ≤ 2% |
| HHEM (hallucination) | ≥ 0.80 |
| BERTScore | ≥ 0.88 |
| ROUGE-L | ≥ 0.42 |
| FactScore | ≥ 0.75 |
| GPT-4o Preference | ≥ 70% |
| Latency | ≤ 8 sec |
