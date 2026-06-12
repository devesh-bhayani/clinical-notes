# Clinical Note Summarizer

[![CI](https://github.com/devesh-bhayani/clinical-notes/actions/workflows/ci.yml/badge.svg)](https://github.com/devesh-bhayani/clinical-notes/actions/workflows/ci.yml)

Hallucination-resistant summarization of unstructured clinical discharge notes
into validated, structured JSON. A two-stage QLoRA + ORPO fine-tune of
**BioMistral-7B**, served behind a FastAPI endpoint with a DrugBank medication
guardrail that flags any drug name the model invents.

> ⚠️ **Research project.** Not for clinical use without regulatory approval. See
> [docs/model_card.md](docs/model_card.md).

---

## How it works

```
MIMIC-IV notes ──▶ data/pipeline.py ──▶ chosen summaries (regex + spaCy NER,
                                          DrugBank-validated)
                          │
                          ▼
              data/rejection_gen.py ──▶ ORPO preference pairs
                          │              (5 adversarial failure classes A–E)
                          ▼
              train/orpo_train.py ──▶ QLoRA adapter (4-bit NF4, r=32, α=64)
                          │
                          ▼
   api/main.py  ◀── api/inference.py (BioMistral backend)
        │
        ├── api/guardrail.py  (DrugBank NER validation → confidence_flags)
        └── eval/run_eval_suite.py (DEER, HHEM, BERTScore, ROUGE-L, FactScore…)
```

The inference layer is abstracted behind a `Summarizer` protocol with two
backends: the production **BioMistral** model and a GPU-free **stub** for tests,
CI, and local demos. The stub is opt-in only (`ALLOW_STUB_INFERENCE=1`) so a
misconfigured production deploy fails loudly (HTTP 503) instead of serving fakes.

## Quickstart

### A. Demo / development (no GPU, no data)

Runs the full API, UI, and evaluation against the **stub** backend and a small
committed sample DrugBank vocabulary.

```bash
pip install -r requirements-dev.txt        # lightweight deps (no torch/transformers)

make test            # 64 unit + integration tests
make synthetic       # generate a synthetic, PHI-free dataset under data/splits/
make eval-smoke      # run the eval suite end-to-end on synthetic data
make serve           # FastAPI on :8000 (stub backend)
make ui              # Streamlit UI on :8501  (needs `pip install streamlit`)
```

Or with Docker (stub backend, runs anywhere):

```bash
docker compose up --build      # API on :8000, UI on :8501
```

### B. Real training run (GPU)

Two data sources are supported. **Asclepius** is the zero-friction path: a
public, **DUA-free** corpus of 157k synthetic discharge summaries — no PhysioNet
credentialing, no HIPAA exposure.

```bash
pip install -r requirements.txt            # full ML/serving stack (CUDA box)
cp .env.example .env                        # set DRUGBANK_VOCAB_PATH (CC0 download)

python train/vram_test.py                              # preflight GPU check

# Build chosen summaries + ORPO splits from Asclepius (recommended):
python -m data.asclepius --output data/splits --limit 20000 --make-splits

# ── or, with a MIMIC-IV DUA, the medication-dense alternative: ──
# python -m data.pipeline --input "$MIMIC_DATA_DIR" --output data/splits --sample 10000
# python -m data.rejection_gen --input data/splits/chosen_summaries.jsonl \
#         --output data/splits/train_orpo.jsonl --batch_size 5000

python train/orpo_train.py --config configs/orpo_base.yaml      # produces the adapter
python eval/run_eval_suite.py --split test                      # gated pass/fail

CHECKPOINT_DIR=models/checkpoints uvicorn api.main:app --port 8000   # serve the real model
```

> **Note on Asclepius:** diagnoses, procedures, and instructions extract well;
> medications are frequently absent in the source notes, so the `medications`
> field is often empty (faithful to the note). The medication guardrail/DEER
> metric still apply, and the Class-A hallucination rejections still teach the
> model not to invent drugs. For a medication-dense corpus, prefer MIMIC-IV.

## Configuration

All paths and keys are read from `.env` (see [.env.example](.env.example)).

| Variable | Used by | Notes |
|---|---|---|
| `MIMIC_DATA_DIR` | data pipeline | MIMIC-IV notes (HIPAA — never committed) |
| `DRUGBANK_VOCAB_PATH` | guardrail, eval, pipeline | Full DrugBank vocabulary CSV |
| `SPLITS_DIR` | training, eval | Where JSONL splits live (default `data/splits`) |
| `BASE_MODEL_NAME` | training, serving | Default `BioMistral/BioMistral-7B` |
| `CHECKPOINT_DIR` | serving | Trained adapter dir; if set, real inference is used |
| `ALLOW_STUB_INFERENCE` | serving | `1` enables the stub backend (demos/CI only) |
| `ANTHROPIC_API_KEY` | rejection gen (optional) | LLM-assisted rejection synthesis |
| `OPENAI_API_KEY` | eval (optional) | Enables the GPT-4o preference gate |
| `WANDB_API_KEY` | training (optional) | Enables W&B logging |

## Make targets

| Target | Description |
|---|---|
| `make test` | Full unit + integration suite |
| `make test-cov` | Coverage report + enforce guardrail ≥ 90% |
| `make synthetic` | Generate a synthetic PHI-free dataset |
| `make eval-smoke` | Build synthetic data, then run the eval suite |
| `make compliance` | HIPAA PHI scan over tracked data/source files |
| `make serve` / `make ui` | Serve the API (stub) / Streamlit UI |
| `make train` | Launch ORPO training (GPU) |
| `make clean` | Remove caches and generated artifacts |

## Output schema

Every model response is validated against:

```json
{
  "diagnoses": [],
  "medications": [{"name": "", "dose": "", "freq": "", "route": ""}],
  "procedures": [],
  "discharge_instructions": "",
  "confidence_flags": []
}
```

`confidence_flags` carries guardrail warnings — e.g. a medication name not found
in DrugBank is flagged here rather than silently dropped.

## Evaluation targets

| Metric | Target | Computed by |
|---|---|---|
| Drug Entity Error Rate | ≤ 2% | local (DrugBank) |
| HHEM (hallucination) | ≥ 0.80 | Vectara model / lexical fallback |
| BERTScore F1 | ≥ 0.88 | `bert-score` (optional dep) |
| ROUGE-L | ≥ 0.42 | local |
| FactScore | ≥ 0.75 | local structured-fact precision |
| GPT-4o Preference | ≥ 0.70 | LLM judge (`OPENAI_API_KEY`) |
| Latency | ≤ 8 sec | serving |

Metrics whose backend isn't configured (BERTScore, GPT-4o) are reported as
`skipped` and never fabricate a score or fail the run.

## Data onboarding (expected formats)

- **Asclepius** (recommended, no DUA): pulled automatically by
  `python -m data.asclepius` from
  [`starmpcc/Asclepius-Synthetic-Clinical-Notes`](https://huggingface.co/datasets/starmpcc/Asclepius-Synthetic-Clinical-Notes)
  (requires the `datasets` library + network on first run).
- **MIMIC-IV notes** (`MIMIC_DATA_DIR`): a CSV with a note-text column
  (`text`/`note`/`TEXT`/`note_text`) and, ideally, a `category` column so the
  pipeline can filter to discharge summaries.
- **DrugBank vocabulary** (`DRUGBANK_VOCAB_PATH`): a CSV with a `Common name`
  column and an optional `Synonyms` column (`|`-delimited). A single-column /
  headerless name list also works. See
  [data/drugbank_vocabulary.sample.csv](data/drugbank_vocabulary.sample.csv).
- **Demographics for the bias audit** (optional): per-record `age`,
  `ethnicity`/`race`, and `admission_type` fields under a `metadata` key.

## Data & HIPAA rules

- **Never commit or log anything under `data/mimic/`** — it is gitignored.
- All data paths come from `.env`; nothing is hardcoded in source.
- `scripts/check_no_mimic_data.py` scans for PHI signatures; CI runs it over all
  tracked data/source files on every push and PR.

## Testing & CI

`make test` runs 64 tests covering the guardrail, API, eval metrics + gating, the
data layer, an end-to-end synthetic integration test, and the compliance checker.
CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)) runs the suite on
Python 3.11/3.12, enforces ≥ 90% guardrail coverage, runs an eval smoke test on
synthetic data, and runs the HIPAA scan — all without GPU or ML dependencies.

## Repository layout

```
api/        FastAPI app, inference backends, DrugBank guardrail
data/       MIMIC extraction pipeline, rejection generation, bias audit
train/      ORPO/QLoRA trainer, VRAM preflight
eval/       metrics + gated evaluation suite
ui/         Streamlit demo
scripts/    synthetic data generator, HIPAA compliance checker
configs/    orpo_base.yaml (hyperparameters)
docs/       model card
```
