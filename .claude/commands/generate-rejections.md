Generate a batch of adversarial rejected summaries for ORPO preference training.

Arguments (parse from user message if provided, otherwise use defaults):
- `--batch_size` (default: 500)
- `--failure_class`: A, B, C, D, E, or random (default: random)
  - A: Wrong medication dose or route
  - B: Fabricated diagnoses not in the note
  - C: Missing required schema fields
  - D: Hallucinated procedures
  - E: Truncated/incomplete response

Steps:
1. Load `--batch_size` records from `data/chosen_summaries.jsonl`
2. For each record, select failure class (per `--failure_class` arg, or sample weighted-randomly if "random")
3. Call Anthropic API using `claude-haiku-4-5-20251001` with the appropriate adversarial prompt for the chosen failure class
4. Validate every generated output parses as valid JSON matching the schema:
   `{ "diagnoses": [], "medications": [{"name","dose","freq","route"}], "procedures": [], "discharge_instructions": "", "confidence_flags": [] }`
5. Discard and retry (up to 2 times) any record that fails JSON validation
6. Append valid pairs to `data/preference_dataset.jsonl` as `{"prompt": ..., "chosen": ..., "rejected": ...}`
7. Log total input tokens, output tokens, and estimated cost to `logs/api_cost.txt` with a timestamp
8. Print a summary: records requested, records written, records discarded, total API cost
