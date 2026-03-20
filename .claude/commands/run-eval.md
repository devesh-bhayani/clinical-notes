Run the full evaluation suite on the held-out test split.

1. Execute: `python eval/run_eval_suite.py --split test`
2. Parse the output JSON from `eval/results/latest.json`
3. Compare all three metrics against targets:
   - Drug Entity Error Rate ≤ 2%
   - HHEM ≥ 0.80
   - BERTScore ≥ 0.88
4. If any metric misses its target, open a GitHub issue titled `[Eval Failure] <metric> missed target by <delta>` with the metric name, actual value, target, and delta in the body
5. Print a pass/fail summary table to terminal in this format:

| Metric | Target | Actual | Status |
|---|---|---|---|
| Drug Error Rate | ≤ 2% | X% | PASS/FAIL |
| HHEM | ≥ 0.80 | X.XX | PASS/FAIL |
| BERTScore | ≥ 0.88 | X.XX | PASS/FAIL |
