# Final Agent quality gates

The production path is web research only. ChromaDB, sentence-transformers and
local knowledge packs are not runtime dependencies; the search cache is only a
short-lived request cache.

## Claim-Evidence contract

Search results are fetched, converted to validated `Evidence`, bound to
`ReportClaim`, and checked before a formal report is emitted. A claim can be
`entailed`, `contradicted`, or `unknown`. Unknown evidence IDs, uncited external
numbers, and unsupported claims are downgraded to an evidence-insufficient
report.

The independent judge is optional. Configure `EVAL_JUDGE_API_KEY`,
`EVAL_JUDGE_MODEL`, and optionally `EVAL_JUDGE_BASE_URL` to enable it. Without
those variables the deterministic rules still run, while the entailment metric
is explicitly reported as `unavailable`.

## Reproducible measurement

Run the 10-case online baseline with:

```powershell
python -m eval.online_eval --runs 3
```

Configure `LLM_INPUT_COST_PER_1K` and `LLM_OUTPUT_COST_PER_1K` to expose cost.
The API reports token counts, search/fetch calls, retries, failures, latency,
and an explicitly labelled equal-split per-dimension cost estimate.

The current baseline and four-version ablation outputs live under
`eval/results/`. Authentication, PostgreSQL checkpoints, and distributed
queues are documented migration work, not hidden claims of the internship
demo.
