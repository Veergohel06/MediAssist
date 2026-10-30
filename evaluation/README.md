# MediAssist Evaluation Framework

This directory contains a version-controlled experiment harness separate from production retrieval and generation code.

## Dataset

`dataset.jsonl` contains synthetic, public-source-grounded scenarios. Each row records:

- stable evaluation ID and question;
- relevant passage IDs and optional graded relevance;
- expected behavior and answer criteria;
- evidence sufficiency and expected abstention;
- expected safety behavior and escalation requirement.

The initial dataset is a deterministic smoke set. Larger datasets should be reviewed by a named clinical-content reviewer and approved through a dataset version change. Do not add private patient information.

## Commands

From the repository root:

```powershell
# CI-safe deterministic smoke test; does not call external APIs
python evaluation/run_eval.py --mode mock

# API-backed retrieval experiments, opt-in
python evaluation/run_eval.py --mode keyword
python evaluation/run_eval.py --mode semantic
python evaluation/run_eval.py --mode hybrid
python evaluation/run_eval.py --mode hybrid_reranked

# Full generation capture, opt-in and quota-sensitive
python evaluation/run_eval.py --mode full_generation
```

Each run writes raw JSONL predictions under `predictions/`, machine-readable metrics under `reports/*.json`, and a Markdown summary under `reports/*.md`. Reports record the dataset hash, configured models, index, mode, failure count, and whether the run was mocked.

## Metrics and limitations

Retrieval reports compute Precision@5, Recall@5, MRR, nDCG@5, latency p50/p95, and retrieval failures. Citation validity and abstention metrics are computed only when the prediction records contain those fields. Full-generation answer relevance, factual support, token usage, cost, and memory are `null` unless measured or reviewed; automated structure is not ground truth.

Safety behavior must be reviewed against the deterministic safety-rule tests and all escalation cases. Structured citations improve provenance auditing but do not establish medical factual accuracy. API-backed runs can be affected by provider quota, model drift, network latency, and embedding/index version changes.
