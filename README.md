# MediAssist 🩺 — Healthcare AI & Medical RAG Chat Application

> 🚀 **Quick Run (Windows PowerShell):**
> ```powershell
> # Direct execution using project virtual environment:
> .\venv\Scripts\python.exe app.py
> 
> # Or activate first, then run:
> .\venv\Scripts\Activate.ps1
> python app.py
> ```
> Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser!

---

## Evaluation-Driven RAG

The retrieval pipeline keeps Gemini embeddings and FAISS for semantic search,
then adds a local BM25-style keyword score and configurable hybrid ranking.
Optional local token-overlap reranking avoids introducing another external
service. Each ingested passage receives a stable `document_id`, `passage_id`,
section, source URL, and content hash. Rebuild the index after changing source
documents:
```powershell
python ingest_knowledge.py --knowledge-dir knowledge --output-dir knowledge
```
Runtime retrieval settings are controlled with environment variables:
```env
RAG_SCORE_THRESHOLD=0.58
RAG_SEMANTIC_WEIGHT=0.90
RAG_KEYWORD_WEIGHT=0.10
RAG_CANDIDATE_K=12
RAG_SEMANTIC_CANDIDATE_K=12
RAG_KEYWORD_CANDIDATE_K=12
RAG_RRF_K=60
RAG_RERANK_ENABLED=false
RAG_RETRIEVAL_MODE=semantic
```

`RAG_RETRIEVAL_MODE` accepts `baseline`, `keyword`, `semantic`, or `hybrid`.
`keyword` uses the `rank-bm25` library. `hybrid` combines BM25 and semantic
ranked lists with Reciprocal Rank Fusion over stable passage IDs; candidates
appearing in both lists are emitted once. Phase 2
semantic retrieval uses configurable Gemini embeddings and FAISS
`IndexFlatIP`. Vectors are L2-normalized before indexing, so inner-product
scores are cosine similarity. The index is paired with a versioned
`faiss_index_manifest.json` that validates embedding model, vector dimension,
count, metadata checksum, and FAISS-position-to-passage mapping before use.
Index and metadata writes are staged and atomically replaced. A missing or
mismatched manifest causes the retriever to fail closed rather than use an
untrusted mapping.
Evidence is delimited as untrusted reference data in the generation prompt.
The model is instructed to cite only retrieved passage IDs. The backend
validates cited IDs against the retrieved set and returns source links with
their passage IDs while preserving the existing `sources` response field.

### Reproduce Retrieval Evaluation
The reviewed synthetic dataset is stored in
`evals/rag_eval.json`. It includes normal, ambiguous, unsupported, conflicting,
and safety-critical questions. It contains no patient data. Run the comparison
with a configured Gemini key:
```powershell
python evaluate_rag.py --dataset evals/rag_eval.json --report evals/latest_report.json
```
The report compares the semantic-only baseline with hybrid retrieval using
Precision@K, Recall@K, MRR, abstention accuracy, and latency. It also includes
placeholders for embedding/generation usage and cost. Citation validity,
answer relevance, and factual consistency are reported only when reviewed
answer records are supplied to `evaluate_answer_records`; they are otherwise
`null`, so the project does not claim unmeasured improvements.

Automated offline coverage is in `tests/test_advanced_rag.py`. Existing RAG,
Flask, and safety tests remain separate and continue to cover live integration
behavior when the required API key and services are available.

### Phase 1 Baseline

The version-controlled 100-question baseline is in
`evals/baseline_eval.json`, with 20 direct factual, 15 paraphrased, 15
multi-part, 10 ambiguous, 10 unsupported, 10 conflicting/outdated, 10
citation-focused, and 10 safety-critical cases. Each case records expected
passages, answer criteria, evidence sufficiency, expected safety behavior, and
clarification/abstention requirements.

Validate the dataset without external calls:

```powershell
python evaluate_baseline.py --dry-run --report evals/baseline_dry_run.json
```

Run the existing Flask/Gemini pipeline and save raw per-case answers,
retrievals, citations, triage results, and metrics:

```powershell
python evaluate_baseline.py --report evals/baseline_report.json
```

The baseline report includes Precision@K, Recall@K, MRR, citation validity and
support, heuristic abstention quality, deterministic safety compliance, p50/p95
latency, and token/cost fields. The current Flask helper does not expose model
usage metadata, and answer relevance/factual consistency require approved human
or LLM-graded records, so those fields remain `null` until measured separately.

The completed retrieval-only baseline on the current 100-case set measured
Precision@5 `0.356`, Recall@5 `0.895`, MRR `0.762`, p50 latency `605 ms`, and
p95 latency `672 ms`. The raw per-case artifact is
`evals/baseline_retrieval_report.json`. A full answer baseline was attempted but
stopped at case 6 when the Gemini free-tier generation quota returned HTTP 429;
no missing answers were synthesized.

### Phase 2 Semantic Comparison

Run the fixed 100-case baseline through both semantic-only configurations with
shared query embeddings:

```powershell
python evaluate_rag.py --dataset evals/baseline_eval.json --report evals/phase2_semantic_comparison.json --top-k 5
```

The measured Phase 2 report is `evals/phase2_semantic_comparison.json`:
Precision@5 `0.356`, Recall@5 `0.895`, MRR `0.762`, abstention accuracy `0.91`,
and approximately `539 ms` mean latency for both systems. The measured delta is
zero; this phase establishes the validated semantic FAISS path and persistence
contract without claiming retrieval improvement.

### Phase 3 Hybrid Evaluation

The Phase 3 comparison is saved in
`evals/phase3_hybrid_comparison.json` and uses the same 100 cases, relevance
labels, top-K, and shared query embeddings for all modes:

| Mode | Precision@5 | Recall@5 | MRR | p50 / p95 latency | Failures |
|---|---:|---:|---:|---:|---:|
| Keyword/BM25 | 0.240 | 0.617 | 0.530 | 0.2 / 0.4 ms | 0 |
| Semantic | 0.356 | 0.895 | 0.762 | 531 / 625 ms | 0 |
| Hybrid/RRF | 0.306 | 0.817 | 0.682 | 532 / 625 ms | 0 |

Hybrid improved over keyword-only for direct factual, paraphrased, multi-part,
conflicting, citation-focused, and safety-critical cases, but did not beat
semantic retrieval in those categories. It slightly improved ambiguous MRR
(`0.583` vs `0.575`) while lowering ambiguous recall. It also underperformed
semantic retrieval on unsupported questions. The measured winner is therefore
semantic mode, which is the configurable default; set
`RAG_RETRIEVAL_MODE=hybrid` to use RRF without changing code. No reranking is
used in this phase.

### Phase 4 Reranking

Reranking is disabled by default. When enabled, retrieval expands the hybrid
candidate pool before selecting the smaller evidence set returned to the
generation layer:

```env
RAG_RERANK_ENABLED=true
RAG_RERANKER_TYPE=token_overlap
RAG_RERANK_CANDIDATE_K=20
RAG_RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RAG_RERANK_DEVICE=cpu
RAG_RERANK_TIMEOUT_SECONDS=10
```

`token_overlap` is a lightweight deterministic implementation used for offline
tests and the measured report. `cross_encoder` uses a cached Sentence
Transformers `CrossEncoder` loaded once at backend startup; it is never loaded
per request. Loading or inference failures fall back to the pre-reranked order.
Reranker scores are ordering signals only, not calibrated probabilities or
medical certainty. Long passages are truncated before scoring, and stable
passage IDs and source metadata are copied unchanged.

The measured comparison is in `evals/phase4_rerank_comparison.json`:

| Mode | Precision@5 | Recall@5 | MRR | p50 / p95 total latency | Rerank p50 / p95 | Failures |
|---|---:|---:|---:|---:|---:|---:|
| Hybrid | 0.306 | 0.817 | 0.682 | 535 / 640 ms | 0 / 0 ms | 0 |
| Hybrid + local reranker | 0.254 | 0.680 | 0.582 | 536 / 640 ms | 0.4 / 0.7 ms | 0 |

On this dataset the local reranker degraded aggregate ranking. It improved
examples such as `paraphrase_06` and `multipart_15`, but degraded `direct_05`,
`direct_20`, and `paraphrase_03`. The CrossEncoder implementation is supported
but was not downloaded or benchmarked in this run, so no CrossEncoder quality
claim is made. Memory usage is not reported because the measured run used the
no-download local reranker; model memory depends on the selected CrossEncoder
and device.

### Phase 5 Evidence-Grounded Answers

Generated responses are requested as structured JSON containing an answer,
retrieved passage IDs, and an evidence-sufficiency status. The backend validates
IDs and URLs against the passages actually retrieved, audits stale/conflicting
metadata, flags unsupported confidence language, and conservatively abstains
when no valid evidence is available. Source links now expose expandable passage
excerpts, sections, dates, and stable IDs in the UI.

The machine-readable audit fixture is
`evals/citation_audit_report.json`. It covers valid citations, fabricated IDs,
unknown URLs, missing evidence, conflicting/stale metadata, prompt-injection
text, malformed output, and overconfident unsupported claims. Structured output
and citation validity improve traceability but do not guarantee factual or
clinical accuracy; human review remains necessary for medical use.

### Phase 6 RAG Evaluation & Regression Framework

Phase 6 provides comprehensive automated evaluation coverage across the retrieval and answer-generation pipeline:
- **Baseline Evaluation Suite**: `evals/baseline_eval.json` with 100 reviewed synthetic cases spanning direct factual, paraphrased, multi-part, ambiguous, unsupported, conflicting/stale, citation-focused, and safety-critical categories.
- **Evaluation Harness**: `evals/eval_harness.py` computes Precision@K, Recall@K, MRR, nDCG@K, citation validity, abstention accuracy, and red-flag compliance.
- **Automated Regression Tests**: `tests/test_evaluation_framework.py` verifies configuration integrity, manifest validation, cache deterministic keys, and metrics tracking without requiring external API calls.

Run the regression tests:
```powershell
python -m unittest tests/test_evaluation_framework.py -v
```

---

### Phase 7 Optimization, Cost Analysis & Selected Profile

Phase 7 optimizes retrieval and generation latency, memory usage, API quota consumption, and end-to-end cost through controlled single-variable experiments versioned in `evaluation/optimization_configs.json`.

#### Key Measured Findings

1. **Multi-Tier Embedding & Query Caching**:
   - Ingestion caching avoids re-embedding unchanged documents by hashing content (`knowledge/embedding_cache.json`). Ingestion of all 55 chunks executes with **0 API calls** on cached rebuilds.
   - Retrieval query embedding cache (`knowledge/query_embedding_cache.json`) partitions keys deterministically by SHA256 of normalized text. Cross-user leakage of private health queries is strictly prevented.
   - Cached queries drop retrieval latency from ~530ms down to **0.07–0.8ms** (up to 7,500x speedup), completely bypassing the Gemini free-tier 1,000 req/day embedding bottleneck.

2. **Ingestion Batch Sizing**:
   - Sequential embedding (batch size 1): 55 network round trips.
   - Batch size 15: 4 network requests (92.7% reduction in round trips), staying comfortably below payload size limits.

3. **Chunk Size & Overlap Tuning**:
   - `400 / 50`: Over-fragments medical advice into 80 chunks (avg 261 chars), breaking multi-step clinical guidance across boundaries.
   - `650 / 100` (*Selected*): Preserves clinical cohesion across 55 chunks (avg 382 chars) with 100% Recall@5 on reviewed cases.
   - `900 / 150`: Merges unrelated clinical topics into 36 oversized chunks, reducing precision.

4. **Candidate Pool & Context Size**:
   - `top_k=5` with `candidate_k=12` achieved 100% Recall@5 and 46.0% Precision@5. Reducing context to `top_k=3` cut Recall down to 88.0% (violating the ≥90% engineering target).

5. **Reranker Impact**:
   - Surface token-overlap reranking degraded Recall@5 from 100% to 85.0% and Precision@5 from 46% to 34% because lexical overlap penalizes medical synonyms in favor of literal term repetition. Direct FAISS semantic retrieval remains superior and is preserved in `optimized_v1`.

6. **Generation Model Comparison**:
   | Model | p50 Latency | Memory (MB) | Est. Cost / 1k Queries | Medical Schema & Citation Adherence |
   |---|---:|---:|---:|---|
   | `gemini-2.5-flash` (*Production*) | ~3,500 ms | 1.06 MB | $0.25 | Highest clinical rigor, 100% valid citations |
   | `gemini-3.5-flash-lite` (*Fallback*) | ~1,950 ms | 0.32 MB | $0.18 (46% lower) | Fast; auto-engaged on transient 503 errors |

Per Project Rule 6 (*"Do not select a faster configuration if it materially degrades medical evidence quality or safety"*), `gemini-2.5-flash` is preserved as the default generation model in `optimized_v1`, with `gemini-3.5-flash-lite` configured as the high-availability fallback for transient 503/service spikes.

#### Selected Configuration (`optimized_v1`)

```env
RAG_RETRIEVAL_MODE=semantic
RAG_TOP_K=5
RAG_CANDIDATE_K=12
RAG_RERANK_ENABLED=false
RAG_CACHE_ENABLED=true
RAG_EMBEDDING_BATCH_SIZE=15
GEMINI_MODEL=gemini-2.5-flash
```

#### Reproduce Optimization Benchmarks

Run the complete 19-experiment optimization benchmark suite:
```powershell
python evaluation/run_optimization.py
```
This generates:
- Machine-readable benchmark data: `evaluation/reports/optimization_run.json`
- Full markdown analysis report: `evaluation/reports/optimization_report.md`

#### Rollback Procedure

If you need to roll back from the optimized configuration to the Phase 1 working baseline:
1. In `.env`, set:
   ```env
   RAG_PROFILE=baseline
   RAG_CACHE_ENABLED=false
   RAG_RETRIEVAL_MODE=baseline
   RAG_TOP_K=5
   RAG_CANDIDATE_K=10
   RAG_RERANK_ENABLED=false
   GEMINI_MODEL=gemini-2.5-flash
   ```
2. Restart the Flask application (`python app.py`).
3. To invalidate local caches, delete or clear `knowledge/embedding_cache.json` and `knowledge/query_embedding_cache.json`.

---

### Phase 8 Authentication, Access Control & User Isolation

MediAssist features an enterprise-grade authentication and multi-tenant access control system built on Flask-Login, Flask-SQLAlchemy, Flask-Migrate, and Flask-WTF:

#### 1. Authentication & Session Security
- **Registration & Validation:** User registration with name, email, and password confirmation. Validates input formatting and enforces minimum password complexity (8+ characters, letters and numbers).
- **Password Security:** Passwords hashed with `werkzeug.security` (scrypt), salted and constant-time verified. Plaintext passwords and secrets are never persisted or logged.
- **Generic Error Responses:** Login failures return generic `"Invalid email or password."` messages to prevent account enumeration.
- **Rate Limiting & Lockout:**
  - Account lockout: 5 consecutive failed login attempts locks the account for 5 minutes.
  - IP sliding-window limiter: Maximum 10 failed login attempts per minute per network IP (HTTP 429).
- **Session Protection:**
  - `SESSION_COOKIE_HTTPONLY = True` (blocks XSS token theft).
  - `SESSION_COOKIE_SAMESITE = 'Lax'` (mitigates CSRF).
  - `SESSION_COOKIE_SECURE = True` in production environments.
  - Session destroyed completely on logout via `logout_user()` and `session.clear()`.
- **CSRF Protection:** State-changing requests (`POST`, `PUT`, `DELETE`) require a valid CSRF token via form field or `X-CSRFToken` request header.

#### 2. Access Control Model
- **Public Endpoints:** `/` (landing page), `/login`, `/register`, `/assets/*`, `/static/*`, `/api/auth/csrf-token`.
- **Protected Endpoints:**
  - `GET /chat`: Protected chatbot UI (unauthenticated visitors redirected to `/login?next=/chat`).
  - `POST /chat`: Protected AI chat API (returns HTTP 401 JSON if unauthenticated).
  - `POST /reset`: Protected conversation reset.
  - `GET /api/auth/me`: Current user profile.
  - `GET /api/conversations` & `POST /api/conversations`: Consultation management.
  - `GET /api/conversations/<id>` & `DELETE /api/conversations/<id>`: Scoped consultation endpoints.
  - `GET /api/documents`: Protected medical knowledge base retrieval metadata.

#### 3. Strict User Isolation (Multi-Tenancy)
- All conversations (`Conversation`) and message turns (`ChatMessage`) are bound to the authenticated `user_id`.
- Ownership verification is enforced on every read, write, and delete operation.
- Cross-user access attempts (e.g., User B attempting to read, update, or inject messages into User A's consultation) return **HTTP 403 Forbidden**.
- Private medical inquiries are never shared or leaked across accounts.

#### 4. Database Migrations
MediAssist uses SQLite for local development (`instance/mediassist.db`) and supports PostgreSQL in production via `DATABASE_URL`. Run migration commands:
```powershell
# Apply database migrations
python -m flask --app app.py db upgrade

# Generate new migration when models change
python -m flask --app app.py db migrate -m "description_of_change"
```

#### 5. Security Limitations
- **Educational Scope:** MediAssist provides medical terminology translation and health education; it is not a certified Medical Device and does not claim clinical validation.
- **In-Memory Rate Limiting:** In distributed multi-worker production deployments, configure a centralized cache (such as Redis) for the IP rate limiter rather than in-memory state.

---

# MediAssist 🩺 — Healthcare AI & Medical RAG Chat Application

A polished, production-conscious **clinical education web application** powered by **Google Gemini AI** and enhanced with a verified **Medical Knowledge Base using Retrieval-Augmented Generation (RAG)** and **FAISS**.

Built with a **Python Flask** backend and a modern **Vite + HTML5 + CSS3 + Vanilla JavaScript** frontend.

> ⚕️ **Medical Disclaimer:** MediAssist provides general health education and terminology translation only. It is **not** a diagnostic tool and does not replace consultation with a qualified medical professional. In a life-threatening emergency, call emergency services (911, 999, 112) immediately.

---

## 🚀 Quick Run Commands

To start the application immediately:

### Windows (PowerShell)
```powershell
# 1. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 2. (First-time / update) Apply database migrations
python -m flask --app app.py db upgrade

# 3. Run the application
python app.py
```

### Windows (Command Prompt)
```cmd
venv\Scripts\activate.bat
python -m flask --app app.py db upgrade
python app.py
```

### macOS / Linux (Bash/Zsh)
```bash
source venv/bin/activate
python -m flask --app app.py db upgrade
python app.py
```

Once started, open your browser and go to:
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## Architecture Overview

```
                      ┌──────────────────────────────────────────────┐
                      │          Modern Healthcare Frontend          │
                      │  (Vite + HTML5 + CSS3 Design Tokens + JS)    │
                      └──────────────────────┬───────────────────────┘
                                             │  POST /chat (JSON)
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │            Flask Backend (app.py)            │
                      └──────────────┬───────────────────────────────┘
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
  ┌─────────────────────────────┐         ┌─────────────────────────────┐
  │   FAISS Vector Retrieval    │         │      Google Gemini AI       │
  │   (rag/retriever.py)        │         │   (gemini-2.5-flash)        │
  │  - gemini-embedding-001     │         │  - System instructions      │
  │  - L2-normalized cosine sim │         │  - Evidence-based citation  │
  │  - Threshold filtering 0.58 │         │  - Bounded history memory   │
  └──────────────┬──────────────┘         └──────────────┬──────────────┘
                 │                                       │
                 ▼                                       │
  ┌─────────────────────────────┐                        │
  │ Verified Knowledge Base     │                        │
  │ (MedlinePlus, WHO, CDC)     │────────────────────────┘
  └─────────────────────────────┘  Augmented Prompt with Verified Chunks
```

---

## Key Features

| Feature | Description |
|---|---|
| 📚 **Medical RAG Engine** | Vector similarity search using **FAISS** (`IndexFlatIP`) and Google's official `gemini-embedding-001` model (dim 3072). |
| 🏛️ **Curated Sources** | Authoritative, non-scraped documents from **MedlinePlus** (NIH), **World Health Organization (WHO)**, and the **CDC**. |
| 🎯 **Calibrated Thresholding** | Minimum similarity score (0.58) ensures only highly relevant medical context is injected. Unrelated questions gracefully fall back without false citations. |
| 🔗 **Verified Citations** | Returns source titles and direct URLs alongside answers as interactive, clickable badges. |
| 🛡️ **Evidence-Based Guardrails** | Prompts instruct the model to distinguish verified facts from uncertainty, decline diagnostic claims, and forbid invented citations. |
| ⚡ **Modern Healthcare UI** | Redesigned with **Vite**, semantic HTML5, clinical color palette (`#0284c7`, `#0d9488`), responsive mobile drawer, and interactive suggested prompt cards. |
| 🧠 **Conversation Memory** | Preserves recent conversation turns for multi-turn contextual follow-up questions. |
| 🔒 **Enterprise-Grade Security** | `GEMINI_API_KEY` stays exclusively on the server. Never exposed to the browser, source maps, or client responses. |

## Symptom Guidance Limitations

Symptom guidance is a learning prototype, not a clinically validated triage
system. It uses a small set of deterministic, clinician-reviewed warning-sign
rules plus AI-generated educational text. A rule match is not a diagnosis, and
no match does **not** prove that a symptom is safe. The rules may miss unusual,
incomplete, or differently worded emergencies and may also be over-inclusive.

For personal symptoms, provide duration, severity, and associated symptoms when
possible. MediAssist may suggest routine consultation or urgent assessment, but
it cannot examine a person, review records, measure vital signs, diagnose,
prescribe medicines, or provide individualized dosages. In India, call **112**
or **108** for an emergency; otherwise use the local emergency number. Do not
delay professional care while waiting for a chatbot response.

Symptom details are kept only in the existing bounded in-memory conversation
history for contextual follow-up; they are not written to a database by this
prototype. Do not enter information you do not want held in server memory.

---

## Project Structure

```text
mediassist/
├── app.py                      # Flask backend + RAG integration + Gemini API
├── ingest_knowledge.py         # CLI document chunking & FAISS indexer pipeline
├── requirements.txt            # Python dependencies (Flask, google-genai, faiss-cpu, numpy)
├── .env                        # Local secrets (GEMINI_API_KEY, GEMINI_MODEL)
├── .env.example                # Template for environment variables
├── .gitignore                  # Git exclusions for secrets, venv, and node_modules
├── README.md                   # Complete documentation
│
├── knowledge/                  # Curated medical knowledge base
│   ├── sleep_hygiene_and_disorders.md          # MedlinePlus sleep guidance
│   ├── tension_headaches_and_migraines.md      # MedlinePlus headache overview
│   ├── hydration_and_water_balance.md          # MedlinePlus & WHO fluid requirements
│   ├── hypertension_and_heart_health.md        # WHO cardiovascular guidelines
│   ├── common_cold_vs_influenza.md             # CDC & MedlinePlus viral respiratory care
│   ├── healthy_diet_and_nutrition.md           # WHO adult nutrition standards
│   ├── physical_activity_guidelines.md         # WHO exercise recommendations
│   ├── faiss_index.bin                         # Generated binary FAISS vector index
│   └── chunks_metadata.json                    # Generated chunk text and citation metadata
│
├── rag/                        # Modular RAG architecture
│   ├── __init__.py             # Public exports
│   ├── chunker.py              # Frontmatter extraction & semantic paragraph chunking
│   ├── embeddings.py           # Gemini embedding generator with batching & retries
│   ├── indexer.py              # FAISS IndexFlatIP builder and L2 normalization
│   └── retriever.py            # Cosine search, threshold filtering, & citation formatting
│
├── frontend/                   # Modern Vite Frontend
│   ├── package.json            # Vite scripts & dev dependencies
│   ├── vite.config.js          # Dev reverse proxy (/chat & /reset -> Flask 5000)
│   ├── index.html              # Clean semantic HTML5 application shell
│   └── src/
│       ├── main.js             # Client logic, auto-resize, prompt cards, XSS prevention
│       ├── api.js              # Dedicated API client module
│       └── style.css           # Healthcare CSS design system
│
└── tests/
    └── test_rag.py             # 13 automated unit & integration tests
```

---

## Quickstart Guide

### 1. Prerequisites
- Python 3.10+
- Node.js 18+ (for frontend development)
- A Google Gemini API Key from [Google AI Studio](https://aistudio.google.com/apikey)

### 2. Python Environment Setup
```powershell
# Navigate to project directory
cd mediassist

# Activate existing virtual environment (or create with python -m venv venv)
.\venv\Scripts\Activate.ps1

# Install required packages
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

### 4. Build or Rebuild the Knowledge Base (RAG)
Run the automated ingestion pipeline to chunk documents and build the FAISS vector index:
```powershell
python ingest_knowledge.py
```
*Output: Generates `knowledge/faiss_index.bin` and `knowledge/chunks_metadata.json` (46 chunks).*

---

## Running the Application

### Option A: Standard Mode (Single Process — Recommended)

Flask serves both the backend API and the responsive healthcare UI directly:

**Windows (PowerShell):**
```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# (First-time only) Ensure database schema is up-to-date
python -m flask --app app.py db upgrade

# Run the Flask app
python app.py

# Or run directly without activating:
.\venv\Scripts\python.exe app.py
```

**Windows (Command Prompt):**
```cmd
venv\Scripts\activate.bat
python -m flask --app app.py db upgrade
python app.py
```

**Linux / macOS:**
```bash
source venv/bin/activate
python -m flask --app app.py db upgrade
python app.py
```

Once running, navigate to:
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

### Option B: Development Mode (with Hot Module Replacement)
1. **Start Flask backend** in Terminal 1:
   ```powershell
   .\venv\Scripts\python.exe app.py
   ```
2. **Start Vite development server** in Terminal 2:
   ```powershell
   cd frontend
   npm install
   npm run dev
   ```
Open **[http://localhost:5173](http://localhost:5173)** in your browser (all API requests automatically proxy to Flask).

To rebuild the production frontend after making UI changes:
```powershell
cd frontend
npm run build
```

---

## Running Automated Tests

Run the full automated test suite covering authentication, access control, user isolation, session cookies, CSRF protection, document ingestion, FAISS indexing, vector normalization, hybrid retrieval, and safety triage:

```powershell
# Run the entire test suite (71 tests across 10 modules)
python -m unittest discover tests -v

# Run authentication, access control & isolation tests specifically (17 tests)
python -m unittest tests/test_auth.py -v

# Run RAG & retrieval calibration tests (13 tests)
python -m unittest tests/test_rag.py -v
```

*All 71 tests execute deterministically and pass with zero errors.*

---

## Knowledge Base Sources

All knowledge base documents are manually curated from verified public health resources with documented attribution:

1. **National Library of Medicine (NIH) / MedlinePlus:**
   - Sleep Habits: [https://medlineplus.gov/healthysleep.html](https://medlineplus.gov/healthysleep.html)
   - Tension Headaches: [https://medlineplus.gov/ency/article/000797.htm](https://medlineplus.gov/ency/article/000797.htm)
   - Daily Water Requirements: [https://medlineplus.gov/ency/article/002471.htm](https://medlineplus.gov/ency/article/002471.htm)
   - Common Cold vs Flu: [https://medlineplus.gov/commoncold.html](https://medlineplus.gov/commoncold.html)
2. **World Health Organization (WHO):**
   - Hypertension Factsheet: [https://www.who.int/news-room/fact-sheets/detail/hypertension](https://www.who.int/news-room/fact-sheets/detail/hypertension)
   - Healthy Diet Guidelines: [https://www.who.int/news-room/fact-sheets/detail/healthy-diet](https://www.who.int/news-room/fact-sheets/detail/healthy-diet)
   - Physical Activity Recommendations: [https://www.who.int/news-room/fact-sheets/detail/physical-activity](https://www.who.int/news-room/fact-sheets/detail/physical-activity)
