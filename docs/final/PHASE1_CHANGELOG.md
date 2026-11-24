# Phase 1 — Stabilisation, defect fixes and re-measurement

Branch: `final-milestone` · Date: 25 Nov 2026

The post-Milestone-1 code audit found defects that made the system look
more capable than it was, or that slowed it down. Every fix below is pinned
by an automated test (`backend/tests/unit/test_regression_fixes.py`,
`backend/tests/integration/test_api.py`).

## 1. Clinical-safety defects fixed

| # | Defect (before) | Effect on the user | Fix (after) |
|---|---|---|---|
| S1 | When the LLM API failed, a **template "mock" answer** was silently returned with `status="answer"` | A fake answer could be shown as verified evidence | Explicit abstention: *"answer generator unavailable"*, with the error recorded. No unverified text is shown (`openrouter.py::llm_unavailable_answer`) |
| S2 | When the safety gate **abstained**, the LLM's synthesis was **still returned and displayed** | Text that failed verification still reached the clinician | On abstention the API returns no synthesis sections, only the reason and the retrieved sources for manual review (`app.py::persist_and_build_response`) |
| S3 | `HER2+`, `HR+`, `EGFR+`, `ALK+` were **never detected** (a regex `\b` cannot follow `+`) | Population filtering silently switched off for the most common notation | `(?!\w)` look-ahead (`clinical_intent.py`, `population_matcher.py`) |
| S4 | `her2-` matched inside "HER2-positive" / "HER2-low", so a HER2-positive query listed **HER2-positive as a rejected subgroup** | The LLM was told to ignore the very population asked about | Bare `HER2-` must not be followed by a word; a requested biomarker can never be rejected |
| S5 | The normalizer expanded the English word **"or" → "odds ratio"**, "os", and the "HR" in "HR+" → **"hazard ratio"** | The search query was corrupted | Ambiguous acronyms expand only when typed in UPPERCASE as a standalone token; one left-to-right pass, so nothing is expanded twice |
| S6 | Abstention read `hierarchy_rank` from the wrong object, so **every answer** got the warning *"All remaining evidence is Case Series or lower"* | Spurious low-quality warning on every answer | Read the rank set on the candidate |
| S7 | A single paper could be reported as **contradicting itself** | False conflict alerts | Pairs must come from different sources (`_first_distinct_pair`) |
| S8 | Any RCT was labelled **"Phase III"**; any text containing the word "recommendation" was ranked as a **guideline** | Evidence strength overstated | Phase III only when stated; phase II RCTs rank lower; a guideline needs an explicit guideline signal |
| S9 | Unauthenticated requests ran as a seeded **default clinician** account; the UI **silently re-logged in** with hard-coded credentials | Authentication was effectively off | No anonymous fallback (HTTP 401); the UI returns to the login screen; `GET /auth/me` restores a valid session |
| S10 | Traces recorded the model as `"mock-llm"` even when real models ran | The audit trail was wrong | Real model IDs, token usage and an error flag are stored per run |

## 1b. Retrieval-correctness defect: "hybrid" search was secretly dense-only

| # | Defect (before) | Evidence | Fix (after) |
|---|---|---|---|
| R1 | Query-side **BGE-M3 sparse vectors were always empty**, so the lexical channel returned nothing and "hybrid RRF retrieval" was really **dense-only**. Cause: with `use_fp16=True` the XLM-R encoder runs in float16 while the `sparse_linear` head stayed float32, so every sparse forward pass raised *"mat1 and mat2 must have the same dtype"*. FlagEmbedding's batch-size retry loop swallowed the error, and a local patch of `M3Embedder.encode` also dropped the `return_sparse` flag | Retrieval log: `Sparse terms: 0` on every query. The stored document vectors are fine (60–197 non-zero terms each) | Cast the heads to the encoder dtype; call `encode_single_device` with explicit flags. Queries now get 10–18 weighted terms. Dense vectors are unchanged (cosine 1.000 vs. the old path), so **no re-indexing was needed** |

## 2. Performance defects fixed

| # | Defect (before) | Fix (after) |
|---|---|---|
| P1 | Qdrant `should` filters used payload keys that are never written (`disease`, `biomarker`). Qdrant requires at least one `should` match, so **every filtered query returned 0 hits and was re-run** | Removed. Population constraints are applied by the population matcher |
| P2 | Dense and sparse searches ran **one after the other** | Run concurrently (`asyncio.gather`) |
| P3 | Hydration ran **~5 SQL queries per candidate** (~250 per query) | One batched 5-table join |
| P4 | The context assembler **re-embedded all 50 candidates on the GPU** for MMR on every query | Reuses the dense vectors already stored in Qdrant (one `retrieve` call) |
| P5 | MMR redundancy used pure-Python cosine loops, O(n²·k·1024) | One numpy similarity matrix, computed once |
| P6 | GPU inference ran on the async event loop, blocking the server; `gc.collect()` ran on every call | Runs in a worker thread behind a lock; per-call GC removed |

## 3. Engineering changes
- **Shared pipeline:** `modules/pipeline/query_pipeline.py` is now the only code path, used by both the API and the evaluation scripts. It has a `retrieval_mode` switch (`sparse | dense | hybrid | hybrid_rerank | full`) for the ablation study.
- **Citations resolve to real sources:** the API returns `evidence_sources` for E01…En (title, year, study type, section, reranker score, population-match reason, PMC URL). The UI shows clickable citation chips and a References panel.
- **Audit tab shows recorded data:** models, a per-stage latency breakdown, and per-candidate dense/sparse ranks, RRF, reranker and population scores. These were previously hard-coded labels.
- **Setup files:** `requirements.txt` and `.env.example` now match the real stack (torch, FlagEmbedding, httpx, OpenRouter settings).
- **Evaluation:** the old `scripts/run_eval.py` **fakes its results** (it always ranks the expected papers first). It is superseded by the real evaluation harness in `evals/`, which calls the live pipeline.

## 4. Tests
| | Before | After |
|---|---|---|
| Backend tests passing | 72 | 88 |
| Regression tests for the defects above | 0 | 16 |

## 5. Latency — measured, not estimated
Sources: `evals/results/latency_before.json` (52 historical traces from the pre-fix system) and `evals/results/latency_after_retrieval_only.json` (10 fixed oncology queries, warm GPU).

| Stage | Before p50 | Before p95 | After p50 | After p95 | Speed-up (p50) |
|---|---|---|---|---|---|
| Retrieval (embed + dense‖sparse + RRF + hydrate) | 3.86 s | 10.09 s | **0.28 s** | 0.38 s | 13.7× |
| Rerank + MMR context assembly | 5.76 s | 7.83 s | **1.17 s** | 1.26 s | 4.9× |
| Whole pipeline excluding the LLM | ≈ 9.6 s | — | **1.46 s** | 1.62 s | ≈ 6.6× |
| LLM synthesis (llama-3.3-70b) | 34.8 s | 96.3 s | see §6 | | |

## 6. LLM provider findings (OpenRouter)
- `meta-llama/llama-3.3-70b-instruct` takes **36–38 s** per synthesis, and once returned a JSON object followed by extra text. The parser now keeps the first complete object (`json.raw_decode`).
- `openai/gpt-4.1-mini`: **10.2 s**, valid JSON, 0 of 26 cited IDs invalid. `mistral-small-3.2-24b`: 26.0 s, valid, 0 of 66 invalid.
- httpx timeouts apply per network operation, so a slow provider could hold a request open for minutes (this explains the old p95 of 96 s). There is now a hard overall deadline (`LLM_TIMEOUT_S`). A timeout becomes an explicit abstention.
- **The account's credit is exhausted** (HTTP 402 *"maximum cost exceeds your available credits"*). Before the fix this case produced a silent fake answer. Now the UI shows *"answer generator unavailable"* and lists the retrieved sources for manual review.

## 7. Zero-cost operation: free-model fallback chain
- **Ordered model chain:** `LLM_MODEL_NAME` accepts a comma-separated chain. Each request tries the models in order. A model that returns 429/402 cools down for 90 s. Empty or invalid output moves on to the next model. If every model fails, the pipeline abstains explicitly. Prefixes such as `gemini:`/`groq:`/`cerebras:` let free keys from other providers join the chain.
- **Free models tested on the real synthesis prompt (25 Nov, 01:00 IST):**

| Model (OpenRouter `:free`) | Result |
|---|---|
| `nvidia/nemotron-3-ultra-550b-a55b` | **28.6 s, valid JSON**, 4 atomic claims (with reasoning set to "low" and the compact format) |
| `baidu/ernie-5.5-flash-preview` | 2.0 s on short prompts. On the full prompt its mandatory hidden reasoning used all 4,000 tokens, so it serves as the **verifier**, not the generator |
| `nvidia/nemotron-3.5-super-vl` | reasoning exhausted the token budget |
| `google/gemma-4-31b/26b`, `ling-3.2-flash`, `particle-1` | rate-limited upstream (HTTP 429) at test time; kept as fallbacks |

- **Fixes that made free models usable:** `reasoning: {effort: "low"}` on every request; strict length limits in the synthesis prompt (≤60 words per field, ≤6 atomic claims, no redundant summary claims); a 60 s overall deadline.
- **Batched NLI verification:** all claims are now checked in **one** call instead of one call per claim. That is about 2 LLM calls per question instead of ~8, which keeps the system within the free-tier quota (~50 requests/day without credit).
- **Live result:** "Does osimertinib improve OS in EGFR-mutant advanced NSCLC?" took **18.7 s** end to end. The model correctly **declined to use outside knowledge**: the retrieved FLAURA passages did not contain the OS result, so it abstained and explained why.
