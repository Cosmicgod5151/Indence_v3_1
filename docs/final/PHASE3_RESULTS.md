# Phase 3: Evaluation results (25 Nov 2026)

Every number here comes from `evals/results/*.json`. Each result can be regenerated with the scripts in `evals/`, which all call the same `QueryPipeline` the app uses, with real models and the real corpus (19,057 papers, 412,846 indexed chunks).

## 1. Retrieval ablation: 119-question silver known-item benchmark
Method:
- Stratified sample of oncology RCTs and meta-analyses (seed 20261125).
- An LLM wrote one paraphrased clinical question per paper; the source paper is the relevant item.
- Ranking is scored at document level, with 95% bootstrap CIs.
- 88 questions share no 5-word phrase with the paper title (the strict subset).

| Configuration | R@5 | R@10 (95% CI) | R@20 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|
| Sparse only (BGE-M3 lexical) | 0.723 | 0.807 (0.73–0.87) | 0.891 | 0.609 | 0.656 |
| Dense only (BGE-M3) | 0.824 | 0.866 (0.80–0.92) | 0.908 | 0.702 | 0.742 |
| Hybrid (RRF k=60) | 0.782 | 0.882 (0.82–0.93) | 0.916 | 0.688 | 0.734 |
| Hybrid + cross-encoder | **0.916** | **0.924** (0.87–0.97) | **0.941** | **0.788** | **0.822** |
| Full Indence, original Oct-22 blend | 0.521 | 0.672 (0.59–0.76) | 0.874 | 0.394 | 0.459 |
| **Full Indence, relevance-first blend** | 0.798 | 0.891 (0.83–0.94) | 0.916 | 0.653 | 0.710 |

Findings:
- **Each retrieval layer adds value.** Hybrid fusion beats either channel alone on Recall@10/20, and the cross-encoder adds +0.10 MRR.
- **The evaluation exposed a ranking defect.** The October score blend let study-type and population priors (up to ~1.4) swamp the reranker's relevance score (at most 0.3), and **48 relevant papers were demoted**.
  - The blend was redesigned a priori: priors became bounded tie-breakers (|prior| ≤ 0.10).
  - No weights were searched on this benchmark.
  - Recall@10 rose from 0.672 to 0.891, statistically level with hybrid + reranking (CIs overlap).
- **The remaining MRR gap (0.653 vs 0.788) is a deliberate trade-off.** Indence nudges guidelines and phase III trials above equally relevant lower-level evidence. A known-item benchmark penalises that by construction.
- **Caveats:**
  - Silver labels (one relevant paper per question) make every value a **lower bound**.
  - The population hard filter wrongly removed 2 of 119 known items.

## 2. Safety and grounding: Indence vs naive RAG vs LLM-only (40 questions)
All three systems use the **same generator (gemini-3.1-flash-lite)**, so the differences come from the architecture alone.
- Naive RAG = dense top-5 passages plus an "answer with citations" prompt.
- LLM-only references were checked against PubMed E-utilities.
- Indence's column was re-run with the final verifier (Gemini Decision 1). The chat-verifier version is kept in `safety_eval_v1_chat_verifier.json`.

| Metric | LLM only | Naive RAG | **Indence** |
|---|---|---|---|
| Correct abstention (out-of-scope 6, fictional drug 8, PHI 6) | 40% | 95% | **100%** |
| Fictional-drug questions answered (hallucinated) | 6 / 8 | 1 / 8 | **0 / 8** |
| PHI sent to the external model provider | 6 / 6 | 6 / 6 | **0 / 6** |
| Answer rate on the 20 answerable questions | 100% | 100% | 90% |
| Citations that resolve to real/provided sources | **4%** (47/49 fabricated or misattributed PMIDs) | 100% | 100% |
| Numeric faithfulness (numbers in the answer found in the evidence) | n/a | 95.7% | **97.3%** |
| Wrong-subgroup citations (e.g. HER2-low cited for HER2-positive) | n/a | 6.0% | **0%** |
| Latency p50 / p95 (answered) | 8.2 s / 26.1 s | 5.4 s / 17.4 s | 9.7 s / 14.7 s |

The PubMed check was spot-checked by hand. The LLM cites the right *titles* (FLAURA, the FOLFIRINOX NEJM trial, DESTINY-Breast04) with PMIDs that belong to unrelated papers: vascular surgery, lung stem cells, plant root biology.

Indence's two conservative abstentions:
- **Cetuximab:** too few claims were verified.
- **EGFR TKIs:** the verifier detected that a generated claim **contradicted its own cited source**.

That is the intended trade-off: an unverifiable answer is withheld.

## 3. Patient Case Mode (8 synthetic oncology cases, 85 hand-labelled gold facts)
| Metric | Result |
|---|---|
| Gold-fact recall, LLM (gemini-2.5-flash) + span verification | **98.8%** (84/85) |
| Gold-fact recall, deterministic regex fallback | 51.8% |
| Traceability (verified ÷ proposed facts) | 95.8%. The 3 untraceable facts were dropped automatically |
| Conflict detection (PD-L1 80% vs 45%) | detected |
| Mean applicability of the top-8 studies: general → case-aware search | **0.704 → 0.878** |
| "Different population" studies in the top-8 (summed over cases) | **19 → 6** |
| Cases improved | 7 / 8 |

The applicability rows were computed with facts from the regex fallback, because the Gemini Flash daily quota was exhausted during that run. The improvement therefore holds even with the weaker extractor.

## 4. Claim verifier and safety check
- **Gemini Decision 1 as the NLI verifier:** 4/4 on a labelled probe with calibrated confidence 0.98–0.996, including the 18.9-vs-19.8-month numeric trap. One request verifies all claims, and the free tier allows 10,000 requests/day.
- **Patient Safety Check:**
  - Live HR+ breast case: 10 curated flags plus AI-screen flags in 1.5 s.
  - The AI screen caught capivasertib × clarithromycin (p = 0.94) and elacestrant × clarithromycin (p = 0.87). Both are real CYP3A4 label warnings, and both were then added to the curated rules.
  - It also produced one false positive (alpelisib × denosumab, p = 0.73). This led to a 0.85 display threshold and a separate "needs review" severity.

## 5. Latency (excluding the LLM)
| Stage | Before (52 historical runs) | After (10 runs) |
|---|---|---|
| Retrieval | 3.86 s | 0.28 s (13.7×) |
| Rerank + MMR | 5.76 s | 1.17 s (4.9×) |

## 6. Engineering
- Tests: **72 → 122** passing (18 regression, 8 effect-map, 15 case-mode, 4 safety-check and other suites).
- **Free-tier constraints found and handled:**
  - Gemini Flash models allow 20 requests/day each, so the app uses a 10-model chain and honours each model's reset time.
  - Transient 503s get a retry with backoff.
  - Provider outages are counted as errors in the evaluations, never as abstentions.

## Figures
- `evals/results/figures/retrieval_ablation_all_queries.png`, `retrieval_ablation_paraphrased_only.png`
- `evals/results/figures/safety_comparison.png`
- `evals/results/figures/case_mode_eval.png`
- `evals/results/figures/latency_before_after.png`

## Limitations
- The retrieval labels are silver.
- The safety set is small: 40 questions, written by the author.
- The synthetic cases were labelled by the author.
- No clinician validation has been done yet.
- The free-tier generator (flash-lite) was used in the comparisons. The live demo uses gemini-2.5-flash whenever its quota allows.

## Post-evaluation fix: verifier truncation (25 Nov 2026)
- **Defect:** the passages cited by a claim or synthesis section were joined and cut at 4,000 characters before being sent to Gemini Decision 1. Text citing three or more passages was therefore judged on only part of its evidence.
- **Example:** in the trastuzumab deruxtecan recommendation, which cited 5 passages (7,705 characters), the verdict was "not established" (48%) on truncated passages and "supported" (95%) on full passages. Its three sentences checked separately scored 90–96% supported.
- **Fix:** every cited passage is sent once, in full, and each claim is judged against exactly its own passages (`gemini_decision.py`, `abstention.py`, `query_pipeline._verify_sections`). Two regression tests were added (122 passing).
- **Effect on reported numbers:** Section 2 was measured before the fix and is unchanged. The 90% answer rate may be an underestimate, because the "too few claims verified" abstention could have been caused by this defect. Re-run `evals/rerun_indence_column.py` to re-measure when the free-tier quota allows.
