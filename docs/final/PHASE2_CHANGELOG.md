# Phase 2: Capabilities beyond a RAG chatbot

Branch `final-milestone`, commits `04e770a`, `e6e69fd`, `50de3b4` (25 Nov 2026).

| Capability | What it does | Why a plain RAG chatbot cannot do it |
|---|---|---|
| **Patient Case Mode** (Milestone-1 Feature B) | 1. A local PHI guard scans the case text.<br>2. The LLM extracts facts; each must quote the case text verbatim.<br>3. The quote is checked against the text, and untraceable facts are dropped.<br>4. Conflicting values are flagged, and missing decision-critical variables are listed.<br>5. The clinician confirms the facts; the case-aware search runs alongside a general-search baseline. | A chatbot reads a pasted case as free text. It cannot prove where a fact came from, refuse identifiable data, or show which studies fit *this* patient. |
| **Applicability Matrix** | For each study and each patient fact, a deterministic judgement: ✓ match / ✗ differs / ? not reported, with the justifying sentence. It knows drug→biomarker implications (osimertinib → EGFR-mutant), that EGFR exon 20 insertions are a distinct subgroup, and that adjuvant trials are early-stage. | RAG ranks by text similarity. A trial in the wrong subgroup looks just as "relevant". |
| **Quantitative Evidence Map** | Extracts HR/OR/RR with 95% CI and p-values deterministically, keeping exact source spans. Separates treatment effects from prognostic associations. Computes Cochran's Q, I², τ² and an exploratory DerSimonian–Laird pooled estimate, and draws a forest plot. | An LLM paraphrases numbers and can invent them. Here every number is copied from a sentence you can hover. |
| **GRADE-inspired certainty engine** | Starts High or Low from study design, then downgrades for risk of bias, inconsistency, indirectness and imprecision, using measured signals. Publication bias is marked "not assessed". | It replaces the LLM's self-assigned "High quality" label, which had no basis. |

## Measured results
- **Model selection** (identical real evidence; `evals/results/llm_quality_benchmark.json`):
  - gemini-2.5-flash: 0 fabricated citation IDs, 100% of the numbers in its answers appear verbatim in the evidence, 18–30 s.
  - The Pro models are not available on the free tier.
- **Claim verifier** (gemini-3.1-flash-lite, low reasoning): 8/8 correct on a labelled probe, including a 18.9 vs 19.8 numeric trap, in 2.4 s. Before, verification was one LLM call per claim.
- **Case extraction** (live, gemini-2.5-flash):
  - Case 2: 17/17 proposed facts anchored to the source text; the PD-L1 80% vs 45% conflict was flagged; ROS1 was listed as missing.
  - Case 1: 17/17 anchored.
- **PHI guard:** the demo text with a name, UHID and phone number was blocked locally; none of the 8 synthetic cases was blocked.
- **Applicability:** on case 2, mean applicability was 1.00 for the case-aware search vs 0.876 for the general search.
- **Live review caught clinical errors, now fixed and covered by tests:**
  - exon 20 trials were counted as matches for an exon 19 patient;
  - adjuvant trials were counted as matches for a stage IV patient;
  - KRAS was reported as "missing" for an EGFR-positive patient;
  - "Treatment-naive" was treated as a prior therapy.

## Tests
99 → **111** passing. New suites: `test_effect_map.py` (8 tests), `test_case_mode.py` (12 tests).

## Known limitations (carried into the report)
- Applicability uses the retrieved passage plus the title, not the full eligibility criteria, so many cells are honestly "not reported".
- Effect extraction reads ratios with CIs only. Medians, absolute risks and Kaplan–Meier figures are not extracted.
- The risk-of-bias rating is a wording proxy, not a RoB-2 appraisal.
- The pooled estimate is screening only, not a systematic meta-analysis.
- The synthetic cases were written and labelled by the author; no clinician has validated them yet.
