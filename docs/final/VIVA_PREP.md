# Indence: viva preparation

Short, defensible answers. Numbers are from `evals/results/*.json` and match the report.
**Rule for the viva:** if you do not know, say what you would measure to find out. Never guess a number.

---

## A. The big questions

**1. "Isn't this just a RAG chatbot?"**
RAG is two of our seven stages: retrieve and generate. The other five are not RAG: local identifier and scope gates; population and subgroup logic; deterministic appraisal (effect sizes, I², GRADE-style certainty); claim- and section-level verification with an abstention policy; and, in Case Mode, applicability and a medication safety check. We measured the difference: with the **same generator**, naive RAG sent identifiers to the provider in 6/6 cases, answered 1/8 fictional-drug questions and cited a wrong subgroup in 6% of answers. Indence: 0/6, 0/8 and 0%.

**2. Why would anyone use this instead of ChatGPT or OpenEvidence?**
ChatGPT: in our test, 47 of 49 PubMed IDs it produced belonged to the wrong paper. OpenEvidence is strong but closed: you cannot inspect its applicability logic or check how often it abstains. Indence's value is inspectability. Every number links to a sentence, every model is named, and it runs on-premises on one GPU, which matters for hospitals that cannot send patient data to the cloud.

**3. What is your own technical contribution? You used pre-trained models.**
We did not train a model. The contribution is the system and its evaluation: the hybrid retrieval and relevance-first ranking design, the clinical rule layer (population matching, drug→biomarker implications, exon 20, adjuvant vs metastatic), span-verified fact extraction, the effect-size extractor with heterogeneity statistics, the certainty engine, the verification and abstention policy, the safety rules, and a benchmark harness with three systems compared under the same generator.

**4. What was the hardest problem?**
Finding that "hybrid" retrieval was secretly dense-only. With fp16 on, BGE-M3's encoder output float16 while the sparse head stayed float32. The dtype error was swallowed by the library's batch-size retry loop, so sparse vectors were silently empty. We found it from retrieval logs showing `Sparse terms: 0`, fixed it by casting the heads, and confirmed stored document vectors were unaffected (cosine 1.0 on re-encoding).

---

## B. Retrieval

**5. Why BGE-M3?**
One model gives both a 1024-d dense vector and learned sparse term weights, so we get semantic and lexical matching from a single encoder that fits in 8 GB alongside the reranker.

**6. What is RRF and why k = 60?**
RRF(d) = Σ 1/(k + rank_r(d)) over the dense and sparse lists. It fuses rankings without calibrating scores, which matters because dense cosine and sparse dot-product scores are on different scales. k = 60 is the standard value from Cormack et al. (2009); it damps the influence of top ranks. We did not tune it on the benchmark.

**7. Why a cross-encoder after RRF?**
Bi-encoders embed the query and passage separately, so they are fast but approximate. A cross-encoder reads them together and is more accurate but expensive, so we run it only on the top 50. It added about +0.10 MRR@10 (0.688 → 0.788).

**8. What does MMR do?**
Maximal marginal relevance picks passages that are relevant but not redundant, so the generator does not get five copies of one abstract. We reuse the dense vectors already stored in Qdrant instead of re-encoding 50 candidates; that was one of the latency fixes.

**9. Indence's MRR (0.653) is lower than hybrid + rerank (0.788). Isn't your system worse?**
For "find this exact paper", yes, slightly, and we say so. The benchmark has one correct paper per question. Indence deliberately nudges guidelines, meta-analyses and better population matches above near-ties. In 71 of 119 questions the rank is the same and in 10 Indence ranks the paper higher. In 24 the paper drops from #1 to #2–#4, which halves its reciprocal rank and explains about 0.12 of the 0.135 gap. Recall@10 is statistically tied (0.891 vs 0.924, overlapping 95% CIs). Two real errors remain: the population filter removed 2 target papers.

**10. Didn't you tune the ranking on the test set?**
No. The October blend demoted 48 relevant papers, which the benchmark revealed. The fix bounded the priors to tie-breakers (|prior| ≤ 0.10), with weights chosen a priori from the score ranges, not searched on the benchmark. That is why the full system still scores below hybrid + rerank on MRR. A tuned system would not.

**11. How was the retrieval benchmark built? Isn't it biased?**
119 RCTs and meta-analyses were sampled with stratification and a fixed seed. An LLM wrote one clinician-style question per paper, and that paper is the relevant item. It is a silver standard, so the values are lower bounds. To check title-copying bias, 88 strict questions share no 5-word phrase with the title; the ranking order of systems is the same on that subset.

**12. What are Recall@k, MRR and nDCG? Why bootstrap?**
- **Recall@k:** is the relevant paper in the top k?
- **MRR:** 1/rank of the first relevant item.
- **nDCG:** gain discounted by log₂ of the rank, normalised by the ideal ranking.

Bootstrap resampling of the 119 questions gives 95% confidence intervals without assuming a distribution.

---

## C. Generation, verification, abstention

**13. How do you stop the LLM from inventing citations?**
The prompt supplies evidence IDs (E01.S3, …) and the model must return JSON claims citing only those IDs. The server audits every ID against the bundle. An unknown ID invalidates the claim, so a citation cannot point outside the retrieved text. Measured invalid-citation rate: 0%.

**14. How does verification work? Why Gemini Decision 1?**
All claims and prose sections are sent in one request. Each cited passage is included once, in full, and each claim is judged against exactly its own passages. Decision 1 returns supported, contradicted or not established, with a probability for each. We chose it for the calibration, the speed, and its separate free quota of 10,000 requests per day; Flash models allow about 20.

**14a. What is Gemini Decision 1? Is it a new technology?**
Google released it on 11 November 2026. It answers closed-set questions (yes/no, multiple choice, rubric scores) by **scoring the given options in one pass** and returning a probability for each. It does not generate text, and one request can hold many questions over a shared input. That makes it a discriminative judge rather than a chatbot. In spirit it resembles non-generative approaches such as JEPA (joint-embedding predictive architecture), which predict in representation space instead of producing tokens. But Google has not published its architecture, so say "behaves like", never "is built like". We treat it as a black box and judge it by measurements.

**14b. How do you know it works? What was your experiment?** (report §6.6, `evals/run_verifier_eval.py`)
- **Design:** 25 real corpus passages, each reporting a hazard or odds ratio with a 95% CI. From each we generated 5 claims whose correct answer is known by construction: a faithful claim, a changed number, swapped digits (0.53 → 0.35), reversed significance, and a subgroup result the passage never reports. That gave 119 claims. Decision 1 and the chat-model verifier it replaced (gemini-3.1-flash-lite) judged the same pairs.
- **Accuracy:** Decision 1 98.3%, chat model 100%. Both caught every changed number, swapped digit and reversed significance. Decision 1's 2 errors were "contradicted" instead of "not established" for unreported facts. So **it is not more accurate**; say this openly.
- **What it adds:**
  - **Calibration** (expected calibration error 0.05): at confidence ≥ 0.90 it was 100% correct on 76% of claims. That makes thresholds meaningful, like the 0.85 cut-off for AI interaction flags and the confidence shown on every badge.
  - **Speed:** about 18× faster per claim (30 ms vs 538 ms).
  - **Quota:** its own 10,000/day, so verification never uses up the generator's ~20/day.
  - **No parsing failures:** the output is a closed set, not JSON text.
- **Lesson from the first run:** we first wrote claims with the endpoint name taken from our extractor. Manual review showed the extractor sometimes guessed wrong, for example an OlympiA disease-free-survival HR labelled as mortality. In those cases the verifier was right to reject the claim. We fixed the test design, not the result.
- **Limits:** the free tier also caps input at 50,000 tokens per minute (we hit it when batching 75 passages). One answer is about 5,000 tokens, so production use is far below that.

**15. Why abstain when more than 1/3 of claims are contradicted or fewer than 50% are supported? Aren't those arbitrary?**
They are a priori policy choices, not learned thresholds. Earlier, a single contradicted claim killed the whole answer, which caused over-abstention. Now a contradicted claim is removed and shown, and the answer is withheld only when the core is unreliable. A sensitivity analysis of these thresholds is future work, and we say so.

**16. Can the verifier itself be wrong?**
Yes. It is an LLM judge. Mitigations: calibrated confidence is shown, never hidden; contradicted content is listed rather than deleted silently; and the planned clinician study measures agreement (Cohen's κ) between clinicians and the verifier.

**17. Why did Indence answer only 90% of answerable questions?**
It withheld 2 of 20. In one, too few claims could be verified. In the other, the verifier caught a generated claim contradicting its own cited source. For a clinical tool we prefer "I can't verify this, here are the sources" to a fluent unsupported answer.

**17b. Did you find bugs after the evaluation?**
Yes, one in the verifier. The passages a claim cited were joined and cut at 4,000 characters, so text citing three or more passages was judged on half its evidence. The T-DXd recommendation went from "not established, 48%" to "supported, 95%" once it saw the full passages. We fixed it (each passage sent once, in full), added regression tests, and reported the benchmark numbers unchanged, with a note that the 90% answer rate may be an underestimate. It shows why per-claim confidence is displayed rather than hidden.

**18. Why free models? Doesn't that lower quality?**
There was no budget. We benchmarked models on identical evidence: gemini-2.5-flash had 0 invalid citation IDs and all of its numbers appeared verbatim in the evidence. Primary models get the full prompt. Only the last-resort free OpenRouter models get compact output. All three systems in the comparison used the same generator, so the comparison is fair regardless of model strength.

---

## D. Appraisal statistics

**19. Explain Q, I² and τ².**
Log-ratios yᵢ get weights wᵢ = 1/SEᵢ², with SE recovered from the 95% CI as (ln U − ln L)/3.92.
- **Q** = Σwᵢ(yᵢ − ȳ)², a test of heterogeneity.
- **I²** = (Q − df)/Q, the share of variation from real between-study differences.
- **τ²** is the DerSimonian–Laird between-study variance, added to each study's variance for the random-effects estimate.

**20. Is your pooled estimate a meta-analysis?**
No, and the UI says "screening only". There is no systematic search, no risk-of-bias assessment and no check that the estimates are truly comparable. It flags heterogeneity and conflicting directions for the clinician to read.

**21. How is certainty decided?**
Like GRADE: RCTs and meta-analyses start high, observational evidence starts low. Then one level is lost for each of these:
- risk of bias (phase II, single-arm, small n);
- inconsistency (I² > 50% or contradiction flags);
- indirectness (population mismatch);
- imprecision (CI crosses 1 or is wide).

Publication bias is shown as "not assessed". It replaced the LLM's self-assigned quality label.

---

## E. Patient Case Mode and safety

**22. How do you know an extracted fact is real?**
The LLM must return a verbatim quote for every fact. We search for that quote in the case text and record character offsets. If it is not found, the fact is dropped and counted. Traceability was 95.8%, and the 3 untraceable facts were dropped automatically. Gold-fact recall was 98.8% (84/85), against 51.8% for the regex fallback.

**23. How does the PHI guard work, and can it miss things?**
It runs locally before any external call, using pattern rules for the HIPAA Safe Harbor categories plus Indian identifiers (Aadhaar-like numbers, UHID/MRN, phone numbers, emails, name fields, exact dates of birth). It blocked the demo case and all 6 PHI benchmark cases, with no false blocks on the 8 synthetic cases. Pattern rules can miss unusual formats, such as a name written without a label. That is why the scope is synthetic or de-identified data only, and a real pilot would need a validated de-identification tool.

**24. How is the Applicability Matrix decided?**
Deterministic rules compare each patient fact with the study's retrieved text and title: matches, differs or not reported, with the justifying sentence. It encodes clinical knowledge such as osimertinib → EGFR-mutant, EGFR exon 20 as a separate subgroup, adjuvant trials not applying to stage IV, and treatment-naive status. Live review caught errors in exactly these rules, and they are now regression tests.

**25. Does personalisation create confirmation bias?**
That was Milestone-1 risk R15. The general search always runs and stays visible beside the case-aware results. Across 8 cases, mean applicability rose from 0.704 to 0.878, wrong-population studies fell from 19 to 6, and 7 of 8 cases improved. The one exception is shown openly.

**26. How does the safety check use earlier prescriptions?**
Drugs come from the answer and from the retrieved studies. They are checked against:
- current medications (10 curated label-based interaction rules, for example CYP3A4 inhibitors);
- comorbidities (9 rules);
- organ function;
- earlier treatments (re-using a drug or class the patient progressed on).

A Decision 1 screen adds pairs only above p = 0.85, as "needs review". It found capivasertib × clarithromycin, which we then added as a rule, and one false positive, which set the threshold.

---

## F. Engineering, scale, ethics

**27. How would this scale to all of PubMed?**
Qdrant shards horizontally and the corpus pipeline is already batch-based, so the main cost is re-embedding. BGE-M3 on one GPU managed our 420k chunks. Full PMC open access would need a few GPU-days or a cloud batch job. Retrieval latency grows sub-linearly with HNSW. The LLM calls, not the search, dominate latency.

**28. Latency: you missed the 8 s target.**
The median for answered questions is 9.7 s and the 95th percentile is 14.7 s, so the 15 s target is met. Local work went from about 9.6 s to 1.46 s (13.7× faster retrieval) through parallel dense/sparse search, batched SQL, vector reuse for MMR and moving GPU work off the event loop. The rest is two external calls on free tiers. Streaming the answer and a paid tier would bring the median under target.

**29. How did you test it?**
- **Automated tests:** 122, up from 72, including 18 regression tests for the defects found.
- **Benchmarks:** retrieval ablation (119 questions), a 3-system safety benchmark (40) and Case Mode (8 cases, 85 gold facts).
- **Live checks:** every UI flow tested in the browser.

Provider outages count as errors, never as abstentions.

**30. Legal and ethical position?**
- It is decision support for professionals, not a medical device, and makes no outcome claims.
- No identifiable data is processed; identifiers are blocked locally.
- A real deployment would need clinician validation, DPDP Rules compliance, a security review (TLS, roles), and an assessment against CDSS guidance such as the FDA's.
- Corpus licences are PMC open access.

**31. What would you do with three more months?**
1. Run the clinician study.
2. Extract full eligibility criteria for applicability.
3. Add licensed guidelines and Indian drug terminology.
4. Run a threshold sensitivity analysis.
5. Add the feedback UI.
6. Do security hardening.

**32. What is the business model?**
An on-premises evidence workspace for oncology departments and medical colleges, licensed per department, with tumour-board briefs. Adjacent products run on the same verified layer: literature surveillance alerts, review screening and pharmacy interaction review.

**33. Did you use AI tools to build this?**
Yes, and it is disclosed in Appendix E. Gemini and OpenRouter models are runtime components. An AI coding assistant (Claude) helped with auditing, implementation, tests and drafting. Every reported number comes from running the code, and the team reviewed the code and results. Be ready to explain any module yourself.

**34. Individual contributions** *(fill in before the viva; each member should own 2–3 modules)*
- [Member 1]: …
- [Member 2]: …
- [Member 3]: …

---

## Numbers to remember

| | |
|---|---|
| Corpus | 19,057 PMC oncology papers · 420,482 chunks · 412,846 indexed |
| Retrieval R@10 | sparse 0.807 · dense 0.866 · hybrid 0.882 · +rerank 0.924 · Indence 0.891 (was 0.672) |
| MRR@10 | hybrid + rerank 0.788 · Indence 0.653 |
| Safety (LLM / naive RAG / Indence) | correct abstention 40% / 95% / 100% · PHI leaked 6/6 / 6/6 / 0/6 · wrong subgroup n/a / 6% / 0% |
| Fabricated PMIDs, LLM only | 47 of 49 |
| Case Mode | fact recall 98.8% · applicability 0.704 → 0.878 · 7/8 cases improved |
| Latency | local 1.46 s (was ≈ 9.6 s) · answered p50 9.7 s, p95 14.7 s |
| Tests | 72 → 122 |
