# Indence: live demo script (about 6 minutes)

The demo follows the last slide: **verified answer → forest plot → Patient Case Mode → abstention and PHI block**.
Every flow below was run successfully while the report screenshots were captured (25 Nov 2026).

---

## 0. Before the panel (do this 20 minutes early)

| Step | What to do | Check |
|---|---|---|
| 1 | Close other GPU programs (games, browsers with video, a second backend). The 8 GB GPU holds BGE-M3 and the reranker together and nothing else. | Task Manager → GPU memory mostly free |
| 2 | Run `start_indence.bat` from the project folder. It clears ports 8000/3000/6333, starts the backend (which starts Qdrant itself) and the production frontend, then opens the browser. | Two console windows stay open |
| 3 | Wait for the backend console to show `Application startup complete`. The first start loads the models (about 30–60 s). | No red tracebacks |
| 4 | Sign in at http://localhost:3000 (the form is pre-filled with the local demo account). | Header shows the three tabs |
| 5 | **Warm-up query:** run one Clinical Search (any question from §1). The first query is slow because CUDA kernels compile; later ones are fast. | An answer or an abstention appears |
| 6 | Open the Execution Audit tab once, so it is ready to show. | Stage timings visible |

### Free-tier quota: read this

- Gemini Flash free tier: about **20 requests per day per model**. One clinical question uses about 2 generator calls and 1 verifier call. Case Mode uses 1 extraction call plus 2 searches.
- The quota **resets at midnight Pacific time = 13:30 IST**. If the viva is in the morning, rehearse **the afternoon before**, not on the morning itself.
- The app rotates through 9 models automatically. If every model is exhausted it says **"answer generator unavailable"** and still lists the retrieved sources. That is a designed behaviour you can explain, not a crash.
- The claim verifier (Gemini Decision 1) allows 10,000 requests/day, so it will not run out.

---

## 1. Verified answer and model attribution (90 s)

**Tab:** Clinical Search. Leave the default filters (RCT, meta-analysis, systematic review, guideline; from 2020).

**Type:** `Is trastuzumab deruxtecan effective in patients with HER2-positive breast cancer?`

**Say while it runs (about 10–20 s):**
> "The question is checked for identifiers locally, encoded on the laptop GPU into dense and sparse vectors, searched in Qdrant, reranked, appraised, and only then written by Gemini. Then every claim is checked against the passage it cites."

**Point to:**
1. **Extracted Clinical Intent.** HER2-positive is recognised as the target biomarker, and any excluded subgroups are listed below it. "This is the bug we fixed: before, HER2-positive itself was listed as a rejected subgroup."
2. **Which model did what.** BGE-M3 and the reranker run locally, the clinical logic is rule-based with no LLM, the writer is a Gemini Flash model, and the verifier is Gemini Decision 1 with its timing.
3. **Verification badges** on each section (✓ verified 90%, ? not fully established 63%). "The percentage is the verifier's calibrated confidence."
4. Click a **citation chip** (E02.S3) and open a **References** entry with a PMC link. "Every citation resolves to a real stored passage."
5. If present, **Removed by verification**: "content its own source contradicted is removed and shown, never silently dropped."

## 2. Quantitative evidence map and certainty (60 s)

**Type:** `Does adding immunotherapy to chemotherapy improve overall survival in advanced non-small cell lung cancer?`

**Point to:**
1. **Quantitative Evidence Map** (forest plot). Hover a square: "the exact sentence this hazard ratio was copied from. The model never writes these numbers."
2. **Certainty card** (⊕⊕⊕◯ style): "the starting level comes from study design, then a downgrade per GRADE domain, with the reason shown."
3. If the map says no ratios were found, say: "When no ratio with a confidence interval is in the evidence, Indence says so instead of inventing one." Then move on.

## 3. Patient Case Mode (2 min)

**Tab:** Patient Case.

**3a. Conflict detection.** Click the sample **"NSCLC · conflicting PD-L1"** → **Screen for identifiers & extract facts**.
- Green bar: "No direct identifiers detected (local scan; nothing has left this device)."
- Facts table: each fact has **chars a–b** offsets and is underlined in the case text. "Every fact is proven against the source text; anything the model cannot quote is dropped."
- Red box: **Conflicting PD-L1 values, 80% vs 45%**. "Indence will not choose; the clinician ticks the verified value."

**3b. Applicability and safety.** Click **"Breast · HR+ after CDK4/6i"** → extract → **Find applicable evidence**. This takes about 30–40 s because it runs a general search and a case-aware search.
- **Applicability Matrix:** ✓ / ✗ / ? for each study against each patient fact. Point to a ✗ under *Prior therapy*: "a first-line palbociclib trial does not apply to a patient who has already progressed on palbociclib."
- **Patient Safety Check:** palbociclib × clarithromycin (CYP3A4, major) and palbociclib already received (treatment history). "Earlier prescriptions and current medicines are used for safety, not ignored."
- **General-search baseline** (expandable): "kept on screen so personalisation cannot hide evidence."
- If the case-aware score is lower than the general score: "this is why we show both; in the benchmark it was better in 7 of 8 cases."

## 4. Safe failure (60 s)

**4a. Fictional drug.** Clinical Search → `Is zorvatinib effective for EGFR-mutant non-small cell lung cancer?`
→ **Abstention Triggered: no unverified answer shown.** "No retrieved evidence mentions zorvatinib. The LLM-only baseline answered six of eight questions like this."

**4b. Identifiers.** Patient Case → sample **"Privacy demo (blocked)"** → extract.
→ ⛔ blocked locally, listing the name, UHID and phone number. "Nothing was sent to any model provider. Both baselines sent all six such cases."

## 5. Close (15 s)

Open **Execution Audit**: "Every run is traced: stage timings, every candidate's dense, sparse, RRF and reranker scores, and the models used, so any answer can be reproduced."

---

## Fallback if something fails live

| Problem | Recovery |
|---|---|
| "Answer generator unavailable" | Explain the quota (above) and that this is the designed fail-closed path. Show the screenshots in `report/figures/` (`ss_answer.png`, `ss_forest.png`, `ss_case_facts.png`, `ss_applicability.png`, `ss_safety.png`, `ss_abstain_entity.png`) or Figures 5.1–5.6 in the report. |
| Backend console shows CUDA out of memory | Close other GPU apps, close the backend window, run `start_indence.bat` again. |
| Port already in use | `start_indence.bat` kills stale processes on 8000/3000/6333; run it again. |
| Page stuck on the login screen | Refresh; the session cookie expires after logout. |
| Slow first query | Expected (model warm-up). Talk through the pipeline slide while it runs. |
