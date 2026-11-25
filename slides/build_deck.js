// Build Indence_Final_Presentation.pptx (14 slides, 16:9 wide).
//   cd slides && node build_deck.js
// Numbers come from evals/results/*.json; images from report/figures and evals/results/figures.
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require("./apply_theme.js");

const ROOT = path.resolve(__dirname, "..");
const FIG = (f) => path.join(ROOT, "report", "figures", f);
const RES = (f) => JSON.parse(fs.readFileSync(path.join(ROOT, "evals", "results", f), "utf8"));
const OUT = path.join(ROOT, "Indence_Final_Presentation.pptx");

const ABL = RES("retrieval_ablation.json");
const OLD = RES("retrieval_ablation_v1_original_blend.json");
const SAF = RES("safety_eval.json").summary;
const CASE = RES("case_mode_eval.json").summary;

const THEME = {
  name: "Indence Clinical",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "13293D", lt1: "FFFFFF", dk2: "3B5568", lt2: "EEF4F3",
    accent1: "0E8C7F", accent2: "E4572E", accent3: "F2A541", accent4: "5B7C99",
    accent5: "BFDCD7", accent6: "D9E1E6", hlink: "0E8C7F", folHlink: "5B7C99",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Indence: Final Evaluation";
const C = pres.SchemeColor;
const W = 13.333;

// ------------------------------------------------------------------ layouts
pres.defineSlideMaster({
  title: "DARK_TITLE",
  background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.1, w: 11.7, h: 1.5, fontSize: 54, bold: true, color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.8, w: 11.7, h: 2.3, fontSize: 22, color: C.accent5, valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  margin: [0.5, 0.6, 0.6, 0.6],
  slideNumber: { x: 12.3, y: 7.0, w: 0.6, h: 0.3, fontSize: 11, color: C.text2, align: "right" },
  objects: [
    { text: { text: "Indence · CSET400C final evaluation", options: { x: 0.6, y: 7.0, w: 6, h: 0.3, fontSize: 11, color: C.text2, margin: 0 } } },
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.95, fontSize: 34, bold: true, color: C.text1, valign: "middle", margin: 0 }, text: "" } },
  ],
});

// ------------------------------------------------------------------ helpers
let n = 0;
const uid = (p) => `${p}_${++n}`;

function text(slide, t, o) {
  slide.addText(t, { isTextBox: true, margin: 0, fontSize: 16, color: C.text1, valign: "top", objectName: uid("txt"), ...o });
}
function card(slide, x, y, w, h, fill) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, rectRadius: 0.12, fill: { color: fill || C.background2 }, line: { color: fill || C.background2 },
    objectName: uid("card"),
  });
}
function dot(slide, x, y, label, fill) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: 0.5, h: 0.5, fill: { color: fill || C.accent1 }, line: { color: fill || C.accent1 }, objectName: uid("dot") });
  slide.addText(label, { isTextBox: true, x, y, w: 0.5, h: 0.5, fontSize: 16, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, objectName: uid("dotlbl") });
}
function img(slide, file, x, y, w, h, alt) {
  slide.addImage({ path: FIG(file), x, y, w, h, altText: alt, objectName: uid("img") });
}
function content(title, section) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  return s;
}
const pct = (x) => `${Math.round(x * 100)}%`;
const chartText = { catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt", titleFontFace: "+mn-lt" };

// ================================================================== 1 title
pres.addSection({ title: "Opening" });
let s = pres.addSlide({ masterName: "DARK_TITLE", sectionTitle: "Opening" });
s.addText("Indence", { placeholder: "title" });
s.addText([
  { text: "A verifiable clinical evidence reasoning engine for oncology", options: { breakLine: true } },
  { text: "Case → Retrieve → Appraise → Verify → Brief or Abstain", options: { italic: true, color: C.background1, fontSize: 18 } },
], { placeholder: "body" });
text(s, "CSET400C Capstone · Final evaluation · 26 November 2026\n[Team members] · Guide: [Guide name]", { x: 0.8, y: 6.2, w: 11, h: 0.8, fontSize: 14, color: C.accent6 });
s.addNotes("Indence answers oncology evidence questions only when every claim is verified against its source, and it says so when it cannot. Today: the problem, how Indence works, the measured results against two baselines, the honest trade-offs, then a live demo.");

// ================================================================== 2 problem
pres.addSection({ title: "Problem" });
s = content("General AI fails where clinicians need it most", "Problem");
text(s, `What an LLM alone did on our 40-question oncology benchmark (same generator used for every system later)`, { x: 0.6, y: 1.35, w: 12, h: 0.5, fontSize: 16, color: C.text2 });
const L = SAF.llm_only;
const stats = [
  [L.fabricated_or_misattributed_refs.replace("/", " / "), "PubMed IDs that point to the wrong paper or none", "It names the real trial, then attaches an ID belonging to vascular surgery or plant biology"],
  [`${Math.round((1 - L.abstention_by_category.fictional_drug) * 8)} / 8`, "questions about drugs that do not exist, answered", "Confident efficacy claims for invented compounds"],
  [L.phi_sent_to_provider.replace("/", " / "), "cases with patient identifiers sent to the provider", "Names, hospital IDs and phone numbers forwarded unchanged"],
];
stats.forEach(([big, label, sub], i) => {
  const x = 0.6 + i * 4.1;
  card(s, x, 2.15, 3.8, 3.9);
  text(s, big, { x: x + 0.35, y: 2.45, w: 3.1, h: 1.2, fontSize: 60, bold: true, color: C.accent2, fontFace: "Cambria" });
  text(s, label, { x: x + 0.35, y: 3.75, w: 3.1, h: 0.9, fontSize: 18, bold: true });
  text(s, sub, { x: x + 0.35, y: 4.75, w: 3.1, h: 1.1, fontSize: 14, color: C.text2 });
});
text(s, "Plain RAG fixes some of this, but not identifier leakage or wrong-subgroup evidence (slide 7).", { x: 0.6, y: 6.3, w: 12, h: 0.45, fontSize: 16, italic: true, color: C.accent1 });
s.addNotes("These numbers are from our own benchmark, not from the literature. The LLM-only system cited FLAURA and DESTINY-Breast04 by name, but 47 of 49 PubMed identifiers belonged to unrelated papers, checked with NCBI E-utilities. It answered six of eight questions about fictional drugs and sent all six identifier-containing cases to the provider. This is the failure profile Indence is designed against.");

// ================================================================== 3 what it is
pres.addSection({ title: "Solution" });
s = content("Seven stages, and the LLM is only one of them", "Solution");
img(s, "fig_pipeline.png", 0.9, 1.4, 11.5, 5.38, "Seven-stage pipeline: gate, understand, retrieve, rank, appraise, synthesise, verify; ending in a verified brief or an explicit abstention");
s.addNotes("Every question passes these seven stages. Only stage 6, and the entailment judgement inside stage 7, use an LLM. Safety gates, retrieval, ranking, population matching, effect sizes, heterogeneity, certainty and the abstention decision are deterministic code with unit tests. That is the answer to 'isn't this just RAG': RAG is stages 3 and 6.");

// ================================================================== 4 architecture
s = content("Runs on one laptop GPU, with free model tiers", "Solution");
img(s, "fig_architecture.png", 0.6, 1.4, 7.6, 5.47, "System architecture: Next.js UI, FastAPI pipeline, local GPU models, Qdrant and SQLite, external LLM services behind the PHI gate");
const arch = [
  ["19,057", "open-access oncology papers, 412,846 indexed passages (Qdrant + SQLite)"],
  ["8 GB", "RTX 3070 Ti runs BGE-M3 (dense + sparse) and the cross-encoder reranker"],
  ["0 PHI", "leaves the machine: a local identifier guard runs before any external call"],
  ["9-model", "Gemini Flash chain with quota-aware fallback; Gemini Decision 1 verifies claims"],
];
arch.forEach(([big, sub], i) => {
  const y = 1.45 + i * 1.35;
  card(s, 8.6, y, 4.1, 1.17);
  text(s, big, { x: 8.85, y: y + 0.14, w: 3.7, h: 0.45, fontSize: 22, bold: true, color: C.accent1, fontFace: "Cambria" });
  text(s, sub, { x: 8.85, y: y + 0.58, w: 3.7, h: 0.55, fontSize: 14, color: C.text2 });
});
s.addNotes("A modular FastAPI monolith. Embedding and reranking are local on the laptop GPU; vectors in Qdrant, documents and full query traces in SQLite. External models are only used for writing, case-fact extraction and claim verification, and only after the identifier check passes. The free tier allows about 20 requests per day per Flash model, so the adapter rotates a chain of models and honours each one's reset time.");

// ================================================================== 5 M1 -> final
pres.addSection({ title: "Since Milestone-1" });
s = content("The audit found defects hidden in October", "Since Milestone-1");
const rows = [
  ["‘Hybrid’ search", "Sparse vectors silently empty: dense-only", "Real dense + sparse, each measured"],
  ["Evaluation script", "Put the expected papers first by construction", "Live-pipeline harness, bootstrap CIs, 3 systems"],
  ["Provider failure", "Returned a template answer marked verified", "Explicit ‘generator unavailable’ abstention"],
  ["HER2+ / EGFR+", "Never detected; HER2-positive listed as rejected", "Fixed; 0% wrong-subgroup citations"],
  ["Patient context", "Proposed only", "Case Mode, Applicability Matrix, Safety Check"],
  ["Latency excl. LLM", "≈ 9.6 s", "1.46 s"],
  ["Automated tests", "72", "122"],
];
const head = (t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text1 } } });
s.addTable([
  [head("Area"), head("Milestone-1 build"), head("Final system")],
  ...rows.map(([a, b, c]) => [{ text: a, options: { bold: true } }, { text: b, options: { color: C.accent2 } }, { text: c, options: { color: C.accent1 } }]),
], {
  x: 0.6, y: 1.5, w: 12.1, colW: [2.6, 4.75, 4.75], fontSize: 16, fontFace: "Calibri", color: C.text1,
  border: { type: "solid", pt: 1, color: C.accent6 }, fill: { color: C.background1 }, rowH: 0.62, valign: "middle",
  margin: [0.05, 0.12, 0.05, 0.12], objectName: uid("table"),
});
s.addNotes("We audited the Milestone-1 code before adding anything. The most serious finding: 'hybrid' retrieval had been dense-only because of an fp16 versus fp32 mismatch that a library retry loop swallowed. The old evaluation script reported results by placing the expected papers first. Ten safety defects and six performance defects were fixed, each pinned by a regression test.");

// ================================================================== 6 retrieval
pres.addSection({ title: "Results" });
s = content("Retrieval: every layer adds recall", "Results");
const cfg = [["sparse", "Sparse"], ["dense", "Dense"], ["hybrid", "Hybrid (RRF)"], ["hybrid_rerank", "Hybrid + rerank"]];
const labels = cfg.map((c) => c[1]).concat(["Indence (Oct blend)", "Indence (final)"]);
const r10 = cfg.map((c) => ABL.all_queries[c[0]]["recall@10"].mean).concat([OLD.all_queries.full["recall@10"].mean, ABL.all_queries.full["recall@10"].mean]);
const mrr = cfg.map((c) => ABL.all_queries[c[0]]["mrr@10"].mean).concat([OLD.all_queries.full["mrr@10"].mean, ABL.all_queries.full["mrr@10"].mean]);
s.addChart(pres.charts.BAR, [
  { name: "Recall@10", labels, values: r10 },
  { name: "MRR@10", labels, values: mrr },
], {
  x: 0.6, y: 1.45, w: 7.9, h: 5.3, barDir: "col", barGrouping: "clustered", barGapWidthPct: 60,
  chartColors: [HEX.accent1, HEX.accent4], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00",
  dataLabelFontSize: 11, dataLabelColor: HEX.dk2, catAxisLabelFontSize: 12, valAxisLabelFontSize: 11,
  catAxisLabelColor: HEX.dk2, valAxisLabelColor: HEX.dk2, valAxisMinVal: 0, valAxisMaxVal: 1, valAxisLabelFormatCode: "0.0",
  valGridLine: { color: "E3E8EB", size: 0.75 }, catGridLine: { style: "none" },
  showLegend: true, legendPos: "t", legendFontSize: 12, legendColor: HEX.dk2,
  showTitle: true, title: `Known-item benchmark, ${ABL.n_queries} questions`, titleFontSize: 14, titleColor: HEX.dk1,
  ...chartText, objectName: uid("chart"),
});
const pts = [
  ["Hybrid + reranker is best for finding one paper", `R@10 ${r10[3].toFixed(3)}; the cross-encoder adds ≈ 0.10 MRR`],
  ["The benchmark caught our own ranking bug", `Priors swamped relevance: R@10 ${r10[4].toFixed(3)} → ${r10[5].toFixed(3)} after the fix`],
  ["The remaining MRR gap is deliberate", "Same rank in 71 of 119; in 24 the paper moved #1 → #2–4 because a guideline or better-fitting study was ranked above it"],
];
pts.forEach(([h, b], i) => {
  const y = 1.55 + i * 1.75;
  dot(s, 8.85, y, String(i + 1));
  text(s, h, { x: 9.5, y, w: 3.25, h: 0.6, fontSize: 16, bold: true });
  text(s, b, { x: 9.5, y: y + 0.62, w: 3.25, h: 1.0, fontSize: 14, color: C.text2 });
});
s.addNotes("This is the benchmark where Indence looks weakest, so we explain it. The silver benchmark has exactly one correct paper per question. Indence deliberately nudges guidelines, meta-analyses and better population matches above near-ties. When the target moves from first to second place, reciprocal rank halves, and those 24 small moves explain about 0.12 of the 0.135 MRR gap. Recall@10 is statistically tied with overlapping confidence intervals. Two genuine errors: the population filter removed two target papers.");

// ================================================================== 7 safety
s = content("Same generator, different architecture", "Results");
const N = SAF.naive_rag, I = SAF.indence;
const scats = ["Correct abstention", "Fictional drugs refused", "Patient identifiers kept local", "Citations resolve to the source"];
const fic = (x) => x.abstention_by_category.fictional_drug;
s.addChart(pres.charts.BAR, [
  { name: "LLM only", labels: scats, values: [L.correct_abstention_rate, fic(L), 0, 1 - L.fabricated_ref_rate].map((v) => v * 100) },
  { name: "Naive RAG", labels: scats, values: [N.correct_abstention_rate, fic(N), 0, 1 - N.invalid_citation_rate].map((v) => v * 100) },
  { name: "Indence", labels: scats, values: [I.correct_abstention_rate, fic(I), 1, 1 - I.invalid_citation_rate].map((v) => v * 100) },
], {
  x: 0.6, y: 1.45, w: 8.2, h: 5.3, barDir: "col", barGrouping: "clustered", barGapWidthPct: 55,
  chartColors: [HEX.accent2, HEX.accent4, HEX.accent1], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0\"%\"",
  dataLabelFontSize: 11, dataLabelColor: HEX.dk2, catAxisLabelFontSize: 12, valAxisLabelFontSize: 11,
  catAxisLabelColor: HEX.dk2, valAxisLabelColor: HEX.dk2, valAxisMinVal: 0, valAxisMaxVal: 110, valAxisMajorUnit: 25, valAxisLabelFormatCode: "0",
  valGridLine: { color: "E3E8EB", size: 0.75 }, catGridLine: { style: "none" },
  showLegend: true, legendPos: "t", legendFontSize: 12, legendColor: HEX.dk2,
  showTitle: true, title: "40 questions, generator gemini-3.1-flash-lite for all systems (%)", titleFontSize: 14, titleColor: HEX.dk1,
  ...chartText, objectName: uid("chart"),
});
const sb = [
  [`${pct(I.subgroup_leakage_rate)} vs ${pct(N.subgroup_leakage_rate)}`, "wrong-subgroup citations (Indence vs naive RAG)"],
  [`${(I.numeric_faithfulness * 100).toFixed(1)}%`, "numbers in answers found verbatim in the evidence"],
  [pct(I.answer_rate_on_answerable), "of answerable questions answered: 2 withheld because verification failed"],
];
sb.forEach(([big, sub], i) => {
  const y = 1.5 + i * 1.75;
  card(s, 9.1, y, 3.6, 1.55);
  text(s, big, { x: 9.35, y: y + 0.15, w: 3.1, h: 0.6, fontSize: 28, bold: true, color: C.accent1, fontFace: "Cambria" });
  text(s, sub, { x: 9.35, y: y + 0.78, w: 3.1, h: 0.7, fontSize: 14, color: C.text2 });
});
s.addNotes("All three systems used the same generator, so the differences come from architecture. Naive RAG is dense top-5 retrieval plus an 'answer with citations' prompt. Indence refused every out-of-scope, fictional-drug and identifier case, never sent identifiers out, and cited no wrong subgroup. The price: it withheld two of twenty answerable questions, one because the verifier caught a claim contradicting its own source. We think that is the right trade for a clinical tool.");

// ================================================================== 8 verification
s = content("Every claim is checked against its source", "Results");
img(s, "ss_answer_crop.png", 0.6, 1.45, 3.37, 5.35, "Indence answer with per-section verification badges and citation chips");
const ver = [
  ["Cite only what was retrieved", "The generator may cite evidence IDs from the bundle only; others invalidate the claim"],
  ["Verify with calibrated confidence", "Gemini Decision 1 labels each claim and section supported, contradicted or not established, in one request"],
  ["Remove and show, never hide", "Content its own source contradicts is removed and listed under ‘Removed by verification’"],
  ["Abstain when support is thin", "Withhold the answer if > 1/3 of claims are contradicted or < 50% are supported"],
];
ver.forEach(([h, b], i) => {
  const y = 1.5 + i * 1.32;
  dot(s, 4.6, y, String(i + 1));
  text(s, h, { x: 5.3, y, w: 7.4, h: 0.45, fontSize: 18, bold: true });
  text(s, b, { x: 5.3, y: y + 0.47, w: 7.4, h: 0.75, fontSize: 15, color: C.text2 });
});
s.addNotes("The screenshot is a real answer. Each section shows the verifier's verdict and confidence; 'not fully established' means the cited passages only partly support it. The verifier was 4 of 4 on a labelled probe, including a trap where the claim said 18.9 months and the source said 19.8. A single request covers all claims, so we stay inside the free tier of 10,000 requests a day.");

// ================================================================== 9 appraisal
s = content("Appraisal a chatbot cannot do", "Results");
img(s, "ss_forest.png", 0.6, 1.45, 7.3, 3.88, "Forest plot of hazard ratios extracted verbatim from retrieved passages");
text(s, "Every ratio is copied from a sentence in a retrieved passage; hovering shows the sentence. Q, I² and τ² are computed per endpoint; a pooled estimate is labelled screening only.", { x: 0.6, y: 5.55, w: 7.3, h: 1.2, fontSize: 15, color: C.text2 });
const grade = [["Start", "RCT / meta-analysis: high; observational: low"], ["Risk of bias", "phase II, single-arm, small n"], ["Inconsistency", "I² > 50% or contradiction flags"], ["Indirectness", "population mismatch"], ["Imprecision", "CI crosses 1 or is wide"]];
text(s, "GRADE-inspired certainty, by rule", { x: 8.4, y: 1.45, w: 4.3, h: 0.45, fontSize: 18, bold: true });
grade.forEach(([h, b], i) => {
  const y = 2.05 + i * 0.93;
  card(s, 8.4, y, 4.3, 0.8);
  text(s, h, { x: 8.6, y: y + 0.08, w: 3.9, h: 0.32, fontSize: 15, bold: true, color: C.accent1 });
  text(s, b, { x: 8.6, y: y + 0.42, w: 3.9, h: 0.32, fontSize: 14, color: C.text2 });
});
s.addNotes("The LLM used to write its own 'High quality' label. Now certainty is computed: start from study design and downgrade one level per domain with a measured signal; publication bias is shown as not assessed. The forest plot never contains a number the model wrote; if no ratio with a confidence interval is found, the panel says so instead of inventing one.");

// ================================================================== 10 case mode
pres.addSection({ title: "Patient Case Mode" });
s = content("Patient Case Mode: evidence that fits", "Patient Case Mode");
img(s, "ss_case_facts_crop.png", 0.6, 1.45, 3.69, 5.35, "Extracted case facts with character offsets and a flagged PD-L1 conflict");
const steps = [
  ["Local PHI guard", "Names, IDs, phone numbers and dates of birth are blocked before anything leaves the laptop"],
  ["Facts with proof", "Each fact must quote the case text; untraceable facts are dropped and conflicts flagged"],
  ["Applicability Matrix", "Each study × each patient fact: matches, differs or not reported, with the justifying sentence"],
  ["General search kept alongside", "Guards against confirmation bias"],
];
steps.forEach(([h, b], i) => {
  const y = 1.45 + i * 1.27;
  dot(s, 4.85, y, String(i + 1));
  text(s, h, { x: 5.5, y, w: 4.0, h: 0.4, fontSize: 17, bold: true });
  text(s, b, { x: 5.5, y: y + 0.4, w: 4.0, h: 0.85, fontSize: 14, color: C.text2 });
});
const cm = [["98.8%", "gold-fact recall (84/85)"], [`${CASE.applicability.general_mean_of_means.toFixed(2)} → ${CASE.applicability.case_aware_mean_of_means.toFixed(2)}`, "mean applicability of top-8 studies"], [`${CASE.applicability.general_different_population_total} → ${CASE.applicability.case_different_population_total}`, "wrong-population studies (top 8)"]];
cm.forEach(([big, sub], i) => {
  const y = 1.45 + i * 1.5;
  card(s, 9.85, y, 2.85, 1.3);
  text(s, big, { x: 10.05, y: y + 0.12, w: 2.5, h: 0.55, fontSize: 26, bold: true, color: C.accent1, fontFace: "Cambria" });
  text(s, sub, { x: 10.05, y: y + 0.7, w: 2.5, h: 0.55, fontSize: 13, color: C.text2 });
});
text(s, `8 synthetic cases; case-aware search better in ${CASE.applicability.cases_improved} of ${CASE.n_cases}`, { x: 9.85, y: 6.0, w: 2.85, h: 0.7, fontSize: 13, italic: true, color: C.text2 });
s.addNotes("This is Milestone-1 Feature B, delivered. The clinician confirms the facts before they influence search. The matrix knows clinical rules such as osimertinib implying EGFR-mutant disease, exon 20 insertions being a separate subgroup, and adjuvant trials not applying to stage IV. Live review found and fixed exactly those errors. Cases are synthetic and author-labelled; no real patient data is used.");

// ================================================================== 11 safety check
s = content("Safety check uses medicines and history", "Patient Case Mode");
img(s, "ss_safety_crop.png", 0.6, 1.4, 12.1, 3.86, "Patient Safety Check flagging palbociclib and ribociclib with clarithromycin");
const sc = [
  ["Curated rules", "10 interaction and 9 comorbidity rules from drug labels, plus organ function"],
  ["Treatment history", "Re-using a drug or class the patient already progressed on is flagged"],
  ["AI screen", "Decision 1 flags other pairs only at p ≥ 0.85, as ‘needs review’"],
];
sc.forEach(([h, b], i) => {
  const x = 0.6 + i * 4.1;
  dot(s, x, 5.55, String(i + 1));
  text(s, h, { x: x + 0.65, y: 5.5, w: 3.2, h: 0.4, fontSize: 17, bold: true });
  text(s, b, { x: x + 0.65, y: 5.92, w: 3.2, h: 0.9, fontSize: 14, color: C.text2 });
});
s.addNotes("Drugs come from both the answer and the retrieved studies, so the check still works when the answer is withheld. In this synthetic HR-positive breast case it raised four major flags and one moderate: CYP3A4 interactions with clarithromycin and re-use of palbociclib after progression. The AI screen earlier found capivasertib with clarithromycin, which we then added as a curated rule; its one false positive is why the threshold is 0.85. It is a screening aid, and the panel says so.");

// ================================================================== 12 trade-offs
pres.addSection({ title: "Discussion" });
s = content("What it costs, and what is not done yet", "Discussion");
const tr = [
  ["0.653 vs 0.788", "MRR@10, Indence vs hybrid + rerank", "Quality and population priors reorder near-ties; Recall@10 is statistically tied"],
  ["90%", "answer rate on answerable questions", "Two answers withheld because verification failed"],
  [`${I.latency_answered_s.p50.toFixed(1)} s`, "median latency (p95 " + I.latency_answered_s.p95.toFixed(1) + " s)", "p95 target of 15 s met; p50 target of 8 s not; 1.46 s is local work"],
  ["0", "clinicians have reviewed it", "Labels are silver or author-written; a review protocol is ready"],
];
tr.forEach(([big, label, sub], i) => {
  const x = 0.6 + (i % 2) * 6.15, y = 1.5 + Math.floor(i / 2) * 2.7;
  card(s, x, y, 5.95, 2.45);
  text(s, big, { x: x + 0.35, y: y + 0.25, w: 5.3, h: 0.75, fontSize: 36, bold: true, color: i === 3 ? C.accent2 : C.text1, fontFace: "Cambria" });
  text(s, label, { x: x + 0.35, y: y + 1.05, w: 5.3, h: 0.45, fontSize: 17, bold: true });
  text(s, sub, { x: x + 0.35, y: y + 1.55, w: 5.3, h: 0.75, fontSize: 14, color: C.text2 });
});
s.addNotes("We report the costs as clearly as the wins. The MRR gap and the answer rate are design choices. Latency is dominated by two external calls on free tiers. The biggest open item is clinician validation: three to five oncologists reviewing ten briefs and two case outputs each, with agreement against the automated verifier. Other limits: open-access corpus only, applicability judged from the passage rather than full eligibility criteria, and a small curated rule set.");

// ================================================================== 13 impact
s = content("Impact and a path beyond the capstone", "Discussion");
const sdg = [["3", "Good health", "Applicable, verified evidence and medication-aware checks for oncology teams"], ["9", "Innovation", "Auditable evidence infrastructure on commodity hardware"], ["10", "Reduced inequalities", "One laptop GPU and free model tiers: affordable for smaller hospitals and colleges"]];
text(s, "UN Sustainable Development Goals", { x: 0.6, y: 1.45, w: 5.8, h: 0.45, fontSize: 18, bold: true });
sdg.forEach(([num, h, b], i) => {
  const y = 2.05 + i * 1.5;
  s.addShape(pres.shapes.OVAL, { x: 0.6, y, w: 0.9, h: 0.9, fill: { color: C.accent1 }, line: { color: C.accent1 }, objectName: uid("sdg") });
  text(s, num, { x: 0.6, y, w: 0.9, h: 0.9, fontSize: 26, bold: true, color: C.background1, align: "center", valign: "middle", fontFace: "Cambria" });
  text(s, `SDG ${num}: ${h}`, { x: 1.75, y, w: 4.7, h: 0.4, fontSize: 17, bold: true });
  text(s, b, { x: 1.75, y: y + 0.42, w: 4.7, h: 0.8, fontSize: 14, color: C.text2 });
});
card(s, 6.9, 1.45, 5.8, 5.3, C.text1);
text(s, "Startup path", { x: 7.25, y: 1.7, w: 5.1, h: 0.5, fontSize: 22, bold: true, color: C.background1, fontFace: "Cambria" });
s.addText([
  { text: "On-premises evidence workspace for oncology departments and medical colleges in India", options: { bullet: true, breakLine: true } },
  { text: "Tumour-board briefs with applicability and interaction checks", options: { bullet: true, breakLine: true } },
  { text: "Adjacent products on the same verified layer: literature alerts, review screening, pharmacy checks", options: { bullet: true, breakLine: true } },
  { text: "Before any pilot: clinician validation, licensed guidelines, security review, DPDP compliance", options: { bullet: true } },
], { isTextBox: true, x: 7.25, y: 2.4, w: 5.15, h: 4.2, fontSize: 17, color: C.background1, paraSpaceAfter: 16, valign: "top", margin: 0, objectName: uid("txt") });
s.addNotes("The design choices that made this run on a student budget, a local GPU and free model tiers, are also what makes on-premises deployment possible in hospitals that will not send patient data to the cloud. The verified-evidence layer is reusable: the same pipeline can power literature surveillance and pharmacy review. We are clear that a clinical pilot needs validation and compliance work first.");

// ================================================================== 14 close
pres.addSection({ title: "Close" });
s = pres.addSlide({ masterName: "DARK_TITLE", sectionTitle: "Close" });
s.addText("Evidence before confidence", { placeholder: "title" });
s.addText([
  { text: "Zero fabricated references, zero identifier leaks, zero wrong-subgroup citations against two baselines", options: { bullet: true, breakLine: true } },
  { text: "Evidence appraised, fitted to the patient and checked for interactions, with every model named", options: { bullet: true, breakLine: true } },
  { text: "Honest about the cost: a lower MRR, withheld answers and no clinician validation yet", options: { bullet: true } },
], { placeholder: "body" });
text(s, "Live demo next: verified answer → forest plot → Patient Case Mode → abstention and PHI block", { x: 0.8, y: 6.3, w: 11.7, h: 0.5, fontSize: 18, italic: true, color: C.accent3 });
s.addNotes("To close: Indence shows that the useful part of a clinical AI is not the language model but the reasoning and checking around it, and that this can be built and measured on a laptop. Now the demo.");

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
