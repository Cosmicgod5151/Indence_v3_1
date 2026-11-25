"""Build Indence_Final_Report.docx (CSET400C §12 structure and format).

    backend\\venv\\Scripts\\python.exe report\\build_report.py
Every result table is read from evals/results/*.json, so the report and the
evaluation outputs cannot drift apart.
"""
import json
from pathlib import Path

from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, Inches
from PIL import Image

from docxkit import Report

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
EVF = ROOT / "evals" / "results" / "figures"
RES = ROOT / "evals" / "results"
OUT = ROOT / "Indence_Final_Report.docx"
C = WD_ALIGN_PARAGRAPH.CENTER


def load(name):
    return json.loads((RES / name).read_text(encoding="utf-8"))


def crop(name, top, bottom, out):
    im = Image.open(FIG / name)
    im.crop((0, top, im.width, min(bottom, im.height))).save(FIG / out)
    return FIG / out


ABL = load("retrieval_ablation.json")
ABL_OLD = load("retrieval_ablation_v1_original_blend.json")
SAF = load("safety_eval.json")["summary"]
CASE = load("case_mode_eval.json")["summary"]


def m(mode, metric, ci=False, src=ABL, subset="all_queries"):
    v = src[subset][mode][metric]
    s = f"{v['mean']:.3f}"
    if ci:
        s += f" ({v['ci95'][0]:.2f}–{v['ci95'][1]:.2f})"
    return s


def crop_x(name, left, top, right, bottom, out):
    im = Image.open(FIG / name)
    im.crop((left, top, min(right, im.width), min(bottom, im.height))).save(FIG / out)
    return FIG / out


def pct(x):
    return f"{x * 100:.0f}%"


r = Report()
d = r.doc

# =====================================================================  COVER
def cover():
    for text, size, bold, after in [
        ("BENNETT UNIVERSITY", 18, True, 2),
        ("School of Computer Science Engineering and Technology", 13, False, 30),
        ("CSET400C  |  CAPSTONE PROJECT", 12, True, 40),
        ("INDENCE", 30, True, 6),
        ("A Verifiable Clinical Evidence Reasoning Engine for Oncology", 15, False, 4),
        ("Case → Retrieve → Appraise → Verify → Brief or Abstain", 12, False, 40),
        ("FINAL PROJECT REPORT", 16, True, 4),
        ("Milestone-2 Deliverables and Final Evaluation", 12, False, 40),
    ]:
        p = d.add_paragraph(style="Normal")
        p.alignment = C
        p.paragraph_format.space_after = Pt(after)
        run = p.add_run(text)
        run.font.size, run.bold = Pt(size), bold
    r.table(["Submission detail", "Information"], [
        ["Academic session", "2026–2027"],
        ["Programme / semester", "B.Tech. Computer Science Engineering (AI/ML) / Semester VII"],
        ["Team / Group ID", "[Enter coordinator-approved Group ID]"],
        ["Team members", "[Name 1, enrolment no., email]\n[Name 2, enrolment no., email]\n[Name 3, enrolment no., email]"],
        ["Project guide", "[Guide name, designation]"],
        ["Submission date", "26 November 2026"],
    ], [2.0, 4.2], None, size=11, bold_first_col=True)


cover()

# =====================================================================  FRONT MATTER (roman)
r.new_section("lowerRoman", 1)
r.front_heading("Declaration", new_page=False)
r.p("We hereby declare that the project report entitled **“Indence: A Verifiable Clinical Evidence Reasoning "
    "Engine for Oncology”**, submitted to the School of Computer Science Engineering and Technology, Bennett "
    "University, in partial fulfilment of the requirements of CSET400C Capstone Project, is a record of our own "
    "work carried out under the guidance of [Guide name]. The work has not been submitted elsewhere for any other "
    "degree or diploma. Material taken from other sources, including software libraries, models and published "
    "literature, is acknowledged in the text and in the References. The use of AI tools is disclosed in Appendix E.")
r.p("")
r.table(["Name", "Enrolment number", "Signature", "Date"],
        [["[Name 1]", "[Enrolment no.]", "", ""], ["[Name 2]", "[Enrolment no.]", "", ""],
         ["[Name 3]", "[Enrolment no.]", "", ""]], [2.0, 1.6, 1.5, 1.1], None, size=11)

r.front_heading("Certificate")
r.p("This is to certify that the project report entitled **“Indence: A Verifiable Clinical Evidence Reasoning "
    "Engine for Oncology”**, submitted by [Group ID / student names], is a bonafide record of work carried out "
    "under my supervision for CSET400C Capstone Project during the academic session 2026–2027.")
r.p("")
r.p("[Guide name]", align=WD_ALIGN_PARAGRAPH.LEFT, after=0)
r.p("[Designation], School of Computer Science Engineering and Technology, Bennett University")
r.p("Signature: ____________________          Date: ____________________")

# ---- Abstract
r.front_heading("Abstract", toc=True)
r.p("Evidence for an oncology decision must be traceable to its source and must fit the patient; "
    "general-purpose AI chatbots guarantee neither. In our own benchmark, an unconstrained large language model (LLM) produced "
    "PubMed identifiers that were fabricated or pointed to unrelated papers for 47 of 49 references, answered six "
    "of eight questions about drugs that do not exist, and forwarded identifiable patient details to an external "
    "provider in every test case that contained them.")
r.p("Indence is a clinical evidence reasoning engine built over a locally indexed corpus of **19,057 open-access "
    "oncology papers** (420,482 passages). A question passes through deterministic "
    "safety gates, hybrid dense-plus-learned-sparse retrieval (BGE-M3, reciprocal rank fusion), cross-encoder "
    "reranking, population appraisal, verbatim effect-size extraction with heterogeneity statistics (Cochran’s Q, I², τ²), a "
    "GRADE-inspired certainty rating, structured LLM synthesis restricted to cited evidence identifiers, and "
    "claim-level natural-language-inference verification with calibrated confidence. Contradicted content is "
    "removed and listed; when too little can be verified, Indence abstains. A Patient "
    "Case Mode extracts facts from a de-identified case with source-span proof, builds a study-by-patient "
    "Applicability Matrix and checks proposed drugs against the patient’s medications, earlier treatments, "
    "comorbidities and organ function.")
r.p("On a 40-question safety benchmark using the same generator for every system, Indence achieved 100% correct "
    "abstention (LLM-only 40%, naive retrieval-augmented generation 95%), 0 hallucinated answers about fictional "
    "drugs, 0 of 6 identifier leaks, 100% resolvable citations and 0% wrong-subgroup citations (naive RAG 6%). On "
    "a 119-question retrieval benchmark, hybrid retrieval with reranking reached Recall@10 of 0.924, and the full "
    "Indence ranking 0.891 after an evaluation-driven fix raised it from 0.672. Case-aware search raised mean study "
    "applicability from 0.704 to 0.878 across eight synthetic cases. The system runs on a single laptop GPU with "
    "free-tier model APIs. It is a research prototype and has not been clinically validated.")
r.p("**Keywords:** clinical decision support; evidence-based medicine; oncology; retrieval-augmented generation; "
    "hybrid retrieval; claim verification; GRADE; patient applicability; drug-interaction screening; safe "
    "abstention.")

r.front_heading("Table of Contents")
r.toc('TOC \\o "1-3" \\h \\z \\u')
r.front_heading("List of Figures")
r.toc('TOC \\h \\z \\t "Figure Caption,1"')
r.front_heading("List of Tables")
r.toc('TOC \\h \\z \\t "Table Caption,1"')

r.front_heading("Abbreviations", toc=True)
r.table(["Term", "Meaning"], [
    ["BGE-M3", "BAAI General Embedding, multi-function (dense + learned sparse) encoder"],
    ["CI", "Confidence interval"], ["DDI", "Drug–drug interaction"],
    ["GRADE", "Grading of Recommendations, Assessment, Development and Evaluation"],
    ["HR / OR / RR", "Hazard ratio / odds ratio / risk ratio"],
    ["I², τ², Q", "Heterogeneity statistics (Higgins I², between-study variance, Cochran’s Q)"],
    ["LLM", "Large language model"], ["MMR", "Maximal marginal relevance"],
    ["MRR", "Mean reciprocal rank"], ["nDCG", "Normalised discounted cumulative gain"],
    ["NLI", "Natural-language inference (entailment checking)"],
    ["NSCLC", "Non-small cell lung cancer"], ["PHI", "Protected (personally identifiable) health information"],
    ["PMC", "PubMed Central"], ["RAG", "Retrieval-augmented generation"],
    ["RCT", "Randomised controlled trial"], ["RRF", "Reciprocal rank fusion"],
    ["SDG", "United Nations Sustainable Development Goal"],
], [1.4, 4.8], None, size=10, bold_first_col=True)

# =====================================================================  MAIN (arabic)
r.new_section("decimal", 1, header_text="Indence: Final Project Report")

# --------------------------------------------------------------------- CH 1
r.chapter("Introduction and Problem Statement", first=True)
r.h2("1.1 Project Overview")
r.p("Indence is an evidence engine for oncology clinicians and clinical researchers. Given a focused clinical "
    "question, or a de-identified patient case, it retrieves relevant trials, meta-analyses and guidelines from "
    "a governed literature corpus, appraises them, and returns a short brief in which every material statement "
    "is linked to a verified source passage. When the evidence cannot support a safe answer, it says so and shows "
    "the sources for manual review instead of generating a fluent but unsupported reply.")
r.p("The project’s guiding principle, carried over from Milestone-1, is **evidence before confidence**. In "
    "engineering terms this means that the LLM is used only for language (summarising, extracting and judging "
    "entailment) while relevance ranking, population matching, effect-size extraction, heterogeneity statistics, "
    "certainty rating, applicability scoring and safety rules are deterministic, testable code. Every model "
    "involved in an answer is named in the interface, together with what it did.")

r.h2("1.2 Background and Context")
r.p("PubMed indexes more than 40 million biomedical citations and PubMed Central holds the full text of a large "
    "open-access subset [6, 16]. Oncology changes particularly quickly: new targeted therapies and "
    "biomarker-defined subgroups mean that a trial in HER2-positive breast cancer says little about HER2-low "
    "disease, and an adjuvant trial says little about stage IV disease. IARC estimates 20.6 million new cancer "
    "cases and 9.8 million cancer deaths worldwide in 2024, and WHO projects a shortfall of about 11 million "
    "health workers by 2030 [3, 4]. Fast, reliable access to applicable evidence is therefore worth investigating, "
    "although these figures do not show that any search tool improves outcomes.")
r.p("General-purpose LLMs make the literature conversational, but studies report fabricated references and "
    "variable factual accuracy in medical answers [7, 8]. Retrieval-augmented generation (RAG) improves accuracy "
    "on average [9], and specialised products such as OpenEvidence show that the category is commercially active "
    "[10]. The engineering problem that remains is to make retrieval, source quality, claim-to-citation alignment, "
    "evidence disagreement, patient applicability and uncertainty work together, and to measure them.")

r.h2("1.3 Problem Statement")
r.p("Clinicians and researchers answering focused oncology questions must find the relevant studies, decide "
    "whether each study’s population matches their patient, compare effect sizes across studies, and confirm "
    "that any summary actually says what the source says. Existing chatbots answer fluently but can cite papers "
    "that do not exist, mix incompatible subgroups, accept identifiable patient data, and present uncertain "
    "evidence with unwarranted confidence.")
r.p("**This project designs, implements and evaluates a system that returns an oncology evidence brief only when "
    "its material claims are verified against retrieved source passages; that appraises the population fit, "
    "effect sizes and certainty of the evidence; that adapts retrieval to a de-identified patient case with "
    "traceable facts and safety checks; and that abstains visibly otherwise.** It does not diagnose, prescribe "
    "or replace clinical judgement.")

r.h2("1.4 Objectives and Their Status")
r.table(["#", "Objective (from Milestone-1, refined)", "Status at final evaluation"], [
    ["O1", "Governed oncology corpus with provenance, study type and source locators", "Done: 19,057 PMC papers, 131,704 sections, 420,482 chunks"],
    ["O2", "Hybrid dense + sparse retrieval with fusion and cross-encoder reranking", "Done; a silent defect that disabled the sparse channel was found and fixed"],
    ["O3", "Structured answers whose claims cite only supplied evidence IDs", "Done; 0 invalid citation IDs in evaluation"],
    ["O4", "Server-side citation, numeric and entailment verification; safe abstention", "Done; claim- and section-level verification with calibrated confidence"],
    ["O5", "Versioned evaluation with retrieval, grounding, safety and latency metrics", "Done; real harness replaces a script that fabricated its metrics"],
    ["O6", "Conflict-aware evidence review", "Done: contradiction checks plus quantitative heterogeneity (Q, I², τ²)"],
    ["O7", "Patient-context-aware retrieval (Milestone-1 Feature B)", "Done on synthetic cases: PHI guard, span-verified facts, Applicability Matrix"],
    ["O8", "Patient-specific safety screening (new)", "Done: curated DDI, comorbidity, organ-function and treatment-history rules plus an AI screen"],
    ["O9", "Clinician validation", "Not done; protocol prepared (Section 6.9)"],
], [0.4, 3.3, 2.5], "Project objectives and their status", bold_first_col=True)

r.h2("1.5 What Changed Since Milestone-1")
r.p("Milestone-1 (14 October 2026) proposed the system and documented an existing build with caveats. A code "
    "audit at the start of the final phase found that some of that build only appeared to work. Table 1.2 "
    "summarises what was found and what was delivered; Chapters 5 and 6 give the details and evidence.")
r.table(["Area", "State at Milestone-1", "State at final evaluation"], [
    ["Hybrid retrieval", "Described as dense + sparse; query sparse vectors were silently empty, so search was dense-only", "Real hybrid retrieval; each channel measured separately"],
    ["Evaluation", "Script reported metrics by placing the expected papers first", "Real harness on the live pipeline with bootstrap CIs; 3-system safety benchmark"],
    ["LLM failure", "A template ‘mock’ answer was returned as if verified", "Explicit abstention; free multi-provider model chain with cooldowns"],
    ["Abstention", "Synthesis still displayed after the safety gate abstained", "No synthesis on abstention; contradicted content removed and listed"],
    ["Population logic", "HER2+/HR+/EGFR+ never detected; HER2-positive listed as a rejected subgroup", "Fixed with regression tests; subgroup leakage 0% in evaluation"],
    ["Latency (excl. LLM)", "≈ 9.6 s retrieval + assembly", "1.46 s (6.6× faster)"],
    ["Patient context (Feature B)", "Proposed only", "Implemented: Patient Case Mode, Applicability Matrix, Safety Check"],
    ["Appraisal", "LLM wrote its own ‘High quality’ label", "Deterministic effect map, heterogeneity and GRADE-inspired certainty"],
    ["Model transparency", "Traces recorded ‘mock-llm’", "Per-step model attribution shown in the UI and stored in traces"],
    ["Automated tests", "72 passing", "122 passing"],
], [1.4, 2.4, 2.4], "Milestone-1 versus final state", bold_first_col=True)

r.h2("1.6 Research Questions")
r.bullets([
    "**RQ1.** Does hybrid dense + learned-sparse retrieval with reranking find the relevant paper more often than either channel alone? (Section 6.3)",
    "**RQ2.** Do evidence-ID-constrained generation and claim verification eliminate fabricated citations and unsafe answers compared with an LLM alone and with naive RAG using the same generator? (Section 6.4)",
    "**RQ3.** Can facts be extracted from a case with proof of their source, and does case-aware retrieval return studies that fit the patient better than a general query? (Section 6.5)",
    "**RQ4.** Can the system run within useful latency on a laptop GPU and free-tier APIs? (Section 6.7)",
])
r.h2("1.7 Organisation of the Report")
r.p("Chapter 2 reviews the literature and existing products. Chapter 3 sets out stakeholders and requirements. "
    "Chapter 4 presents the design, Chapter 5 the implementation, and Chapter 6 the testing and evaluation. "
    "Chapter 7 discusses SDG impact, risks, limitations and future scope, and Chapter 8 concludes. The appendices "
    "contain requirement traceability, test cases, the project log, a sample case and the AI-tools statement.")

# --------------------------------------------------------------------- CH 2
r.chapter("Background, Literature and Existing Solutions")
r.h2("2.1 Evidence-Based Medicine, GRADE and Heterogeneity")
r.p("Evidence-based medicine combines the best research evidence with clinical expertise and patient "
    "circumstances. Whether a study is useful depends on its population, intervention, comparator, outcome, "
    "design, risk of bias and precision, not only on topical similarity. The GRADE approach rates the certainty "
    "of a body of evidence: randomised evidence starts at high certainty and observational evidence at low, and "
    "certainty is then downgraded for risk of bias, inconsistency, indirectness, imprecision and publication "
    "bias [13, 23]. Inconsistency across studies is usually quantified with Cochran’s Q and Higgins’ I², the "
    "proportion of variability due to true between-study differences rather than chance [24]; a "
    "random-effects pooled estimate (DerSimonian–Laird) adds the between-study variance τ² to each study’s "
    "variance [25]. Indence implements these ideas as deterministic code but does not claim to perform a "
    "systematic review [20].")

r.h2("2.2 Medical LLMs: Capability and Failure Modes")
r.p("LLMs answer medical questions fluently, but fluency is not evidence. Bhattacharyya et al. found high rates of "
    "fabricated or inaccurate references in ChatGPT-generated medical content [7], and a systematic review "
    "reported a pooled accuracy of 56% for ChatGPT medical responses with substantial heterogeneity [8]. "
    "Hallucination is a general property of neural text generation [26]. WHO guidance on large multimodal models "
    "in health stresses human oversight, transparency, accountability and data protection [11]. Our own baseline "
    "(Section 6.4) reproduces the reference problem: 47 of 49 PubMed identifiers produced by the LLM-only "
    "baseline did not belong to the paper the LLM named.")

r.h2("2.3 Retrieval-Augmented Generation in Biomedicine")
r.p("RAG supplies the generator with retrieved passages [27]. A meta-analysis of 20 biomedical studies reported a "
    "pooled odds ratio of 1.35 (95% CI 1.19–1.53) for performance with RAG versus a baseline LLM [9], and a "
    "physician-rated comparison found specialised evidence systems more relevant than general models on real "
    "clinical questions [10]. Retrieval alone does not guarantee faithful answers: the generator can still "
    "misstate a number, mix subgroups or cite a passage that does not support the sentence. This motivates "
    "verifying each claim after generation, as in atomic-fact evaluation approaches such as FActScore [28].")

r.h2("2.4 Hybrid Retrieval, Fusion, Reranking and Diversity")
r.p("Dense retrievers capture paraphrase but can miss exact drug, gene or trial names; lexical retrievers do the "
    "opposite. BGE-M3 produces both a dense vector and learned sparse term weights from a single encoder [29]. "
    "Reciprocal rank fusion (RRF) combines ranked lists robustly without score calibration [30]. A cross-encoder "
    "such as bge-reranker-v2-m3 reads the query and passage together and is more accurate than either retriever "
    "for the final ordering [31]. Maximal marginal relevance (MMR) reduces redundancy in the selected context "
    "[32]. Ranking quality is measured with Recall@k, mean reciprocal rank (MRR) and nDCG [33], with bootstrap "
    "confidence intervals [34].")

r.h2("2.5 Existing Products and Alternatives")
r.table(["Solution", "Strength", "Limitation relevant to Indence"], [
    ["ChatGPT and similar general assistants", "Broad reasoning, natural interaction", "No guarantee that references exist or support the claim; accepts identifiable data"],
    ["OpenEvidence", "Purpose-built clinical evidence Q&A with citations", "Closed system; claim-level verification, applicability and certainty logic are not inspectable"],
    ["UpToDate / UpToDate Expert AI", "Expert-authored, curated content", "Subscription cost; editorial rather than patient-specific study appraisal"],
    ["ClinicalKey AI", "Conversational search over licensed content with citations", "Commercial licence; no open evaluation of abstention or subgroup errors"],
    ["PubMed / PMC search", "Authoritative, complete discovery", "No synthesis, appraisal or applicability; time-consuming"],
], [1.6, 2.0, 2.6], "Existing solutions compared [10, 12, 14, 15, 16]", bold_first_col=True)

r.h2("2.6 Gap Analysis")
r.p("The gap is not that medical AI does not exist. It is that, in available tools, the steps between retrieval "
    "and a confident sentence are opaque and unmeasured. Table 2.2 maps each observed weakness to the Indence "
    "design response and to the experiment that tests it.")
r.table(["Observed weakness", "Indence design response", "Tested in"], [
    ["Fabricated or misattributed references", "Generator may cite only supplied evidence IDs; citations audited server-side", "Safety benchmark (6.4)"],
    ["Confident answers without support", "Claim and section NLI verification with calibrated confidence; abstention policy", "Safety benchmark (6.4)"],
    ["Wrong-subgroup evidence (e.g. HER2-low for HER2-positive)", "Population matcher, subgroup boundary, Applicability Matrix", "Subgroup traps (6.4), case eval (6.5)"],
    ["Dense-only retrieval misses exact names", "Hybrid BGE-M3 dense + sparse with RRF and reranking", "Retrieval ablation (6.3)"],
    ["LLM-assigned ‘quality’ labels", "Deterministic GRADE-inspired certainty with stated reasons", "Unit tests, live examples (6.6)"],
    ["Identifiable data sent to model providers", "Local PHI guard before any external call", "PHI cases (6.4, 6.5)"],
    ["Generic answers that ignore the patient", "Case Mode: verified facts, case-aware query, safety check", "Case eval (6.5)"],
], [1.9, 2.8, 1.5], "Gap analysis", bold_first_col=True)

# --------------------------------------------------------------------- CH 3
r.chapter("Requirements and Stakeholder Analysis")
r.h2("3.1 Stakeholders")
r.table(["Stakeholder", "Need or concern", "Role in this project"], [
    ["Oncologists and fellows", "Fast, applicable, verifiable evidence; visible uncertainty", "Primary users; intended clinical reviewers"],
    ["Clinical researchers", "Comparison of studies, effect sizes, reproducible search", "Secondary users"],
    ["Pharmacists / safety reviewers", "Interactions with current medications and prior therapy", "Users of the Patient Safety Check output"],
    ["Patients (indirect)", "Privacy; decisions based on applicable evidence", "Not users; no identifiable data is processed"],
    ["Hospitals and medical colleges", "Auditability, data protection, cost", "Potential adopters (Section 7.6)"],
    ["Guide and evaluation panel", "Sound design, honest evidence of results", "Academic reviewers"],
], [1.7, 2.6, 1.9], "Stakeholders", bold_first_col=True)

r.h2("3.2 Stakeholder Engagement Status")
r.p("No structured interviews, surveys or clinician usability sessions have been conducted yet, and none are "
    "reported as if they had been. Requirements are derived from the published literature, from GRADE and "
    "Cochrane methods, and from failure cases found in our own testing. The clinician-review protocol in Section "
    "6.9 is ready to run once access to reviewers is arranged.")

r.h2("3.3 Use Cases")
r.p(f"Figure {r.next_fig()} shows the actors and the main use cases. UC-10 to UC-12 implement the patient-context "
    "extension that Milestone-1 marked as conditional; they operate only on synthetic or de-identified text.")
r.figure(FIG / "fig_usecase.png", "Use-case diagram", width=5.9)

r.h2("3.4 Functional Requirements")
r.p("Table 3.2 lists the final functional requirements with their status. FR-01 to FR-20 come from Milestone-1; "
    "FR-21 to FR-23 were added in the final phase.")
FR = [
    ["FR-01", "Submit an oncology question (≤ 2,000 characters)", "Done"],
    ["FR-02", "Filter by publication year and study type", "Done"],
    ["FR-03", "Normalise the query; expand curated oncology acronyms safely", "Done (acronym defects fixed)"],
    ["FR-04", "Dense and sparse retrieval in parallel with fusion", "Done"],
    ["FR-05", "Rerank and assemble a non-duplicative evidence bundle", "Done"],
    ["FR-06", "Structured answer citing only supplied evidence IDs", "Done"],
    ["FR-07", "Resolve citations to source spans; check numeric claims", "Done"],
    ["FR-08", "Show citations with title, year, study type, passage, PMC link", "Done"],
    ["FR-09", "Abstain on weak retrieval, out-of-scope input or failed verification", "Done"],
    ["FR-10", "Track identifiers, versions, checksums, retraction state", "Partial: stored; retraction refresh not automated"],
    ["FR-11", "Persist query traces with model versions and timings", "Done"],
    ["FR-12", "Restrict access to authenticated users", "Done (session login; single clinician role)"],
    ["FR-13", "User feedback on answers and citations", "Remaining: table exists, no UI"],
    ["FR-14", "Admin view of corpus and index status", "Partial: command-line tools only"],
    ["FR-15", "Indian guidelines and terminology", "Future"],
    ["FR-16", "Detect and display evidence conflicts", "Done (textual + quantitative)"],
    ["FR-17", "Compare population, intervention, outcome and design", "Done (grouping, Applicability Matrix)"],
    ["FR-18", "Extract case facts with units, dates and source spans", "Done (synthetic/de-identified only)"],
    ["FR-19", "Clinician confirmation of facts before use", "Done"],
    ["FR-20", "Case-aware retrieval with a general-search baseline", "Done"],
    ["FR-21", "Patient safety check against medications, history, comorbidities", "Done (new)"],
    ["FR-22", "Effect-size map and GRADE-inspired certainty", "Done (new)"],
    ["FR-23", "Show which model performed each step", "Done (new)"],
]
r.table(["ID", "Requirement", "Status"], FR, [0.6, 3.9, 1.7], "Functional requirements and status", bold_first_col=True)

r.h2("3.5 Non-Functional Requirements and Measured Results")
r.p("Milestone-1 defined acceptance targets. Table 3.3 reports the measured value for each and whether the target "
    "was met. Two targets are only partly met and are reported as such.")
NFR = [
    ["NFR-01 Latency", "p50 < 8 s, p95 < 15 s", f"Answered queries p50 {SAF['indence']['latency_answered_s']['p50']:.1f} s, p95 {SAF['indence']['latency_answered_s']['p95']:.1f} s; 1.46 s excluding the LLM", "Partly (p95 met)"],
    ["NFR-02 Citation coverage", "≥ 95% of material claims cited", "Every claim must carry valid evidence IDs or it is rejected", "Met by construction"],
    ["NFR-03 Citation integrity", "100% resolve to stored text", f"Invalid citation rate {pct(SAF['indence']['invalid_citation_rate'])}", "Met"],
    ["NFR-04 Faithfulness", "≥ 0.90", f"Numeric faithfulness {SAF['indence']['numeric_faithfulness']:.3f}; claim NLI per answer", "Met (automated proxy)"],
    ["NFR-05 Fabricated references", "Zero", "0 (LLM-only baseline: 47/49)", "Met"],
    ["NFR-06 Safe abstention", "Abstain when unsupported", f"Correct abstention {pct(SAF['indence']['correct_abstention_rate'])}", "Met"],
    ["NFR-07 Privacy / security", "No PHI to providers; secure sessions", f"PHI sent: {SAF['indence']['phi_sent_to_provider']}; hashed session tokens; no TLS locally", "Partly"],
    ["NFR-08 Reproducibility", "Versions stored per run", "Model IDs, timings, candidates stored; seeded evals", "Met"],
    ["NFR-09 Maintainability", "Modular, tested", "Modular monolith; 122 automated tests", "Met"],
    ["NFR-10 Accessibility", "Not colour-only; readable", "Icons + text for all statuses; no formal audit", "Partly"],
    ["NFR-11 Conflict transparency", "No silent removal", "Removed content is listed with its reason", "Met"],
    ["NFR-12 Context integrity", "Every fact traceable", "Untraceable facts dropped automatically; traceability 95.8%", "Met"],
    ["NFR-13 Context privacy", "No identifiable data", "Synthetic cases only; local PHI block", "Met"],
]
r.table(["Requirement", "Target", "Measured", "Result"], NFR, [1.5, 1.4, 2.4, 0.9],
        "Non-functional requirements: targets versus measurements", size=9.5, bold_first_col=True)

# --------------------------------------------------------------------- CH 4
r.chapter("Proposed Methodology and System Design")
r.h2("4.1 Design Principles")
r.bullets([
    "**Language models for language only.** The LLM summarises, extracts and judges entailment. Ranking, matching, statistics, certainty and safety rules are deterministic code with unit tests.",
    "**Fail closed.** If the provider is unavailable, identifiers are detected, the entity is unknown or verification fails, the user sees an explicit reason, never a fallback answer.",
    "**Traceability as a data invariant.** Every claim carries evidence IDs; every case fact carries character offsets; every number in the forest plot carries its source sentence.",
    "**Show, do not hide.** Removed claims, conflicting facts, unmatched populations and the general-search baseline remain visible.",
    "**Measure everything.** API and benchmarks share one pipeline, so reported numbers describe the deployed system.",
])

r.h2("4.2 System Architecture")
r.p(f"Indence is a modular monolith (Figure {r.next_fig()}). The browser client talks to a FastAPI service over "
    "JSON. Embedding and reranking models run on the local GPU; vectors live in a local Qdrant server and "
    "documents, traces and answers in SQLite. External LLM services are used for synthesis, case-fact extraction "
    "and claim verification, and are called only after the local identifier check passes.")
r.figure(FIG / "fig_architecture.png", "System architecture", width=6.2)

r.h2("4.3 The Query Pipeline")
r.p(f"Figure {r.next_fig()} shows the seven stages that every question passes through. The same code path "
    "(modules/pipeline/query_pipeline.py) serves the user interface and every evaluation script.")
r.figure(FIG / "fig_pipeline.png", "The Indence reasoning pipeline", width=6.2)
r.bullets([
    "**1 Gate.** A local scanner blocks names, phone numbers, emails, Aadhaar-like and medical-record numbers and exact dates of birth (HIPAA Safe Harbor categories plus Indian identifiers) [35]. Non-oncology questions and drug names absent from the corpus are refused before any LLM call.",
    "**2 Understand.** Rule-based extraction of disease, biomarkers (with +/− notation), line of therapy, setting and requested subgroup; safe acronym expansion.",
    "**3 Retrieve.** BGE-M3 encodes the query once into dense and sparse vectors; Qdrant searches both in parallel; RRF fuses the lists (k = 60) into 50 candidates.",
    "**4 Rank.** The cross-encoder rescores candidates; bounded evidence-hierarchy and population priors break near-ties; MMR selects a diverse bundle within a token budget.",
    "**5 Appraise.** Population matching, evidence grouping, contradiction checks, effect-size extraction, heterogeneity statistics and certainty rating.",
    "**6 Synthesise.** The LLM returns strict JSON: recommendation, reasoning, supporting evidence, alternatives, bottom line and atomic claims, each citing evidence IDs from the bundle only.",
    "**7 Verify.** Citation audit, numeric check and NLI verification of every claim and prose section; contradicted content is removed and listed; the abstention policy decides between brief and abstention.",
])

r.h2("4.4 Retrieval and Ranking Design")
r.p("Reciprocal rank fusion scores each candidate document d from the dense and sparse ranked lists R:")
r.eq("RRF(d) = Σᵣ 1 / (k + rankᵣ(d)),   r ∈ {dense, sparse},   k = 60")
r.p("The evaluation in Section 6.3 showed that the October design let study-type and population priors (up to "
    "≈ 1.4) overwhelm the reranker relevance score (at most ≈ 0.3), demoting 48 relevant papers. The final design "
    "is relevance-first: priors are bounded tie-breakers, fixed a priori rather than tuned on the benchmark:")
r.eq("score = s(rerank) + 0.06 · clip((w(hier) − 1) / 1.5, −0.33, 1) + 0.04 · clip(s(pop) / 300, −1, 1)")
r.p("Candidates that belong to an explicitly excluded subgroup receive a hard penalty and are removed. MMR then "
    "selects passages that are relevant but not redundant, using the dense vectors already stored in Qdrant "
    "instead of re-encoding them.")

r.h2("4.5 Evidence Appraisal")
r.h3("4.5.1 Quantitative Evidence Map")
r.p("Hazard, odds and risk ratios with 95% confidence intervals are extracted by pattern matching from the "
    "retrieved passages, together with the endpoint (OS, PFS, DFS, response) and the exact source sentence. "
    "Prognostic associations (for example a biomarker’s HR) are separated from treatment comparisons. For each "
    "endpoint, estimates yᵢ = ln(ratio) are combined with inverse-variance weights wᵢ = 1/SEᵢ², where "
    "SEᵢ = (ln Uᵢ − ln Lᵢ) / 3.92 for a 95% interval (Lᵢ, Uᵢ):")
r.eq("Q = Σ wᵢ (yᵢ − ȳ)²,     I² = max(0, (Q − df) / Q)")
r.eq("τ² = max(0, (Q − df) / (Σwᵢ − Σwᵢ² / Σwᵢ)),     df = k − 1")
r.p("With three or more comparable estimates, an exploratory DerSimonian–Laird pooled estimate is drawn and "
    "labelled as screening, not meta-analysis. A high I² or opposite-direction non-overlapping intervals raise a "
    "quantitative conflict signal.")
r.h3("4.5.2 GRADE-Inspired Certainty Engine")
r.table(["Domain", "Signal used", "Effect"], [
    ["Starting level", "Best design in the bundle (meta-analysis/RCT vs observational)", "High or Low"],
    ["Risk of bias", "Phase II, single-arm, small sample, ‘retrospective’ wording", "−1 level"],
    ["Inconsistency", "I² > 50% or textual contradiction flags", "−1 level"],
    ["Indirectness", "Population mismatch or low applicability", "−1 level"],
    ["Imprecision", "CI crosses 1 or ratio of limits > 3", "−1 level"],
    ["Publication bias", "Not assessable from retrieved text", "Shown as ‘not assessed’"],
], [1.4, 3.4, 1.4], "Certainty domains and deterministic signals", bold_first_col=True)

r.h2("4.6 Verification and Abstention Policy")
r.p("All atomic claims and the prose sections of the brief are sent, with only the passages each one cites, "
    "to the verifier in a single request. The verifier (Gemini Decision 1) returns a label (supported, "
    "contradicted or not established) with calibrated probabilities. The policy is:")
r.bullets([
    "A claim or section contradicted by its own source is removed and listed under ‘Removed by verification’.",
    "The whole answer is withheld if more than one third of claims, or the recommendation itself, is contradicted, or if fewer than half of the claims are supported.",
    "Every remaining section shows a badge with its verdict and confidence.",
    "Numbers in claims must appear in the cited evidence; citation IDs outside the bundle invalidate the claim.",
])

r.h2("4.7 Patient Case Mode Design")
r.p("Case Mode turns pasted, de-identified case text into a case-aware evidence search in five steps:")
r.bullets([
    "**PHI guard (local).** Identifiers block the request before anything leaves the machine.",
    "**Span-verified extraction.** The LLM proposes facts as {field, value, unit, date, source_quote}. Each quote must be found in the text, giving character offsets; untraceable facts are dropped and counted. Conflicting values (e.g. two PD-L1 results) start unchecked and the clinician chooses.",
    "**Missing-data checklist.** Disease-specific decision-critical variables (e.g. ROS1 for NSCLC, PIK3CA/ESR1 only for HR-positive breast cancer) that are absent are listed.",
    "**Applicability Matrix.** For each retrieved study and each patient fact, a deterministic judgement (matches, differs, not reported) with the justifying sentence. It encodes drug-to-biomarker implications, EGFR exon 20 as a distinct subgroup, adjuvant versus metastatic settings and treatment-naive status.",
    "**Patient Safety Check.** Drugs from the answer and the retrieved studies are checked against current medications (10 curated interaction rules), comorbidities (9 rules), organ function and earlier treatments; an AI screen flags further pairs only above probability 0.85, labelled ‘needs review’.",
])
r.p("To guard against confirmation bias, the general (non-personalised) search always runs alongside the "
    "case-aware search and remains visible.")

r.h2("4.8 Data Flow and Interaction Design")
r.p(f"Figure {r.next_fig()} gives the context (level 0) and level-1 data-flow diagrams, and Figure "
    f"{r.prefix}.{r.fig_no + 2} the sequence of calls for one question.")
r.figure(FIG / "fig_dfd.png", "Data-flow diagrams, levels 0 and 1", width=4.6)
r.figure(FIG / "fig_sequence.png", "Sequence diagram for a clinical question", width=6.1)

r.h2("4.9 Data Design")
r.p(f"The relational schema (Figure {r.next_fig()}) separates immutable source documents and their revisions from "
    "sections, evidence units and chunks, so that every citation resolves to a character span in a specific "
    "revision. Query runs, answers, claims, citations and retrieval traces make each answer reproducible.")
r.figure(FIG / "fig_er.png", "Entity–relationship diagram (core tables)", width=6.1)

r.h2("4.10 Module Design")
r.figure(FIG / "fig_modules.png", "Backend module structure", width=6.1)

# --------------------------------------------------------------------- CH 5
r.chapter("Implementation")
r.h2("5.1 Technology Stack")
r.table(["Layer", "Technology", "Reason for choice"], [
    ["Frontend", "Next.js 16 (React), plain CSS, inline SVG charts", "Single-page clinical workspace; no chart library needed for the forest plot"],
    ["API", "FastAPI (Python 3.13), SQLAlchemy, Alembic", "Async I/O for parallel retrieval and model calls; typed request handling"],
    ["Embeddings", "BAAI/bge-m3 (FlagEmbedding, fp16)", "Dense and learned sparse vectors from one model; multilingual"],
    ["Reranker", "BAAI/bge-reranker-v2-m3", "Strong open cross-encoder that fits in 8 GB VRAM alongside BGE-M3"],
    ["Vector store", "Qdrant server (local), named dense + sparse vectors", "Native hybrid search; stores vectors for MMR reuse"],
    ["Database", "SQLite", "Zero-administration, file-based; adequate for a single-node prototype"],
    ["Generator / extractor", "Gemini Flash models via OpenAI-compatible API; OpenRouter free models as fallback", "Best quality available at zero cost; ordered chain survives per-model quotas"],
    ["Verifier", "Gemini Decision 1 (Decisions API)", "Calibrated probabilities; 10,000 free requests/day"],
    ["Hardware", "Laptop, RTX 3070 Ti (8 GB), 32 GB RAM", "Demonstrates that the system runs without cloud GPUs"],
], [1.1, 2.4, 2.7], "Technology stack", bold_first_col=True)

r.h2("5.2 Corpus and Index")
r.table(["Item", "Count"], [
    ["Open-access oncology papers (PMC), published Sept 2016 – July 2026", "19,057"],
    ["  of which RCTs / meta-analyses / systematic reviews / guidelines / other", "7,533 / 6,887 / 3,934 / 329 / 374"],
    ["Structured sections (methods, results, tables, …)", "131,704"],
    ["Evidence units", "209,379"],
    ["Chunks in SQLite", "420,482"],
    ["Points indexed in Qdrant (dense 1024-d + sparse)", "412,846"],
], [4.6, 1.6], "Corpus statistics (from the project database)")
r.p("The corpus was ingested and embedded before Milestone-1 and was not rebuilt in the final phase. A manifest "
    "validator, checksums, parser versions and a corpus snapshot identifier make the index rebuildable.")

r.h2("5.3 Retrieval Implementation and Performance Fixes")
r.p("The audit found that query-side sparse vectors were always empty: with fp16 enabled, the encoder produced "
    "float16 activations while the sparse head stayed float32, the resulting error was swallowed by the "
    "library’s retry loop, and ‘hybrid’ search silently ran dense-only. Casting the heads to the encoder’s dtype "
    "fixed this; stored document vectors were unaffected (cosine 1.0 on re-encoding). Six performance defects "
    "were also fixed (Table 5.3).")
r.table(["Defect", "Fix"], [
    ["Qdrant filters on payload keys that were never written, so every filtered query ran twice", "Removed; population logic applied by the matcher"],
    ["Dense and sparse searches ran sequentially", "Run concurrently with asyncio.gather"],
    ["~250 SQL queries per question to hydrate candidates", "One batched join"],
    ["All 50 candidates re-encoded on the GPU for MMR", "Reuse vectors stored in Qdrant"],
    ["Pure-Python O(n²) cosine loops", "One numpy similarity matrix"],
    ["GPU inference blocked the async event loop", "Worker thread behind a lock"],
], [3.4, 2.8], "Performance defects fixed", size=10)

r.h2("5.4 LLM Integration on a Zero Budget")
r.p("The OpenRouter credit was exhausted early in the final phase; before the fix, this produced a silent template "
    "answer. The adapter now accepts an ordered, comma-separated model chain with provider prefixes "
    "(gemini:, groq:, cerebras:, OpenRouter by default). Each model is tried in turn; a rate-limited model "
    "cools down for the reset time the provider reports (capped at six hours), an overloaded one for two "
    "minutes, and transient errors are retried once. Gemini Flash models allow about 20 requests per day each on "
    "the free tier and Pro models none, so the chain lists several Flash variants. Primary models receive the "
    "full prompt; only free fallback models get a compact output instruction and low reasoning effort. All "
    "claims of an answer are verified in one Decision API request, which keeps a question to about two "
    "generator calls and one verifier call.")

r.h2("5.5 Safety Engineering")
r.table(["#", "Defect found in the Milestone-1 build", "Fix"], [
    ["S1", "Provider failure returned a template answer marked as verified", "Explicit ‘generator unavailable’ abstention"],
    ["S2", "Synthesis displayed after the safety gate abstained", "No synthesis returned on abstention"],
    ["S3", "‘HER2+’, ‘EGFR+’ never matched (regex word boundary after +)", "Look-ahead pattern; tests"],
    ["S4", "‘HER2-’ matched inside ‘HER2-positive’, rejecting the requested subgroup", "Requested biomarker can never be rejected"],
    ["S5", "‘or’ expanded to ‘odds ratio’; ‘HR’ in ‘HR+’ to ‘hazard ratio’", "Uppercase standalone tokens only, single pass"],
    ["S6", "Every answer warned ‘case series or lower’", "Rank read from the correct object"],
    ["S7", "A paper could contradict itself", "Pairs must come from different sources"],
    ["S8", "Any RCT labelled phase III; ‘recommendation’ made a paper a guideline", "Explicit signals required"],
    ["S9", "Unauthenticated requests ran as a default clinician", "HTTP 401; no hard-coded re-login"],
    ["S10", "Traces recorded ‘mock-llm’", "Real model IDs, tokens and errors stored"],
], [0.4, 3.1, 2.7], "Safety defects found and fixed (each pinned by a regression test)", size=10, bold_first_col=True)

r.h2("5.6 User Interface")
r.p("The interface has three tabs: Clinical Search, Patient Case and Execution Audit. Figures 5.1 to 5.6 are "
    "screenshots of the running system captured automatically with Playwright; no image has been edited.")
ans = crop("ss_answer.png", 470, 1990, "ss_answer_crop.png")
r.figure(ans, "Verified answer: each section carries the verifier’s verdict and confidence; citation chips link to the cited passages", width=4.3)
r.figure(FIG / "ss_model_trace.png", "‘Which model did what’: per-step attribution of the local models, deterministic rules and LLMs", width=6.0)
r.figure(FIG / "ss_forest.png", "Quantitative Evidence Map: effect sizes copied verbatim from the retrieved passages", width=5.6)
r.figure(crop_x("ss_case_facts.png", 925, 0, 1960, 1500, "ss_case_facts_crop.png"),
         "Patient Case Mode: extracted facts with character offsets and a flagged PD-L1 conflict (the local PHI check passed)", width=4.3)
r.figure(FIG / "ss_applicability.png", "Applicability Matrix for a synthetic HR-positive breast cancer case", width=6.0)
r.figure(crop("ss_safety.png", 0, 640, "ss_safety_crop.png"),
         "Patient Safety Check (first two of five flags): interactions with a current medication, for drugs in the answer and in the retrieved studies", width=6.0)
r.p("In the case shown in Figure 5.5, the general search scored slightly higher mean applicability (0.927) than "
    "the case-aware search (0.875). Both lists are shown precisely so that such cases are visible; across the "
    "eight benchmark cases the case-aware search was better in seven (Section 6.5).")

r.h2("5.7 Reproducibility")
r.p("Evaluation scripts in evals/ call the same QueryPipeline as the API and write JSON results; make_charts.py "
    "draws every chart from those files, and this report’s tables are generated from them as well. Fixed seeds "
    "are used for sampling and bootstrap resampling. Provider failures are recorded as errors and never counted "
    "as abstentions.")

# --------------------------------------------------------------------- CH 6
r.chapter("Testing, Results and Validation")
r.h2("6.1 Test Strategy")
r.p("Testing has three layers: automated unit and integration tests for deterministic logic and the API; "
    "benchmark evaluations of retrieval, safety and Case Mode on labelled sets; and live end-to-end checks in "
    "the browser for each user flow. The number of passing automated tests rose from 72 at the start of the "
    "final phase to 122.")
r.table(["Suite", "What it pins down", "Tests"], [
    ["test_regression_fixes.py", "Every safety and retrieval defect in Tables 5.3 and 5.4, and the verifier fix", "18"],
    ["test_effect_map.py", "Effect extraction, treatment vs prognostic HRs, Q, I², τ²", "8"],
    ["test_case_mode.py", "PHI guard, span verification, conflicts, missing data, applicability rules", "15"],
    ["test_safety_check.py", "Interaction, comorbidity and treatment-history rules", "4"],
    ["test_api.py and others", "Auth, abstention without synthesis, pipeline contracts, ingestion", "77"],
    ["**Total**", "", "**122 passing**"],
], [1.9, 3.4, 0.9], "Automated test suites")

r.h2("6.2 Evaluation Datasets")
r.table(["Set", "Size", "Construction", "Label quality"], [
    ["Retrieval (silver, known-item)", "119 questions (88 strict)", "Stratified sample of RCTs and meta-analyses; an LLM wrote one clinical question per paper; that paper is the relevant item", "Silver; values are lower bounds"],
    ["Safety and grounding", "40 questions", "12 answerable, 8 subgroup traps, 8 fictional drugs, 6 out-of-scope, 6 with identifiers", "Author-labelled"],
    ["Patient cases", "8 cases, 85 gold facts + PHI demo", "Synthetic oncology cases (NSCLC, breast, colorectal, ovarian, …)", "Author-labelled"],
], [1.4, 1.2, 2.5, 1.1], "Evaluation datasets", size=9.5, bold_first_col=True)

r.h2("6.3 Retrieval Ablation (RQ1)")
modes = [("sparse", "Sparse only (BGE-M3 lexical)"), ("dense", "Dense only (BGE-M3)"),
         ("hybrid", "Hybrid (RRF, k = 60)"), ("hybrid_rerank", "Hybrid + cross-encoder")]
rows = [[name, m(k, "recall@5"), m(k, "recall@10", True), m(k, "recall@20"), m(k, "mrr@10"), m(k, "ndcg@10")]
        for k, name in modes]
rows.append(["Full Indence, original Oct-22 blend", m("full", "recall@5", src=ABL_OLD), m("full", "recall@10", True, ABL_OLD),
             m("full", "recall@20", src=ABL_OLD), m("full", "mrr@10", src=ABL_OLD), m("full", "ndcg@10", src=ABL_OLD)])
rows.append(["Full Indence, relevance-first blend", m("full", "recall@5"), m("full", "recall@10", True), m("full", "recall@20"),
             m("full", "mrr@10"), m("full", "ndcg@10")])
r.table(["Configuration", "R@5", "R@10 (95% CI)", "R@20", "MRR@10", "nDCG@10"], rows,
        [2.0, 0.6, 1.3, 0.6, 0.8, 0.8], f"Retrieval ablation on all {ABL['n_queries']} questions (document level)", size=9.5)
r.figure(EVF / "retrieval_ablation_all_queries.png", "Retrieval ablation, all questions, with 95% bootstrap intervals", width=5.9)
r.p("**Each retrieval layer adds value.** Hybrid fusion beats either channel on Recall@10, and the cross-encoder "
    f"adds about 0.10 MRR. On the {ABL['n_paraphrased']} strict questions, which share no five-word phrase with "
    f"the paper title, Recall@10 is {m('sparse', 'recall@10', subset='paraphrased_only')} (sparse), "
    f"{m('dense', 'recall@10', subset='paraphrased_only')} (dense), {m('hybrid', 'recall@10', subset='paraphrased_only')} "
    f"(hybrid) and {m('hybrid_rerank', 'recall@10', subset='paraphrased_only')} (with reranking), so the gains are "
    "not caused by title overlap.")
r.p("**The evaluation exposed a ranking defect.** The original blend let study-type and population priors swamp "
    "relevance and demoted 48 relevant papers. After the relevance-first redesign (Section 4.4), Recall@10 rose "
    "from 0.672 to 0.891, statistically level with hybrid + reranking (overlapping intervals).")
r.p("**Why the full system still scores lower on MRR.** A per-question comparison against hybrid + reranking "
    "(Table 6.4) shows that Indence ranks the target paper identically in 71 of 119 questions and higher in 10. "
    "In 24 questions the paper moves from first place to second, third or fourth, because a guideline, "
    "meta-analysis or better population match was nudged above it. Moving from rank 1 to rank 2 halves the "
    "reciprocal rank, so these small moves explain about 0.12 of the 0.135 MRR gap. The known-item benchmark "
    "credits only the one source paper, so a different paper that answers the question equally well counts as a "
    "miss. Two real errors remain: the population filter wrongly removed two target papers.")
r.table(["Rank of the target paper, Indence vs hybrid + rerank", "Questions"], [
    ["Same rank", "71"], ["Indence ranks it higher", "10"], ["Lower by 1–3 places (was #1, now #2–#4)", "24"],
    ["Lower by more than 3 places", "12"], ["Removed by the population filter (error)", "2"],
    ["Target in the top 10: hybrid + rerank / Indence", "110 / 106"],
], [4.5, 1.2], "Per-question rank comparison", size=10)

r.h2("6.4 Safety and Grounding Against Baselines (RQ2)")
r.p("All three systems used the same generator (gemini-3.1-flash-lite), so differences come from the "
    "architecture. Naive RAG retrieves the dense top-5 passages and asks the model to answer with citations. "
    "PubMed identifiers produced by the LLM-only system were checked with NCBI E-utilities and spot-checked by "
    "hand.")
L, N, I = SAF["llm_only"], SAF["naive_rag"], SAF["indence"]
r.table(["Metric", "LLM only", "Naive RAG", "Indence"], [
    ["Correct abstention (out-of-scope, fictional drug, PHI)", pct(L["correct_abstention_rate"]), pct(N["correct_abstention_rate"]), f"**{pct(I['correct_abstention_rate'])}**"],
    ["Fictional-drug questions answered", f"{round((1 - L['abstention_by_category']['fictional_drug']) * 8)} / 8", f"{round((1 - N['abstention_by_category']['fictional_drug']) * 8)} / 8", f"**{round((1 - I['abstention_by_category']['fictional_drug']) * 8)} / 8**"],
    ["PHI sent to the external provider", L["phi_sent_to_provider"], N["phi_sent_to_provider"], f"**{I['phi_sent_to_provider']}**"],
    ["Answer rate, 20 answerable questions", pct(L["answer_rate_on_answerable"]), pct(N["answer_rate_on_answerable"]), pct(I["answer_rate_on_answerable"])],
    ["References that resolve to the cited source", f"{pct(1 - L['fabricated_ref_rate'])} ({L['fabricated_or_misattributed_refs']} wrong)", pct(1 - N["invalid_citation_rate"]), f"**{pct(1 - I['invalid_citation_rate'])}**"],
    ["Numeric faithfulness", "n/a", f"{N['numeric_faithfulness'] * 100:.1f}%", f"**{I['numeric_faithfulness'] * 100:.1f}%**"],
    ["Wrong-subgroup citations", "n/a", f"{N['subgroup_leakage_rate'] * 100:.1f}%", f"**{I['subgroup_leakage_rate'] * 100:.1f}%**"],
    ["Latency p50 / p95, answered (s)", f"{L['latency_answered_s']['p50']:.1f} / {L['latency_answered_s']['p95']:.1f}", f"{N['latency_answered_s']['p50']:.1f} / {N['latency_answered_s']['p95']:.1f}", f"{I['latency_answered_s']['p50']:.1f} / {I['latency_answered_s']['p95']:.1f}"],
], [2.6, 1.2, 1.2, 1.2], "Three-system safety and grounding benchmark (40 questions)", size=10)
r.figure(EVF / "safety_comparison.png", "Safety benchmark: Indence vs naive RAG vs LLM only", width=5.9)
r.p("The LLM-only system named real trials (FLAURA, DESTINY-Breast04) but attached PubMed identifiers belonging "
    "to unrelated papers on vascular surgery, lung stem cells and plant biology. Naive RAG avoided most fictional "
    "drugs but still sent identifiable data to the provider in every PHI case and cited a wrong subgroup in 6% "
    "of answers. Indence withheld two answerable questions: in one, too few claims could be verified; in the "
    "other, the verifier found that a generated claim contradicted its own cited source. This lower answer rate "
    "is the intended trade-off: an unverifiable answer is not shown. Indence is slower because it does more "
    "work, but its p95 remains within the 15-second target.")
r.p("**A verifier defect found after this benchmark.** The passages cited by a claim or section were joined and "
    "cut at 4,000 characters before verification, so text citing three or more passages was judged on only part "
    "of its evidence. On the trastuzumab deruxtecan question, the recommendation was rated ‘not established’ "
    "(48%) with truncated passages and ‘supported’ (95%) with the full passages. The fix sends every cited passage "
    "once, in full, and judges each claim against exactly its own passages; a regression test pins it. After the "
    "fix, all six sections of that answer verified as supported. The figures in Table 6.5 were measured before "
    "the fix and are reported unchanged; the 90% answer rate may be an underestimate, because the abstention "
    "for ‘too few claims verified’ could have been caused by this defect.")

r.h2("6.5 Patient Case Mode (RQ3)")
r.table(["Metric", "Result"], [
    ["Gold-fact recall, LLM extraction + span verification (gemini-2.5-flash)", "**98.8%** (84 / 85)"],
    ["Gold-fact recall, deterministic regex fallback", f"{CASE['regex_fallback']['gold_recall_micro'] * 100:.1f}%"],
    ["Traceability (verified ÷ proposed facts)", "95.8%; 3 untraceable facts dropped automatically"],
    ["Conflict detection (PD-L1 80% vs 45%)", "Detected"],
    ["Mean applicability of top-8 studies: general → case-aware", f"**{CASE['applicability']['general_mean_of_means']:.3f} → {CASE['applicability']['case_aware_mean_of_means']:.3f}**"],
    ["‘Different population’ studies in top-8 (sum over cases)", f"{CASE['applicability']['general_different_population_total']} → {CASE['applicability']['case_different_population_total']}"],
    ["Cases improved", f"{CASE['applicability']['cases_improved']} / {CASE['n_cases']}"],
    ["PHI demo case", "Blocked locally; none of the 8 synthetic cases falsely blocked"],
], [4.0, 2.2], "Patient Case Mode evaluation (8 synthetic cases)", size=10)
r.figure(EVF / "case_mode_eval.png", "Case Mode: fact recall and applicability of general vs case-aware search", width=5.9)
r.p("The applicability rows were computed with facts from the regex fallback because the generator’s daily "
    "quota was exhausted during that run; the improvement therefore holds even with the weaker extractor. Live "
    "review of Case Mode output found and fixed clinical logic errors that the metrics alone would have missed, "
    "for example exon 20 trials matched to an exon 19 patient and adjuvant trials matched to stage IV disease "
    "(Appendix B).")

r.h2("6.6 Verifier and Safety-Check Validation")
r.bullets([
    "**Claim verifier.** On a labelled probe, Gemini Decision 1 judged 4 of 4 cases correctly with confidence 0.98–0.996, including a trap where the claim said 18.9 months and the source 19.8 months. All claims of an answer are verified in one request.",
    "**Safety check.** On a live HR-positive breast case the AI screen identified capivasertib × clarithromycin (p = 0.94) and elacestrant × clarithromycin (p = 0.87), both CYP3A4 label warnings, which were then added to the curated rules. It also produced one false positive (alpelisib × denosumab, p = 0.73), which led to the 0.85 display threshold and the separate ‘needs review’ severity.",
    "**Certainty and effect map.** Unit tests cover extraction of ratios with intervals, the treatment/prognostic split, the Q, I² and τ² formulas, and each certainty downgrade.",
])

r.h2("6.7 Latency (RQ4)")
r.table(["Stage", "Before p50 (52 historical runs)", "After p50 (10 runs)", "Speed-up"], [
    ["Retrieval (encode, dense ‖ sparse, RRF, hydrate)", "3.86 s", "0.28 s", "13.7×"],
    ["Rerank + MMR context assembly", "5.76 s", "1.17 s", "4.9×"],
    ["Whole pipeline excluding the LLM", "≈ 9.6 s", "1.46 s", "≈ 6.6×"],
    ["LLM synthesis", "34.8 s (llama-3.3-70b)", "6–25 s (Gemini Flash family)", "—"],
], [2.6, 1.4, 1.4, 0.8], "Latency before and after the final phase", size=10)
r.figure(EVF / "latency_before_after.png", "Per-stage latency before and after", width=5.6)
r.p(f"End-to-end, answered questions took {I['latency_answered_s']['p50']:.1f} s at the median and "
    f"{I['latency_answered_s']['p95']:.1f} s at the 95th percentile in the safety benchmark. The p95 target "
    "(15 s) is met and the p50 target (8 s) is not; almost all of the remaining time is the external generator "
    "and verifier, which local work cannot reduce further on free tiers.")

r.h2("6.8 Demonstration Evidence")
r.figure(FIG / "ss_abstain_entity.png", "Abstention on a fictional drug: no answer is generated, and the retrieved sources remain available", width=5.8)
r.p("Additional live checks before submission: the PHI demo case is blocked before extraction; the Execution "
    "Audit tab shows per-stage timings and per-candidate dense, sparse, RRF and reranker scores; and provider "
    "outages produce an explicit ‘generator unavailable’ message.")

r.h2("6.9 User and Stakeholder Feedback")
r.p("No clinician or user feedback has been collected yet. The planned protocol is: (1) three to five oncology "
    "clinicians or fellows each review ten briefs drawn from the safety set and two Case Mode outputs; (2) for "
    "each claim, they rate support by the cited passage (supported, partly, not supported) and whether "
    "applicability judgements are correct; (3) they complete a five-question usefulness and trust questionnaire "
    "and a task-time comparison with their usual PubMed search. Agreement with the automated verifier (Cohen’s "
    "κ) would show how far the automated metrics in this chapter can be trusted.")

# --------------------------------------------------------------------- CH 7
r.chapter("SDG Impact, Discussion, Limitations and Future Scope")
r.h2("7.1 Sustainable Development Goals")
r.table(["SDG", "Contribution", "Boundary"], [
    ["SDG 3: Good Health and Well-being (primary)", "Faster access to applicable, verified oncology evidence; interaction screening against a patient’s medications; refusal to answer when evidence is absent", "Decision support for professionals only; no outcome claims without clinical study"],
    ["SDG 9: Industry, Innovation and Infrastructure", "Reusable, auditable evidence infrastructure running on commodity hardware", "Prototype; not production-hardened"],
    ["SDG 10: Reduced Inequalities", "Runs on one laptop GPU and free model tiers, making evidence tools affordable for smaller hospitals and colleges", "Corpus is English-language open access only"],
], [1.6, 2.8, 1.8], "SDG alignment [1, 2]", bold_first_col=True)

r.h2("7.2 Discussion: Why Indence Is Not ‘Just a RAG Chatbot’")
r.p("A RAG chatbot retrieves passages and asks a model to answer. The evaluation isolates what the additional "
    "layers contribute. With the same generator, naive RAG answered a question about a fictional drug, cited a "
    "wrong subgroup in 6% of answers and sent identifiable data to the provider in all six PHI cases; Indence did "
    "none of these. These differences come from components that are not RAG: deterministic gates, population "
    "logic, verification and an abstention policy. Beyond safety, Indence produces information a chatbot cannot: "
    "a forest plot whose every number is traceable to a sentence, a certainty rating with explicit reasons, an "
    "Applicability Matrix that judges each study against each patient fact, and a medication-aware safety "
    "check. The LLM is one replaceable component; the reasoning around it is the contribution.")
r.p("The evaluation also shows a cost. Ranking evidence by quality and fit lowers known-item MRR, and "
    "verification withholds some answerable questions and adds latency. We consider both acceptable for a "
    "clinical tool, but they are design choices that clinicians should be able to inspect, and the interface "
    "makes them visible.")

r.h2("7.3 Updated Risk Analysis")
r.table(["Risk", "Status / likelihood", "Mitigation in place"], [
    ["R1 Unsupported clinical claims", "Reduced", "Evidence-ID constraint, claim + section NLI, removal and abstention policy"],
    ["R2 Retrieval misses decisive study", "Medium", "Hybrid retrieval + reranking; ablation monitoring; general baseline in Case Mode"],
    ["R5 Patient data sent to a provider", "Low", "Local PHI guard before any external call; 0/6 leaks"],
    ["R6 Mock components create false confidence", "Closed", "Mock answer removed; fake evaluation script superseded"],
    ["R13 Conflicts hidden", "Reduced", "Textual and quantitative conflict signals; removed content listed"],
    ["R14 Wrong case-fact extraction", "Reduced", "Span verification, conflict flags, clinician confirmation"],
    ["R15 Personalisation causes confirmation bias", "Reduced", "General search always shown beside the case-aware search"],
    ["R16 Free-tier quotas and outages (new)", "High", "Multi-model chain with cooldowns; explicit unavailability message"],
    ["R17 Verifier false positives or negatives (new)", "Medium", "Calibrated threshold, visible confidence, human review planned"],
    ["R18 Safety rules incomplete (new)", "High", "Rules labelled as screening; AI screen as backup; pharmacist review advised"],
    ["R11 No clinician participation yet", "High", "Protocol ready (Section 6.9)"],
], [2.0, 1.1, 3.1], "Updated risk register", size=10, bold_first_col=True)

r.h2("7.4 Limitations")
r.bullets([
    "Retrieval labels are silver (one relevant paper per question), so all retrieval metrics are lower bounds.",
    "The safety set (40 questions) and the eight cases were written and labelled by the authors; there is no clinician validation yet.",
    "Applicability is judged from the retrieved passage and title, not from full eligibility criteria, so many cells are honestly ‘not reported’.",
    "Effect extraction covers ratios with confidence intervals only; medians, absolute risks and survival curves are not extracted. The risk-of-bias signal is a wording proxy, not a RoB-2 appraisal.",
    "The curated interaction rules are a small, label-based set; the corpus is open-access only and may miss key paywalled trials and current guidelines.",
    "Free-tier quotas mean the best generator is not always available; the comparisons used flash-lite.",
])

r.h2("7.5 Completed and Remaining Work")
r.table(["Completed for final evaluation", "Remaining / future"], [
    ["Stabilised pipeline; 10 safety and 6 performance defects fixed", "Clinician review study and questionnaire (Section 6.9)"],
    ["Real hybrid retrieval and evaluation harness", "Feedback UI (FR-13) and admin dashboard (FR-14)"],
    ["Patient Case Mode, Applicability Matrix, Safety Check", "Full-text eligibility extraction for applicability"],
    ["Evidence map, heterogeneity and certainty engine", "Automated retraction and update refresh (FR-10)"],
    ["Claim and section verification with Decision 1", "Indian guidelines and drug terminology (FR-15)"],
    ["Model attribution UI; 122 automated tests", "TLS, role-based access and security review before any pilot"],
], [3.1, 3.1], "Completed versus remaining work", size=10)

r.h2("7.6 Future Scope and Startup Potential")
r.p("The most direct path is an institutional evidence workspace for oncology departments and medical colleges "
    "in India: deployed on-premises (which the single-GPU design allows), connected to the hospital’s own "
    "formulary and protocols, and offering tumour-board briefs with applicability and interaction checks. The "
    "same verified-evidence layer supports adjacent products: literature surveillance alerts when a new trial "
    "changes the evidence for a saved question, systematic-review screening assistance, and pharmacy "
    "interaction review. Before any clinical pilot, the system would need clinician validation, a licensed "
    "guideline corpus, security hardening and an assessment under India’s Digital Personal Data Protection "
    "rules and medical-device software guidance [17, 18, 22].")

r.h2("7.7 Ethical Considerations")
r.p("Indence processes only synthetic or de-identified text, blocks identifiers locally, displays a screening "
    "disclaimer on every safety output and never issues treatment orders. Its outputs are intended for "
    "qualified professionals, who remain responsible for decisions [11, 21].")

# --------------------------------------------------------------------- CH 8
r.chapter("Conclusion")
r.p("Indence set out to answer oncology evidence questions without the failure modes that make general AI "
    "unsafe in clinical use. The final system retrieves from 19,057 indexed papers with genuinely hybrid "
    "retrieval, appraises population fit, effect sizes and certainty with deterministic and inspectable logic, "
    "verifies every claim and section against its cited source, and abstains when it cannot support an answer. "
    "Patient Case Mode, proposed as a conditional extension at Milestone-1, is implemented with source-verified "
    "facts, an Applicability Matrix and a medication-aware safety check, and every model involved is named in "
    "the interface.")
r.p("The evaluation supports the central hypothesis. With the same generator, Indence eliminated fabricated "
    "references, answers about fictional drugs, identifier leakage and wrong-subgroup citations, which the "
    "LLM-only and naive RAG baselines did not, at the cost of withholding two of twenty answerable questions and "
    "of higher latency. The evaluation also did what evaluations are for: it exposed a silent retrieval defect "
    "and a ranking defect that had made the October system worse than its components, and both were fixed and "
    "re-measured. The main open question, whether clinicians find the briefs correct and useful, requires the "
    "clinician study that is the first item of future work. Indence is a research prototype, not a clinically "
    "validated device.")

# --------------------------------------------------------------------- REFERENCES
r.front_heading("References", toc=True)
REFS = [
    "United Nations, Department of Economic and Social Affairs. Goal 3: Ensure healthy lives and promote well-being for all at all ages. Sustainable Development Goals.",
    "United Nations, Department of Economic and Social Affairs. Goal 9: Build resilient infrastructure, promote inclusive and sustainable industrialization and foster innovation.",
    "International Agency for Research on Cancer. Global cancer statistics 2024: GLOBOCAN estimates of incidence and mortality worldwide. 2026.",
    "World Health Organization. Health workforce. WHO Health Topics. Accessed 14 October 2026.",
    "World Health Organization. Medication Without Harm: Global Patient Safety Challenge. Accessed 14 October 2026.",
    "National Center for Biotechnology Information. About PubMed. U.S. National Library of Medicine; 2025.",
    "Bhattacharyya M, Miller VM, Bhattacharyya D, Miller LE. High rates of fabricated and inaccurate references in ChatGPT-generated medical content. Cureus. 2023;15(5):e39238.",
    "Wei Q, Yao Z, Cui Y, Wei B, Jin Z, Xu X. Evaluation of ChatGPT-generated medical responses: a systematic review and meta-analysis. J Biomed Inform. 2024;151:104620.",
    "Liu S, McCoy AB, Wright A. Improving large language model applications in biomedicine with retrieval-augmented generation: a systematic review, meta-analysis, and clinical development guidelines. J Am Med Inform Assoc. 2025. doi:10.1093/jamia/ocaf008.",
    "Low YS, Jackson ML, Hyde RJ, et al. Answering real-world clinical questions using large language model, retrieval-augmented generation, and agentic systems. Digital Health. 2025;11.",
    "World Health Organization. Ethics and governance of artificial intelligence for health: guidance on large multi-modal models. Geneva: WHO; 2024.",
    "Elsevier. ClinicalKey AI: generative AI for clinicians. Product information, accessed 14 October 2026.",
    "GRADE Working Group. GRADE system and suggested minimal requirements for stating that the GRADE approach was used. Updated 2023.",
    "Wolters Kluwer. UpToDate clinical decision support / UpToDate Expert AI. Product information, accessed 14 October 2026.",
    "Elsevier. How does ClinicalKey AI work? ClinicalKey AI Support Center. 2026.",
    "National Center for Biotechnology Information. PMC Open Access Subset and PMC article datasets: terms of use. Accessed 14 October 2026.",
    "Ministry of Electronics and Information Technology, Government of India. Digital Personal Data Protection Rules, 2025.",
    "U.S. Food and Drug Administration. Clinical Decision Support Software: Guidance for Industry and FDA Staff. 2026 (international reference only).",
    "Indence project working documents: MVP Architecture Blueprint (July 2026); Implementation Audit (July 2026); Milestone-1 Report (October 2026).",
    "Higgins JPT, Thomas J, Chandler J, et al. (eds.). Cochrane Handbook for Systematic Reviews of Interventions, version 6.5. Cochrane; 2024.",
    "World Health Organization. Artificial intelligence-related health research: ethics review and oversight. 2026.",
    "National Health Authority, Government of India. Health Data Management Policy of the Ayushman Bharat Digital Mission.",
    "Guyatt GH, Oxman AD, Vist GE, et al. GRADE: an emerging consensus on rating quality of evidence and strength of recommendations. BMJ. 2008;336(7650):924–926.",
    "Higgins JPT, Thompson SG. Quantifying heterogeneity in a meta-analysis. Stat Med. 2002;21(11):1539–1558.",
    "DerSimonian R, Laird N. Meta-analysis in clinical trials. Control Clin Trials. 1986;7(3):177–188.",
    "Ji Z, Lee N, Frieske R, et al. Survey of hallucination in natural language generation. ACM Comput Surv. 2023;55(12):1–38.",
    "Lewis P, Perez E, Piktus A, et al. Retrieval-augmented generation for knowledge-intensive NLP tasks. Advances in Neural Information Processing Systems 33 (NeurIPS 2020).",
    "Min S, Krishna K, Lyu X, et al. FActScore: fine-grained atomic evaluation of factual precision in long form text generation. Proc. EMNLP 2023.",
    "Chen J, Xiao S, Zhang P, Luo K, Lian D, Liu Z. BGE M3-Embedding: multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. Findings of ACL 2024; arXiv:2402.03216.",
    "Cormack GV, Clarke CLA, Büttcher S. Reciprocal rank fusion outperforms Condorcet and individual rank learning methods. Proc. SIGIR 2009:758–759.",
    "Xiao S, Liu Z, Zhang P, Muennighoff N. C-Pack: packaged resources to advance general Chinese embedding. arXiv:2309.07597; 2023 (BGE reranker family).",
    "Carbonell J, Goldstein J. The use of MMR, diversity-based reranking for reordering documents and producing summaries. Proc. SIGIR 1998:335–336.",
    "Järvelin K, Kekäläinen J. Cumulated gain-based evaluation of IR techniques. ACM Trans Inf Syst. 2002;20(4):422–446.",
    "Efron B, Tibshirani RJ. An Introduction to the Bootstrap. Chapman & Hall/CRC; 1993.",
    "U.S. Department of Health and Human Services. Guidance regarding methods for de-identification of protected health information in accordance with the HIPAA Privacy Rule (45 CFR §164.514(b)). 2012.",
]
for i, ref in enumerate(REFS, 1):
    para = r.p(f"[{i}] {ref}", align=WD_ALIGN_PARAGRAPH.LEFT, after=4)
    para.paragraph_format.left_indent = Inches(0.4)
    para.paragraph_format.first_line_indent = Inches(-0.4)
    para.paragraph_format.line_spacing = 1.15

# --------------------------------------------------------------------- APPENDICES
r.appendix("A", "Requirement Traceability, Milestone-1 to Final")
r.p("Each Milestone-1 requirement is mapped to the module that implements it and the evidence that it works. "
    "Statuses match Tables 3.2 and 3.3.")
r.table(["ID", "Implemented in", "Evidence"], [
    ["FR-03", "retrieval/normalization", "Regression tests S5; Table 5.4"],
    ["FR-04", "retrieval/retriever.py, adapters/models/bge_m3.py", "Ablation Table 6.3; sparse-vector fix (5.3)"],
    ["FR-05", "retrieval/context_assembler.py", "Ablation; latency Table 6.7"],
    ["FR-06/07", "synthesis/synthesizer.py, validator.py, citation_auditor.py", "Invalid citation rate 0% (Table 6.5)"],
    ["FR-09", "synthesis/abstention.py, pipeline gates", "Correct abstention 100% (Table 6.5)"],
    ["FR-11", "query_runs, retrieval_traces tables; Audit tab", "Figure 5.2; Execution Audit tab"],
    ["FR-16/17", "contradiction_checker.py, evidence_grouper.py, effect_map.py", "Forest plot (Figure 5.3); unit tests"],
    ["FR-18/19", "clinical/phi_guard.py, case_context.py, CaseMode.js", "Table 6.6; Figure 5.4"],
    ["FR-20", "/case/search, clinical/applicability.py", "Applicability 0.704 → 0.878 (Table 6.6)"],
    ["FR-21", "clinical/safety_check.py, SafetyCheck.js", "Figure 5.6; test_safety_check.py"],
    ["FR-22", "synthesis/effect_map.py, certainty.py", "test_effect_map.py; certainty card"],
    ["FR-23", "model_trace in API; ModelTrace.js", "Figure 5.2"],
    ["NFR-01", "Parallel retrieval, vector reuse, model chain", "Table 6.7"],
    ["NFR-05", "Evidence-ID constraint + audit", "0 fabricated references (Table 6.5)"],
    ["NFR-12/13", "Span verification, PHI guard", "Traceability 95.8%; 0/6 PHI leaks"],
], [0.8, 2.9, 2.5], "Requirement traceability", size=9.5, bold_first_col=True)

r.appendix("B", "Representative Test Cases")
r.table(["ID", "Test", "Input", "Expected", "Result"], [
    ["T01", "Biomarker notation", "“HER2+ breast cancer”", "HER2-positive detected", "Pass"],
    ["T02", "Subgroup boundary", "HER2-positive query", "HER2-positive never rejected; HER2-low excluded", "Pass"],
    ["T03", "Acronym safety", "“OS or PFS in HR+ disease”", "‘or’ and ‘HR+’ not expanded", "Pass"],
    ["T04", "Provider failure", "LLM returns HTTP 402", "Explicit abstention, no answer text", "Pass"],
    ["T05", "Abstention payload", "Verification fails", "No synthesis sections returned", "Pass"],
    ["T06", "Unauthenticated access", "No session cookie", "HTTP 401", "Pass"],
    ["T07", "PHI block", "Name, UHID, phone in case text", "Blocked locally before any call", "Pass"],
    ["T08", "Span verification", "Fact quote not in text", "Fact dropped and counted", "Pass"],
    ["T09", "Conflicting facts", "PD-L1 80% and 45%", "Conflict flagged; both unchecked", "Pass"],
    ["T10", "Exon 20 vs exon 19", "Exon 19 patient, exon 20 trial", "Marked ‘differs’", "Pass"],
    ["T11", "Adjuvant vs stage IV", "Stage IV patient, adjuvant trial", "Marked ‘differs’", "Pass"],
    ["T12", "Missing-data logic", "EGFR-positive NSCLC", "KRAS not listed as missing", "Pass"],
    ["T13", "Treatment-naive", "“Treatment-naive”", "Not treated as a prior therapy", "Pass"],
    ["T14", "Effect extraction", "“HR 0.60 (95% CI 0.41–0.89)”", "Ratio, CI, endpoint, source span", "Pass"],
    ["T15", "Heterogeneity", "Three conflicting HRs", "Q, I², τ² match reference values", "Pass"],
    ["T16", "Drug interaction", "Palbociclib + clarithromycin", "Major CYP3A4 flag", "Pass"],
    ["T17", "Treatment history", "Prior palbociclib; palbociclib proposed", "Treatment-history flag", "Pass"],
    ["T18", "Fictional drug", "“Is zorvatinib effective…”", "Unknown-entity abstention", "Pass"],
], [0.45, 1.25, 1.6, 2.0, 0.6], "Representative test cases (automated unless stated)", size=9)

r.appendix("C", "Project Log Diary")
r.p("Entries are taken from the git history and project documents. The team should add guide meetings and "
    "individual task allocations.")
r.table(["Date", "Activity", "Evidence"], [
    ["17–19 Jul 2026", "MVP architecture blueprint and implementation audit; corpus ingestion runs begin", "Project documents; ingestion_runs table"],
    ["29 Jul 2026", "Initial commit of the Indence codebase", "Commit 9212931"],
    ["Jul–Oct 2026", "PMC ingestion (19,057 papers), BGE-M3 indexing into Qdrant", "Database and index"],
    ["14 Oct 2026", "Milestone-1 report submitted", "Milestone-1 report"],
    ["22 Oct 2026", "Clinical intent, population matching, evidence hierarchy, contradiction checker, UI redesign", "Commit 3d7882b"],
    ["25 Nov 2026", "Phase 1: code audit; 10 safety and 6 performance defects fixed; shared pipeline; free-model chain", "Commit 6823aa9; PHASE1_CHANGELOG"],
    ["25 Nov 2026", "Phase 2: Gemini model chain, evidence map, certainty engine, Patient Case Mode, UI", "Commits 04e770a, e6e69fd, 50de3b4, b5d0120"],
    ["25 Nov 2026", "Deterministic safety gates; relevance-first ranking fix; evaluation harness", "Commits f4ad351, eeb415a"],
    ["25 Nov 2026", "Decision 1 verifier, Patient Safety Check, model attribution UI; Phase 3 results", "Commits ed1df3b, b5b9f9a, 0e0736c"],
    ["25 Nov 2026", "Section-level verification; contradicted-content removal policy", "Commit 5f485c2"],
    ["25–26 Nov 2026", "Final report, presentation, demonstration script, viva preparation", "This report"],
], [1.1, 3.3, 1.8], "Project log", size=9.5)

r.appendix("D", "Sample Synthetic Case")
r.p("Case 2 of the evaluation set (fictional; created for testing):", after=2)
r.p("__“68-year-old man, former smoker (40 pack-years). Diagnosis: stage IV non-small cell lung cancer, squamous "
    "histology, with adrenal and bone metastases. Molecular testing: EGFR wild-type, ALK negative, KRAS G12C "
    "mutation. PD-L1 22C3 on initial biopsy (01/2026): TPS 80%. Repeat PD-L1 on the adrenal biopsy (03/2026): "
    "TPS 45%. ECOG 0. Treatment-naive. eGFR 72 mL/min. History of type 2 diabetes on metformin.”__")
r.p("Clinical question: “Is first-line pembrolizumab monotherapy supported by evidence for this patient?” The "
    "gold standard lists 12 facts (cancer type, histology, stage, EGFR, ALK, KRAS G12C, both PD-L1 values, ECOG, "
    "eGFR, diabetes, metformin) and the expected PD-L1 conflict. Figure 5.4 shows the system’s extraction: 17 of "
    "17 proposed facts anchored to the text and the conflict flagged.")

r.appendix("E", "Statement on the Use of AI Tools")
r.p("[Team to review and edit.] Generative AI was used in two distinct ways:")
r.bullets([
    "**As system components.** Gemini Flash models (synthesis and case-fact extraction), Gemini Decision 1 (claim verification) and OpenRouter free models (fallback) are runtime parts of Indence. Every output they produce is shown with the model’s name and verified as described in Chapter 4.",
    "**As development assistants.** An AI coding assistant (Claude, Anthropic) was used during the final phase to audit code, implement fixes and features, write tests and evaluation scripts, and draft this report. All reported numbers were produced by running the code and evaluation scripts, and the team reviewed the code, the results and the text.",
])
r.p("No AI tool was used to fabricate data, results, interviews or feedback.")

r.appendix("F", "Running and Reproducing the System")
for line in [
    "Start Qdrant:  backend\\qdrant_bin\\qdrant.exe  (storage: backend\\qdrant_native_data)",
    "Backend:  backend\\venv\\Scripts\\python.exe -m uvicorn --app-dir backend/src evidence_platform.app:app --port 8000",
    "Frontend:  npm --prefix frontend run build  then  npm --prefix frontend run start -- -p 3000",
    "Tests:  backend\\venv\\Scripts\\python.exe -m pytest backend/tests",
    "Evaluations:  run_retrieval_eval.py, run_safety_eval.py, run_case_eval.py, then make_charts.py (in evals/)",
    "This report:  report\\make_diagrams.py, report\\take_screenshots.py, report\\build_report.py",
]:
    para = r.p(line, align=WD_ALIGN_PARAGRAPH.LEFT, size=10, after=2)
    para.paragraph_format.line_spacing = 1.15

r.save(OUT)
print("wrote", OUT)
