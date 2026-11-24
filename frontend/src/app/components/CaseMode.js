"use client";

import React, { useMemo, useState } from "react";
import EvidenceMap from "./EvidenceMap";
import CertaintyCard from "./CertaintyCard";
import ModelTrace from "./ModelTrace";
import SafetyCheck from "./SafetyCheck";
import ClaimVerification, { VerifyBadge, RemovedByVerification } from "./ClaimVerification";

// Synthetic, fictional, de-identified sample cases (see evals/cases/synthetic_cases.json).
const SAMPLES = [
  {
    label: "NSCLC · EGFR exon 19",
    question: "What first-line systemic treatment has the strongest survival evidence for this patient?",
    text: "62-year-old woman, never smoker. Biopsy of right upper lobe mass: lung adenocarcinoma. Staging PET-CT and MRI brain: stage IVB with multiple liver metastases and two asymptomatic brain metastases (largest 8 mm). Tissue NGS: EGFR exon 19 deletion detected; ALK rearrangement negative; ROS1 fusion negative. PD-L1 (22C3) TPS 10%. ECOG performance status 1. No prior systemic therapy. Creatinine 0.9 mg/dL, bilirubin 0.6 mg/dL. Comorbidity: well-controlled hypothyroidism on levothyroxine.",
  },
  {
    label: "NSCLC · conflicting PD-L1",
    question: "Is first-line pembrolizumab monotherapy supported by evidence for this patient?",
    text: "68-year-old man, former smoker (40 pack-years). Diagnosis: stage IV non-small cell lung cancer, squamous histology, with adrenal and bone metastases. Molecular testing: EGFR wild-type, ALK negative, KRAS G12C mutation. PD-L1 22C3 on initial biopsy (01/2026): TPS 80%. Repeat PD-L1 on the adrenal biopsy (03/2026): TPS 45%. ECOG 0. Treatment-naive. eGFR 72 mL/min. History of type 2 diabetes on metformin.",
  },
  {
    label: "Breast · HER2+ after 1st line",
    question: "What is the best-supported second-line HER2-targeted treatment after progression?",
    text: "54-year-old woman with metastatic breast cancer. Invasive ductal carcinoma, ER negative, PR negative, HER2 positive (IHC 3+). Metastases to liver and lungs. First-line treatment with trastuzumab, pertuzumab and docetaxel for 14 months; now progressive disease in the liver. No brain metastases on MRI. ECOG 1. LVEF 58%. ALT 48 U/L.",
  },
  {
    label: "Colorectal · left-sided RAS wt",
    question: "Should an anti-EGFR antibody be added to first-line chemotherapy?",
    text: "59-year-old man with adenocarcinoma of the sigmoid colon (left-sided) and synchronous unresectable liver metastases. Extended RAS testing: KRAS and NRAS wild-type. BRAF V600E not detected. Microsatellite stable (MSS), pMMR. No prior chemotherapy. ECOG 0. Bilirubin 0.8 mg/dL, platelets 260 x10^9/L.",
  },
  {
    label: "Privacy demo (blocked)",
    question: "What first-line therapy is recommended?",
    text: "Patient name: Rakesh Verma, UHID: AIIMS2026/44871, phone 98765 43210. 61-year-old man with stage IV lung adenocarcinoma, EGFR exon 21 L858R mutation, ECOG 1.",
  },
];

const FIELD_LABELS = {
  cancer_type: "Cancer", histology: "Histology", stage: "Stage", metastatic_site: "Metastasis",
  biomarker: "Biomarker", prior_therapy: "Prior therapy", line_of_therapy: "Line", performance_status: "Performance",
  age: "Age", sex: "Sex", lab: "Lab", comorbidity: "Comorbidity", medication: "Medication",
};
const STATUS_ICON = { match: "✓", differs: "✗", not_reported: "?" };
const STATUS_TEXT = { match: "Matches", differs: "Differs", not_reported: "Not reported" };

function HighlightedText({ text, spans, activeIdx }) {
  // spans: [{start,end,kind}] non-overlapping, sorted
  const parts = [];
  let last = 0;
  [...spans].sort((a, b) => a.start - b.start).forEach((s, i) => {
    if (s.start < last) return;
    parts.push(<span key={`t${i}`}>{text.slice(last, s.start)}</span>);
    parts.push(
      <mark key={`m${i}`} className={`case-mark ${s.kind} ${activeIdx === s.idx ? "active" : ""}`} title={s.title}>
        {text.slice(s.start, s.end)}
      </mark>
    );
    last = s.end;
  });
  parts.push(<span key="tail">{text.slice(last)}</span>);
  return <p className="case-source">{parts}</p>;
}

export default function CaseMode({ apiPost }) {
  const [text, setText] = useState(SAMPLES[0].text);
  const [question, setQuestion] = useState(SAMPLES[0].question);
  const [extracting, setExtracting] = useState(false);
  const [extraction, setExtraction] = useState(null);
  const [facts, setFacts] = useState([]);
  const [activeFact, setActiveFact] = useState(null);
  const [searching, setSearching] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [extractedText, setExtractedText] = useState("");

  const loadSample = (s) => {
    setText(s.text); setQuestion(s.question); setExtraction(null); setFacts([]); setResult(null); setError("");
  };

  const runExtract = async () => {
    setExtracting(true); setError(""); setResult(null); setExtraction(null); setFacts([]);
    try {
      const data = await apiPost("/case/extract", { text });
      setExtraction(data);
      setExtractedText(text);
      setFacts((data.facts || []).map((f) => ({ ...f, confirmed: !(f.conflict_with && f.conflict_with.length) })));
    } catch (e) {
      setError(e.message);
    } finally {
      setExtracting(false);
    }
  };

  const runSearch = async () => {
    setSearching(true); setError(""); setResult(null);
    try {
      const confirmed = facts.filter((f) => f.confirmed).map(({ confirmed, ...f }) => f);
      const data = await apiPost("/case/search", { question, facts: confirmed, filters: {} });
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setSearching(false);
    }
  };

  const spans = useMemo(() => {
    if (!extraction) return [];
    if (extraction.phi && extraction.phi.blocked) {
      return extraction.phi.findings.map((f, i) => ({ start: f.start, end: f.end, kind: "phi", title: f.kind, idx: `p${i}` }));
    }
    return facts
      .map((f, i) => ({ start: f.start, end: f.end, kind: f.conflict_with && f.conflict_with.length ? "conflict" : "fact",
        title: `${FIELD_LABELS[f.field] || f.field}: ${f.value}`, idx: i }))
      .filter((s) => s.start !== null && s.start !== undefined);
  }, [extraction, facts]);

  const cm = result && result.case_mode;
  const dims = useMemo(() => {
    if (!cm) return [];
    const set = [];
    cm.applicability.rows.forEach((r) => r.cells.forEach((c) => { if (!set.includes(c.dimension)) set.push(c.dimension); }));
    return set;
  }, [cm]);

  return (
    <div className="case-mode">
      <div className="case-intro glass">
        <div>
          <span className="section-kicker">Patient Case Mode</span>
          <h2 className="case-title">From a patient&rsquo;s reports to the evidence that actually applies to them</h2>
          <p className="case-lede">
            Paste a <strong>de-identified</strong> case summary. Indence screens it for identifiers on this device,
            extracts clinical facts that are each anchored to the exact words in the text, asks you to confirm them,
            then searches the literature and checks every study&rsquo;s population against this patient.
          </p>
        </div>
        <div className="sample-row">
          <span className="filter-label">Synthetic sample cases</span>
          <div className="sample-pills">
            {SAMPLES.map((s) => (
              <button key={s.label} type="button" className="sample-pill" onClick={() => loadSample(s)}>{s.label}</button>
            ))}
          </div>
        </div>
      </div>

      <div className="case-grid">
        <section className="glass case-panel">
          <h3 className="panel-title"><span className="step-num">1</span> Case summary</h3>
          <textarea className="input-text case-textarea" value={text} onChange={(e) => setText(e.target.value)}
            aria-label="De-identified case summary" rows={9} />
          <button type="button" className="search-btn" onClick={runExtract} disabled={extracting || !text.trim()}>
            {extracting ? "Screening & extracting…" : "Screen for identifiers & extract facts"}
          </button>
          {extraction && extractedText && (
            <>
              <div className={`phi-status ${extraction.phi.blocked ? "blocked" : "ok"}`} role="status">
                <span aria-hidden="true">{extraction.phi.blocked ? "⛔" : "🔒"}</span>
                <span>{extraction.phi.message}</span>
              </div>
              <HighlightedText text={extractedText} spans={spans} activeIdx={activeFact} />
            </>
          )}
        </section>

        <section className="glass case-panel">
          <h3 className="panel-title"><span className="step-num">2</span> Confirm extracted facts</h3>
          {!extraction && <p className="muted-note">Extracted facts appear here, each linked to its source words.</p>}
          {extraction && extraction.phi && extraction.phi.blocked && (
            <ul className="phi-list">
              {extraction.phi.findings.map((f, i) => (
                <li key={i}><strong>{f.kind}:</strong> &ldquo;{f.text}&rdquo;</li>
              ))}
            </ul>
          )}
          {extraction && !extraction.phi.blocked && (
            <>
              <div className="trace-badges">
                <span className="trace-badge ok">
                  {extraction.traceability.verified}/{extraction.traceability.proposed} facts anchored to source text
                </span>
                {extraction.traceability.dropped > 0 && (
                  <span className="trace-badge warn">{extraction.traceability.dropped} untraceable facts dropped</span>
                )}
                <span className="trace-badge">{extraction.extraction_method}</span>
              </div>
              {extraction.conflicts.length > 0 && (
                <div className="case-conflict" role="status">
                  <span aria-hidden="true">⚠</span>
                  <div>
                    {extraction.conflicts.map((c, i) => (
                      <p key={i}>
                        <strong>{c.message}</strong>{" "}
                        {c.values.map((v) => `${v.value}${v.date ? ` (${v.date})` : ""}`).join(" vs ")}
                      </p>
                    ))}
                    <p className="muted-note">Conflicting facts start unchecked — tick the one you have verified.</p>
                  </div>
                </div>
              )}
              <table className="facts-table">
                <thead><tr><th>Use</th><th>Field</th><th>Value (editable)</th><th>Source</th></tr></thead>
                <tbody>
                  {facts.map((f, i) => (
                    <tr key={i} className={f.conflict_with && f.conflict_with.length ? "conflict" : ""}
                      onMouseEnter={() => setActiveFact(i)} onMouseLeave={() => setActiveFact(null)}>
                      <td>
                        <input type="checkbox" checked={!!f.confirmed} aria-label={`Use ${f.value}`}
                          onChange={(e) => setFacts(facts.map((x, j) => (j === i ? { ...x, confirmed: e.target.checked } : x)))} />
                      </td>
                      <td className="fact-field">{FIELD_LABELS[f.field] || f.field}{f.name ? ` · ${f.name}` : ""}</td>
                      <td>
                        <input className="fact-input" value={f.value}
                          onChange={(e) => setFacts(facts.map((x, j) => (j === i ? { ...x, value: e.target.value } : x)))} />
                      </td>
                      <td className="fact-quote" title={f.source_quote}>chars {f.start}–{f.end}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <ModelTrace trace={extraction.model_trace} title="Extraction — which model did what" />
              {extraction.missing_data.length > 0 && (
                <div className="missing-box">
                  <span className="filter-label">Missing decision-critical information</span>
                  <ul>
                    {extraction.missing_data.map((m) => (
                      <li key={m.variable}><strong>{m.variable}</strong> — {m.why_it_matters}</li>
                    ))}
                  </ul>
                </div>
              )}
            </>
          )}
        </section>
      </div>

      {extraction && !extraction.phi.blocked && (
        <section className="glass case-panel">
          <h3 className="panel-title"><span className="step-num">3</span> Ask the clinical question</h3>
          <div className="case-question-row">
            <input className="input-text" value={question} onChange={(e) => setQuestion(e.target.value)}
              aria-label="Clinical question" />
            <button type="button" className="search-btn" onClick={runSearch}
              disabled={searching || !question.trim() || !facts.some((f) => f.confirmed)}>
              {searching ? "Searching & appraising…" : "Find applicable evidence"}
            </button>
          </div>
        </section>
      )}

      {error && <div className="glass error-box">Error: {error}</div>}

      {searching && (
        <div className="case-loading">
          <div className="loading-spinner" />
          <p>Running the general search and the case-aware search, appraising certainty and checking each study&rsquo;s population against this patient…</p>
        </div>
      )}

      {result && cm && (
        <div className="case-results">
          <section className="glass case-panel">
            <span className="section-kicker">Case-aware query</span>
            <p className="case-query">{cm.case_query}</p>
            {result.status === "insufficient_evidence" ? (
              <div className="abstain-box">
                <strong>Abstained — no unverified answer shown.</strong> {result.abstention_reason}
              </div>
            ) : (
              <div className="case-answer">
                {result.synthesis && result.synthesis.recommendation && (
                  <p><strong>Answer:</strong> <VerifyBadge section={result.synthesis.recommendation} /> {result.synthesis.recommendation.text}</p>
                )}
                {result.synthesis && result.synthesis.bottom_line && (
                  <p className="case-bottom"><strong>Bottom line:</strong> {result.synthesis.bottom_line.text}</p>
                )}
                <RemovedByVerification items={result.synthesis && result.synthesis.removed_by_verification} />
                {result.synthesis && result.synthesis.clinical_context && result.synthesis.clinical_context.assumptions && (
                  <p className="muted-note">Assumptions: {result.synthesis.clinical_context.assumptions}</p>
                )}
              </div>
            )}
          </section>

          <SafetyCheck safety={cm.safety_check} />

          <section className="glass case-panel">
            <div className="matrix-head">
              <div>
                <span className="section-kicker">Applicability Matrix</span>
                <p className="muted-note">
                  Does each study&rsquo;s population match this patient? Hover a cell for the sentence that justifies it.
                </p>
              </div>
              <div className="matrix-compare">
                <div className="compare-stat">
                  <span className="cs-num">{cm.comparison.case_mean_applicability ?? "—"}</span>
                  <span className="cs-label">mean applicability · case-aware</span>
                </div>
                <div className="compare-stat">
                  <span className="cs-num muted">{cm.comparison.general_mean_applicability ?? "—"}</span>
                  <span className="cs-label">mean applicability · general search</span>
                </div>
              </div>
            </div>
            <div className="matrix-scroll">
              <table className="matrix-table">
                <thead>
                  <tr>
                    <th>Study</th>
                    {dims.map((d) => <th key={d}>{d}</th>)}
                    <th>Verdict</th>
                  </tr>
                </thead>
                <tbody>
                  {cm.applicability.rows.map((r) => (
                    <tr key={r.pmcid}>
                      <td className="matrix-study">
                        {r.evidence_id && <span className="reference-id">{r.evidence_id}</span>}{" "}
                        <a href={`https://pmc.ncbi.nlm.nih.gov/articles/${r.pmcid}/`} target="_blank" rel="noopener noreferrer">
                          {(r.title || r.pmcid).slice(0, 70)}{(r.title || "").length > 70 ? "…" : ""}
                        </a>
                        <span className="matrix-meta">{r.study_type}{r.year ? ` · ${r.year}` : ""}</span>
                      </td>
                      {dims.map((d) => {
                        const c = r.cells.find((x) => x.dimension === d);
                        if (!c) return <td key={d} className="cell na">—</td>;
                        return (
                          <td key={d} className={`cell ${c.status}`}
                            title={`${STATUS_TEXT[c.status]} — patient: ${c.patient}${c.note ? ` · ${c.note}` : ""}${c.quote ? `\n“${c.quote}”` : ""}`}>
                            <span aria-hidden="true">{STATUS_ICON[c.status]}</span>
                            <span className="sr-only">{STATUS_TEXT[c.status]}</span>
                          </td>
                        );
                      })}
                      <td className={`verdict ${r.verdict.startsWith("Different") ? "bad" : r.verdict.startsWith("Closely") ? "good" : ""}`}>
                        {r.verdict}{r.score !== null && r.score !== undefined ? ` (${r.score})` : ""}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="muted-note">
              ✓ matches · ✗ differs · ? not reported in the retrieved text. {cm.comparison.note}
            </p>
            <details className="baseline-details">
              <summary>General-search baseline ({cm.general_baseline.sources.length} studies, {cm.comparison.only_in_general} not in the case-aware list)</summary>
              <ul>
                {cm.general_baseline.sources.map((s) => (
                  <li key={s.pmcid}>{(s.title || s.pmcid).slice(0, 90)} — <em>{s.verdict}</em></li>
                ))}
              </ul>
            </details>
          </section>

          <CertaintyCard certainty={result.certainty} />
          <ClaimVerification claims={result.direct_answer_claims} />
          <EvidenceMap map={result.evidence_map} />
          <ModelTrace trace={{ ...(extraction && extraction.model_trace), ...result.model_trace }} />
        </div>
      )}
    </div>
  );
}
