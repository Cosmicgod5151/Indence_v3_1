"use client";

import React from "react";

const CATEGORY = {
  treatment_history: "Treatment history",
  drug_interaction: "Drug interaction",
  comorbidity: "Comorbidity",
  organ_function: "Organ function",
  ai_screen: "AI-screened interaction",
};
const SEV_ICON = { major: "⛔", moderate: "⚠", review: "🔎", info: "ℹ" };
const METHOD = {
  curated_rule: "Curated rule (label-based)",
  treatment_history: "Matched against prior therapy",
  ai_screen: "Gemini Decision 1 — calibrated probability",
};

export default function SafetyCheck({ safety }) {
  if (!safety) return null;
  const { flags = [], counts = {}, proposed_drugs = [], patient_medications = [], prior_therapies = [] } = safety;
  return (
    <section className="glass case-panel safety-check">
      <div className="sc-head">
        <div>
          <span className="section-kicker">Patient Safety Check</span>
          <p className="muted-note">
            Proposed drugs from the question and answer, checked against this patient&rsquo;s current medications,
            earlier treatments, comorbidities and organ function.
          </p>
        </div>
        <div className="sc-counts">
          <span className="sc-count major">{counts.major || 0} major</span>
          <span className="sc-count moderate">{counts.moderate || 0} moderate</span>
          <span className="sc-count review">{counts.ai_review || 0} AI review</span>
        </div>
      </div>

      <div className="sc-context">
        <div><span className="filter-label">Proposed drugs</span><p>{proposed_drugs.join(", ") || "none detected"}</p></div>
        <div><span className="filter-label">Current medications</span><p>{patient_medications.join(", ") || "none recorded"}</p></div>
        <div><span className="filter-label">Earlier treatments</span><p>{prior_therapies.join("; ") || "none recorded"}</p></div>
      </div>

      {flags.length === 0 ? (
        <p className="sc-none">No safety flags raised by the curated rules or the AI screen. This does not mean the
          absence of risk.</p>
      ) : (
        <ul className="sc-list">
          {flags.map((f, i) => (
            <li key={i} className={`sc-flag ${f.severity}`}>
              <div className="sc-flag-head">
                <span aria-hidden="true">{SEV_ICON[f.severity] || "•"}</span>
                <strong>{f.proposed}</strong>
                <span className="sc-x">×</span>
                <span>{f.patient_factor}</span>
                <span className="sc-cat">{CATEGORY[f.category] || f.category}</span>
                <span className={`sc-sev ${f.severity}`}>{f.severity === "review" ? "needs review" : f.severity}</span>
              </div>
              <p className="sc-msg">{f.message}</p>
              <p className="sc-action"><strong>Action:</strong> {f.action}</p>
              <p className="sc-method">
                {METHOD[f.method] || f.method}
                {f.probability !== null && f.probability !== undefined ? ` · p = ${f.probability.toFixed(2)}` : ""}
                {f.source_quote ? ` · source: “${f.source_quote}”` : ""}
              </p>
            </li>
          ))}
        </ul>
      )}
      <p className="cc-disclaimer">{safety.disclaimer}</p>
    </section>
  );
}
