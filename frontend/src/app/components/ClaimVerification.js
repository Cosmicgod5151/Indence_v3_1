"use client";

import React from "react";

const STATUS = {
  entailment: { label: "Supported", cls: "ok", icon: "✓" },
  contradiction: { label: "Contradicted", cls: "bad", icon: "✗" },
  neutral: { label: "Not established", cls: "warn", icon: "?" },
  unverified: { label: "Unverified", cls: "na", icon: "–" },
};

// Badge for a synthesis section verified by the claim verifier (calibrated confidence).
export function VerifyBadge({ section }) {
  const v = section && section.verification;
  if (!v) return null;
  const st = STATUS[v.label] || STATUS.unverified;
  const pct = typeof v.confidence === "number" ? ` ${(v.confidence * 100).toFixed(0)}%` : "";
  const text = v.label === "entailment" ? `verified${pct}` : v.label === "neutral" ? `not fully established${pct}` : st.label;
  return (
    <span className={`verify-badge ${st.cls}`} title="Checked against the passages this section cites (Gemini Decision 1)">
      <span aria-hidden="true">{st.icon}</span> {text}
    </span>
  );
}

// Content the verifier removed: shown openly, never silently dropped.
export function RemovedByVerification({ items }) {
  if (!items || items.length === 0) return null;
  return (
    <div className="removed-box">
      <strong>Removed by verification ({items.length}):</strong>
      <ul>
        {items.map((r, i) => (
          <li key={i}><em>{r.section}</em>: &ldquo;{r.text}&rdquo; — contradicted by its cited source
            {typeof r.verification?.confidence === "number" ? ` (${(r.verification.confidence * 100).toFixed(0)}%)` : ""}</li>
        ))}
      </ul>
    </div>
  );
}

// Every atomic claim of the answer, judged against the exact passages it cites.
export default function ClaimVerification({ claims }) {
  if (!claims || claims.length === 0) return null;
  return (
    <section className="claim-verification glass">
      <span className="section-kicker">Claim-by-claim verification</span>
      <p className="muted-note">
        Each statement is checked against only the passages it cites. Confidence is the verifier&rsquo;s calibrated
        probability for its verdict.
      </p>
      <table className="cc-table">
        <thead>
          <tr><th>Claim</th><th>Cited</th><th>Verdict</th><th>Confidence</th><th>Numbers</th></tr>
        </thead>
        <tbody>
          {claims.map((c, i) => {
            const st = STATUS[c.entailment_status] || STATUS.unverified;
            return (
              <tr key={i} className={c.removed_by_verification ? "cv-removed" : ""}>
                <td>{c.text}</td>
                <td className="cv-ids">{(c.evidence_ids || []).join(", ")}</td>
                <td>
                  <span className={`cc-judgement ${st.cls}`}><span aria-hidden="true">{st.icon}</span> {st.label}</span>
                  {c.removed_by_verification && <span className="cv-removed-tag">removed from answer</span>}
                </td>
                <td className="cv-conf">
                  {typeof c.entailment_confidence === "number" ? (
                    <span className="cv-bar" title={JSON.stringify(c.entailment_probabilities || {})}>
                      <span className="cv-fill" style={{ width: `${Math.round(c.entailment_confidence * 100)}%` }} />
                      <span className="cv-num">{(c.entailment_confidence * 100).toFixed(0)}%</span>
                    </span>
                  ) : "—"}
                </td>
                <td>{c.numeric_validation_passed === false ? <span className="cc-judgement bad">mismatch</span> : <span className="cc-judgement ok">ok</span>}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </section>
  );
}
