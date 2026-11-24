"use client";

import React from "react";

const STATUS = {
  entailment: { label: "Supported", cls: "ok", icon: "✓" },
  contradiction: { label: "Contradicted", cls: "bad", icon: "✗" },
  neutral: { label: "Not established", cls: "warn", icon: "?" },
  unverified: { label: "Unverified", cls: "na", icon: "–" },
};

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
              <tr key={i}>
                <td>{c.text}</td>
                <td className="cv-ids">{(c.evidence_ids || []).join(", ")}</td>
                <td><span className={`cc-judgement ${st.cls}`}><span aria-hidden="true">{st.icon}</span> {st.label}</span></td>
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
