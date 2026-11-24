"use client";

import React from "react";

const JUDGEMENT_CLASS = {
  "no serious concern": "ok",
  serious: "warn",
  "very serious": "bad",
  "not assessed": "na",
};
const JUDGEMENT_ICON = { "no serious concern": "✓", serious: "↓", "very serious": "↓↓", "not assessed": "–" };

export default function CertaintyCard({ certainty }) {
  if (!certainty || !certainty.level) return null;
  return (
    <div className="certainty-card glass">
      <div className="cc-head">
        <div>
          <span className="section-kicker">Certainty of evidence (GRADE-inspired)</span>
          <div className="cc-level">
            <span className={`cc-symbols level-${certainty.level}`} aria-hidden="true">{certainty.symbols}</span>
            <span className="cc-label">{certainty.label}</span>
          </div>
        </div>
        <p className="cc-start">Starting level: <strong>{certainty.start_level}</strong>. {certainty.start_reason}</p>
      </div>
      <table className="cc-table">
        <thead>
          <tr><th>Domain</th><th>Judgement</th><th>Why (measured from the retrieved evidence)</th></tr>
        </thead>
        <tbody>
          {certainty.domains.map((d) => (
            <tr key={d.domain}>
              <td className="cc-domain">{d.domain}</td>
              <td>
                <span className={`cc-judgement ${JUDGEMENT_CLASS[d.judgement] || "na"}`}>
                  <span aria-hidden="true">{JUDGEMENT_ICON[d.judgement] || "–"}</span> {d.judgement}
                </span>
              </td>
              <td className="cc-reason">{d.reason}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="cc-disclaimer">{certainty.disclaimer}</p>
    </div>
  );
}
