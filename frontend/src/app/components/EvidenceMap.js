"use client";

import React, { useState } from "react";

// Log-scale forest plot for ratio estimates (HR / OR / RR).
// One series colour; the reference line at 1 = no difference; markers sized by
// inverse-variance weight; hover shows the exact source sentence.
const W = 720;
const LABEL_W = 250;
const VALUE_W = 140;
const PLOT_W = W - LABEL_W - VALUE_W;
const ROW_H = 30;
const TICKS = [0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 4, 5, 10, 20];

function niceDomain(values) {
  const lo = Math.max(0.02, Math.min(...values, 1) * 0.8);
  const hi = Math.min(50, Math.max(...values, 1) * 1.25);
  return [lo, hi];
}

function ForestPlot({ group, onHover, hovered }) {
  const ests = group.estimates || [];
  const pooled = group.pooled;
  const vals = ests.flatMap((e) => [e.ci_low, e.ci_high]).concat(pooled ? [pooled.ci_low, pooled.ci_high] : []);
  const [lo, hi] = niceDomain(vals);
  const x = (v) => LABEL_W + ((Math.log(Math.max(v, lo)) - Math.log(lo)) / (Math.log(hi) - Math.log(lo))) * PLOT_W;
  const weights = ests.map((e) => {
    const se = (Math.log(e.ci_high) - Math.log(e.ci_low)) / 3.92;
    return se > 0 ? 1 / (se * se) : 1;
  });
  const wMax = Math.max(...weights, 1);
  const rows = ests.length + (pooled ? 1 : 0);
  const H = 28 + rows * ROW_H + 44;
  const ticks = TICKS.filter((t) => t >= lo && t <= hi);
  const prognostic = group.context === "prognostic_association";

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="forest-svg" role="img"
      aria-label={`Forest plot of ${group.measure} for ${group.endpoint}, ${ests.length} studies`}>
      <text x={LABEL_W - 8} y={16} textAnchor="end" className="fp-head">Study</text>
      <text x={W - 4} y={16} textAnchor="end" className="fp-head">{group.measure} (95% CI)</text>
      <line x1={x(1)} x2={x(1)} y1={24} y2={H - 40} className="fp-ref" />
      {ests.map((e, i) => {
        const y = 28 + i * ROW_H + ROW_H / 2;
        const size = 8 + 8 * Math.sqrt(weights[i] / wMax);
        const label = `${e.evidence_id ? e.evidence_id + " · " : ""}${(e.title || e.pmcid || "").slice(0, 34)}${(e.title || "").length > 34 ? "…" : ""}`;
        const isHover = hovered === `${group.endpoint}-${i}`;
        return (
          <g key={i} onMouseEnter={() => onHover({ key: `${group.endpoint}-${i}`, e })} onMouseLeave={() => onHover(null)}
            className={`fp-row ${isHover ? "hover" : ""}`}>
            <rect x={0} y={y - ROW_H / 2} width={W} height={ROW_H} className="fp-hit" />
            <text x={LABEL_W - 8} y={y + 4} textAnchor="end" className="fp-label">{label}{e.year ? ` (${e.year})` : ""}</text>
            <line x1={x(e.ci_low)} x2={x(e.ci_high)} y1={y} y2={y} className={`fp-ci ${prognostic ? "muted" : ""}`} />
            <rect x={x(e.estimate) - size / 2} y={y - size / 2} width={size} height={size} rx={2}
              className={`fp-point ${prognostic ? "muted" : ""}`} />
            <text x={W - 4} y={y + 4} textAnchor="end" className="fp-value">
              {e.estimate.toFixed(2)} ({e.ci_low.toFixed(2)}–{e.ci_high.toFixed(2)})
            </text>
          </g>
        );
      })}
      {pooled && (() => {
        const y = 28 + ests.length * ROW_H + ROW_H / 2;
        const xl = x(pooled.ci_low), xm = x(pooled.estimate), xh = x(pooled.ci_high);
        return (
          <g>
            <text x={LABEL_W - 8} y={y + 4} textAnchor="end" className="fp-label strong">Exploratory pooled (RE)</text>
            <polygon points={`${xl},${y} ${xm},${y - 8} ${xh},${y} ${xm},${y + 8}`} className="fp-diamond" />
            <text x={W - 4} y={y + 4} textAnchor="end" className="fp-value strong">
              {pooled.estimate.toFixed(2)} ({pooled.ci_low.toFixed(2)}–{pooled.ci_high.toFixed(2)})
            </text>
          </g>
        );
      })()}
      <line x1={LABEL_W} x2={LABEL_W + PLOT_W} y1={H - 40} y2={H - 40} className="fp-axis" />
      {ticks.map((t) => (
        <g key={t}>
          <line x1={x(t)} x2={x(t)} y1={H - 40} y2={H - 35} className="fp-axis" />
          <text x={x(t)} y={H - 24} textAnchor="middle" className="fp-tick">{t}</text>
        </g>
      ))}
      <text x={x(1) - 6} y={H - 6} textAnchor="end" className="fp-tick">◀ lower ratio</text>
      <text x={x(1) + 6} y={H - 6} textAnchor="start" className="fp-tick">higher ratio ▶</text>
    </svg>
  );
}

export default function EvidenceMap({ map }) {
  const [hover, setHover] = useState(null);
  if (!map || !map.groups || map.groups.length === 0) {
    return (
      <div className="evidence-map glass">
        <div className="em-header">
          <span className="section-kicker">Quantitative Evidence Map</span>
        </div>
        <p className="em-empty">
          No hazard/odds/risk ratios with 95% confidence intervals were found in the retrieved passages, so no forest
          plot can be drawn. Indence does not invent numbers.
        </p>
      </div>
    );
  }
  return (
    <div className="evidence-map glass">
      <div className="em-header">
        <span className="section-kicker">Quantitative Evidence Map</span>
        <span className="em-sub">
          {map.n_estimates} effect estimates extracted verbatim from {map.n_papers} papers
        </span>
      </div>
      {(map.quantitative_conflicts || []).length > 0 && (
        <div className="em-conflict" role="status">
          <span aria-hidden="true">⚠</span>
          <span>
            <strong>Quantitative conflict:</strong>{" "}
            {map.quantitative_conflicts.map((q) => `${q.endpoint} (${q.measure}) — ${q.description}`).join("; ")}
          </span>
        </div>
      )}
      {map.groups.map((g) => (
        <div key={`${g.measure}-${g.endpoint}-${g.context}`} className="em-group">
          <div className="em-group-head">
            <span className="em-endpoint">{g.endpoint}</span>
            <span className="em-measure">{g.measure}</span>
            <span className={`em-context ${g.context === "prognostic_association" ? "prog" : ""}`}>
              {g.context === "prognostic_association" ? "Prognostic association — not a treatment effect" : "Treatment comparison"}
            </span>
            {g.k >= 2 && g.i2 !== null && g.i2 !== undefined && (
              <span className="em-stat" title="Share of variability due to between-study heterogeneity rather than chance">
                I² = {g.i2.toFixed(0)}% · Q = {g.q} · k = {g.k}
              </span>
            )}
          </div>
          <ForestPlot group={g} onHover={(h) => setHover(h)} hovered={hover && hover.key} />
        </div>
      ))}
      <div className="em-tooltip" aria-live="polite">
        {hover ? (
          <>
            <span className="em-tt-src">
              {hover.e.evidence_id ? `${hover.e.evidence_id} · ` : ""}{hover.e.pmcid}
              {hover.e.p_value ? ` · p${hover.e.p_value}` : ""}
            </span>
            <span className="em-tt-quote">&ldquo;{hover.e.quote.replace(/\s+/g, " ")}&rdquo;</span>
          </>
        ) : (
          <span className="em-tt-hint">Hover a study to see the exact sentence each number was extracted from.</span>
        )}
      </div>
      <p className="em-note">{map.note}</p>
    </div>
  );
}
