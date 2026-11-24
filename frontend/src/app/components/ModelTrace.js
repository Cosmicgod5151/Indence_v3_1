"use client";

import React from "react";

// Shows which model (or deterministic component) performed each step of a run,
// including fallbacks and retries, so every output is attributable.
const ORDER = ["extractor", "span_verifier", "embedding", "reranker", "deterministic", "generator", "verifier",
  "safety_rules", "safety_ai_screen"];
const STEP_LABEL = {
  extractor: "Case fact extraction",
  span_verifier: "Fact source check",
  embedding: "Query encoding",
  reranker: "Relevance reranking",
  deterministic: "Clinical logic",
  generator: "Answer writing",
  verifier: "Claim verification",
  safety_rules: "Safety rules",
  safety_ai_screen: "Interaction AI screen",
};

function kind(model) {
  const m = String(model || "").toLowerCase();
  if (!m || m === "null") return "none";
  if (m.includes("no llm") || m.includes("indence")) return "rule";
  if (m.includes("bge")) return "local";
  return "llm";
}

export default function ModelTrace({ trace, title = "Which model did what" }) {
  if (!trace || Object.keys(trace).length === 0) return null;
  const keys = Object.keys(trace).sort((a, b) => {
    const ia = ORDER.indexOf(a), ib = ORDER.indexOf(b);
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
  });
  return (
    <details className="model-trace glass" open>
      <summary>
        <span className="section-kicker">{title}</span>
        <span className="mt-legend">
          <span className="mt-chip llm">LLM / AI model</span>
          <span className="mt-chip local">local GPU model</span>
          <span className="mt-chip rule">deterministic rules</span>
        </span>
      </summary>
      <table className="mt-table">
        <thead>
          <tr><th>Step</th><th>Performed by</th><th>Details</th></tr>
        </thead>
        <tbody>
          {keys.map((k) => {
            const t = trace[k] || {};
            const model = t.model || (t.error ? "unavailable" : "—");
            const details = [];
            if (t.task) details.push(t.task);
            if (t.runs_on) details.push(t.runs_on);
            if (t.latency_s !== undefined && t.latency_s !== null) details.push(`${Number(t.latency_s).toFixed(2)} s`);
            if (t.retries) details.push(`${t.retries} retr${t.retries === 1 ? "y" : "ies"}`);
            if (t.fallback_used) details.push("fallback verifier used");
            if (t.attempted_models && t.attempted_models.length > 1) {
              details.push(`fallback chain: ${t.attempted_models.join(" → ")}`);
            }
            if (t.pairs_screened !== undefined) details.push(`${t.pairs_screened} drug pairs screened`);
            if (t.claims !== undefined) details.push(`${t.claims} claims in one call`);
            if (t.skipped) details.push(`skipped: ${t.skipped}`);
            if (t.error) details.push(`error: ${String(t.error).slice(0, 80)}`);
            return (
              <tr key={k}>
                <td className="mt-step">{STEP_LABEL[k] || k}</td>
                <td><span className={`mt-chip ${kind(model)}`}>{model}</span></td>
                <td className="mt-details">{details.join(" · ")}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </details>
  );
}
