"""Patient Case Mode evaluation on the 8 synthetic cases.

1. Fact extraction (LLM + span verification) vs the deterministic regex fallback:
   - gold recall     gold facts matched (same field; value contains all gold tokens)
   - traceability    verified (anchored) facts / facts proposed by the extractor
   - conflicts       expected conflict detected (case 2: PD-L1 80% vs 45%)
2. Applicability: case-aware retrieval vs general retrieval of the same question
   (no LLM; ranking only) — mean applicability score of the top-8 studies and the
   number of studies judged "Different population".

    backend\\venv\\Scripts\\python.exe evals\\run_case_eval.py
"""
from __future__ import annotations

import asyncio
import json

from _common import ROOT, make_session, make_pipeline, clients, save_json, now_iso


def gold_recall(facts: list[dict], gold: list[dict]) -> tuple[int, list[dict]]:
    from evidence_platform.modules.clinical.case_context import fact_label
    used, hit, missed = set(), 0, []
    for g in gold:
        ok = False
        for i, f in enumerate(facts):
            if i in used or f["field"] != g["field"]:
                continue
            label = fact_label(f).lower()
            if all(tok.lower() in label for tok in g["must_contain"]):
                used.add(i)
                ok = True
                break
        hit += ok
        if not ok:
            missed.append(g)
    return hit, missed


async def main():
    from evidence_platform.modules.clinical.case_context import extract_case_facts, build_case_query
    from evidence_platform.modules.clinical.applicability import build_applicability_matrix
    from evidence_platform.modules.pipeline.query_pipeline import PipelineOptions
    cases = json.loads((ROOT / "evals" / "cases" / "synthetic_cases.json").read_text(encoding="utf-8"))["cases"]
    import os
    llm = clients(os.environ.get("EXTRACT_MODEL") or None)[2]
    pipe = make_pipeline(make_session())
    rows = []
    for case in cases:
        row = {"id": case["id"], "n_gold": len(case["gold_facts"])}
        for method, client in (("llm", llm), ("regex", None)):
            out = await extract_case_facts(case["text"], client)
            hit, missed = gold_recall(out["facts"], case["gold_facts"])
            row[method] = {
                "method": out["extraction_method"], "proposed": out["traceability"]["proposed"],
                "verified": out["traceability"]["verified"], "dropped": out["traceability"]["dropped"],
                "traceability": out["traceability"]["rate"], "gold_hit": hit,
                "gold_recall": round(hit / len(case["gold_facts"]), 3),
                "missed": [f"{m['field']}:{'+'.join(m['must_contain'])}" for m in missed],
                "conflicts": [c["name"] for c in out["conflicts"]],
                "missing_data": [m["variable"] for m in out["missing_data"]],
            }
            if method == "llm":
                facts = [f for f in out["facts"] if not f.get("conflict_with")]
                row["llm_facts"] = out["facts"]
            await asyncio.sleep(2 if method == "llm" else 0)
        if case.get("expected_conflict"):
            row["conflict_detected"] = any(case["expected_conflict"].lower() in (c or "").lower() for c in row["llm"]["conflicts"])

        # Applicability: general vs case-aware ranking (no LLM)
        q = case["question"]
        case_query, case_intent = build_case_query(q, facts)
        gen = await pipe.run(q, {}, PipelineOptions(run_synthesis=False))
        cas = await pipe.run(case_query, {}, PipelineOptions(run_synthesis=False, intent_override=case_intent))
        pool = lambda r: (list(r.assembled.selected_candidates) + list(r.ranked_candidates)) if r.assembled else list(r.ranked_candidates)  # noqa: E731
        gm = build_applicability_matrix(pool(gen), facts, limit=8)
        cm = build_applicability_matrix(pool(cas), facts, limit=8)
        row["applicability"] = {
            "general_mean": gm["mean_applicability"], "case_mean": cm["mean_applicability"],
            "general_different_population": gm["n_different_population"],
            "case_different_population": cm["n_different_population"],
            "shared_studies": len({r["pmcid"] for r in gm["rows"]} & {r["pmcid"] for r in cm["rows"]}),
            "case_query": case_query,
        }
        rows.append(row)
        print(f"{case['id']:30s} LLM recall {row['llm']['gold_recall']:.2f} trace {row['llm']['traceability']} | "
              f"regex recall {row['regex']['gold_recall']:.2f} | applic general {gm['mean_applicability']} -> case {cm['mean_applicability']} "
              f"| diff-pop {gm['n_different_population']} -> {cm['n_different_population']}", flush=True)

    def agg(method):
        gold = sum(r["n_gold"] for r in rows)
        return {
            "gold_recall_micro": round(sum(r[method]["gold_hit"] for r in rows) / gold, 3),
            "traceability_micro": round(sum(r[method]["verified"] for r in rows) / max(1, sum(r[method]["proposed"] for r in rows)), 3),
            "facts_dropped": sum(r[method]["dropped"] for r in rows),
            "facts_verified": sum(r[method]["verified"] for r in rows),
        }
    gm_vals = [r["applicability"]["general_mean"] for r in rows if r["applicability"]["general_mean"] is not None]
    cm_vals = [r["applicability"]["case_mean"] for r in rows if r["applicability"]["case_mean"] is not None]
    summary = {
        "n_cases": len(rows), "n_gold_facts": sum(r["n_gold"] for r in rows),
        "llm_extraction": agg("llm"), "regex_fallback": agg("regex"),
        "conflict_detection": [r.get("conflict_detected") for r in rows if "conflict_detected" in r],
        "applicability": {
            "general_mean_of_means": round(sum(gm_vals) / len(gm_vals), 3) if gm_vals else None,
            "case_aware_mean_of_means": round(sum(cm_vals) / len(cm_vals), 3) if cm_vals else None,
            "general_different_population_total": sum(r["applicability"]["general_different_population"] for r in rows),
            "case_different_population_total": sum(r["applicability"]["case_different_population"] for r in rows),
            "cases_improved": sum(1 for r in rows if (r["applicability"]["case_mean"] or 0) > (r["applicability"]["general_mean"] or 0)),
        },
    }
    import os
    print(save_json("case_mode_eval.json", {"generated_at": now_iso(), "extraction_model": os.environ.get("EXTRACT_MODEL") or "app default chain",
                                            "summary": summary, "rows": rows}))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
