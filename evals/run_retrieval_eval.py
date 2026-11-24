"""Retrieval ablation on the silver known-item benchmark (no LLM calls).

Configurations (same corpus snapshot, same queries):
  sparse        BGE-M3 learned-sparse (lexical) channel only
  dense         BGE-M3 dense channel only
  hybrid        dense + sparse fused with Reciprocal Rank Fusion (k=60)
  hybrid_rerank hybrid, re-ordered by the bge-reranker-v2-m3 cross-encoder
  full          Indence: hybrid -> population filter -> evidence hierarchy ->
                cross-encoder blend (the ranking used before MMR selection)

Rankings are evaluated at DOCUMENT level (chunks de-duplicated by PMCID).
Metrics: Recall@5/10/20, MRR@10, nDCG@10 with 95% bootstrap CIs (2,000 resamples).

    backend\\venv\\Scripts\\python.exe evals\\run_retrieval_eval.py
"""
from __future__ import annotations

import asyncio
import json
import math
import random
import time

from _common import ROOT, SEED, make_session, make_pipeline, save_json, now_iso, summarize_latency

import sys
ALL_MODES = ["sparse", "dense", "hybrid", "hybrid_rerank", "full"]
RUN_MODES = sys.argv[1].split(",") if len(sys.argv) > 1 else ALL_MODES
MODES = ALL_MODES
KS = (5, 10, 20)


def doc_ranking(cands) -> list[str]:
    seen, out = set(), []
    for c in cands:
        if c.pmcid and c.pmcid not in seen:
            seen.add(c.pmcid)
            out.append(c.pmcid)
    return out


def metrics_for(rank: int | None) -> dict[str, float]:
    m = {f"recall@{k}": float(rank is not None and rank <= k) for k in KS}
    m["mrr@10"] = 1.0 / rank if rank is not None and rank <= 10 else 0.0
    m["ndcg@10"] = 1.0 / math.log2(rank + 1) if rank is not None and rank <= 10 else 0.0
    return m


def bootstrap_ci(values: list[float], n: int = 2000) -> tuple[float, float]:
    rng = random.Random(SEED)
    means = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(n))
    return round(means[int(0.025 * n)], 3), round(means[int(0.975 * n)], 3)


async def main():
    from evidence_platform.modules.pipeline.query_pipeline import PipelineOptions
    gold = json.loads((ROOT / "evals" / "gold_sets" / "retrieval_silver_v1.json").read_text(encoding="utf-8"))
    items = gold["items"]
    pipe = make_pipeline(make_session())
    await pipe.run("warm up the GPU models", {}, PipelineOptions(retrieval_mode="hybrid_rerank"))
    per_query = []
    lat = {m: [] for m in MODES}
    for n, it in enumerate(items, 1):
        row = {"id": it["id"], "question": it["question"], "relevant": it["relevant_pmcids"][0],
               "paraphrased": not it.get("copies_title_5gram", False), "ranks": {}}
        for mode in RUN_MODES:
            t0 = time.perf_counter()
            res = await pipe.run(it["question"], {}, PipelineOptions(retrieval_mode=mode, run_synthesis=False))
            lat[mode].append(time.perf_counter() - t0)
            ranking = doc_ranking(res.ranked_candidates)
            row["ranks"][mode] = ranking.index(row["relevant"]) + 1 if row["relevant"] in ranking else None
        per_query.append(row)
        if n % 10 == 0:
            print(f"{n}/{len(items)} done", flush=True)

    def summarise(rows):
        out = {}
        for mode in MODES:
            ms = [metrics_for(r["ranks"][mode]) for r in rows]
            out[mode] = {}
            for key in ms[0]:
                vals = [m[key] for m in ms]
                lo, hi = bootstrap_ci(vals)
                out[mode][key] = {"mean": round(sum(vals) / len(vals), 3), "ci95": [lo, hi]}
        return out

    # Merge modes that were not re-run from the previous result file (unchanged code paths).
    prev_path = ROOT / "evals" / "results" / "retrieval_ablation.json"
    if set(RUN_MODES) != set(ALL_MODES) and prev_path.exists():
        prev = {r["id"]: r for r in json.loads(prev_path.read_text(encoding="utf-8"))["per_query"]}
        for r in per_query:
            for m in ALL_MODES:
                if m not in r["ranks"]:
                    r["ranks"][m] = prev[r["id"]]["ranks"][m]
        old_lat = json.loads(prev_path.read_text(encoding="utf-8"))["latency_s"]
    else:
        old_lat = {}
    paraphrased = [r for r in per_query if r["paraphrased"]]
    result = {
        "generated_at": now_iso(),
        "benchmark": "evals/gold_sets/retrieval_silver_v1.json",
        "n_queries": len(per_query), "n_paraphrased": len(paraphrased),
        "modes": MODES,
        "all_queries": summarise(per_query),
        "paraphrased_only": summarise(paraphrased),
        "latency_s": {m: (summarize_latency(v) if v else old_lat.get(m)) for m, v in lat.items()},
        "modes_rerun": RUN_MODES,
        "per_query": per_query,
        "notes": "Document-level known-item evaluation; silver labels (one relevant paper per question) make these "
                 "lower bounds. 'full' excludes papers removed by the population hard filter.",
    }
    print(save_json("retrieval_ablation.json", result))
    for subset in ("all_queries", "paraphrased_only"):
        print(f"\n{subset} (n={result['n_queries'] if subset == 'all_queries' else result['n_paraphrased']})")
        for mode in MODES:
            s = result[subset][mode]
            print(f"  {mode:14s} R@5 {s['recall@5']['mean']:.3f}  R@10 {s['recall@10']['mean']:.3f} {s['recall@10']['ci95']}  "
                  f"R@20 {s['recall@20']['mean']:.3f}  MRR@10 {s['mrr@10']['mean']:.3f}  nDCG@10 {s['ndcg@10']['mean']:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
