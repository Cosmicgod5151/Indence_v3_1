"""Latency + LLM-model benchmark for the Indence pipeline.

Usage (from repo root, backend venv):
    backend\\venv\\Scripts\\python.exe evals\\bench_latency.py models      # compare LLMs on identical evidence
    backend\\venv\\Scripts\\python.exe evals\\bench_latency.py e2e         # end-to-end latency, default model
    backend\\venv\\Scripts\\python.exe evals\\bench_latency.py retrieval   # pipeline latency without the LLM stages

Outputs: evals/results/llm_model_benchmark.json, evals/results/latency_after.json
"""
from __future__ import annotations

import asyncio
import re
import sys
import time

from _common import make_session, make_pipeline, summarize_latency, save_json, now_iso, pct

QUERIES = [
    "Is trastuzumab deruxtecan effective in HER2-positive metastatic breast cancer after prior trastuzumab?",
    "Does osimertinib improve overall survival in EGFR-mutant advanced non-small cell lung cancer?",
    "What is the benefit of pembrolizumab plus chemotherapy in first-line metastatic non-small cell lung cancer?",
    "Do CDK4/6 inhibitors improve progression-free survival in HR-positive HER2-negative advanced breast cancer?",
    "Is cetuximab beneficial in RAS wild-type metastatic colorectal cancer?",
    "Does olaparib maintenance improve survival in BRCA-mutated ovarian cancer?",
    "What is the role of immunotherapy in MSI-high colorectal cancer?",
    "Does adjuvant radiotherapy after breast-conserving surgery reduce recurrence?",
    "Is abiraterone effective in metastatic castration-sensitive prostate cancer?",
    "Does neoadjuvant chemotherapy improve outcomes in triple-negative breast cancer?",
]

MODELS = [
    "meta-llama/llama-3.3-70b-instruct",
    "google/gemini-2.5-flash",
    "openai/gpt-4.1-mini",
    "mistralai/mistral-small-3.2-24b-instruct",
]


def citation_check(res, evidence_map) -> dict:
    """Do all cited evidence/sentence ids exist in the evidence bundle given to the model?"""
    cited, bad = 0, 0
    sections = [res.recommendation, res.clinical_reasoning, res.alternatives, res.bottom_line,
                *(res.supporting_evidence or []), *(res.direct_claims or [])]
    for sec in sections:
        if not isinstance(sec, dict):
            continue
        for eid in sec.get("evidence_ids", []) or []:
            cited += 1
            bad += eid not in evidence_map
        for sid in sec.get("sentence_ids", []) or []:
            cited += 1
            bad += sid.split(".S")[0] not in evidence_map
    return {"cited_ids": cited, "invalid_ids": bad}


async def bench_models(n_queries: int = 3):
    from evidence_platform.modules.pipeline.query_pipeline import PipelineOptions
    from evidence_platform.modules.synthesis.synthesizer import AnswerSynthesizer
    from _common import clients
    db = make_session()
    pipe = make_pipeline(db)
    await pipe.run(QUERIES[0], {}, PipelineOptions(run_synthesis=False))  # warm-up (GPU load)
    rows = []
    for q in QUERIES[:n_queries]:
        prep = await pipe.run(q, {}, PipelineOptions(run_synthesis=False))
        for model in MODELS:
            await asyncio.sleep(3)  # let OpenRouter's in-flight budget settle between calls
            llm = clients(model)[2]
            synth = AnswerSynthesizer(llm)
            t0 = time.perf_counter()
            res = await synth.synthesize(q, prep.assembled, intent=prep.intent, clusters=prep.clusters,
                                         contradiction_info=prep.contradiction_info)
            dt = time.perf_counter() - t0
            ok = res.llm_error is None and res.status in ("answer", "insufficient_evidence")
            cc = citation_check(res, res.evidence_map)
            usage = (res.llm_meta or {}).get("usage", {}) or {}
            rows.append({
                "query": q, "model": model, "latency_s": round(dt, 2), "valid_json": ok,
                "status": res.status, "llm_error": res.llm_error,
                "has_recommendation": bool(res.recommendation and res.recommendation.get("text")),
                "n_claims": len(res.direct_claims or []), **cc,
                "completion_tokens": usage.get("completion_tokens"), "prompt_tokens": usage.get("prompt_tokens"),
            })
            print(f"{model:45s} {dt:6.1f}s valid={ok} status={res.status} claims={len(res.direct_claims or [])} invalid_ids={cc['invalid_ids']}/{cc['cited_ids']}")
    summary = {}
    for model in MODELS:
        r = [x for x in rows if x["model"] == model]
        lat = [x["latency_s"] for x in r]
        cited = sum(x["cited_ids"] for x in r) or 1
        summary[model] = {
            **summarize_latency(lat),
            "valid_json_rate": round(sum(x["valid_json"] for x in r) / len(r), 3),
            "answer_rate": round(sum(x["status"] == "answer" for x in r) / len(r), 3),
            "invalid_citation_rate": round(sum(x["invalid_ids"] for x in r) / cited, 4),
            "mean_claims": round(sum(x["n_claims"] for x in r) / len(r), 2),
        }
    out = {"generated_at": now_iso(), "n_queries": n_queries, "models": MODELS, "summary": summary, "rows": rows}
    print(save_json("llm_model_benchmark.json", out))
    for m, s in summary.items():
        print(m, s)


async def bench_e2e(synthesis: bool = True):
    from evidence_platform.core.config import settings
    from evidence_platform.modules.pipeline.query_pipeline import PipelineOptions
    db = make_session()
    pipe = make_pipeline(db)
    await pipe.run(QUERIES[0], {}, PipelineOptions(run_synthesis=False))  # warm-up
    rows = []
    for q in QUERIES:
        t0 = time.perf_counter()
        res = await pipe.run(q, {}, PipelineOptions(retrieval_mode="full", run_synthesis=synthesis))
        wall = time.perf_counter() - t0
        rows.append({"query": q, "status": res.final_status, "wall_s": round(wall, 3), **res.timings,
                     "n_claims": len(res.final_claims), "abstention_reason": res.abstention_reason,
                     "llm_error": res.synth_res.llm_error if res.synth_res else None})
        print(f"{wall:6.1f}s {res.final_status:22s} {q[:70]}")
    stages = ["retrieval_s", "context_assembly_s", "synthesis_s", "grounding_s", "total_e2e_s"]
    summary = {s: summarize_latency([r[s] for r in rows if s in r]) for s in stages if any(s in r for r in rows)}
    out = {"generated_at": now_iso(), "llm_model": settings.LLM_MODEL_NAME,
           "entailment_model": settings.ENTAILMENT_MODEL_NAME or settings.LLM_MODEL_NAME,
           "summary": summary, "rows": rows}
    print(save_json("latency_after.json" if synthesis else "latency_after_retrieval_only.json", out))
    for s, v in summary.items():
        print(s, v)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "e2e"
    if mode == "models":
        asyncio.run(bench_models())
    else:
        asyncio.run(bench_e2e(synthesis=(mode != "retrieval")))
