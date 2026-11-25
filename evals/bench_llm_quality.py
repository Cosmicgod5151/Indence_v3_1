"""LLM quality/latency benchmark on IDENTICAL real evidence bundles.

For each question the evidence bundle is built once by the live pipeline
(retrieval + rerank + MMR), then every candidate model receives exactly the same
system prompt + evidence. Objective quality signals:
  * valid_json            - parsable structured answer
  * invalid_citation_ids  - cited E/S ids that were not in the bundle (fabricated)
  * sources_used          - distinct evidence ids cited (breadth of synthesis)
  * numeric_faithfulness  - share of numbers in the answer text that occur in the
                            evidence text (catches invented statistics)
  * latency_s

    backend\\venv\\Scripts\\python.exe evals\\bench_llm_quality.py gemini:gemini-3.8-flash gemini:gemini-2.5-pro ...
"""
from __future__ import annotations

import asyncio
import re
import sys
import time

from _common import make_session, make_pipeline, save_json, now_iso

QUESTIONS = [
    "Is trastuzumab deruxtecan effective in HER2-positive metastatic breast cancer after prior trastuzumab?",
    "Do CDK4/6 inhibitors improve progression-free survival in HR-positive HER2-negative advanced breast cancer?",
]
NUM_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])")


def answer_texts(j: dict) -> list[str]:
    out = []
    for k in ("recommendation", "clinical_reasoning", "alternatives", "bottom_line"):
        if isinstance(j.get(k), dict) and j[k].get("text"):
            out.append(j[k]["text"])
    for k in ("supporting_evidence", "direct_answer_claims"):
        for it in j.get(k) or []:
            if isinstance(it, dict) and it.get("text"):
                out.append(it["text"])
    return out


def cited_ids(j: dict) -> list[str]:
    ids = []
    for k in ("recommendation", "clinical_reasoning", "alternatives", "bottom_line"):
        sec = j.get(k)
        if isinstance(sec, dict):
            ids += list(sec.get("evidence_ids") or []) + [s.split(".S")[0] for s in sec.get("sentence_ids") or []]
    for k in ("supporting_evidence", "direct_answer_claims"):
        for it in j.get(k) or []:
            if isinstance(it, dict):
                ids += list(it.get("evidence_ids") or []) + [s.split(".S")[0] for s in it.get("sentence_ids") or []]
    return ids


async def main(models: list[str]):
    from evidence_platform.modules.pipeline.query_pipeline import PipelineOptions
    from evidence_platform.adapters.models import openrouter as orm
    db = make_session()
    pipe = make_pipeline(db)
    bundles = []
    captured = {}

    async def spy(self, provider, model, system_prompt, user_content, max_tokens, temperature, compact_suffix=""):
        captured["sys"], captured["user"] = system_prompt, user_content
        raise orm.LLMUnavailableError("captured")

    real_call = orm.OpenRouterLLMClient._call_one
    for q in QUESTIONS:
        prep = await pipe.run(q, {}, PipelineOptions(run_synthesis=False))
        orm.OpenRouterLLMClient._call_one = spy
        await pipe.synthesizer.synthesize(q, prep.assembled, intent=prep.intent, clusters=prep.clusters,
                                          contradiction_info=prep.contradiction_info)
        orm.OpenRouterLLMClient._call_one = real_call
        evidence_text = " ".join(c.text for c in prep.assembled.selected_candidates)
        valid_ids = {f"E{i + 1:02d}" for i in range(len(prep.assembled.selected_candidates))}
        bundles.append((q, captured["sys"], captured["user"], evidence_text, valid_ids))

    from evidence_platform.core.config import settings
    rows = []
    for q, sys_p, user_p, ev_text, valid_ids in bundles:
        ev_nums = set(NUM_RE.findall(ev_text))
        for spec in models:
            provider, model = orm.parse_model_chain(spec)[0]
            client = orm.OpenRouterLLMClient(settings.LLM_API_KEY, spec, timeout_s=120)
            t0 = time.perf_counter()
            row = {"question": q, "model": spec}
            try:
                j, meta = await client._call_one(provider, model, sys_p, user_p, 8000, 0.0)
                texts = answer_texts(j)
                nums = [n for t in texts for n in NUM_RE.findall(t)]
                ids = cited_ids(j)
                row.update({
                    "ok": True, "latency_s": round(time.perf_counter() - t0, 1), "status": j.get("status"),
                    "invalid_citation_ids": sum(1 for i in ids if i not in valid_ids),
                    "cited_ids": len(ids), "sources_used": len({i for i in ids if i in valid_ids}),
                    "n_claims": len(j.get("direct_answer_claims") or []),
                    "answer_words": sum(len(t.split()) for t in texts),
                    "numbers_in_answer": len(nums),
                    "numeric_faithfulness": round(sum(n in ev_nums for n in nums) / len(nums), 3) if nums else None,
                    "completion_tokens": (meta.get("usage") or {}).get("completion_tokens"),
                })
            except Exception as e:
                row.update({"ok": False, "latency_s": round(time.perf_counter() - t0, 1), "error": str(e)[:200]})
            rows.append(row)
            print({k: v for k, v in row.items() if k != "question"}, flush=True)
            await asyncio.sleep(2)
    print(save_json("llm_quality_benchmark.json", {"generated_at": now_iso(), "questions": QUESTIONS, "rows": rows}))


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
