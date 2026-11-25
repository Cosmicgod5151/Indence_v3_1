"""Re-run only the (item, system) pairs that failed with a transient error
(e.g. DNS / network blips) in evals/results/safety_eval.json, then recompute the
summary. Successful results are never re-run or overwritten.

    backend\\venv\\Scripts\\python.exe evals\\retry_safety_errors.py
"""
import asyncio
import json

from _common import RESULTS_DIR, make_session, make_pipeline, clients, save_json, now_iso
from run_safety_eval import GEN_MODEL, run_llm_only, run_naive_rag, run_indence, summarise


async def main():
    data = json.loads((RESULTS_DIR / "safety_eval.json").read_text(encoding="utf-8"))
    rows = data["rows"]
    todo = [(r, s) for r in rows for s in ("llm_only", "naive_rag", "indence") if "error" in r.get(s, {})]
    print(f"{len(todo)} failed calls to retry")
    if not todo:
        return
    pipe = make_pipeline(make_session())
    llm = clients(GEN_MODEL)[2]
    for r, s in todo:
        q = r["question"]
        r[s] = (await run_llm_only(llm, q) if s == "llm_only" else
                await run_naive_rag(pipe, llm, q) if s == "naive_rag" else await run_indence(pipe, q))
        print(r["id"], s, "ok" if "error" not in r[s] else "still failing: " + r[s]["error"][:80], flush=True)
        await asyncio.sleep(4)
    data["summary"] = {s: summarise(rows, s) for s in ("llm_only", "naive_rag", "indence")}
    data["retried_at"] = now_iso()
    print(save_json("safety_eval.json", data))
    print(json.dumps(data["summary"], indent=1))


if __name__ == "__main__":
    asyncio.run(main())
