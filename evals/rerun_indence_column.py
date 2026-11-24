"""Re-run ONLY the Indence column of the safety benchmark with the current
pipeline (e.g. after switching the claim verifier to Gemini Decision 1).
Baseline columns (LLM-only, naive RAG) and the generator model are unchanged.
The previous file is kept as safety_eval_v1_chat_verifier.json.

    backend\\venv\\Scripts\\python.exe evals\\rerun_indence_column.py [categories]
"""
import asyncio
import json
import shutil
import sys

from _common import RESULTS_DIR, make_session, make_pipeline, save_json, now_iso
from run_safety_eval import GEN_MODEL, run_indence, summarise

CATS = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else {"answerable", "subgroup_trap"}


async def main():
    path = RESULTS_DIR / "safety_eval.json"
    backup = RESULTS_DIR / "safety_eval_v1_chat_verifier.json"
    if not backup.exists():
        shutil.copy(path, backup)
    data = json.loads(path.read_text(encoding="utf-8"))
    pipe = make_pipeline(make_session(), GEN_MODEL)
    for r in data["rows"]:
        if r["category"] not in CATS:
            continue
        r["indence"] = await run_indence(pipe, r["question"])
        print(r["id"], "abstained" if r["indence"].get("abstained") else "answered",
              r["indence"].get("abstention_reason", "")[:90] if r["indence"].get("abstained") else "",
              r["indence"].get("error", "")[:90], flush=True)
        await asyncio.sleep(4)
    data["summary"] = {s: summarise(data["rows"], s) for s in ("llm_only", "naive_rag", "indence")}
    data["indence_rerun"] = {"at": now_iso(), "categories": sorted(CATS),
                             "verifier": "gemini-decision-1 (Gemini Decisions API)"}
    print(save_json("safety_eval.json", data))
    print(json.dumps(data["summary"]["indence"], indent=1))


if __name__ == "__main__":
    asyncio.run(main())
