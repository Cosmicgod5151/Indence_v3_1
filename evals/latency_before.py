"""Historical ("before") latency baseline, computed from the audit traces the
pre-fix system wrote to backend/evidence_platform.db (retrieval_traces.timings).

Only traces created before the Phase-1 fixes are used (traces written by the
new pipeline carry a `request_wall_s` key and are excluded).

    backend\\venv\\Scripts\\python.exe evals\\latency_before.py
"""
import json
import sqlite3

from _common import ROOT, summarize_latency, save_json, now_iso

DB = ROOT / "backend" / "evidence_platform.db"
STAGES = ["retrieval_s", "context_assembly_s", "synthesis_s", "grounding_s", "total_e2e_s"]


def main():
    con = sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True)
    rows = []
    for (t,) in con.execute("SELECT timings FROM retrieval_traces WHERE timings IS NOT NULL"):
        d = json.loads(t) if isinstance(t, str) else t
        if isinstance(d, dict) and "request_wall_s" not in d and "total_e2e_s" in d:
            rows.append(d)
    summary = {s: summarize_latency([r[s] for r in rows if s in r]) for s in STAGES}
    out = {
        "generated_at": now_iso(),
        "source": "backend/evidence_platform.db :: retrieval_traces (pre-fix runs)",
        "note": "total_e2e_s in the old code summed the timed stages (not wall clock).",
        "n_traces": len(rows),
        "summary": summary,
    }
    print(save_json("latency_before.json", out))
    for s, v in summary.items():
        print(f"{s:22s} {v}")


if __name__ == "__main__":
    main()
