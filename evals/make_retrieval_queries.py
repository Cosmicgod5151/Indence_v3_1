"""Build the silver-standard known-item retrieval benchmark.

Method (InPars / Promptagator-style synthetic query generation):
  1. Stratified random sample (fixed seed) of oncology RCTs and meta-analyses that
     have a Conclusion section.
  2. An LLM writes ONE natural clinician question per paper from its title and
     conclusion, WITHOUT copying the title (paraphrase, no trial acronyms).
  3. The source paper (PMCID) is the known relevant item.

Caveat (reported): other papers may also answer a generated question, so
known-item recall is a LOWER BOUND on true retrieval quality.

    backend\\venv\\Scripts\\python.exe evals\\make_retrieval_queries.py 120
"""
from __future__ import annotations

import asyncio
import json
import random
import sqlite3
import sys

from _common import ROOT, SEED, save_json, now_iso, clients

N = int(sys.argv[1]) if len(sys.argv) > 1 else 120
DB = ROOT / "backend" / "evidence_platform.db"
ONCO = ("cancer", "carcinoma", "tumor", "tumour", "lymphoma", "leukemia", "leukaemia", "melanoma", "myeloma", "sarcoma", "glioma", "neoplasm")

PROMPT = """You write realistic search questions that an oncologist would type into a clinical evidence engine.
For EACH paper below, write ONE specific clinical question that this paper answers.
Rules:
- Do NOT copy the title. Paraphrase; use clinical wording (population, intervention, comparator, outcome).
- Do NOT mention trial names/acronyms, author names, journal or year.
- 10-25 words, phrased as a question.
Return ONLY JSON: {"questions": [{"id": "<paper id>", "question": "..."}]}"""


def sample_papers(n: int) -> list[dict]:
    con = sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True)
    rows = con.execute("""
        SELECT d.pmcid, d.title, r.study_type, s.text
        FROM source_documents d
        JOIN document_revisions r ON r.document_id = d.id
        JOIN document_sections s ON s.revision_id = r.id
        WHERE r.study_type IN ('rct','meta_analysis') AND s.section_kind = 'conclusion'
          AND length(s.text) > 200 AND d.pmcid IS NOT NULL
    """).fetchall()
    seen, papers = set(), []
    for pmcid, title, st, concl in rows:
        if pmcid in seen or not title or not any(k in title.lower() for k in ONCO):
            continue
        seen.add(pmcid)
        papers.append({"pmcid": pmcid, "title": title, "study_type": st, "conclusion": concl[:700]})
    rng = random.Random(SEED)
    rct = [p for p in papers if p["study_type"] == "rct"]
    ma = [p for p in papers if p["study_type"] == "meta_analysis"]
    pick = rng.sample(rct, int(n * 0.6)) + rng.sample(ma, n - int(n * 0.6))
    rng.shuffle(pick)
    print(f"eligible: {len(rct)} RCT + {len(ma)} meta-analyses; sampled {len(pick)}")
    return pick


async def main():
    papers = sample_papers(N)
    llm = clients()[2]
    items = []
    for i in range(0, len(papers), 10):
        batch = papers[i:i + 10]
        body = "\n\n".join(f"[id={p['pmcid']}] TITLE: {p['title']}\nCONCLUSION: {p['conclusion']}" for p in batch)
        for attempt in range(3):
            try:
                parsed, meta = await llm.complete_json(PROMPT, body, max_tokens=4000, temperature=0.3)
                qs = {q["id"]: q["question"] for q in parsed.get("questions", []) if q.get("id") and q.get("question")}
                break
            except Exception as e:  # rate limit etc.
                print("retry", attempt, str(e)[:120])
                await asyncio.sleep(20)
                qs = {}
        for p in batch:
            if p["pmcid"] in qs:
                items.append({"id": f"ret_{len(items) + 1:03d}", "question": qs[p["pmcid"]], "relevant_pmcids": [p["pmcid"]],
                              "source_title": p["title"], "study_type": p["study_type"]})
        print(f"batch {i // 10 + 1}: {len(qs)}/{len(batch)} questions ({meta.get('model') if qs else 'failed'})")
        await asyncio.sleep(4)
    # title-copy check: flag questions sharing a long n-gram with the title
    for it in items:
        tw = it["source_title"].lower().split()
        grams = {" ".join(tw[j:j + 5]) for j in range(max(0, len(tw) - 4))}
        it["copies_title_5gram"] = any(g in it["question"].lower() for g in grams)
    out = {"version": "1.0.0", "generated_at": now_iso(), "seed": SEED, "method": __doc__.strip().splitlines()[0],
           "n": len(items), "items": items}
    path = ROOT / "evals" / "gold_sets" / "retrieval_silver_v1.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(path, "| title 5-gram copies:", sum(i["copies_title_5gram"] for i in items))


if __name__ == "__main__":
    asyncio.run(main())
