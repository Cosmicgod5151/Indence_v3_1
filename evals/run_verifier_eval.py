"""Verifier experiment: Gemini Decision 1 vs a chat-LLM NLI verifier.

Gold labels are known by construction (controlled perturbation, in the spirit of
FactCC-style synthetic consistency tests). From real corpus passages that report a
treatment-comparison ratio with a 95% CI, five claims are generated per passage:

  supported        - the ratio and interval exactly as reported
  number_changed   - point estimate shifted by +0.17          -> contradicted
  digits_swapped   - decimals transposed (0.68 -> 0.86)       -> contradicted
  significance_flip- "no significant difference" although the CI excludes 1 -> contradicted
  unreported       - a subgroup result the passage never mentions -> not established

Experiments
  E1  same claims, same passages: Decision 1 (calibrated choice) vs chat verifier
  E2  evidence visibility: Decision 1 with the old joined-and-truncated premise
      (target passage placed after two other passages, cut at 4,000 chars) vs the
      new per-passage input.

    backend\\venv\\Scripts\\python.exe evals\\run_verifier_eval.py
"""
from __future__ import annotations

import asyncio
import json
import random
import sqlite3
import time

from _common import ROOT, RESULTS_DIR, SEED, save_json, now_iso

from evidence_platform.core.config import settings  # noqa: E402  (loads backend/.env)
from evidence_platform.adapters.models.gemini_decision import GeminiDecisionEntailmentClient  # noqa: E402
from evidence_platform.adapters.models.openrouter import OpenRouterEntailmentClient  # noqa: E402
from evidence_platform.modules.synthesis.effect_map import extract_effects  # noqa: E402

N_PASSAGES = 25
GOLD = {"supported": "entailment", "number_changed": "contradiction", "digits_swapped": "contradiction",
        "significance_flip": "contradiction", "unreported": "neutral"}
MEASURE = {"HR": "hazard ratio", "OR": "odds ratio", "RR": "risk ratio"}
ENDPOINT = {"OS": "overall survival", "PFS": "progression-free survival", "DFS": "disease-free survival",
            "EFS": "event-free survival", "ORR": "objective response rate", "RFS": "relapse-free survival"}


def _f(x: float) -> str:
    return f"{x:.2f}"


def sample_items(rng: random.Random):
    db = sqlite3.connect(str(ROOT / "backend" / "evidence_platform.db"))
    rows = db.execute("select id, text from chunks where text like '%95% CI%' and "
                      "(text like '%HR %' or text like '%hazard ratio%') limit 6000").fetchall()
    rng.shuffle(rows)
    items = []
    for cid, text in rows:
        effs = [e for e in extract_effects(text) if e.context == "treatment_comparison" and e.endpoint
                and e.significant and 0.1 <= e.estimate <= 0.9 and len(text) < 3500]
        if not effs:
            continue
        e = effs[0]
        items.append({"chunk_id": cid, "text": text, "measure": e.measure, "est": e.estimate, "lo": e.ci_low,
                      "hi": e.ci_high, "endpoint": ENDPOINT.get(e.endpoint, e.endpoint), "quote": e.quote})
        if len(items) == N_PASSAGES:
            break
    return items


def claims_for(it: dict) -> list[tuple[str, str]]:
    """Claims anchored to the reported interval, not to an extracted endpoint name.

    A first version named the endpoint ("HR for mortality") from the extractor's guess; manual review
    showed the guess was wrong for some passages (e.g. a DFS or PFS ratio labelled mortality), which
    made the gold label wrong. Anchoring on the exact interval removes that ambiguity. Perturbed
    values that happen to appear anywhere in the passage are skipped.
    """
    m, est, lo, hi, text = MEASURE[it["measure"]], it["est"], it["lo"], it["hi"], it["text"]
    ci = f"95% CI {_f(lo)} to {_f(hi)}"
    out = [("supported", f"The study reported a {m} of {_f(est)} ({ci})."),
           ("significance_flip", f"The {m} of {_f(est)} ({ci}) was not statistically significant, because its "
                                 f"confidence interval included 1."),
           ("unreported", f"In the subgroup of patients aged 80 years or older, the study reported a {m} of 0.41.")]
    wrong = _f(min(est + 0.17, 0.99))
    if wrong not in text:
        out.insert(1, ("number_changed", f"The study reported a {m} of {wrong} ({ci})."))
    s = _f(est)
    swapped = s[:2] + s[3] + s[2] if len(s) == 4 else s
    if swapped != s and swapped not in text:
        out.insert(2, ("digits_swapped", f"The study reported a {m} of {swapped} ({ci})."))
    return out


def distractors(rng: random.Random, n: int) -> list[str]:
    """Passages from other papers that contain no ratio or interval (background / methods text)."""
    db = sqlite3.connect(str(ROOT / "backend" / "evidence_platform.db"))
    rows = db.execute("select text from chunks where length(text) between 1700 and 2300 and text not like '%CI%' "
                      "and text not like '%ratio%' and text not like '%HR%' limit 400").fetchall()
    rng.shuffle(rows)
    return [r[0] for r in rows[:n]]


def metrics(golds: list[str], preds: list[str]) -> dict:
    labels = ["entailment", "contradiction", "neutral"]
    acc = sum(g == p for g, p in zip(golds, preds)) / len(golds)
    f1s = []
    for lab in labels:
        tp = sum(g == lab and p == lab for g, p in zip(golds, preds))
        fp = sum(g != lab and p == lab for g, p in zip(golds, preds))
        fn = sum(g == lab and p != lab for g, p in zip(golds, preds))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return {"accuracy": round(acc, 4), "macro_f1": round(sum(f1s) / 3, 4), "n": len(golds)}


def calibration(golds, details) -> dict:
    """Brier score, expected calibration error and selective accuracy for the calibrated verifier."""
    labels = ["entailment", "contradiction", "neutral"]
    brier, bins = 0.0, [[] for _ in range(10)]
    for g, d in zip(golds, details):
        probs = d.get("probabilities") or {}
        brier += sum((probs.get(l, 0.0) - (1.0 if l == g else 0.0)) ** 2 for l in labels)
        conf = d.get("confidence") or 0.0
        bins[min(int(conf * 10), 9)].append((conf, d["label"] == g))
    n = len(golds)
    ece = sum(len(b) / n * abs(sum(c for c, _ in b) / len(b) - sum(ok for _, ok in b) / len(b)) for b in bins if b)
    curve = []
    for t in (0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
        kept = [(d["label"] == g) for g, d in zip(golds, details) if (d.get("confidence") or 0) >= t]
        curve.append({"threshold": t, "coverage": round(len(kept) / n, 4),
                      "accuracy": round(sum(kept) / len(kept), 4) if kept else None})
    reliability = [{"bin": f"{i / 10:.1f}-{(i + 1) / 10:.1f}", "n": len(b),
                    "mean_conf": round(sum(c for c, _ in b) / len(b), 4) if b else None,
                    "accuracy": round(sum(ok for _, ok in b) / len(b), 4) if b else None} for i, b in enumerate(bins)]
    return {"brier": round(brier / n, 4), "ece": round(ece, 4), "risk_coverage": curve, "reliability": reliability}


async def run_batches(fn, pairs, size, pause=1.0):
    out, lat = [], []
    for i in range(0, len(pairs), size):
        t0 = time.perf_counter()
        res = await fn(pairs[i:i + size])
        if all((r["label"] if isinstance(r, dict) else r) == "unverified" for r in res):
            await asyncio.sleep(max(5, pause))          # transient failure or rate limit: retry once
            t0 = time.perf_counter()
            res = await fn(pairs[i:i + size])
        out.extend(res)
        lat.append(round(time.perf_counter() - t0, 3))
        await asyncio.sleep(pause)
    return out, lat


async def run_e2(items, rows, decision):
    # E2: evidence visibility (truncation) on the supported + number_changed claims
    e2_rows = [r for r in rows if r["type"] in ("supported", "number_changed")]
    old_pairs, new_pairs = [], []
    pool = distractors(random.Random(SEED + 1), 2 * len(items))
    for r in e2_rows:
        # Two ratio-free passages from other papers come first, as when a claim cites several passages.
        others = [pool[2 * r["item"]], pool[2 * r["item"] + 1]]
        target = items[r["item"]]["text"]
        old_pairs.append(((" ".join(others + [target]))[:4000], r["claim"]))     # old behaviour
        new_pairs.append((others + [target], r["claim"]))                        # fixed behaviour
    # Free tier: 50,000 input tokens/minute for gemini-decision-1, so small, paced batches.
    await asyncio.sleep(40)
    old_out, _ = await run_batches(decision.check_entailment_batch_detailed, old_pairs, 10, pause=35)
    new_out, _ = await run_batches(decision.check_entailment_batch_detailed, new_pairs, 10, pause=35)
    e2 = {}
    for t in ("supported", "number_changed"):
        idx = [i for i, r in enumerate(e2_rows) if r["type"] == t]
        e2[t] = {"n": len(idx),
                 "truncated_join_accuracy": round(sum(old_out[i]["label"] == e2_rows[i]["gold"] for i in idx) / len(idx), 4),
                 "full_passages_accuracy": round(sum(new_out[i]["label"] == e2_rows[i]["gold"] for i in idx) / len(idx), 4),
                 "truncated_join_not_established": round(sum(old_out[i]["label"] == "neutral" for i in idx) / len(idx), 4)}
    for t in e2:
        e2[t]["unverified_outputs"] = sum(o["label"] == "unverified" for o in old_out + new_out)
    return e2


async def main(mode: str = "all"):
    rng = random.Random(SEED)
    items = sample_items(rng)
    if mode == "e2":   # re-run only experiment 2 into the existing results file
        data = json.loads((RESULTS_DIR / "verifier_eval.json").read_text(encoding="utf-8"))
        data["summary"]["e2_evidence_visibility"] = await run_e2(items, data["rows"], GeminiDecisionEntailmentClient())
        data["e2_rerun_at"] = now_iso()
        print(save_json("verifier_eval.json", data))
        print(json.dumps(data["summary"]["e2_evidence_visibility"], indent=1))
        return
    rows = []
    for k, it in enumerate(items):
        for ctype, claim in claims_for(it):
            rows.append({"item": k, "type": ctype, "gold": GOLD[ctype], "claim": claim})
    print(f"{len(items)} passages, {len(rows)} claims")

    decision = GeminiDecisionEntailmentClient()  # GEMINI_API_KEY from backend/.env; no fallback, so failures stay visible
    chat = OpenRouterEntailmentClient(settings.LLM_API_KEY)

    # E1: Decision 1 (per-passage input) vs chat verifier, identical claim/passage pairs
    pairs = [([items[r["item"]]["text"]], r["claim"]) for r in rows]
    d_out, d_lat = await run_batches(decision.check_entailment_batch_detailed, pairs, 40)
    for r, d in zip(rows, d_out):
        r["decision"] = d
    chat_pairs = [(items[r["item"]]["text"], r["claim"]) for r in rows]
    c_out, c_lat = await run_batches(chat.check_entailment_batch, chat_pairs, 20)
    for r, lab in zip(rows, c_out):
        r["chat"] = lab
    chat_model = (getattr(chat, "last_meta", {}) or {}).get("model")

    golds = [r["gold"] for r in rows]
    summary = {
        "decision_1": {**metrics(golds, [r["decision"]["label"] for r in rows]),
                       **calibration(golds, [r["decision"] for r in rows]),
                       "requests": len(d_lat), "latency_s_per_request": d_lat,
                       "latency_ms_per_claim": round(1000 * sum(d_lat) / len(rows), 1)},
        "chat_verifier": {**metrics(golds, [r["chat"] for r in rows]), "model": chat_model,
                          "unverified": sum(r["chat"] == "unverified" for r in rows),
                          "requests": len(c_lat), "latency_s_per_request": c_lat,
                          "latency_ms_per_claim": round(1000 * sum(c_lat) / len(rows), 1)},
        "by_type": {},
    }
    for t in GOLD:
        sub = [r for r in rows if r["type"] == t]
        if sub:
            summary["by_type"][t] = {
                "n": len(sub),
                "decision_1": round(sum(r["decision"]["label"] == r["gold"] for r in sub) / len(sub), 4),
                "chat_verifier": round(sum(r["chat"] == r["gold"] for r in sub) / len(sub), 4)}

    summary["e2_evidence_visibility"] = await run_e2(items, rows, decision)

    out = {"generated_at": now_iso(), "seed": SEED, "n_passages": len(items), "n_claims": len(rows),
           "design": __doc__, "summary": summary, "rows": rows,
           "passages": [{k: v for k, v in it.items() if k != "text"} for it in items]}
    print(save_json("verifier_eval.json", out))
    print(json.dumps({k: v for k, v in summary.items() if k != "by_type"}, indent=1)[:2500])
    print(json.dumps(summary["by_type"], indent=1))


if __name__ == "__main__":
    import sys
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "all"))
