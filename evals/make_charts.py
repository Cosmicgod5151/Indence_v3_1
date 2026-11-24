"""Render report figures from evals/results/*.json (light mode, print-ready, 200 dpi).

Palette: dataviz reference palette. Ordered configurations use one blue ramp
(light -> dark = more pipeline layers); the three compared systems use the fixed
categorical order (slot 1 blue = Indence, slot 2 orange = Naive RAG, slot 3 aqua =
LLM-only). Text stays in ink colours; values are direct-labelled selectively.

    backend\\venv\\Scripts\\python.exe evals\\make_charts.py
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from _common import RESULTS_DIR  # noqa: E402

FIG = RESULTS_DIR / "figures"
FIG.mkdir(exist_ok=True)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
RAMP = ["#b7d3f6", "#86b6ef", "#5598e7", "#256abf", "#0d366b"]       # ordered configs
SYS = {"indence": "#2a78d6", "naive_rag": "#eb6834", "llm_only": "#1baf7a"}
BEFORE = "#b4b2a9"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK, "axes.titleweight": "bold",
    "axes.titlesize": 12, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


def load(name):
    p = RESULTS_DIR / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _finish(ax, fig, path, ylabel=None):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote", path)


def retrieval_chart():
    d = load("retrieval_ablation.json")
    if not d:
        return
    modes = d["modes"]
    labels = {"sparse": "Sparse only", "dense": "Dense only", "hybrid": "Hybrid (RRF)",
              "hybrid_rerank": "Hybrid + cross-encoder", "full": "Full Indence"}
    metrics = [("recall@5", "Recall@5"), ("recall@10", "Recall@10"), ("recall@20", "Recall@20"),
               ("mrr@10", "MRR@10"), ("ndcg@10", "nDCG@10")]
    for subset, title in (("all_queries", f"all {d['n_queries']} questions"),
                          ("paraphrased_only", f"{d['n_paraphrased']} strictly paraphrased questions")):
        fig, ax = plt.subplots(figsize=(9, 4.2))
        w = 0.16
        for i, m in enumerate(modes):
            vals = [d[subset][m][k]["mean"] for k, _ in metrics]
            los = [d[subset][m][k]["mean"] - d[subset][m][k]["ci95"][0] for k, _ in metrics]
            his = [d[subset][m][k]["ci95"][1] - d[subset][m][k]["mean"] for k, _ in metrics]
            xs = [j + (i - 2) * w for j in range(len(metrics))]
            ax.bar(xs, vals, w * 0.9, color=RAMP[i], label=labels[m], yerr=[los, his],
                   error_kw={"elinewidth": 0.8, "ecolor": INK2, "capsize": 0})
            if m == "full":
                for x, v in zip(xs, vals):
                    ax.text(x, v + 0.03, f"{v:.2f}", ha="center", fontsize=8, color=INK)
        ax.set_xticks(range(len(metrics)), [l for _, l in metrics])
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Retrieval ablation — known-item benchmark ({title})", loc="left")
        ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=8.5)
        _finish(ax, fig, FIG / f"retrieval_ablation_{subset}.png", "Score (95% bootstrap CI)")


def safety_chart():
    d = load("safety_eval.json")
    if not d:
        return
    s = d["summary"]

    def frac(txt):
        a, b = (int(x) for x in txt.split("/"))
        return 1 - a / b if b else None

    rows = [
        ("Correct abstention\n(out-of-scope, fictional drug, PHI)", lambda x: x["correct_abstention_rate"]),
        ("Answered answerable\nquestions", lambda x: x["answer_rate_on_answerable"]),
        ("PHI kept off the\nmodel provider", lambda x: frac(x["phi_sent_to_provider"])),
        ("Numeric faithfulness\n(numbers found in evidence)", lambda x: x["numeric_faithfulness"]),
        ("Citations resolve to\nreal / provided sources", lambda x: (1 - x["fabricated_ref_rate"]) if x.get("fabricated_ref_rate") is not None
         else (1 - x["invalid_citation_rate"]) if x.get("invalid_citation_rate") is not None else None),
        ("No wrong-subgroup\ncitations", lambda x: (1 - x["subgroup_leakage_rate"]) if x.get("subgroup_leakage_rate") is not None else None),
    ]
    order = [("indence", "Indence"), ("naive_rag", "Naive RAG"), ("llm_only", "LLM only")]
    fig, ax = plt.subplots(figsize=(9, 5.6))
    h = 0.26
    for i, (key, name) in enumerate(order):
        vals = [f(s[key]) for _, f in rows]
        ys = [j + (i - 1) * h for j in range(len(rows))]
        ax.barh(ys, [v if v is not None else 0 for v in vals], h * 0.88, color=SYS[key], label=name)
        for y, v in zip(ys, vals):
            ax.text((v or 0) + 0.01, y, "n/a" if v is None else f"{v:.0%}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(rows)), [r for r, _ in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title(f"Safety & grounding on {d['n_items']} questions\n"
                 f"all systems use the same generator: {d['generator_model'].split(',')[0].split(':')[-1].split('#')[0].strip()}",
                 loc="left")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.07))
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "safety_comparison.png", dpi=200)
    plt.close(fig)
    print("wrote", FIG / "safety_comparison.png")


def latency_chart():
    b, a = load("latency_before.json"), load("latency_after_retrieval_only.json")
    if not (b and a):
        return
    stages = [("retrieval_s", "Retrieval"), ("context_assembly_s", "Rerank + MMR assembly")]
    fig, ax = plt.subplots(figsize=(8, 2.8))
    for j, (k, name) in enumerate(stages):
        bv, av = b["summary"][k]["p50"], a["summary"][k]["p50"]
        ax.barh(j - 0.18, bv, 0.34, color=BEFORE, label="Before (Milestone-1 code, n=52)" if j == 0 else None)
        ax.barh(j + 0.18, av, 0.34, color=SYS["indence"], label="After Phase-1 fixes (n=10)" if j == 0 else None)
        ax.text(bv + 0.08, j - 0.18, f"{bv:.2f} s", va="center", fontsize=8.5, color=INK2)
        ax.text(av + 0.08, j + 0.18, f"{av:.2f} s  ({bv / av:.1f}× faster)", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(stages)), [n for _, n in stages])
    ax.invert_yaxis()
    ax.set_xlabel("Median latency (s)")
    ax.set_title("Pipeline latency before vs after (excluding the LLM)", loc="left")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "latency_before_after.png", dpi=200)
    plt.close(fig)
    print("wrote", FIG / "latency_before_after.png")


def case_chart():
    d = load("case_mode_eval.json")
    v1 = load("case_mode_eval_v1.json")      # extraction measured with gemini-2.5-flash
    if not d:
        return
    rows = d["rows"]
    ext_rows = (v1 or d)["rows"]
    names = [r["id"].split("_", 2)[2].replace("_", " ") for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.6))
    xs = range(len(rows))
    a1.bar([x - 0.2 for x in xs], [r["llm"]["gold_recall"] for r in ext_rows], 0.38, color=SYS["indence"], label="LLM (gemini-2.5-flash) + span verification")
    a1.bar([x + 0.2 for x in xs], [r["regex"]["gold_recall"] for r in ext_rows], 0.38, color=BEFORE, label="Deterministic regex fallback")
    a1.set_xticks(list(xs), names, rotation=35, ha="right", fontsize=8)
    a1.set_ylim(0, 1.05)
    a1.set_title("Fact extraction — recall of gold facts", loc="left")
    a1.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.38), ncol=1)
    a1.grid(axis="y", color=GRID, linewidth=0.8)
    a1.set_axisbelow(True)
    gm = [r["applicability"]["general_mean"] or 0 for r in rows]
    cm = [r["applicability"]["case_mean"] or 0 for r in rows]
    a2.bar([x - 0.2 for x in xs], gm, 0.38, color=BEFORE, label="General search")
    a2.bar([x + 0.2 for x in xs], cm, 0.38, color=SYS["indence"], label="Case-aware search")
    a2.set_xticks(list(xs), names, rotation=35, ha="right", fontsize=8)
    a2.set_ylim(0, 1.05)
    a2.set_title("Mean applicability of top-8 studies", loc="left")
    a2.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.38), ncol=2)
    for x, r in zip(xs, rows):
        if r["applicability"]["general_mean"] is None:
            a2.text(x - 0.2, 0.02, "n/a", ha="center", fontsize=7.5, color=INK2, rotation=90)
    a2.grid(axis="y", color=GRID, linewidth=0.8)
    a2.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "case_mode_eval.png", dpi=200)
    plt.close(fig)
    print("wrote", FIG / "case_mode_eval.png")


if __name__ == "__main__":
    retrieval_chart()
    safety_chart()
    latency_chart()
    case_chart()
