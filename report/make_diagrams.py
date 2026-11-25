"""Draw the report's design diagrams with matplotlib (no graphviz needed).

    backend\\venv\\Scripts\\python.exe report\\make_diagrams.py
Outputs PNGs to report/figures/.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse, FancyArrowPatch

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams["font.family"] = "Times New Roman"

# Palette: one hue per layer, light fills, dark ink.
C = {
    "ui": "#dbe9f6", "api": "#e3f1e6", "pipe": "#fdf1d8", "gate": "#f9dede",
    "model": "#ece4f4", "store": "#e6e6e6", "ext": "#fbe6d4", "ink": "#1d1d1d",
    "edge": "#4a4a4a",
}


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    ax.set_xlim(0, w * 10)
    ax.set_ylim(0, h * 10)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, fill, fs=9, bold=False, style="round,pad=0.02,rounding_size=1.2", lw=0.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fill, ec=C["edge"], lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=C["ink"],
            fontweight="bold" if bold else "normal", wrap=True, linespacing=1.15)
    return (x, y, w, h)


def arrow(ax, p, q, label=None, fs=7.5, style="-|>", ls="-", rad=0.0, lpos=0.5, loff=(0, 1.2), color=None):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=9, lw=0.9, ls=ls,
                                 color=color or C["edge"], connectionstyle=f"arc3,rad={rad}"))
    if label:
        mx = p[0] + (q[0] - p[0]) * lpos + loff[0]
        my = p[1] + (q[1] - p[1]) * lpos + loff[1]
        ax.text(mx, my, label, ha="center", va="center", fontsize=fs, color="#333",
                bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.9))


def region(ax, x, y, w, h, title, fill="#fafafa"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=1.5",
                                fc=fill, ec="#9a9a9a", lw=0.8, ls="--"))
    ax.text(x + 1.2, y + h - 1.6, title, ha="left", va="top", fontsize=8.5, color="#444", fontstyle="italic")


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)
    print("wrote", OUT / name)


# ---------------------------------------------------------------- architecture
def architecture():
    fig, ax = canvas(10, 7.2)
    region(ax, 1, 60, 98, 11, "Presentation layer (browser)")
    box(ax, 4, 61.5, 28, 7, "Next.js 16 UI\nEvidence Search · Patient Case · Audit", C["ui"], 8.5)
    box(ax, 36, 61.5, 28, 7, "Verified answer, forest plot,\ncertainty card, model trace", C["ui"], 8.5)
    box(ax, 68, 61.5, 28, 7, "Applicability matrix,\nsafety check, references", C["ui"], 8.5)

    region(ax, 1, 13, 59, 45, "Application layer: FastAPI modular monolith (local machine)")
    box(ax, 3.5, 49, 54, 6, "REST API  /search · /case/extract · /case/search · /auth · /traces", C["api"], 8.3, True)
    cols = [3.5, 22, 40.5]
    grid = [["Safety gates\nPHI · scope · entity", "Clinical intent &\nquery normaliser", "Hybrid retriever\ndense ‖ sparse, RRF"],
            ["Rerank + MMR\ncontext assembler", "Population match &\nevidence hierarchy", "Effect map, I², τ²\nGRADE certainty"],
            ["Synthesis\n(structured JSON)", "Claim & section\nverification (NLI)", "Abstention &\ncitation audit"]]
    fills = [[C["gate"], C["pipe"], C["pipe"]], [C["pipe"]] * 3, [C["pipe"], C["gate"], C["gate"]]]
    for r, ys in enumerate((40.5, 31.5, 22.5)):
        for c, x in enumerate(cols):
            box(ax, x, ys, 17, 6.5, grid[r][c], fills[r][c], 8)
    box(ax, 3.5, 14.5, 54, 5.5, "Patient Case Mode: fact extraction → span check → applicability → safety check", C["pipe"], 8.3)

    region(ax, 68, 31, 31, 27, "Local GPU (RTX 3070 Ti, 8 GB)")
    box(ax, 71, 44, 25, 9, "BGE-M3 encoder\n1024-d dense +\nlearned sparse", C["model"], 8.5)
    box(ax, 71, 33.5, 25, 8, "bge-reranker-v2-m3\ncross-encoder", C["model"], 8.5)

    region(ax, 68, 13, 31, 16, "Local storage")
    box(ax, 70, 15, 13, 11, "Qdrant\n412,846\nvectors", C["store"], 8)
    box(ax, 85, 15, 12.5, 11, "SQLite\n19,057 papers\ntraces", C["store"], 8)

    region(ax, 1, 0.5, 98, 10.5, "External services (called only after the PHI gate passes)")
    box(ax, 4, 2, 29, 6.5, "Gemini API (Flash chain)\ngenerator, fact extractor", C["ext"], 8)
    box(ax, 36, 2, 29, 6.5, "Gemini Decision 1\ncalibrated claim verifier", C["ext"], 8)
    box(ax, 68, 2, 29, 6.5, "OpenRouter free models\nfallback chain", C["ext"], 8)

    arrow(ax, (50, 61.5), (32, 55.2), "HTTPS / JSON", style="<|-|>")
    arrow(ax, (60, 46), (68, 46), style="<|-|>")
    ax.text(64, 48.3, "embed,\nrerank", ha="center", fontsize=7.3)
    arrow(ax, (60, 21), (68, 21), style="<|-|>")
    ax.text(64, 23.6, "vectors,\ntraces", ha="center", fontsize=7.3)
    arrow(ax, (18, 13), (18, 8.6), style="<|-|>")
    ax.text(19.2, 11, "prompt", fontsize=7.3, va="center")
    arrow(ax, (50, 13), (50, 8.6), style="<|-|>")
    ax.text(51.2, 11, "verify", fontsize=7.3, va="center")
    arrow(ax, (58, 13), (74, 8.6), ls="--")
    ax.text(68, 11.6, "fallback", fontsize=7.3, va="center")
    save(fig, "fig_architecture.png")


# ---------------------------------------------------------------- pipeline flow
def pipeline():
    fig, ax = canvas(10, 4.6)
    stages = [
        ("1  Gate", "PHI, out-of-\nscope, unknown\nentity", C["gate"]),
        ("2  Understand", "intent, disease,\nbiomarkers,\nsetting", C["pipe"]),
        ("3  Retrieve", "BGE-M3 dense\n+ sparse\nRRF k=60", C["pipe"]),
        ("4  Rank", "cross-encoder,\nMMR, bounded\npriors", C["pipe"]),
        ("5  Appraise", "population fit,\neffect sizes,\nI², GRADE", C["pipe"]),
        ("6  Synthesise", "structured\nclaims citing\nE-IDs only", C["model"]),
        ("7  Verify", "claim & section\nNLI, numbers,\ncitations", C["gate"]),
    ]
    w, gap, y = 12.2, 1.75, 22
    for i, (t, sub, col) in enumerate(stages):
        x = 1 + i * (w + gap)
        box(ax, x, y, w, 16, "", col)
        ax.text(x + w / 2, y + 13.2, t, ha="center", fontsize=9, fontweight="bold")
        ax.text(x + w / 2, y + 6.4, sub, ha="center", va="center", fontsize=7.8, linespacing=1.2)
        if i:
            arrow(ax, (x - gap, y + 8), (x, y + 8))
    # outcomes
    box(ax, 6, 2, 26, 10, "Explicit abstention\nreason + sources for\nmanual review", C["gate"], 8.2)
    box(ax, 54, 2, 30, 10, "Verified brief\nclaims + certainty + forest plot\n+ applicability + model trace", C["api"], 8.2)
    xv = 1 + 6 * (w + gap) + w / 2
    arrow(ax, (xv, y), (74, 12.2), "≥50% entailed,\n≤1/3 contradicted", lpos=0.5, loff=(7, 0))
    arrow(ax, (xv - 4, y), (26, 12.2), "fails verification", lpos=0.45, loff=(0, 1.2))
    arrow(ax, (7, y), (12, 12.2), "blocked", ls="--", lpos=0.5, loff=(-4, 0))
    ax.text(50, 44, "Case → Retrieve → Appraise → Verify → Brief or Abstain", ha="center", fontsize=10.5,
            fontstyle="italic")
    save(fig, "fig_pipeline.png")


# ---------------------------------------------------------------- use cases
def actor(ax, x, y, name):
    ax.add_patch(Ellipse((x, y + 7), 3, 3, fc="white", ec=C["ink"], lw=1))
    ax.plot([x, x], [y + 5.5, y + 1.5], color=C["ink"], lw=1)
    ax.plot([x - 2.2, x + 2.2], [y + 4.3, y + 4.3], color=C["ink"], lw=1)
    ax.plot([x, x - 2], [y + 1.5, y - 1.5], color=C["ink"], lw=1)
    ax.plot([x, x + 2], [y + 1.5, y - 1.5], color=C["ink"], lw=1)
    ax.text(x, y - 3.2, name, ha="center", va="top", fontsize=8.5, linespacing=1.1)


def usecase():
    fig, ax = canvas(10, 7)
    ax.add_patch(FancyBboxPatch((20, 2), 60, 66, boxstyle="round,pad=0.02,rounding_size=1", fc="#fbfbfb",
                                ec=C["edge"], lw=1))
    ax.text(50, 66, "Indence system boundary", ha="center", fontsize=9.5, fontweight="bold")
    ucs = [
        (50, 60, "UC-01 Ask an oncology evidence question"),
        (50, 54, "UC-02 Filter by year / study type"),
        (50, 48, "UC-03 Open a citation to its source passage"),
        (50, 42, "UC-05 Receive a safe abstention"),
        (50, 36, "UC-08/09 Inspect conflicts & forest plot"),
        (50, 30, "UC-10 Paste a de-identified case, confirm facts"),
        (50, 24, "UC-11 Case-aware search + applicability matrix"),
        (50, 18, "UC-12 Patient safety check (DDI, history)"),
        (50, 12, "UC-06 Inspect model trace & audit timings"),
        (50, 6, "UC-07 Check corpus / index health"),
    ]
    for x, y, t in ucs:
        ax.add_patch(Ellipse((x, y), 50, 4.8, fc="#eef4fb", ec=C["edge"], lw=0.8))
        ax.text(x, y, t, ha="center", va="center", fontsize=8.2)
    actor(ax, 8, 46, "Oncologist /\nfellow")
    actor(ax, 8, 16, "Clinical\nresearcher")
    actor(ax, 92, 8, "Administrator")
    actor(ax, 92, 44, "External LLM\n(secondary)")
    for _, y, _ in ucs[:8]:
        arrow(ax, (11, 49), (25, y), style="-")
    for _, y, _ in ucs[:5]:
        arrow(ax, (11, 18), (25, y), style="-", color="#9a9a9a")
    arrow(ax, (11, 18), (25, 12), style="-")
    arrow(ax, (89, 10), (75, 6), style="-")
    arrow(ax, (89, 10), (75, 12), style="-")
    for y in (60, 30, 18):
        arrow(ax, (89, 47), (75, y), style="-", ls="--", color="#9a9a9a")
    save(fig, "fig_usecase.png")


# ---------------------------------------------------------------- DFD
def dfd():
    fig, ax = canvas(10, 8.4)
    ax.text(50, 82, "Level 0 (context diagram)", ha="center", fontsize=10, fontweight="bold")
    box(ax, 3, 66, 20, 9, "Clinician /\nresearcher", C["ui"], 9, style="square,pad=0.02")
    ax.add_patch(Ellipse((50, 70.5), 26, 13, fc=C["pipe"], ec=C["edge"]))
    ax.text(50, 70.5, "0  Indence\nevidence engine", ha="center", va="center", fontsize=9.5, fontweight="bold")
    box(ax, 77, 73, 20, 7, "PMC corpus\n(ingested)", C["store"], 8.5, style="square,pad=0.02")
    box(ax, 77, 62, 20, 7, "LLM providers", C["ext"], 8.5, style="square,pad=0.02")
    arrow(ax, (23, 72.5), (37, 72.5), "question, case text", lpos=0.5, loff=(0, 1.6))
    arrow(ax, (37, 68), (23, 68), "verified brief / abstention", lpos=0.5, loff=(0, -1.8))
    arrow(ax, (77, 76), (63, 73), "papers, chunks")
    arrow(ax, (63, 68), (77, 66.5), "de-identified prompt", loff=(0, 1.8))
    arrow(ax, (77, 63.5), (63, 66), None)

    ax.text(50, 56, "Level 1", ha="center", fontsize=10, fontweight="bold")
    procs = [
        (12, 44, "1.0\nGate &\nnormalise"), (32, 44, "2.0\nHybrid\nretrieve"), (52, 44, "3.0\nRank &\nassemble"),
        (72, 44, "4.0\nAppraise"), (88, 28, "5.0\nSynthesise"), (64, 14, "6.0\nVerify &\nabstain"),
        (36, 14, "7.0\nPersist\ntrace"), (12, 26, "8.0\nCase facts\n& safety"),
    ]
    for x, y, t in procs:
        ax.add_patch(Ellipse((x, y), 15, 10, fc=C["pipe"], ec=C["edge"]))
        ax.text(x, y, t, ha="center", va="center", fontsize=8)
    # data stores (open rectangles)
    for x, y, t in [(26, 30, "D1  Qdrant vectors"), (46, 30, "D2  SQLite papers / chunks"),
                    (14, 2, "D3  query_runs, traces, answers")]:
        ax.plot([x, x + 18], [y + 3, y + 3], color=C["ink"], lw=1)
        ax.plot([x, x + 18], [y - 1, y - 1], color=C["ink"], lw=1)
        ax.plot([x, x], [y - 1, y + 3], color=C["ink"], lw=1)
        ax.text(x + 9.5, y + 1, t, ha="center", va="center", fontsize=7.8)
    arrow(ax, (19.5, 44), (24.5, 44), "query")
    arrow(ax, (39.5, 44), (44.5, 44), "50 cand.")
    arrow(ax, (59.5, 44), (64.5, 44), "bundle")
    arrow(ax, (79, 41), (85, 32.5), "evidence\n+ certainty", lpos=0.4, loff=(5, 0))
    arrow(ax, (84, 24), (71, 16.5), "claims JSON")
    arrow(ax, (56.5, 14), (43.5, 14), "result")
    arrow(ax, (33, 33.2), (32, 39), "vectors", lpos=0.5, loff=(-4, 0))
    arrow(ax, (53, 33.2), (52, 39), "text, metadata", lpos=0.5, loff=(7, 0))
    arrow(ax, (33, 9.5), (25, 5.2), "write")
    arrow(ax, (12, 31), (12, 39), "case query", lpos=0.5, loff=(-5, 0))
    save(fig, "fig_dfd.png")


# ---------------------------------------------------------------- sequence
def sequence():
    fig, ax = canvas(10, 7.4)
    lanes = ["Clinician\n(UI)", "FastAPI\n/search", "QueryPipeline", "BGE-M3 +\nreranker", "Qdrant /\nSQLite",
             "Gemini\nFlash", "Decision 1\nverifier"]
    xs = [6 + i * 14.6 for i in range(len(lanes))]
    for x, t in zip(xs, lanes):
        box(ax, x - 6, 66, 12, 7, t, C["ui"] if x == xs[0] else C["api"], 8)
        ax.plot([x, x], [2, 66], color="#9a9a9a", lw=0.8, ls="--")
    msgs = [
        (0, 1, "POST question + filters"), (1, 2, "run(query, options)"), (2, 2, "PHI / scope gates"),
        (2, 3, "encode dense+sparse"), (2, 4, "dense ‖ sparse search"), (4, 2, "50 candidates (RRF)"),
        (2, 3, "rerank top-50"), (2, 2, "MMR, priors, effect map, GRADE"), (2, 5, "synthesis prompt (E-IDs)"),
        (5, 2, "structured claims"), (2, 6, "verify claims + sections"), (6, 2, "label + confidence"),
        (2, 4, "persist run + trace"), (1, 0, "brief / abstention + model_trace"),
    ]
    y = 62
    for a, b, t in msgs:
        if a == b:
            ax.add_patch(FancyArrowPatch((xs[a], y + 0.6), (xs[a], y - 1.8), arrowstyle="-|>",
                                         connectionstyle="arc3,rad=-1.4", mutation_scale=8, lw=0.8))
            ax.text(xs[a] + 3, y - 0.6, t, fontsize=7.6, va="center")
        else:
            ret = b < a
            arrow(ax, (xs[a], y), (xs[b], y), None, ls="--" if ret else "-")
            ax.text((xs[a] + xs[b]) / 2, y + 1.1, t, ha="center", fontsize=7.6)
        y -= 4.3
    save(fig, "fig_sequence.png")


# ---------------------------------------------------------------- ER
def er():
    fig, ax = canvas(10, 6.6)
    ents = {
        "source_documents": (2, 44, ["id PK", "pmid, pmcid, doi", "title, publication_date",
                                     "retraction_status", "display_rights"]),
        "document_revisions": (27, 44, ["id PK", "document_id FK", "corpus_snapshot_id FK", "study_type",
                                        "content_sha256, parser_version"]),
        "document_sections": (52, 44, ["id PK", "revision_id FK", "section_kind", "source_locator"]),
        "evidence_units": (77, 44, ["id PK", "section_id FK", "start/end_char", "content_type"]),
        "chunks": (77, 14, ["id PK", "evidence_unit_id FK", "text, token_count", "qdrant_point_id",
                            "embedding_model"]),
        "query_runs": (2, 14, ["id PK", "user_id FK", "raw/normalized_text", "filters, latency_ms"]),
        "answers": (27, 14, ["id PK", "query_run_id FK", "status, confidence", "generation_model"]),
        "answer_claims": (52, 14, ["id PK", "answer_id FK", "text, claim_type", "verification_status"]),
    }
    geo = {}
    for name, (x, y, cols) in ents.items():
        h = 3.2 * (len(cols) + 1) + 1
        ax.add_patch(FancyBboxPatch((x, y), 21, h, boxstyle="square,pad=0", fc="white", ec=C["edge"], lw=0.9))
        ax.add_patch(FancyBboxPatch((x, y + h - 4), 21, 4, boxstyle="square,pad=0", fc="#dbe9f6", ec=C["edge"],
                                    lw=0.9))
        ax.text(x + 10.5, y + h - 2, name, ha="center", va="center", fontsize=8.2, fontweight="bold")
        for i, c in enumerate(cols):
            ax.text(x + 1, y + h - 6.2 - i * 3.2, c, fontsize=7.4, va="center")
        geo[name] = (x, y, 21, h)

    def link(a, b, la="1", lb="N", side="h"):
        xa, ya, wa, ha = geo[a]
        xb, yb, wb, hb = geo[b]
        if side == "h":
            p, q = (xa + wa, ya + ha / 2), (xb, yb + hb / 2)
        else:
            p, q = (xa + wa / 2, ya), (xb + wb / 2, yb + hb)
        ax.plot([p[0], q[0]], [p[1], q[1]], color=C["edge"], lw=0.9)
        ax.text(p[0] + (0.8 if side == "h" else 1.2), p[1] + (1 if side == "h" else -1.5), la, fontsize=7.5)
        ax.text(q[0] - (2 if side == "h" else -1.2), q[1] + (1 if side == "h" else 1.2), lb, fontsize=7.5)

    link("source_documents", "document_revisions")
    link("document_revisions", "document_sections")
    link("document_sections", "evidence_units")
    link("evidence_units", "chunks", side="v")
    link("query_runs", "answers")
    link("answers", "answer_claims")
    xa, ya, wa, ha = geo["answer_claims"]
    xb, yb, wb, hb = geo["chunks"]
    ax.plot([xa + wa, xb], [ya + ha / 2, yb + hb / 2], color=C["edge"], lw=0.9, ls="--")
    ax.text((xa + wa + xb) / 2, ya + ha / 2 - 5.5, "citations\n(N:M, span)", ha="center", fontsize=7.4)
    ax.text(50, 4, "Also: users, sessions, retrieval_traces, retrieval_candidates, corpus_snapshots, "
                   "ingestion_runs/jobs, feedback, audit_events", ha="center", fontsize=7.8, fontstyle="italic")
    save(fig, "fig_er.png")


# ---------------------------------------------------------------- modules
def modules():
    fig, ax = canvas(10, 5.6)
    groups = [
        ("ingestion/", ["importer", "manifest_validator", "handlers"], 2, C["store"]),
        ("corpus/", ["chunker"], 2, C["store"]),
        ("retrieval/", ["normalization", "retriever", "context_assembler"], 1, C["pipe"]),
        ("clinical/", ["clinical_intent", "population_matcher", "evidence_hierarchy", "phi_guard",
                       "case_context", "applicability", "safety_check"], 1, C["gate"]),
        ("synthesis/", ["synthesizer", "validator", "abstention", "effect_map", "certainty",
                        "contradiction_checker", "evidence_grouper", "citation_auditor"], 1, C["model"]),
        ("adapters/models/", ["bge_m3", "bge_reranker", "openrouter (LLM chain)", "gemini_decision"], 1,
         C["ext"]),
    ]
    x = 1
    for title, items, _, col in groups:
        h = 6 + 4.6 * len(items)
        top = 44
        ax.add_patch(FancyBboxPatch((x, top - h), 15.2, h, boxstyle="round,pad=0.02,rounding_size=1", fc=col,
                                    ec=C["edge"], lw=0.9))
        ax.text(x + 7.6, top - 2.6, title, ha="center", fontsize=8.6, fontweight="bold")
        for i, it in enumerate(items):
            box(ax, x + 1, top - 8.8 - i * 4.6, 13.2, 3.6, it, "white", 7.2, lw=0.6)
        x += 16.4
    box(ax, 1, 48, 97, 6, "modules/pipeline/query_pipeline.py: the single code path used by app.py (API) and evals/ "
                          "(benchmarks)", C["api"], 8.6, True)
    save(fig, "fig_modules.png")


if __name__ == "__main__":
    architecture()
    pipeline()
    usecase()
    dfd()
    sequence()
    er()
    modules()
