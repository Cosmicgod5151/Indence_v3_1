"""Shared helpers for Indence evaluation scripts.

All evaluation code calls the SAME QueryPipeline the API uses
(backend/src/evidence_platform/modules/pipeline/query_pipeline.py), with real
models (BGE-M3, bge-reranker-v2-m3 on GPU; OpenRouter LLM) and the real corpus.
"""
from __future__ import annotations

import json
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
BACKEND_SRC = ROOT / "backend" / "src"
RESULTS_DIR = ROOT / "evals" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(BACKEND_SRC))

SEED = 20261125
random.seed(SEED)


def make_session():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from evidence_platform.core.config import settings
    engine = create_engine(settings.DATABASE_URL, connect_args={"check_same_thread": False})
    return sessionmaker(bind=engine, autoflush=False)()


_CLIENTS: dict[str, Any] = {}


def clients(llm_model: str | None = None):
    """Real model clients, loaded once per process."""
    from evidence_platform.core.config import settings
    from evidence_platform.adapters.models.factory import get_embedding_client, get_reranker_client, get_entailment_client
    from evidence_platform.adapters.models.openrouter import OpenRouterLLMClient
    from evidence_platform.adapters.qdrant.qdrant_client import QdrantIndexClient
    if "emb" not in _CLIENTS:
        _CLIENTS["emb"] = get_embedding_client()
        _CLIENTS["rer"] = get_reranker_client()
        _CLIENTS["ent"] = get_entailment_client()
        _CLIENTS["qdr"] = QdrantIndexClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
    llm = OpenRouterLLMClient(settings.LLM_API_KEY, llm_model or settings.LLM_MODEL_NAME, timeout_s=settings.LLM_TIMEOUT_S)
    return _CLIENTS["emb"], _CLIENTS["rer"], llm, _CLIENTS["ent"], _CLIENTS["qdr"]


def make_pipeline(db, llm_model: str | None = None):
    from evidence_platform.modules.pipeline.query_pipeline import QueryPipeline
    emb, rer, llm, ent, qdr = clients(llm_model)
    return QueryPipeline(db, emb, rer, llm, ent, qdr)


def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    v = sorted(values)
    k = (len(v) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def summarize_latency(values: list[float]) -> dict[str, float]:
    return {
        "n": len(values),
        "p50": round(pct(values, 0.5), 3),
        "p95": round(pct(values, 0.95), 3),
        "mean": round(statistics.mean(values), 3) if values else float("nan"),
    }


def save_json(name: str, obj: Any) -> Path:
    path = RESULTS_DIR / name
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return path


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")
