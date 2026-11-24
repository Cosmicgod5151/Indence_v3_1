"""Claim verification with Google's Gemini Decisions API (model: gemini-decision-1).

The Decisions API answers closed-set questions (yes/no, multiple choice, rubric
score) and returns a calibrated probability per option instead of generated
text. Indence uses it as the NLI verifier: every generated claim becomes one
multiple-choice question {supported, contradicted, not_established} judged
against the exact evidence passages the claim cites. All claims of an answer
are verified in ONE request (passages are shared input, de-duplicated).

Why: calibrated per-claim confidence, ~10k free requests/day (vs ~20/day for
chat Flash models), sub-second latency, and no competition with the answer
generator for quota. If the API fails, the chat-model verifier is used.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx

from ...ports.models import EntailmentClient

logger = logging.getLogger("evidence_platform.gemini_decision")

ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/decisions"
LABEL_MAP = {"supported": "entailment", "contradicted": "contradiction", "not_established": "neutral"}
CRITERIA = {
    "supported": "The cited passage(s) explicitly state the claim; every number, direction and population matches.",
    "contradicted": "The cited passage(s) state something incompatible with the claim (different number, opposite "
                    "direction of effect, or a different population).",
    "not_established": "The cited passage(s) do not contain enough information to confirm or refute the claim.",
}


class GeminiDecisionEntailmentClient(EntailmentClient):
    def __init__(self, api_key: str | None = None, fallback: EntailmentClient | None = None,
                 model: str = "gemini-decision-1", timeout_s: float = 30.0):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.fallback = fallback
        self.model = model
        self.timeout_s = timeout_s
        self.model_name = model
        self.last_meta: dict[str, Any] = {}

    async def check_entailment_batch_detailed(self, pairs: list[tuple[str, str]]) -> list[dict[str, Any]]:
        """Return one {label, confidence, probabilities} dict per (premise, claim) pair."""
        if not pairs:
            return []
        t0 = time.perf_counter()
        if self.api_key:
            passages: list[str] = []
            for premise, _ in pairs:
                if premise not in passages:
                    passages.append(premise)
            shared = "\n\n".join(f"[PASSAGE P{i + 1}]\n{p[:4000]}" for i, p in enumerate(passages))
            questions = {}
            for i, (premise, claim) in enumerate(pairs):
                pid = passages.index(premise) + 1
                questions[f"claim_{i}"] = {
                    "type": "choice",
                    "instructions": f"Judge the CLAIM using ONLY [PASSAGE P{pid}]. CLAIM: {claim}",
                    "criteria": CRITERIA,
                }
            try:
                async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                    r = await client.post(ENDPOINT, headers={"x-goog-api-key": self.api_key,
                                                             "Content-Type": "application/json"},
                                          json={"model": self.model, "input": shared, "questions": questions})
                if r.status_code != 200:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                answers = r.json().get("answers", {})
                out = []
                for i in range(len(pairs)):
                    a = answers.get(f"claim_{i}", {})
                    out.append({
                        "label": LABEL_MAP.get(a.get("choice"), "unverified"),
                        "confidence": round(float(a.get("confidence", 0.0)), 4),
                        "probabilities": {LABEL_MAP.get(k, k): round(float(v), 4) for k, v in (a.get("probabilities") or {}).items()},
                    })
                self.last_meta = {"model": self.model, "provider": "Gemini Decisions API",
                                  "latency_s": round(time.perf_counter() - t0, 3), "claims": len(pairs),
                                  "fallback_used": False}
                return out
            except Exception as e:
                logger.warning("Gemini Decisions API failed (%s); using fallback verifier.", str(e)[:200])
                err = str(e)[:200]
        else:
            err = "GEMINI_API_KEY not set"
        if self.fallback is None:
            self.last_meta = {"model": None, "error": err}
            return [{"label": "unverified", "confidence": 0.0, "probabilities": {}} for _ in pairs]
        labels = await self.fallback.check_entailment_batch(pairs)
        fb_meta = getattr(self.fallback, "last_meta", {}) or {}
        self.last_meta = {"model": fb_meta.get("model", getattr(self.fallback, "model_name", "fallback")),
                          "provider": "chat-model fallback", "latency_s": round(time.perf_counter() - t0, 3),
                          "claims": len(pairs), "fallback_used": True, "decision_api_error": err}
        return [{"label": lb, "confidence": None, "probabilities": {}} for lb in labels]

    async def check_entailment_batch(self, pairs: list[tuple[str, str]]) -> list[str]:
        return [d["label"] for d in await self.check_entailment_batch_detailed(pairs)]

    async def check_entailment(self, premise: str, hypothesis: str) -> str:
        return (await self.check_entailment_batch([(premise, hypothesis)]))[0]
