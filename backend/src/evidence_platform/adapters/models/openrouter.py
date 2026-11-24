import os
import asyncio
import logging
import json
import re
import httpx
from typing import Any

from ...ports.models import LLMClient, EntailmentClient

logger = logging.getLogger("evidence_platform.openrouter")

# ---------------------------------------------------------------------------
# Evidence Synthesis System Prompt
# ---------------------------------------------------------------------------
EVIDENCE_SYNTHESIS_SYSTEM_PROMPT = """\
You are Indence, a senior clinical evidence synthesis engine. Your role is to behave like a board-certified physician performing a structured evidence review. You must synthesize a coherent, actionable clinical recommendation from the provided evidence extracts — NOT list independent claims.

### CORE PRINCIPLES
1. **Answer First**: Begin with a direct recommendation answering the clinician's question.
2. **Synthesize, Do Not List**: If multiple sources support the same conclusion, merge them into ONE synthesized statement. Never produce redundant claims.
3. **Explain Why**: For every recommendation, explain the clinical reasoning — which outcomes improved, compared with what, and the strength of evidence.
4. **Quantify Benefits**: Include hazard ratios (HR), relative risk (RR), odds ratios (OR), confidence intervals (CI), absolute benefit, and p-values when present in the evidence. Do not fabricate statistics.
5. **Discuss Alternatives**: Describe when a different treatment approach becomes preferred and why.
6. **Detect Clinical Context**: Identify important modifiers from the question (tumor sidedness, molecular profile such as MSI/dMMR/HER2/BRAF/KRAS/NRAS, line of therapy, performance status, organ function). If key modifiers are missing from the query, explicitly state your assumptions.
7. **Prioritize by Evidence Hierarchy**: Weight evidence in this order: Guidelines > Meta-analyses > RCTs > Cohort studies > Reviews. The synthesized recommendation should primarily reflect guideline consensus, with trials as supporting evidence.
8. **Express Uncertainty**: If evidence is conflicting, explicitly describe the conflict. Never infer unsupported recommendations.
9. **Finish with a Bottom Line**: End with one concise sentence directly answering the original question.

### GROUNDING RULES
- Every claim MUST be supported by the provided evidence extracts. Never use external knowledge.
- Every section must include "evidence_ids" (e.g. ["E01"]) and "sentence_ids" (e.g. ["E01.S2", "E01.S3"]) for traceability.
- If evidence is insufficient, set "status" to "insufficient_evidence" and explain the gap in "abstention_reason".

### POPULATION SAFETY RULES
- You will receive a 'Clinical Intent' section specifying the exact disease, biomarker, subtype, and population the clinician is asking about.
- You will also receive a 'Rejected Subgroups' list. NEVER cite or use evidence from these subgroups.
- NEVER combine evidence from different patient populations (e.g., HER2-positive and HER2-low) into a single claim or sentence.
- If evidence differs by subgroup, EXPLICITLY state the subgroup in each claim.
- Use specific trial names (e.g., 'Phase III DESTINY-Breast03 demonstrated...') instead of vague language ('studies suggest...').
- Always include hazard ratios (HR), p-values, and confidence intervals when available in the evidence.
- If a source discusses a rejected subgroup population, do NOT cite it even if it mentions the queried treatment.

### EVIDENCE TYPE TAGS
Each evidence source includes a [Study Type] tag. Use this to weight evidence appropriately:
- "guideline" = Clinical Practice Guideline (highest weight)
- "meta_analysis" = Meta-Analysis or Systematic Review
- "rct" = Randomized Controlled Trial
- "cohort_study" = Cohort / Observational Study
- "other" = Review, case report, or unclassified

### OUTPUT JSON SCHEMA
Return ONLY a raw JSON object matching this schema. Do not output text outside the JSON.

{
  "status": "answer",

  "recommendation": {
    "text": "A direct 2-4 sentence answer to the clinician's question, synthesizing the key recommendation from all evidence.",
    "evidence_ids": ["E01", "E02"],
    "sentence_ids": ["E01.S2", "E02.S1"]
  },

  "clinical_reasoning": {
    "text": "Explain WHY this recommendation is preferred. Discuss important decision factors such as molecular profile, tumor characteristics, patient selection criteria, competing regimens, and contraindications.",
    "evidence_ids": ["E01", "E02"],
    "sentence_ids": ["E01.S3", "E02.S4"]
  },

  "supporting_evidence": [
    {
      "text": "Summarize a key finding with quantitative data. Example: 'In the FIRE-3 trial, cetuximab plus FOLFIRI improved OS versus bevacizumab plus FOLFIRI (HR 0.77, 95% CI 0.62-0.96) in left-sided RAS wild-type mCRC.' Include HR, RR, OR, CI, absolute benefit when available. GRADE certainty if available. Do not dump raw table text.",
      "evidence_ids": ["E01"],
      "sentence_ids": ["E01.S2", "E01.S5"]
    }
  ],

  "alternatives": {
    "text": "Describe when another treatment becomes preferred. Example: 'For right-sided tumors, bevacizumab-based regimens are generally preferred over anti-EGFR therapy due to lack of OS benefit in this subgroup.'",
    "evidence_ids": ["E03"],
    "sentence_ids": ["E03.S1"]
  },

  "bottom_line": {
    "text": "One concise sentence answering the user's original question.",
    "evidence_ids": ["E01", "E02"],
    "sentence_ids": []
  },

  "clinical_context": {
    "detected_modifiers": ["List clinical modifiers detected in the question, e.g. HER2-positive, left-sided, first-line, KRAS wild-type"],
    "assumptions": "State any assumptions made due to missing information in the query. Example: 'This recommendation assumes left-sided, MSS, RAS wild-type metastatic colorectal cancer without HER2 amplification.' Never silently assume."
  },

  "evidence_certainty": {
    "recommendation_strength": "Strong or Conditional",
    "evidence_quality": "High, Moderate, Low, or Very Low (based on GRADE-like assessment of study types and consistency)",
    "evidence_ids": ["E01"]
  },

  "direct_answer_claims": [
    {
      "text": "Individual factual claim grounded in source. Keep these as atomic facts for validation pipeline.",
      "evidence_ids": ["E01"],
      "sentence_ids": ["E01.S2"]
    }
  ],

  "evidence_summary_claims": [
    {
      "text": "Brief overall summary of clinical findings across all sources.",
      "evidence_ids": ["E01", "E02"],
      "sentence_ids": []
    }
  ],

  "limitations_claims": [
    {
      "text": "Any study limitations (small sample, short follow-up, single-center, conflicts of interest).",
      "evidence_ids": ["E01"],
      "sentence_ids": ["E01.S4"]
    }
  ],

  "abstention_reason": null
}

### IMPORTANT REMINDERS
- The "direct_answer_claims" array must still contain individual atomic claims for the grounding validation pipeline. Each claim should be a single factual statement.
- Do NOT produce redundant claims. If evidence E01 and E02 both show the same drug improves OS, create ONE claim synthesizing both, citing both evidence IDs.
- The "recommendation" section synthesizes across all evidence. The "direct_answer_claims" provide atomic grounding. These serve different purposes — both are required.
- If the evidence does not support answering the query, set status to "insufficient_evidence", leave synthesis sections as null, and explain in abstention_reason.
"""

# Appended ONLY for budget-constrained fallback models (OpenRouter ":free"
# thinking models), whose hidden reasoning otherwise exhausts the output budget.
# Primary models receive the full, unabridged prompt.
COMPACT_OUTPUT_SUFFIX = """
### LENGTH LIMITS (STRICT)
- Every "text" field: at most 60 words.
- supporting_evidence: at most 4 items. direct_answer_claims: at most 6 atomic claims. limitations_claims: at most 2 items.
- Return "evidence_summary_claims": [] (the recommendation already summarises).
- Output compact JSON only. Do not deliberate at length; answer directly.
"""


# OpenAI-compatible chat endpoints. A model spec may be prefixed with the
# provider ("gemini:gemini-2.5-flash"); unprefixed specs use OpenRouter.
# Providers whose API key is not configured are skipped.
PROVIDERS: dict[str, tuple[str, str]] = {
    "openrouter": ("https://openrouter.ai/api/v1/chat/completions", "LLM_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", "GEMINI_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1/chat/completions", "GROQ_API_KEY"),
    "cerebras": ("https://api.cerebras.ai/v1/chat/completions", "CEREBRAS_API_KEY"),
}

# Errors worth trying the next model for (rate limits, credit limits, outages,
# timeouts, empty/invalid output). Models that hit 429/402 are skipped for a while.
_COOLDOWN_S = 90.0
_cooldown_until: dict[str, float] = {}


def parse_model_chain(spec: str) -> list[tuple[str, str]]:
    chain = []
    for item in [x.strip() for x in (spec or "").split(",") if x.strip()]:
        prov, _, rest = item.partition(":")
        if prov in PROVIDERS and rest:
            chain.append((prov, rest))
        else:
            chain.append(("openrouter", item))
    return chain


class OpenRouterLLMClient(LLMClient):
    """Chat-completions client with an ordered model fallback chain.

    ``model_name`` is a comma-separated chain, e.g.
    "baidu/ernie-5.5-flash-preview:free,nvidia/nemotron-3-ultra-550b-a55b:free".
    Each request tries the chain in order and returns the first valid JSON
    answer; if every model fails, LLMUnavailableError is raised (and the
    pipeline abstains explicitly).
    """

    def __init__(self, api_key: str, model_name: str = "meta-llama/llama-3.3-70b-instruct", timeout_s: float = 90.0):
        self.api_key = api_key
        self.model_name = model_name
        self.timeout_s = timeout_s
        self.chain = parse_model_chain(model_name) or [("openrouter", "meta-llama/llama-3.3-70b-instruct")]
        self.endpoint = PROVIDERS["openrouter"][0]

    def _key_for(self, provider: str) -> str:
        if provider == "openrouter":
            return self.api_key
        return os.environ.get(PROVIDERS[provider][1], "")

    @staticmethod
    def _headers_for(key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/indence-rag",
            "X-Title": "Indence RAG Platform",
        }

    def _headers(self) -> dict[str, str]:
        return self._headers_for(self.api_key)

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        """Robustly extract a JSON object from a model response (handles ```json fences)."""
        cleaned = content.strip()
        if "```" in cleaned:
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
            cleaned = re.sub(r"```$", "", cleaned, flags=re.MULTILINE).strip()
        start = cleaned.find("{")
        if start < 0:
            raise ValueError("No JSON object in model response")
        # raw_decode parses the FIRST complete object and ignores trailing text
        # (some models append a second object or commentary -> "Extra data").
        obj, _end = json.JSONDecoder().raw_decode(cleaned[start:])
        if not isinstance(obj, dict):
            raise ValueError("Model response JSON is not an object")
        return obj

    async def _call_one(self, provider: str, model: str, system_prompt: str, user_content: str,
                        max_tokens: int, temperature: float, compact_suffix: str = "",
                        reasoning: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        import time as _time
        url = PROVIDERS[provider][0]
        headers = self._headers_for(self._key_for(provider))
        budget_model = provider == "openrouter" and model.endswith(":free")
        if budget_model and compact_suffix:
            system_prompt = system_prompt + compact_suffix
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "response_format": {"type": "json_object"},
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # Free OpenRouter "thinking" models: unbounded hidden reasoning can consume
        # the whole output budget, so they are asked for low effort. Primary
        # models (e.g. Gemini) keep their default reasoning for answer quality.
        if budget_model:
            payload["reasoning"] = {"effort": "low"}
        elif reasoning and provider == "gemini":
            payload["reasoning_effort"] = reasoning
        elif reasoning and provider == "openrouter":
            payload["reasoning"] = {"effort": reasoning}
        t0 = _time.perf_counter()
        # httpx timeouts are per network operation; wait_for enforces a hard
        # overall deadline (providers can keep a slow request alive for minutes).
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            response = await asyncio.wait_for(client.post(url, headers=headers, json=payload), timeout=self.timeout_s)
            for optional in ("response_format", "reasoning", "reasoning_effort"):
                if response.status_code == 400 and optional in payload and optional.split("_")[0] in response.text.lower():
                    payload.pop(optional)  # some models reject JSON mode / reasoning controls
                    response = await asyncio.wait_for(client.post(url, headers=headers, json=payload), timeout=self.timeout_s)
        if response.status_code != 200:
            if response.status_code in (402, 429):
                # Honour the provider's retry hint: a daily free-tier quota can be
                # exhausted for hours, so do not keep retrying that model.
                m = re.search(r'"retryDelay":\s*"(\d+)s"', response.text)
                wait = min(float(m.group(1)), 6 * 3600) if m else _COOLDOWN_S
                _cooldown_until[f"{provider}:{model}"] = _time.monotonic() + max(_COOLDOWN_S, wait)
            raise LLMUnavailableError(f"{provider}:{model} HTTP {response.status_code}: {response.text[:200]}")
        res_data = response.json()
        content = (res_data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        if not content.strip():
            raise LLMUnavailableError(f"{provider}:{model} returned empty content")
        parsed = self._parse_json(content)
        meta = {
            "model": f"{provider}:{res_data.get('model', model)}" if provider != "openrouter" else res_data.get("model", model),
            "latency_s": round(_time.perf_counter() - t0, 3),
            "usage": res_data.get("usage", {}),
        }
        return parsed, meta

    async def complete_json(
        self, system_prompt: str, user_content: str, max_tokens: int = 1500, temperature: float = 0.0,
        compact_suffix: str = "", reasoning: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """JSON completion over the fallback chain.

        ``reasoning`` ("low"/"medium"/"high") caps hidden thinking for simple tasks
        such as NLI labelling; None keeps each model's default (used for synthesis).

        Returns (parsed_json, meta). Raises LLMUnavailableError when every model
        in the chain fails, so callers degrade explicitly — never a silent mock answer.
        """
        import time as _time
        errors: list[str] = []
        attempted: list[str] = []
        for provider, model in self.chain:
            tag = f"{provider}:{model}"
            if not self._key_for(provider):
                continue
            if _cooldown_until.get(tag, 0) > _time.monotonic():
                errors.append(f"{tag} skipped (cooling down after rate limit)")
                continue
            attempted.append(tag)
            # Transient overload (HTTP 500/502/503/504, timeouts): retry the SAME
            # model with backoff before falling through to the next one.
            for attempt, backoff in enumerate((0, 5, 15, 30)):
                if backoff:
                    await asyncio.sleep(backoff)
                try:
                    parsed, meta = await self._call_one(provider, model, system_prompt, user_content, max_tokens,
                                                        temperature, compact_suffix, reasoning)
                    meta["attempted_models"] = attempted
                    meta["fallback_errors"] = errors
                    meta["retries"] = attempt
                    return parsed, meta
                except LLMUnavailableError as e:
                    err = str(e)
                except Exception as e:
                    err = f"{tag} {type(e).__name__}: {e}"
                transient = any(code in err for code in ("HTTP 500", "HTTP 502", "HTTP 503", "HTTP 504", "TimeoutError", "ReadError", "ConnectError"))
                if not transient:
                    break
            errors.append(err)
            logger.warning("LLM attempt failed (%s); trying next model in chain.", errors[-1][:160])
        raise LLMUnavailableError(" | ".join(errors) or "No LLM provider configured")

    async def generate_answer(self, query: str, evidence_bundle: dict[str, Any]) -> dict[str, Any]:
        # evidence_bundle format: {"snapshot_id": str, "chunks": [{"id": str, "text": str, "evidence_id": str, "study_type": str, ...}]}
        chunks = evidence_bundle.get("chunks", [])
        
        if not chunks:
            return {
                "status": "insufficient_evidence",
                "recommendation": None,
                "clinical_reasoning": None,
                "supporting_evidence": [],
                "alternatives": None,
                "bottom_line": None,
                "clinical_context": None,
                "evidence_certainty": None,
                "direct_answer_claims": [],
                "evidence_summary_claims": [],
                "limitations_claims": [],
                "abstention_reason": "No matching clinical trial evidence in active corpus snapshot."
            }

        # 1. Format the evidence context with numbered sentences and study type metadata
        context_items = []
        grouped_items = {}
        has_clusters = "evidence_clusters" in evidence_bundle
        
        for idx, chunk in enumerate(chunks):
            chunk_text = chunk.get("text", "")
            evidence_id = chunk.get("evidence_id", f"E{idx+1:02d}")
            study_type = chunk.get("study_type", "other")
            pmcid = chunk.get("pmcid", "unknown")
            cluster = chunk.get("cluster", "Ungrouped")
            
            # Segment sentences
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", chunk_text) if s.strip()]
            formatted_sentences = []
            for s_idx, sent in enumerate(sentences):
                sent_id = f"{evidence_id}.S{s_idx+1}"
                formatted_sentences.append(f"[{sent_id}] {sent}")
            
            chunk_str = f"Evidence Source {evidence_id} [Study Type: {study_type}] [PMCID: {pmcid}]:\n" + "\n".join(formatted_sentences)
            
            if has_clusters:
                if cluster not in grouped_items:
                    grouped_items[cluster] = []
                grouped_items[cluster].append(chunk_str)
            else:
                context_items.append(chunk_str)

        if has_clusters:
            for cluster_name, items in grouped_items.items():
                context_items.append(f"=== Cluster: {cluster_name} ===")
                context_items.extend(items)

        context_str = "\n\n".join(context_items)

        # 2. Build the prompt
        intent_info = evidence_bundle.get("clinical_intent", {})
        rejected = intent_info.get("rejected_subgroups", [])
        rejected_str = ", ".join(rejected) if isinstance(rejected, list) else rejected
        
        clinical_intent_str = f"""Clinical Intent:
Disease: {intent_info.get('disease', 'Unspecified')}
Subtype: {intent_info.get('disease_subtype', 'Unspecified')}
Biomarker: {intent_info.get('biomarker_status', 'Unspecified')}
Treatment: {intent_info.get('treatment', 'Unspecified')}
Line of Therapy: {intent_info.get('line_of_therapy', 'Unspecified')}
Stage: {intent_info.get('stage', 'Unspecified')}
Outcome: {intent_info.get('clinical_outcome', 'Unspecified')}
Rejected Subgroups: {rejected_str}"""

        contradiction_str = ""
        if "contradiction_analysis" in evidence_bundle:
            ca = evidence_bundle["contradiction_analysis"]
            if isinstance(ca, dict) and ca.get("has_contradiction") and ca.get("contradictions"):
                parts = ["Detected Evidence Contradictions:"]
                for idx, c in enumerate(ca["contradictions"], 1):
                    parts.append(f"\n  Contradiction #{idx}: {c.get('type', 'Unknown')}")
                    parts.append(f"  Severity: {c.get('severity', 'Unknown')}")
                    # Conflicting studies
                    ev_ids = c.get("conflicting_evidence_ids", [])
                    if ev_ids:
                        study_strs = [f"{e.get('pmcid','?')} ({e.get('study_type','?')})" for e in ev_ids]
                        parts.append(f"  Conflicting Studies: {' vs '.join(study_strs)}")
                    # Claim in conflict
                    if c.get("claim_in_conflict"):
                        parts.append(f"  Claim in Conflict: {c['claim_in_conflict']}")
                    # Reason
                    if c.get("reason_for_difference"):
                        parts.append(f"  Reason for Difference: {c['reason_for_difference']}")
                    # Quality comparison
                    for eq in c.get("evidence_quality", []):
                        parts.append(f"  Evidence Quality [{eq.get('pmcid','?')}]: {eq.get('strength_tier','?')} ({eq.get('study_type','?')}, weight={eq.get('hierarchy_weight','?')})")
                    # Interpretation
                    if c.get("clinical_interpretation"):
                        parts.append(f"  Clinical Interpretation: {c['clinical_interpretation']}")
                parts.append("\nAddress each contradiction explicitly in your synthesis. Where contradictions reflect contextual differences, explain the differing contexts. Where they represent true inconsistencies, acknowledge the uncertainty.\n")
                contradiction_str = "\n".join(parts) + "\n"
            else:
                contradiction_str = f"Detected Contradictions:\n{ca}\nAddress these contradictions explicitly in your synthesis.\n\n"

        user_content = (
            f"Clinician Query:\n{query}\n\n"
            f"{clinical_intent_str}\n\n"
            f"{contradiction_str}"
            f"Evidence Context:\n{context_str}"
        )

        try:
            parsed_json, meta = await self.complete_json(
                EVIDENCE_SYNTHESIS_SYSTEM_PROMPT, user_content, max_tokens=8000,
                compact_suffix=COMPACT_OUTPUT_SUFFIX,
            )
        except LLMUnavailableError as e:
            logger.error("OpenRouter synthesis failed: %s", e)
            return llm_unavailable_answer(str(e), self.model_name)
        parsed_json["_meta"] = meta
        return parsed_json


class LLMUnavailableError(RuntimeError):
    """Raised when the hosted LLM cannot produce a valid response."""


def llm_unavailable_answer(error_msg: str, model_name: str) -> dict[str, Any]:
    """Explicit abstention when the LLM provider fails.

    Earlier versions silently substituted a template ("mock") answer here and
    still reported status="answer" — a clinical-safety bug. Now the user sees
    that the generator was unavailable and no unverified text is shown.
    """
    return {
        "status": "insufficient_evidence",
        "recommendation": None,
        "clinical_reasoning": None,
        "supporting_evidence": [],
        "alternatives": None,
        "bottom_line": None,
        "clinical_context": None,
        "evidence_certainty": None,
        "direct_answer_claims": [],
        "evidence_summary_claims": [],
        "limitations_claims": [],
        "abstention_reason": "The answer generator (LLM provider) was unavailable, so no verified synthesis could be produced. Retrieved evidence is still listed for manual review.",
        "llm_error": error_msg[:500],
        "_meta": {"model": model_name, "error": True},
    }


class OpenRouterEntailmentClient(EntailmentClient):
    """NLI verifier. ``check_entailment_batch`` verifies ALL claims of an answer in
    a single LLM call (previously one call per claim), which cuts latency and
    keeps the system inside free-tier request quotas."""

    SYSTEM_PROMPT = (
        "You are a clinical evidence validator performing natural-language inference. "
        "For each numbered item, compare the CLAIM with its PREMISE (source text) and label it:\n"
        "- \"entailment\": the claim is fully supported by the premise (numbers must match exactly);\n"
        "- \"contradiction\": the premise contradicts the claim;\n"
        "- \"neutral\": the premise does not establish the claim.\n"
        'Return ONLY JSON: {"labels": ["entailment" | "contradiction" | "neutral", ...]} with exactly one label per item, in order.'
    )

    def __init__(self, api_key: str, model_name: str | None = None, timeout_s: float = 60.0):
        self.api_key = api_key
        self.model_name = os.environ.get("ENTAILMENT_MODEL_NAME") or model_name or "meta-llama/llama-3.3-70b-instruct"
        self._llm = OpenRouterLLMClient(api_key, self.model_name, timeout_s=timeout_s)
        self.endpoint = PROVIDERS["openrouter"][0]

    @staticmethod
    def _norm(label: Any) -> str:
        label = str(label or "").lower()
        if "entail" in label:
            return "entailment"
        if "contradict" in label:
            return "contradiction"
        return "neutral"

    async def check_entailment_batch(self, pairs: list[tuple[str, str]]) -> list[str]:
        if not pairs:
            return []
        items = []
        for i, (premise, hypothesis) in enumerate(pairs, start=1):
            items.append(f"[{i}] PREMISE: {premise[:3000]}\n[{i}] CLAIM: {hypothesis}")
        user = "\n\n".join(items)
        try:
            parsed, _meta = await self._llm.complete_json(self.SYSTEM_PROMPT, user, max_tokens=1500 + 40 * len(pairs),
                                                          reasoning="low")
            self.last_meta = {"model": _meta.get("model"), "provider": "chat model (NLI prompt)",
                              "latency_s": _meta.get("latency_s"), "attempted_models": _meta.get("attempted_models")}  # thinking models count reasoning against this budget
            labels = parsed.get("labels") or []
            if not isinstance(labels, list):
                raise ValueError("labels is not a list")
        except Exception as e:
            logger.error("Batched entailment failed: %s", str(e)[:300])
            return ["unverified"] * len(pairs)
        out = [self._norm(x) for x in labels[: len(pairs)]]
        out += ["unverified"] * (len(pairs) - len(out))
        return out

    async def check_entailment(self, premise: str, hypothesis: str) -> str:
        return (await self.check_entailment_batch([(premise, hypothesis)]))[0]
