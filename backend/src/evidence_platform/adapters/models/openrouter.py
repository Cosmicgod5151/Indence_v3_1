import os
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


class OpenRouterLLMClient(LLMClient):
    """LLM client using OpenRouter API with evidence synthesis prompt."""

    def __init__(self, api_key: str, model_name: str = "nvidia/nemotron-3-super-120b-a12b"):
        self.api_key = api_key
        self.model_name = model_name
        self.endpoint = "https://openrouter.ai/api/v1/chat/completions"

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

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/indence-rag",
            "X-Title": "Indence RAG Platform"
        }

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": EVIDENCE_SYNTHESIS_SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
            "max_tokens": 4096
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(self.endpoint, headers=headers, json=payload)
                if response.status_code != 200:
                    logger.error(f"OpenRouter returned status {response.status_code}: {response.text}")
                    return await self._fallback_answer(query, chunks, f"OpenRouter API error: {response.text}")
                
                res_data = response.json()
                content = res_data["choices"][0]["message"]["content"].strip()
                
                # Robustly extract JSON object from response (handling markdown ```json blocks)
                cleaned = content
                if "```" in cleaned:
                    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
                    cleaned = re.sub(r"```$", "", cleaned, flags=re.MULTILINE).strip()
                
                match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
                if match:
                    cleaned = match.group(1)
                
                parsed_json = json.loads(cleaned)
                return parsed_json
        except Exception as e:
            logger.error(f"Failed to generate answer via OpenRouter: {e}")
            return await self._fallback_answer(query, chunks, f"Error calling OpenRouter: {str(e)}")

    async def _fallback_answer(self, query: str, chunks: list[dict], error_msg: str) -> dict[str, Any]:
        """Gracefully fall back to local structured evidence synthesis if LLM API is unreachable."""
        from .mock import MockLLMClient
        logger.warning(f"OpenRouter API call failed ({error_msg}). Falling back to local evidence synthesis.")
        mock = MockLLMClient()
        return await mock.generate_answer(query, {"chunks": chunks})


class OpenRouterEntailmentClient(EntailmentClient):
    """Entailment checker using OpenRouter API with Nvidia Nemotron-3 Super."""

    def __init__(self, api_key: str, model_name: str | None = None):
        self.api_key = api_key
        # Use fast, low-latency classification model for NLI entailment unless explicitly specified
        self.model_name = model_name if model_name and "nemotron" not in model_name else os.environ.get("ENTAILMENT_MODEL_NAME", "meta-llama/llama-3.3-70b-instruct")
        self.endpoint = "https://openrouter.ai/api/v1/chat/completions"

    async def check_entailment(self, premise: str, hypothesis: str) -> str:
        system_prompt = (
            "You are a clinical evidence validator. Compare the premise and hypothesis and output exactly one label: 'entailment', 'contradiction', or 'neutral'.\n"
            "Rules:\n"
            "- Output 'entailment' if the hypothesis is fully supported and justified by the premise.\n"
            "- Output 'contradiction' if the hypothesis directly contradicts or denies the premise.\n"
            "- Output 'neutral' if the premise does not provide enough information to confirm or deny the hypothesis.\n"
            "Return ONLY the single word label, lowercase, with no punctuation or other text."
        )

        user_content = (
            f"Premise (Source Text):\n{premise}\n\n"
            f"Hypothesis (Claim):\n{hypothesis}"
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/indence-rag",
            "X-Title": "Indence RAG Platform"
        }

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.0,
            "max_tokens": 100
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(self.endpoint, headers=headers, json=payload)
                if response.status_code != 200:
                    logger.error(f"OpenRouter entailment returned status {response.status_code}: {response.text}")
                    return "neutral"
                
                res_data = response.json()
                label = res_data["choices"][0]["message"]["content"].strip().lower()
                
                # Sanitize response to match exact strings
                if "entailment" in label:
                    return "entailment"
                elif "contradiction" in label:
                    return "contradiction"
                else:
                    return "neutral"
        except Exception as e:
            logger.error(f"Failed to check entailment via OpenRouter: {e}")
            return "neutral"
