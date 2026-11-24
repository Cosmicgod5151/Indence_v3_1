import logging
from typing import Any, NamedTuple
from ...ports.models import LLMClient
from ..retrieval.context_assembler import AssembledContext
from ..retrieval.retriever import RetrievalCandidate
from ..clinical.clinical_intent import ClinicalQueryIntent

logger = logging.getLogger("evidence_platform.synthesizer")

class SynthesizedAnswer(NamedTuple):
    status: str  # "answer" | "insufficient_evidence"
    direct_claims: list[dict[str, Any]]
    summary_claims: list[dict[str, Any]]
    limitations: list[dict[str, Any]]
    abstention_reason: str | None
    evidence_map: dict[str, RetrievalCandidate]
    # Evidence synthesis sections
    recommendation: dict[str, Any] | None
    clinical_reasoning: dict[str, Any] | None
    supporting_evidence: list[dict[str, Any]]
    alternatives: dict[str, Any] | None
    bottom_line: dict[str, Any] | None
    clinical_context: dict[str, Any] | None
    evidence_certainty: dict[str, Any] | None
    # 12-Step OpenEvidence additions
    clinical_intent: dict[str, Any] | None
    evidence_clusters: list[dict[str, Any]]
    contradiction_analysis: dict[str, Any] | None
    citation_audit: list[dict[str, Any]]
    failure_analysis: list[dict[str, Any]]
    # Provider metadata (model id, latency, token usage, error flag)
    llm_meta: dict[str, Any] | None = None
    llm_error: str | None = None


_EMPTY_SYNTHESIS = dict(
    recommendation=None,
    clinical_reasoning=None,
    supporting_evidence=[],
    alternatives=None,
    bottom_line=None,
    clinical_context=None,
    evidence_certainty=None,
    clinical_intent=None,
    evidence_clusters=[],
    contradiction_analysis=None,
    citation_audit=[],
    failure_analysis=[],
)


class AnswerSynthesizer:
    """Invokes structured LLM generation client with evidence maps and context."""

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    async def synthesize(
        self,
        query: str,
        assembled: AssembledContext,
        intent: ClinicalQueryIntent | None = None,
        clusters: list[Any] | None = None,
        contradiction_info: dict[str, Any] | None = None,
        citation_audit: list[dict[str, Any]] | None = None,
        failure_analysis: list[dict[str, Any]] | None = None,
    ) -> SynthesizedAnswer:
        """Construct the evidence bundle, query the LLM, and map results to original chunks."""
        intent_dict = intent.to_dict() if intent else None
        clusters_dict = [c.to_dict() if hasattr(c, "to_dict") else c for c in (clusters or [])]
        audit_list = citation_audit or []
        failures_list = failure_analysis or []

        if not assembled.selected_candidates:
            return SynthesizedAnswer(
                status="insufficient_evidence",
                direct_claims=[],
                summary_claims=[],
                limitations=[],
                abstention_reason="No relevant clinical trials found matching requested disease/biomarker population in active snapshot.",
                evidence_map={},
                recommendation=None,
                clinical_reasoning=None,
                supporting_evidence=[],
                alternatives=None,
                bottom_line=None,
                clinical_context=None,
                evidence_certainty=None,
                clinical_intent=intent_dict,
                evidence_clusters=clusters_dict,
                contradiction_analysis=contradiction_info,
                citation_audit=audit_list,
                failure_analysis=failures_list,
            )

        # 1. Create a structured evidence bundle mapping reference IDs (E01, E02...) to chunks
        chunks_list = []
        evidence_map = {}
        
        for idx, cand in enumerate(assembled.selected_candidates):
            ref_id = f"E{idx + 1:02d}"
            evidence_map[ref_id] = cand
            
            chunks_list.append({
                "id": cand.chunk_id,
                "text": cand.text,
                "evidence_id": ref_id,
                "pmcid": cand.pmcid,
                "study_type": cand.study_type or "other",
                "hierarchy_label": getattr(cand, "hierarchy_label", cand.study_type or "other"),
                "population_match_score": getattr(cand, "population_match_score", 0),
            })

        evidence_bundle = {
            "snapshot_id": "active_snapshot",
            "clinical_intent": intent_dict,
            "evidence_clusters": clusters_dict,
            "contradiction_analysis": contradiction_info,
            "chunks": chunks_list
        }

        # 2. Invoke the structured LLM client
        logger.info(f"Invoking LLM structured generation for query with {len(chunks_list)} candidates...")
        res = await self.llm_client.generate_answer(query, evidence_bundle)

        status = res.get("status", "answer")
        abstention_reason = res.get("abstention_reason")

        # Extract claims
        direct_claims = res.get("direct_answer_claims", [])
        summary_claims = res.get("evidence_summary_claims", [])
        limitations = res.get("limitations_claims", [])

        # Extract synthesis sections
        recommendation = res.get("recommendation")
        clinical_reasoning = res.get("clinical_reasoning")
        supporting_evidence = res.get("supporting_evidence", [])
        alternatives = res.get("alternatives")
        bottom_line = res.get("bottom_line")
        clinical_context = res.get("clinical_context")
        evidence_certainty = res.get("evidence_certainty")

        return SynthesizedAnswer(
            status=status,
            direct_claims=direct_claims,
            summary_claims=summary_claims,
            limitations=limitations,
            abstention_reason=abstention_reason,
            evidence_map=evidence_map,
            recommendation=recommendation,
            clinical_reasoning=clinical_reasoning,
            supporting_evidence=supporting_evidence,
            alternatives=alternatives,
            bottom_line=bottom_line,
            clinical_context=clinical_context,
            evidence_certainty=evidence_certainty,
            clinical_intent=intent_dict,
            evidence_clusters=clusters_dict,
            contradiction_analysis=contradiction_info,
            citation_audit=audit_list,
            failure_analysis=failures_list,
            llm_meta=res.get("_meta"),
            llm_error=res.get("llm_error"),
        )
