from typing import Any, Protocol, Union

class EmbeddingClient(Protocol):
    """Protocol for generating dense and sparse embeddings from text."""

    async def embed_query(self, query: str) -> tuple[list[float], dict[str, Any]]:
        """Embed a single query text.
        
        Returns:
            Tuple of (dense_vector, sparse_vector)
            where sparse_vector is a dict with {"indices": list[int], "values": list[float]}
        """
        ...

    async def embed_documents(self, documents: list[str]) -> list[tuple[list[float], dict[str, Any]]]:
        """Embed a batch of document texts.
        
        Returns:
            List of (dense_vector, sparse_vector) tuples
        """
        ...


class RerankerClient(Protocol):
    """Protocol for reranking candidate passages relative to a query."""

    async def rerank(self, query: str, passages: list[str]) -> list[float]:
        """Compute relevance scores for a list of passages against a query.
        
        Returns:
            List of float scores corresponding to the input passages.
        """
        ...


class LLMClient(Protocol):
    """Protocol for generating structured, evidence-grounded answers with clinical synthesis."""

    async def generate_answer(self, query: str, evidence_bundle: dict[str, Any]) -> dict[str, Any]:
        """Generate a structured, synthesized answer from a query and a closed evidence bundle.

        Returns:
            A dict conforming to the structured output contract:
            {
                "status": "answer | insufficient_evidence",

                # Evidence synthesis sections (each with span-level grounding)
                "recommendation": {"text": str, "evidence_ids": list[str], "sentence_ids": list[str]} | None,
                "clinical_reasoning": {"text": str, "evidence_ids": list[str], "sentence_ids": list[str]} | None,
                "supporting_evidence": [{"text": str, "evidence_ids": list[str], "sentence_ids": list[str]}],
                "alternatives": {"text": str, "evidence_ids": list[str], "sentence_ids": list[str]} | None,
                "bottom_line": {"text": str, "evidence_ids": list[str], "sentence_ids": list[str]} | None,
                "clinical_context": {"detected_modifiers": list[str], "assumptions": str} | None,
                "evidence_certainty": {
                    "recommendation_strength": "Strong | Conditional",
                    "evidence_quality": "High | Moderate | Low | Very Low",
                    "evidence_ids": list[str]
                } | None,

                # Legacy claim arrays (preserved for grounding pipeline compatibility)
                "direct_answer_claims": [{"text": str, "evidence_ids": list[str], "sentence_ids": list[str]}],
                "evidence_summary_claims": [...],
                "limitations_claims": [...],
                "abstention_reason": str | None
            }
        """
        ...


class EntailmentClient(Protocol):
    """Protocol for evaluating whether a claim is semantically supported by an evidence chunk."""

    async def check_entailment(self, premise: str, hypothesis: str) -> str:
        """Assess semantic entailment between premise (source chunk) and hypothesis (claim).
        
        Returns:
            One of: "entailment", "neutral", "contradiction"
        """
        ...
