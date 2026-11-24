import logging
import asyncio
from typing import Any
from ...ports.models import EntailmentClient
from ..retrieval.retriever import RetrievalCandidate

logger = logging.getLogger("evidence_platform.abstention")

class AbstentionLayer:
    """Enforces clinical grounding policies by running parallel NLI checks and triggering abstention on safety failures."""

    def __init__(self, entailment_client: EntailmentClient, min_entailment_ratio: float = 0.5) -> None:
        self.entailment_client = entailment_client
        self.min_entailment_ratio = min_entailment_ratio
        self.last_verifier_meta: dict[str, Any] = {}

    def population_quality_check(
        self, candidates: list[RetrievalCandidate], intent: Any
    ) -> tuple[bool, str, bool]:
        """
        Check if population quality is sufficient.
        Returns: (passed_check, abstention_reason, all_low_quality)
        """
        if not candidates or len(candidates) < 2:
            return False, "Insufficient population-matched evidence", False

        all_low_quality = True
        for cand in candidates:
            # hierarchy_rank lives on the candidate (set by EvidenceHierarchyRanker),
            # not on paper_metadata; reading it from metadata always defaulted to 10.
            rank = getattr(cand, "hierarchy_rank", 11) or 11
            if rank < 9:
                all_low_quality = False
                break

        return True, "", all_low_quality

    async def verify_and_calibrate(
        self,
        status: str,
        direct_claims: list[dict[str, Any]],
        evidence_map: dict[str, RetrievalCandidate],
        abstention_reason: str | None = None,
        candidates: list[RetrievalCandidate] | None = None,
        intent: Any = None
    ) -> tuple[str, list[dict[str, Any]], str | None]:
        """Perform parallel NLI checks on all claims. Trigger abstention if grounding ratio is too low."""
        # 0. Population quality check
        if candidates is not None and intent is not None:
            passed, pop_reason, all_low = self.population_quality_check(candidates, intent)
            if not passed:
                return "insufficient_evidence", [], pop_reason
            if all_low:
                # lower confidence, add limitation note to claims
                for claim in direct_claims:
                    if "limitations" not in claim:
                        claim["limitations"] = []
                    if isinstance(claim["limitations"], list):
                        claim["limitations"].append("All remaining evidence is Case Series or lower quality.")

        # 1. If already in abstention state, pass through
        if status == "insufficient_evidence":
            return status, direct_claims, abstention_reason or "Insufficient evidence to answer query."

        if not direct_claims:
            return "insufficient_evidence", [], "No grounded clinical claims could be formulated."

        # 2. Prepare parallel tasks for each claim
        async def verify_single_claim(claim: dict[str, Any]) -> dict[str, Any]:
            claim_text = claim.get("text", "")
            evidence_ids = claim.get("evidence_ids", [])
            
            premises = []
            for ev_id in evidence_ids:
                if ev_id in evidence_map:
                    premises.append(evidence_map[ev_id].text)
            
            premise_text = " ".join(premises)
            
            if not premise_text:
                entailment_result = "neutral"
            else:
                entailment_result = await self.entailment_client.check_entailment(
                    premise=premise_text,
                    hypothesis=claim_text
                )
                
            enriched_claim = dict(claim)
            enriched_claim["entailment_status"] = entailment_result
            return enriched_claim

        # Batched NLI: one call verifies every claim when the client supports it;
        # otherwise fall back to concurrent per-claim checks.
        detailed_fn = getattr(self.entailment_client, "check_entailment_batch_detailed", None)
        batch_fn = getattr(self.entailment_client, "check_entailment_batch", None)
        if detailed_fn is not None:
            logger.info(f"Running calibrated batched verification for {len(direct_claims)} claims...")
            pairs, idx_with_premise = [], []
            for i, claim in enumerate(direct_claims):
                premise = " ".join(evidence_map[e].text for e in claim.get("evidence_ids", []) if e in evidence_map)
                if premise:
                    pairs.append((premise, claim.get("text", "")))
                    idx_with_premise.append(i)
            details = await detailed_fn(pairs) if pairs else []
            by_idx = dict(zip(idx_with_premise, details))
            verified_claims = []
            for i, claim in enumerate(direct_claims):
                enriched = dict(claim)
                d = by_idx.get(i, {"label": "neutral", "confidence": None, "probabilities": {}})
                enriched["entailment_status"] = d["label"]
                enriched["entailment_confidence"] = d.get("confidence")
                enriched["entailment_probabilities"] = d.get("probabilities")
                verified_claims.append(enriched)
        elif batch_fn is not None:
            logger.info(f"Running batched NLI verification for {len(direct_claims)} claims...")
            pairs, idx_with_premise = [], []
            for i, claim in enumerate(direct_claims):
                premise = " ".join(
                    evidence_map[e].text for e in claim.get("evidence_ids", []) if e in evidence_map
                )
                if premise:
                    pairs.append((premise, claim.get("text", "")))
                    idx_with_premise.append(i)
            labels = await batch_fn(pairs) if pairs else []
            label_by_idx = dict(zip(idx_with_premise, labels))
            verified_claims = []
            for i, claim in enumerate(direct_claims):
                enriched = dict(claim)
                enriched["entailment_status"] = label_by_idx.get(i, "neutral")
                verified_claims.append(enriched)
        else:
            logger.info(f"Running parallel NLI verification for {len(direct_claims)} claims...")
            verified_claims = await asyncio.gather(*[verify_single_claim(c) for c in direct_claims])

        self.last_verifier_meta = dict(getattr(self.entailment_client, "last_meta", {}) or {})

        # 3. Calculate grounding ratio
        entailed_count = sum(1 for c in verified_claims if c.get("entailment_status") == "entailment")
        grounding_ratio = entailed_count / len(direct_claims)
        logger.info(f"NLI Verification grounding ratio: {grounding_ratio:.2f} ({entailed_count}/{len(direct_claims)})")

        # 4. Enforce grounding threshold policy
        if grounding_ratio < self.min_entailment_ratio:
            logger.warning(
                f"Grounding ratio {grounding_ratio:.2f} is below minimum threshold {self.min_entailment_ratio:.2f}. "
                "Triggering dynamic abstention."
            )
            return (
                "insufficient_evidence",
                [],
                f"Answer verification failed. Grounded evidence ratio was too low ({grounding_ratio:.1%})."
            )

        # 5. Contradicted claims: remove them (they are reported, never displayed as
        #    part of the answer). Abstain only when contradictions are a substantial
        #    share of the answer (> 1/3), since that signals an unreliable generation.
        contradicted = [c for c in verified_claims if c["entailment_status"] == "contradiction"]
        if contradicted:
            logger.warning("%d contradicted claim(s) removed by verification.", len(contradicted))
            if len(contradicted) / len(verified_claims) > 1 / 3:
                return (
                    "insufficient_evidence",
                    [],
                    f"Answer verification failed: {len(contradicted)} of {len(verified_claims)} generated claims "
                    "contradict their own cited sources."
                )
            for c in verified_claims:
                if c["entailment_status"] == "contradiction":
                    c["removed_by_verification"] = True

        return status, list(verified_claims), abstention_reason
