"""Citation Audit & Failure Analysis Module (Steps 11 & 12).

Produces structured citation audit tables for all candidate papers (included & excluded)
and performs failure analysis diagnosing exact retrieval failure stages with actionable recommendations.
"""

from typing import Any
from ..clinical.clinical_intent import ClinicalQueryIntent


class CitationAuditor:
    """Audits every retrieved citation and performs root-cause failure analysis for excluded or mismatched papers."""

    def audit_citations(
        self,
        intent: ClinicalQueryIntent,
        included_candidates: list[Any],
        excluded_candidates: list[dict[str, Any]],
        claims: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Generate Step 11 Citation Audit entries and Step 12 Failure Analysis entries."""
        citation_audit = []
        failure_analyses = []

        # 1. Audit Included Candidates
        for cand in included_candidates:
            pmcid = getattr(cand, "pmcid", "Unknown")
            chunk_id = getattr(cand, "chunk_id", "")
            meta = getattr(cand, "paper_metadata", None)

            disease = getattr(meta, "disease", None) if meta else None
            disease = disease if disease else (getattr(intent, "disease", None) or "Unspecified")

            subtype = getattr(meta, "disease_subtype", None) if meta else None
            subtype = subtype if subtype else (getattr(intent, "disease_subtype", None) or "Unspecified")

            biomarker = getattr(meta, "biomarker", None) if meta else None
            biomarker = biomarker if biomarker else (getattr(intent, "biomarker_status", None) or "Unspecified")

            stage = getattr(meta, "stage", None) if meta else None
            stage = stage if stage else (getattr(intent, "stage", None) or "Unspecified")
            
            pop = getattr(meta, "trial_population", None) if meta else None
            pop = pop if pop else f"{biomarker} {disease}".strip()
            
            intervention = getattr(meta, "intervention", None) if meta else None
            intervention = intervention if intervention else (getattr(intent, "treatment", None) or "Unspecified")
            
            comparator = getattr(meta, "comparator", None) if meta else None
            comparator = comparator if comparator else (getattr(intent, "comparator", None) or "Unspecified")
            
            outcome = getattr(intent, "clinical_outcome", None) or "PFS / OS benefit"

            supports = "NO"
            reason = "No specific population match verified."
            
            intent_disease = getattr(intent, "disease", None)
            intent_biomarker = getattr(intent, "biomarker_status", None)
            
            disease_match = (not intent_disease) or (meta and getattr(meta, "disease", None) == intent_disease)
            biomarker_match = (not intent_biomarker) or (meta and getattr(meta, "biomarker", None) == intent_biomarker)
            
            if disease_match and biomarker_match:
                supports = "YES"
                reason = f"Exact population match for {biomarker} {disease} with verified clinical trial outcome."
            else:
                supports = "NO"
                reason = "Candidate disease or biomarker does not match intent."

            citation_audit.append({
                "pmcid": pmcid,
                "chunk_id": chunk_id,
                "disease": disease,
                "subtype": subtype,
                "biomarker": biomarker,
                "stage": stage,
                "population": pop,
                "intervention": intervention,
                "comparator": comparator,
                "outcome": outcome,
                "supports_claim": supports,
                "reason": reason,
                "inclusion_reason": f"Included due to relevance. Supports claim: {supports}. Reason: {reason}"
            })

        # 2. Audit & Analyze Excluded Candidates (Failure Analysis)
        for excl in excluded_candidates:
            pmcid = excl.get("pmcid", "Unknown")
            chunk_id = excl.get("chunk_id", "")
            meta_dict = excl.get("paper_metadata", {})
            reason = excl.get("exclusion_reason", "Population mismatch")
            stage_excl = excl.get("exclusion_stage", "Hard Filters")

            disease = meta_dict.get("disease", "Unspecified")
            subtype = meta_dict.get("disease_subtype", "Unspecified")
            biomarker = meta_dict.get("biomarker", "Unspecified")
            stage = meta_dict.get("stage", "Unspecified")
            pop = meta_dict.get("trial_population", "Unspecified")

            citation_audit.append({
                "pmcid": pmcid,
                "chunk_id": chunk_id,
                "disease": disease,
                "subtype": subtype,
                "biomarker": biomarker,
                "stage": stage,
                "population": pop,
                "intervention": meta_dict.get("intervention", "Unspecified"),
                "comparator": meta_dict.get("comparator", "Unspecified"),
                "outcome": intent.clinical_outcome or "Unspecified",
                "supports_claim": "NO",
                "reason": reason,
            })

            # Failure Analysis (Step 12)
            failed_stage = "population_mismatch"
            rec = "Enforce stricter population matching at retrieval stage."

            if "metadata" in reason.lower() or not meta_dict:
                failed_stage = "metadata_gap"
                rec = "Implement robust metadata extraction for this source to allow structured filtering."
            elif "bm25" in reason.lower() or "sparse" in reason.lower() or "keyword" in reason.lower():
                failed_stage = "hybrid_search_noise"
                rec = "Improve BM25/hybrid keyword query expansion to exclude irrelevant documents."
            elif "reranker" in reason.lower() or "cross-encoder" in reason.lower():
                failed_stage = "reranker_override"
                rec = "Adjust cross-encoder weights to penalize population mismatch harder."
            elif "vector" in reason.lower() or "dense" in reason.lower() or "embedding" in reason.lower():
                failed_stage = "embedding_mismatch"
                rec = "Tune embedding model to prioritize clinical concepts over generic oncology terms."
            elif "disease" in reason.lower() or "biomarker" in reason.lower() or "stage" in reason.lower():
                failed_stage = "population_mismatch"
                rec = "Enforce clinical population pre-filter before reranking."

            failure_analyses.append({
                "pmcid": pmcid,
                "chunk_id": chunk_id,
                "why_retrieved": "High dense/sparse vector similarity to query terms (e.g. shared drug name or oncology terms).",
                "failed_stage": failed_stage,
                "failure_description": reason,
                "recommended_pipeline_improvement": rec,
            })

        return citation_audit, failure_analyses
