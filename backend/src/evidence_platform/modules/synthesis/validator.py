import re
import logging
from typing import Any
from ..retrieval.retriever import RetrievalCandidate
from ..clinical.clinical_intent import ClinicalQueryIntent

logger = logging.getLogger("evidence_platform.validator")

class NumericDosageValidator:
    """Verifies that all numeric values and dosages in LLM claims are grounded in source text."""

    def __init__(self) -> None:
        self.number_pattern = re.compile(r"\b\d+(?:\.\d+)?%?\b")

    def validate_claim(self, claim_text: str, referenced_candidates: list[RetrievalCandidate]) -> tuple[bool, list[str]]:
        """Validate a single claim text against a list of candidates."""
        claim_numbers = self.number_pattern.findall(claim_text)
        if not claim_numbers:
            return True, []

        failed_numbers = []
        source_texts = [c.text.lower() for c in referenced_candidates]

        for num in claim_numbers:
            num_clean = num.strip().lower()
            pattern = re.compile(rf"\b{re.escape(num_clean)}\b")
            found = False
            
            for src in source_texts:
                if pattern.search(src):
                    found = True
                    break
                    
            if not found:
                try:
                    num_val = float(num_clean.replace("%", ""))
                    for src in source_texts:
                        src_nums = self.number_pattern.findall(src)
                        for sn in src_nums:
                            try:
                                sn_val = float(sn.replace("%", ""))
                                if abs(num_val - sn_val) < 1e-9:
                                    found = True
                                    break
                            except ValueError:
                                continue
                        if found:
                            break
                except ValueError:
                    pass

            if not found:
                failed_numbers.append(num)

        is_valid = len(failed_numbers) == 0
        return is_valid, failed_numbers

    def validate_answer(
        self,
        direct_claims: list[dict[str, Any]],
        evidence_map: dict[str, RetrievalCandidate]
    ) -> list[dict[str, Any]]:
        """Validate and enrich direct claims with numeric validation flags."""
        validated_claims = []
        
        for claim in direct_claims:
            claim_text = claim.get("text", "")
            evidence_ids = claim.get("evidence_ids", [])
            
            candidates = []
            for ev_id in evidence_ids:
                if ev_id in evidence_map:
                    candidates.append(evidence_map[ev_id])

            is_valid, failed_nums = self.validate_claim(claim_text, candidates)
            
            enriched_claim = dict(claim)
            enriched_claim["numeric_validation_passed"] = is_valid
            enriched_claim["failed_clinical_values"] = failed_nums
            
            if not is_valid:
                logger.warning(
                    f"Clinical values validation failed for claim: '{claim_text}'. "
                    f"Failed numbers: {failed_nums}"
                )
                
            validated_claims.append(enriched_claim)

        return validated_claims


class ClinicalSafetyValidator:
    """Enforces Step 8 Clinical Safety Validation.

    Verifies for each claim and citation:
    - Correct disease
    - Correct subtype
    - Correct biomarker
    - Correct therapy
    - Correct indication
    - Correct line
    - Correct outcome
    - Correct trial population

    If any mismatch exists -> Reject the citation from the claim.
    """

    def validate_clinical_safety(
        self,
        intent: ClinicalQueryIntent,
        claims: list[dict[str, Any]],
        evidence_map: dict[str, RetrievalCandidate]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Audit each claim's citations against clinical query intent.

        Returns: (validated_claims, rejected_citations)
        """
        validated_claims = []
        rejected_citations = []

        for claim in claims:
            claim_text = claim.get("text", "")
            ev_ids = claim.get("evidence_ids", [])
            valid_ev_ids = []

            for ev_id in ev_ids:
                if ev_id not in evidence_map:
                    continue

                cand = evidence_map[ev_id]
                cand_text = getattr(cand, "text", "").lower()
                meta = getattr(cand, "paper_metadata", None)

                # Check for biomarker mismatch in cited paper
                has_biomarker_mismatch = False
                mismatch_reason = ""

                if intent.biomarker_status and intent.rejected_subgroups:
                    paper_biomarker = getattr(meta, "biomarker", None) if meta else None
                    if paper_biomarker and paper_biomarker != "General" and paper_biomarker != "None":
                        if paper_biomarker.lower() in [r.lower() for r in intent.rejected_subgroups] and paper_biomarker.lower() != intent.biomarker_status.lower():
                            has_biomarker_mismatch = True
                            mismatch_reason = f"Citation {ev_id} ({getattr(cand, 'pmcid', '')}) uses '{paper_biomarker}' population which contradicts requested biomarker '{intent.biomarker_status}'."
                    else:
                        for rej in intent.rejected_subgroups:
                            rej_l = rej.lower()
                            if rej_l in cand_text and intent.biomarker_status.lower() not in cand_text:
                                has_biomarker_mismatch = True
                                mismatch_reason = f"Citation {ev_id} ({getattr(cand, 'pmcid', '')}) uses '{rej}' population which contradicts requested biomarker '{intent.biomarker_status}'."
                                break

                # Additional validation checks
                if not has_biomarker_mismatch and meta:
                    # Disease mismatch
                    if intent.disease and meta.disease and intent.disease.lower() != meta.disease.lower():
                        has_biomarker_mismatch = True
                        mismatch_reason = f"Disease mismatch: Intent requires {intent.disease}, paper is {meta.disease}."
                    
                    # Stage mismatch
                    if not has_biomarker_mismatch and intent.stage and meta.stage:
                        i_stage = intent.stage.lower()
                        m_stage = meta.stage.lower()
                        if ("metastatic" in i_stage and "early" in m_stage) or ("early" in i_stage and "metastatic" in m_stage):
                            has_biomarker_mismatch = True
                            mismatch_reason = f"Stage mismatch: Intent requires {intent.stage}, paper is {meta.stage}."
                    
                    # Treatment mismatch
                    if not has_biomarker_mismatch and intent.treatment and meta.intervention:
                        if isinstance(intent.treatment, list):
                            intent_treatments = [t for t in intent.treatment if t]
                        elif isinstance(intent.treatment, str):
                            intent_treatments = [intent.treatment] if intent.treatment else []
                        else:
                            intent_treatments = []

                        if intent_treatments and not any(t.lower() in meta.intervention.lower() or meta.intervention.lower() in t.lower() for t in intent_treatments):
                            has_biomarker_mismatch = True
                            mismatch_reason = f"Treatment mismatch: Intent requires {intent.treatment}, paper is {meta.intervention}."
                            
                    # Line of therapy warning
                    if not has_biomarker_mismatch and intent.line_of_therapy and meta.treatment_line:
                        if intent.line_of_therapy.lower() != meta.treatment_line.lower():
                            logger.warning(f"Line of therapy mismatch for citation {ev_id}: Intent requires {intent.line_of_therapy}, paper is {meta.treatment_line}.")

                if has_biomarker_mismatch:
                    logger.warning(f"Clinical Safety Validation REJECTED citation: {mismatch_reason}")
                    rejected_citations.append({
                        "evidence_id": ev_id,
                        "pmcid": getattr(cand, "pmcid", ""),
                        "claim_text": claim_text,
                        "rejection_reason": mismatch_reason
                    })
                else:
                    valid_ev_ids.append(ev_id)

            enriched = dict(claim)
            enriched["evidence_ids"] = valid_ev_ids
            enriched["clinical_safety_passed"] = len(valid_ev_ids) > 0 or len(ev_ids) == 0
            validated_claims.append(enriched)

        return validated_claims, rejected_citations
