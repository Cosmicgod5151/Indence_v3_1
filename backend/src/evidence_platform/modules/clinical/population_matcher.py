"""Evidence Population Matching & Hard Filters Module (Steps 2 & 3).

Extracts paper population metadata, computes Population Match Score (+100/-300/Reject),
and enforces strict hard filters to reject population/biomarker/disease mismatches.
"""

import re
from dataclasses import dataclass, field
from typing import Any
from .clinical_intent import ClinicalQueryIntent


@dataclass
class PaperPopulationMetadata:
    disease: str | None = None
    disease_subtype: str | None = None
    biomarker: str | None = None
    trial_population: str | None = None
    inclusion_criteria: list[str] = field(default_factory=list)
    exclusion_criteria: list[str] = field(default_factory=list)
    treatment_line: str | None = None
    trial_phase: str | None = None
    intervention: str | None = None
    comparator: str | None = None
    outcomes: list[str] = field(default_factory=list)
    stage: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "disease": self.disease,
            "disease_subtype": self.disease_subtype,
            "biomarker": self.biomarker,
            "trial_population": self.trial_population,
            "inclusion_criteria": self.inclusion_criteria,
            "exclusion_criteria": self.exclusion_criteria,
            "treatment_line": self.treatment_line,
            "trial_phase": self.trial_phase,
            "intervention": self.intervention,
            "comparator": self.comparator,
            "outcomes": self.outcomes,
            "stage": self.stage,
        }


class PopulationMatchEvaluator:
    """Extracts paper population metadata, evaluates Population Match Score, and executes Hard Filters."""

    def extract_paper_metadata(self, passage_text: str, study_type: str = "") -> PaperPopulationMetadata:
        """Extract population metadata from passage text and study type."""
        text_lower = passage_text.lower()

        # Disease
        disease = None
        if "breast cancer" in text_lower or "breast carcinoma" in text_lower or "mbc" in text_lower:
            disease = "Breast Cancer"
        elif "non-small cell lung cancer" in text_lower or "nsclc" in text_lower:
            disease = "Non-Small Cell Lung Cancer"
        elif "lung cancer" in text_lower:
            disease = "Lung Cancer"
        elif "colorectal cancer" in text_lower or "mcrc" in text_lower or "colon cancer" in text_lower:
            disease = "Colorectal Cancer"
        elif "cervical cancer" in text_lower or "cervical carcinoma" in text_lower:
            disease = "Cervical Cancer"
        elif "prostate cancer" in text_lower:
            disease = "Prostate Cancer"
        elif "prostate" in text_lower:
            disease = "Prostate Cancer"
        elif "diabetes" in text_lower or "metformin" in text_lower:
            disease = "Diabetes"
        if disease is None:
            # Fall back to the full disease vocabulary used for queries
            # (ovarian, melanoma, pancreatic, gastric, RCC, urothelial, ...).
            from .clinical_intent import ClinicalIntentExtractor
            for pattern, label in ClinicalIntentExtractor.DISEASE_PATTERNS:
                if re.search(pattern, text_lower):
                    disease = label
                    break

        # Context-aware Biomarker Extraction
        biomarker = None
        
        sentences = [s.strip() for s in re.split(r'[.!?]', text_lower) if s.strip()]
        
        def has_negative_context(text_chunk, term):
            term = term.lower()
            neg_patterns = [f"excluding {term}", f"{term} were excluded", f"not {term}", f"excluded {term}", f"without {term}", "excluding her2-positive", "her2-positive were excluded", "not her2-positive"]
            return any(np in text_chunk for np in neg_patterns)
            
        def has_positive_context(text_chunk):
            pos_patterns = ["patients with", "enrolled", "included", "eligible", "population", "study population"]
            return any(pp in text_chunk for pp in pos_patterns)

        b_mappings = [
            (r"\bher2[- ]positive\b|\bher2\s*\+(?!\w)|\bher2[- ]amplified\b|\bher2\s+positive\b", "HER2-positive"),
            (r"\bher2[- ]low\b|\bher2\s+low\b", "HER2-low"),
            (r"\bher2[- ]ultralow\b", "HER2-ultralow"),
            (r"\bher2[- ]negative\b|\bher2\s*-(?![\w-])", "HER2-negative"),
            (r"\btriple[- ]negative\b|\btnbc\b", "Triple-negative"),
            (r"\bhr[- ]positive\b|\bhr\s*\+(?!\w)|\ber[- ]positive\b|\ber\s*\+(?!\w)|\bhormone[- ]receptor[- ]positive\b", "HR-positive"),
            (r"\begfr[- ]mutant\b|\begfr[- ]mutated\b|\begfr\s+mutation\b|\begfr\s*\+(?!\w)", "EGFR-mutant"),
            (r"\begfr\s+wild[- ]type\b|\begfr\s+wt\b", "EGFR wild-type"),
            (r"\bkras\s+wild[- ]type\b|\bras\s+wild[- ]type\b|\bkras\s+wt\b", "KRAS/RAS wild-type")
        ]

        for sentence in sentences:
            if not has_positive_context(sentence):
                continue
            for pattern, label in b_mappings:
                if re.search(pattern, sentence):
                    b_term_match = re.search(pattern, sentence).group(0)
                    if not has_negative_context(sentence, b_term_match) and not has_negative_context(sentence, label):
                        biomarker = label
                        break
            if biomarker:
                break
        
        if not biomarker:
            for pattern, label in b_mappings:
                if re.search(pattern, text_lower):
                    b_term_match = re.search(pattern, text_lower).group(0)
                    if not has_negative_context(text_lower, b_term_match) and not has_negative_context(text_lower, label):
                        biomarker = label
                        break

        # Stage
        stage = None
        if "metastatic" in text_lower or "stage iv" in text_lower or "advanced" in text_lower or re.search(r"\bstage\s*iii[abc]?\b", text_lower):
            stage = "Metastatic / Stage IV" if ("metastatic" in text_lower or "stage iv" in text_lower) else "Advanced (stage III-IV)"
        elif "stage ib2" in text_lower or "figo stage ib2" in text_lower:
            stage = "FIGO Stage IB2"
        elif "localized" in text_lower or "early" in text_lower or "stage i" in text_lower:
            stage = "Early / Localized"

        # Treatment Line
        line = None
        if re.search(r"\bfirst[- ]line\b|\b1l\b|\bfront[- ]line\b", text_lower):
            line = "First-line (1L)"
        elif re.search(r"\bsecond[- ]line\b|\b2l\b|\bpreviously treated\b", text_lower):
            line = "Second-line (2L)"

        # Trial Phase
        phase = None
        if "phase iii" in text_lower or "phase 3" in text_lower:
            phase = "Phase III"
        elif "phase ii" in text_lower or "phase 2" in text_lower:
            phase = "Phase II"
        elif "phase i" in text_lower or "phase 1" in text_lower:
            phase = "Phase I"

        # Intervention
        intervention = None
        treatment_patterns = [
            (r"trastuzumab deruxtecan|t-dxd|enhertu", "Trastuzumab deruxtecan"),
            (r"trastuzumab emtansine|t-dm1|kadcyla", "Trastuzumab emtansine"),
            (r"osimertinib|tagrisso", "Osimertinib"),
            (r"pembrolizumab|keytruda", "Pembrolizumab"),
            (r"nivolumab|opdivo", "Nivolumab"),
            (r"atenolizumab|atezolizumab|tecentriq", "Atezolizumab"),
            (r"trastuzumab|herceptin", "Trastuzumab"),
            (r"pertuzumab|perjeta", "Pertuzumab"),
            (r"bevacizumab|avastin", "Bevacizumab"),
            (r"cetuximab|erbitux", "Cetuximab"),
            (r"sacituzumab govitecan|trodelvy", "Sacituzumab govitecan"),
            (r"tucatinib", "Tucatinib"),
            (r"lapatinib", "Lapatinib"),
            (r"neratinib", "Neratinib"),
            (r"palbociclib", "Palbociclib"),
            (r"ribociclib", "Ribociclib"),
            (r"abemaciclib", "Abemaciclib"),
            (r"olaparib", "Olaparib"),
            (r"talazoparib", "Talazoparib"),
            (r"capecitabine", "Capecitabine"),
            (r"docetaxel", "Docetaxel"),
            (r"paclitaxel", "Paclitaxel"),
            (r"carboplatin", "Carboplatin"),
            (r"cisplatin", "Cisplatin"),
            (r"durvalumab", "Durvalumab"),
            (r"ipilimumab", "Ipilimumab"),
            (r"sotorasib", "Sotorasib"),
            (r"adagrasib", "Adagrasib"),
        ]
        for pattern, label in treatment_patterns:
            if re.search(rf"\b(?:{pattern})\b", text_lower):
                intervention = label
                break

        # Trial population description
        trial_pop = f"{biomarker or ''} {stage or ''} {disease or ''}".strip()

        return PaperPopulationMetadata(
            disease=disease,
            disease_subtype=biomarker,
            biomarker=biomarker,
            trial_population=trial_pop,
            inclusion_criteria=[],
            exclusion_criteria=[],
            treatment_line=line,
            trial_phase=phase,
            intervention=intervention,
            comparator=None,
            outcomes=[],
            stage=stage,
        )

    def calculate_match_score(
        self, intent: ClinicalQueryIntent, paper_meta: PaperPopulationMetadata
    ) -> tuple[int, bool, str]:
        """Calculate Population Match Score (+100/-300/Reject).

        Returns: (score: int, is_hard_rejected: bool, reason: str)
        """
        score = 0
        reasons = []

        # 1. Disease Matching (CRITICAL)
        if intent.disease:
            if paper_meta.disease and paper_meta.disease.lower() != intent.disease.lower():
                # Disease mismatch -> REJECT (-1000)
                return -1000, True, f"Disease mismatch: query asks for '{intent.disease}', paper is for '{paper_meta.disease}'"
            elif paper_meta.disease and paper_meta.disease.lower() == intent.disease.lower():
                score += 100
                reasons.append("Exact disease match (+100)")

        # 2. Biomarker & Subtype Matching (CRITICAL)
        if intent.biomarker_status:
            intent_bio = intent.biomarker_status.lower()
            paper_bio = (paper_meta.biomarker or "").lower()

            if paper_bio:
                if paper_bio == intent_bio:
                    score += 100
                    reasons.append("Exact biomarker match (+100)")
                elif intent.rejected_subgroups and any(rej.lower() in paper_bio for rej in intent.rejected_subgroups):
                    if not getattr(intent, "explicitly_broad_scope", False):
                        # Hard mismatch
                        score -= 300
                        return score, True, f"Biomarker population mismatch: query requires '{intent.biomarker_status}', paper is for '{paper_meta.biomarker}'"
                    else:
                        reasons.append(f"Broad scope overrides biomarker mismatch ('{paper_meta.biomarker}')")
                elif paper_bio != intent_bio:
                    if not getattr(intent, "explicitly_broad_scope", False):
                        score -= 300
                        reasons.append(f"Different biomarker (-300: '{paper_meta.biomarker}' vs '{intent.biomarker_status}')")
                    else:
                        reasons.append(f"Broad scope overrides biomarker mismatch ('{paper_meta.biomarker}')")

        # 2.5 Treatment Matching
        intent_treatments = intent.treatment if isinstance(intent.treatment, list) else ([intent.treatment] if intent.treatment else [])
        if intent_treatments and paper_meta.intervention:
            paper_intervention = paper_meta.intervention.lower()
            matched = False
            for t in intent_treatments:
                t_lower = t.lower()
                if t_lower == paper_intervention:
                    score += 80
                    reasons.append(f"Treatment match (+80: {t})")
                    matched = True
                    break
            if not matched:
                if "trastuzumab deruxtecan" in [t.lower() for t in intent_treatments] and paper_intervention == "trastuzumab emtansine":
                    score -= 100
                    reasons.append("Treatment mismatch: query asks for T-DXd but paper is T-DM1 (-100)")
                else:
                    score -= 200
                    reasons.append(f"Treatment mismatch: query asks for {intent_treatments[0]} but paper is {paper_meta.intervention} (-200)")

        # 3. Line of Therapy Matching
        if intent.line_of_therapy and paper_meta.treatment_line:
            if intent.line_of_therapy.lower() in paper_meta.treatment_line.lower() or paper_meta.treatment_line.lower() in intent.line_of_therapy.lower():
                score += 60
                reasons.append("Exact therapy line match (+60)")

        # 4. Stage Matching
        if intent.stage and paper_meta.stage:
            i_stage = intent.stage.lower()
            p_stage = paper_meta.stage.lower()
            if i_stage in p_stage or p_stage in i_stage:
                score += 40
                reasons.append("Exact stage match (+40)")
            else:
                if ("early" in i_stage or "localized" in i_stage) and ("metastatic" in p_stage or "stage iv" in p_stage):
                    score -= 200
                    reasons.append("Stage mismatch: query is early-stage, paper is metastatic (-200)")
                elif ("metastatic" in i_stage or "stage iv" in i_stage) and ("early" in p_stage or "localized" in p_stage):
                    score -= 200
                    reasons.append("Stage mismatch: query is metastatic, paper is early-stage (-200)")

        # 5. Same trial population
        if score >= 200:
            score += 50
            reasons.append("Same trial population match (+50)")

        reason_summary = "; ".join(reasons) if reasons else "Partial match"
        return score, False, reason_summary

    def filter_and_score_candidates(
        self, intent: ClinicalQueryIntent, candidates: list[Any]
    ) -> tuple[list[Any], list[dict[str, Any]]]:
        """Apply Hard Filters (Step 3) and attach Population Match Scores to candidates.

        Returns: (included_candidates, excluded_audit_entries)
        """
        included = []
        excluded = []

        for cand in candidates:
            # Candidate can be RetrievalCandidate or object with .text and .study_type
            passage_text = getattr(cand, "text", str(cand))
            study_type = getattr(cand, "study_type", "")
            pmcid = getattr(cand, "pmcid", "Unknown")

            paper_meta = self.extract_paper_metadata(passage_text, study_type)
            score, is_rejected, reason = self.calculate_match_score(intent, paper_meta)

            if is_rejected:
                excluded.append({
                    "pmcid": pmcid,
                    "chunk_id": getattr(cand, "chunk_id", ""),
                    "study_type": study_type,
                    "paper_metadata": paper_meta.to_dict(),
                    "population_match_score": score,
                    "exclusion_stage": "Step 3 - Hard Filters",
                    "exclusion_reason": reason,
                    "scored_breakdown": reason,
                    "text_snippet": passage_text[:150] + "...",
                })
            else:
                # Attach population match score to candidate object dynamically or return tuple
                setattr(cand, "population_match_score", score)
                setattr(cand, "paper_metadata", paper_meta)
                setattr(cand, "population_match_reason", reason)
                included.append(cand)

        return included, excluded
