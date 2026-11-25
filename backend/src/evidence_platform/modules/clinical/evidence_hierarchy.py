"""Evidence Hierarchy Module (Step 4).

Ranks evidence by clinical quality according to an 11-level hierarchy:
1. Clinical Practice Guidelines
2. Phase III Randomized Controlled Trials
3. Phase II Trials
4. Systematic Reviews
5. Meta-analysis
6. Large Cohort Studies
7. Prospective Studies
8. Retrospective Studies
9. Case Series
10. Narrative Reviews
11. Expert Opinion
"""

from typing import Any


EVIDENCE_HIERARCHY = [
    ("guideline", "Clinical Practice Guidelines", 1, 2.5),
    ("phase_3_rct", "Phase III Randomized Controlled Trial", 2, 2.2),
    ("phase_2_trial", "Phase II Trial", 3, 1.9),
    ("systematic_review", "Systematic Review", 4, 1.7),
    ("meta_analysis", "Meta-analysis", 5, 1.6),
    ("cohort_study", "Large Cohort Study", 6, 1.4),
    ("prospective_study", "Prospective Study", 7, 1.2),
    ("retrospective_study", "Retrospective Study", 8, 1.0),
    ("case_series", "Case Series", 9, 0.8),
    ("narrative_review", "Narrative Review", 10, 0.6),
    ("expert_opinion", "Expert Opinion", 11, 0.5),
]

KNOWN_TRIALS = {
    "DESTINY-Breast03": "phase_3_rct", 
    "DESTINY-Breast04": "phase_3_rct", 
    "DESTINY-Breast01": "phase_2_trial",
    "CLEOPATRA": "phase_3_rct", 
    "EMILIA": "phase_3_rct", 
    "KATHERINE": "phase_3_rct",
    "FLAURA": "phase_3_rct", 
    "FLAURA2": "phase_3_rct", 
    "KEYNOTE": "varies",
    "FIRE-3": "phase_3_rct", 
    "CRYSTAL": "phase_3_rct", 
    "CheckMate": "varies",
    "MONALEESA-2": "phase_3_rct", 
    "MONALEESA-3": "phase_3_rct", 
    "MONALEESA-7": "phase_3_rct",
    "PALOMA-2": "phase_3_rct", 
    "PALOMA-3": "phase_3_rct", 
    "MONARCH-3": "phase_3_rct",
    "SOLAR-1": "phase_3_rct", 
    "OlympiAD": "phase_3_rct",
}


class EvidenceHierarchyRanker:
    """Classifies candidate study type into the 11-tier hierarchy and assigns priority weights."""

    def classify_and_rank(self, study_type_raw: str, passage_text: str = "", publication_date: str | None = None) -> tuple[int, float, str]:
        """Classify study into hierarchy rank (1-11), priority weight, and human-readable label.

        Returns: (rank: int, weight: float, display_label: str)
        """
        st_clean = (study_type_raw or "").lower().strip()
        text_lower = passage_text.lower()
        
        is_phase3_known = False
        is_phase2_known = False
        for trial, t_type in KNOWN_TRIALS.items():
            if trial.lower() in text_lower:
                if t_type == "phase_3_rct":
                    is_phase3_known = True
                elif t_type == "phase_2_trial":
                    is_phase2_known = True

        # Primary signal from structured metadata
        if st_clean and st_clean != "other":
            if "guideline" in st_clean:
                rank, weight, label = 1, 2.5, "Clinical Practice Guidelines"
            elif "phase 3" in st_clean or "phase iii" in st_clean or st_clean == "phase_3_rct":
                rank, weight, label = 2, 2.2, "Phase III Randomized Controlled Trial"
            elif "phase 2" in st_clean or "phase ii" in st_clean or st_clean == "phase_2_trial":
                rank, weight, label = 3, 1.9, "Phase II Trial"
            elif "rct" in st_clean or "randomized" in st_clean:
                # Metadata says "RCT" but not the phase: only call it Phase III when
                # the text says so; phase II randomised trials rank one tier lower.
                if is_phase3_known or "phase iii" in text_lower or "phase 3" in text_lower:
                    rank, weight, label = 2, 2.2, "Phase III Randomized Controlled Trial"
                elif is_phase2_known or "phase ii" in text_lower or "phase 2" in text_lower:
                    rank, weight, label = 3, 1.9, "Phase II Randomized Trial"
                else:
                    rank, weight, label = 2, 2.1, "Randomized Controlled Trial"
            elif "systematic_review" in st_clean or "systematic review" in st_clean:
                rank, weight, label = 4, 1.7, "Systematic Review"
            elif "meta_analysis" in st_clean or "meta-analysis" in st_clean:
                rank, weight, label = 5, 1.6, "Meta-analysis"
            elif "cohort" in st_clean:
                rank, weight, label = 6, 1.4, "Large Cohort Study"
            elif "prospective" in st_clean:
                rank, weight, label = 7, 1.2, "Prospective Study"
            elif "retrospective" in st_clean:
                rank, weight, label = 8, 1.0, "Retrospective Study"
            elif "case" in st_clean:
                rank, weight, label = 9, 0.8, "Case Series"
            elif "review" in st_clean:
                rank, weight, label = 10, 0.6, "Narrative Review"
            else:
                rank, weight, label = 11, 0.5, "Expert Opinion"
        else:
            # Secondary signal from text-based classification
            # A guideline needs an explicit guideline signal; the bare word
            # "recommendation" (common in trial discussions) is not enough.
            if "guideline" in text_lower and (
                "recommend" in text_lower or "nccn" in text_lower or "asco" in text_lower or "esmo" in text_lower
            ):
                rank, weight, label = 1, 2.5, "Clinical Practice Guidelines"
            elif is_phase3_known or ("randomized" in text_lower or "phase iii" in text_lower or "phase 3" in text_lower) and ("phase iii" in text_lower or "phase 3" in text_lower):
                rank, weight, label = 2, 2.2, "Phase III Randomized Controlled Trial"
            elif is_phase2_known or "phase ii" in text_lower or "phase 2" in text_lower:
                rank, weight, label = 3, 1.9, "Phase II Trial"
            elif "randomized" in text_lower or "randomised" in text_lower:
                rank, weight, label = 2, 2.2, "Phase III Randomized Controlled Trial"
            elif "systematic review" in text_lower:
                rank, weight, label = 4, 1.7, "Systematic Review"
            elif "meta-analysis" in text_lower:
                rank, weight, label = 5, 1.6, "Meta-analysis"
            elif "cohort study" in text_lower or "large cohort" in text_lower:
                rank, weight, label = 6, 1.4, "Large Cohort Study"
            elif "prospective" in text_lower:
                rank, weight, label = 7, 1.2, "Prospective Study"
            elif "retrospective" in text_lower:
                rank, weight, label = 8, 1.0, "Retrospective Study"
            elif "case series" in text_lower or "case report" in text_lower:
                rank, weight, label = 9, 0.8, "Case Series"
            elif "review" in text_lower:
                rank, weight, label = 10, 0.6, "Narrative Review"
            else:
                rank, weight, label = 11, 0.5, "Expert Opinion"

        if publication_date:
            import datetime
            try:
                year_str = publication_date[:4]
                pub_year = int(year_str)
                current_year = datetime.datetime.now().year
                if current_year - pub_year <= 3:
                    weight += 0.2
            except ValueError:
                pass
                
        return rank, weight, label

    def rank_candidates(self, candidates: list[Any]) -> list[Any]:
        """Rank candidates according to clinical quality hierarchy."""
        for cand in candidates:
            st = getattr(cand, "study_type", "")
            text = getattr(cand, "text", "")
            pub_date = getattr(cand, "publication_date", getattr(getattr(cand, "paper_metadata", None), "publication_date", None))
            rank, weight, label = self.classify_and_rank(st, text, pub_date)
            setattr(cand, "hierarchy_rank", rank)
            setattr(cand, "hierarchy_weight", weight)
            setattr(cand, "hierarchy_label", label)

        # Sort candidates primarily by hierarchy rank (1 is highest), secondarily by population match score if present
        def sort_key(c):
            rank = getattr(c, "hierarchy_rank", 11)
            pop_score = getattr(c, "population_match_score", 0)
            return (rank, -pop_score)

        return sorted(candidates, key=sort_key)
