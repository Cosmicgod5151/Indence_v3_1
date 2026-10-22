"""Evidence Grouping Module (Step 6).

Clusters retrieved evidence by:
- Population
- Trial Name (e.g. DESTINY-Breast03, CLEOPATRA)
- Outcome (PFS, OS, ORR)
- Therapy Line (1L, 2L)
- Biomarker (HER2+, EGFR mut, etc.)

Generates separate evidence clusters to prevent merging heterogeneous evidence into single claims.
"""

from dataclasses import dataclass, field
from typing import Any


import re

@dataclass
class EvidenceCluster:
    cluster_id: str
    population: str
    trial_name: str | None
    biomarker: str | None
    therapy_line: str | None
    outcome: str | None
    candidates: list[Any] = field(default_factory=list)
    summary: str | None = None
    quality_score: float = 0.0
    population_label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "population": self.population,
            "population_label": self.population_label,
            "trial_name": self.trial_name,
            "biomarker": self.biomarker,
            "therapy_line": self.therapy_line,
            "outcome": self.outcome,
            "quality_score": self.quality_score,
            "candidate_pmcids": [getattr(c, "pmcid", "") for c in self.candidates],
            "candidate_chunk_ids": [getattr(c, "chunk_id", "") for c in self.candidates],
            "count": len(self.candidates),
        }

def normalize_trial_name(raw_name: str) -> str:
    name = raw_name.lower().replace(" ", "").replace("-", "")
    if name.startswith("destinybreast"):
        num = name.replace("destinybreast", "")
        return f"DESTINY-BREAST{num.zfill(2)}"
    if name.startswith("db"):
        num = name.replace("db", "")
        return f"DESTINY-BREAST{num.zfill(2)}"
    if name.startswith("keynote"):
        num = name.replace("keynote", "")
        return f"KEYNOTE-{num}"
    if name.startswith("kn"):
        num = name.replace("kn", "")
        return f"KEYNOTE-{num}"
    if name.startswith("checkmate"):
        num = name.replace("checkmate", "")
        return f"CHECKMATE-{num}"
    if name.startswith("cm"):
        num = name.replace("cm", "")
        return f"CHECKMATE-{num}"
    return raw_name.upper()

def extract_outcomes(text: str) -> list[str]:
    outcomes = []
    if re.search(r"\b(pfs|progression[- ]free survival)\b", text, re.IGNORECASE):
        outcomes.append("PFS")
    if re.search(r"\b(os|overall survival)\b", text, re.IGNORECASE):
        outcomes.append("OS")
    if re.search(r"\b(orr|objective response rate)\b", text, re.IGNORECASE):
        outcomes.append("ORR")
    if re.search(r"\b(dor|duration of response)\b", text, re.IGNORECASE):
        outcomes.append("DOR")
    if re.search(r"\b(cr|complete response)\b", text, re.IGNORECASE):
        outcomes.append("CR")
    if re.search(r"\b(pcr|pathologic complete response)\b", text, re.IGNORECASE):
        outcomes.append("pCR")
    return outcomes

def _norm_attr(val: str | None) -> str:
    if not val or val.lower() == "general":
        return "_unspecified_"
    return val

class EvidenceGrouper:
    """Groups candidate passages into homogenous clinical evidence clusters."""

    def group_candidates(self, candidates: list[Any]) -> list[EvidenceCluster]:
        """Cluster candidates into homogeneous clinical population/trial clusters."""
        if not candidates:
            return []

        clusters_map: dict[str, EvidenceCluster] = {}

        for idx, cand in enumerate(candidates):
            passage_text = getattr(cand, "text", "")
            meta = getattr(cand, "paper_metadata", None)

            # Extract trial name if present
            trial_name = None
            trial_match = re.search(r"\b(destiny[- ]breast\d*|db[- ]\d+|cleopatra|emilia|fire[- ]3|flaura2?|keynote[- ]\d*|kn[- ]\d+|checkmate[- ]\d*|cm[- ]\d+|monaleesa[- ]\d+|paloma[- ]\d+|monarch[- ]\d+|solar[- ]1|olympiad)\b", passage_text.lower())
            if trial_match:
                trial_name = normalize_trial_name(trial_match.group(1))

            biomarker = _norm_attr(meta.biomarker if meta else None)
            disease = _norm_attr(meta.disease if meta else None)
            line = _norm_attr(meta.treatment_line if meta else None)
            stage = _norm_attr(meta.stage if meta else None)
            
            outcomes = extract_outcomes(passage_text)
            outcome_str = ",".join(outcomes) if outcomes else "_unspecified_"

            # Form key
            pop_key = f"{biomarker}_{disease}_{line}_{stage}_{trial_name or 'NoTrial'}_{outcome_str}"

            if pop_key not in clusters_map:
                cluster_id = f"Cluster_{len(clusters_map) + 1:02d}"
                pop_label = f"{biomarker} {disease}, {line}, {outcome_str} outcome".replace("_unspecified_ ", "").replace(", _unspecified_ outcome", "")
                clusters_map[pop_key] = EvidenceCluster(
                    cluster_id=cluster_id,
                    population=f"{biomarker} {stage} {disease}".replace("_unspecified_", "").strip(),
                    population_label=pop_label,
                    trial_name=trial_name,
                    biomarker=biomarker,
                    therapy_line=line,
                    outcome=outcome_str,
                    candidates=[cand],
                    quality_score=getattr(cand, "hierarchy_weight", 0.0)
                )
            else:
                clusters_map[pop_key].candidates.append(cand)
                clusters_map[pop_key].quality_score += getattr(cand, "hierarchy_weight", 0.0)

        return list(clusters_map.values())
