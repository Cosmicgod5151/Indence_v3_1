"""Applicability Matrix: does this study's population match THIS patient?

For every retrieved study and every confirmed patient fact, a deterministic
comparison yields one of:
    match        — the study passage/title describes the same characteristic
    differs      — the study describes a different/excluded characteristic
    not_reported — the retrieved text does not say (honest unknown)
Each judged cell carries the sentence from the study that justifies it, so the
clinician can verify the reasoning. No LLM is involved.
"""
from __future__ import annotations

import re
from typing import Any

from .clinical_intent import ClinicalIntentExtractor
from .population_matcher import PopulationMatchEvaluator

_extractor = ClinicalIntentExtractor()
_popmatch = PopulationMatchEvaluator()

_BIOMARKER_FAMILY = {
    "HER2-positive": "HER2", "HER2-low": "HER2", "HER2-ultralow": "HER2", "HER2-negative": "HER2", "Triple-negative": "HER2",
    "HR-positive": "HR", "EGFR-mutant": "EGFR", "EGFR wild-type": "EGFR", "KRAS/RAS wild-type": "RAS",
    "KRAS-mutant": "RAS", "BRAF-mutant": "BRAF", "ALK-positive": "ALK", "PD-L1 positive": "PD-L1",
    "MSI-H/dMMR": "MSI", "MSS/pMMR": "MSI", "BRCA-mutant": "BRCA", "HRD-positive": "BRCA",
}


# Targeted drugs imply the biomarker-defined population a trial enrolled.
_DRUG_IMPLIES: list[tuple[str, str]] = [
    (r"osimertinib|gefitinib|erlotinib|afatinib|dacomitinib|amivantamab|lazertinib", "EGFR-mutant"),
    (r"alectinib|crizotinib|lorlatinib|brigatinib|ceritinib", "ALK-positive"),
    (r"sotorasib|adagrasib|divarasib", "KRAS-mutant"),
    (r"trastuzumab|pertuzumab|tucatinib|neratinib|lapatinib|t-dxd|deruxtecan|t-dm1|emtansine", "HER2-positive"),
    (r"cetuximab|panitumumab", "KRAS/RAS wild-type"),
    (r"dabrafenib|encorafenib|vemurafenib", "BRAF-mutant"),
]
_EXPLICIT_OPPOSITES = {"EGFR wild-type": "EGFR-mutant", "EGFR-mutant": "EGFR wild-type",
                       "KRAS/RAS wild-type": "KRAS-mutant", "KRAS-mutant": "KRAS/RAS wild-type"}


def _implied_biomarkers(text: str) -> list[tuple[str, str]]:
    out = []
    for rx, label in _DRUG_IMPLIES:
        m = re.search(rf"\b(?:{rx})\b", text, re.I)
        if m:
            out.append((label, m.group(0)))
    return out


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?;])\s+|\n+", text or "") if s.strip()]


def _sentence_with(text: str, pattern: str) -> str | None:
    rx = re.compile(pattern, re.I)
    for s in _sentences(text):
        if rx.search(s):
            return s[:300]
    return None


def _cell(dimension: str, patient_value: str, status: str, quote: str | None = None, note: str | None = None) -> dict[str, Any]:
    return {"dimension": dimension, "patient": patient_value, "status": status, "quote": quote, "note": note}


def _patient_profile(facts: list[dict[str, Any]]) -> dict[str, Any]:
    from .case_context import fact_label
    vals = lambda fld: [fact_label(f) for f in facts if f.get("field") == fld]  # noqa: E731
    disease = _extractor.extract_intent(" ".join(vals("cancer_type") + vals("histology"))).disease
    biomarkers = []
    for v in vals("biomarker"):
        it = _extractor.extract_intent(v)
        labels = [b for b in [it.biomarker_status, *it.secondary_biomarkers] if b]
        if not labels and re.search(r"\b(?:er|pr|estrogen|progesterone)\b[^.]{0,15}(?:positive|\+)", v, re.I):
            labels = ["HR-positive"]
        if not labels and re.search(r"\bher2\b[^.]{0,10}(?:3\+|amplified)", v, re.I):
            labels = ["HER2-positive"]
        if not labels and re.search(r"\bher2\b[^.]{0,10}(?:1\+|2\+)", v, re.I):
            labels = ["HER2-low"]
        biomarkers += [(v, b) for b in labels]
    stage_txt = " ".join(vals("stage") + vals("metastatic_site"))
    stage = None
    if re.search(r"metasta|stage\s*iv[abc]?\b|\bm1\b", stage_txt, re.I):
        stage = "Metastatic / Stage IV"
    elif re.search(r"stage\s*iii[abc]?\d?\b|locally advanced|\badvanced\b", stage_txt, re.I):
        stage = "Advanced (stage III-IV)"
    elif re.search(r"stage\s*(?:i|ii)[abc]?\d?\b|locali[sz]ed|early", stage_txt, re.I):
        stage = "Early / Localized"
    ecog = None
    for v in vals("performance_status"):
        m = re.search(r"ecog\D{0,6}([0-4])", v, re.I)
        if m:
            ecog = int(m.group(1))
    from .case_context import is_treatment_naive
    priors = [p for p in vals("prior_therapy") if not is_treatment_naive(p)]
    naive = any(is_treatment_naive(p) for p in vals("prior_therapy"))
    prior_agents = []
    for p in priors:
        prior_agents += _extractor.extract_intent(p).treatment
    return {"disease": disease, "biomarkers": biomarkers, "stage": stage, "stage_text": stage_txt,
            "ecog": ecog, "priors": priors, "prior_agents": prior_agents, "treatment_naive": naive and not priors}


def assess_study(study_text: str, title: str, profile: dict[str, Any]) -> dict[str, Any]:
    text = f"{title}. {study_text}" if title else study_text
    meta = _popmatch.extract_paper_metadata(text)
    study_intent = _extractor.extract_intent(text)
    study_bios = {b for b in [study_intent.biomarker_status, *study_intent.secondary_biomarkers] if b}
    cells: list[dict[str, Any]] = []

    # Disease
    if profile["disease"]:
        if meta.disease and meta.disease.lower() == profile["disease"].lower():
            cells.append(_cell("Disease", profile["disease"], "match", _sentence_with(text, re.escape(profile["disease"].split()[0]))))
        elif meta.disease:
            cells.append(_cell("Disease", profile["disease"], "differs",
                               _sentence_with(text, re.escape(meta.disease.split()[0])), f"Study population: {meta.disease}"))
        else:
            cells.append(_cell("Disease", profile["disease"], "not_reported"))

    # Biomarkers
    for raw_value, label in profile["biomarkers"]:
        family = _BIOMARKER_FAMILY.get(label, label)
        same_family = {b for b in study_bios if _BIOMARKER_FAMILY.get(b, b) == family}
        fam_rx = {"HER2": r"her2|erbb2", "HR": r"hormone[- ]receptor|\bhr\b|\ber\b|estrogen", "EGFR": r"egfr",
                  "RAS": r"\b[kn]?ras\b", "BRAF": r"braf", "ALK": r"\balk\b", "PD-L1": r"pd-?l1", "MSI": r"msi|mmr",
                  "BRCA": r"brca|homologous recombination|hrd"}.get(family, re.escape(family))
        implied = [(lab, drug) for lab, drug in _implied_biomarkers(text) if _BIOMARKER_FAMILY.get(lab, lab) == family]
        if family == "EGFR" and label == "EGFR-mutant":
            pt_ex20 = bool(re.search(r"exon\s*20|ex20ins|insertion", raw_value, re.I))
            st_ex20 = bool(re.search(r"exon\s*20\s*insertion|ex20ins|exon\s*20", text, re.I)) or bool(
                re.search(r"mobocertinib|amivantamab|sunvozertinib|furmonertinib", text, re.I))
            st_common = bool(re.search(r"exon\s*19|l858r|common (?:egfr )?mutation|sensitizing", text, re.I))
            if st_ex20 and not st_common and not pt_ex20:
                cells.append(_cell("Biomarker (EGFR)", raw_value, "differs",
                                   _sentence_with(text, r"exon\s*20|mobocertinib|amivantamab|insertion"),
                                   "Study enrolled EGFR exon 20 insertions — a distinct, TKI-resistant subgroup"))
                continue
            if pt_ex20 and st_common and not st_ex20:
                cells.append(_cell("Biomarker (EGFR)", raw_value, "differs", _sentence_with(text, r"exon\s*19|l858r"),
                                   "Study enrolled common EGFR mutations, not exon 20 insertions"))
                continue
        if label in study_bios:
            cells.append(_cell(f"Biomarker ({family})", raw_value, "match", _sentence_with(text, fam_rx)))
        elif same_family:
            cells.append(_cell(f"Biomarker ({family})", raw_value, "differs", _sentence_with(text, fam_rx),
                               f"Study subgroup: {', '.join(sorted(same_family))}"))
        elif implied:
            lab, drug = implied[0]
            compatible = lab == label or (family == "HER2" and label == "HER2-positive" and lab == "HER2-positive")
            if label == "KRAS-mutant" and lab == "KRAS-mutant" and "g12c" not in raw_value.lower():
                compatible = False  # KRAS G12C inhibitors need G12C specifically
            cells.append(_cell(f"Biomarker ({family})", raw_value, "match" if compatible else "differs",
                               _sentence_with(text, re.escape(drug)),
                               f"Inferred: {drug} trials enrol {lab} patients"))
        else:
            cells.append(_cell(f"Biomarker ({family})", raw_value, "not_reported"))

    # Stage / extent ("adjuvant"/"resected" trials enrol early-stage patients)
    early_setting = re.search(r"(?<!neo)adjuvant|resected|resectable|post-?operative|stage\s*i{1,3}[ab]?\s*(?:-|–|to)\s*iii", text, re.I)
    if profile["stage"] and early_setting and profile["stage"].startswith(("Metastatic", "Advanced")) and not re.search(r"maintenance", text, re.I):
        cells.append(_cell("Stage", profile["stage_text"] or profile["stage"], "differs",
                           _sentence_with(text, r"adjuvant|resect|post-?operative|stage"),
                           "Study is in the adjuvant / resected (early-stage) setting"))
    elif profile["stage"]:
        advanced_family = ("metastatic", "advanced")
        same_band = meta.stage and (
            profile["stage"].split()[0].lower() in meta.stage.lower()
            or (profile["stage"].split()[0].lower() in advanced_family and meta.stage.split()[0].lower() in advanced_family)
        )
        if same_band:
            cells.append(_cell("Stage", profile["stage_text"] or profile["stage"], "match",
                               _sentence_with(text, r"metasta|stage\s*iv|advanced|early|locali[sz]ed|stage\s*i")))
        elif meta.stage:
            cells.append(_cell("Stage", profile["stage_text"] or profile["stage"], "differs",
                               _sentence_with(text, r"metasta|stage|advanced|early|locali[sz]ed|adjuvant"), f"Study: {meta.stage}"))
        else:
            cells.append(_cell("Stage", profile["stage_text"] or profile["stage"], "not_reported"))

    # Line of therapy / prior treatment
    if profile["priors"]:
        pre = _sentence_with(text, r"previously treated|pre-?treated|after (?:progression|failure)|second[- ]line|refractory|resistan|prior (?:therapy|treatment)")
        naive = _sentence_with(text, r"first[- ]line|treatment[- ]na[iï]ve|untreated|no prior (?:systemic )?(?:therapy|treatment)|previously untreated")
        agent_naive = None
        for a in profile["prior_agents"]:
            agent_naive = agent_naive or _sentence_with(text, rf"{re.escape(a.split()[0])}[- ]na[iï]ve|no prior {re.escape(a.split()[0])}")
        label = "; ".join(profile["priors"][:2])
        maint = _sentence_with(text, r"maintenance")
        if maint and not agent_naive:
            # Maintenance trials enrol patients who responded to prior induction therapy.
            cells.append(_cell("Prior therapy", label, "match", maint, "Maintenance setting after prior therapy"))
        elif agent_naive:
            cells.append(_cell("Prior therapy", label, "differs", agent_naive, "Study required patients naive to a drug this patient already received"))
        elif pre and not naive:
            cells.append(_cell("Prior therapy", label, "match", pre))
        elif naive and not pre:
            cells.append(_cell("Prior therapy", label, "differs", naive, "Study population was treatment-naive / first-line"))
        else:
            cells.append(_cell("Prior therapy", label, "not_reported"))

    elif profile.get("treatment_naive"):
        pre = _sentence_with(text, r"previously treated|pre-?treated|after (?:progression|failure)|second[- ]line|refractory|resistan")
        naive_s = _sentence_with(text, r"first[- ]line|treatment[- ]na[iï]ve|untreated|previously untreated")
        if naive_s and not pre:
            cells.append(_cell("Line of therapy", "Treatment-naive (first-line)", "match", naive_s))
        elif pre and not naive_s:
            cells.append(_cell("Line of therapy", "Treatment-naive (first-line)", "differs", pre, "Study population was previously treated"))
        else:
            cells.append(_cell("Line of therapy", "Treatment-naive (first-line)", "not_reported"))

    # Performance status
    if profile["ecog"] is not None:
        s = _sentence_with(text, r"ecog|performance status")
        rng = re.search(r"(?:ecog|performance status)[^.;]{0,25}?\b([0-4])\s*(?:-|–|to|or)\s*([0-4])\b", s or "", re.I)
        if rng:
            lo, hi = int(rng.group(1)), int(rng.group(2))
            st = "match" if lo <= profile["ecog"] <= hi else "differs"
            cells.append(_cell("Performance status", f"ECOG {profile['ecog']}", st, s,
                               None if st == "match" else f"Study enrolled ECOG {lo}–{hi}"))
        else:
            cells.append(_cell("Performance status", f"ECOG {profile['ecog']}", "not_reported", s))

    judged = [c for c in cells if c["status"] != "not_reported"]
    matches = sum(c["status"] == "match" for c in judged)
    differs = sum(c["status"] == "differs" for c in judged)
    score = round(matches / len(judged), 2) if judged else None
    if differs and any(c["dimension"].startswith(("Disease", "Biomarker")) and c["status"] == "differs" for c in cells):
        verdict = "Different population — use with caution"
    elif score is not None and score >= 0.75 and len(judged) >= 2:
        verdict = "Closely matches this patient"
    elif score is not None and score >= 0.5:
        verdict = "Partially applicable"
    elif differs:
        verdict = "Limited applicability"
    else:
        verdict = "Insufficient detail to judge"
    return {"cells": cells, "matches": matches, "differs": differs,
            "not_reported": len(cells) - len(judged), "score": score, "verdict": verdict}


def build_applicability_matrix(candidates: list[Any], facts: list[dict[str, Any]],
                               evidence_ids: dict[str, str] | None = None, limit: int = 8) -> dict[str, Any]:
    profile = _patient_profile(facts)
    evidence_ids = evidence_ids or {}
    rows, seen = [], set()
    for c in candidates:
        pmcid = getattr(c, "pmcid", "") or ""
        if pmcid in seen:
            continue
        seen.add(pmcid)
        a = assess_study(getattr(c, "text", ""), getattr(c, "title", ""), profile)
        rows.append({
            "evidence_id": evidence_ids.get(getattr(c, "chunk_id", "")),
            "pmcid": pmcid, "title": getattr(c, "title", ""),
            "year": (getattr(c, "publication_date", "") or "")[:4] or None,
            "study_type": getattr(c, "study_type", ""), **a,
        })
        if len(rows) >= limit:
            break
    scored = [r["score"] for r in rows if r["score"] is not None]
    return {
        "patient_profile": {k: v for k, v in profile.items() if k != "prior_agents"},
        "rows": rows,
        "mean_applicability": round(sum(scored) / len(scored), 3) if scored else None,
        "n_different_population": sum(r["verdict"].startswith("Different") for r in rows),
    }
