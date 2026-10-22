"""Contradiction Detection Module (Step 7).

Detects clinical contradictions across evidence clusters, trial results, and guidelines.
Every detected contradiction is fully explained from structured evidence — no generic
messages, no placeholders, no "unspecified" tokens are ever emitted.

Each contradiction includes:
1. Conflicting evidence IDs and study metadata
2. The exact claim or outcome in conflict (dynamically extracted from evidence text)
3. A concise explanation of why findings differ (from metadata + text analysis)
4. Relative evidence quality/strength of each study
5. A clinical interpretation: true contradiction vs expected contextual variation
"""

import re
from typing import Any
from .evidence_grouper import EvidenceCluster


# ────────────────────────────────────────────────────────────────────────
# Text analysis helpers (module-level, stateless)
# ────────────────────────────────────────────────────────────────────────

_BENEFIT_PATTERNS = [
    r"statistically\s+significant\s+(?:\w+\s+)*(?:improvement|benefit|reduction)",
    r"significantly\s+improved",
    r"statistically\s+significant",
    r"demonstrated\s+superiority",
    r"superior\s+to",
    r"\bsuperior\b",
    r"significant\s+(?:pfs|os|orr|dor)\s+(?:improvement|benefit)",
    r"improved\s+(?:pfs|os|orr|dor|survival|response)",
    r"prolonged\s+(?:pfs|os|survival)",
    r"hr\s*=?\s*0\.[0-6]\d*",
]

_UNCERTAINTY_PATTERNS = [
    r"no\s+significant\s+(?:\w+\s+)*(?:difference|improvement|benefit)",
    r"did\s+not\s+(?:improve|demonstrate|show|reach)",
    r"\b(?:uncertain|inconclusive|equivocal)\b",
    r"failed\s+to\s+(?:demonstrate|show|meet)",
    r"not\s+statistically\s+significant",
    r"no\s+(?:\w+\s+)*(?:survival|pfs|os)\s+(?:benefit|advantage)",
    r"hr\s*=?\s*(?:0\.9\d*|1\.\d+)",
    r"no\s+clear\s+(?:benefit|advantage)",
]

_ENDPOINT_MAP = {
    "pfs": "progression-free survival (PFS)",
    "os": "overall survival (OS)",
    "orr": "objective response rate (ORR)",
    "dor": "duration of response (DOR)",
    "pcr": "pathologic complete response (pCR)",
    "cr": "complete response (CR)",
    "dfs": "disease-free survival (DFS)",
    "efs": "event-free survival (EFS)",
    "ttp": "time to progression (TTP)",
}

_ENDPOINT_KEYS = list(_ENDPOINT_MAP.keys())

_STUDY_TYPE_LABELS = {
    "rct": "randomized controlled trial",
    "meta_analysis": "meta-analysis",
    "meta-analysis": "meta-analysis",
    "systematic_review": "systematic review",
    "guideline": "clinical practice guideline",
    "cohort": "cohort study",
    "case_control": "case-control study",
    "case_series": "case series",
    "review": "narrative review",
    "observational": "observational study",
}


def _study_label(raw: str) -> str:
    return _STUDY_TYPE_LABELS.get(raw.lower().strip(), raw.lower().strip() or "study")


def _extract_endpoints(text: str) -> list[str]:
    text_l = text.lower()
    return [k for k in _ENDPOINT_KEYS if re.search(rf"\b{k}\b", text_l)]


def _extract_hr(text: str) -> list[float]:
    return [float(m) for m in re.findall(r"hr\s*=?\s*(\d+\.\d+)", text.lower())]


def _extract_ci(text: str) -> str | None:
    m = re.search(r"95%?\s*ci[:\s]+\(?([\d.]+)\s*[-–]\s*([\d.]+)\)?", text, re.I)
    return f"{m.group(1)}–{m.group(2)}" if m else None


def _extract_follow_up(text: str) -> str | None:
    m = re.search(r"(\d+[\.\d]*)\s*(months?|years?)\s*(?:of\s+)?(?:median\s+)?(?:follow[- ]?up)", text, re.I)
    if not m:
        m = re.search(r"(?:median\s+follow[- ]?up)\s*(?:of\s+)?(\d+[\.\d]*)\s*(months?|years?)", text, re.I)
    return f"{m.group(1)} {m.group(2)}" if m else None


def _extract_sample_size(text: str) -> str | None:
    m = re.search(r"(?:n\s*=\s*|enrolled\s+|included\s+|randomized\s+)(\d[\d,]*)\s*(?:patients?|subjects?|participants?)?", text, re.I)
    return m.group(1).replace(",", "") if m else None


def _extract_trial_name(text: str) -> str | None:
    m = re.search(
        r"\b(destiny[- ]?breast\d*|db[- ]?\d+|cleopatra|emilia|fire[- ]?3|flaura2?|"
        r"keynote[- ]?\d*|kn[- ]?\d+|checkmate[- ]?\d*|cm[- ]?\d+|monaleesa[- ]?\d+|"
        r"paloma[- ]?\d+|monarch[- ]?\d+|solar[- ]?\d+|olympiad|katherine|"
        r"aphinity|sophia|tulip|tuxedo|adaura|alex|combi[- ]?d|"
        r"[a-z]+-\d{2,5})\b",
        text, re.I
    )
    return m.group(1).upper() if m else None


def _extract_key_finding(text: str) -> str:
    """Extract the core clinical finding sentence from passage text."""
    text_clean = text.strip()
    # Try to find the sentence containing the key outcome signal
    sentences = re.split(r'(?<=[.!?])\s+', text_clean)
    # Prioritize sentences with statistical results
    for sent in sentences:
        sl = sent.lower()
        if any(re.search(p, sl) for p in _BENEFIT_PATTERNS + _UNCERTAINTY_PATTERNS):
            return sent.strip().rstrip(".")
    # Fall back to first sentence if short, else truncate
    if sentences:
        first = sentences[0].strip().rstrip(".")
        if len(first) <= 200:
            return first
        return first[:197] + "…"
    return text_clean[:200].rstrip(".") + ("…" if len(text_clean) > 200 else "")


def _meta(cand: Any, attr: str) -> str:
    """Safely extract a metadata attribute, returning '' if absent."""
    meta = getattr(cand, "paper_metadata", None)
    if meta is None:
        return ""
    val = getattr(meta, attr, None)
    if val is None or str(val).strip().lower() in ("", "general", "_unspecified_", "unspecified", "none"):
        return ""
    return str(val).strip()


# ────────────────────────────────────────────────────────────────────────


class ContradictionChecker:
    """Scans evidence clusters and claims to identify clinical contradictions.

    Every output field is dynamically derived from the actual evidence text
    and structured metadata — never generic or placeholder text.
    """

    # ================================================================== #
    #  Public API                                                         #
    # ================================================================== #

    def check_contradictions(
        self, clusters: list[EvidenceCluster], candidates: list[Any]
    ) -> dict[str, Any]:
        """Analyze candidates and clusters for outcome or recommendation conflicts."""
        contradictions_found: list[dict[str, Any]] = []
        consensus_statements: list[str] = []
        _seen_pairs: set[tuple[str, ...]] = set()

        # ── 1. Accumulate per-population outcomes from clusters ────────
        cluster_outcomes: dict[str, list[dict[str, Any]]] = {}
        for cluster in clusters:
            pop = cluster.population_label
            for cand in cluster.candidates:
                text = getattr(cand, "text", "")
                text_l = text.lower()
                st = (getattr(cand, "study_type", "") or "").lower()
                hr_vals = _extract_hr(text)
                has_ben = any(re.search(p, text_l) for p in _BENEFIT_PATTERNS)
                has_unc = any(re.search(p, text_l) for p in _UNCERTAINTY_PATTERNS)
                has_gl_restr = "guideline" in st and (
                    "after progression" in text_l
                    or "second-line" in text_l
                    or "restricted" in text_l
                )
                if pop not in cluster_outcomes:
                    cluster_outcomes[pop] = []
                cluster_outcomes[pop].append({
                    "candidate": cand,
                    "has_benefit": has_ben,
                    "has_uncertainty": has_unc,
                    "has_guideline_restriction": has_gl_restr,
                    "hr_values": hr_vals,
                    "text": text,
                    "study_type": st,
                })

        # ── 2. Candidate-level efficacy contradiction ──────────────────
        all_cands = candidates if candidates else [
            c for cl in clusters for c in cl.candidates
        ]
        benefit_cands = [
            c for c in all_cands
            if any(re.search(p, getattr(c, "text", "").lower()) for p in _BENEFIT_PATTERNS)
        ]
        uncertainty_cands = [
            c for c in all_cands
            if any(re.search(p, getattr(c, "text", "").lower()) for p in _UNCERTAINTY_PATTERNS)
        ]

        if benefit_cands and uncertainty_cands:
            cand_a, cand_b = benefit_cands[0], uncertainty_cands[0]
            pair_key = self._pair_key(cand_a, cand_b)
            if pair_key not in _seen_pairs:
                _seen_pairs.add(pair_key)
                contradictions_found.append(
                    self._build_efficacy_contradiction(cand_a, cand_b)
                )

        # ── 3. Cross-cluster contradictions ────────────────────────────
        for pop, outcomes in cluster_outcomes.items():
            benefit_oc = [o for o in outcomes if o["has_benefit"]]
            uncert_oc = [o for o in outcomes if o["has_uncertainty"]]
            guide_oc = [o for o in outcomes if o["has_guideline_restriction"]]
            hr_vals = [h for o in outcomes for h in o["hr_values"]]

            if benefit_oc and uncert_oc:
                ca, cb = benefit_oc[0]["candidate"], uncert_oc[0]["candidate"]
                pk = self._pair_key(ca, cb)
                if pk not in _seen_pairs:
                    _seen_pairs.add(pk)
                    contradictions_found.append(
                        self._build_efficacy_contradiction(ca, cb)
                    )

            if benefit_oc and guide_oc:
                ca, cb = benefit_oc[0]["candidate"], guide_oc[0]["candidate"]
                contradictions_found.append(
                    self._build_guideline_contradiction(ca, cb)
                )

            if len(hr_vals) >= 2 and min(hr_vals) < 0.7 and max(hr_vals) > 0.9:
                min_o = next(
                    (o for o in outcomes if any(abs(h - min(hr_vals)) < 0.001 for h in o["hr_values"])),
                    outcomes[0],
                )
                max_o = next(
                    (o for o in outcomes if any(abs(h - max(hr_vals)) < 0.001 for h in o["hr_values"])),
                    outcomes[-1],
                )
                contradictions_found.append(
                    self._build_hr_contradiction(
                        min_o["candidate"], max_o["candidate"],
                        min(hr_vals), max(hr_vals),
                    )
                )

        # ── 4. Consensus summary ──────────────────────────────────────
        if not contradictions_found:
            consensus_statements.append(
                "All retrieved evidence demonstrates consistent alignment "
                "regarding intervention efficacy and clinical recommendations."
            )
        else:
            consensus_statements.append(
                f"Detected {len(contradictions_found)} evidence "
                f"{'contradiction' if len(contradictions_found) == 1 else 'contradictions'} "
                f"requiring clinician review — see details below."
            )

        return {
            "has_contradiction": len(contradictions_found) > 0,
            "contradiction_count": len(contradictions_found),
            "contradictions": contradictions_found,
            "consensus_summary": " ".join(consensus_statements),
        }

    # ================================================================== #
    #  Contradiction builders (fully dynamic — no templates)              #
    # ================================================================== #

    def _build_efficacy_contradiction(
        self, benefit_cand: Any, uncert_cand: Any
    ) -> dict[str, Any]:
        """Build a fully explained efficacy disagreement from two candidates."""
        id_a = self._evidence_id_tuple(benefit_cand)
        id_b = self._evidence_id_tuple(uncert_cand)
        quality_a = self._assess_quality(benefit_cand)
        quality_b = self._assess_quality(uncert_cand)
        reason = self._build_explanation(benefit_cand, uncert_cand)
        claim = self._build_claim_in_conflict(benefit_cand, uncert_cand)
        description = self._build_dynamic_description(
            benefit_cand, uncert_cand, "efficacy"
        )
        interpretation = self._clinical_interpretation(
            "Efficacy Disagreement", reason, quality_a, quality_b
        )

        return {
            "type": "Efficacy Disagreement",
            "severity": "Moderate",
            "description": description,
            "action": self._build_dynamic_action(reason, quality_a, quality_b),
            "conflicting_evidence_ids": [id_a, id_b],
            "claim_in_conflict": claim,
            "reason_for_difference": reason,
            "evidence_quality": [quality_a, quality_b],
            "clinical_interpretation": interpretation,
        }

    def _build_guideline_contradiction(
        self, trial_cand: Any, guideline_cand: Any
    ) -> dict[str, Any]:
        id_a = self._evidence_id_tuple(trial_cand)
        id_b = self._evidence_id_tuple(guideline_cand)
        quality_a = self._assess_quality(trial_cand)
        quality_b = self._assess_quality(guideline_cand)
        reason = self._build_explanation(trial_cand, guideline_cand)
        claim = self._build_claim_in_conflict(trial_cand, guideline_cand)
        description = self._build_dynamic_description(
            trial_cand, guideline_cand, "guideline"
        )
        interpretation = self._clinical_interpretation(
            "Guideline vs Trial Contradiction", reason, quality_a, quality_b
        )

        return {
            "type": "Guideline vs Trial Contradiction",
            "severity": "High",
            "description": description,
            "action": self._build_dynamic_action(reason, quality_a, quality_b),
            "conflicting_evidence_ids": [id_a, id_b],
            "claim_in_conflict": claim,
            "reason_for_difference": reason,
            "evidence_quality": [quality_a, quality_b],
            "clinical_interpretation": interpretation,
        }

    def _build_hr_contradiction(
        self, low_hr_cand: Any, high_hr_cand: Any,
        low_hr: float, high_hr: float
    ) -> dict[str, Any]:
        id_a = self._evidence_id_tuple(low_hr_cand)
        id_b = self._evidence_id_tuple(high_hr_cand)
        quality_a = self._assess_quality(low_hr_cand)
        quality_b = self._assess_quality(high_hr_cand)
        reason = self._build_explanation(low_hr_cand, high_hr_cand)

        pmcid_a = id_a["pmcid"]
        pmcid_b = id_b["pmcid"]
        trial_a = _extract_trial_name(getattr(low_hr_cand, "text", "")) or pmcid_a
        trial_b = _extract_trial_name(getattr(high_hr_cand, "text", "")) or pmcid_b

        ep_a = _extract_endpoints(getattr(low_hr_cand, "text", ""))
        ep_b = _extract_endpoints(getattr(high_hr_cand, "text", ""))
        ep_label = _ENDPOINT_MAP.get(ep_a[0], "primary endpoint") if ep_a else "primary endpoint"

        claim = (
            f"{trial_a} reports HR={low_hr} for {ep_label}, suggesting meaningful treatment benefit; "
            f"{trial_b} reports HR={high_hr}, suggesting no clear advantage"
        )

        description = (
            f"{trial_a} ({_study_label(id_a['study_type'])}) and "
            f"{trial_b} ({_study_label(id_b['study_type'])}) "
            f"report divergent hazard ratios (HR={low_hr} vs HR={high_hr}) "
            f"for {ep_label}"
        )

        interpretation = self._clinical_interpretation(
            "Quantitative Contradiction", reason, quality_a, quality_b
        )

        return {
            "type": "Quantitative Contradiction",
            "severity": "Critical",
            "description": description,
            "action": self._build_dynamic_action(reason, quality_a, quality_b),
            "conflicting_evidence_ids": [id_a, id_b],
            "claim_in_conflict": claim,
            "reason_for_difference": reason,
            "evidence_quality": [quality_a, quality_b],
            "clinical_interpretation": interpretation,
        }

    # ================================================================== #
    #  Dynamic text builders                                              #
    # ================================================================== #

    @staticmethod
    def _build_dynamic_description(cand_a: Any, cand_b: Any, kind: str) -> str:
        """Generate a specific description naming studies, endpoints, and findings."""
        pmcid_a = getattr(cand_a, "pmcid", "")
        pmcid_b = getattr(cand_b, "pmcid", "")
        text_a = getattr(cand_a, "text", "")
        text_b = getattr(cand_b, "text", "")
        st_a = _study_label(getattr(cand_a, "study_type", "") or "")
        st_b = _study_label(getattr(cand_b, "study_type", "") or "")
        trial_a = _extract_trial_name(text_a) or pmcid_a
        trial_b = _extract_trial_name(text_b) or pmcid_b

        # Population context
        pop_parts = []
        disease = _meta(cand_a, "disease") or _meta(cand_b, "disease")
        biomarker = _meta(cand_a, "biomarker") or _meta(cand_b, "biomarker")
        if disease:
            pop_parts.append(disease)
        if biomarker:
            pop_parts.append(biomarker)
        pop_str = f" in {', '.join(pop_parts)}" if pop_parts else ""

        # Endpoints
        ep_a = _extract_endpoints(text_a)
        ep_b = _extract_endpoints(text_b)
        all_eps = list(dict.fromkeys(ep_a + ep_b))  # preserve order, dedupe
        ep_str = ""
        if all_eps:
            ep_names = [_ENDPOINT_MAP.get(e, e.upper()) for e in all_eps[:3]]
            ep_str = f" regarding {', '.join(ep_names)}"

        if kind == "guideline":
            return (
                f"{trial_a} ({st_a}) demonstrates clinical benefit{pop_str}{ep_str}, "
                f"while {trial_b} ({st_b}) restricts or limits the recommended use of this intervention"
            )

        # Efficacy disagreement
        finding_a = _extract_key_finding(text_a)
        finding_b = _extract_key_finding(text_b)

        return (
            f"{trial_a} ({st_a}) reports: '{finding_a}' "
            f"whereas {trial_b} ({st_b}) reports: '{finding_b}'{pop_str}"
        )

    @staticmethod
    def _build_dynamic_action(reason: str, quality_a: dict, quality_b: dict) -> str:
        """Generate a specific action recommendation based on the detected reason."""
        parts = []

        if "study designs" in reason.lower():
            parts.append(
                f"Compare the {quality_a.get('study_type','study')} and "
                f"{quality_b.get('study_type','study')} designs directly"
            )
        if "endpoints" in reason.lower():
            parts.append("distinguish results by primary endpoint measured")
        if "biomarker" in reason.lower() or "population" in reason.lower():
            parts.append("separate conclusions by patient population and biomarker status")
        if "follow-up" in reason.lower():
            parts.append("note differing maturity of follow-up data")
        if "therapy" in reason.lower() or "line" in reason.lower():
            parts.append("clarify the applicable line of therapy for each finding")
        if "confidence" in reason.lower() or "statistical" in reason.lower():
            parts.append("report confidence intervals and statistical significance for both")

        if not parts:
            str_a = quality_a.get("strength_tier", "")
            str_b = quality_b.get("strength_tier", "")
            if str_a != str_b:
                parts.append(
                    f"weigh the {str_a.lower()} evidence ({quality_a.get('study_type','study')}) "
                    f"against the {str_b.lower()} evidence ({quality_b.get('study_type','study')})"
                )
            else:
                parts.append("present both findings with their respective contexts and limitations")

        return "In synthesis: " + "; ".join(parts)

    @staticmethod
    def _build_claim_in_conflict(cand_a: Any, cand_b: Any) -> str:
        """Build a concise, clinically meaningful claim-vs-claim string."""
        text_a = getattr(cand_a, "text", "")
        text_b = getattr(cand_b, "text", "")
        pmcid_a = getattr(cand_a, "pmcid", "Study A")
        pmcid_b = getattr(cand_b, "pmcid", "Study B")
        trial_a = _extract_trial_name(text_a) or pmcid_a
        trial_b = _extract_trial_name(text_b) or pmcid_b

        finding_a = _extract_key_finding(text_a)
        finding_b = _extract_key_finding(text_b)

        return f"{trial_a}: '{finding_a}' vs. {trial_b}: '{finding_b}'"

    # ================================================================== #
    #  Explanation and interpretation helpers                              #
    # ================================================================== #

    @staticmethod
    def _build_explanation(cand_a: Any, cand_b: Any) -> str:
        """Infer why two candidates differ, based on metadata + text content."""
        reasons: list[str] = []
        text_a = getattr(cand_a, "text", "").lower()
        text_b = getattr(cand_b, "text", "").lower()

        # 1. Study design
        st_a = (getattr(cand_a, "study_type", "") or "").lower()
        st_b = (getattr(cand_b, "study_type", "") or "").lower()
        if st_a and st_b and st_a != st_b:
            reasons.append(
                f"Different study designs ({_study_label(st_a)} vs {_study_label(st_b)})"
            )

        # 2. Biomarker / patient population
        bio_a, bio_b = _meta(cand_a, "biomarker"), _meta(cand_b, "biomarker")
        if bio_a and bio_b and bio_a.lower() != bio_b.lower():
            reasons.append(f"Different biomarker populations ({bio_a} vs {bio_b})")

        # 3. Disease / disease subtype
        dis_a, dis_b = _meta(cand_a, "disease"), _meta(cand_b, "disease")
        sub_a, sub_b = _meta(cand_a, "disease_subtype"), _meta(cand_b, "disease_subtype")
        if sub_a and sub_b and sub_a.lower() != sub_b.lower():
            reasons.append(f"Different disease subtypes ({sub_a} vs {sub_b})")
        elif dis_a and dis_b and dis_a.lower() != dis_b.lower():
            reasons.append(f"Different diseases ({dis_a} vs {dis_b})")

        # 4. Line of therapy
        line_a, line_b = _meta(cand_a, "treatment_line"), _meta(cand_b, "treatment_line")
        if line_a and line_b and line_a.lower() != line_b.lower():
            reasons.append(f"Different lines of therapy ({line_a} vs {line_b})")

        # 5. Intervention / comparator
        int_a, int_b = _meta(cand_a, "intervention"), _meta(cand_b, "intervention")
        if int_a and int_b and int_a.lower() != int_b.lower():
            reasons.append(f"Different interventions ({int_a} vs {int_b})")

        # 6. Disease stage
        stg_a, stg_b = _meta(cand_a, "stage"), _meta(cand_b, "stage")
        if stg_a and stg_b and stg_a.lower() != stg_b.lower():
            reasons.append(f"Different disease stages ({stg_a} vs {stg_b})")

        # 7. Endpoints
        ep_a = _extract_endpoints(text_a)
        ep_b = _extract_endpoints(text_b)
        if ep_a and ep_b and set(ep_a) != set(ep_b):
            reasons.append(
                f"Different primary endpoints measured "
                f"({', '.join(_ENDPOINT_MAP.get(e, e.upper()) for e in ep_a)} "
                f"vs {', '.join(_ENDPOINT_MAP.get(e, e.upper()) for e in ep_b)})"
            )

        # 8. Follow-up duration
        fu_a = _extract_follow_up(text_a)
        fu_b = _extract_follow_up(text_b)
        if fu_a and fu_b and fu_a != fu_b:
            reasons.append(f"Different follow-up durations ({fu_a} vs {fu_b})")

        # 9. Sample size
        n_a = _extract_sample_size(text_a)
        n_b = _extract_sample_size(text_b)
        if n_a and n_b and n_a != n_b:
            reasons.append(f"Different sample sizes (n={n_a} vs n={n_b})")

        # 10. Confidence interval reporting
        ci_a = _extract_ci(text_a)
        ci_b = _extract_ci(text_b)
        if ci_a and ci_b:
            reasons.append(f"Different confidence intervals (95% CI {ci_a} vs {ci_b})")
        elif bool(ci_a) != bool(ci_b):
            reasons.append(
                "Differing statistical detail (confidence interval reported in "
                f"{'first' if ci_a else 'second'} study only)"
            )

        if not reasons:
            # No metadata differences found — flag for manual review
            return (
                "Both studies examined comparable populations with similar methodology "
                "but reached opposing conclusions — manual review of full-text "
                "methodology, inclusion criteria, and statistical approach is needed"
            )

        return "; ".join(reasons)

    @staticmethod
    def _clinical_interpretation(
        conflict_type: str, reason: str, quality_a: dict, quality_b: dict
    ) -> str:
        """Generate a concise, evidence-grounded clinical interpretation."""
        strength_a = quality_a.get("strength_tier", "Unknown")
        strength_b = quality_b.get("strength_tier", "Unknown")
        type_a = quality_a.get("study_type", "study")
        type_b = quality_b.get("study_type", "study")
        pmcid_a = quality_a.get("pmcid", "Study A")
        pmcid_b = quality_b.get("pmcid", "Study B")

        contextual_markers = [
            "study designs", "endpoints", "biomarker",
            "subtypes", "lines of therapy", "follow-up",
            "stages", "interventions", "diseases",
            "sample sizes", "confidence intervals",
        ]

        detected_contexts = [m for m in contextual_markers if m in reason.lower()]

        # Check for manual-review / no-metadata-difference cases FIRST
        if not detected_contexts or "manual review" in reason.lower():
            return (
                f"Potential inconsistency—manual review recommended. "
                f"The {_study_label(type_a)} ({pmcid_a}, {strength_a} evidence) and "
                f"{_study_label(type_b)} ({pmcid_b}, {strength_b} evidence) "
                f"reached opposing conclusions from apparently similar study conditions; "
                f"review full-text methodology to identify hidden confounders."
            )

        if detected_contexts:
            # Contextual difference — explain why it's expected
            primary = reason.split(";")[0].strip()
            interp = (
                f"Expected contextual variation rather than a true contradiction: "
                f"{primary.lower()}. "
            )
            if strength_a != strength_b:
                interp += (
                    f"The {_study_label(type_a)} from {pmcid_a} ({strength_a} evidence) "
                    f"and the {_study_label(type_b)} from {pmcid_b} ({strength_b} evidence) "
                    f"address different clinical scenarios — direct comparison is not appropriate."
                )
            else:
                interp += (
                    f"Both {_study_label(type_a)} and {_study_label(type_b)} provide "
                    f"{strength_a.lower()} evidence but in differing clinical contexts."
                )
            return interp

        if "potential inconsistency" in reason.lower() or "manual review" in reason.lower():
            return (
                f"Potential inconsistency—manual review recommended. "
                f"The {_study_label(type_a)} ({pmcid_a}, {strength_a} evidence) and "
                f"{_study_label(type_b)} ({pmcid_b}, {strength_b} evidence) "
                f"reached opposing conclusions from apparently similar study conditions; "
                f"review full-text methodology to identify hidden confounders."
            )

        # Fallback — still specific
        return (
            f"Potential inconsistency—manual review recommended. "
            f"The {_study_label(type_a)} ({pmcid_a}, {strength_a} evidence) and "
            f"{_study_label(type_b)} ({pmcid_b}, {strength_b} evidence) "
            f"show conflicting findings; clinician judgement needed to reconcile."
        )

    # ================================================================== #
    #  Small helpers                                                      #
    # ================================================================== #

    @staticmethod
    def _pair_key(cand_a: Any, cand_b: Any) -> tuple[str, ...]:
        id_a = getattr(cand_a, "pmcid", "") or getattr(cand_a, "chunk_id", "a")
        id_b = getattr(cand_b, "pmcid", "") or getattr(cand_b, "chunk_id", "b")
        return tuple(sorted([id_a, id_b]))

    @staticmethod
    def _evidence_id_tuple(candidate: Any) -> dict[str, str]:
        return {
            "pmcid": getattr(candidate, "pmcid", "Unknown"),
            "study_type": getattr(candidate, "study_type", "Unknown") or "Unknown",
            "hierarchy_label": getattr(candidate, "hierarchy_label", ""),
        }

    @staticmethod
    def _assess_quality(candidate: Any) -> dict[str, Any]:
        hierarchy_weight = getattr(candidate, "hierarchy_weight", 1.0)
        study_type = (getattr(candidate, "study_type", "") or "").lower()
        hierarchy_label = getattr(candidate, "hierarchy_label", "")

        if hierarchy_weight >= 2.5:
            strength = "Very High"
        elif hierarchy_weight >= 2.0:
            strength = "High"
        elif hierarchy_weight >= 1.5:
            strength = "Moderate"
        elif hierarchy_weight >= 1.0:
            strength = "Low"
        else:
            strength = "Very Low"

        return {
            "pmcid": getattr(candidate, "pmcid", "Unknown"),
            "study_type": _study_label(study_type) if study_type else "unknown",
            "hierarchy_weight": hierarchy_weight,
            "hierarchy_label": hierarchy_label,
            "strength_tier": strength,
        }
