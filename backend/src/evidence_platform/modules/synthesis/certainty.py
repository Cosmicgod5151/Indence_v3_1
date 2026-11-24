"""GRADE-inspired certainty-of-evidence engine (deterministic, explainable).

GRADE rates the certainty of a body of evidence on four levels (High, Moderate,
Low, Very low). Randomised evidence starts High and observational evidence
starts Low; certainty is then downgraded for risk of bias, inconsistency,
indirectness, imprecision and publication bias.

This module applies those domains to the *retrieved evidence bundle* using
measurable signals only — study design labels, trial-phase/design wording,
reported sample sizes, the Quantitative Evidence Map (I^2, opposite-direction
effects, confidence intervals) and population-match scores. Every downgrade
carries a human-readable reason.

It replaces the previous behaviour where the LLM simply *wrote* a quality label.
It is NOT a formal GRADE assessment by a guideline panel; publication bias is
reported as "not assessed".
"""
from __future__ import annotations

import re
from typing import Any

LEVELS = {4: "High", 3: "Moderate", 2: "Low", 1: "Very low"}
SYMBOLS = {4: "⊕⊕⊕⊕", 3: "⊕⊕⊕◯", 2: "⊕⊕◯◯", 1: "⊕◯◯◯"}
RANDOMISED = {"rct", "meta_analysis", "systematic_review", "guideline"}

_N_RE = re.compile(
    r"(?:\bn\s*=\s*(\d{1,3}(?:,\d{3})+|\d{2,6})\b)|(\d{1,3}(?:,\d{3})+|\d{2,6})\s+(?:patients|participants|women|men|subjects|individuals)\b",
    re.I,
)
_WEAK_DESIGN = re.compile(r"\bphase\s*(?:ii|2)\b(?!i)|single[- ]arm|non[- ]?randomi[sz]ed|retrospective|open[- ]label", re.I)


def _sample_size(text: str) -> int | None:
    best = None
    for m in _N_RE.finditer(text or ""):
        raw = (m.group(1) or m.group(2) or "").replace(",", "")
        try:
            n = int(raw)
        except ValueError:
            continue
        if 10 <= n <= 500000 and (best is None or n > best):
            best = n
    return best


def _domain(name: str, downgrade: int, reason: str, assessed: bool = True) -> dict[str, Any]:
    judgement = "not assessed" if not assessed else {0: "no serious concern", 1: "serious", 2: "very serious"}[downgrade]
    return {"domain": name, "judgement": judgement, "downgrade": downgrade, "reason": reason}


def assess_certainty(candidates: list[Any], evidence_map: dict[str, Any] | None,
                     contradiction_info: dict[str, Any] | None = None) -> dict[str, Any]:
    """Rate certainty for the evidence bundle given to the generator."""
    if not candidates:
        return {"level": None, "label": "Not assessable", "symbols": "", "domains": [],
                "basis": "No evidence passages were available."}

    designs = [(getattr(c, "study_type", "") or "other").lower() for c in candidates]
    n_rand = sum(d in RANDOMISED for d in designs)
    start = 4 if n_rand >= len(designs) / 2 else 2
    start_reason = (f"{n_rand}/{len(designs)} passages come from randomised trials, meta-analyses, systematic reviews "
                    f"or guidelines → start at High." if start == 4 else
                    f"Only {n_rand}/{len(designs)} passages are randomised/synthesised evidence → start at Low.")
    domains: list[dict[str, Any]] = []

    # 1. Risk of bias (proxy): weak design wording among trial passages.
    weak = sum(bool(_WEAK_DESIGN.search(getattr(c, "text", "") or "")) for c in candidates)
    if weak >= max(2, len(candidates) / 2):
        domains.append(_domain("Risk of bias", 1, f"{weak}/{len(candidates)} passages describe phase II, single-arm, "
                                                  "open-label, non-randomised or retrospective designs."))
    else:
        domains.append(_domain("Risk of bias", 0, f"Weak-design wording in {weak}/{len(candidates)} passages "
                                                  "(proxy only; full risk-of-bias appraisal needs the full text)."))

    # 2. Inconsistency: quantitative conflicts from the evidence map + narrative contradictions.
    em = evidence_map or {}
    qconf = em.get("quantitative_conflicts") or []
    opposite = [q for q in qconf if "OPPOSITE" in (q.get("description") or "")]
    high_i2 = [q for q in qconf if q not in opposite]
    narrative = (contradiction_info or {}).get("contradiction_count", 0) or 0
    if opposite:
        domains.append(_domain("Inconsistency", 2 if len(opposite) > 1 else 1,
                               "Significant effects in opposite directions for "
                               + ", ".join(f"{q['endpoint']} ({q['measure']})" for q in opposite) + "."))
    elif high_i2:
        domains.append(_domain("Inconsistency", 1, "; ".join(
            f"{q['endpoint']} {q['measure']}: I² = {q['i2']:.0f}% across {q['k']} papers" for q in high_i2) + "."))
    elif narrative:
        domains.append(_domain("Inconsistency", 1, f"{narrative} narrative contradiction(s) detected between sources."))
    else:
        groups = [g for g in (em.get("groups") or []) if g.get("context", "treatment_comparison") == "treatment_comparison"]
        detail = (f" Largest effect group: {groups[0]['endpoint']} {groups[0]['measure']}, k={groups[0]['k']}"
                  + (f", I² = {groups[0]['i2']:.0f}%" if groups[0].get("i2") is not None else "") + ".") if groups else ""
        domains.append(_domain("Inconsistency", 0, "No quantitative or narrative conflict detected." + detail))

    # 3. Indirectness: how well the evidence population matches the question.
    pops = [getattr(c, "population_match_score", 0) or 0 for c in candidates]
    direct = sum(p >= 100 for p in pops)
    mismatched = sum(p < 0 for p in pops)
    if mismatched >= len(pops) / 2:
        domains.append(_domain("Indirectness", 2, f"{mismatched}/{len(pops)} passages show population or treatment "
                                                  "mismatches with the question."))
    elif direct < len(pops) / 2:
        domains.append(_domain("Indirectness", 1, f"Only {direct}/{len(pops)} passages confirm the queried disease/"
                                                  "subgroup; the rest are not clearly the same population."))
    else:
        domains.append(_domain("Indirectness", 0, f"{direct}/{len(pops)} passages match the queried population."))

    # 4. Imprecision: confidence interval of the main effect + total sample size.
    groups = [g for g in (em.get("groups") or []) if g.get("context", "treatment_comparison") == "treatment_comparison"]
    sizes = [n for n in (_sample_size(getattr(c, "text", "")) for c in candidates) if n]
    total_n = sum(sizes)
    if groups:
        g = groups[0]
        ref = g.get("pooled") or (g["estimates"][0] if g.get("estimates") else None)
        if ref:
            lo, hi = ref["ci_low"], ref["ci_high"]
            crosses = lo < 1.0 < hi
            wide = hi / lo > 3 if lo > 0 else True
            src = "exploratory pooled" if g.get("pooled") else "single-study"
            if crosses or wide:
                domains.append(_domain("Imprecision", 1, f"{src} {g['measure']} for {g['endpoint']} has 95% CI "
                                                         f"{lo:.2f}–{hi:.2f}" + (" (crosses no effect)" if crosses else " (wide)") + "."))
            else:
                domains.append(_domain("Imprecision", 0, f"{src} {g['measure']} for {g['endpoint']} 95% CI "
                                                         f"{lo:.2f}–{hi:.2f} excludes 1 (no effect) and is reasonably narrow."))
        else:
            domains.append(_domain("Imprecision", 0, "No usable interval.", assessed=False))
    elif len(sizes) >= 2 and total_n < 400:
        domains.append(_domain("Imprecision", 1, f"No effect estimates with CIs; reported sample sizes total only {total_n}."))
    else:
        domains.append(_domain("Imprecision", 0, "No effect estimate with a 95% CI was found in the retrieved passages.",
                               assessed=False))

    # 5. Publication bias: cannot be assessed from a handful of retrieved passages.
    domains.append(_domain("Publication bias", 0, "Requires a funnel plot or registry search over a complete set of "
                                                  "studies; not assessable from retrieved passages.", assessed=False))

    total_down = min(3, sum(d["downgrade"] for d in domains))
    level = max(1, start - total_down)
    return {
        "level": level,
        "label": LEVELS[level],
        "symbols": SYMBOLS[level],
        "start_level": LEVELS[start],
        "start_reason": start_reason,
        "domains": domains,
        "basis": {"passages": len(candidates), "randomised_or_synthesised": n_rand,
                  "reported_sample_sizes": sizes[:10], "total_reported_n": total_n},
        "disclaimer": "GRADE-inspired automated rating from measurable signals in the retrieved passages; "
                      "not a formal GRADE panel assessment.",
    }
