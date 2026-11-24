"""Tests for the Quantitative Evidence Map (effect extraction + heterogeneity)."""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from evidence_platform.modules.synthesis.effect_map import (
    extract_effects, heterogeneity, build_evidence_map, EffectEstimate,
)


def _one(text):
    r = extract_effects(text)
    assert len(r) == 1, r
    return r[0]


def test_extracts_common_reporting_formats():
    e = _one("Osimertinib improved overall survival (HR 0.80; 95% CI, 0.64 to 1.00; P = 0.046).")
    assert (e.measure, e.estimate, e.ci_low, e.ci_high) == ("HR", 0.80, 0.64, 1.00)
    assert e.endpoint == "Overall survival" and e.p_value == "=0.046"
    e = _one("HR 1·13 [95% CI 0·94–1·35]; log-rank p=0·20")          # Lancet mid-dot decimals
    assert (e.estimate, e.ci_low, e.ci_high) == (1.13, 0.94, 1.35)
    e = _one("(HR [95 % CI]: 0.85 [0.71; 1.01], p = 0.065)")
    assert (e.estimate, e.ci_low, e.ci_high) == (0.85, 0.71, 1.01)
    e = _one("(OR=0.86, 95% CI=[0.50–1.48],")
    assert e.measure == "OR"
    e = _one("relative risk: 2.057, 95% CI [0.645 to 6.50], P=0.360")
    assert e.measure == "RR" and e.ci_high == 6.50


def test_quote_is_the_exact_source_span():
    text = "Median PFS improved (HR 0.62, 95% CI 0.51-0.75, p<0.001) in arm A."
    e = _one(text)
    assert text[e.span_start:e.span_end] == e.quote
    assert e.quote.startswith("HR 0.62")


def test_rejects_non_estimates():
    assert extract_effects("the pooled OR rate was 26% (95% CI, 23–29%, I") == []   # percentages
    assert extract_effects("Number of studies | HR | 95% CI | P value") == []        # table header
    assert extract_effects("stage II or 3 patients (95% CI 1-4)") == []              # English 'or'
    assert extract_effects("HR 0.50 (95% CI 0.60-0.90)") == []                        # estimate outside CI


def test_endpoint_prefers_specific_term():
    assert _one("overall mortality (HR 0.80, 95% CI 0.72–0.90)").endpoint == "Overall survival"


def _est(hr, lo, hi, pmcid):
    return EffectEstimate("HR", hr, lo, hi, "Overall survival", None, "", 0, 0, pmcid=pmcid)


def test_heterogeneity_matches_hand_calculation():
    ests = [_est(0.70, 0.60, 0.82, "A"), _est(0.75, 0.62, 0.91, "B"), _est(1.10, 0.95, 1.27, "C")]
    h = heterogeneity(ests)
    ys = [math.log(e.estimate) for e in ests]
    ses = [(math.log(e.ci_high) - math.log(e.ci_low)) / (2 * 1.959963984540054) for e in ests]
    ws = [1 / s ** 2 for s in ses]
    yf = sum(w * y for w, y in zip(ws, ys)) / sum(ws)
    q = sum(w * (y - yf) ** 2 for w, y in zip(ws, ys))
    assert abs(h["q"] - round(q, 3)) < 1e-6
    assert abs(h["i2"] - round(max(0, (q - 2) / q) * 100, 1)) < 1e-6
    assert h["pooled"] is not None and h["pooled"]["ci_low"] < h["pooled"]["estimate"] < h["pooled"]["ci_high"]


def test_no_pooling_below_three_studies_and_identical_studies_have_zero_i2():
    two = heterogeneity([_est(0.7, 0.6, 0.8, "A"), _est(0.7, 0.6, 0.8, "B")])
    assert two["pooled"] is None and two["i2"] == 0.0


def test_evidence_map_flags_opposite_significant_effects_once_per_paper():
    cands = [
        SimpleNamespace(chunk_id="c1", pmcid="P1", title="t1", publication_date="2021", study_type="rct",
                        text="Overall survival favoured arm A (HR 0.60, 95% CI 0.50-0.72). OS HR 0.61, 95% CI 0.50-0.74."),
        SimpleNamespace(chunk_id="c2", pmcid="P2", title="t2", publication_date="2022", study_type="rct",
                        text="Overall survival was worse (HR 1.40, 95% CI 1.15-1.70)."),
    ]
    m = build_evidence_map(cands, {"c1": "E01"})
    g = m["groups"][0]
    assert g["endpoint"] == "Overall survival" and len(g["estimates"]) == 2   # one estimate per paper
    assert "OPPOSITE" in (g["conflict"] or "")
    assert m["quantitative_conflicts"]
    assert g["estimates"][0]["evidence_id"] == "E01"


def test_prognostic_estimates_never_conflict_with_treatment_effects():
    cands = [
        SimpleNamespace(chunk_id="c1", pmcid="P1", title="", publication_date="2021", study_type="rct",
                        text="Progression-free survival was longer with osimertinib versus comparator (HR 0.62, 95% CI 0.43-0.89)."),
        SimpleNamespace(chunk_id="c2", pmcid="P2", title="", publication_date="2022", study_type="other",
                        text="In multivariate analysis, high expression was independently associated with shorter PFS (HR 2.66, 95% CI 1.58-4.49)."),
    ]
    m = build_evidence_map(cands)
    assert m["quantitative_conflicts"] == []
    ctx = {g["context"] for g in m["groups"]}
    assert ctx == {"treatment_comparison", "prognostic_association"}
