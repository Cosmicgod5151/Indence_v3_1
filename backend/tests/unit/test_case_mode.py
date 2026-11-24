"""Tests for Patient Case Mode: PHI guard, traceable extraction, conflicts,
missing data, case query construction and the applicability matrix."""
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "backend" / "src"))

from evidence_platform.modules.clinical.phi_guard import scan_phi
from evidence_platform.modules.clinical.case_context import (
    verify_facts, flag_conflicts, missing_data, build_case_query, extract_case_facts, locate_quote,
)
from evidence_platform.modules.clinical.applicability import assess_study, _patient_profile

CASES = json.loads((ROOT / "evals" / "cases" / "synthetic_cases.json").read_text(encoding="utf-8"))


def test_phi_guard_blocks_identifiers_and_passes_clinical_text():
    blocked = scan_phi(CASES["phi_demo"]["text"])
    kinds = {f["kind"] for f in blocked["findings"]}
    assert blocked["blocked"]
    assert {"Patient name field", "Medical record / hospital number", "Phone number"} <= kinds
    for case in CASES["cases"]:
        assert scan_phi(case["text"])["blocked"] is False, case["id"]


def test_untraceable_or_altered_facts_are_dropped():
    text = "PD-L1 22C3 TPS 80%. ECOG 0. EGFR wild-type."
    raw = [
        {"field": "biomarker", "name": "PD-L1", "value": "PD-L1 TPS 80%", "source_quote": "PD-L1 22C3 TPS 80%"},
        {"field": "biomarker", "name": "PD-L1", "value": "PD-L1 TPS 90%", "source_quote": "PD-L1 22C3 TPS 80%"},  # altered number
        {"field": "performance_status", "value": "ECOG 1", "source_quote": "ECOG 1"},                             # not in text
        {"field": "favourite_colour", "value": "blue", "source_quote": "EGFR"},                                    # unknown field
    ]
    kept, dropped = verify_facts(text, raw)
    assert [f.value for f in kept] == ["PD-L1 TPS 80%"]
    assert len(dropped) == 3
    assert kept[0].start == text.index("PD-L1 22C3")


def test_quote_location_tolerates_whitespace_and_case():
    text = "Molecular testing:  EGFR   exon 19\ndeletion detected."
    s, e = locate_quote(text, "egfr exon 19 deletion")
    assert text[s:e].lower().split() == ["egfr", "exon", "19", "deletion"]


def test_conflicting_biomarker_values_are_flagged_not_resolved():
    text = CASES["cases"][1]["text"]
    raw = [
        {"field": "biomarker", "name": "PD-L1", "value": "TPS 80%", "source_quote": "TPS 80%"},
        {"field": "biomarker", "name": "PD-L1", "value": "TPS 45%", "source_quote": "TPS 45%"},
    ]
    kept, _ = verify_facts(text, raw)
    conflicts = flag_conflicts(kept)
    assert len(conflicts) == 1 and len(conflicts[0]["values"]) == 2
    assert kept[0].conflict_with == [1] and kept[1].conflict_with == [0]


def test_missing_decision_critical_variables_are_listed():
    miss = {m["variable"] for m in missing_data("Non-Small Cell Lung Cancer", [], "stage IV NSCLC, ECOG 1, PD-L1 TPS 60%")}
    assert {"EGFR", "ALK", "ROS1", "KRAS"} <= miss
    assert "PD-L1" not in miss and "Performance status" not in miss


def test_regex_fallback_when_no_llm_is_available():
    case = CASES["cases"][0]
    out = asyncio.run(extract_case_facts(case["text"], llm_client=None))
    assert out["extraction_method"] == "regex-fallback"
    assert out["traceability"]["rate"] == 1.0          # every fallback fact is anchored
    values = " ".join(f["value"].lower() for f in out["facts"])
    assert "egfr exon 19" in values and "ecog" in values
    assert out["disease"] == "Non-Small Cell Lung Cancer"


def test_case_query_adds_context_and_line_of_therapy():
    facts = [
        {"field": "cancer_type", "value": "metastatic breast cancer"},
        {"field": "biomarker", "value": "HER2 positive (IHC 3+)"},
        {"field": "prior_therapy", "value": "trastuzumab, pertuzumab and docetaxel"},
    ]
    q, intent = build_case_query("What is the best second-line option?", facts)
    assert "HER2 positive" in q and "previously treated" in q
    assert intent.disease == "Breast Cancer" and intent.biomarker_status == "HER2-positive"
    assert intent.line_of_therapy


def test_applicability_matrix_match_and_differs():
    profile = _patient_profile([
        {"field": "cancer_type", "value": "breast cancer"},
        {"field": "biomarker", "value": "HER2 positive (IHC 3+)"},
        {"field": "stage", "value": "metastatic"},
        {"field": "performance_status", "value": "ECOG 2"},
    ])
    same = assess_study("Patients with HER2-positive metastatic breast cancer and ECOG 0-1 were enrolled.", "", profile)
    dims = {c["dimension"]: c["status"] for c in same["cells"]}
    assert dims["Disease"] == "match" and dims["Biomarker (HER2)"] == "match" and dims["Stage"] == "match"
    assert dims["Performance status"] == "differs"          # patient ECOG 2 vs trial 0-1
    other = assess_study("Patients with HER2-low metastatic breast cancer were enrolled.", "", profile)
    assert {c["dimension"]: c["status"] for c in other["cells"]}["Biomarker (HER2)"] == "differs"
    assert other["verdict"].startswith("Different population")


def test_treatment_naive_is_not_a_prior_therapy():
    facts = [{"field": "cancer_type", "value": "non-small cell lung cancer"},
             {"field": "prior_therapy", "value": "Treatment-naive"}]
    q, intent = build_case_query("Is pembrolizumab appropriate?", facts)
    assert "previously treated" not in q and intent.line_of_therapy == "First-line (1L)"
    prof = _patient_profile(facts)
    assert prof["priors"] == [] and prof["treatment_naive"] is True
    row = assess_study("Previously untreated patients with NSCLC received first-line pembrolizumab.", "", prof)
    assert {c["dimension"]: c["status"] for c in row["cells"]}["Line of therapy"] == "match"
