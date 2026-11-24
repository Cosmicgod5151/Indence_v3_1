"""Tests for the Patient Safety Check (curated layer; AI screen disabled)."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from evidence_platform.modules.clinical.safety_check import run_safety_check, proposed_drugs


def _run(facts, question, answers=(), monkeypatch=None):
    return asyncio.run(run_safety_check(facts, question, list(answers)))


def _cats(res):
    return {(f["category"], f["proposed"], f["severity"]) for f in res["flags"]}


def test_treatment_history_same_drug_and_same_class(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    facts = [{"field": "prior_therapy", "value": "letrozole plus palbociclib for 22 months"}]
    res = _run(facts, "What next after CDK4/6 inhibitor?", ["Consider palbociclib rechallenge or abemaciclib."])
    cats = _cats(res)
    assert ("treatment_history", "palbociclib", "major") in cats       # same agent
    assert ("treatment_history", "abemaciclib", "moderate") in cats    # same class


def test_curated_interactions_comorbidity_and_organ_function(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    facts = [{"field": "medication", "value": "omeprazole 20 mg daily"},
             {"field": "medication", "value": "levothyroxine"},
             {"field": "comorbidity", "value": "hypothyroidism"},
             {"field": "lab", "name": "eGFR", "value": "52 mL/min"},
             {"field": "lab", "name": "LVEF", "value": "45%"}]
    res = _run(facts, "Options?", ["Erlotinib, cisplatin-based chemotherapy, pembrolizumab or trastuzumab could be considered."])
    cats = {(f["category"], f["proposed"]) for f in res["flags"]}
    assert ("drug_interaction", "erlotinib") in cats           # PPI reduces absorption
    assert ("organ_function", "cisplatin") in cats             # eGFR < 60
    assert ("comorbidity", "pembrolizumab") in cats            # thyroid monitoring
    assert ("organ_function", "trastuzumab") in cats           # LVEF < 50
    assert all(f["method"] in ("curated_rule", "treatment_history") for f in res["flags"])


def test_no_false_alarm_for_osimertinib_with_ppi(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    res = _run([{"field": "medication", "value": "pantoprazole"}], "Is osimertinib appropriate?")
    assert not [f for f in res["flags"] if f["category"] == "drug_interaction"]


def test_proposed_drug_detection():
    d = proposed_drugs("Should she get trastuzumab deruxtecan?", "Alternatives include tucatinib plus capecitabine.")
    assert {"trastuzumab deruxtecan", "tucatinib", "capecitabine"} <= set(d)
