"""Regression tests for defects found in the post-Milestone-1 code audit.

Each test pins one bug that previously produced wrong clinical behaviour.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from evidence_platform.modules.clinical.clinical_intent import ClinicalIntentExtractor
from evidence_platform.modules.clinical.population_matcher import PopulationMatchEvaluator
from evidence_platform.modules.clinical.evidence_hierarchy import EvidenceHierarchyRanker
from evidence_platform.modules.retrieval.normalization.normalizer import QueryNormalizer
from evidence_platform.modules.synthesis.contradiction_checker import ContradictionChecker
from evidence_platform.modules.synthesis.abstention import AbstentionLayer
from evidence_platform.adapters.models.openrouter import llm_unavailable_answer

extractor = ClinicalIntentExtractor()
normalizer = QueryNormalizer()


# ---- Biomarker detection ---------------------------------------------------

def test_her2_plus_symbol_is_detected():
    intent = extractor.extract_intent("Is T-DXd effective in HER2+ metastatic breast cancer?")
    assert intent.biomarker_status == "HER2-positive"


def test_her2_positive_is_never_its_own_rejected_subgroup():
    intent = extractor.extract_intent("Trastuzumab in HER2-positive breast cancer")
    assert intent.biomarker_status == "HER2-positive"
    assert "HER2-positive" not in intent.rejected_subgroups
    assert "HER2-low" in intent.rejected_subgroups


def test_hr_plus_her2_minus_combination():
    intent = extractor.extract_intent("First-line therapy for HR+/HER2- metastatic breast cancer")
    found = {intent.biomarker_status, *intent.secondary_biomarkers}
    assert {"HR-positive", "HER2-negative"} <= found
    assert "HR-positive" not in intent.rejected_subgroups


def test_egfr_plus_and_alk_plus_detected():
    assert extractor.extract_intent("osimertinib in EGFR+ NSCLC").biomarker_status == "EGFR-mutant"
    assert extractor.extract_intent("alectinib in ALK+ NSCLC").biomarker_status == "ALK-positive"


def test_paper_metadata_her2_low_not_misread_as_negative():
    meta = PopulationMatchEvaluator().extract_paper_metadata(
        "Patients with HER2-low metastatic breast cancer were enrolled.", "rct"
    )
    assert meta.biomarker == "HER2-low"


# ---- Query normalisation -----------------------------------------------------

def test_common_words_are_not_expanded():
    q = normalizer.normalize("osimertinib or chemotherapy in nsclc").rewritten_query
    assert "odds ratio" not in q
    assert "non-small cell lung cancer" in q


def test_hr_plus_is_receptor_status_not_hazard_ratio():
    q = normalizer.normalize("HR+ breast cancer").rewritten_query
    assert "hazard ratio" not in q
    assert "hormone receptor positive" in q


def test_uppercase_acronym_still_expands_once():
    q = normalizer.normalize("Does T-DXd improve OS? OS data").rewritten_query
    assert q.count("overall survival") == 2  # once per occurrence, never nested
    assert "(overall survival) (overall survival)" not in q


# ---- Evidence hierarchy --------------------------------------------------------

def test_rct_without_phase_is_not_labelled_phase_iii():
    rank, _, label = EvidenceHierarchyRanker().classify_and_rank("rct", "A randomised trial of drug X.")
    assert rank == 2 and "Phase III" not in label


def test_phase_ii_rct_ranks_below_phase_iii():
    r2, _, _ = EvidenceHierarchyRanker().classify_and_rank("rct", "This phase II randomized trial ...")
    r3, _, _ = EvidenceHierarchyRanker().classify_and_rank("rct", "This phase III randomized trial ...")
    assert r3 < r2


def test_word_recommendation_alone_is_not_a_guideline():
    rank, _, _ = EvidenceHierarchyRanker().classify_and_rank("", "Our recommendation is further study.")
    assert rank != 1


# ---- Contradictions ------------------------------------------------------------

def _cand(cid, pmcid, text):
    return SimpleNamespace(chunk_id=cid, pmcid=pmcid, text=text, study_type="rct",
                           hierarchy_rank=2, hierarchy_weight=2.2, hierarchy_label="RCT")


def test_single_paper_cannot_contradict_itself():
    one = _cand("c1", "PMC1", "Drug X significantly improved PFS; however OS benefit was not significant.")
    result = ContradictionChecker().check_contradictions([], [one])
    assert result["has_contradiction"] is False


# ---- Abstention / LLM failure ------------------------------------------------------

def test_low_quality_flag_reads_candidate_rank():
    layer = AbstentionLayer(entailment_client=None)
    good = [_cand("a", "P1", "t"), _cand("b", "P2", "t")]
    passed, _, all_low = layer.population_quality_check(good, intent=object())
    assert passed and all_low is False


def test_json_parser_ignores_trailing_text_and_fences():
    from evidence_platform.adapters.models.openrouter import OpenRouterLLMClient
    parse = OpenRouterLLMClient._parse_json
    assert parse('```json\n{"status": "answer"}\n```') == {"status": "answer"}
    assert parse('{"status": "answer", "x": {"y": 1}}\n{"extra": true}') == {"status": "answer", "x": {"y": 1}}
    assert parse('Here is the result: {"a": "}"} trailing') == {"a": "}"}


def test_llm_failure_is_an_explicit_abstention_not_a_mock_answer():
    res = llm_unavailable_answer("HTTP 503", "some/model")
    assert res["status"] == "insufficient_evidence"
    assert res["recommendation"] is None and res["direct_answer_claims"] == []
    assert "unavailable" in res["abstention_reason"]


# ---- Free-model fallback chain & batched NLI ------------------------------------

def test_model_chain_parsing_keeps_free_suffix_and_providers():
    from evidence_platform.adapters.models.openrouter import parse_model_chain
    chain = parse_model_chain("baidu/ernie-5.5-flash-preview:free, gemini:gemini-2.5-flash ,nvidia/nemotron-3-ultra-550b-a55b:free")
    assert chain == [
        ("openrouter", "baidu/ernie-5.5-flash-preview:free"),
        ("gemini", "gemini-2.5-flash"),
        ("openrouter", "nvidia/nemotron-3-ultra-550b-a55b:free"),
    ]


def test_abstention_uses_one_batched_nli_call():
    import asyncio

    class BatchNLI:
        def __init__(self):
            self.calls = 0

        async def check_entailment_batch(self, pairs):
            self.calls += 1
            return ["entailment"] * len(pairs)

    nli = BatchNLI()
    layer = AbstentionLayer(nli)
    ev = {"E01": _cand("a", "P1", "Drug X improved PFS."), "E02": _cand("b", "P2", "Drug X improved OS.")}
    claims = [{"text": "Drug X improved PFS.", "evidence_ids": ["E01"]},
              {"text": "Drug X improved OS.", "evidence_ids": ["E02"]},
              {"text": "Drug X is safe.", "evidence_ids": ["E01", "E02"]}]
    status, out, _ = asyncio.run(layer.verify_and_calibrate("answer", claims, ev))
    assert nli.calls == 1
    assert status == "answer" and all(c["entailment_status"] == "entailment" for c in out)


# ---- Pre-generation safety gates -------------------------------------------------

def test_scope_and_entity_gates():
    from evidence_platform.modules.pipeline.query_pipeline import out_of_scope, unknown_entities
    from evidence_platform.modules.clinical.clinical_intent import ClinicalIntentExtractor
    ex = ClinicalIntentExtractor()
    for q in ["What is the capital of France?", "Write a poem about the ocean", "Best gaming laptop under 1000 dollars?"]:
        assert out_of_scope(q, ex.extract_intent(q)), q
    for q in ["Is cetuximab beneficial in RAS wild-type colorectal cancer?", "Does adjuvant radiotherapy reduce recurrence?",
              "What is the toxicity of sotorasib?"]:
        assert not out_of_scope(q, ex.extract_intent(q)), q
    passages = [_cand("a", "P1", "Trastuzumab deruxtecan improved PFS in HER2-positive breast cancer.")]
    assert unknown_entities("Is zorvatinib effective in HER2-positive breast cancer?", passages) == ["zorvatinib"]
    assert unknown_entities("Is trastuzumab effective in HER2-positive breast cancer?", passages) == []
