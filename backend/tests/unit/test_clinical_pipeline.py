"""Unit and Integration Test Suite for 12-Step Clinical Evidence Pipeline.

Verifies:
- Step 1: Clinical Intent Extraction
- Step 2 & 3: Evidence Population Matching & Hard Filters (+100 / -300 / Reject)
- Step 4: Evidence Hierarchy 11-level ranking
- Step 5 & 6: Evidence Grouping & Clustering
- Step 7: Contradiction Detection
- Step 8: Clinical Safety Validation
- Step 10 & 11: Citation Audit
- Step 12: Retrieval Failure Analysis
- Success Criteria: HER2-low rejection for HER2-positive query
"""

import pytest
from evidence_platform.modules.clinical.clinical_intent import ClinicalIntentExtractor, ClinicalQueryIntent
from evidence_platform.modules.clinical.population_matcher import PopulationMatchEvaluator, PaperPopulationMetadata
from evidence_platform.modules.clinical.evidence_hierarchy import EvidenceHierarchyRanker
from evidence_platform.modules.synthesis.evidence_grouper import EvidenceGrouper
from evidence_platform.modules.synthesis.contradiction_checker import ContradictionChecker
from evidence_platform.modules.synthesis.citation_auditor import CitationAuditor
from evidence_platform.modules.synthesis.validator import ClinicalSafetyValidator


class DummyCandidate:
    def __init__(self, chunk_id: str, pmcid: str, text: str, study_type: str):
        self.chunk_id = chunk_id
        self.pmcid = pmcid
        self.text = text
        self.study_type = study_type
        self.score = 1.0


def test_step_1_clinical_intent_extraction():
    extractor = ClinicalIntentExtractor()
    query = "Is trastuzumab deruxtecan effective in HER2-positive breast cancer?"
    intent = extractor.extract_intent(query)

    assert intent.disease == "Breast Cancer"
    assert intent.biomarker_status == "HER2-positive"
    assert "Trastuzumab deruxtecan" in intent.treatment
    assert "HER2-low" in intent.rejected_subgroups
    assert "Triple-negative" in intent.rejected_subgroups


def test_step_2_and_3_population_matching_and_hard_filters():
    extractor = ClinicalIntentExtractor()
    intent = extractor.extract_intent("Is trastuzumab deruxtecan effective in HER2-positive breast cancer?")

    evaluator = PopulationMatchEvaluator()

    # Candidate 1: Exact HER2-positive breast cancer paper
    cand1 = DummyCandidate(
        chunk_id="chk_01",
        pmcid="PMC1001",
        text="Phase III DESTINY-Breast03 trial demonstrated significant PFS improvement with trastuzumab deruxtecan in HER2-positive metastatic breast cancer.",
        study_type="rct"
    )

    # Candidate 2: HER2-low breast cancer paper (DESTINY-Breast04)
    cand2 = DummyCandidate(
        chunk_id="chk_02",
        pmcid="PMC9635907",
        text="DESTINY-Breast04 investigated trastuzumab deruxtecan in patients with HER2-low metastatic breast cancer.",
        study_type="rct"
    )

    # Candidate 3: Prostate cancer paper (Different disease)
    cand3 = DummyCandidate(
        chunk_id="chk_03",
        pmcid="PMC3003",
        text="Phase III trial evaluating docetaxel in metastatic prostate cancer.",
        study_type="rct"
    )

    included, excluded = evaluator.filter_and_score_candidates(intent, [cand1, cand2, cand3])

    # Verified expectations
    assert len(included) == 1
    assert included[0].pmcid == "PMC1001"
    assert getattr(included[0], "population_match_score") >= 200

    # HER2-low and Prostate cancer papers must be hard filtered out!
    assert len(excluded) == 2
    excl_pmcids = [e["pmcid"] for e in excluded]
    assert "PMC9635907" in excl_pmcids
    assert "PMC3003" in excl_pmcids


def test_step_4_evidence_hierarchy_ranking():
    ranker = EvidenceHierarchyRanker()

    guideline_cand = DummyCandidate("c1", "PMC1", "NCCN Clinical Practice Guidelines in Oncology recommend...", "guideline")
    rct_cand = DummyCandidate("c2", "PMC2", "Phase III randomized trial DESTINY-Breast03...", "rct")
    review_cand = DummyCandidate("c3", "PMC3", "Narrative review of HER2-targeted therapy...", "review")

    ranked = ranker.rank_candidates([review_cand, guideline_cand, rct_cand])

    assert ranked[0].pmcid == "PMC1"
    assert ranked[0].hierarchy_rank == 1
    assert ranked[1].pmcid == "PMC2"
    assert ranked[1].hierarchy_rank == 2
    assert ranked[2].pmcid == "PMC3"
    assert ranked[2].hierarchy_rank == 10


def test_step_6_and_7_evidence_grouping_and_contradictions():
    grouper = EvidenceGrouper()
    checker = ContradictionChecker()

    c1 = DummyCandidate("c1", "PMC1", "Phase III trial showed statistically significant PFS improvement.", "rct")
    setattr(c1, "paper_metadata", PaperPopulationMetadata(disease="Breast Cancer", biomarker="HER2-positive", treatment_line="First-line (1L)"))
    setattr(c1, "hierarchy_weight", 2.2)

    c2 = DummyCandidate("c2", "PMC2", "Meta-analysis showed no significant overall survival benefit.", "meta_analysis")
    setattr(c2, "paper_metadata", PaperPopulationMetadata(disease="Breast Cancer", biomarker="HER2-positive", treatment_line="First-line (1L)"))
    setattr(c2, "hierarchy_weight", 1.6)

    clusters = grouper.group_candidates([c1, c2])
    assert len(clusters) >= 1

    contradiction_res = checker.check_contradictions(clusters, [c1, c2])
    assert contradiction_res["has_contradiction"] is True
    assert contradiction_res["contradiction_count"] >= 1

    # Verify all explainable fields are present AND dynamically generated
    for item in contradiction_res["contradictions"]:
        # 1. Conflicting evidence IDs — must name the actual PMCIDs
        assert "conflicting_evidence_ids" in item
        assert len(item["conflicting_evidence_ids"]) >= 2
        for ev in item["conflicting_evidence_ids"]:
            assert "pmcid" in ev
            assert "study_type" in ev

        # 2. Claim in conflict — must quote actual findings from evidence text
        assert "claim_in_conflict" in item
        claim = item["claim_in_conflict"]
        assert len(claim) > 0
        # Must contain text derived from the actual evidence, not generic placeholder
        assert "Study A claims" not in claim, "claim_in_conflict must not use generic 'Study A/B' labels"

        # 3. Reason for difference — must be evidence-specific
        assert "reason_for_difference" in item
        reason = item["reason_for_difference"]
        assert len(reason) > 0
        assert "unspecified" not in reason.lower()

        # 4. Evidence quality — per-study
        assert "evidence_quality" in item
        assert len(item["evidence_quality"]) >= 2
        for eq in item["evidence_quality"]:
            assert "pmcid" in eq
            assert "study_type" in eq
            assert "strength_tier" in eq
            assert "hierarchy_weight" in eq

        # 5. Clinical interpretation — fully specific
        assert "clinical_interpretation" in item
        interp = item["clinical_interpretation"]
        assert len(interp) > 0

        # 6. Description — must be dynamic, naming studies/findings
        assert "description" in item
        desc = item["description"]
        assert len(desc) > 20, "Description must be a substantive evidence-specific sentence"
        # Must NOT contain banned generic phrases
        for banned in ["Conflicting efficacy findings across retrieved evidence",
                       "unspecified", "_unspecified_"]:
            assert banned not in desc, f"Description must not contain generic text: '{banned}'"

    # Verify the specific contradiction references the correct PMCIDs
    first_c = contradiction_res["contradictions"][0]
    pmcids = [e["pmcid"] for e in first_c["conflicting_evidence_ids"]]
    assert "PMC1" in pmcids
    assert "PMC2" in pmcids

    # Reason must identify the study design difference
    assert "rct" in first_c["reason_for_difference"].lower() or \
           "meta" in first_c["reason_for_difference"].lower() or \
           "randomized" in first_c["reason_for_difference"].lower()

    # Description must reference actual study identifiers, not generics
    desc = first_c["description"]
    assert "PMC1" in desc or "PMC2" in desc or "trial" in desc.lower()

    # Claim must quote from the actual evidence text
    claim = first_c["claim_in_conflict"]
    assert "significant" in claim.lower() or "improvement" in claim.lower() or "pfs" in claim.lower()


def test_contradiction_no_generic_messages():
    """Verify that no contradiction field ever contains generic/placeholder text."""
    checker = ContradictionChecker()

    c1 = DummyCandidate("c1", "PMC5001", "KEYNOTE-522 demonstrated statistically significant improvement in PFS with pembrolizumab.", "rct")
    setattr(c1, "paper_metadata", PaperPopulationMetadata(disease="Breast Cancer", biomarker="HER2-positive"))
    setattr(c1, "hierarchy_weight", 2.0)

    c2 = DummyCandidate("c2", "PMC5002", "This meta-analysis found no significant overall survival benefit for the combination regimen.", "meta_analysis")
    setattr(c2, "paper_metadata", PaperPopulationMetadata(disease="Breast Cancer", biomarker="HER2-positive"))
    setattr(c2, "hierarchy_weight", 1.5)

    result = checker.check_contradictions([], [c1, c2])
    assert result["has_contradiction"] is True

    banned_phrases = [
        "Conflicting efficacy findings across retrieved evidence",
        "Conflicting efficacy findings in population",
        "_unspecified_",
        "unspecified",
        "Study A claims",
        "Study B claims",
    ]

    for item in result["contradictions"]:
        for field in ["description", "claim_in_conflict", "reason_for_difference",
                      "clinical_interpretation", "action"]:
            value = item.get(field, "")
            for phrase in banned_phrases:
                assert phrase not in value, (
                    f"Field '{field}' contains banned generic text '{phrase}': {value}"
                )

        # Description must mention at least one PMCID or trial name
        desc = item["description"]
        assert any(x in desc for x in ["PMC5001", "PMC5002", "KEYNOTE"]), \
            f"Description must reference actual studies, got: {desc}"

        # Claim must reference actual evidence content
        claim = item["claim_in_conflict"]
        assert any(x in claim.lower() for x in ["pfs", "significant", "survival", "benefit", "pembrolizumab"]), \
            f"Claim must reference actual evidence findings, got: {claim}"


def test_contradiction_fallback_label():
    """When metadata is identical and no structural difference can be inferred,
    the interpretation must still be specific (naming studies) and flag manual review."""
    checker = ContradictionChecker()

    c1 = DummyCandidate("c1", "PMC100", "Treatment X showed statistically significant improvement.", "rct")
    setattr(c1, "paper_metadata", PaperPopulationMetadata(disease="Breast Cancer", biomarker="HER2-positive"))
    setattr(c1, "hierarchy_weight", 1.0)

    c2 = DummyCandidate("c2", "PMC101", "Treatment X did not improve outcomes in this population.", "rct")
    setattr(c2, "paper_metadata", PaperPopulationMetadata(disease="Breast Cancer", biomarker="HER2-positive"))
    setattr(c2, "hierarchy_weight", 1.0)

    result = checker.check_contradictions([], [c1, c2])
    assert result["has_contradiction"] is True

    for item in result["contradictions"]:
        interp = item["clinical_interpretation"]
        # Must flag manual review
        assert "manual review" in interp.lower(), \
            f"Expected manual review recommendation, got: {interp}"
        # Must still be specific — naming PMCIDs
        assert "PMC100" in interp or "PMC101" in interp, \
            f"Even fallback interpretation must name studies, got: {interp}"
        # Description must not be generic
        assert "unspecified" not in item["description"].lower()


def test_step_8_clinical_safety_validation():
    extractor = ClinicalIntentExtractor()
    intent = extractor.extract_intent("Is trastuzumab deruxtecan effective in HER2-positive breast cancer?")

    validator = ClinicalSafetyValidator()

    ev1 = DummyCandidate("c1", "PMC1001", "Trastuzumab deruxtecan in HER2-positive breast cancer.", "rct")
    setattr(ev1, "paper_metadata", PaperPopulationMetadata(disease="Breast Cancer", biomarker="HER2-positive", intervention="Trastuzumab deruxtecan"))

    ev2 = DummyCandidate("c2", "PMC9635907", "Trastuzumab deruxtecan in HER2-low breast cancer.", "rct")
    setattr(ev2, "paper_metadata", PaperPopulationMetadata(disease="Breast Cancer", biomarker="HER2-low", intervention="Trastuzumab deruxtecan"))

    evidence_map = {"E01": ev1, "E02": ev2}

    claims = [
        {"text": "T-DXd improves PFS in HER2-positive breast cancer.", "evidence_ids": ["E01"]},
        {"text": "T-DXd improves PFS in HER2-positive breast cancer.", "evidence_ids": ["E02"]},  # Mismatched ref E02 (HER2-low)
    ]

    validated_claims, rejected_cits = validator.validate_clinical_safety(intent, claims, evidence_map)

    assert len(rejected_cits) == 1
    assert rejected_cits[0]["evidence_id"] == "E02"
    assert "E02" not in validated_claims[1]["evidence_ids"]


def test_step_11_and_12_citation_audit_and_failure_analysis():
    auditor = CitationAuditor()
    extractor = ClinicalIntentExtractor()
    intent = extractor.extract_intent("Is trastuzumab deruxtecan effective in HER2-positive breast cancer?")

    c1 = DummyCandidate("c1", "PMC1001", "DESTINY-Breast03 in HER2-positive breast cancer", "rct")
    setattr(c1, "paper_metadata", PaperPopulationMetadata(disease="Breast Cancer", biomarker="HER2-positive"))

    excl = [{
        "pmcid": "PMC9635907",
        "chunk_id": "c2",
        "study_type": "rct",
        "paper_metadata": {"disease": "Breast Cancer", "biomarker": "HER2-low"},
        "exclusion_reason": "Biomarker population mismatch: query requires HER2-positive, paper is for HER2-low",
        "exclusion_stage": "Step 3 - Hard Filters",
    }]

    claims = [{"text": "T-DXd improves PFS.", "evidence_ids": ["E01"]}]

    audit, failure_analysis = auditor.audit_citations(intent, [c1], excl, claims)

    assert len(audit) == 2
    included_audit = [a for a in audit if a["supports_claim"] == "YES"]
    excluded_audit = [a for a in audit if a["supports_claim"] == "NO"]

    assert len(included_audit) == 1
    assert len(excluded_audit) == 1
    assert excluded_audit[0]["pmcid"] == "PMC9635907"

    assert len(failure_analysis) == 1
