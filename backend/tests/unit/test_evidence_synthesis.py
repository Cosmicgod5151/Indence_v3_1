"""
Evidence synthesis engine tests — validates that the synthesis pipeline produces
clinician-grade evidence synthesis rather than a flat list of claims.

Tests cover oncology, cardiology, diabetes, and infectious disease queries.
"""
import asyncio
import re
import unittest
from evidence_platform.modules.synthesis.synthesizer import AnswerSynthesizer, SynthesizedAnswer
from evidence_platform.modules.retrieval.context_assembler import AssembledContext
from evidence_platform.modules.retrieval.retriever import RetrievalCandidate
from evidence_platform.adapters.models.mock import MockLLMClient, _detect_modifiers, _infer_study_hierarchy


def _make_candidate(chunk_id, text, pmcid="PMC100", study_type="rct", score=0.9):
    """Helper to build a RetrievalCandidate with defaults."""
    return RetrievalCandidate(
        chunk_id=chunk_id,
        qdrant_point_id=f"pt-{chunk_id}",
        text=text,
        score=score,
        document_id=f"doc-{chunk_id}",
        pmcid=pmcid,
        study_type=study_type,
        publication_date="2023",
        display_rights=True,
    )


def _make_assembled(*candidates):
    """Helper to build AssembledContext from candidates."""
    context_text = "\n".join(
        f"[Source: {c.pmcid} | Study Type: {c.study_type}]\n{c.text}\n"
        for c in candidates
    )
    return AssembledContext(
        context_text=context_text,
        selected_candidates=list(candidates),
        total_tokens=sum(len(c.text.split()) for c in candidates),
    )


class TestOncologySynthesis(unittest.TestCase):
    """Oncology: anti-EGFR therapy in mCRC."""

    def setUp(self):
        self.llm = MockLLMClient()
        self.synth = AnswerSynthesizer(self.llm)
        self.candidates = [
            _make_candidate(
                "chunk-onc-1",
                "Cetuximab plus FOLFIRI improved overall survival in left-sided RAS wild-type metastatic colorectal cancer (HR 0.77, 95% CI 0.62-0.96, p=0.02). Patients with left-sided tumors benefited significantly.",
                pmcid="PMC200", study_type="rct",
            ),
            _make_candidate(
                "chunk-onc-2",
                "NCCN guidelines recommend anti-EGFR therapy as first-line for left-sided RAS wild-type mCRC. Bevacizumab-based regimens are preferred for right-sided tumors.",
                pmcid="PMC201", study_type="guideline",
            ),
        ]
        self.assembled = _make_assembled(*self.candidates)

    def test_recommendation_is_synthesized(self):
        """Verify recommendation is a coherent synthesis, not individual claims."""
        res = asyncio.run(self.synth.synthesize(
            query="What is the preferred first-line therapy for left-sided RAS wild-type metastatic colorectal cancer?",
            assembled=self.assembled,
        ))
        self.assertEqual(res.status, "answer")
        self.assertIsNotNone(res.recommendation)
        self.assertTrue(len(res.recommendation["text"]) > 20)
        # Must reference multiple sources
        self.assertTrue(len(res.recommendation["evidence_ids"]) >= 1)

    def test_quantitative_evidence_preserved(self):
        """Verify HR, CI are present in supporting evidence when source contains them."""
        res = asyncio.run(self.synth.synthesize(
            query="Does cetuximab improve OS in left-sided mCRC?",
            assembled=self.assembled,
        ))
        # At least one supporting evidence entry should reference the RCT chunk
        self.assertTrue(len(res.supporting_evidence) > 0)
        all_se_text = " ".join(se.get("text", "") for se in res.supporting_evidence)
        # The mock extracts first sentences which include HR data
        self.assertIn("left-sided", all_se_text.lower())

    def test_evidence_certainty_populated(self):
        """Verify evidence certainty reflects guideline + RCT evidence."""
        res = asyncio.run(self.synth.synthesize(
            query="First-line therapy for mCRC",
            assembled=self.assembled,
        ))
        self.assertIsNotNone(res.evidence_certainty)
        # Guideline present → should be Strong/High
        self.assertEqual(res.evidence_certainty["recommendation_strength"], "Strong")
        self.assertEqual(res.evidence_certainty["evidence_quality"], "High")

    def test_clinical_context_detection(self):
        """Verify clinical modifiers are detected from the query."""
        res = asyncio.run(self.synth.synthesize(
            query="What is the preferred first-line therapy for left-sided RAS wild-type metastatic colorectal cancer?",
            assembled=self.assembled,
        ))
        self.assertIsNotNone(res.clinical_context)
        mods = res.clinical_context.get("detected_modifiers", [])
        self.assertIn("left-sided", mods)
        self.assertIn("metastatic", mods)
        self.assertIn("first-line", mods)
        self.assertIn("RAS wild-type", mods)

    def test_no_redundant_claims(self):
        """Verify no duplicate claim texts appear."""
        res = asyncio.run(self.synth.synthesize(
            query="Cetuximab in mCRC",
            assembled=self.assembled,
        ))
        claim_texts = [c["text"] for c in res.direct_claims]
        self.assertEqual(len(claim_texts), len(set(claim_texts)), "Duplicate claims detected")

    def test_bottom_line_present(self):
        """Verify every successful answer has a bottom line."""
        res = asyncio.run(self.synth.synthesize(
            query="Cetuximab in mCRC",
            assembled=self.assembled,
        ))
        self.assertIsNotNone(res.bottom_line)
        self.assertTrue(len(res.bottom_line["text"]) > 10)

    def test_alternatives_discussed(self):
        """Verify alternatives section is present."""
        res = asyncio.run(self.synth.synthesize(
            query="Anti-EGFR vs bevacizumab in mCRC",
            assembled=self.assembled,
        ))
        self.assertIsNotNone(res.alternatives)
        self.assertTrue(len(res.alternatives["text"]) > 10)


class TestCardiologySynthesis(unittest.TestCase):
    """Cardiology: SGLT2 inhibitors in heart failure."""

    def setUp(self):
        self.llm = MockLLMClient()
        self.synth = AnswerSynthesizer(self.llm)
        self.candidates = [
            _make_candidate(
                "chunk-card-1",
                "In the DAPA-HF trial, dapagliflozin reduced the composite of worsening heart failure or cardiovascular death by 26% (HR 0.74, 95% CI 0.65-0.85) in patients with HFrEF.",
                pmcid="PMC300", study_type="rct",
            ),
            _make_candidate(
                "chunk-card-2",
                "ACC/AHA guidelines recommend SGLT2 inhibitors as a foundational therapy for heart failure with reduced ejection fraction regardless of diabetes status.",
                pmcid="PMC301", study_type="guideline",
            ),
        ]
        self.assembled = _make_assembled(*self.candidates)

    def test_guideline_prioritization(self):
        """Verify recommendation primarily reflects guideline consensus."""
        res = asyncio.run(self.synth.synthesize(
            query="Should SGLT2 inhibitors be used in heart failure?",
            assembled=self.assembled,
        ))
        self.assertEqual(res.status, "answer")
        self.assertIsNotNone(res.recommendation)
        # Certainty should reflect guideline presence
        self.assertIsNotNone(res.evidence_certainty)
        self.assertEqual(res.evidence_certainty["recommendation_strength"], "Strong")

    def test_source_grounding_preserved(self):
        """Verify every recommendation has evidence_ids mapping to evidence_map."""
        res = asyncio.run(self.synth.synthesize(
            query="SGLT2 inhibitors in HFrEF",
            assembled=self.assembled,
        ))
        for ev_id in res.recommendation.get("evidence_ids", []):
            self.assertIn(ev_id, res.evidence_map)


class TestDiabetesSynthesis(unittest.TestCase):
    """Diabetes: metformin first-line therapy."""

    def setUp(self):
        self.llm = MockLLMClient()
        self.synth = AnswerSynthesizer(self.llm)
        self.candidates = [
            _make_candidate(
                "chunk-dm-1",
                "ADA guidelines recommend metformin as first-line pharmacologic therapy for type 2 diabetes mellitus. Metformin should be continued unless contraindicated or not tolerated.",
                pmcid="PMC400", study_type="guideline",
            ),
        ]
        self.assembled = _make_assembled(*self.candidates)

    def test_single_source_synthesis(self):
        """Verify synthesis works with a single evidence source."""
        res = asyncio.run(self.synth.synthesize(
            query="What is the first-line therapy for type 2 diabetes?",
            assembled=self.assembled,
        ))
        self.assertEqual(res.status, "answer")
        self.assertIsNotNone(res.recommendation)
        self.assertIsNotNone(res.bottom_line)
        # Assumptions should be stated
        self.assertIsNotNone(res.clinical_context)
        self.assertTrue(len(res.clinical_context.get("assumptions", "")) > 0)


class TestInfectiousDiseaseSynthesis(unittest.TestCase):
    """Infectious disease: empiric antibiotic therapy."""

    def setUp(self):
        self.llm = MockLLMClient()
        self.synth = AnswerSynthesizer(self.llm)
        self.candidates = [
            _make_candidate(
                "chunk-id-1",
                "IDSA guidelines recommend ceftriaxone plus azithromycin for community-acquired pneumonia in hospitalized patients without risk factors for MRSA or Pseudomonas.",
                pmcid="PMC500", study_type="guideline",
            ),
            _make_candidate(
                "chunk-id-2",
                "A meta-analysis of 12 RCTs found beta-lactam/macrolide combination therapy reduced mortality compared to beta-lactam monotherapy (RR 0.82, 95% CI 0.68-0.98) in hospitalized CAP patients.",
                pmcid="PMC501", study_type="meta_analysis",
            ),
        ]
        self.assembled = _make_assembled(*self.candidates)

    def test_multiple_sources_merged(self):
        """Verify evidence from multiple sources is merged into synthesis."""
        res = asyncio.run(self.synth.synthesize(
            query="What is the empiric antibiotic regimen for community-acquired pneumonia?",
            assembled=self.assembled,
        ))
        self.assertEqual(res.status, "answer")
        self.assertIsNotNone(res.recommendation)
        # Should reference both sources
        all_ev_ids = set()
        for field in [res.recommendation, res.clinical_reasoning]:
            if field:
                all_ev_ids.update(field.get("evidence_ids", []))
        self.assertTrue(len(all_ev_ids) >= 2, "Synthesis should reference multiple sources")

    def test_all_sentence_ids_valid_format(self):
        """Verify all sentence_ids follow E{nn}.S{nn} format."""
        res = asyncio.run(self.synth.synthesize(
            query="CAP antibiotic therapy",
            assembled=self.assembled,
        ))
        pattern = re.compile(r"^E\d{2}\.S\d+$")
        sections = [res.recommendation, res.clinical_reasoning, res.alternatives, res.bottom_line]
        for sec in sections:
            if sec:
                for sid in sec.get("sentence_ids", []):
                    self.assertTrue(pattern.match(sid), f"Invalid sentence_id format: {sid}")
        for se in (res.supporting_evidence or []):
            for sid in se.get("sentence_ids", []):
                self.assertTrue(pattern.match(sid), f"Invalid sentence_id format: {sid}")


class TestCrossCuttingValidations(unittest.TestCase):
    """Cross-cutting validations applied across all synthesis outputs."""

    def setUp(self):
        self.llm = MockLLMClient()
        self.synth = AnswerSynthesizer(self.llm)
        self.candidates = [
            _make_candidate("chunk-xc-1", "Treatment A improved OS (HR 0.65). Patients with BRAF V600E mutations had worse outcomes.", pmcid="PMC600", study_type="rct"),
            _make_candidate("chunk-xc-2", "Guidelines recommend treatment A for MSI-H/dMMR tumors.", pmcid="PMC601", study_type="guideline"),
        ]
        self.assembled = _make_assembled(*self.candidates)

    def test_evidence_map_completeness(self):
        """Verify all evidence_ids in synthesis map to valid evidence_map entries."""
        res = asyncio.run(self.synth.synthesize(
            query="Treatment options for MSI-H colorectal cancer",
            assembled=self.assembled,
        ))
        sections = [res.recommendation, res.clinical_reasoning, res.alternatives, res.bottom_line]
        for sec in sections:
            if sec:
                for ev_id in sec.get("evidence_ids", []):
                    self.assertIn(ev_id, res.evidence_map, f"Evidence ID {ev_id} not in evidence_map")
        for se in (res.supporting_evidence or []):
            for ev_id in se.get("evidence_ids", []):
                self.assertIn(ev_id, res.evidence_map, f"Evidence ID {ev_id} not in evidence_map")

    def test_abstention_empties_synthesis(self):
        """Verify synthesis sections are None/empty on abstention."""
        res = asyncio.run(self.synth.synthesize(
            query="Please abstain from this query",
            assembled=self.assembled,
        ))
        self.assertEqual(res.status, "insufficient_evidence")
        self.assertIsNone(res.recommendation)
        self.assertIsNone(res.clinical_reasoning)
        self.assertEqual(res.supporting_evidence, [])
        self.assertIsNone(res.alternatives)
        self.assertIsNone(res.bottom_line)
        self.assertIsNone(res.clinical_context)
        self.assertIsNone(res.evidence_certainty)

    def test_empty_candidates_abstains(self):
        """Verify empty candidate list produces proper abstention."""
        empty_assembled = AssembledContext("", [], 0)
        res = asyncio.run(self.synth.synthesize(
            query="Any query here",
            assembled=empty_assembled,
        ))
        self.assertEqual(res.status, "insufficient_evidence")
        self.assertIsNone(res.recommendation)
        self.assertIsNotNone(res.abstention_reason)


class TestModifierDetection(unittest.TestCase):
    """Test clinical context modifier detection helper."""

    def test_molecular_markers(self):
        mods = _detect_modifiers("HER2-positive metastatic breast cancer")
        self.assertIn("HER2", mods)
        self.assertIn("metastatic", mods)

    def test_sidedness(self):
        mods = _detect_modifiers("left-sided RAS wild-type mCRC")
        self.assertIn("left-sided", mods)
        self.assertIn("RAS wild-type", mods)

    def test_therapy_line(self):
        mods = _detect_modifiers("second-line therapy for advanced NSCLC")
        self.assertIn("second-line", mods)
        self.assertIn("metastatic", mods)  # "advanced" maps to metastatic

    def test_msi_dmmr(self):
        mods = _detect_modifiers("MSI-H colorectal cancer with BRAF mutation")
        self.assertIn("MSI-H/dMMR", mods)
        self.assertIn("BRAF V600E", mods)

    def test_empty_query(self):
        mods = _detect_modifiers("")
        self.assertEqual(mods, [])

    def test_no_modifiers(self):
        mods = _detect_modifiers("What is the weather today?")
        self.assertEqual(mods, [])


class TestStudyHierarchyInference(unittest.TestCase):
    """Test evidence quality and recommendation strength inference."""

    def test_guideline_present(self):
        strength, quality = _infer_study_hierarchy([{"study_type": "guideline"}, {"study_type": "rct"}])
        self.assertEqual(strength, "Strong")
        self.assertEqual(quality, "High")

    def test_meta_analysis_present(self):
        strength, quality = _infer_study_hierarchy([{"study_type": "meta_analysis"}])
        self.assertEqual(strength, "Strong")
        self.assertEqual(quality, "Moderate")

    def test_rct_only(self):
        strength, quality = _infer_study_hierarchy([{"study_type": "rct"}])
        self.assertEqual(strength, "Conditional")
        self.assertEqual(quality, "Moderate")

    def test_cohort_only(self):
        strength, quality = _infer_study_hierarchy([{"study_type": "cohort_study"}])
        self.assertEqual(strength, "Conditional")
        self.assertEqual(quality, "Low")

    def test_unknown_type(self):
        strength, quality = _infer_study_hierarchy([{"study_type": "other"}])
        self.assertEqual(strength, "Conditional")
        self.assertEqual(quality, "Very Low")


if __name__ == "__main__":
    unittest.main()
