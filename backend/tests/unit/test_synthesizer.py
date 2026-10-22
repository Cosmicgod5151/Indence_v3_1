import unittest
import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from evidence_platform.adapters.models.mock import MockLLMClient
from evidence_platform.modules.retrieval.retriever import RetrievalCandidate
from evidence_platform.modules.retrieval.context_assembler import AssembledContext
from evidence_platform.modules.synthesis.synthesizer import AnswerSynthesizer


class TestAnswerSynthesizer(unittest.TestCase):

    def setUp(self):
        self.llm_client = MockLLMClient()
        self.synthesizer = AnswerSynthesizer(self.llm_client)

        # Build some mock candidates and context
        self.candidate = RetrievalCandidate(
            chunk_id="chunk-101", qdrant_point_id="pt-101",
            text="Trastuzumab deruxtecan is highly effective in patients.",
            score=0.9, document_id="doc-1", pmcid="PMC101",
            study_type="rct", publication_date="2022", display_rights=True
        )

        self.assembled = AssembledContext(
            context_text="[Source: PMC101 | Study Type: rct]\nTrastuzumab deruxtecan is highly effective in patients.\n",
            selected_candidates=[self.candidate],
            total_tokens=25
        )

    def test_synthesis_success(self):
        res = asyncio.run(
            self.synthesizer.synthesize(
                query="Is trastuzumab deruxtecan effective?",
                assembled=self.assembled
            )
        )

        self.assertEqual(res.status, "answer")
        self.assertIsNone(res.abstention_reason)
        
        # Verify evidence mapping
        self.assertIn("E01", res.evidence_map)
        self.assertEqual(res.evidence_map["E01"].chunk_id, "chunk-101")

        # Verify claims are generated
        self.assertTrue(len(res.direct_claims) > 0)
        self.assertEqual(res.direct_claims[0]["evidence_ids"], ["E01"])

        # Verify synthesis sections are populated
        self.assertIsNotNone(res.recommendation)
        self.assertTrue(len(res.recommendation["text"]) > 0)
        self.assertIsNotNone(res.bottom_line)
        self.assertTrue(len(res.bottom_line["text"]) > 0)
        self.assertIsNotNone(res.clinical_context)
        self.assertIsNotNone(res.evidence_certainty)
        self.assertIn(res.evidence_certainty["recommendation_strength"], ["Strong", "Conditional"])
        self.assertIn(res.evidence_certainty["evidence_quality"], ["High", "Moderate", "Low", "Very Low"])
        self.assertTrue(len(res.supporting_evidence) > 0)

    def test_synthesis_abstain(self):
        # Trigger mock LLM abstention by putting "abstain" in query
        res = asyncio.run(
            self.synthesizer.synthesize(
                query="Please abstain from answering this.",
                assembled=self.assembled
            )
        )

        self.assertEqual(res.status, "insufficient_evidence")
        self.assertIsNotNone(res.abstention_reason)
        self.assertEqual(len(res.direct_claims), 0)

        # Verify synthesis sections are empty/None on abstention
        self.assertIsNone(res.recommendation)
        self.assertIsNone(res.clinical_reasoning)
        self.assertEqual(res.supporting_evidence, [])
        self.assertIsNone(res.alternatives)
        self.assertIsNone(res.bottom_line)
        self.assertIsNone(res.clinical_context)
        self.assertIsNone(res.evidence_certainty)


if __name__ == "__main__":
    unittest.main()
