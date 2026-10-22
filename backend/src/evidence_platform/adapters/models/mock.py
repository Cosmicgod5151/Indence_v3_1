import hashlib
import re
from typing import Any
from ...ports.models import EmbeddingClient, RerankerClient, LLMClient, EntailmentClient

class MockEmbeddingClient(EmbeddingClient):
    """Deterministic mock embedding client for testing and local development."""

    def _hash_text(self, text: str, dim: int = 1024) -> list[float]:
        # Generate a deterministic vector using MD5 hash
        h = hashlib.md5(text.encode("utf-8")).digest()
        vector = []
        for i in range(dim):
            # Deterministic float between -1.0 and 1.0
            byte_val = h[i % len(h)]
            val = (byte_val / 127.5) - 1.0
            vector.append(round(val, 4))
        return vector

    def _generate_sparse(self, text: str) -> dict[str, Any]:
        # Extract alphanumeric words and generate mock sparse weights
        words = re.findall(r"\w+", text.lower())
        indices = []
        values = []
        
        seen = set()
        for word in words:
            if word in seen:
                continue
            seen.add(word)
            # Map word to a deterministic index between 1 and 100000
            word_hash = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            idx = (word_hash % 100000) + 1
            indices.append(idx)
            # Mock weight based on length and frequency
            weight = min(1.0, len(word) * 0.1)
            values.append(round(weight, 4))
            
        # Ensure Qdrant expects sorted indices
        sorted_pairs = sorted(zip(indices, values))
        if sorted_pairs:
            indices, values = zip(*sorted_pairs)
        else:
            indices, values = [], []
            
        return {"indices": list(indices), "values": list(values)}

    async def embed_query(self, query: str) -> tuple[list[float], dict[str, Any]]:
        dense = self._hash_text(query, dim=1024)
        sparse = self._generate_sparse(query)
        return dense, sparse

    async def embed_documents(self, documents: list[str]) -> list[tuple[list[float], dict[str, Any]]]:
        results = []
        for doc in documents:
            dense = self._hash_text(doc, dim=1024)
            sparse = self._generate_sparse(doc)
            results.append((dense, sparse))
        return results


class MockRerankerClient(RerankerClient):
    """Deterministic mock rerank client scoring based on keyword overlap."""

    async def rerank(self, query: str, passages: list[str]) -> list[float]:
        query_words = set(re.findall(r"\w+", query.lower()))
        scores = []
        for passage in passages:
            passage_words = set(re.findall(r"\w+", passage.lower()))
            overlap = len(query_words.intersection(passage_words))
            # Base score on overlap + descending baseline index to break ties
            base_score = overlap / max(1, len(query_words))
            scores.append(round(base_score, 4))
        return scores


# ---------------------------------------------------------------------------
# Clinical context modifier detection for mock synthesis
# ---------------------------------------------------------------------------
_CLINICAL_MODIFIERS = [
    # Molecular markers
    ("HER2", r"\bher2\b|\bher-2\b|\berbb2\b"),
    ("EGFR", r"\begfr\b"),
    ("KRAS wild-type", r"\bkras\s*(?:wild[- ]?type|wt)\b"),
    ("KRAS mutant", r"\bkras\s*(?:mut|mutant|mutation)\b"),
    ("NRAS", r"\bnras\b"),
    ("RAS wild-type", r"\bras\s*(?:wild[- ]?type|wt)\b"),
    ("BRAF V600E", r"\bbraf\b"),
    ("MSI-H/dMMR", r"\bmsi[- ]?h\b|\bdmmr\b|\bmicrosatellite\s*instability\b"),
    ("MSS/pMMR", r"\bmss\b|\bpmmr\b|\bmicrosatellite\s*stable\b"),
    ("ALK", r"\balk\b"),
    ("PD-L1", r"\bpd[- ]?l1\b"),
    # Tumor characteristics
    ("left-sided", r"\bleft[- ]?sided\b"),
    ("right-sided", r"\bright[- ]?sided\b"),
    ("metastatic", r"\bmetastatic\b|\badvanced\b|\bstage\s*iv\b"),
    # Lines of therapy
    ("first-line", r"\bfirst[- ]?line\b|\b1l\b|\bfront[- ]?line\b"),
    ("second-line", r"\bsecond[- ]?line\b|\b2l\b"),
    ("third-line", r"\bthird[- ]?line\b|\b3l\b"),
    # Performance / function
    ("ECOG PS", r"\becog\b|\bperformance\s*status\b"),
    ("renal impairment", r"\brenal\b|\bkidney\b"),
    ("hepatic impairment", r"\bhepatic\b|\bliver\b"),
]


def _detect_modifiers(text: str) -> list[str]:
    """Scan text for clinically relevant modifiers."""
    found = []
    lower = text.lower()
    for label, pattern in _CLINICAL_MODIFIERS:
        if re.search(pattern, lower):
            found.append(label)
    return found


def _infer_study_hierarchy(chunks: list[dict[str, Any]]) -> tuple[str, str]:
    """Determine recommendation strength and evidence quality from study types present."""
    types = {c.get("study_type", "other").lower().strip() for c in chunks}

    if "guideline" in types:
        return "Strong", "High"
    if "meta_analysis" in types or "systematic_review" in types:
        return "Strong", "Moderate"
    if "rct" in types:
        return "Conditional", "Moderate"
    if "cohort_study" in types:
        return "Conditional", "Low"
    return "Conditional", "Very Low"


class MockLLMClient(LLMClient):
    """Mock LLM client that generates structured evidence-synthesis-compliant answers."""

    async def generate_answer(self, query: str, evidence_bundle: dict[str, Any]) -> dict[str, Any]:
        # evidence_bundle format: {"snapshot_id": str, "chunks": [{"id": str, "text": str, "evidence_id": str, "study_type": str, ...}]}
        chunks = evidence_bundle.get("chunks", [])
        
        # Test for abstention triggers
        if "abstain" in query.lower() or not chunks:
            return {
                "status": "insufficient_evidence",
                "recommendation": None,
                "clinical_reasoning": None,
                "supporting_evidence": [],
                "alternatives": None,
                "bottom_line": None,
                "clinical_context": None,
                "evidence_certainty": None,
                "direct_answer_claims": [],
                "evidence_summary_claims": [],
                "limitations_claims": [],
                "abstention_reason": "No matching clinical trial evidence in active corpus snapshot."
            }

        # --- Build atomic claims (legacy, for grounding pipeline) ---
        direct_claims = []
        limitations_claims = []
        all_evidence_ids = []
        all_sentence_ids = []
        first_sentences = []

        for idx, chunk in enumerate(chunks):
            chunk_text = chunk.get("text", "")
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", chunk_text) if s.strip()]
            first_sentence = sentences[0] if sentences else chunk_text[:60]
            first_sentences.append(first_sentence)

            evidence_id = chunk.get("evidence_id", f"E{idx+1:02d}")
            sentence_id = f"{evidence_id}.S1"
            all_evidence_ids.append(evidence_id)
            all_sentence_ids.append(sentence_id)

            claim_text = f"According to study {evidence_id}, {first_sentence}"
            direct_claims.append({
                "text": claim_text,
                "evidence_ids": [evidence_id],
                "sentence_ids": [sentence_id]
            })

            if any(w in chunk_text.lower() for w in ["limit", "small", "n =", "patients"]):
                limitations_claims.append({
                    "text": f"Trial {evidence_id} has limited sample size or population constraints.",
                    "evidence_ids": [evidence_id],
                    "sentence_ids": [sentence_id]
                })

        # --- Build synthesized sections ---
        rec_strength, ev_quality = _infer_study_hierarchy(chunks)
        detected_mods = _detect_modifiers(query)

        # Recommendation: synthesize across all chunks
        recommendation = {
            "text": f"Based on the available evidence from {len(chunks)} source(s), {first_sentences[0]} This finding is supported across the reviewed literature.",
            "evidence_ids": all_evidence_ids,
            "sentence_ids": all_sentence_ids[:2],
        }

        # Clinical reasoning
        study_types_present = list({c.get("study_type", "other") for c in chunks})
        clinical_reasoning = {
            "text": f"This recommendation is supported by evidence from {', '.join(study_types_present)} studies. The available data consistently supports this approach based on the evaluated clinical endpoints.",
            "evidence_ids": all_evidence_ids,
            "sentence_ids": all_sentence_ids[:2],
        }

        # Supporting evidence — one entry per chunk
        supporting_evidence = []
        for idx, chunk in enumerate(chunks):
            evidence_id = chunk.get("evidence_id", f"E{idx+1:02d}")
            sentence_id = f"{evidence_id}.S1"
            supporting_evidence.append({
                "text": first_sentences[idx] if idx < len(first_sentences) else "Supporting data available.",
                "evidence_ids": [evidence_id],
                "sentence_ids": [sentence_id],
            })

        # Alternatives
        alternatives = {
            "text": "Alternative treatment strategies should be considered based on individual patient characteristics, comorbidities, and treatment goals.",
            "evidence_ids": [all_evidence_ids[0]] if all_evidence_ids else [],
            "sentence_ids": [],
        }

        # Bottom line
        bottom_line = {
            "text": f"In summary, the evidence supports the clinical approach described, based on {len(chunks)} reviewed source(s).",
            "evidence_ids": all_evidence_ids,
            "sentence_ids": [],
        }

        # Clinical context
        assumptions_parts = []
        if not detected_mods:
            assumptions_parts.append("No specific clinical modifiers were detected in the query. This recommendation is generalized and may need adjustment for specific molecular profiles or patient characteristics.")
        else:
            assumptions_parts.append(f"Clinical modifiers detected: {', '.join(detected_mods)}. Recommendation is contextualized to these modifiers.")

        clinical_context = {
            "detected_modifiers": detected_mods,
            "assumptions": " ".join(assumptions_parts),
        }

        # Evidence certainty
        evidence_certainty = {
            "recommendation_strength": rec_strength,
            "evidence_quality": ev_quality,
            "evidence_ids": all_evidence_ids,
        }

        return {
            "status": "answer",
            "recommendation": recommendation,
            "clinical_reasoning": clinical_reasoning,
            "supporting_evidence": supporting_evidence,
            "alternatives": alternatives,
            "bottom_line": bottom_line,
            "clinical_context": clinical_context,
            "evidence_certainty": evidence_certainty,
            "direct_answer_claims": direct_claims,
            "evidence_summary_claims": [
                {
                    "text": f"Summary of {len(chunks)} clinical trials examined for this query.",
                    "evidence_ids": [c.get("evidence_id", f"E{i+1:02d}") for i, c in enumerate(chunks)],
                    "sentence_ids": []
                }
            ],
            "limitations_claims": limitations_claims,
            "abstention_reason": None
        }


class MockEntailmentClient(EntailmentClient):
    """Mock entailment client deciding NLI support based on word similarity."""

    async def check_entailment(self, premise: str, hypothesis: str) -> str:
        if "contradict" in hypothesis.lower() or "not supported" in hypothesis.lower():
            return "contradiction"
            
        # Standard keyword overlap check
        premise_words = set(re.findall(r"\w+", premise.lower()))
        hyp_words = set(re.findall(r"\w+", hypothesis.lower()))
        
        # Strip out direct citations/source numbering like E01, E02 from matching
        hyp_words = {w for w in hyp_words if not re.match(r"^e\d+$", w)}
        
        # If the hypothesis has a high proportion of words in premise, it entails
        overlap = len(hyp_words.intersection(premise_words))
        ratio = overlap / max(1, len(hyp_words))
        
        if ratio >= 0.35:  # Low threshold to make mock generation pass easily
            return "entailment"
        return "neutral"
