import math
import logging
import numpy as np
from typing import Any, NamedTuple
from ...ports.models import RerankerClient, EmbeddingClient
from .retriever import RetrievalCandidate
from ..corpus.chunker import count_tokens

logger = logging.getLogger("evidence_platform.context_assembler")

def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Compute cosine similarity between two float vectors."""
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_a = math.sqrt(sum(a * a for a in v1))
    norm_b = math.sqrt(sum(b * b for b in v2))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)


class AssembledContext(NamedTuple):
    context_text: str
    selected_candidates: list[RetrievalCandidate]
    total_tokens: int
    # Full candidate list ordered by the blended relevance score (before MMR /
    # token budget). Used by the evaluation harness to compute Recall@k etc.
    ranked_candidates: list[RetrievalCandidate] = []


class ContextAssembler:
    """Reranks retrieval candidates and selects diverse chunks under a token budget using MMR, Population Matching, and Evidence Hierarchy."""

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        reranker_client: RerankerClient,
        lambda_val: float = 0.5
    ) -> None:
        self.embedding_client = embedding_client
        self.reranker_client = reranker_client
        self.lambda_val = lambda_val

    async def assemble_context(
        self,
        query: str,
        candidates: list[RetrievalCandidate],
        max_tokens_budget: int = 2048
    ) -> AssembledContext:
        if not candidates:
            return AssembledContext("", [], 0)

        # 1. Rerank all candidates using the Cross-Encoder Reranker
        passage_texts = [c.text for c in candidates]
        raw_rerank_scores = await self.reranker_client.rerank(query, passage_texts)
        
        STUDY_TYPE_WEIGHTS = {
            "guideline": 1.35,
            "meta_analysis": 1.3,
            "systematic_review": 1.25,
            "rct": 1.2,
            "cohort_study": 1.0,
            "observational": 0.9,
            "case_report": 0.8,
            "other": 1.0
        }

        # Combine Cross-Encoder Reranker score with Population Match Score and Evidence Hierarchy Weight
        combined_scores = []
        for candidate, raw_score in zip(candidates, raw_rerank_scores):
            try:
                candidate.rerank_score = float(raw_score)
            except Exception:
                pass
            st = (candidate.study_type or "other").lower().strip()
            # Population Match Score factor (+100 to -300)
            pop_score = getattr(candidate, "population_match_score", 0)
            # Hierarchy Weight factor (0.5 to 2.5)
            h_weight = getattr(candidate, "hierarchy_weight", STUDY_TYPE_WEIGHTS.get(st, 1.0))

            # Relevance-first blending (revised after the retrieval ablation).
            # The previous blend (0.3*rerank + 0.4*hierarchy + 0.004*population)
            # let study type and population priors (up to ~1.4) swamp the
            # cross-encoder relevance (max 0.3): on the 119-question known-item
            # benchmark it demoted 48 relevant papers and cut Recall@10 from 0.92
            # to 0.67. Priors are now bounded tie-breakers (|prior| <= 0.10) that
            # reorder near-ties but never override a clearly more relevant passage.
            # Severe population mismatches are still hard-excluded.
            if pop_score < -100:
                adjusted_score = -999.0
            else:
                hier_prior = max(-0.33, min(1.0, (h_weight - 1.0) / 1.5)) * 0.06
                pop_prior = max(-1.0, min(1.0, pop_score / 300.0)) * 0.04
                adjusted_score = float(raw_score) + hier_prior + pop_prior
                
            combined_scores.append(adjusted_score)
        
        # 2. Dense vectors for the MMR redundancy term. The retriever attaches the
        #    vectors already stored in Qdrant; only candidates without one (e.g.
        #    SQL-fallback results) are re-embedded.
        candidate_vectors: list[list[float] | None] = [getattr(c, "dense_vector", None) for c in candidates]
        missing = [i for i, v in enumerate(candidate_vectors) if not v]
        if missing:
            embeddings = await self.embedding_client.embed_documents([passage_texts[i] for i in missing])
            for i, (dense, _) in zip(missing, embeddings):
                candidate_vectors[i] = dense
        mat = np.asarray(candidate_vectors, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat = mat / norms
        sim_matrix = mat @ mat.T  # pairwise cosine similarity, computed once

        # Normalize combined scores to [0, 1] for MMR calculation
        min_score = min(combined_scores) if combined_scores else 0.0
        max_score = max(combined_scores) if combined_scores else 1.0
        score_range = max_score - min_score if max_score != min_score else 1.0

        normalized_scores = [
            (score - min_score) / score_range for score in combined_scores
        ]

        # 3. Perform Maximal Marginal Relevance (MMR) selection loop
        selected_indices: list[int] = []
        remaining_indices = list(range(len(candidates)))
        
        current_tokens = 0
        selected_candidates: list[RetrievalCandidate] = []
        selected_pmcid_outcomes: dict[str, set[str]] = {}
        outcome_keywords = {"PFS", "OS", "ORR"}

        while remaining_indices:
            best_mmr_score = -float("inf")
            best_idx = -1

            for idx in remaining_indices:
                relevance = normalized_scores[idx]
                
                # Calculate redundancy: max similarity with already selected items
                redundancy = 0.0
                if selected_indices:
                    redundancy = float(sim_matrix[idx, selected_indices].max())
                
                # MMR formula
                mmr_score = self.lambda_val * relevance - (1.0 - self.lambda_val) * redundancy
                
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_idx = idx

            if best_idx == -1:
                break

            # Check if candidate fits in token budget and is not severely mismatched
            candidate = candidates[best_idx]
            pop_score = getattr(candidate, "population_match_score", 0)

            if pop_score >= -50:
                pmcid = getattr(candidate, "pmcid", "") or ""
                cand_text = candidate.text.upper()
                cand_outcomes = {kw for kw in outcome_keywords if kw in cand_text}
                
                allow_selection = True
                if pmcid and pmcid in selected_pmcid_outcomes:
                    existing_outcomes = selected_pmcid_outcomes[pmcid]
                    if not (cand_outcomes - existing_outcomes):
                        allow_selection = False
                
                if allow_selection:
                    cand_tokens = count_tokens(candidate.text)
                    if current_tokens + cand_tokens <= max_tokens_budget:
                        selected_indices.append(best_idx)
                        selected_candidates.append(candidate)
                        current_tokens += cand_tokens
                        if pmcid:
                            if pmcid not in selected_pmcid_outcomes:
                                selected_pmcid_outcomes[pmcid] = set()
                            selected_pmcid_outcomes[pmcid].update(cand_outcomes)
                
            remaining_indices.remove(best_idx)

        # 4. Sort selected candidates by Evidence Hierarchy Rank (1 is highest), then study weight
        STUDY_TYPE_RANKS = {
            "guideline": 1,
            "rct": 2,
            "phase_3_rct": 2,
            "meta_analysis": 3,
            "systematic_review": 4,
            "cohort_study": 5,
            "observational": 6,
            "case_report": 7,
            "other": 8
        }
        def get_cand_rank(c):
            if hasattr(c, "hierarchy_rank"):
                return getattr(c, "hierarchy_rank")
            st = (getattr(c, "study_type", "") or "other").lower().strip()
            return STUDY_TYPE_RANKS.get(st, 8)

        selected_candidates.sort(key=lambda x: (get_cand_rank(x), getattr(x, "pmcid", "") or "", getattr(x, "chunk_id", "") or ""))

        # 5. Format assembled context block
        context_blocks = []
        for c in selected_candidates:
            h_label = getattr(c, "hierarchy_label", c.study_type or "Evidence")
            h_rank = get_cand_rank(c)
            pop_score = getattr(c, "population_match_score", 0)
            ref_header = f"[Source: {c.pmcid} | Hierarchy Rank: {h_rank} ({h_label}) | PopMatchScore: {pop_score}]"
            context_blocks.append(f"{ref_header}\n{c.text}\n")

        assembled_text = "\n".join(context_blocks)
        total_tokens = count_tokens(assembled_text)

        order = sorted(range(len(candidates)), key=lambda i: combined_scores[i], reverse=True)
        return AssembledContext(
            context_text=assembled_text,
            selected_candidates=selected_candidates,
            total_tokens=total_tokens,
            ranked_candidates=[candidates[i] for i in order],
        )
