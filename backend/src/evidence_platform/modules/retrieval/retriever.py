import logging
import asyncio
from typing import Any, NamedTuple
from sqlalchemy.orm import Session

from ...db.models.models import Chunk, EvidenceUnit, DocumentSection, SourceDocument, DocumentRevision
from ...ports.models import EmbeddingClient
from ...adapters.qdrant.qdrant_client import QdrantIndexClient
from ..retrieval.normalization.normalizer import QueryNormalizer
from ..clinical.clinical_intent import ClinicalQueryIntent

logger = logging.getLogger("evidence_platform.retriever")

from dataclasses import dataclass, field

@dataclass
class RetrievalCandidate:
    chunk_id: str
    qdrant_point_id: str
    text: str
    score: float
    document_id: str
    pmcid: str
    study_type: str
    publication_date: str
    display_rights: bool
    population_match_score: int = 0
    paper_metadata: Any = None
    hierarchy_rank: int = 11
    hierarchy_weight: float = 1.0
    hierarchy_label: str = ""
    unfiltered_retrieval: bool = False
    title: str = ""
    section: str = ""
    dense_vector: list[float] | None = None
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rerank_score: float | None = None


RETRIEVAL_MODES = ("dense", "sparse", "hybrid")


class RetrievalDebugInfo(NamedTuple):
    raw_query: str
    normalized_query: str
    rewritten_query: str
    expanded_terms: dict[str, str]
    dense_vector_dim: int
    sparse_term_count: int
    pre_rerank_candidates: list[dict[str, Any]]
    clinical_intent: Any = None


class HybridRetriever:
    """Performs hybrid dense-sparse searches with RRF fusion, query rewriting, and SQL metadata hydration."""

    def __init__(
        self,
        db_session: Session,
        embedding_client: EmbeddingClient,
        qdrant_client: QdrantIndexClient,
        rrf_k: int = 60
    ) -> None:
        self.db = db_session
        self.embedding_client = embedding_client
        self.qdrant_client = qdrant_client
        self.normalizer = QueryNormalizer()
        self.rrf_k = rrf_k
        self.collection_name = "evidence_chunks_1_0_0"

    def _mock_local_retrieval(self, query_text: str, filters: dict[str, Any], limit: int, clinical_intent: Any = None) -> list[RetrievalCandidate]:
        """Offline mock search using simple term-matching on database chunks."""
        logger.info("[Mock Retriever] Performing fallback database term matching.")
        
        query = self.db.query(Chunk, SourceDocument, DocumentRevision).join(
            EvidenceUnit, Chunk.evidence_unit_id == EvidenceUnit.id
        ).join(
            DocumentSection, EvidenceUnit.section_id == DocumentSection.id
        ).join(
            DocumentRevision, DocumentSection.revision_id == DocumentRevision.id
        ).join(
            SourceDocument, DocumentRevision.document_id == SourceDocument.id
        ).filter(
            DocumentRevision.status == "published",
            SourceDocument.retraction_status == "not_retracted"
        )

        if "study_types" in filters and filters["study_types"]:
            query = query.filter(DocumentRevision.study_type.in_(filters["study_types"]))
        if "year_from" in filters:
            year_str = f"{filters['year_from']}-01-01"
            query = query.filter(SourceDocument.publication_date >= year_str)

        import re
        from sqlalchemy import or_

        COMMON_STOPWORDS = {
            "is", "are", "was", "were", "effective", "in", "patients", "with", "for", "and",
            "the", "a", "an", "of", "to", "or", "on", "at", "by", "from", "study", "studies",
            "group", "treatment", "trials", "trial", "results", "about", "which", "their",
            "this", "that", "been", "have", "has", "had", "does", "what", "how", "can"
        }
        COMMON_MEDICAL_TERMS = {
            "breast", "cancer", "positive", "negative", "patients", "study", "group",
            "survival", "clinical", "therapy", "trial", "outcomes", "analysis", "overall",
            "surgery", "disease", "early", "stage", "equivalent", "compared", "versus"
        }

        words_in_query = [w for w in re.findall(r"\w+", query_text.lower()) if len(w) > 2 and w not in COMMON_STOPWORDS]
        # Deduplicate while preserving order
        seen_words: set[str] = set()
        unique_words = [w for w in words_in_query if not (w in seen_words or seen_words.add(w))]

        seen_rare: set[str] = set()
        rare_terms = [w for w in unique_words if w not in COMMON_MEDICAL_TERMS and not (w in seen_rare or seen_rare.add(w))]

        # Key intervention & drug synonyms
        key_drug_terms = []
        for kw in ["trastuzumab deruxtecan", "t-dxd", "enhertu", "destiny", "trastuzumab", "deruxtecan"]:
            if kw in query_text.lower():
                key_drug_terms.append(kw)

        # 1. Fetch targeted drug/trial hits first
        drug_results = []
        if key_drug_terms:
            drug_filters = [Chunk.text.ilike(f"%{kw}%") for kw in key_drug_terms]
            drug_results = query.filter(or_(*drug_filters)).limit(200).all()

        # 2. Fetch general rare term hits
        target_terms = rare_terms if rare_terms else unique_words
        if target_terms:
            word_filters = [Chunk.text.ilike(f"%{w}%") for w in target_terms[:8]]
            general_results = query.filter(or_(*word_filters)).limit(200).all()
        else:
            general_results = query.limit(200).all()

        # Combine results preserving uniqueness
        seen_ids: set[str] = set()
        all_results = []
        for row in drug_results + general_results:
            c_obj = row[0]
            if c_obj.id not in seen_ids:
                seen_ids.add(c_obj.id)
                all_results.append(row)

        if not all_results and unique_words:
            word_filters = [Chunk.text.ilike(f"%{w}%") for w in unique_words[:8]]
            all_results = query.filter(or_(*word_filters)).limit(200).all()

        if not all_results:
            all_results = query.limit(200).all()

        scored_candidates = []
        for c, doc, rev in all_results:
            text_lower = c.text.lower()
            title_lower = (doc.title or "").lower()

            matched_rare = sum(1 for w in rare_terms if w in text_lower or w in title_lower)
            matched_common = sum(1 for w in unique_words if w in text_lower or w in title_lower)
            matched_drug = sum(1 for kw in key_drug_terms if kw in text_lower or kw in title_lower)

            score = (matched_drug * 50.0) + (matched_rare * 10.0) + float(matched_common)
            
            if clinical_intent:
                if getattr(clinical_intent, 'disease', None) and clinical_intent.disease.lower() in text_lower:
                    score += 20.0
                if getattr(clinical_intent, 'biomarker_status', None) and clinical_intent.biomarker_status.lower() in text_lower:
                    score += 20.0

            if score > 0:
                scored_candidates.append(
                    RetrievalCandidate(
                        chunk_id=c.id,
                        qdrant_point_id=c.qdrant_point_id or "",
                        text=c.text,
                        score=score,
                        document_id=doc.id,
                        pmcid=doc.pmcid,
                        study_type=rev.study_type,
                        publication_date=doc.publication_date,
                        display_rights=doc.display_rights
                    )
                )

        scored_candidates.sort(key=lambda x: x.score, reverse=True)
        return scored_candidates[:limit]

    def _apply_rrf(self, dense_results: list[str], sparse_results: list[str]) -> list[tuple[str, float]]:
        """Combine dense and sparse result IDs using Reciprocal Rank Fusion."""
        rrf_scores: dict[str, float] = {}

        for rank, item_id in enumerate(dense_results):
            rrf_scores[item_id] = rrf_scores.get(item_id, 0.0) + (1.0 / (self.rrf_k + (rank + 1)))

        for rank, item_id in enumerate(sparse_results):
            rrf_scores[item_id] = rrf_scores.get(item_id, 0.0) + (1.0 / (self.rrf_k + (rank + 1)))

        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items

    async def retrieve_with_debug(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        limit: int = 50,
        clinical_intent: ClinicalQueryIntent | None = None,
        mode: str = "hybrid",
    ) -> tuple[list[RetrievalCandidate], RetrievalDebugInfo]:
        """Perform query normalization, rewriting, vector search (dense, sparse or
        hybrid RRF fusion) and batched metadata hydration.

        ``mode`` exists so the evaluation harness can ablate each retrieval
        channel against the same corpus snapshot.
        """
        import time
        if mode not in RETRIEVAL_MODES:
            raise ValueError(f"Unknown retrieval mode '{mode}'. Expected one of {RETRIEVAL_MODES}.")
        filters = filters or {}
        collection_name = self.collection_name

        # 1. Normalize and rewrite query
        t0 = time.perf_counter()
        norm_res = self.normalizer.normalize(query)
        search_query_text = norm_res.rewritten_query or norm_res.normalized_query
        t_normalize = time.perf_counter() - t0
        logger.info(
            "[Retrieval] Normalization: %.4fs | Rewritten query: %.100s...",
            t_normalize, search_query_text,
        )

        # 2. Determine retrieval mode — skip GPU embedding if collection is empty
        use_sql_fallback = self.qdrant_client.is_mock()
        if not use_sql_fallback:
            t0 = time.perf_counter()
            point_count = self.qdrant_client.collection_point_count(collection_name)
            t_validate = time.perf_counter() - t0
            if point_count == 0:
                logger.warning(
                    "[Retrieval] Collection '%s' has 0 indexed vectors "
                    "(validated in %.4fs). Skipping GPU embedding. Using SQL fallback.",
                    collection_name, t_validate,
                )
                use_sql_fallback = True
            else:
                logger.info(
                    "[Retrieval] Collection '%s' validated: %d vectors (%.4fs).",
                    collection_name, point_count, t_validate,
                )

        if use_sql_fallback:
            t0 = time.perf_counter()
            cands = self._mock_local_retrieval(search_query_text, filters, limit, clinical_intent)
            t_sql = time.perf_counter() - t0
            logger.info(
                "[Retrieval] SQL fallback complete: %d candidates in %.4fs",
                len(cands), t_sql,
            )
            debug = RetrievalDebugInfo(
                raw_query=query,
                normalized_query=norm_res.normalized_query,
                rewritten_query=search_query_text,
                expanded_terms=norm_res.expanded_terms,
                dense_vector_dim=0,
                sparse_term_count=0,
                pre_rerank_candidates=[{"chunk_id": c.chunk_id, "score": c.score, "pmcid": c.pmcid} for c in cands],
                clinical_intent=clinical_intent
            )
            return cands, debug

        # 3. Generate real query embeddings (BGE-M3)
        t0 = time.perf_counter()
        dense, sparse = await self.embedding_client.embed_query(search_query_text)
        t_embed = time.perf_counter() - t0
        logger.info(
            "[Retrieval] Embedding: %.4fs | Dense dim: %d | Sparse terms: %d",
            t_embed, len(dense), len(sparse.get("indices", [])),
        )

        # 4. Query Qdrant with metadata filters.
        # NOTE: earlier versions added `should` conditions on `disease` /
        # `biomarker` payload keys that are never written at ingestion time.
        # Qdrant requires >=1 `should` match, so every such query returned zero
        # hits and was silently re-run. Population constraints are applied
        # downstream by PopulationMatchEvaluator instead.
        from qdrant_client.http import models as qmodels

        qdrant_filters = [
            qmodels.FieldCondition(key="retraction_status", match=qmodels.MatchValue(value="not_retracted")),
            qmodels.FieldCondition(key="is_retrievable", match=qmodels.MatchValue(value=True))
        ]

        if "study_types" in filters and filters["study_types"]:
            qdrant_filters.append(qmodels.FieldCondition(
                key="study_type",
                match=qmodels.MatchAny(any=filters["study_types"])
            ))

        if filters.get("year_from"):
            qdrant_filters.append(qmodels.FieldCondition(
                key="publication_year",
                range=qmodels.Range(gte=int(filters["year_from"]))
            ))

        filter_obj = qmodels.Filter(must=qdrant_filters)
        fetch_limit = max(100, limit * 2)
        client = self.qdrant_client.client

        def run_dense_search():
            return client.query_points(
                collection_name=collection_name,
                query=dense,
                using="dense",
                query_filter=filter_obj,
                limit=fetch_limit,
                with_payload=["chunk_id"],
                timeout=30,
            ).points

        def run_sparse_search():
            return client.query_points(
                collection_name=collection_name,
                query=qmodels.SparseVector(indices=sparse["indices"], values=sparse["values"]),
                using="sparse",
                query_filter=filter_obj,
                limit=fetch_limit,
                with_payload=["chunk_id"],
                timeout=30,
            ).points

        async def _safe(fn, label):
            try:
                return await asyncio.to_thread(fn)
            except Exception as e:  # one channel failing must not kill the query
                logger.warning("%s search failed: %s", label, e)
                return []

        # Dense and sparse channels run concurrently.
        t0 = time.perf_counter()
        dense_task = _safe(run_dense_search, "Dense") if mode in ("dense", "hybrid") else asyncio.sleep(0, result=[])
        sparse_task = _safe(run_sparse_search, "Sparse") if mode in ("sparse", "hybrid") else asyncio.sleep(0, result=[])
        dense_hits, sparse_hits = await asyncio.gather(dense_task, sparse_task)
        t_search = time.perf_counter() - t0

        if not dense_hits and not sparse_hits:
            logger.warning(
                "[Retrieval] Qdrant returned 0 hits (search took %.4fs). Falling back to SQL retrieval.",
                t_search,
            )
            cands = self._mock_local_retrieval(search_query_text, filters, limit, clinical_intent)
            debug = RetrievalDebugInfo(
                raw_query=query,
                normalized_query=norm_res.normalized_query,
                rewritten_query=search_query_text,
                expanded_terms=norm_res.expanded_terms,
                dense_vector_dim=len(dense),
                sparse_term_count=len(sparse.get("indices", [])),
                pre_rerank_candidates=[{"chunk_id": c.chunk_id, "score": c.score, "pmcid": c.pmcid} for c in cands],
                clinical_intent=clinical_intent
            )
            return cands, debug

        # 5. Fuse rankings. In single-channel modes the RRF score reduces to
        #    1/(k+rank) of that channel, so ordering is preserved.
        t0 = time.perf_counter()
        point_id_by_chunk: dict[str, Any] = {}
        dense_ids: list[str] = []
        sparse_ids: list[str] = []
        for hit in dense_hits:
            if hit.payload and hit.payload.get("chunk_id"):
                dense_ids.append(hit.payload["chunk_id"])
                point_id_by_chunk[hit.payload["chunk_id"]] = hit.id
        for hit in sparse_hits:
            if hit.payload and hit.payload.get("chunk_id"):
                sparse_ids.append(hit.payload["chunk_id"])
                point_id_by_chunk.setdefault(hit.payload["chunk_id"], hit.id)
        dense_rank = {cid: i + 1 for i, cid in enumerate(dense_ids)}
        sparse_rank = {cid: i + 1 for i, cid in enumerate(sparse_ids)}

        rrf_ranked = self._apply_rrf(dense_ids, sparse_ids)[:limit]
        t_fusion = time.perf_counter() - t0

        # 6. Fetch stored dense vectors for the fused top-k in ONE call, so the
        #    context assembler can run MMR without re-embedding every passage.
        top_chunk_ids = [cid for cid, _ in rrf_ranked]
        vectors_by_chunk: dict[str, list[float]] = {}
        try:
            pts = await asyncio.to_thread(
                client.retrieve,
                collection_name=collection_name,
                ids=[point_id_by_chunk[c] for c in top_chunk_ids if c in point_id_by_chunk],
                with_vectors=["dense"],
                with_payload=["chunk_id"],
            )
            for p in pts:
                vec = p.vector.get("dense") if isinstance(p.vector, dict) else p.vector
                if p.payload and vec is not None:
                    vectors_by_chunk[p.payload["chunk_id"]] = list(vec)
        except Exception as e:
            logger.warning("[Retrieval] Could not fetch stored vectors (MMR will re-embed): %s", e)

        # 7. Hydrate metadata with a single batched SQL join (was ~5 queries/candidate).
        t0 = time.perf_counter()
        rows = (
            self.db.query(Chunk, DocumentSection, DocumentRevision, SourceDocument)
            .join(EvidenceUnit, Chunk.evidence_unit_id == EvidenceUnit.id)
            .join(DocumentSection, EvidenceUnit.section_id == DocumentSection.id)
            .join(DocumentRevision, DocumentSection.revision_id == DocumentRevision.id)
            .join(SourceDocument, DocumentRevision.document_id == SourceDocument.id)
            .filter(Chunk.id.in_(top_chunk_ids))
            .all()
        ) if top_chunk_ids else []
        by_id = {r[0].id: r for r in rows}

        candidates = []
        pre_rerank_debug_list = []
        for rank, (chunk_id, rrf_score) in enumerate(rrf_ranked, start=1):
            row = by_id.get(chunk_id)
            if not row:
                continue
            c, sec, rev, doc = row
            cand = RetrievalCandidate(
                chunk_id=c.id,
                qdrant_point_id=c.qdrant_point_id or "",
                text=c.text,
                score=rrf_score,
                document_id=doc.id,
                pmcid=doc.pmcid,
                study_type=rev.study_type,
                publication_date=doc.publication_date,
                display_rights=doc.display_rights,
                title=doc.title or "",
                section=sec.section_kind or "",
                dense_vector=vectors_by_chunk.get(chunk_id),
                dense_rank=dense_rank.get(chunk_id),
                sparse_rank=sparse_rank.get(chunk_id),
            )
            candidates.append(cand)
            pre_rerank_debug_list.append({
                "chunk_id": c.id,
                "rank": rank,
                "pmcid": doc.pmcid,
                "title": doc.title,
                "study_type": rev.study_type,
                "publication_date": doc.publication_date,
                "fusion_score": round(rrf_score, 5),
                "dense_rank": dense_rank.get(chunk_id),
                "sparse_rank": sparse_rank.get(chunk_id),
                "section": sec.section_kind if sec else "Unknown"
            })
        t_hydrate = time.perf_counter() - t0
        logger.info(
            "[Retrieval] mode=%s | Search: %.4fs | Fusion: %.4fs | Hydration: %.4fs | Candidates: %d",
            mode, t_search, t_fusion, t_hydrate, len(candidates),
        )

        debug = RetrievalDebugInfo(
            raw_query=query,
            normalized_query=norm_res.normalized_query,
            rewritten_query=search_query_text,
            expanded_terms=norm_res.expanded_terms,
            dense_vector_dim=len(dense),
            sparse_term_count=len(sparse.get("indices", [])),
            pre_rerank_candidates=pre_rerank_debug_list,
            clinical_intent=clinical_intent
        )

        return candidates, debug

    async def retrieve(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        limit: int = 50,
        clinical_intent: ClinicalQueryIntent | None = None,
        mode: str = "hybrid",
    ) -> list[RetrievalCandidate]:
        candidates, _ = await self.retrieve_with_debug(query, filters, limit, clinical_intent, mode=mode)
        return candidates
