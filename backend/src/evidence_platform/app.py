import os
import uuid
import logging
import datetime
import hashlib
import time
from typing import Any
from fastapi import FastAPI, Depends, Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

# Import utilities and modules
from .core.config import settings
from .core.rate_limiter import rate_limiter
from .core.csrf import verify_csrf_token, generate_csrf_token
from .db.models.models import Base, User, QueryRun, Answer, RetrievalTrace, Chunk, RetrievalCandidate as DBRetrievalCandidate, AnswerClaim, Citation
from .modules.auth.auth_manager import AuthManager
from .modules.retrieval.retriever import HybridRetriever
from .modules.retrieval.context_assembler import ContextAssembler
from .modules.synthesis.synthesizer import AnswerSynthesizer
from .modules.synthesis.validator import NumericDosageValidator, ClinicalSafetyValidator
from .modules.synthesis.highlighter import SpanHighlighter
from .modules.synthesis.abstention import AbstentionLayer
from .modules.clinical.clinical_intent import ClinicalIntentExtractor
from .modules.clinical.population_matcher import PopulationMatchEvaluator
from .modules.clinical.evidence_hierarchy import EvidenceHierarchyRanker
from .modules.synthesis.evidence_grouper import EvidenceGrouper
from .modules.synthesis.contradiction_checker import ContradictionChecker
from .modules.synthesis.citation_auditor import CitationAuditor
from .adapters.models.factory import (
    get_embedding_client,
    get_reranker_client,
    get_llm_client,
    get_entailment_client,
)
from .adapters.qdrant.qdrant_client import QdrantIndexClient

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evidence_platform.api")

# Database Setup
connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(settings.DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Ensure all database tables exist
Base.metadata.create_all(bind=engine)

# Initialize managers
auth_manager = AuthManager()

# Seed default clinician user if not exists
def seed_default_user():
    db = SessionLocal()
    try:
        user = db.query(User).filter_by(email="clinician@indence.org").first()
        if not user:
            default_user = User(
                id=str(uuid.uuid4()),
                email="clinician@indence.org",
                password_hash=auth_manager.hash_password("DoctorPassword1"),
                role="clinician",
                status="active"
            )
            db.add(default_user)
            db.commit()
            logger.info("Created default clinician user (clinician@indence.org).")
    except Exception as e:
        logger.error(f"Error seeding default user: {e}")
    finally:
        db.close()

seed_default_user()

_qdrant_client_singleton: QdrantIndexClient | None = None

def get_qdrant_client() -> QdrantIndexClient:
    """Return a singleton QdrantIndexClient to avoid per-request init overhead."""
    global _qdrant_client_singleton
    if _qdrant_client_singleton is None:
        _qdrant_client_singleton = QdrantIndexClient(
            host=settings.QDRANT_HOST, port=settings.QDRANT_PORT
        )
    return _qdrant_client_singleton

app = FastAPI(
    title="Indence Oncology Evidence Engine API",
    description="FastAPI backend for oncology evidence retrieval and synthesis.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Auth helper
def get_current_user(request: Request, db: Session = Depends(get_db)) -> User:
    token = request.cookies.get("session_token")
    if token:
        user = auth_manager.verify_user_session(db, token)
        if user:
            return user

    # Fallback to default clinician user for seamless local development
    default_user = db.query(User).filter_by(email="clinician@indence.org").first()
    if default_user:
        return default_user

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated"
    )


# CSRF cookie setter
@app.get("/api/v1/csrf-token")
def get_csrf_token(response: Response):
    token = generate_csrf_token()
    # Set non-HttpOnly cookie so frontend JS can read and send in custom headers
    response.set_cookie(
        key="csrf_token",
        value=token,
        samesite="lax",
        secure=False  # True in production HTTPS
    )
    return {"detail": "CSRF cookie set successfully.", "csrf_token": token}


# AUTH ROUTES
@app.post("/api/v1/auth/login")
def login(request: Request, response: Response, payload: dict[str, Any], db: Session = Depends(get_db)):
    # Rate Limit checking
    client_ip = request.client.host if request.client else "unknown"
    if not rate_limiter.is_allowed(f"login:{client_ip}", limit=5, window_seconds=60):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many login attempts. Please try again later."
        )

    email = payload.get("email")
    password = payload.get("password")
    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password required")

    user = db.query(User).filter_by(email=email).first()
    if not user or not auth_manager.verify_password(user.password_hash, password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    # Create session
    session_record = auth_manager.create_user_session(db, user.id)
    
    # Set secure HttpOnly session cookie
    response.set_cookie(
        key="session_token",
        value=session_record.raw_token,
        httponly=True,
        samesite="lax",
        secure=False,  # True in production HTTPS
        max_age=7 * 24 * 3600
    )

    return {
        "user_id": user.id,
        "email": user.email,
        "role": user.role
    }


@app.post("/api/v1/auth/logout")
def logout(request: Request, response: Response, db: Session = Depends(get_db)):
    token = request.cookies.get("session_token")
    if token:
        auth_manager.invalidate_user_session(db, token)
    
    response.delete_cookie("session_token")
    return {"detail": "Successfully logged out."}


# RAG ENGINE ENGINE ROUTE
@app.post("/api/v1/search", dependencies=[Depends(verify_csrf_token)])
async def clinical_search(
    request: Request,
    payload: dict[str, Any],
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Rate limit check per user
    if not rate_limiter.is_allowed(f"user_search:{user.id}", limit=30, window_seconds=60):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded."
        )

    query = payload.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="Query string is required")
        
    filters = payload.get("filters", {})

    # Instantiate services dynamically with current factories
    embedding_client = get_embedding_client()
    reranker_client = get_reranker_client()
    llm_client = get_llm_client()
    entailment_client = get_entailment_client()

    retriever = HybridRetriever(db, embedding_client, get_qdrant_client())
    assembler = ContextAssembler(embedding_client, reranker_client)
    synthesizer = AnswerSynthesizer(llm_client)
    numeric_validator = NumericDosageValidator()
    highlighter = SpanHighlighter(db)
    abstention_layer = AbstentionLayer(entailment_client)

    # 12-Step OpenEvidence Engine Pipeline
    intent_extractor = ClinicalIntentExtractor()
    population_evaluator = PopulationMatchEvaluator()
    hierarchy_ranker = EvidenceHierarchyRanker()
    evidence_grouper = EvidenceGrouper()
    contradiction_checker = ContradictionChecker()
    citation_auditor = CitationAuditor()
    safety_validator = ClinicalSafetyValidator()

    # --- Profiling infrastructure ---
    _stage_timings: dict[str, float] = {}
    _candidate_counts: dict[str, int] = {}
    try:
        import psutil as _psutil
        _proc = _psutil.Process()
        _mem_start = _proc.memory_info().rss / (1024 * 1024)
    except (ImportError, Exception):
        _proc = None
        _mem_start = 0.0

    def _mem_mb() -> float:
        if _proc:
            return _proc.memory_info().rss / (1024 * 1024)
        return 0.0

    # Step 1: Clinical Intent Extraction
    _t0 = time.perf_counter()
    intent = intent_extractor.extract_intent(query)
    _stage_timings["intent_extraction_s"] = round(time.perf_counter() - _t0, 4)

    # Step 1.5: Retrieve initial vector/hybrid candidates (up to 50)
    _t0 = time.perf_counter()
    raw_candidates, debug_info = await retriever.retrieve_with_debug(query, filters, limit=50, clinical_intent=intent)
    _stage_timings["retrieval_s"] = round(time.perf_counter() - _t0, 4)
    _candidate_counts["pre_filter"] = len(raw_candidates)
    _candidate_counts["unfiltered_retrieval"] = sum(1 for c in raw_candidates if getattr(c, "unfiltered_retrieval", False))

    # Step 2 & Step 3: Evidence Population Matching & Hard Filters
    _t0 = time.perf_counter()
    matched_candidates, excluded_audit_entries = population_evaluator.filter_and_score_candidates(intent, raw_candidates)
    _stage_timings["population_matching_s"] = round(time.perf_counter() - _t0, 4)
    _candidate_counts["post_hard_filter"] = len(matched_candidates)

    # Step 4: Evidence Hierarchy Ranking
    _t0 = time.perf_counter()
    ranked_candidates = hierarchy_ranker.rank_candidates(matched_candidates)
    _stage_timings["evidence_hierarchy_s"] = round(time.perf_counter() - _t0, 4)

    # Step 5: MMR Diversification & Context Assembly
    _t0 = time.perf_counter()
    assembled = await assembler.assemble_context(query, ranked_candidates, max_tokens_budget=2048)
    _stage_timings["context_assembly_s"] = round(time.perf_counter() - _t0, 4)
    _candidate_counts["post_assembly"] = len(assembled.selected_candidates)

    # Step 6: Evidence Grouping
    _t0 = time.perf_counter()
    clusters = evidence_grouper.group_candidates(assembled.selected_candidates)
    _stage_timings["evidence_grouping_s"] = round(time.perf_counter() - _t0, 4)

    # Step 7: Contradiction Check
    _t0 = time.perf_counter()
    contradiction_info = contradiction_checker.check_contradictions(clusters, assembled.selected_candidates)
    _stage_timings["contradiction_check_s"] = round(time.perf_counter() - _t0, 4)

    # Step 8: Synthesis & Clinical Safety Validation
    _t0 = time.perf_counter()
    synth_res = await synthesizer.synthesize(
        query,
        assembled,
        intent=intent,
        clusters=clusters,
        contradiction_info=contradiction_info
    )
    _stage_timings["synthesis_s"] = round(time.perf_counter() - _t0, 4)
    _candidate_counts["synthesis_claims"] = len(synth_res.direct_claims)

    # Numeric and dosage validation
    validated_claims = numeric_validator.validate_answer(synth_res.direct_claims, synth_res.evidence_map)

    # Step 8 Safety Validation: Audit claim citations against target population
    safe_claims, rejected_citations = safety_validator.validate_clinical_safety(intent, validated_claims, synth_res.evidence_map)

    # Step 9: Span Alignment & Grounding Highlighting
    aligned_claims = highlighter.align_claims(safe_claims, synth_res.evidence_map)

    # Grounding NLI verification & dynamic safety abstention gating
    _t0 = time.perf_counter()
    final_status, final_claims, abstention_reason = await abstention_layer.verify_and_calibrate(
        status=synth_res.status,
        direct_claims=aligned_claims,
        evidence_map=synth_res.evidence_map,
        abstention_reason=synth_res.abstention_reason,
        candidates=assembled.selected_candidates,
        intent=intent
    )
    _stage_timings["grounding_s"] = round(time.perf_counter() - _t0, 4)
    _candidate_counts["final_claims"] = len(final_claims)

    # Step 11 & Step 12: Citation Audit & Failure Analysis
    _t0 = time.perf_counter()
    citation_audit, failure_analysis = citation_auditor.audit_citations(
        intent,
        assembled.selected_candidates,
        excluded_audit_entries,
        final_claims
    )
    _stage_timings["citation_audit_s"] = round(time.perf_counter() - _t0, 4)

    _mem_end = _mem_mb()
    _stage_timings["total_e2e_s"] = round(sum(_stage_timings.values()), 4)
    logger.info(
        "12-Step Clinical Pipeline Complete: status=%s | %d claims | Total: %.4fs",
        final_status, len(final_claims), _stage_timings["total_e2e_s"]
    )

    # Save Trace to Database
    run_id = str(uuid.uuid4())
    query_run = QueryRun(
        id=run_id,
        user_id=user.id,
        raw_text=query,
        normalized_text=query,
        filters_json=filters,
        status="abstained" if final_status == "insufficient_evidence" else "answered"
    )
    db.add(query_run)

    trace_id = str(uuid.uuid4())
    trace_record = RetrievalTrace(
        id=trace_id,
        query_run_id=run_id,
        normalization={"query_text": query, "clinical_intent": intent.to_dict()},
        config={"filters": filters},
        model_versions={"embedding": "BGE-M3", "llm": "mock-llm"},
        timings=_stage_timings
    )
    db.add(trace_record)

    # Save candidates
    for rank, cand in enumerate(raw_candidates, start=1):
        db_cand = DBRetrievalCandidate(
            trace_id=trace_id,
            chunk_id=cand.chunk_id,
            stage="fused",
            rank=rank,
            score=cand.score,
            score_details={}
        )
        db.add(db_cand)

    # Apply span highlighting to synthesis sections
    def _align_synthesis_section(section: dict | None) -> dict | None:
        if not section or not section.get("text"):
            return section
        pseudo_claim = {
            "text": section.get("text", ""),
            "sentence_ids": section.get("sentence_ids", []),
            "evidence_ids": section.get("evidence_ids", []),
        }
        aligned = highlighter.align_claims([pseudo_claim], synth_res.evidence_map)
        if aligned:
            section_copy = dict(section)
            section_copy["aligned_spans"] = aligned[0].get("aligned_spans", [])
            return section_copy
        return section

    aligned_recommendation = _align_synthesis_section(synth_res.recommendation)
    aligned_clinical_reasoning = _align_synthesis_section(synth_res.clinical_reasoning)
    aligned_alternatives = _align_synthesis_section(synth_res.alternatives)
    aligned_bottom_line = _align_synthesis_section(synth_res.bottom_line)
    aligned_supporting_evidence = [
        _align_synthesis_section(se) or se
        for se in (synth_res.supporting_evidence or [])
    ]

    synthesis_response = {
        "recommendation": aligned_recommendation,
        "clinical_reasoning": aligned_clinical_reasoning,
        "supporting_evidence": aligned_supporting_evidence,
        "alternatives": aligned_alternatives,
        "bottom_line": aligned_bottom_line,
        "clinical_context": synth_res.clinical_context,
        "evidence_certainty": synth_res.evidence_certainty,
    }

    answer_id = str(uuid.uuid4())
    # Render structured synthesis markdown
    markdown_lines = []
    if final_status == "insufficient_evidence":
        markdown_lines.append(f"**Abstained**: {abstention_reason}")
    else:
        if synth_res.recommendation:
            markdown_lines.append(f"## Recommendation\n{synth_res.recommendation.get('text', '')}")
        if synth_res.clinical_reasoning:
            markdown_lines.append(f"## Clinical Reasoning\n{synth_res.clinical_reasoning.get('text', '')}")
        if synth_res.supporting_evidence:
            markdown_lines.append("## Supporting Evidence")
            for se in synth_res.supporting_evidence:
                markdown_lines.append(f"- {se.get('text', '')}")
        if synth_res.alternatives:
            markdown_lines.append(f"## Alternatives\n{synth_res.alternatives.get('text', '')}")
        if synth_res.bottom_line:
            markdown_lines.append(f"## Bottom Line\n{synth_res.bottom_line.get('text', '')}")
    rendered_markdown = "\n\n".join(markdown_lines)

    db_answer = Answer(
        id=answer_id,
        query_run_id=run_id,
        status="abstained" if final_status == "insufficient_evidence" else "published",
        rendered_markdown=rendered_markdown,
        confidence="high" if final_status == "answer" else "low",
        generation_model="mock-llm",
        prompt_version="2.0.0",
        verification_summary={"failed_nums_count": sum(1 for c in final_claims if not c.get("numeric_validation_passed", True))}
    )
    db.add(db_answer)

    # Save AnswerClaims
    claim_ordinal = 0
    for idx, claim in enumerate(final_claims, start=1):
        claim_ordinal += 1
        claim_id = str(uuid.uuid4())
        db_claim = AnswerClaim(
            id=claim_id,
            answer_id=answer_id,
            ordinal=claim_ordinal,
            text=claim.get("text", ""),
            claim_type="direct_answer",
            is_material=True,
            verification_status="verified" if claim.get("numeric_validation_passed", True) else "unverified"
        )
        db.add(db_claim)

        # Citations
        seen_chunks_for_claim = set()
        for s_idx, span in enumerate(claim.get("aligned_spans", []), start=1):
            cit_id = str(uuid.uuid4())
            supporting_hash = hashlib.sha256(span.get("text", "").encode("utf-8")).hexdigest()
            parts = span.get("sentence_id", "").split(".S")
            ref_id = parts[0] if parts else ""
            candidate_ref = synth_res.evidence_map.get(ref_id)
            chunk_id = candidate_ref.chunk_id if candidate_ref else None

            if chunk_id and chunk_id not in seen_chunks_for_claim:
                seen_chunks_for_claim.add(chunk_id)
                db_citation = Citation(
                    id=cit_id,
                    claim_id=claim_id,
                    chunk_id=chunk_id,
                    marker_index=s_idx,
                    span_start=span.get("section_start_char", 0),
                    span_end=span.get("section_end_char", 0),
                    verification_status="verified" if claim.get("numeric_validation_passed", True) else "unverified",
                    supporting_text_hash=supporting_hash
                )
                db.add(db_citation)

    db.commit()

    # Step 10: Complete Structured Output Response
    supporting_pmcids = list({getattr(c, "pmcid", "") for c in assembled.selected_candidates if getattr(c, "pmcid", None)})
    supporting_trials = list({getattr(c, "hierarchy_label", c.study_type) for c in assembled.selected_candidates})
    population_scores = {getattr(c, "chunk_id", f"chk_{i}"): getattr(c, "population_match_score", 0) for i, c in enumerate(assembled.selected_candidates)}

    clinical_answer = (
        (synth_res.recommendation or {}).get("text")
        or (synth_res.bottom_line or {}).get("text")
        or rendered_markdown
    )

    return {
        "answer_id": answer_id,
        "trace_id": trace_id,
        "status": final_status,
        "clinical_answer": clinical_answer,
        "confidence": "High" if final_status == "answer" else "Low",
        "extraction_confidence": intent.confidence,
        "supporting_claims": final_claims,
        "rejected_citations": rejected_citations,
        "evidence_strength": synth_res.evidence_certainty,
        "supporting_trials": supporting_trials,
        "supporting_pmcid": supporting_pmcids,
        "evidence_spans": [s for c in final_claims for s in c.get("aligned_spans", [])],
        "population_match_score": population_scores,
        "clinical_intent": intent.to_dict(),
        "evidence_clusters": [c.to_dict() for c in clusters],
        "contradiction_analysis": contradiction_info,
        "citation_audit": citation_audit,
        "failure_analysis": failure_analysis,
        "synthesis": synthesis_response,
        "direct_answer_claims": final_claims,
        "evidence_summary_claims": synth_res.summary_claims,
        "limitations_claims": synth_res.limitations,
        "abstention_reason": abstention_reason,
        "_debug": {
            "timings": _stage_timings,
            "candidate_counts": _candidate_counts,
            "retrieval_mode": "qdrant_vector" if not get_qdrant_client().is_mock() else "sqlite_fallback",
            "clinical_intent_filters_applied": {
                "disease_filter": intent.disease is not None,
                "biomarker_filter": intent.biomarker_status is not None,
                "broad_scope": intent.explicitly_broad_scope,
            },
            "population_match_breakdown": {
                getattr(c, "chunk_id", f"chk_{i}"): {
                    "score": getattr(c, "population_match_score", 0),
                    "hierarchy_rank": getattr(c, "hierarchy_rank", None),
                    "hierarchy_label": getattr(c, "hierarchy_label", None),
                } for i, c in enumerate(assembled.selected_candidates)
            },
        }
    }


# AUDIT TRACE LOOKUP ROUTE
@app.get("/api/v1/traces/{trace_id}")
def get_trace_audit(trace_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    trace = db.query(RetrievalTrace).filter_by(id=trace_id).first()
    if not trace:
        raise HTTPException(status_code=404, detail="Audit trace not found.")

    query_run = db.query(QueryRun).filter_by(id=trace.query_run_id).first()
    answer = db.query(Answer).filter_by(query_run_id=trace.query_run_id).first()

    # Retrieve claims from DB
    claims = []
    if answer:
        claims_records = db.query(AnswerClaim).filter_by(answer_id=answer.id).order_by(AnswerClaim.ordinal).all()
        for cr in claims_records:
            # Query citations for this claim
            cits = db.query(Citation).filter_by(claim_id=cr.id).order_by(Citation.marker_index).all()
            aligned_spans = []
            for cit in cits:
                aligned_spans.append({
                    "sentence_id": f"E{cit.marker_index}",  # reconstruct a display ID
                    "section_start_char": cit.span_start,
                    "section_end_char": cit.span_end
                })
            claims.append({
                "text": cr.text,
                "claim_type": cr.claim_type,
                "verification_status": cr.verification_status,
                "aligned_spans": aligned_spans
            })

    return {
        "trace_id": trace.id,
        "query": query_run.raw_text if query_run else None,
        "filters": query_run.filters_json if query_run else None,
        "status": answer.status if answer else None,
        "claims": claims,
        "generation_model": f"{settings.LLM_MODEL_NAME} ({settings.LLM_PROVIDER.title()})",
        "normalization_context": trace.normalization,
        "retrieved_chunk_ids": [c.chunk_id for c in db.query(DBRetrievalCandidate).filter_by(trace_id=trace.id).all()],
        "timestamp": query_run.created_at.isoformat() if (query_run and query_run.created_at) else datetime.datetime.utcnow().isoformat()
    }


# RETRIEVAL DEBUG MODE ROUTE (PHASE 2 AUDIT)
@app.post("/api/v1/search/debug")
async def search_debug(
    request: Request,
    payload: dict[str, Any],
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    query = payload.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="Query string is required")
    filters = payload.get("filters", {})

    embedding_client = get_embedding_client()
    reranker_client = get_reranker_client()

    retriever = HybridRetriever(db, embedding_client, get_qdrant_client())
    assembler = ContextAssembler(embedding_client, reranker_client)

    # 1. Retrieve candidates & extract pre-rerank debug info
    candidates, debug_info = await retriever.retrieve_with_debug(query, filters, limit=50)

    # 2. Assemble context with cross-encoder reranker
    assembled = await assembler.assemble_context(query, candidates, max_tokens_budget=2048)

    # 3. Format post-rerank top candidates
    post_rerank_candidates = []
    for rank, cand in enumerate(assembled.selected_candidates, start=1):
        post_rerank_candidates.append({
            "final_rank": rank,
            "chunk_id": cand.chunk_id,
            "pmcid": cand.pmcid,
            "study_type": cand.study_type,
            "publication_date": cand.publication_date,
            "score": round(cand.score, 5),
            "text_preview": cand.text[:180] + "...",
            "survival_reason": "High cross-encoder relevance score + EBM study-type boost + MMR diversity selection"
        })

    return {
        "raw_query": debug_info.raw_query,
        "normalized_query": debug_info.normalized_query,
        "rewritten_query": debug_info.rewritten_query,
        "expanded_terms": debug_info.expanded_terms,
        "dense_vector_dim": debug_info.dense_vector_dim,
        "sparse_term_count": debug_info.sparse_term_count,
        "pre_rerank_candidate_count": len(debug_info.pre_rerank_candidates),
        "pre_rerank_top_50": debug_info.pre_rerank_candidates,
        "post_rerank_top_selected": post_rerank_candidates,
        "assembled_token_count": assembled.total_tokens
    }

