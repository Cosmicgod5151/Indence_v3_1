import uuid
import logging
import datetime
import hashlib
import time
from typing import Any
from fastapi import FastAPI, Depends, Request, Response, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

# Import utilities and modules
from .core.config import settings
from .core.rate_limiter import rate_limiter
from .core.csrf import verify_csrf_token, generate_csrf_token
from .db.models.models import Base, User, QueryRun, Answer, RetrievalTrace, RetrievalCandidate as DBRetrievalCandidate, AnswerClaim, Citation
from .modules.auth.auth_manager import AuthManager
from .modules.retrieval.retriever import HybridRetriever
from .modules.retrieval.context_assembler import ContextAssembler
from .modules.synthesis.highlighter import SpanHighlighter
from .adapters.models.factory import (
    get_embedding_client,
    get_reranker_client,
    get_llm_client,
    get_entailment_client,
)
from .adapters.qdrant.qdrant_client import QdrantIndexClient
from .modules.pipeline.query_pipeline import QueryPipeline, PipelineOptions, PipelineResult, evidence_sources
from .modules.clinical.phi_guard import scan_phi
from .modules.clinical.case_context import extract_case_facts, build_case_query, missing_data, detect_disease, CaseFact
from .modules.clinical.applicability import build_applicability_matrix

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


@app.get("/api/v1/auth/me")
def whoami(user: User = Depends(get_current_user)):
    """Lets the UI restore a valid session after a page refresh."""
    return {"user_id": user.id, "email": user.email, "role": user.role}


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

    _t_request = time.perf_counter()
    pipeline = build_pipeline(db)
    result = await pipeline.run(query, filters, PipelineOptions(retrieval_mode="full"))
    response = persist_and_build_response(db, user, result, _t_request)
    return response


def build_pipeline(db: Session) -> QueryPipeline:
    return QueryPipeline(
        db,
        get_embedding_client(),
        get_reranker_client(),
        get_llm_client(),
        get_entailment_client(),
        get_qdrant_client(),
    )


def _model_versions(result: PipelineResult) -> dict[str, Any]:
    meta = (result.synth_res.llm_meta if result.synth_res else None) or {}
    return {
        "embedding": "BAAI/bge-m3" if settings.EMBEDDING_PROVIDER.lower() in ("bge-m3", "bge", "real") else "mock",
        "reranker": "BAAI/bge-reranker-v2-m3" if settings.RERANKER_PROVIDER.lower() in ("bge-reranker", "bge", "real") else "mock",
        "llm": meta.get("model") or (settings.LLM_MODEL_NAME if settings.LLM_PROVIDER.lower() in ("openrouter", "real") else "mock"),
        "entailment": (settings.ENTAILMENT_MODEL_NAME or settings.LLM_MODEL_NAME) if settings.ENTAILMENT_PROVIDER.lower() in ("openrouter", "real") else "mock",
        "llm_usage": meta.get("usage"),
        "llm_error": bool(meta.get("error")),
    }


def persist_and_build_response(db: Session, user: User, result: PipelineResult, t_request: float) -> dict[str, Any]:
    """Persist the audit trace (query run, trace, candidates, answer, claims, citations) and build the API response."""
    synth_res = result.synth_res
    final_status = result.final_status
    final_claims = result.final_claims
    abstention_reason = result.abstention_reason
    intent = result.intent
    assembled = result.assembled
    highlighter = SpanHighlighter(db)
    abstained = final_status == "insufficient_evidence"
    models = _model_versions(result)
    wall_s = round(time.perf_counter() - t_request, 4)
    timings = dict(result.timings)
    timings["request_wall_s"] = wall_s

    run_id = str(uuid.uuid4())
    db.add(QueryRun(
        id=run_id,
        user_id=user.id,
        raw_text=result.query,
        normalized_text=getattr(result.debug_info, "rewritten_query", result.query) if result.debug_info else result.query,
        filters_json=result.filters,
        status="abstained" if abstained else "answered",
        latency_ms=int(wall_s * 1000),
    ))

    trace_id = str(uuid.uuid4())
    db.add(RetrievalTrace(
        id=trace_id,
        query_run_id=run_id,
        normalization={
            "query_text": result.query,
            "rewritten_query": getattr(result.debug_info, "rewritten_query", None),
            "expanded_terms": getattr(result.debug_info, "expanded_terms", None),
            "clinical_intent": intent.to_dict(),
        },
        config={"filters": result.filters, "retrieval_mode": result.options.retrieval_mode,
                "retrieval_limit": result.options.retrieval_limit, "max_context_tokens": result.options.max_context_tokens},
        model_versions=models,
        timings=timings,
    ))

    selected_ids = {c.chunk_id for c in (assembled.selected_candidates if assembled else [])}
    for rank, cand in enumerate(result.raw_candidates, start=1):
        db.add(DBRetrievalCandidate(
            trace_id=trace_id,
            chunk_id=cand.chunk_id,
            stage="selected" if cand.chunk_id in selected_ids else "fused",
            rank=rank,
            score=cand.score,
            score_details={
                "dense_rank": cand.dense_rank,
                "sparse_rank": cand.sparse_rank,
                "rerank_score": cand.rerank_score,
                "population_match_score": getattr(cand, "population_match_score", None),
                "hierarchy_rank": getattr(cand, "hierarchy_rank", None),
            },
        ))

    evidence_map = synth_res.evidence_map if synth_res else {}

    def _align_section(section: dict | None) -> dict | None:
        if not section or not section.get("text"):
            return section
        pseudo = {"text": section.get("text", ""), "sentence_ids": section.get("sentence_ids", []),
                  "evidence_ids": section.get("evidence_ids", [])}
        aligned = highlighter.align_claims([pseudo], evidence_map)
        out = dict(section)
        out["aligned_spans"] = aligned[0].get("aligned_spans", []) if aligned else []
        return out

    # SAFETY: when the gate abstains, no generated synthesis text is returned —
    # only the reason and the retrieved sources for manual review.
    if abstained or not synth_res:
        synthesis_response = {k: None for k in ("recommendation", "clinical_reasoning", "alternatives",
                                                 "bottom_line", "clinical_context", "evidence_certainty")}
        synthesis_response["supporting_evidence"] = []
    else:
        synthesis_response = {
            "recommendation": _align_section(synth_res.recommendation),
            "clinical_reasoning": _align_section(synth_res.clinical_reasoning),
            "supporting_evidence": [_align_section(se) or se for se in (synth_res.supporting_evidence or [])],
            "alternatives": _align_section(synth_res.alternatives),
            "bottom_line": _align_section(synth_res.bottom_line),
            "clinical_context": synth_res.clinical_context,
            "evidence_certainty": synth_res.evidence_certainty,
        }

    markdown_lines = []
    if abstained:
        markdown_lines.append(f"**Abstained**: {abstention_reason}")
    else:
        for key, title in (("recommendation", "Recommendation"), ("clinical_reasoning", "Clinical Reasoning"),
                           ("alternatives", "Alternatives"), ("bottom_line", "Bottom Line")):
            sec = synthesis_response.get(key)
            if sec and sec.get("text"):
                markdown_lines.append(f"## {title}\n{sec['text']}")
    rendered_markdown = "\n\n".join(markdown_lines)

    answer_id = str(uuid.uuid4())
    db.add(Answer(
        id=answer_id,
        query_run_id=run_id,
        status="abstained" if abstained else "published",
        rendered_markdown=rendered_markdown,
        confidence="low" if abstained else "high",
        generation_model=models["llm"],
        prompt_version="2.1.0",
        verification_summary={
            "failed_nums_count": sum(1 for c in final_claims if not c.get("numeric_validation_passed", True)),
            "llm_error": synth_res.llm_error if synth_res else None,
        },
    ))

    for ordinal, claim in enumerate(final_claims, start=1):
        claim_id = str(uuid.uuid4())
        db.add(AnswerClaim(
            id=claim_id, answer_id=answer_id, ordinal=ordinal, text=claim.get("text", ""),
            claim_type="direct_answer", is_material=True,
            verification_status="verified" if claim.get("numeric_validation_passed", True) else "unverified",
        ))
        seen_chunks: set[str] = set()
        for s_idx, span in enumerate(claim.get("aligned_spans", []), start=1):
            ref_id = (span.get("sentence_id", "").split(".S") or [""])[0]
            cand_ref = evidence_map.get(ref_id)
            chunk_id = cand_ref.chunk_id if cand_ref else None
            if chunk_id and chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                db.add(Citation(
                    id=str(uuid.uuid4()), claim_id=claim_id, chunk_id=chunk_id, marker_index=s_idx,
                    span_start=span.get("section_start_char", 0), span_end=span.get("section_end_char", 0),
                    verification_status="verified" if claim.get("numeric_validation_passed", True) else "unverified",
                    supporting_text_hash=hashlib.sha256(span.get("text", "").encode("utf-8")).hexdigest(),
                ))
    db.commit()

    selected = assembled.selected_candidates if assembled else []
    clinical_answer = abstention_reason if abstained else (
        (synthesis_response.get("recommendation") or {}).get("text")
        or (synthesis_response.get("bottom_line") or {}).get("text")
        or rendered_markdown
    )

    return {
        "answer_id": answer_id,
        "trace_id": trace_id,
        "status": final_status,
        "clinical_answer": clinical_answer,
        "confidence": "Low" if abstained else "High",
        "extraction_confidence": intent.confidence,
        "supporting_claims": final_claims,
        "rejected_citations": result.rejected_citations,
        "evidence_strength": synthesis_response.get("evidence_certainty"),
        "supporting_trials": list({getattr(c, "hierarchy_label", c.study_type) for c in selected}),
        "supporting_pmcid": list({c.pmcid for c in selected if c.pmcid}),
        "evidence_spans": [s for c in final_claims for s in c.get("aligned_spans", [])],
        "evidence_sources": evidence_sources(result),
        "population_match_score": {c.chunk_id: getattr(c, "population_match_score", 0) for c in selected},
        "clinical_intent": intent.to_dict(),
        "evidence_clusters": [c.to_dict() for c in result.clusters],
        "contradiction_analysis": result.contradiction_info,
        "evidence_map": result.evidence_map,
        "certainty": result.certainty,
        "citation_audit": result.citation_audit,
        "failure_analysis": result.failure_analysis,
        "synthesis": synthesis_response,
        "direct_answer_claims": final_claims,
        "evidence_summary_claims": [] if abstained or not synth_res else synth_res.summary_claims,
        "limitations_claims": [] if abstained or not synth_res else synth_res.limitations,
        "abstention_reason": abstention_reason,
        "llm_error": synth_res.llm_error if synth_res else None,
        "model_versions": models,
        "_debug": {
            "timings": timings,
            "candidate_counts": result.candidate_counts,
            "retrieval_mode": "qdrant_vector" if not get_qdrant_client().is_mock() else "sqlite_fallback",
            "rewritten_query": getattr(result.debug_info, "rewritten_query", None),
            "clinical_intent_filters_applied": {
                "disease_filter": intent.disease is not None,
                "biomarker_filter": intent.biomarker_status is not None,
                "broad_scope": intent.explicitly_broad_scope,
            },
            "excluded_count": len(result.excluded_audit_entries),
        },
    }


# PATIENT CASE MODE ROUTES
@app.post("/api/v1/case/extract", dependencies=[Depends(verify_csrf_token)])
async def case_extract(payload: dict[str, Any], user: User = Depends(get_current_user)):
    """PHI guard (local) -> source-anchored fact extraction -> verification."""
    if not rate_limiter.is_allowed(f"case_extract:{user.id}", limit=15, window_seconds=60):
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded.")
    text = (payload.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Case text is required")
    if len(text) > 12000:
        raise HTTPException(status_code=400, detail="Case text is limited to 12,000 characters")
    phi = scan_phi(text)
    if phi["blocked"]:
        return {"phi": phi, "facts": [], "dropped_facts": [], "conflicts": [], "missing_data": [],
                "disease": None, "extraction_method": None, "traceability": None}
    t0 = time.perf_counter()
    result = await extract_case_facts(text, get_llm_client())
    result["phi"] = phi
    result["latency_s"] = round(time.perf_counter() - t0, 2)
    return result


@app.post("/api/v1/case/search", dependencies=[Depends(verify_csrf_token)])
async def case_search(payload: dict[str, Any], user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Case-aware evidence search with a general-search baseline and Applicability Matrix."""
    if not rate_limiter.is_allowed(f"user_search:{user.id}", limit=30, window_seconds=60):
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded.")
    question = (payload.get("question") or "").strip()
    facts = [f for f in (payload.get("facts") or []) if isinstance(f, dict) and f.get("value")]
    filters = payload.get("filters", {}) or {}
    if not question:
        raise HTTPException(status_code=400, detail="A clinical question is required")
    if not facts:
        raise HTTPException(status_code=400, detail="Confirm at least one patient fact")
    phi = scan_phi(question + "\n" + "\n".join(str(f.get("value")) for f in facts))
    if phi["blocked"]:
        raise HTTPException(status_code=400, detail="Possible identifiers in question/facts: " +
                            ", ".join(f["kind"] for f in phi["findings"]))

    t_request = time.perf_counter()
    case_query, case_intent = build_case_query(question, facts)
    pipeline = build_pipeline(db)
    # Baseline: the same question WITHOUT patient context (retrieval + ranking only).
    general = await pipeline.run(question, filters, PipelineOptions(run_synthesis=False))
    # Case-aware run: patient facts expand the query and set population constraints.
    case = await pipeline.run(case_query, filters, PipelineOptions(intent_override=case_intent))
    response = persist_and_build_response(db, user, case, t_request)

    ev_ids = {c.chunk_id: f"E{i + 1:02d}" for i, c in enumerate(case.assembled.selected_candidates)} if case.assembled else {}
    case_pool = list(case.assembled.selected_candidates) + list(case.ranked_candidates) if case.assembled else list(case.ranked_candidates)
    general_pool = list(general.assembled.selected_candidates) + list(general.ranked_candidates) if general.assembled else list(general.ranked_candidates)
    case_matrix = build_applicability_matrix(case_pool, facts, ev_ids, limit=8)
    general_matrix = build_applicability_matrix(general_pool, facts, {}, limit=8)
    case_ids = {r["pmcid"] for r in case_matrix["rows"]}
    general_ids = {r["pmcid"] for r in general_matrix["rows"]}
    fact_text = " ".join(f"{f.get('name') or ''} {f.get('value')}" for f in facts)
    disease = detect_disease([CaseFact(field=f.get("field", ""), value=str(f.get("value"))) for f in facts], fact_text)
    response["case_mode"] = {
        "case_query": case_query,
        "case_intent": case_intent.to_dict(),
        "applicability": case_matrix,
        "general_baseline": {
            "query": question,
            "applicability": general_matrix,
            "sources": [{"pmcid": r["pmcid"], "title": r["title"], "year": r["year"], "verdict": r["verdict"],
                         "score": r["score"]} for r in general_matrix["rows"]],
        },
        "comparison": {
            "case_mean_applicability": case_matrix["mean_applicability"],
            "general_mean_applicability": general_matrix["mean_applicability"],
            "shared_studies": len(case_ids & general_ids),
            "only_in_case_aware": len(case_ids - general_ids),
            "only_in_general": len(general_ids - case_ids),
            "note": "The general search is kept alongside the case-aware search so that personalisation cannot "
                    "silently hide evidence (anti-confirmation-bias control from the Milestone-1 design).",
        },
        "missing_data": missing_data(disease, [], fact_text),
        "confirmed_facts": facts,
    }
    return response


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
        "generation_model": (answer.generation_model if answer else None) or settings.LLM_MODEL_NAME,
        "model_versions": trace.model_versions,
        "timings": trace.timings,
        "candidates": [
            {"chunk_id": c.chunk_id, "rank": c.rank, "stage": c.stage, "score": c.score, "score_details": c.score_details}
            for c in db.query(DBRetrievalCandidate).filter_by(trace_id=trace.id).order_by(DBRetrievalCandidate.rank).all()
        ],
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

