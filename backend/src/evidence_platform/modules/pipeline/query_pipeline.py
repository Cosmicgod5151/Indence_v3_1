"""Indence query pipeline — the single code path shared by the API and the
evaluation harness.

Stages (each timed):
    1  clinical intent extraction            (ClinicalIntentExtractor)
    2  hybrid retrieval: dense + sparse, RRF  (HybridRetriever)
    3  population matching + hard filters     (PopulationMatchEvaluator)
    4  evidence-hierarchy ranking              (EvidenceHierarchyRanker)
    5  cross-encoder rerank + MMR assembly     (ContextAssembler)
    6  evidence grouping                       (EvidenceGrouper)
    7  contradiction analysis                  (ContradictionChecker)
    8  LLM synthesis (structured JSON)         (AnswerSynthesizer)
    9  numeric + population safety validation  (validators)
    10 span alignment                          (SpanHighlighter)
    11 NLI grounding + abstention gate         (AbstentionLayer)
    12 citation audit                          (CitationAuditor)

``PipelineOptions.retrieval_mode`` lets the evaluation harness ablate layers:
    "sparse" | "dense" | "hybrid"  -> retrieval channel only, fusion order
    "hybrid_rerank"                -> hybrid + cross-encoder ordering only
    "full"                         -> complete Indence pipeline (default)
"""
from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from ..retrieval.retriever import HybridRetriever, RetrievalCandidate
from ..retrieval.context_assembler import ContextAssembler, AssembledContext
from ..synthesis.synthesizer import AnswerSynthesizer, SynthesizedAnswer
from ..synthesis.validator import NumericDosageValidator, ClinicalSafetyValidator
from ..synthesis.highlighter import SpanHighlighter
from ..synthesis.abstention import AbstentionLayer
from ..synthesis.evidence_grouper import EvidenceGrouper
from ..synthesis.contradiction_checker import ContradictionChecker
from ..synthesis.citation_auditor import CitationAuditor
from ..synthesis.effect_map import build_evidence_map
from ..synthesis.certainty import assess_certainty
from ..clinical.phi_guard import scan_phi
import re
from ..clinical.clinical_intent import ClinicalIntentExtractor, ClinicalQueryIntent
from ..clinical.population_matcher import PopulationMatchEvaluator
from ..clinical.evidence_hierarchy import EvidenceHierarchyRanker

logger = logging.getLogger("evidence_platform.pipeline")

PIPELINE_MODES = ("sparse", "dense", "hybrid", "hybrid_rerank", "full")


@dataclass
class PipelineOptions:
    retrieval_mode: str = "full"
    retrieval_limit: int = 50
    max_context_tokens: int = 2048
    run_synthesis: bool = True
    # Optional pre-built intent (Case Mode builds it from confirmed patient facts).
    intent_override: ClinicalQueryIntent | None = None
    # Text whose drug names must appear in the evidence (Case Mode passes the
    # clinician's question, not the patient's prior-therapy context).
    entity_check_text: str | None = None


@dataclass
class PipelineResult:
    query: str
    filters: dict[str, Any]
    options: PipelineOptions
    intent: ClinicalQueryIntent
    raw_candidates: list[RetrievalCandidate]
    ranked_candidates: list[RetrievalCandidate]          # final ranking used for eval
    excluded_audit_entries: list[dict[str, Any]] = field(default_factory=list)
    assembled: AssembledContext | None = None
    clusters: list[Any] = field(default_factory=list)
    contradiction_info: dict[str, Any] = field(default_factory=dict)
    synth_res: SynthesizedAnswer | None = None
    final_status: str = "insufficient_evidence"
    final_claims: list[dict[str, Any]] = field(default_factory=list)
    abstention_reason: str | None = None
    rejected_citations: list[dict[str, Any]] = field(default_factory=list)
    citation_audit: list[dict[str, Any]] = field(default_factory=list)
    failure_analysis: list[dict[str, Any]] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    candidate_counts: dict[str, int] = field(default_factory=dict)
    debug_info: Any = None
    evidence_map: dict[str, Any] = field(default_factory=dict)
    certainty: dict[str, Any] = field(default_factory=dict)
    safety_gate: str | None = None   # phi | out_of_scope | low_relevance | unknown_entity
    model_trace: dict[str, Any] = field(default_factory=dict)  # which model performed which step


class QueryPipeline:
    """Builds every stage once and runs the 12-stage evidence pipeline."""

    def __init__(self, db: Session, embedding_client, reranker_client, llm_client, entailment_client, qdrant_client):
        self.db = db
        self.retriever = HybridRetriever(db, embedding_client, qdrant_client)
        self.assembler = ContextAssembler(embedding_client, reranker_client)
        self.reranker_client = reranker_client
        self.synthesizer = AnswerSynthesizer(llm_client)
        self.numeric_validator = NumericDosageValidator()
        self.highlighter = SpanHighlighter(db)
        self.abstention_layer = AbstentionLayer(entailment_client)
        self.intent_extractor = ClinicalIntentExtractor()
        self.population_evaluator = PopulationMatchEvaluator()
        self.hierarchy_ranker = EvidenceHierarchyRanker()
        self.evidence_grouper = EvidenceGrouper()
        self.contradiction_checker = ContradictionChecker()
        self.citation_auditor = CitationAuditor()
        self.safety_validator = ClinicalSafetyValidator()

    async def run(self, query: str, filters: dict[str, Any] | None = None,
                  options: PipelineOptions | None = None) -> PipelineResult:
        options = options or PipelineOptions()
        if options.retrieval_mode not in PIPELINE_MODES:
            raise ValueError(f"retrieval_mode must be one of {PIPELINE_MODES}")
        filters = filters or {}
        timings: dict[str, float] = {}
        counts: dict[str, int] = {}
        t_wall = time.perf_counter()

        def _tick(name: str, t0: float) -> None:
            timings[name] = round(time.perf_counter() - t0, 4)

        # 1. Intent
        t0 = time.perf_counter()
        intent = options.intent_override or self.intent_extractor.extract_intent(query)
        _tick("intent_extraction_s", t0)

        def _gate(kind: str, reason: str, raw: list | None = None, dbg: Any = None) -> PipelineResult:
            res = PipelineResult(query=query, filters=filters, options=options, intent=intent,
                                 raw_candidates=raw or [], ranked_candidates=list(raw or []),
                                 timings=timings, candidate_counts=counts, debug_info=dbg)
            res.final_status, res.abstention_reason, res.safety_gate = "insufficient_evidence", reason, kind
            timings["total_e2e_s"] = round(time.perf_counter() - t_wall, 4)
            logger.info("Safety gate '%s' fired: %s", kind, reason)
            return res

        # Gate 1: identifiable patient data never leaves the machine.
        phi = scan_phi(query)
        if phi["blocked"]:
            return _gate("phi", "The question appears to contain patient identifiers ("
                         + ", ".join(sorted({f['kind'] for f in phi['findings']}))
                         + "). It was blocked locally before any model call; please remove them.")
        # Gate 2: out-of-scope (no clinical entity or vocabulary at all).
        if options.retrieval_mode == "full" and out_of_scope(query, intent):
            return _gate("out_of_scope", "This question is outside Indence's scope (clinical evidence from the "
                         "oncology literature), so no answer is generated.")

        # 2. Retrieval
        channel = options.retrieval_mode if options.retrieval_mode in ("sparse", "dense", "hybrid") else "hybrid"
        t0 = time.perf_counter()
        raw_candidates, debug_info = await self.retriever.retrieve_with_debug(
            query, filters, limit=options.retrieval_limit, clinical_intent=intent, mode=channel
        )
        _tick("retrieval_s", t0)
        counts["pre_filter"] = len(raw_candidates)

        result = PipelineResult(
            query=query, filters=filters, options=options, intent=intent,
            raw_candidates=raw_candidates, ranked_candidates=list(raw_candidates),
            timings=timings, candidate_counts=counts, debug_info=debug_info,
        )

        # Retrieval-only ablations stop here.
        if options.retrieval_mode in ("sparse", "dense", "hybrid"):
            timings["total_e2e_s"] = round(time.perf_counter() - t_wall, 4)
            return result

        if options.retrieval_mode == "hybrid_rerank":
            t0 = time.perf_counter()
            scores = await self.reranker_client.rerank(query, [c.text for c in raw_candidates])
            for c, s in zip(raw_candidates, scores):
                c.rerank_score = float(s)
            result.ranked_candidates = sorted(raw_candidates, key=lambda c: c.rerank_score or 0.0, reverse=True)
            _tick("rerank_s", t0)
            timings["total_e2e_s"] = round(time.perf_counter() - t_wall, 4)
            return result

        # 3. Population matching / hard filters
        t0 = time.perf_counter()
        matched, excluded = self.population_evaluator.filter_and_score_candidates(intent, raw_candidates)
        _tick("population_matching_s", t0)
        counts["post_hard_filter"] = len(matched)
        result.excluded_audit_entries = excluded

        # 4. Evidence hierarchy
        t0 = time.perf_counter()
        ranked = self.hierarchy_ranker.rank_candidates(matched)
        _tick("evidence_hierarchy_s", t0)

        # 5. Rerank + MMR assembly
        t0 = time.perf_counter()
        assembled = await self.assembler.assemble_context(query, ranked, max_tokens_budget=options.max_context_tokens)
        _tick("context_assembly_s", t0)
        counts["post_assembly"] = len(assembled.selected_candidates)
        result.assembled = assembled
        result.ranked_candidates = list(assembled.ranked_candidates) or list(assembled.selected_candidates)

        # Gate 3: evidence relevance and entity coverage (before any LLM call).
        if options.run_synthesis:
            strong = sum(1 for c in assembled.selected_candidates if (c.rerank_score or 0) >= RELEVANCE_MIN)
            missing = unknown_entities(options.entity_check_text or query, raw_candidates)
            if missing:
                g = _gate("unknown_entity", "No retrieved evidence mentions " + ", ".join(missing)
                          + ". Indence does not answer about treatments it has no evidence for.", raw_candidates, debug_info)
                g.assembled, g.ranked_candidates = assembled, result.ranked_candidates
                return g
            if strong < MIN_RELEVANT_PASSAGES:
                g = _gate("low_relevance", f"Only {strong} retrieved passage(s) are strongly relevant to the question "
                          f"(cross-encoder ≥ {RELEVANCE_MIN}); at least {MIN_RELEVANT_PASSAGES} are required to answer.",
                          raw_candidates, debug_info)
                g.assembled, g.ranked_candidates = assembled, result.ranked_candidates
                return g

        # 6. Grouping
        t0 = time.perf_counter()
        clusters = self.evidence_grouper.group_candidates(assembled.selected_candidates)
        _tick("evidence_grouping_s", t0)
        result.clusters = clusters

        # 7. Contradictions
        t0 = time.perf_counter()
        contradiction_info = self.contradiction_checker.check_contradictions(clusters, assembled.selected_candidates)
        _tick("contradiction_check_s", t0)

        # 7b. Quantitative Evidence Map (effect sizes + heterogeneity) and the
        #     GRADE-inspired certainty rating. Quantitative conflicts are merged
        #     into the contradiction analysis so the generator must address them.
        t0 = time.perf_counter()
        ev_ids = {c.chunk_id: f"E{i + 1:02d}" for i, c in enumerate(assembled.selected_candidates)}
        map_pool = list(assembled.ranked_candidates[:25]) or list(assembled.selected_candidates)
        for c in assembled.selected_candidates:          # always include the bundle itself
            if c not in map_pool:
                map_pool.append(c)
        evidence_map = build_evidence_map(map_pool, ev_ids)
        contradiction_info = merge_quantitative_conflicts(contradiction_info, evidence_map)
        result.evidence_map = evidence_map
        result.certainty = assess_certainty(assembled.selected_candidates, evidence_map, contradiction_info)
        _tick("evidence_map_certainty_s", t0)
        result.contradiction_info = contradiction_info

        if not options.run_synthesis:
            timings["total_e2e_s"] = round(time.perf_counter() - t_wall, 4)
            return result

        # 8. Synthesis
        t0 = time.perf_counter()
        synth_res = await self.synthesizer.synthesize(
            query, assembled, intent=intent, clusters=clusters, contradiction_info=contradiction_info
        )
        _tick("synthesis_s", t0)
        counts["synthesis_claims"] = len(synth_res.direct_claims)
        result.synth_res = synth_res

        # 9. Numeric + population safety validation
        t0 = time.perf_counter()
        validated = self.numeric_validator.validate_answer(synth_res.direct_claims, synth_res.evidence_map)
        safe_claims, rejected_citations = self.safety_validator.validate_clinical_safety(
            intent, validated, synth_res.evidence_map
        )
        result.rejected_citations = rejected_citations

        # 10. Span alignment
        aligned = self.highlighter.align_claims(safe_claims, synth_res.evidence_map)
        _tick("validation_alignment_s", t0)

        # 11. NLI grounding + abstention gate
        t0 = time.perf_counter()
        final_status, final_claims, abstention_reason = await self.abstention_layer.verify_and_calibrate(
            status=synth_res.status,
            direct_claims=aligned,
            evidence_map=synth_res.evidence_map,
            abstention_reason=synth_res.abstention_reason,
            candidates=assembled.selected_candidates,
            intent=intent,
        )
        _tick("grounding_s", t0)
        counts["final_claims"] = len(final_claims)
        result.model_trace = {
            "generator": {**(synth_res.llm_meta or {}), "task": "Evidence synthesis (structured JSON answer)"},
            "verifier": {**(getattr(self.abstention_layer, "last_verifier_meta", {}) or {}),
                         "task": "Claim-by-claim verification against cited passages"},
        }
        result.final_status, result.final_claims, result.abstention_reason = final_status, final_claims, abstention_reason

        # 12. Citation audit
        t0 = time.perf_counter()
        result.citation_audit, result.failure_analysis = self.citation_auditor.audit_citations(
            intent, assembled.selected_candidates, excluded, final_claims
        )
        _tick("citation_audit_s", t0)

        timings["total_e2e_s"] = round(time.perf_counter() - t_wall, 4)
        logger.info("Pipeline complete: status=%s | %d claims | %.2fs wall",
                    final_status, len(final_claims), timings["total_e2e_s"])
        return result


# ----------------------------------------------------------------------------
# Pre-generation safety gates (deterministic; no LLM call is made when one fires)
# ----------------------------------------------------------------------------
MEDICAL_VOCAB = re.compile(
    r"cancer|tumou?r|carcinoma|sarcoma|lymphoma|leuk|melanoma|metasta|oncolog|neoplas|malignan|chemo|radiother|"
    r"immunother|therap|treat|drug|dose|trial|surviv|mortality|recurr|remission|response rate|patient|clinical|"
    r"biomarker|mutation|her2|egfr|alk|kras|brca|pd-?l1|msi|hazard ratio|placebo|adjuvant|surgery|resection|"
    r"toxicit|adverse|guideline|prognos|diagnos|screening|staging|stage\s+[iv1-4]", re.I)
DRUG_TOKEN = re.compile(
    r"\b[a-z]{3,}(?:tinib|inib|zumab|ximab|mumab|umab|ciclib|parib|lisib|tecan|platin|rafenib|metinib|degib|sertib|"
    r"clax|zomib|lutamide|terone|rasib|tansine|taxel|rubicin|citabine|fosfamide|trexate|mustine|limus)\b", re.I)
RELEVANCE_MIN = 0.30     # cross-encoder probability; need >= 2 passages above it
MIN_RELEVANT_PASSAGES = 2


def out_of_scope(query: str, intent: ClinicalQueryIntent) -> bool:
    has_entities = any([intent.disease, intent.biomarker_status, intent.treatment, intent.clinical_outcome])
    return not has_entities and not MEDICAL_VOCAB.search(query) and not DRUG_TOKEN.search(query)


def unknown_entities(query: str, candidates: list[Any]) -> list[str]:
    """Drug names in the question that occur in NONE of the retrieved passages/titles."""
    drugs = {m.group(0).lower() for m in DRUG_TOKEN.finditer(query)}
    if not drugs:
        return []
    corpus = " ".join(((getattr(c, "title", "") or "") + " " + (c.text or "")).lower() for c in candidates)
    return sorted(d for d in drugs if d not in corpus)


def pmc_url(pmcid: str | None) -> str | None:
    if not pmcid:
        return None
    pid = pmcid if pmcid.upper().startswith("PMC") else f"PMC{pmcid}"
    return f"https://pmc.ncbi.nlm.nih.gov/articles/{pid}/"


def evidence_sources(result: PipelineResult) -> list[dict[str, Any]]:
    """Citation metadata for every evidence passage given to the generator (E01, E02, ...)."""
    out: list[dict[str, Any]] = []
    if result.synth_res and result.synth_res.evidence_map:
        items = list(result.synth_res.evidence_map.items())
    elif result.assembled:
        items = [(f"E{i + 1:02d}", c) for i, c in enumerate(result.assembled.selected_candidates)]
    else:
        items = []
    for ev_id, c in items:
        out.append({
            "evidence_id": ev_id,
            "chunk_id": c.chunk_id,
            "pmcid": c.pmcid,
            "title": getattr(c, "title", "") or "",
            "year": (c.publication_date or "")[:4] or None,
            "study_type": c.study_type,
            "hierarchy_label": getattr(c, "hierarchy_label", "") or c.study_type,
            "hierarchy_rank": getattr(c, "hierarchy_rank", None),
            "section": getattr(c, "section", "") or "",
            "rerank_score": round(c.rerank_score, 4) if getattr(c, "rerank_score", None) is not None else None,
            "population_match_score": getattr(c, "population_match_score", 0),
            "population_match_reason": getattr(c, "population_match_reason", None),
            "dense_rank": getattr(c, "dense_rank", None),
            "sparse_rank": getattr(c, "sparse_rank", None),
            "url": pmc_url(c.pmcid),
            "snippet": c.text[:400],
        })
    return out


def merge_quantitative_conflicts(contradiction_info: dict[str, Any], evidence_map: dict[str, Any]) -> dict[str, Any]:
    """Add statistical conflicts from the evidence map to the narrative contradiction analysis."""
    info = dict(contradiction_info or {})
    items = list(info.get("contradictions") or [])
    groups = {(g["measure"], g["endpoint"]): g for g in evidence_map.get("groups", [])}
    for qc in evidence_map.get("quantitative_conflicts", []):
        g = groups.get((qc["measure"], qc["endpoint"]))
        ests = (g or {}).get("estimates", [])
        items.append({
            "type": "Quantitative conflict (effect sizes)",
            "severity": "high" if "OPPOSITE" in qc["description"] else "moderate",
            "description": f"{qc['endpoint']} — {qc['description']}.",
            "claim_in_conflict": f"Direction/magnitude of the {qc['measure']} for {qc['endpoint'].lower()}",
            "conflicting_evidence_ids": [{"pmcid": e["pmcid"], "study_type": e["study_type"]} for e in ests[:6]],
            "reason_for_difference": "Studies may differ in population, comparator, line of therapy or follow-up; "
                                     "see the Evidence Map for each estimate and its source sentence.",
            "evidence_quality": [{"pmcid": e["pmcid"], "study_type": e["study_type"],
                                  "strength_tier": f"{e['measure']} {e['estimate']} ({e['ci_low']}–{e['ci_high']})"}
                                 for e in ests[:6]],
            "clinical_interpretation": "Do not rely on a single pooled conclusion; review each study's population "
                                       "(manual review recommended).",
            "action": "Inspect the forest plot and source sentences.",
        })
    info["contradictions"] = items
    info["contradiction_count"] = len(items)
    info["has_contradiction"] = bool(items)
    info["quantitative_conflict_count"] = len(evidence_map.get("quantitative_conflicts", []))
    if items and not info.get("consensus_summary", "").startswith("Detected"):
        info["consensus_summary"] = f"Detected {len(items)} evidence conflict(s) requiring clinician review."
    return info
