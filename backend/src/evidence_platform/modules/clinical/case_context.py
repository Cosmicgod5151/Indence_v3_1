"""Patient Case Mode — source-anchored clinical fact extraction (M1 Feature B).

Workflow (FR-18/19/20, NFR-12/13):
  1. PHI guard (local) — identifiable text never reaches a model provider.
  2. Extraction — LLM returns candidate facts as JSON, each with a verbatim
     ``source_quote``; a deterministic regex extractor is the offline fallback.
  3. Span verification — every quote must be found in the case text and must
     contain the extracted value; untraceable facts are DROPPED and reported
     (an untraceable fact is a failed case, never silently kept).
  4. Conflict + missing-data analysis — contradictory values are flagged (not
     resolved); decision-critical variables that are absent are listed.
  5. The clinician confirms/edits the facts; only confirmed facts build the
     case-aware query and population constraints.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, asdict, field
from typing import Any

from .clinical_intent import ClinicalIntentExtractor, ClinicalQueryIntent

FIELDS = {
    "cancer_type": "Primary cancer / disease",
    "histology": "Histology / subtype",
    "stage": "Stage / extent (e.g. stage IV, metastatic, locally advanced)",
    "metastatic_site": "Site of metastasis (one fact per site)",
    "biomarker": "Biomarker / molecular result (one fact per marker: EGFR, ALK, ROS1, KRAS, BRAF, PD-L1, HER2, ER, PR, BRCA1/2, MSI/MMR, PIK3CA, NTRK, RET, MET, Ki-67)",
    "prior_therapy": "Prior treatment received (one fact per regimen/line, include outcome if stated)",
    "line_of_therapy": "Treatment line now being considered",
    "performance_status": "ECOG or Karnofsky performance status",
    "age": "Age in years",
    "sex": "Sex",
    "lab": "Laboratory value (one fact per test, with unit: creatinine, eGFR, bilirubin, ALT, AST, ANC, platelets, haemoglobin, LVEF)",
    "comorbidity": "Relevant comorbidity",
    "medication": "Current non-cancer medication",
}

EXTRACTION_PROMPT = """You extract structured clinical facts from a DE-IDENTIFIED oncology case summary for evidence retrieval.

Rules:
- Extract ONLY facts explicitly written in the text. Never infer, never guess, never add knowledge.
- Every fact MUST include "source_quote": the shortest EXACT, VERBATIM substring of the case text that states it
  (copy characters exactly, including numbers, units and punctuation).
- "value" must be the fact as written (e.g. "EGFR exon 19 deletion", "PD-L1 TPS 60%", "ECOG 1", "creatinine 1.1 mg/dL").
- "name": for biomarker/lab facts, the marker or test name (e.g. "EGFR", "PD-L1", "creatinine"); otherwise null.
- "date": the date/time phrase attached to the fact if written (e.g. "03/2024", "2 months ago"), else null.
- If the same item appears with DIFFERENT values (e.g. two PD-L1 results), output BOTH facts; do not choose.
- Allowed "field" values: {fields}

Return ONLY JSON: {{"facts": [{{"field": "...", "name": "...|null", "value": "...", "unit": "...|null", "date": "...|null", "source_quote": "..."}}]}}"""

# Decision-critical variables per disease: absence is reported, not guessed.
MISSING_DATA_RULES: dict[str, list[tuple[str, list[str], str]]] = {
    "Non-Small Cell Lung Cancer": [
        ("EGFR", ["egfr"], "EGFR status determines eligibility for EGFR-TKI trials (e.g. osimertinib)."),
        ("ALK", ["alk"], "ALK rearrangement status determines ALK-inhibitor evidence."),
        ("ROS1", ["ros1"], "ROS1 fusion status is a standard first-line biomarker."),
        ("PD-L1", ["pd-l1", "pdl1", "tps"], "PD-L1 TPS stratifies immunotherapy trials (≥50% vs 1–49%)."),
        ("KRAS", ["kras"], "KRAS G12C status determines sotorasib/adagrasib evidence."),
        ("Performance status", ["ecog", "karnofsky", "performance status"], "Most trials enrolled ECOG 0–1 only."),
    ],
    "Breast Cancer": [
        ("HER2", ["her2", "erbb2"], "HER2 status (positive / low / 0) selects entirely different trial populations."),
        ("ER/PR", ["er ", "er+", "er-", "estrogen", "progesterone", "pr ", "hormone receptor", "hr+", "hr-"], "Hormone-receptor status defines endocrine-therapy evidence."),
        ("BRCA", ["brca"], "Germline BRCA status determines PARP-inhibitor evidence."),
        ("PIK3CA/ESR1", ["pik3ca", "esr1"], "PIK3CA/ESR1 mutations select targeted options after endocrine therapy."),
        ("Performance status", ["ecog", "karnofsky", "performance status"], "Most trials enrolled ECOG 0–1 only."),
    ],
    "Colorectal Cancer": [
        ("RAS (KRAS/NRAS)", ["ras", "kras", "nras"], "RAS status determines anti-EGFR (cetuximab/panitumumab) eligibility."),
        ("BRAF", ["braf"], "BRAF V600E selects encorafenib-based evidence."),
        ("MSI/MMR", ["msi", "mmr", "mismatch repair"], "MSI-H/dMMR selects immunotherapy evidence."),
        ("Tumour sidedness", ["left-sided", "right-sided", "left sided", "right sided", "sigmoid", "rectal", "caecum", "cecum", "ascending", "descending"], "Primary-tumour side modifies anti-EGFR benefit."),
        ("HER2", ["her2"], "HER2 amplification selects HER2-targeted evidence."),
    ],
    "Ovarian Cancer": [
        ("BRCA / HRD", ["brca", "hrd", "homologous recombination"], "BRCA/HRD status determines PARP-inhibitor maintenance evidence."),
        ("Platinum sensitivity", ["platinum-sensitive", "platinum sensitive", "platinum-resistant", "platinum resistant", "platinum-free"], "Platinum-free interval defines second-line evidence."),
    ],
    "Prostate Cancer": [
        ("Castration status", ["castration", "crpc", "hormone-sensitive", "castrate"], "Castration-sensitive vs resistant disease selects different trials."),
        ("PSA", ["psa"], "PSA level/kinetics is a key response and eligibility measure."),
        ("HRR/BRCA", ["brca", "hrr", "atm"], "HRR mutations determine PARP-inhibitor evidence."),
    ],
}
_GENERIC_MISSING = [
    ("Stage / extent", ["stage", "metasta", "advanced", "localized", "localised"], "Stage defines which trials are applicable."),
    ("Performance status", ["ecog", "karnofsky", "performance status"], "Most trials enrolled ECOG 0–1 only."),
]


@dataclass
class CaseFact:
    field: str
    value: str
    name: str | None = None
    unit: str | None = None
    date: str | None = None
    source_quote: str = ""
    start: int | None = None
    end: int | None = None
    verified: bool = False
    method: str = "llm"
    conflict_with: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = s.replace("–", "-").replace("—", "-").replace("≥", ">=").replace("≤", "<=")
    return re.sub(r"\s+", " ", s).strip()


def locate_quote(text: str, quote: str) -> tuple[int, int] | None:
    """Find a quote in the text: exact match first, then whitespace/case/dash-insensitive."""
    if not quote:
        return None
    i = text.find(quote)
    if i >= 0:
        return i, i + len(quote)
    # Build a normalised copy of text with an index map back to original offsets.
    norm_chars, index = [], []
    prev_space = False
    for j, ch in enumerate(unicodedata.normalize("NFKC", text)):
        c = ch.lower()
        c = {"–": "-", "—": "-"}.get(c, c)
        if c.isspace():
            if prev_space:
                continue
            c, prev_space = " ", True
        else:
            prev_space = False
        norm_chars.append(c)
        index.append(j)
    hay = "".join(norm_chars)
    needle = _norm(quote)
    k = hay.find(needle)
    if k < 0:
        return None
    return index[k], index[k + len(needle) - 1] + 1


def _value_supported(value: str, quote: str) -> bool:
    """Every number in the value must appear in the quote, and at least one
    substantive token of the value must appear in it."""
    v, q = _norm(value), _norm(quote)
    nums = re.findall(r"\d+(?:\.\d+)?", v)
    if any(n not in q for n in nums):
        return False
    tokens = [t for t in re.findall(r"[a-z0-9+\-]{2,}", v) if t not in {"the", "and", "with", "of", "in"}]
    return not tokens or any(t in q for t in tokens)


def verify_facts(text: str, raw_facts: list[dict[str, Any]], method: str = "llm") -> tuple[list[CaseFact], list[dict[str, Any]]]:
    kept: list[CaseFact] = []
    dropped: list[dict[str, Any]] = []
    for rf in raw_facts or []:
        fld = (rf.get("field") or "").strip()
        val = str(rf.get("value") or "").strip()
        quote = str(rf.get("source_quote") or "").strip()
        if fld not in FIELDS or not val:
            dropped.append({**rf, "drop_reason": "unknown field or empty value"})
            continue
        loc = locate_quote(text, quote)
        if loc is None:
            dropped.append({**rf, "drop_reason": "source_quote not found verbatim in the case text"})
            continue
        if not _value_supported(val, text[loc[0]:loc[1]]):
            dropped.append({**rf, "drop_reason": "value (or its numbers) not present in the quoted text"})
            continue
        kept.append(CaseFact(field=fld, value=val, name=(rf.get("name") or None), unit=(rf.get("unit") or None),
                             date=(rf.get("date") or None), source_quote=text[loc[0]:loc[1]],
                             start=loc[0], end=loc[1], verified=True, method=method))
    # de-duplicate identical facts
    uniq: list[CaseFact] = []
    for f in kept:
        if not any(u.field == f.field and _norm(u.value) == _norm(f.value) for u in uniq):
            uniq.append(f)
    return uniq, dropped


def flag_conflicts(facts: list[CaseFact]) -> list[dict[str, Any]]:
    """Same biomarker/lab/PS with different values -> conflict (shown, never auto-resolved)."""
    conflicts = []
    keyed: dict[tuple[str, str], list[int]] = {}
    for i, f in enumerate(facts):
        if f.field in ("biomarker", "lab", "performance_status", "stage"):
            key = (f.field, _norm(f.name or f.field))
            keyed.setdefault(key, []).append(i)
    for (fld, name), idxs in keyed.items():
        values = {_norm(facts[i].value) for i in idxs}
        if len(idxs) > 1 and len(values) > 1:
            for i in idxs:
                facts[i].conflict_with = [j for j in idxs if j != i]
            conflicts.append({
                "field": fld, "name": name,
                "values": [{"value": facts[i].value, "date": facts[i].date, "quote": facts[i].source_quote} for i in idxs],
                "message": f"Conflicting {name} values in the case documents — confirm which is current; Indence will not choose.",
            })
    return conflicts


def detect_disease(facts: list[CaseFact], text: str) -> str | None:
    probe = " ".join(f.value for f in facts if f.field in ("cancer_type", "histology")) or text
    return ClinicalIntentExtractor().extract_intent(probe).disease


def missing_data(disease: str | None, facts: list[CaseFact], text: str) -> list[dict[str, str]]:
    hay = _norm(text) + " " + " ".join(_norm(f"{f.name or ''} {f.value}") for f in facts)
    rules = MISSING_DATA_RULES.get(disease or "", []) or []
    seen = {r[0] for r in rules}
    rules = rules + [r for r in _GENERIC_MISSING if r[0] not in seen]
    # Oncogenic drivers in NSCLC are (near-)mutually exclusive: once a positive
    # driver is documented, the other driver tests are no longer decision-critical.
    driver_found = bool(re.search(
        r"egfr[^.;,]{0,25}(?:exon|l858r|deletion|mutat|positive)|alk[^.;,]{0,20}(?:positive|rearranged|fusion detected)"
        r"|ros1[^.;,]{0,15}positive|kras[^.;,]{0,10}g12", hay))
    drivers = {"EGFR", "ALK", "ROS1", "KRAS"}
    out = []
    for label, terms, why in rules:
        if driver_found and disease == "Non-Small Cell Lung Cancer" and label in drivers:
            continue
        if not any(t in hay for t in terms):
            out.append({"variable": label, "why_it_matters": why})
    return out


# --------------------------- regex fallback extractor ---------------------------
_FALLBACK_PATTERNS: list[tuple[str, str | None, re.Pattern]] = [
    ("performance_status", "ECOG", re.compile(r"\bECOG(?:\s*(?:PS|performance status))?\s*(?:of\s*|:\s*|=\s*)?[0-4](?:\s*-\s*[0-4])?\b", re.I)),
    ("histology", None, re.compile(r"\b(?:adenocarcinoma|squamous cell carcinoma|ductal carcinoma|lobular carcinoma|serous carcinoma|small cell carcinoma)\b", re.I)),
    ("biomarker", "ROS1", re.compile(r"\bROS1\b[^.;\n]{0,20}?(?:positive|negative|fusion\s+\w+|rearrange\w*)", re.I)),
    ("stage", None, re.compile(r"\b(?:stage\s*(?:IV|III|II|I)[ABC]?\d?|metastatic|locally advanced)\b", re.I)),
    ("age", None, re.compile(r"\b\d{2}[- ](?:year|yr)s?[- ]old\b", re.I)),
    ("biomarker", "PD-L1", re.compile(r"\bPD-?L1\b[^.;\n]{0,25}?\d{1,3}\s*%", re.I)),
    ("biomarker", "EGFR", re.compile(r"\bEGFR\b[^.;\n]{0,40}?(?:exon\s*\d+[^.;,\n]{0,20}|L858R|T790M|mutation|mutant|positive|negative|wild[- ]type)", re.I)),
    ("biomarker", "ALK", re.compile(r"\bALK\b[^.;\n]{0,25}?(?:(?:rearrange\w*|fusion)\s+)?(?:positive|negative|not detected|detected)|\bALK\b[^.;\n]{0,10}?(?:rearrange\w*|fusion)", re.I)),
    ("biomarker", "HER2", re.compile(r"\bHER2\b[^.;\n]{0,25}?(?:positive|negative|low|0|1\+|2\+|3\+|amplified)", re.I)),
    ("biomarker", "KRAS", re.compile(r"\bKRAS\b[^.;\n]{0,25}?(?:G12C|G12D|G12V|mutation|mutant|wild[- ]type)", re.I)),
    ("biomarker", "BRCA", re.compile(r"\b(?:g)?BRCA[12]?\b[^.;\n]{0,25}?(?:mutation|mutated|positive|negative|pathogenic)", re.I)),
    ("biomarker", "MSI/MMR", re.compile(r"\b(?:MSI-H|MSS|dMMR|pMMR|microsatellite (?:instability|stable)[^.;\n]{0,10})", re.I)),
    ("lab", "creatinine", re.compile(r"\bcreatinine\b[^.;\n]{0,15}?\d+(?:\.\d+)?\s*(?:mg/dL|µmol/L|umol/L)", re.I)),
    ("lab", "bilirubin", re.compile(r"\bbilirubin\b[^.;\n]{0,15}?\d+(?:\.\d+)?\s*(?:mg/dL|µmol/L|umol/L)", re.I)),
]


def regex_extract(text: str) -> list[dict[str, Any]]:
    raw: list[dict[str, Any]] = []
    intent = ClinicalIntentExtractor().extract_intent(text)
    if intent.disease:
        for rx_txt, label in ClinicalIntentExtractor.DISEASE_PATTERNS:
            if label == intent.disease:
                m = re.search(rx_txt, text, re.I)
                if m:
                    raw.append({"field": "cancer_type", "value": m.group(0), "source_quote": m.group(0)})
                break
    for fld, name, rx in _FALLBACK_PATTERNS:
        for m in rx.finditer(text):
            raw.append({"field": fld, "name": name, "value": m.group(0).strip(), "source_quote": m.group(0).strip()})
    return raw


async def extract_case_facts(text: str, llm_client: Any | None) -> dict[str, Any]:
    """Run the full extraction workflow. ``llm_client`` must expose complete_json()."""
    method, error = "llm", None
    raw: list[dict[str, Any]] = []
    if llm_client is not None and hasattr(llm_client, "complete_json"):
        try:
            prompt = EXTRACTION_PROMPT.format(fields=", ".join(f"{k} ({v})" for k, v in FIELDS.items()))
            parsed, meta = await llm_client.complete_json(prompt, f"CASE TEXT:\n{text}", max_tokens=6000)
            raw = parsed.get("facts") or []
            method = f"llm:{meta.get('model')}"
        except Exception as e:  # provider unavailable -> deterministic fallback
            error = str(e)[:300]
    if not raw:
        raw = regex_extract(text)
        method = "regex-fallback"
    facts, dropped = verify_facts(text, raw, method="regex" if method == "regex-fallback" else "llm")
    conflicts = flag_conflicts(facts)
    disease = detect_disease(facts, text)
    return {
        "facts": [f.to_dict() for f in facts],
        "dropped_facts": dropped,
        "conflicts": conflicts,
        "disease": disease,
        "missing_data": missing_data(disease, facts, text),
        "extraction_method": method,
        "extraction_error": error,
        "traceability": {
            "proposed": len(raw), "verified": len(facts), "dropped": len(dropped),
            "rate": round(len(facts) / len(raw), 3) if raw else None,
        },
    }


# --------------------------- case-aware query construction ---------------------------
def fact_label(f: dict[str, Any]) -> str:
    """Value with its marker/test name, e.g. {'name': 'EGFR', 'value': 'wild-type'} -> 'EGFR wild-type'.
    Models sometimes put the name only in the separate `name` field."""
    value = str(f.get("value") or "").strip()
    name = str(f.get("name") or "").strip()
    if f.get("field") == "performance_status" and re.fullmatch(r"[0-4](?:\s*-\s*[0-4])?", value):
        name = name or "ECOG"
    if name and name.lower() not in value.lower():
        return f"{name} {value}"
    return value


_NAIVE_RE = re.compile(r"treatment[- ]na[iï]ve|untreated|no prior|not (?:previously )?treated|none\b|newly diagnosed", re.I)


def is_treatment_naive(value: str) -> bool:
    """'Treatment-naive', 'No prior systemic therapy' etc. are NOT prior therapies."""
    return bool(_NAIVE_RE.search(value or ""))


def build_case_query(question: str, facts: list[dict[str, Any]]) -> tuple[str, ClinicalQueryIntent]:
    """Combine the clinician's question with CONFIRMED facts into a retrieval query
    and population constraints (intent). Prior therapies set the line of therapy."""
    facts = [{**f, "value": fact_label(f)} for f in facts]
    by = lambda fld: [f for f in facts if f.get("field") == fld]  # noqa: E731
    parts = [question.strip().rstrip("?")]
    ctx = []
    for f in by("cancer_type")[:1] + by("histology")[:1] + by("stage")[:1]:
        ctx.append(f["value"])
    # Negative / wild-type results constrain the population (via intent) but are
    # kept OUT of the search text: lexical matching on "EGFR" would otherwise pull
    # in EGFR-targeted trials for an EGFR wild-type patient.
    negative_rx = re.compile(r"negative|wild[- ]?type|\bwt\b|not detected|absent|no (?:mutation|alteration)", re.I)
    all_bios = [f["value"] for f in by("biomarker")]
    bios = [v for v in all_bios if not negative_rx.search(v)][:5]
    if bios:
        ctx.append("; ".join(bios))
    priors = [f["value"] for f in by("prior_therapy") if not is_treatment_naive(f["value"])]
    naive = any(is_treatment_naive(f["value"]) for f in by("prior_therapy"))
    if priors:
        ctx.append("previously treated with " + "; ".join(priors[:3]))
    elif naive:
        ctx.append("treatment-naive (first-line)")
    if by("line_of_therapy"):
        ctx.append(by("line_of_therapy")[0]["value"])
    ctx = list(dict.fromkeys(ctx))
    query = parts[0] + (" — patient context: " + ", ".join(ctx) if ctx else "") + "?"

    intent = ClinicalIntentExtractor().extract_intent(" ".join([question] + ctx + all_bios))
    if priors and not intent.line_of_therapy:
        intent.line_of_therapy = "Previously treated / Relapsed"
        intent.population_constraints["line_of_therapy"] = intent.line_of_therapy
    elif naive and not intent.line_of_therapy:
        intent.line_of_therapy = "First-line (1L)"
        intent.population_constraints["line_of_therapy"] = intent.line_of_therapy
    return query, intent
