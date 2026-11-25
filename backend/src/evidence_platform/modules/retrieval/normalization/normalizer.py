import re
import unicodedata
from typing import Any, NamedTuple

class NormalizedQuery(NamedTuple):
    raw_query: str
    normalized_query: str
    rewritten_query: str
    expanded_terms: dict[str, str]


class QueryNormalizer:
    """Cleans and expands acronyms/synonyms in clinical queries for production retrieval."""

    # Comprehensive oncology, cardiology, and clinical staging dictionary
    ONCOLOGY_DICTIONARY: dict[str, str] = {
        "t-dxd": "trastuzumab deruxtecan",
        "dxd": "deruxtecan",
        "sacituzumab govitecan": "sacituzumab govitecan",
        "tucatinib": "tucatinib",
        "lapatinib": "lapatinib",
        "palbociclib": "palbociclib",
        "ribociclib": "ribociclib",
        "abemaciclib": "abemaciclib",
        "olaparib": "olaparib",
        "sotorasib": "sotorasib",
        "adagrasib": "adagrasib",
        "capecitabine": "capecitabine",
        "docetaxel": "docetaxel",
        "paclitaxel": "paclitaxel",
        "carboplatin": "carboplatin",
        "cisplatin": "cisplatin",
        "durvalumab": "durvalumab",
        "ipilimumab": "ipilimumab",
        "mbc": "metastatic breast cancer",
        "nsclc": "non-small cell lung cancer",
        "sclc": "small cell lung cancer",
        "pfs": "progression-free survival",
        "os": "overall survival",
        "rct": "randomized controlled trial",
        "egfr": "epidermal growth factor receptor",
        "her2": "human epidermal growth factor receptor 2",
        "her2-low": "human epidermal growth factor receptor 2",
        "her2+": "human epidermal growth factor receptor 2 positive",
        "hr+": "hormone receptor positive",
        "hr-": "hormone receptor negative",
        "triple-negative": "triple negative",
        "tnbc": "triple-negative breast cancer",
        "chemo": "chemotherapy",
        "nccn": "national comprehensive cancer network",
        "asco": "american society of clinical oncology",
        "esmo": "european society for medical oncology",
        "figo": "international federation of gynecology and obstetrics",
        "ajcc": "american joint committee on cancer",
        "tnm": "tumor node metastasis staging",
        "recist": "response evaluation criteria in solid tumors",
        "mi": "myocardial infarction",
        "hcc": "hepatocellular carcinoma",
        "ccrt": "concurrent chemoradiotherapy",
        "nac": "neoadjuvant chemotherapy",
        "rt": "radiotherapy",
        "brachy": "brachytherapy",
        "or": "odds ratio",
        "hr": "hazard ratio",
        "rr": "relative risk",
        "ci": "confidence interval",
    }

    # Clinical concept expansion map (medically equivalent synonyms)
    SYNONYM_MAP: dict[str, list[str]] = {
        "cervical cancer": ["cervical carcinoma", "cancer of the uterine cervix", "cervix cancer"],
        "cervical carcinoma": ["cervical cancer", "cancer of the uterine cervix"],
        "stage ib2": ["FIGO stage IB2", "stage IB2 cervical cancer", "IB2"],
        "breast-conserving": ["lumpectomy", "breast-conserving surgery", "breast-conserving therapy", "BCT"],
        "mastectomy": ["total mastectomy", "radical mastectomy"],
        "myocardial infarction": ["heart attack", "MI"],
        "non-small cell lung cancer": ["NSCLC", "non-small-cell lung carcinoma"],
        "trastuzumab deruxtecan": ["T-DXd", "Enhertu"],
        "db03": ["DESTINY-Breast03", "DESTINY-Breast 03"],
        "db-03": ["DESTINY-Breast03", "DESTINY-Breast 03"],
        "kn177": ["KEYNOTE-177"],
        "kn-177": ["KEYNOTE-177"],
    }

    # Short acronyms that collide with ordinary English words or with receptor
    # status tokens (e.g. "or", "os", "HR" in "HR+"). These are expanded only
    # when the user typed them in UPPERCASE as a standalone token.
    AMBIGUOUS_ACRONYMS: frozenset[str] = frozenset(
        {"or", "os", "hr", "rr", "ci", "mi", "rt", "nac", "dxd"}
    )

    # A clinical token: letters/digits/hyphens, optionally ending in "+" (HER2+, HR+).
    _TOKEN_RE = re.compile(r"(?<![\w+-])[A-Za-z0-9][A-Za-z0-9-]*\+?(?![\w])")

    def _expansion_for(self, token: str) -> str | None:
        key = token.lower()
        if key not in self.ONCOLOGY_DICTIONARY:
            return None
        if key in self.AMBIGUOUS_ACRONYMS and not token.isupper():
            return None
        return self.ONCOLOGY_DICTIONARY[key]

    def normalize(self, query: str) -> NormalizedQuery:
        """Clean query, expand acronyms, and produce rewritten clinical query variants."""
        if not query:
            return NormalizedQuery("", "", "", {})

        # 1. Unicode normalization and basic cleaning
        cleaned = unicodedata.normalize("NFKC", query)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # 2. Single left-to-right pass: each token is expanded at most once, so
        #    repeated words are never double-expanded and "HR+" is kept intact.
        expanded: dict[str, str] = {}
        parts: list[str] = []
        last = 0
        for m in self._TOKEN_RE.finditer(cleaned):
            token = m.group(0)
            expansion = self._expansion_for(token)
            if expansion is None or expansion.lower() == token.lower():
                continue
            parts.append(cleaned[last:m.end()])
            parts.append(f" ({expansion})")
            last = m.end()
            expanded.setdefault(token, expansion)
        parts.append(cleaned[last:])
        normalized_text = "".join(parts)

        # 3. Medical Concept & Clinical Staging Rewriting
        rewritten_text = normalized_text
        query_lower = cleaned.lower()

        # Specific staging expansion for queries like "stage ib2 cervical cancer"
        if "ib2" in query_lower and "figo" not in query_lower:
            rewritten_text = re.sub(
                r"\b(stage\s+ib2|ib2)\b",
                "FIGO stage IB2 (stage IB2)",
                rewritten_text,
                flags=re.IGNORECASE
            )

        # Concept synonyms are appended to the search text so the sparse
        # (lexical) channel can match papers that use a different term.
        added_synonyms: list[str] = []
        for concept, synonyms in self.SYNONYM_MAP.items():
            if re.search(rf"\b{re.escape(concept)}\b", query_lower) and synonyms:
                new_syns = [s for s in synonyms if s.lower() not in query_lower]
                if new_syns:
                    expanded[concept] = ", ".join(synonyms)
                    added_synonyms.extend(new_syns[:2])
        if added_synonyms:
            rewritten_text = f"{rewritten_text} ({'; '.join(dict.fromkeys(added_synonyms))})"

        return NormalizedQuery(
            raw_query=query,
            normalized_query=normalized_text,
            rewritten_query=rewritten_text,
            expanded_terms=expanded
        )

    def generate_population_query(self, normalized_query: str, intent: str = "") -> str:
        """Produce a population-focused query variant for retrieval."""
        # A simple heuristic example: "Is T-DXd effective in HER2+ breast cancer?" -> 
        # "trastuzumab deruxtecan HER2-positive breast cancer efficacy randomized controlled trial"
        cleaned = re.sub(r"^(is|are|does|do|can|could|would|will)\b", "", normalized_query, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b(effective|efficacious|work|good)\b", "efficacy", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b(in|for|with)\b", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\?", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return f"{cleaned} randomized controlled trial"
