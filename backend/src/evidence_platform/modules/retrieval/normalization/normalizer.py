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

    def normalize(self, query: str) -> NormalizedQuery:
        """Clean query, expand acronyms, and produce rewritten clinical query variants."""
        if not query:
            return NormalizedQuery("", "", "", {})

        # 1. Unicode normalization and basic cleaning
        cleaned = unicodedata.normalize("NFKC", query)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # 2. Case-insensitive acronym and term expansion
        words = re.findall(r"\b[\w+-]+\b", cleaned)
        expanded = {}

        normalized_text = cleaned
        for word in words:
            word_lower = word.lower()
            if word_lower in self.ONCOLOGY_DICTIONARY:
                expansion = self.ONCOLOGY_DICTIONARY[word_lower]
                expanded[word] = expansion
                # Replace exact word boundary match with expansion
                normalized_text = re.sub(
                    rf"\b{re.escape(word)}\b",
                    f"{word} ({expansion})",
                    normalized_text,
                    flags=re.IGNORECASE
                )

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

        # Check concept synonyms
        for concept, synonyms in self.SYNONYM_MAP.items():
            if concept in query_lower and synonyms:
                primary_syn = synonyms[0]
                if primary_syn.lower() not in query_lower:
                    expanded[concept] = ", ".join(synonyms)

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
