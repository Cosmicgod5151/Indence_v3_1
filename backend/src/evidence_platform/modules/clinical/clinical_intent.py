"""Clinical Intent Extraction Module (Step 1).

Extracts structured clinical entities from clinical queries:
- Disease
- Disease Subtype
- Biomarker Status
- Histology
- Stage
- Metastatic vs Localized
- Line of Therapy
- Prior Treatments
- Mutation Status
- Age Group
- Treatment Being Asked About
- Comparator
- Clinical Outcome Requested
- Population Constraints & Rejected Subgroups
"""

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ClinicalQueryIntent:
    disease: str | None = None
    disease_subtype: str | None = None
    biomarker_status: str | None = None
    secondary_biomarkers: list[str] = field(default_factory=list)
    histology: str | None = None
    stage: str | None = None
    metastatic_vs_localized: str | None = None
    line_of_therapy: str | None = None
    prior_treatments: list[str] = field(default_factory=list)
    mutation_status: str | None = None
    performance_status: str | None = None
    age_group: str | None = None
    treatment: list[str] = field(default_factory=list)
    comparator: str | None = None
    clinical_outcome: str | None = None
    population_constraints: dict[str, Any] = field(default_factory=dict)
    rejected_subgroups: list[str] = field(default_factory=list)
    explicitly_broad_scope: bool = False
    confidence: str = "low"

    def to_dict(self) -> dict[str, Any]:
        return {
            "disease": self.disease,
            "disease_subtype": self.disease_subtype,
            "biomarker_status": self.biomarker_status,
            "secondary_biomarkers": self.secondary_biomarkers,
            "histology": self.histology,
            "stage": self.stage,
            "metastatic_vs_localized": self.metastatic_vs_localized,
            "line_of_therapy": self.line_of_therapy,
            "prior_treatments": self.prior_treatments,
            "mutation_status": self.mutation_status,
            "performance_status": self.performance_status,
            "age_group": self.age_group,
            "treatment": self.treatment,
            "comparator": self.comparator,
            "clinical_outcome": self.clinical_outcome,
            "population_constraints": self.population_constraints,
            "rejected_subgroups": self.rejected_subgroups,
            "explicitly_broad_scope": self.explicitly_broad_scope,
            "confidence": self.confidence,
        }


class ClinicalIntentExtractor:
    """Extracts structured clinical entities and population constraints from raw clinical queries."""

    # Disease patterns
    DISEASE_PATTERNS = [
        (r"\bbreast\s+cancer\b|\bbreast\s+carcinoma\b|\bmbc\b|\btnbc\b|\binvasive\s+(?:ductal|lobular)\s+carcinoma\b", "Breast Cancer"),
        (r"\bnon-small[\s-]+cell\s+lung\s+(?:cancer|carcinoma)\b|\bnsclc\b|\blung\s+adenocarcinoma\b|\badenocarcinoma\s+of\s+the\s+lung\b|\blung\s+squamous\b|\bsquamous\s+cell\s+(?:lung\s+)?carcinoma\s+of\s+the\s+lung\b", "Non-Small Cell Lung Cancer"),
        (r"\bsmall\s+cell\s+lung\s+cancer\b|\bsclc\b", "Small Cell Lung Cancer"),
        (r"\blung\s+cancer\b|\blung\s+carcinoma\b", "Lung Cancer"),
        (r"\bcolorectal\s+(?:cancer|carcinoma|adenocarcinoma)\b|\bcrc\b|\bmcrc\b|\bcolon\s+cancer\b|\brectal\s+cancer\b|\b(?:sigmoid|ascending|descending|transverse)\s+colon\b|\bcolon\s+adenocarcinoma\b|\badenocarcinoma\s+of\s+the\s+(?:\w+\s+)?(?:colon|rectum)\b", "Colorectal Cancer"),
        (r"\bprostate\s+(?:cancer|carcinoma|adenocarcinoma)\b|\bmcspc\b|\bmcrpc\b", "Prostate Cancer"),
        (r"\bcervical\s+cancer\b|\bcervical\s+carcinoma\b|\bcervix\s+cancer\b", "Cervical Cancer"),
        (r"\bovarian\s+(?:cancer|carcinoma)\b|\bhigh-grade\s+serous\b", "Ovarian Cancer"),
        (r"\bmelanoma\b", "Melanoma"),
        (r"\bpancreatic\s+cancer\b|\bpancreatic\s+ductal\s+adenocarcinoma\b", "Pancreatic Cancer"),
        (r"\bhead\s+and\s+neck\b|\bhnscc\b", "Head and Neck Cancer"),
        (r"\bgastric\s+cancer\b|\bstomach\s+cancer\b", "Gastric Cancer"),
        (r"\brenal\s+cell\s+carcinoma\b|\brcc\b", "Renal Cell Carcinoma"),
        (r"\bbladder\s+cancer\b|\burothelial\s+carcinoma\b", "Urothelial Carcinoma"),
    ]

    # Biomarker & Subtype patterns
    # Note on "+" / "-" suffixes: "\b" cannot follow "+" (no word boundary between
    # "+" and a space), so receptor-status tokens use "(?!\w)" instead. A bare
    # "her2-" must not match "her2-positive" / "her2-low".
    BIOMARKER_PATTERNS = [
        (r"\bher2[- ]positive\b|\bher2\s*\+(?!\w)|\bher2\s*pos\b|\bher2[- ]amplified\b", "HER2-positive", ["HER2-low", "HER2-ultralow", "HER2-negative", "Triple-negative"]),
        (r"\bher2[- ]low\b", "HER2-low", ["HER2-positive", "HER2-negative", "Triple-negative"]),
        (r"\bher2[- ]ultralow\b", "HER2-ultralow", ["HER2-positive", "HER2-negative"]),
        (r"\bher2[- ]negative\b|\bher2\s*-(?![\w-])", "HER2-negative", ["HER2-positive"]),
        (r"\btriple[- ]negative\b|\btnbc\b", "Triple-negative", ["HER2-positive", "HR-positive"]),
        (r"\bhr[- ]positive\b|\bhr\s*\+(?!\w)|\ber\s*\+(?!\w)|\bpr\s*\+(?!\w)|\ber/pr\s*\+(?!\w)|\bhormone[- ]receptor[- ]positive\b", "HR-positive", ["Triple-negative", "HR-negative"]),
        (r"\begfr[- ]mutant\b|\begfr[- ]mutated\b|\begfr\s*mutation\b|\begfr\s*\+(?!\w)|\begfr\s+exon\s*(?:19|21)\b|\bl858r\b", "EGFR-mutant", ["EGFR wild-type"]),
        (r"\begfr\s*(?:wild[- ]?type|wt)\b", "EGFR wild-type", ["EGFR-mutant"]),
        (r"\bkras\s*(?:wild[- ]?type|wt)\b|\bras\s*(?:wild[- ]?type|wt)\b", "KRAS/RAS wild-type", ["KRAS mutant"]),
        (r"\bkras\s*(?:mutant|mut|g12c)\b", "KRAS-mutant", ["KRAS wild-type"]),
        (r"\bbraf\s*v600e?\b|\bbraf\s*mutant\b", "BRAF-mutant", ["BRAF wild-type"]),
        (r"\balk[- ]positive\b|\balk\s*rearrangement\b|\balk\s*\+(?!\w)|\balk[- ]rearranged\b", "ALK-positive", ["ALK wild-type"]),
        (r"\bpd[- ]l1\s*(?:positive\b|high\b|(?:\u2265|>=)\s*\d{1,3}\s*%)", "PD-L1 positive", ["PD-L1 negative"]),
        (r"\bmsi[- ]h\b|\bdmmr\b", "MSI-H/dMMR", ["MSS/pMMR"]),
        (r"\bmss\b|\bpmmr\b", "MSS/pMMR", ["MSI-H/dMMR"]),
    ]

    # Histology patterns
    HISTOLOGY_PATTERNS = [
        (r"\bductal\b", "Ductal"),
        (r"\blobular\b", "Lobular"),
        (r"\bsquamous\b", "Squamous"),
        (r"\badenocarcinoma\b", "Adenocarcinoma"),
        (r"\bsmall\s+cell\b", "Small cell"),
        (r"\blarge\s+cell\b", "Large cell"),
        (r"\bserous\b", "Serous"),
        (r"\bclear\s+cell\b", "Clear cell"),
        (r"\bpapillary\b", "Papillary"),
        (r"\bmucinous\b", "Mucinous"),
        (r"\bneuroendocrine\b", "Neuroendocrine"),
    ]

    # Mutation patterns
    MUTATION_PATTERNS = [
        (r"\bpik3ca\b", "PIK3CA"),
        (r"\besr1\b", "ESR1"),
        (r"\bntrk\b", "NTRK"),
        (r"\bros1\b", "ROS1"),
        (r"\bmet\s+exon\s+14\b|\bmet\s+ex14\b", "MET exon 14"),
        (r"\bret\b", "RET"),
        (r"\bher2\s+mutat(?:ion|ed)\b", "HER2 mutation"),
    ]

    # Performance status patterns
    PERFORMANCE_STATUS_PATTERNS = [
        (r"\becog\s*(?:[0-4](?:-[0-4])?)\b", "ECOG"),
        (r"\bkarnofsky\b|\bkps\b", "Karnofsky"),
    ]

    # Treatment patterns
    TREATMENT_PATTERNS = [
        (r"\btrastuzumab\s+deruxtecan\b|\bt-dxd\b|\benhertu\b", "Trastuzumab deruxtecan"),
        (r"\bosimertinib\b|\btagrisso\b", "Osimertinib"),
        (r"\bpembrolizumab\b|\bkeytruda\b", "Pembrolizumab"),
        (r"\bnivolumab\b|\bopdivo\b", "Nivolumab"),
        (r"\batenolizumab\b|\batezolizumab\b|\btecentriq\b", "Atezolizumab"),
        (r"\btrastuzumab\b|\bherceptin\b", "Trastuzumab"),
        (r"\bpertuzumab\b|\bperjeta\b", "Pertuzumab"),
        (r"\bbevacizumab\b|\bavastin\b", "Bevacizumab"),
        (r"\bcetuximab\b|\berbitux\b", "Cetuximab"),
        (r"\bchemotherapy\b|\bchemo\b", "Chemotherapy"),
        (r"\bfolfiri\b", "FOLFIRI"),
        (r"\bfolfox\b", "FOLFOX"),
        (r"\bmetformin\b", "Metformin"),
        (r"\blaparoscopy\b|\bsurgery\b|\bmastectomy\b|\blumpectomy\b", "Surgery"),
        (r"\bradiotherapy\b|\brt\b|\bchemoradiotherapy\b|\bccrt\b", "Radiotherapy/Chemoradiotherapy"),
        (r"\bsacituzumab\s+govitecan\b|\btrodelvy\b", "Sacituzumab govitecan"),
        (r"\btucatinib\b", "Tucatinib"),
        (r"\blapatinib\b", "Lapatinib"),
        (r"\bneratinib\b", "Neratinib"),
        (r"\bpalbociclib\b", "Palbociclib"),
        (r"\bribociclib\b", "Ribociclib"),
        (r"\babemaciclib\b", "Abemaciclib"),
        (r"\bolaparib\b", "Olaparib"),
        (r"\btalazoparib\b", "Talazoparib"),
        (r"\bcapecitabine\b", "Capecitabine"),
        (r"\bdocetaxel\b", "Docetaxel"),
        (r"\bpaclitaxel\b", "Paclitaxel"),
        (r"\bcarboplatin\b", "Carboplatin"),
        (r"\bcisplatin\b", "Cisplatin"),
        (r"\bdurvalumab\b", "Durvalumab"),
        (r"\bipilimumab\b", "Ipilimumab"),
        (r"\bsotorasib\b", "Sotorasib"),
        (r"\badagrasib\b", "Adagrasib"),
    ]

    # Line of therapy patterns
    LINE_PATTERNS = [
        (r"\bfirst[- ]line\b|\b1st[- ]line\b|\b1l\b|\bfront[- ]line\b", "First-line (1L)"),
        (r"\bsecond[- ]line\b|\b2nd[- ]line\b|\b2l\b", "Second-line (2L)"),
        (r"\bthird[- ]line\b|\b3rd[- ]line\b|\b3l\b", "Third-line (3L)"),
        (r"\bpreviously\s+treated\b|\bpost[- ]progression\b|\brefractory\b", "Previously treated / Relapsed"),
        (r"\bneoadjuvant\b", "Neoadjuvant"),
        (r"\badjuvant\b", "Adjuvant"),
    ]

    # Stage / Metastatic patterns
    STAGE_PATTERNS = [
        (r"\bmetastatic\b|\badvanced\b|\bstage\s*iv\b|\bstage\s*4\b", "Metastatic", "Metastatic / Stage IV"),
        (r"\blocalized\b|\bearly[- ]stage\b|\bstage\s*i\b|\bstage\s*ii\b|\bstage\s*iii\b", "Localized", "Early / Localized"),
        (r"\bstage\s*ib2\b|\bfigo\s*stage\s*ib2\b", "Localized", "FIGO Stage IB2"),
    ]

    # Clinical outcome patterns
    OUTCOME_PATTERNS = [
        (r"\bprogression[- ]free\s+survival\b|\bpfs\b", "Progression-Free Survival (PFS)"),
        (r"\boverall\s+survival\b|\bos\b", "Overall Survival (OS)"),
        (r"\boverall\s+response\s+rate\b|\borr\b|\bobjective\s+response\s+rate\b", "Overall Response Rate (ORR)"),
        (r"\beffective\b|\befficacy\b|\bbenefit\b|\bsafety\b|\btoxicity\b", "Efficacy and Clinical Benefit"),
    ]

    def extract_intent(self, query: str) -> ClinicalQueryIntent:
        """Extract structured clinical entities from user query."""
        if not query:
            return ClinicalQueryIntent()

        query_lower = query.lower()

        # Extract Disease
        disease = None
        for pattern, label in self.DISEASE_PATTERNS:
            if re.search(pattern, query_lower):
                disease = label
                break

        # Explicitly broad scope
        explicitly_broad_scope = False
        if re.search(r"\b(any her2 status|regardless of biomarker|all subtypes|any subtype|irrespective of)\b", query_lower):
            explicitly_broad_scope = True

        # Extract Biomarker & Rejected Subgroups
        biomarkers = []
        rejected = []
        for pattern, label, rej_list in self.BIOMARKER_PATTERNS:
            if re.search(pattern, query_lower):
                if label not in biomarkers:
                    biomarkers.append(label)
                for r in rej_list:
                    if r not in rejected:
                        rejected.append(r)

        # A subgroup the user explicitly asked about can never be "rejected"
        # (e.g. "HR+/HER2- breast cancer" must not reject HR-positive).
        rejected = [r for r in rejected if r not in biomarkers]

        biomarker = biomarkers[0] if biomarkers else None
        secondary_biomarkers = biomarkers[1:] if len(biomarkers) > 1 else []

        # Extract Subtype
        subtype = biomarker  # Often biomarker defines subtype in oncology

        # Extract Histology
        histology = None
        for pattern, label in self.HISTOLOGY_PATTERNS:
            if re.search(pattern, query_lower):
                histology = label
                break

        # Extract Mutation
        mutation = biomarker
        for pattern, label in self.MUTATION_PATTERNS:
            if re.search(pattern, query_lower):
                mutation = label
                break

        # Extract Performance Status
        performance_status = None
        for pattern, label in self.PERFORMANCE_STATUS_PATTERNS:
            match = re.search(pattern, query_lower)
            if match:
                performance_status = match.group(0).upper() if "ecog" in pattern else label
                break

        # Extract Treatment
        treatments = []
        for pattern, label in self.TREATMENT_PATTERNS:
            if re.search(pattern, query_lower) and label not in treatments:
                treatments.append(label)

        # Extract Prior Treatments
        prior_treatments = []
        for pattern, label in self.TREATMENT_PATTERNS:
            prior_pattern = rf"(?:previously treated with|after|post[- ]|resistant to|refractory to)\s+(?:[a-z0-9\s]+)?(?:{pattern})|(?:{pattern})\s+failure"
            if re.search(prior_pattern, query_lower) and label not in prior_treatments:
                prior_treatments.append(label)
                # Remove from current treatments if accidentally matched
                if label in treatments:
                    treatments.remove(label)

        # Extract Comparator
        comparator = None
        comp_match = re.search(r"\b(?:versus|vs\.?|compared\s+to|compared\s+with)\s+([a-z0-9\s'-]+)", query_lower)
        if comp_match:
            comparator = comp_match.group(1).strip()

        # Extract Line of Therapy
        line = None
        for pattern, label in self.LINE_PATTERNS:
            if re.search(pattern, query_lower):
                line = label
                break

        # Extract Stage & Metastatic status
        metastatic_loc = None
        stage = None
        for pattern, met_label, stage_label in self.STAGE_PATTERNS:
            if re.search(pattern, query_lower):
                metastatic_loc = met_label
                stage = stage_label
                break

        # Extract Clinical Outcome
        outcome = None
        for pattern, label in self.OUTCOME_PATTERNS:
            if re.search(pattern, query_lower):
                outcome = label
                break

        # Calculate Confidence
        components_found = sum([
            1 if disease else 0,
            1 if biomarker else 0,
            1 if treatments else 0
        ])
        if components_found == 3:
            confidence = "high"
        elif components_found == 2:
            confidence = "medium"
        else:
            confidence = "low"

        # Construct population constraints
        population_constraints = {}
        if disease:
            population_constraints["disease"] = disease
        if biomarker:
            population_constraints["biomarker"] = biomarker
        if stage:
            population_constraints["stage"] = stage
        if line:
            population_constraints["line_of_therapy"] = line
        if rejected:
            population_constraints["rejected_subgroups"] = list(set(rejected))

        return ClinicalQueryIntent(
            disease=disease,
            disease_subtype=subtype,
            biomarker_status=biomarker,
            secondary_biomarkers=secondary_biomarkers,
            histology=histology,
            stage=stage,
            metastatic_vs_localized=metastatic_loc,
            line_of_therapy=line,
            prior_treatments=prior_treatments,
            mutation_status=mutation,
            performance_status=performance_status,
            age_group="Adult",
            treatment=treatments,
            comparator=comparator,
            clinical_outcome=outcome,
            population_constraints=population_constraints,
            rejected_subgroups=list(set(rejected)),
            explicitly_broad_scope=explicitly_broad_scope,
            confidence=confidence,
        )
