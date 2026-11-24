"""Patient Safety Check: interactions, treatment history and organ-function cautions.

Inputs: the clinician-CONFIRMED case facts (current medications, prior
therapies, comorbidities, labs, performance status, age) and the drugs that the
question or the generated answer proposes.

Three layers, each labelled with how it was produced:
  1. curated_rule      deterministic rules for well-established interactions and
                       organ-function cautions summarised from product labelling
                       (auditable, no model involved)
  2. treatment_history a proposed drug (or its class) was already received
  3. ai_screen         remaining (proposed drug x current medication) pairs are
                       screened by Gemini Decision 1, which returns a calibrated
                       probability of a clinically significant interaction

This is a SCREENING AID: every flag must be verified against the current product
label / a pharmacist. Absence of a flag does not mean absence of risk.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, asdict
from typing import Any

import httpx

# --------------------------------------------------------------------------------
# Drug classes (lower-case generic names)
# --------------------------------------------------------------------------------
CLASSES: dict[str, set[str]] = {
    "egfr_tki_ph": {"gefitinib", "erlotinib"},                         # pH-dependent absorption
    "egfr_tki": {"gefitinib", "erlotinib", "afatinib", "dacomitinib", "osimertinib", "lazertinib"},
    "alk_tki": {"alectinib", "crizotinib", "lorlatinib", "brigatinib", "ceritinib"},
    "cdk46": {"palbociclib", "ribociclib", "abemaciclib"},
    "parp": {"olaparib", "niraparib", "rucaparib", "talazoparib"},
    "fluoropyrimidine": {"capecitabine", "fluorouracil", "5-fu", "tegafur"},
    "ici": {"pembrolizumab", "nivolumab", "atezolizumab", "durvalumab", "ipilimumab", "cemiplimab",
            "tislelizumab", "toripalimab", "sintilimab", "camrelizumab", "dostarlimab", "tremelimumab"},
    "her2_directed": {"trastuzumab", "pertuzumab", "trastuzumab deruxtecan", "t-dxd", "trastuzumab emtansine", "t-dm1"},
    "anthracycline": {"doxorubicin", "epirubicin", "daunorubicin", "idarubicin"},
    "platinum_cis": {"cisplatin"},
    "vegf": {"bevacizumab", "ramucirumab", "aflibercept"},
    "vegf_tki": {"sunitinib", "sorafenib", "pazopanib", "axitinib", "lenvatinib", "cabozantinib", "regorafenib"},
    "ar_inducer": {"enzalutamide", "apalutamide"},
    "qt_oncology": {"ribociclib", "osimertinib", "vandetanib", "arsenic trioxide", "nilotinib", "lapatinib"},
    # exposure falls with strong CYP3A4 INDUCERS (labels advise avoiding inducers)
    "cyp3a_inducer_sensitive": {"olaparib", "palbociclib", "ribociclib", "abemaciclib", "osimertinib",
                                "alectinib", "lorlatinib", "brigatinib", "crizotinib", "lapatinib", "neratinib",
                                "tucatinib", "irinotecan", "docetaxel", "cabazitaxel", "abiraterone",
                                "capivasertib", "elacestrant"},
    # exposure rises with strong CYP3A4 INHIBITORS (labels advise avoid or dose-reduce)
    "cyp3a_inhibitor_sensitive": {"olaparib", "palbociclib", "ribociclib", "abemaciclib", "crizotinib", "brigatinib",
                                  "lapatinib", "neratinib", "irinotecan", "docetaxel", "cabazitaxel",
                                  "capivasertib", "elacestrant"},
    "tamoxifen": {"tamoxifen"},
    "pi3k": {"alpelisib", "capivasertib", "inavolisib"},
    "pemetrexed": {"pemetrexed"},
    "taxane_irinotecan_hepatic": {"docetaxel", "paclitaxel", "cabazitaxel", "irinotecan"},
    # patient-medication classes
    "ppi_h2": {"omeprazole", "esomeprazole", "pantoprazole", "lansoprazole", "rabeprazole", "famotidine", "ranitidine"},
    "warfarin": {"warfarin", "acenocoumarol"},
    "doac": {"apixaban", "rivaroxaban", "dabigatran", "edoxaban"},
    "strong_cyp3a_inducer": {"rifampicin", "rifampin", "carbamazepine", "phenytoin", "phenobarbital", "st john's wort",
                             "rifabutin"},
    "strong_cyp3a_inhibitor": {"ketoconazole", "itraconazole", "posaconazole", "voriconazole", "clarithromycin",
                               "ritonavir", "cobicistat", "grapefruit"},
    "qt_drug": {"ondansetron", "citalopram", "escitalopram", "haloperidol", "methadone", "amiodarone", "sotalol",
                "azithromycin", "clarithromycin", "levofloxacin", "moxifloxacin", "hydroxychloroquine", "domperidone"},
    "strong_cyp2d6_inhibitor": {"paroxetine", "fluoxetine", "bupropion", "quinidine"},
    "nsaid": {"ibuprofen", "naproxen", "diclofenac", "ketorolac", "indomethacin"},
    "corticosteroid": {"prednisone", "prednisolone", "dexamethasone", "methylprednisolone", "hydrocortisone"},
}


def _classes_of(drug: str) -> set[str]:
    d = drug.lower()
    return {c for c, members in CLASSES.items() if any(m == d or m in d for m in members)}


@dataclass
class Rule:
    proposed_class: str
    patient_class: str
    severity: str        # major | moderate
    effect: str
    action: str


DDI_RULES: list[Rule] = [
    Rule("egfr_tki_ph", "ppi_h2", "major", "Acid suppression markedly lowers gefitinib/erlotinib absorption (pH-dependent solubility).",
         "Avoid PPIs; if acid suppression is essential, use a stagger strategy per label."),
    Rule("fluoropyrimidine", "warfarin", "major", "Capecitabine/5-FU increase INR and bleeding risk with coumarin anticoagulants.",
         "Frequent INR monitoring and warfarin dose adjustment, or consider an alternative anticoagulant."),
    Rule("cyp3a_inducer_sensitive", "strong_cyp3a_inducer", "major", "Strong CYP3A4 inducers substantially reduce exposure to the anticancer drug (loss of efficacy).",
         "Avoid the combination; choose a non-inducing alternative."),
    Rule("cyp3a_inhibitor_sensitive", "strong_cyp3a_inhibitor", "major", "Strong CYP3A4 inhibitors raise exposure to the anticancer drug (toxicity).",
         "Avoid, or reduce the anticancer dose as the label specifies."),
    Rule("qt_oncology", "qt_drug", "major", "Additive QTc prolongation (torsades de pointes risk).",
         "Avoid if possible; otherwise baseline/on-treatment ECG and electrolyte monitoring."),
    Rule("tamoxifen", "strong_cyp2d6_inhibitor", "moderate", "Strong CYP2D6 inhibition reduces conversion of tamoxifen to endoxifen.",
         "Prefer an antidepressant with weak CYP2D6 inhibition (e.g. venlafaxine)."),
    Rule("ar_inducer", "doac", "major", "Enzalutamide/apalutamide (CYP3A4/P-gp induction) lower DOAC concentrations.",
         "Review anticoagulation strategy with the prescriber."),
    Rule("ar_inducer", "warfarin", "major", "Enzalutamide/apalutamide induce CYP2C9/3A4 and lower warfarin effect.",
         "Close INR monitoring."),
    Rule("ici", "corticosteroid", "moderate", "Baseline systemic corticosteroids (≥10 mg/day prednisone-equivalent) have been associated with reduced benefit from immune checkpoint inhibitors.",
         "Review indication and dose of steroids before starting immunotherapy."),
    Rule("pemetrexed", "nsaid", "moderate", "NSAIDs reduce pemetrexed clearance, especially with reduced renal function.",
         "Avoid NSAIDs around pemetrexed doses when CrCl is reduced."),
]

COMORBIDITY_RULES: list[tuple[str, str, str, str, str]] = [
    # (proposed class, comorbidity regex, severity, effect, action)
    ("ici", r"autoimmun|rheumatoid|lupus|crohn|ulcerative colitis|psoriasis|multiple sclerosis|myasthenia|hashimoto",
     "major", "Pre-existing autoimmune disease may flare under immune checkpoint inhibition; such patients were usually excluded from trials.",
     "Discuss risk/benefit; rheumatology/specialist co-management."),
    ("ici", r"hypothyroid|levothyroxine|thyroiditis", "moderate",
     "Immune checkpoint inhibitors commonly cause thyroid dysfunction; existing hypothyroidism needs closer monitoring.",
     "Baseline and periodic thyroid function tests."),
    ("ici", r"transplant", "major", "Immune checkpoint inhibitors can trigger organ-transplant rejection.", "Usually avoided; specialist decision."),
    ("her2_directed", r"heart failure|cardiomyopathy|reduced ejection", "major",
     "HER2-directed therapy can cause left-ventricular dysfunction.", "Cardio-oncology review; LVEF monitoring."),
    ("her2_directed", r"interstitial lung|pneumonitis|ild\b|pulmonary fibrosis", "major",
     "Trastuzumab deruxtecan carries a boxed warning for ILD/pneumonitis; prior ILD was an exclusion criterion.",
     "Avoid T-DXd in active/prior significant ILD; pulmonary assessment."),
    ("vegf", r"hypertension", "moderate", "Anti-VEGF therapy commonly worsens hypertension.", "Control BP before and during treatment."),
    ("vegf_tki", r"hypertension", "moderate", "VEGF-TKIs commonly worsen hypertension.", "Control BP before and during treatment."),
    ("vegf", r"haemoptysis|hemoptysis|bleeding|recent surgery", "major",
     "Anti-VEGF therapy increases bleeding and wound-healing complications.", "Avoid with recent haemoptysis/surgery."),
    ("pi3k", r"diabet", "major", "PI3K/AKT inhibitors (e.g. alpelisib) frequently cause hyperglycaemia; diabetic patients need tight control.",
     "Optimise glucose control; endocrinology input; monitor glucose."),
]


@dataclass
class SafetyFlag:
    category: str          # treatment_history | drug_interaction | organ_function | comorbidity | ai_screen
    severity: str          # major | moderate | info
    proposed: str
    patient_factor: str
    message: str
    action: str
    method: str            # curated_rule | treatment_history | ai_screen
    source_quote: str | None = None
    probability: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DRUG_WORD = re.compile(r"\b[a-z][a-z-]{3,}(?:tinib|inib|zumab|ximab|mumab|umab|ciclib|parib|lisib|tecan|platin|"
                       r"rafenib|metinib|lutamide|terone|rasib|tansine|taxel|rubicin|citabine|fosfamide|trexate|"
                       r"limus|sertib|pemetrexed|fluorouracil|tamoxifen|letrozole|anastrozole|exemestane|strant|degarelix|leuprolide)\b", re.I)


def proposed_drugs(*texts: str) -> list[str]:
    from .clinical_intent import ClinicalIntentExtractor
    found: list[str] = []
    ex = ClinicalIntentExtractor()
    for t in texts:
        if not t:
            continue
        for d in ex.extract_intent(t).treatment:
            if d.lower() not in ("chemotherapy", "surgery", "radiotherapy/chemoradiotherapy") and d.lower() not in found:
                found.append(d.lower())
        for m in DRUG_WORD.finditer(t):
            w = m.group(0).lower()
            if w not in found:
                found.append(w)
    return found


def _lab_value(facts: list[dict], pattern: str) -> tuple[float | None, str | None]:
    for f in facts:
        if f.get("field") != "lab":
            continue
        label = f"{f.get('name') or ''} {f.get('value') or ''}"
        if re.search(pattern, label, re.I):
            m = re.search(r"(\d+(?:\.\d+)?)", f.get("value") or "")
            if m:
                return float(m.group(1)), f.get("source_quote") or label
    return None, None


def _rule_checks(proposed: list[str], facts: list[dict]) -> list[SafetyFlag]:
    from .case_context import fact_label, is_treatment_naive
    flags: list[SafetyFlag] = []
    meds = [(fact_label(f), f.get("source_quote")) for f in facts if f.get("field") == "medication"]
    comorb = " ".join(fact_label(f) for f in facts if f.get("field") in ("comorbidity", "medication"))
    priors = [(fact_label(f), f.get("source_quote")) for f in facts
              if f.get("field") == "prior_therapy" and not is_treatment_naive(f.get("value", ""))]

    for drug in proposed:
        dcls = _classes_of(drug)
        # 1. treatment history
        for prior, quote in priors:
            pl = prior.lower()
            same_agent = drug in pl
            same_class = bool(dcls & _classes_of(pl)) and not same_agent
            if same_agent or (same_class and dcls & {"cdk46", "parp", "egfr_tki", "alk_tki", "ici", "her2_directed"}):
                flags.append(SafetyFlag(
                    "treatment_history", "major" if same_agent else "moderate", drug, prior,
                    f"Patient has already received {'this drug' if same_agent else 'a drug of the same class'} ({prior}). "
                    "Trials of treatment-naive patients may not apply; resistance/progression on prior therapy must be considered.",
                    "Confirm response to and reason for stopping the prior therapy before re-using the drug/class.",
                    "treatment_history", quote))
        # 2. drug–drug interactions with current medications
        for med, quote in meds:
            mcls = _classes_of(med)
            for r in DDI_RULES:
                if r.proposed_class in dcls and r.patient_class in mcls:
                    flags.append(SafetyFlag("drug_interaction", r.severity, drug, med, r.effect, r.action, "curated_rule", quote))
        # 3. comorbidity cautions
        for pcls, rx, sev, effect, action in COMORBIDITY_RULES:
            if pcls in dcls and re.search(rx, comorb, re.I):
                flags.append(SafetyFlag("comorbidity", sev, drug, re.search(rx, comorb, re.I).group(0), effect, action, "curated_rule"))
        # 4. organ function
        crcl, q1 = _lab_value(facts, r"egfr\b|crcl|creatinine clearance|glomerular")
        creat, q2 = _lab_value(facts, r"creatinine(?! clearance)")
        if "platinum_cis" in dcls and ((crcl is not None and crcl < 60) or (creat is not None and creat > 1.5)):
            flags.append(SafetyFlag("organ_function", "major", drug, f"renal function ({q1 or q2})",
                                    "Cisplatin is nephrotoxic and generally avoided when creatinine clearance is below ~60 mL/min.",
                                    "Consider carboplatin; hydrate and monitor renal function.", "curated_rule", q1 or q2))
        if "pemetrexed" in dcls and crcl is not None and crcl < 45:
            flags.append(SafetyFlag("organ_function", "major", drug, f"eGFR/CrCl {crcl}",
                                    "Pemetrexed is not recommended when creatinine clearance is below 45 mL/min.",
                                    "Choose an alternative regimen.", "curated_rule", q1))
        lvef, q3 = _lab_value(facts, r"lvef|ejection fraction")
        if dcls & {"her2_directed", "anthracycline"} and lvef is not None and lvef < 50:
            flags.append(SafetyFlag("organ_function", "major", drug, f"LVEF {lvef}%",
                                    "HER2-directed therapy/anthracyclines can reduce LVEF; baseline LVEF below 50% was an exclusion in pivotal trials.",
                                    "Cardio-oncology assessment before starting.", "curated_rule", q3))
        bili, q4 = _lab_value(facts, r"bilirubin")
        if "taxane_irinotecan_hepatic" in dcls and bili is not None and bili > 1.8:
            flags.append(SafetyFlag("organ_function", "moderate", drug, f"bilirubin {bili} mg/dL",
                                    "Taxanes/irinotecan require dose modification or avoidance with raised bilirubin.",
                                    "Check hepatic dosing guidance in the label.", "curated_rule", q4))
    return flags


AI_SCREEN_THRESHOLD = 0.85   # calibrated probability needed to show an AI-screened pair


async def _ai_screen(proposed: list[str], facts: list[dict], covered: set[tuple[str, str]]) -> tuple[list[SafetyFlag], dict]:
    """Screen uncovered (proposed drug x current medication) pairs with Gemini Decision 1."""
    from .case_context import fact_label
    key = os.environ.get("GEMINI_API_KEY", "")
    meds = [(fact_label(f), f.get("source_quote")) for f in facts if f.get("field") == "medication"]
    pairs = [(d, m, q) for d in proposed for m, q in meds if (d, m) not in covered]
    meta = {"model": "gemini-decision-1", "task": "AI screening of drug-medication pairs not covered by curated rules",
            "pairs_screened": len(pairs)}
    if not key or not pairs:
        meta["skipped"] = "no API key" if not key else "no uncovered pairs"
        return [], meta
    pairs = pairs[:40]
    questions = {f"pair_{i}": {
        "type": "noul",
        "instructions": f"Is there a clinically significant drug-drug interaction between the anticancer drug '{d}' and "
                        f"the patient's current medication '{m}' that would require avoiding the combination, a dose change "
                        "or extra monitoring according to standard drug-interaction references?",
        "criteria": {"true_description": "A clinically significant interaction is documented.",
                     "false_description": "No clinically significant interaction is documented."},
    } for i, (d, m, _) in enumerate(pairs)}
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post("https://generativelanguage.googleapis.com/v1beta/decisions",
                                  headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                                  json={"model": "gemini-decision-1",
                                        "input": "Oncology drug-interaction screening for a clinician. Answer each question independently.",
                                        "questions": questions})
        r.raise_for_status()
        answers = r.json().get("answers", {})
    except Exception as e:
        meta["error"] = str(e)[:200]
        return [], meta
    meta["latency_s"] = round(time.perf_counter() - t0, 3)
    flags = []
    for i, (d, m, q) in enumerate(pairs):
        p = float((answers.get(f"pair_{i}") or {}).get("noul", 0.0))
        if p >= AI_SCREEN_THRESHOLD:
            flags.append(SafetyFlag("ai_screen", "review", d, m,
                                    f"AI screen: possible clinically significant interaction (probability {p:.2f}).",
                                    "Not from a curated rule — verify with a drug-interaction reference or pharmacist.",
                                    "ai_screen", q, round(p, 3)))
    return flags, meta


async def run_safety_check(facts: list[dict], question: str, answer_texts: list[str]) -> dict[str, Any]:
    proposed = proposed_drugs(question, *answer_texts)
    flags = _rule_checks(proposed, facts)
    covered = {(f.proposed, f.patient_factor) for f in flags if f.category == "drug_interaction"}
    ai_flags, ai_meta = await _ai_screen(proposed, facts, covered)
    flags += ai_flags
    order = {"major": 0, "moderate": 1, "review": 2, "info": 3}
    flags.sort(key=lambda f: (order.get(f.severity, 3), f.category))
    from .case_context import fact_label
    return {
        "proposed_drugs": proposed,
        "patient_medications": [fact_label(f) for f in facts if f.get("field") == "medication"],
        "prior_therapies": [fact_label(f) for f in facts if f.get("field") == "prior_therapy"],
        "flags": [f.to_dict() for f in flags],
        "counts": {"major": sum(f.severity == "major" for f in flags), "moderate": sum(f.severity == "moderate" for f in flags),
                   "ai_review": sum(f.severity == "review" for f in flags)},
        "model_trace": {"safety_rules": {"model": "Indence curated rules (no LLM)",
                                         "task": f"{len(DDI_RULES)} interaction + {len(COMORBIDITY_RULES)} comorbidity rules, "
                                                 "organ-function thresholds, treatment-history matching"},
                        "safety_ai_screen": ai_meta},
        "disclaimer": "Screening aid only. Verify every flag against the current product label or with a pharmacist; "
                      "the absence of a flag does not mean the absence of risk.",
    }
