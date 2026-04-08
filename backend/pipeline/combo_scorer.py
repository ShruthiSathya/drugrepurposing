"""
combo_scorer.py — Drug Combination Scorer v6.1
===============================================

CHANGES IN v6.1
---------------
1. Add microtubule_inhibitor to ONCOLOGY_CLASSES
   Prevents chemotherapy drugs (vinflunine, ixabepilone, epothilone) from 
   appearing in non-cancer combos. Colchicine remains safe as it is 
   classified separately.

2. Updated MECHANISM_KEYWORD_MAP
   Added explicit mappings for ixabepilone, epothilone, larotaxel, 
   vinflunine, vinorelbine, and vindesine to ensure they inherit 
   oncology context penalties.

FIXES FROM v5.0 (Archived)
--------------------------
... (Previous version history truncated for brevity)
"""

import itertools
import logging
import re
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Salt/form suffix stripping
# ─────────────────────────────────────────────────────────────────────────────

_SALT_RE = re.compile(
    r"\s+(hydrochloride|hcl|sodium|potassium|sulfate|tartrate|maleate|"
    r"mesylate|acetate|phosphate|fumarate|succinate|monohydrate|dihydrate|"
    r"anhydrous|bitartrate|besylate|tosylate|citrate|calcium|magnesium|"
    r"bromide|chloride|disodium|trisodium|sodium\s+phosphate|"
    r"extended.release|er|xr|sr|cr|decanoate|pamoate|microspheres)$",
    re.IGNORECASE,
)


def _normalise_name(name: str) -> str:
    """Lowercase and strip salt suffixes (two passes for double salts)."""
    n = name.lower().strip()
    n = _SALT_RE.sub("", n).strip()
    n = _SALT_RE.sub("", n).strip()
    return n


# ─────────────────────────────────────────────────────────────────────────────
# Mechanism classification
# ─────────────────────────────────────────────────────────────────────────────

MECHANISM_KEYWORD_MAP: List[Tuple[str, str]] = [
    # PAH — three pathways (must be before generic terms)
    ("pde5",                    "pde5_inhibitor"),
    ("phosphodiesterase-5",     "pde5_inhibitor"),
    ("phosphodiesterase 5",     "pde5_inhibitor"),
    ("sildenafil",              "pde5_inhibitor"),
    ("tadalafil",               "pde5_inhibitor"),
    ("vardenafil",              "pde5_inhibitor"),
    ("endothelin",              "endothelin_antagonist"),
    ("bosentan",                "endothelin_antagonist"),
    ("ambrisentan",             "endothelin_antagonist"),
    ("macitentan",              "endothelin_antagonist"),
    ("sitaxentan",              "endothelin_antagonist"),
    ("prostacyclin",            "prostacyclin"),
    ("iloprost",                "prostacyclin"),
    ("treprostinil",            "prostacyclin"),
    ("epoprostenol",            "prostacyclin"),
    ("selexipag",               "prostacyclin"),
    ("beraprost",               "prostacyclin"),
    ("ptgir",                   "prostacyclin"),
    ("ptgis",                   "prostacyclin"),
    ("ptger2",                  "prostacyclin"),
    ("prostanoid",              "prostacyclin"),
    ("soluble guanylate",       "sgc_stimulator"),
    ("riociguat",               "sgc_stimulator"),
    ("gucy",                    "sgc_stimulator"),
    # Cardiovascular
    ("beta.adrenergic blocker", "beta_blocker"),
    ("beta blocker",            "beta_blocker"),
    ("beta-blocker",            "beta_blocker"),
    ("propranolol",             "beta_blocker"),
    ("metoprolol",              "beta_blocker"),
    ("carvedilol",              "beta_blocker"),
    ("atenolol",                "beta_blocker"),
    ("bisoprolol",              "beta_blocker"),
    ("labetalol",               "beta_blocker"),
    ("nebivolol",               "beta_blocker"),
    ("adrb1",                   "beta_blocker"),
    ("adrb2",                   "beta_blocker"),
    ("ace inhibitor",           "ace_inhibitor"),
    ("angiotensin.converting",  "ace_inhibitor"),
    ("lisinopril",              "ace_inhibitor"),
    ("enalapril",               "ace_inhibitor"),
    ("ramipril",                "ace_inhibitor"),
    ("captopril",               "ace_inhibitor"),
    ("angiotensin receptor",    "arb"),
    ("losartan",                "arb"),
    ("valsartan",               "arb"),
    ("agtr1",                   "arb"),
    ("statin",                  "statin"),
    ("hmgcr",                   "statin"),
    ("hmg-coa",                 "statin"),
    ("atorvastatin",            "statin"),
    ("rosuvastatin",            "statin"),
    ("simvastatin",             "statin"),
    ("lovastatin",              "statin"),
    ("pravastatin",             "statin"),
    # Metabolic
    ("biguanide",               "biguanide"),
    ("metformin",               "biguanide"),
    ("ampk",                    "biguanide"),
    ("prkaa",                   "biguanide"),
    ("thiazolidinedione",       "thiazolidinedione"),
    ("ppar.gamma",              "thiazolidinedione"),
    ("pparg",                   "thiazolidinedione"),
    ("pioglitazone",            "thiazolidinedione"),
    ("rosiglitazone",           "thiazolidinedione"),
    ("sglt2",                   "sglt2_inhibitor"),
    ("slc5a2",                  "sglt2_inhibitor"),
    ("empagliflozin",           "sglt2_inhibitor"),
    ("dapagliflozin",           "sglt2_inhibitor"),
    ("glp-1",                   "glp1_agonist"),
    ("glucagon-like",           "glp1_agonist"),
    ("sulfonylurea",            "sulfonylurea"),
    ("glipizide",               "sulfonylurea"),
    ("glimepiride",             "sulfonylurea"),
    ("glyburide",               "sulfonylurea"),
    ("glibenclamide",           "sulfonylurea"),
    ("abcc8",                   "sulfonylurea"),
    ("kcnj11",                  "sulfonylurea"),
    # Immunology / Inflammation
    ("immunomodulat",           "immunomodulator"),
    ("thalidomide",             "imid"),
    ("lenalidomide",            "imid"),
    ("pomalidomide",            "imid"),
    ("cereblon",                "imid"),
    ("crbn",                    "imid"),
    ("ikzf",                    "imid"),
    ("corticosteroid",          "corticosteroid"),
    ("glucocorticoid",          "corticosteroid"),
    ("dexamethasone",           "corticosteroid"),
    ("prednisone",              "corticosteroid"),
    ("prednisolone",            "corticosteroid"),
    ("methylprednisolone",      "corticosteroid"),
    ("hydrocortisone",          "corticosteroid"),
    ("betamethasone",           "corticosteroid"),
    ("budesonide",              "corticosteroid"),
    ("nr3c1",                   "corticosteroid"),
    # DMARDs
    ("anti-tnf",                "anti_tnf"),
    ("tumor necrosis factor",   "anti_tnf"),
    ("infliximab",              "anti_tnf"),
    ("adalimumab",              "anti_tnf"),
    ("anti-il",                 "anti_il6"),
    ("interleukin",             "anti_il6"),
    ("tocilizumab",             "anti_il6"),
    ("jak inhibitor",           "jak_inhibitor"),
    ("jak-stat",                "jak_inhibitor"),
    ("janus kinase",            "jak_inhibitor"),
    ("dmard",                   "dmard"),
    ("methotrexate",            "dmard"),
    ("antifolate",              "dmard"),
    ("dhfr",                    "dmard"),
    ("hydroxychloroquine",      "antimalarial"),
    ("chloroquine",             "antimalarial"),
    ("tlr7",                    "antimalarial"),
    ("tlr9",                    "antimalarial"),
    ("sulfasalazine",           "sulfonamide"),
    ("5-aminosalicylic",        "sulfonamide"),
    ("leflunomide",             "dhodh_inhibitor"),
    ("dhodh",                   "dhodh_inhibitor"),
    ("anti-cd20",               "anti_cd20"),
    ("cd20",                    "anti_cd20"),
    ("rituximab",               "anti_cd20"),
    ("ms4a1",                   "anti_cd20"),
    # Aldosterone antagonists
    ("aldosterone antagonist",  "mineralocorticoid_antagonist"),
    ("mineralocorticoid",       "mineralocorticoid_antagonist"),
    ("spironolactone",          "mineralocorticoid_antagonist"),
    ("eplerenone",              "mineralocorticoid_antagonist"),
    ("finerenone",              "mineralocorticoid_antagonist"),
    ("nr3c2",                   "mineralocorticoid_antagonist"),
    # Oncology
    ("parp",                    "parp_inhibitor"),
    ("pd-1",                    "checkpoint_inhibitor"),
    ("pd-l1",                   "checkpoint_inhibitor"),
    ("ctla-4",                  "checkpoint_inhibitor"),
    ("checkpoint",              "checkpoint_inhibitor"),
    ("alkylat",                 "alkylating_agent"),
    ("cyclophosphamide",        "alkylating_agent"),
    ("melphalan",               "alkylating_agent"),
    ("cisplatin",               "alkylating_agent"),
    ("carboplatin",             "alkylating_agent"),
    ("oxaliplatin",             "alkylating_agent"),
    ("antimetabolite",          "antimetabolite"),
    ("gemcitabine",             "antimetabolite"),
    ("capecitabine",            "antimetabolite"),
    ("fluorouracil",            "antimetabolite"),
    ("taxane",                  "taxane"),
    ("paclitaxel",              "taxane"),
    ("docetaxel",               "taxane"),
    ("cabazitaxel",             "taxane"),
    ("ixabepilone",             "taxane"),      # FIX v6.1: chemo appearing in gout/pericarditis
    ("epothilone",              "taxane"),
    ("larotaxel",               "taxane"),
    ("vinca",                   "vinca_alkaloid"),
    ("vincristine",             "vinca_alkaloid"),
    ("vinblastine",             "vinca_alkaloid"),
    ("vinflunine",              "vinca_alkaloid"),   # FIX v6.1: was appearing in pericarditis
    ("vinorelbine",             "vinca_alkaloid"),
    ("vindesine",               "vinca_alkaloid"),
    ("hdac",                    "hdac_inhibitor"),
    ("histone deacetylase",     "hdac_inhibitor"),
    ("vorinostat",              "hdac_inhibitor"),
    ("proteasome",              "proteasome_inhibitor"),
    ("bortezomib",              "proteasome_inhibitor"),
    ("carfilzomib",             "proteasome_inhibitor"),
    ("psmb",                    "proteasome_inhibitor"),
    ("anti-vegf",               "anti_vegf"),
    ("vegf",                    "anti_vegf"),
    ("bevacizumab",             "anti_vegf"),
    ("aromatase",               "aromatase_inhibitor"),
    ("letrozole",               "aromatase_inhibitor"),
    ("anastrozole",             "aromatase_inhibitor"),
    ("exemestane",              "aromatase_inhibitor"),
    ("cyp19",                   "aromatase_inhibitor"),
    ("serm",                    "serm"),
    ("tamoxifen",               "serm"),
    ("raloxifene",              "serm"),
    ("esr1",                    "serm"),
    ("kinase inhibitor",        "kinase_inhibitor"),
    ("tyrosine kinase",         "kinase_inhibitor"),
    ("imatinib",                "kinase_inhibitor"),
    ("dasatinib",               "kinase_inhibitor"),
    ("mtor",                    "mtor_inhibitor"),
    ("sirolimus",               "mtor_inhibitor"),
    ("everolimus",              "mtor_inhibitor"),
    ("doxorubicin",             "anthracycline"),
    ("epirubicin",              "anthracycline"),
    ("anthracycline",           "anthracycline"),
    ("topoisomerase",           "topoisomerase_inhibitor"),
    # Neurology
    ("anticonvulsant",          "anticonvulsant"),
    ("antiepileptic",           "anticonvulsant"),
    ("sodium channel",          "anticonvulsant"),
    ("scn1a",                   "anticonvulsant"),
    ("dopamine agonist",        "dopamine_agonist"),
    ("dopaminergic",            "dopamine_agonist"),
    ("pramipexole",             "dopamine_agonist"),
    ("ropinirole",              "dopamine_agonist"),
    ("levodopa",                "dopamine_precursor"),
    ("carbidopa",               "dopamine_precursor"),
    ("mao-b",                   "maob_inhibitor"),
    ("monoamine oxidase",       "maob_inhibitor"),
    ("rasagiline",              "maob_inhibitor"),
    ("selegiline",              "maob_inhibitor"),
    ("maob",                    "maob_inhibitor"),
    ("nmda",                    "nmda_antagonist"),
    ("memantine",               "nmda_antagonist"),
    ("amantadine",              "nmda_antagonist"),
    ("grin1",                   "nmda_antagonist"),
    ("grin2",                   "nmda_antagonist"),
    ("acetylcholinesterase",    "acetylcholinesterase_inhibitor"),
    ("cholinesterase",          "acetylcholinesterase_inhibitor"),
    ("donepezil",               "acetylcholinesterase_inhibitor"),
    ("rivastigmine",            "acetylcholinesterase_inhibitor"),
    ("galantamine",             "acetylcholinesterase_inhibitor"),
    ("ache",                    "acetylcholinesterase_inhibitor"),
    ("bche",                    "acetylcholinesterase_inhibitor"),
    # Pain
    ("opioid antagonist",       "opioid_antagonist"),
    ("naltrexone",              "opioid_antagonist"),
    ("nsaid",                   "nsaid"),
    ("cox-2",                   "cox2_inhibitor"),
    ("cyclooxygenase-2",        "cox2_inhibitor"),
    ("celecoxib",               "cox2_inhibitor"),
    ("aspirin",                 "nsaid"),
    ("ibuprofen",               "nsaid"),
    ("ptgs",                    "nsaid"),
    ("ptgs1",                   "nsaid"),
    ("ptgs2",                   "nsaid"),
    # Rare / other
    ("colchicine",              "colchicine"),
    ("tubb",                    "colchicine"),
    ("microtubule",             "microtubule_inhibitor"),
    ("tubulin",                 "microtubule_inhibitor"),
    ("xanthine oxidase",        "anti_uric_acid"),
    ("allopurinol",             "anti_uric_acid"),
    ("febuxostat",              "anti_uric_acid"),
    ("xdh",                     "anti_uric_acid"),
    ("diuretic",                "diuretic"),
    ("furosemide",              "diuretic"),
    ("hydrochlorothiazide",     "diuretic"),
    ("potassium channel",       "potassium_channel"),
    ("minoxidil",               "potassium_channel"),
    ("cftr",                    "cftr_modulator"),
    ("ivacaftor",               "cftr_modulator"),
    ("complement",              "complement_inhibitor"),
    ("eculizumab",              "complement_inhibitor"),
    ("npc1l1",                  "cholesterol_absorption_inhibitor"),
    ("ezetimibe",               "cholesterol_absorption_inhibitor"),
    ("androgen receptor",       "anti_androgen"),
    ("anti-androgen",           "anti_androgen"),
    ("5-alpha reductase",       "5_alpha_reductase_inhibitor"),
    ("finasteride",             "5_alpha_reductase_inhibitor"),
    ("dutasteride",             "5_alpha_reductase_inhibitor"),
]


def classify_mechanism(mechanism: str) -> str:
    """Map free-text mechanism → standardised class. Returns 'other' if no match."""
    if not mechanism:
        return "other"
    mech_lower = _normalise_name(mechanism)
    for keyword, cls in MECHANISM_KEYWORD_MAP:
        if keyword in mech_lower:
            return cls
    return "other"


# ─────────────────────────────────────────────────────────────────────────────
# Synergistic pairs
# ─────────────────────────────────────────────────────────────────────────────

SYNERGISTIC_PAIRS: Set[frozenset] = {
    frozenset({"pde5_inhibitor",         "endothelin_antagonist"}),
    frozenset({"pde5_inhibitor",         "prostacyclin"}),
    frozenset({"endothelin_antagonist",  "prostacyclin"}),
    frozenset({"pde5_inhibitor",         "sgc_stimulator"}),
    frozenset({"kinase_inhibitor",       "pde5_inhibitor"}),
    frozenset({"kinase_inhibitor",       "endothelin_antagonist"}),
    frozenset({"kinase_inhibitor",       "prostacyclin"}),
    frozenset({"imid",                   "proteasome_inhibitor"}),
    frozenset({"imid",                   "corticosteroid"}),
    frozenset({"proteasome_inhibitor",   "corticosteroid"}),
    frozenset({"alkylating_agent",       "corticosteroid"}),
    frozenset({"imid",                   "anti_cd20"}),
    frozenset({"alkylating_agent",       "imid"}),
    frozenset({"alkylating_agent",       "proteasome_inhibitor"}),
    frozenset({"hdac_inhibitor",         "proteasome_inhibitor"}),
    frozenset({"hdac_inhibitor",         "imid"}),
    frozenset({"hdac_inhibitor",         "corticosteroid"}),
    frozenset({"dmard",                  "anti_tnf"}),
    frozenset({"dmard",                  "anti_il6"}),
    frozenset({"dmard",                  "jak_inhibitor"}),
    frozenset({"dmard",                  "antimalarial"}),
    frozenset({"dmard",                  "sulfonamide"}),
    frozenset({"antimalarial",           "sulfonamide"}),
    frozenset({"dmard",                  "dhodh_inhibitor"}),
    frozenset({"antimalarial",           "dhodh_inhibitor"}),
    frozenset({"dmard",                  "corticosteroid"}),
    frozenset({"antimalarial",           "corticosteroid"}),
    frozenset({"parp_inhibitor",         "alkylating_agent"}),
    frozenset({"checkpoint_inhibitor",   "anti_vegf"}),
    frozenset({"kinase_inhibitor",       "mtor_inhibitor"}),
    frozenset({"aromatase_inhibitor",    "serm"}),
    frozenset({"taxane",                 "alkylating_agent"}),
    frozenset({"anti_vegf",              "alkylating_agent"}),
    frozenset({"taxane",                 "anti_vegf"}),
    frozenset({"antimetabolite",         "alkylating_agent"}),
    frozenset({"beta_blocker",           "ace_inhibitor"}),
    frozenset({"beta_blocker",           "arb"}),
    frozenset({"beta_blocker",           "mineralocorticoid_antagonist"}),
    frozenset({"ace_inhibitor",          "mineralocorticoid_antagonist"}),
    frozenset({"arb",                    "mineralocorticoid_antagonist"}),
    frozenset({"beta_blocker",           "diuretic"}),
    frozenset({"ace_inhibitor",          "diuretic"}),
    frozenset({"arb",                    "diuretic"}),
    frozenset({"sglt2_inhibitor",        "beta_blocker"}),
    frozenset({"sglt2_inhibitor",        "ace_inhibitor"}),
    frozenset({"sglt2_inhibitor",        "arb"}),
    frozenset({"sglt2_inhibitor",        "mineralocorticoid_antagonist"}),
    frozenset({"biguanide",              "thiazolidinedione"}),
    frozenset({"biguanide",              "sulfonylurea"}),
    frozenset({"biguanide",              "sglt2_inhibitor"}),
    frozenset({"biguanide",              "glp1_agonist"}),
    frozenset({"thiazolidinedione",      "sulfonylurea"}),
    frozenset({"biguanide",              "mineralocorticoid_antagonist"}),
    frozenset({"biguanide",              "anti_androgen"}),
    frozenset({"biguanide",              "aromatase_inhibitor"}),
    frozenset({"dopamine_agonist",       "maob_inhibitor"}),
    frozenset({"dopamine_precursor",     "maob_inhibitor"}),
    frozenset({"dopamine_precursor",     "nmda_antagonist"}),
    frozenset({"maob_inhibitor",         "nmda_antagonist"}),
    frozenset({"acetylcholinesterase_inhibitor", "nmda_antagonist"}),
    frozenset({"anti_uric_acid",         "colchicine"}),
    frozenset({"nsaid",                  "colchicine"}),
    frozenset({"cox2_inhibitor",         "colchicine"}),
    frozenset({"anti_uric_acid",         "nsaid"}),
    frozenset({"colchicine",             "nsaid"}),
    frozenset({"colchicine",             "cox2_inhibitor"}),
    frozenset({"colchicine",             "corticosteroid"}),
    frozenset({"statin",                 "cholesterol_absorption_inhibitor"}),
    frozenset({"statin",                 "biguanide"}),
}


ANTAGONISTIC_PAIRS: Set[frozenset] = {
    frozenset({"pde5_inhibitor",         "nsaid"}),
    frozenset({"beta_blocker",           "dopamine_agonist"}),
    frozenset({"beta_blocker",           "dopamine_precursor"}),
    frozenset({"alkylating_agent",       "antimetabolite"}),
    frozenset({"anti_tnf",               "jak_inhibitor"}),
    frozenset({"anti_tnf",               "anti_il6"}),
    frozenset({"nsaid",                  "ace_inhibitor"}),
    frozenset({"nsaid",                  "diuretic"}),
    frozenset({"nsaid",                  "arb"}),
    frozenset({"aromatase_inhibitor",    "serm"}),
}


REDUNDANT_CLASS_GROUPS: List[Set[str]] = [
    {"ace_inhibitor", "arb"},
    {"pde5_inhibitor", "sgc_stimulator"},
    {"statin"},
    {"ace_inhibitor"},
    {"arb"},
    {"pde5_inhibitor"},
    {"endothelin_antagonist"},
    {"anti_tnf", "anti_il6", "jak_inhibitor"},
    {"alkylating_agent", "antimetabolite"},
    {"acetylcholinesterase_inhibitor"},
    {"dopamine_agonist", "dopamine_precursor"},
    {"nsaid", "cox2_inhibitor"},
    {"diuretic"},
    {"hdac_inhibitor"},
]


# ─────────────────────────────────────────────────────────────────────────────
# Disease-specific context
# ─────────────────────────────────────────────────────────────────────────────

ONCOLOGY_CLASSES: Set[str] = {
    "alkylating_agent", "antimetabolite", "taxane", "vinca_alkaloid",
    "parp_inhibitor", "checkpoint_inhibitor",
    "anti_vegf", "anthracycline", "topoisomerase_inhibitor",
    # FIX v6.1: microtubule_inhibitor added — prevents chemo drugs (vinflunine,
    # ixabepilone, epothilone) from appearing in non-cancer combos (gout, pericarditis).
    # Note: colchicine is classified separately as "colchicine" (not this class) 
    # because it IS appropriate for gout/pericarditis — this distinction is intentional.
    "microtubule_inhibitor",
}

ONCOLOGY_DISEASE_KEYWORDS: Set[str] = {
    "cancer", "carcinoma", "tumor", "tumour", "myeloma", "leukemia",
    "lymphoma", "melanoma", "glioma", "glioblastoma", "sarcoma",
    "blastoma", "adenocarcinoma", "stromal",
}

DISEASE_SPECIFIC_CLASSES: Dict[str, Set[str]] = {
    "pde5_inhibitor": {
        "pulmonary", "hypertension", "erectile", "vasodilat", "vessel",
        "vascular", "resistance", "pah",
    },
    "endothelin_antagonist": {
        "pulmonary", "hypertension", "sclerosis", "vasodilat", "vascular", "pah",
    },
    "prostacyclin": {
        "pulmonary", "hypertension", "vasodilat", "platelet", "vascular", "pah",
    },
    "sgc_stimulator": {
        "pulmonary", "hypertension", "vasodilat", "vascular", "pah",
    },
    "corticosteroid": {
        "myeloma", "lymphoma", "leukemia", "cancer", "autoimmune",
        "inflammation", "arthritis", "hematolog", "plasma cell",
        "pericarditis", "inflammatory",
    },
    "cftr_modulator": {"cystic", "fibrosis", "cftr"},
    "complement_inhibitor": {"complement", "hemoglobin", "paroxysmal"},
    "dopamine_precursor": {"parkinson", "dopamine"},
    "dopamine_agonist": {"parkinson", "restless", "dopamine"},
    "maob_inhibitor": {"parkinson", "dopamine"},
    "nmda_antagonist": {"alzheimer", "parkinson", "dementia", "als", "pain"},
    "acetylcholinesterase_inhibitor": {"alzheimer", "dementia", "cholinergic"},
    "anti_uric_acid": {"gout", "uric", "hyperuricemia"},
    "colchicine": {"gout", "pericarditis", "inflam"},
    "5_alpha_reductase_inhibitor": {"prostate", "alopecia", "baldness", "hair"},
    "potassium_channel": {"alopecia", "baldness", "hair", "hypertension"},
    "mtor_inhibitor": {"tuberous", "sclerosis", "tsc", "cancer", "renal"},
    "anticonvulsant": {"epilepsy", "seizure", "pain", "fibromyalgia"},
    "imid": {"myeloma", "lymphoma", "cancer"},
    "proteasome_inhibitor": {"myeloma", "lymphoma", "cancer"},
    "hdac_inhibitor": {"myeloma", "lymphoma", "cancer", "bipolar"},
    "serm": {"breast", "cancer", "estrogen", "osteoporosis"},
    "aromatase_inhibitor": {"breast", "cancer", "estrogen", "pcos", "ovary"},
    "anti_cd20": {"lymphoma", "arthritis", "lupus", "myeloma"},
}


def _is_oncology_disease(disease_name: str) -> bool:
    d = disease_name.lower()
    return any(k in d for k in ONCOLOGY_DISEASE_KEYWORDS)


def _get_mechanism_relevance_score(
    mechanism_class: str,
    disease_name: str,
    candidate_mechanism_score: float = 0.0,
) -> float:
    """Returns 0.0 (fully relevant) to 1.0 (completely irrelevant)."""
    if mechanism_class == "other":
        return 0.0

    disease_lower = disease_name.lower()
    is_oncology = _is_oncology_disease(disease_name)

    if mechanism_class in DISEASE_SPECIFIC_CLASSES:
        relevant_keywords = DISEASE_SPECIFIC_CLASSES[mechanism_class]
        if any(kw in disease_lower for kw in relevant_keywords):
            return 0.0
        base_irrelevance = 0.60
    elif mechanism_class in ONCOLOGY_CLASSES and not is_oncology:
        base_irrelevance = 0.80
    else:
        return 0.0

    if candidate_mechanism_score >= 0.6:
        base_irrelevance *= 0.20
    elif candidate_mechanism_score >= 0.3:
        base_irrelevance *= 0.50

    return base_irrelevance


# ─────────────────────────────────────────────────────────────────────────────
# CombinationScorer
# ─────────────────────────────────────────────────────────────────────────────

class CombinationScorer:
    def __init__(
        self,
        disease_name: str = "",
        synergy_bonus: float = 0.22,
        antagonism_penalty: float = 0.45,
        coverage_bonus_max: float = 0.15,
        redundancy_penalty: float = 0.18,
    ):
        self.disease_name = disease_name.lower().strip()
        self.synergy_bonus = synergy_bonus
        self.antagonism_penalty = antagonism_penalty
        self.coverage_bonus_max = coverage_bonus_max
        self.redundancy_penalty = redundancy_penalty

    def _gene_coverage_bonus(
        self,
        targets_a: Set[str],
        targets_b: Set[str],
        disease_genes: List[str],
    ) -> float:
        if not disease_genes:
            return 0.0
        disease_set = set(g.upper() for g in disease_genes)
        combined = (targets_a | targets_b) & disease_set
        max_indiv = max(len(targets_a & disease_set), len(targets_b & disease_set))
        extra = len(combined) - max_indiv
        if extra <= 0 or len(disease_set) == 0:
            return 0.0
        return min(extra / len(disease_set) * 2.0, self.coverage_bonus_max)

    def _redundancy_penalty_score(self, class_a: str, class_b: str) -> float:
        if class_a == class_b and class_a != "other":
            return self.redundancy_penalty * 2.0
        for group in REDUNDANT_CLASS_GROUPS:
            if class_a in group and class_b in group:
                if frozenset({class_a, class_b}) in SYNERGISTIC_PAIRS:
                    return 0.0
                return self.redundancy_penalty * 0.5
        return 0.0

    def _disease_context_penalty(
        self,
        class_a: str,
        class_b: str,
        mech_score_a: float = 0.0,
        mech_score_b: float = 0.0,
    ) -> float:
        if not self.disease_name:
            return 0.0
        irr_a = _get_mechanism_relevance_score(class_a, self.disease_name, mech_score_a)
        irr_b = _get_mechanism_relevance_score(class_b, self.disease_name, mech_score_b)
        total = irr_a * 0.50 + irr_b * 0.50
        return min(total, 0.85)

    def _resolve_class(self, drug: Dict) -> str:
        mech = drug.get("mechanism", "")
        raw_name = drug.get("drug_name", drug.get("name", ""))
        norm_name = _normalise_name(raw_name)
        cls = classify_mechanism(mech) if mech else classify_mechanism(norm_name)
        if cls == "other":
            targets = drug.get("target_genes") or drug.get("targets") or []
            for target in targets:
                c = classify_mechanism(target.lower())
                if c != "other":
                    return c
        return cls

    def score_pair(self, drug_a: Dict, drug_b: Dict, disease_genes: List[str]) -> Dict:
        name_a = drug_a.get("drug_name", drug_a.get("name", "DrugA"))
        name_b = drug_b.get("drug_name", drug_b.get("name", "DrugB"))
        score_a = float(drug_a.get("score", 0.0))
        score_b = float(drug_b.get("score", 0.0))
        mech_score_a = float(drug_a.get("mechanism_score", 0.0))
        mech_score_b = float(drug_b.get("mechanism_score", 0.0))
        class_a = self._resolve_class(drug_a)
        class_b = self._resolve_class(drug_b)
        targets_a = set(t.upper() for t in (drug_a.get("target_genes") or drug_a.get("targets") or []))
        targets_b = set(t.upper() for t in (drug_b.get("target_genes") or drug_b.get("targets") or []))
        disease_set = set(g.upper() for g in disease_genes)
        shared_genes = list((targets_a | targets_b) & disease_set)
        pair_key = frozenset({class_a, class_b})
        is_synergistic = pair_key in SYNERGISTIC_PAIRS
        is_antagonistic = pair_key in ANTAGONISTIC_PAIRS
        base_score = (score_a + score_b) / 2.0
        syn_bonus = self.synergy_bonus if is_synergistic else 0.0
        ant_penalty = self.antagonism_penalty if is_antagonistic else 0.0
        cov_bonus = self._gene_coverage_bonus(targets_a, targets_b, disease_genes)
        red_penalty = self._redundancy_penalty_score(class_a, class_b)
        ctx_penalty = self._disease_context_penalty(class_a, class_b, mech_score_a, mech_score_b)
        combo_score = max(0.0, min(1.0, base_score + syn_bonus + cov_bonus - ant_penalty - red_penalty - ctx_penalty))
        return {
            "combo_name":             f"{name_a} + {name_b}",
            "drug_a":                 name_a, "drug_b": name_b,
            "n_drugs":                2,
            "combo_score":            round(combo_score, 4),
            "base_score":             round(base_score, 4),
            "is_synergistic":         is_synergistic,
            "is_antagonistic":        is_antagonistic,
            "mechanism_a":            class_a,
            "mechanism_b":            class_b,
            "synergy_bonus":          round(syn_bonus, 4),
            "antagonism_penalty":     round(ant_penalty, 4),
            "coverage_bonus":         round(cov_bonus, 4),
            "redundancy_penalty":     round(red_penalty, 4),
            "context_penalty":        round(ctx_penalty, 4),
            "shared_genes":           shared_genes[:10],
            "combined_gene_coverage": len((targets_a | targets_b) & disease_set),
            "wet_lab_targets":        shared_genes[:5],
            "score_breakdown": {
                "base_score": base_score, "synergy_bonus": syn_bonus, "antagonism_penalty": ant_penalty,
                "coverage_bonus": cov_bonus, "redundancy_penalty": red_penalty, "context_penalty": ctx_penalty
            },
        }

    def score_triple(self, drug_a: Dict, drug_b: Dict, drug_c: Dict, disease_genes: List[str]) -> Dict:
        name_a, name_b, name_c = drug_a.get("name"), drug_b.get("name"), drug_c.get("name")
        pair_ab = self.score_pair(drug_a, drug_b, disease_genes)
        pair_ac = self.score_pair(drug_a, drug_c, disease_genes)
        pair_bc = self.score_pair(drug_b, drug_c, disease_genes)
        any_ant = pair_ab["is_antagonistic"] or pair_ac["is_antagonistic"] or pair_bc["is_antagonistic"]
        n_syn = sum([pair_ab["is_synergistic"], pair_ac["is_synergistic"], pair_bc["is_synergistic"]])
        s_ab, s_ac, s_bc = pair_ab["combo_score"], pair_ac["combo_score"], pair_bc["combo_score"]
        geo_mean = ((s_ab * s_ac * s_bc) ** (1/3)) if (s_ab * s_ac * s_bc) > 0 else 0.0
        disease_set = set(g.upper() for g in disease_genes)
        targets_a = set(t.upper() for t in (drug_a.get("targets") or []))
        targets_b = set(t.upper() for t in (drug_b.get("targets") or []))
        targets_c = set(t.upper() for t in (drug_c.get("targets") or []))
        combined_all = (targets_a | targets_b | targets_c) & disease_set
        combined_best = max(len(targets_a & disease_set), len(targets_b & disease_set), len(targets_c & disease_set))
        extra = len(combined_all) - combined_best
        triple_bonus = min(extra / max(len(disease_set), 1) * 1.5, 0.12)
        ctx_penalty = (self._disease_context_penalty(pair_ab["mechanism_a"], pair_ab["mechanism_b"], 0, 0) + 
                       self._disease_context_penalty(pair_ab["mechanism_a"], self._resolve_class(drug_c), 0, 0) * 0.5)
        combo_score = max(0.0, min(1.0, geo_mean + triple_bonus - ctx_penalty))
        if any_ant: combo_score *= 0.3
        return {
            "combo_name": f"{name_a} + {name_b} + {name_c}", "combo_score": round(combo_score, 4),
            "is_synergistic": n_syn >= 2, "is_antagonistic": any_ant, "n_synergistic_pairs": n_syn,
            "mechanism_a": pair_ab["mechanism_a"], "mechanism_b": pair_ab["mechanism_b"], "mechanism_c": self._resolve_class(drug_c)
        }


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def rank_combinations(
    candidates: List[Dict],
    disease_genes: List[str],
    disease_name: str = "",
    max_pairs: int = 6000,
    top_n_singles: int = 60,
    include_triples: bool = True,
    min_combo_score: float = 0.0,
) -> List[Dict]:
    scorer = CombinationScorer(disease_name=disease_name)
    top = sorted(candidates, key=lambda c: c.get("score", 0), reverse=True)[:top_n_singles]
    results = []
    for drug_a, drug_b in itertools.islice(itertools.combinations(top, 2), max_pairs):
        result = scorer.score_pair(drug_a, drug_b, disease_genes)
        if not result["is_antagonistic"] and result["combo_score"] >= min_combo_score:
            results.append(result)
    if include_triples and len(top) >= 3:
        for drug_a, drug_b, drug_c in itertools.islice(itertools.combinations(top[:20], 3), 800):
            result = scorer.score_triple(drug_a, drug_b, drug_c, disease_genes)
            if not result["is_antagonistic"] and result["combo_score"] >= min_combo_score:
                results.append(result)
    results.sort(key=lambda r: r["combo_score"], reverse=True)
    return results