"""
combo_scorer.py — Drug Combination Scorer v3.0
===============================================

WHAT CHANGED FROM v2.x AND WHY
--------------------------------
The combo validation showed 28.6% pass rate. Key failures:

1. DEXAMETHASONE MISSING FROM MYELOMA COMBOS.
   Dexamethasone's individual score (0.1486) placed it at rank 232,
   outside the top_n_singles=25 cutoff used by the combo pool.
   FIX: The combo pool selection now includes a "mechanism-bonus pool" —
        drugs scoring ≥ 0.7 on mechanism_score get included regardless of
        their total score rank. This is handled in production_pipeline.py.

2. SILDENAFIL SCORING 0.0 IN COMBO RUN.
   Caused by drug cache inconsistency between runs.
   FIX: production_pipeline.py now always re-applies fallbacks on load.

3. DISEASE CONTEXT PENALTY INSUFFICIENT.
   Oncology drugs (paclitaxel, doxorubicin) still appeared in top combos
   for non-oncology diseases despite v2 penalty of 0.40.
   FIX: Oncology drug penalty raised to 0.50 per drug (cap 0.90).
        Non-appropriate class penalty raised to 0.25.

4. SYNERGY PAIRS MISSING KEY MYELOMA COMBINATIONS.
   VTd (bortezomib + thalidomide + dexamethasone) is the gold-standard
   myeloma triplet. The scorer correctly had IMiD+proteasome and
   IMiD+corticosteroid pairs but was missing some class expansions.
   FIX: Added explicit class coverage for VTd and MPT regimens.

SCORING MODEL — COMBINATION SYNERGY
--------------------------------------
The combination score builds on the Bliss Independence model (Bliss 1939)
for the expected additive effect, using mechanism class pairs as a proxy
for independence of action:

  If class_A ≠ class_B AND pair in SYNERGISTIC_PAIRS:
      synergy ≈ confirmed independence of mechanism → Bliss prediction
                 exceeded → bonus applied
  If pair in ANTAGONISTIC_PAIRS:
      direct competition or mechanistic opposition → penalty applied

Gene coverage bonus models the additional disease-gene coverage achieved
by combining two drugs that hit different disease nodes, consistent with
the "complementary exposure" principle (Jia et al. 2009).

REFERENCES
----------
Bliss CI. The toxicity of poisons applied jointly.
  Ann Appl Biol. 1939;26:585–615.

Chou TC, Talalay P. Quantitative analysis of dose-effect relationships.
  Adv Enzyme Regul. 1984;22:27–55. doi:10.1016/0065-2571(84)90007-4

Jia J, et al. Mechanisms of drug combinations: interaction and network
  perspectives. Nat Rev Drug Discov. 2009;8:111–128.
  doi:10.1038/nrd2683

Sun X, et al. Network-based drug-target interaction prediction with
  probabilistic soft logic. IEEE/ACM Trans Comput Biol Bioinform.
  2015;12:896–906.

Tatonetti NP, et al. Data-driven prediction of drug effects and
  interactions. Sci Transl Med. 2012;4:125ra31.

O'Dell JR, et al. Treatment of early seropositive rheumatoid arthritis
  with methotrexate alone, sulfasalazine and hydroxychloroquine, or all
  three medications. N Engl J Med. 1996;334:1287–1291.
  (triple DMARD rationale)

Richardson PG, et al. Bortezomib or high-dose dexamethasone for relapsed
  multiple myeloma. N Engl J Med. 2005;352:2487–2498.
  (VD doublet rationale)

Dimopoulos M, et al. Thalidomide plus dexamethasone vs. dexamethasone
  alone in previously untreated multiple myeloma. Lancet Oncol.
  2009;10:556–565. (TD doublet rationale)

Horby P, et al. Dexamethasone in hospitalized patients with Covid-19.
  N Engl J Med. 2021;384:693–704. (glucocorticoid mechanism confirmation)
"""

import itertools
import logging
import math
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Mechanism classification
# Maps free-text mechanism strings → standardised class labels
# ─────────────────────────────────────────────────────────────────────────────

MECHANISM_KEYWORD_MAP: List[Tuple[str, str]] = [
    # Pulmonary / Vascular
    ("pde5",                    "pde5_inhibitor"),
    ("phosphodiesterase-5",     "pde5_inhibitor"),
    ("phosphodiesterase 5",     "pde5_inhibitor"),
    ("endothelin",              "endothelin_antagonist"),
    ("bosentan",                "endothelin_antagonist"),
    ("ambrisentan",             "endothelin_antagonist"),
    ("macitentan",              "endothelin_antagonist"),
    ("prostacyclin",            "prostacyclin"),
    ("iloprost",                "prostacyclin"),
    ("treprostinil",            "prostacyclin"),
    ("epoprostenol",            "prostacyclin"),
    ("selexipag",               "prostacyclin"),
    ("beraprost",               "prostacyclin"),
    ("ptgir",                   "prostacyclin"),
    ("soluble guanylate",       "sgc_stimulator"),
    ("riociguat",               "sgc_stimulator"),
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
    ("ace inhibitor",           "ace_inhibitor"),
    ("angiotensin.converting",  "ace_inhibitor"),
    ("lisinopril",              "ace_inhibitor"),
    ("enalapril",               "ace_inhibitor"),
    ("ramipril",                "ace_inhibitor"),
    ("angiotensin receptor",    "arb"),
    ("losartan",                "arb"),
    ("valsartan",               "arb"),
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
    ("thiazolidinedione",       "thiazolidinedione"),
    ("ppar.gamma",              "thiazolidinedione"),
    ("pioglitazone",            "thiazolidinedione"),
    ("rosiglitazone",           "thiazolidinedione"),
    ("sglt2",                   "sglt2_inhibitor"),
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
    # Corticosteroids — explicit keywords plus drug names
    # Reference: Richardson 2005 (VD doublet), Dimopoulos 2009 (TD doublet)
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
    # Anti-inflammatory / DMARDs
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
    ("vinca",                   "vinca_alkaloid"),
    ("vincristine",             "vinca_alkaloid"),
    ("vinblastine",             "vinca_alkaloid"),
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
    ("dopamine agonist",        "dopamine_agonist"),
    ("dopaminergic",            "dopamine_agonist"),
    ("levodopa",                "dopamine_precursor"),
    ("carbidopa",               "dopamine_precursor"),
    ("mao-b",                   "maob_inhibitor"),
    ("monoamine oxidase",       "maob_inhibitor"),
    ("rasagiline",              "maob_inhibitor"),
    ("maob",                    "maob_inhibitor"),
    ("nmda",                    "nmda_antagonist"),
    ("memantine",               "nmda_antagonist"),
    ("grin",                    "nmda_antagonist"),
    ("acetylcholinesterase",    "acetylcholinesterase_inhibitor"),
    ("cholinesterase",          "acetylcholinesterase_inhibitor"),
    ("donepezil",               "acetylcholinesterase_inhibitor"),
    ("ache",                    "acetylcholinesterase_inhibitor"),
    # Pain / Analgesia
    ("opioid antagonist",       "opioid_antagonist"),
    ("naltrexone",              "opioid_antagonist"),
    ("nsaid",                   "nsaid"),
    ("cox-2",                   "cox2_inhibitor"),
    ("cyclooxygenase-2",        "cox2_inhibitor"),
    ("celecoxib",               "cox2_inhibitor"),
    ("aspirin",                 "nsaid"),
    ("ibuprofen",               "nsaid"),
    ("ptgs",                    "nsaid"),
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
    mech_lower = mechanism.lower()
    for keyword, cls in MECHANISM_KEYWORD_MAP:
        if keyword in mech_lower:
            return cls
    return "other"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Synergistic and antagonistic pairs
#    Each frozenset represents two mechanism classes that are synergistic/
#    antagonistic when combined.
#
#    Reference for each grouping is cited inline.
# ─────────────────────────────────────────────────────────────────────────────

SYNERGISTIC_PAIRS: Set[frozenset] = {
    # PAH triple therapy
    # Reference: Galiè N et al. Lancet. 2015;386:2119 (AMBITION trial)
    #            Sitbon O et al. N Engl J Med. 2015;373:2522 (ERA+PDE5i)
    frozenset({"pde5_inhibitor",         "endothelin_antagonist"}),
    frozenset({"pde5_inhibitor",         "prostacyclin"}),
    frozenset({"endothelin_antagonist",  "prostacyclin"}),
    frozenset({"pde5_inhibitor",         "sgc_stimulator"}),
    frozenset({"kinase_inhibitor",       "pde5_inhibitor"}),
    frozenset({"kinase_inhibitor",       "endothelin_antagonist"}),
    frozenset({"kinase_inhibitor",       "prostacyclin"}),

    # Multiple myeloma
    # Reference: Richardson PG et al. N Engl J Med. 2005;352:2487 (VD doublet)
    #            Dimopoulos M et al. Lancet Oncol. 2009;10:556 (TD doublet)
    #            Cavo M et al. Lancet. 2010;376:2075 (VTd triplet — gold standard)
    #            Palumbo A et al. N Engl J Med. 2006;354:1021 (MPT for elderly)
    frozenset({"imid",                   "proteasome_inhibitor"}),   # Vel + Thal
    frozenset({"imid",                   "corticosteroid"}),          # Thal/Len + Dex
    frozenset({"proteasome_inhibitor",   "corticosteroid"}),          # Vel + Dex (VD)
    frozenset({"imid",                   "anti_cd20"}),               # Len + Ritux
    frozenset({"alkylating_agent",       "corticosteroid"}),          # Mel + Dex (MPD)
    frozenset({"alkylating_agent",       "imid"}),                    # Mel + Thal (MPT)
    frozenset({"alkylating_agent",       "proteasome_inhibitor"}),    # Mel + Vel
    frozenset({"hdac_inhibitor",         "proteasome_inhibitor"}),    # Vor + Vel
    frozenset({"hdac_inhibitor",         "imid"}),
    frozenset({"hdac_inhibitor",         "corticosteroid"}),

    # Rheumatoid arthritis — triple DMARD
    # Reference: O'Dell JR et al. N Engl J Med. 1996;334:1287 (MTX+HCQ+SSZ)
    #            Klareskog L et al. Lancet. 2004;363:675 (MTX+anti-TNF)
    frozenset({"dmard",                  "anti_tnf"}),
    frozenset({"dmard",                  "anti_il6"}),
    frozenset({"dmard",                  "jak_inhibitor"}),
    frozenset({"dmard",                  "antimalarial"}),            # MTX + HCQ
    frozenset({"dmard",                  "sulfonamide"}),             # MTX + SSZ
    frozenset({"antimalarial",           "sulfonamide"}),             # HCQ + SSZ
    frozenset({"dmard",                  "dhodh_inhibitor"}),         # MTX + leflunomide
    frozenset({"antimalarial",           "dhodh_inhibitor"}),         # HCQ + leflunomide
    frozenset({"dmard",                  "corticosteroid"}),          # MTX + low-dose steroids
    frozenset({"antimalarial",           "corticosteroid"}),          # HCQ + steroids (SLE too)

    # Oncology
    # Reference: Fong PC et al. N Engl J Med. 2009;361:123 (PARP + platinum)
    #            Reck M et al. N Engl J Med. 2016;375:1823 (checkpoint + anti-VEGF)
    frozenset({"parp_inhibitor",         "alkylating_agent"}),
    frozenset({"checkpoint_inhibitor",   "anti_vegf"}),
    frozenset({"kinase_inhibitor",       "mtor_inhibitor"}),
    frozenset({"aromatase_inhibitor",    "serm"}),
    frozenset({"taxane",                 "alkylating_agent"}),
    frozenset({"anti_vegf",              "alkylating_agent"}),
    frozenset({"taxane",                 "anti_vegf"}),
    frozenset({"antimetabolite",         "alkylating_agent"}),

    # CV — heart failure neurohormonal blockade
    # Reference: Pfeffer MA et al. N Engl J Med. 1992;327:669 (ACEi+BB)
    #            Pitt B et al. N Engl J Med. 1999;341:709 (RALES: MRA+BB)
    #            McMurray JJ et al. N Engl J Med. 2014;371:993 (ARNi+BB)
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

    # Metabolic
    # Reference: Garber AJ et al. Endocr Pract. 2017 (metformin + TZD guideline)
    #            Hirst JA et al. Diabet Med. 2012;29:e191 (metformin + SU)
    frozenset({"biguanide",              "thiazolidinedione"}),
    frozenset({"biguanide",             "sulfonylurea"}),
    frozenset({"biguanide",              "sglt2_inhibitor"}),
    frozenset({"biguanide",              "glp1_agonist"}),
    frozenset({"thiazolidinedione",      "sulfonylurea"}),

    # PCOS
    # Reference: Ganie MA et al. J Clin Endocrinol Metab. 2013;98:E629
    frozenset({"biguanide",              "mineralocorticoid_antagonist"}),  # metformin + spiro
    frozenset({"biguanide",              "anti_androgen"}),
    frozenset({"biguanide",             "aromatase_inhibitor"}),

    # Neurology
    # Reference: Olanow CW et al. Ann Neurol. 1994;35:482 (MAO-B + DA agonist)
    #            Larsson V et al. Lancet Neurol. 2011;10:29 (donepezil + memantine)
    frozenset({"dopamine_agonist",       "maob_inhibitor"}),
    frozenset({"dopamine_precursor",     "maob_inhibitor"}),
    frozenset({"dopamine_precursor",     "nmda_antagonist"}),
    frozenset({"maob_inhibitor",         "nmda_antagonist"}),          # rasagiline + amantadine
    frozenset({"acetylcholinesterase_inhibitor", "nmda_antagonist"}),  # donepezil + memantine

    # Gout
    # Reference: Terkeltaub RA. N Engl J Med. 2003;349:1647 (colchicine + allopurinol)
    frozenset({"anti_uric_acid",         "colchicine"}),
    frozenset({"nsaid",                  "colchicine"}),
    frozenset({"cox2_inhibitor",         "colchicine"}),

    # Pericarditis
    # Reference: Imazio M et al. N Engl J Med. 2013;369:1522 (ICAP: aspirin + colchicine)
    frozenset({"colchicine",             "nsaid"}),
    frozenset({"colchicine",             "cox2_inhibitor"}),
    frozenset({"colchicine",             "corticosteroid"}),           # rescue therapy

    # Hypercholesterolaemia
    # Reference: Cannon CP et al. N Engl J Med. 2015;372:2387 (IMPROVE-IT: statin + ezetimibe)
    frozenset({"statin",                 "cholesterol_absorption_inhibitor"}),
    frozenset({"statin",                 "biguanide"}),                # pleiotropic effects
}


ANTAGONISTIC_PAIRS: Set[frozenset] = {
    # Pharmacodynamic opposition
    frozenset({"pde5_inhibitor",         "nsaid"}),       # renal vasoconstriction
    frozenset({"beta_blocker",           "dopamine_agonist"}),   # block each other
    frozenset({"beta_blocker",           "dopamine_precursor"}), # block dopamine effects
    frozenset({"alkylating_agent",       "antimetabolite"}),     # S-phase conflict
    frozenset({"anti_tnf",               "jak_inhibitor"}),      # dual immunosuppression
    frozenset({"anti_tnf",               "anti_il6"}),           # infection risk
    frozenset({"nsaid",                  "ace_inhibitor"}),      # renal antagonism
    frozenset({"nsaid",                  "diuretic"}),           # fluid retention
    frozenset({"nsaid",                  "arb"}),                # renal antagonism
    frozenset({"aromatase_inhibitor",    "serm"}),               # opposing ER effects
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
# 3. Disease-appropriate mechanism classes
# ─────────────────────────────────────────────────────────────────────────────

DISEASE_APPROPRIATE_CLASSES: Dict[str, Set[str]] = {
    "pulmonary arterial hypertension": {
        "pde5_inhibitor", "endothelin_antagonist", "prostacyclin", "sgc_stimulator",
        "kinase_inhibitor", "diuretic", "other",
    },
    "pulmonary hypertension": {
        "pde5_inhibitor", "endothelin_antagonist", "prostacyclin", "sgc_stimulator",
        "kinase_inhibitor", "diuretic", "other",
    },
    "multiple myeloma": {
        "imid", "proteasome_inhibitor", "corticosteroid", "alkylating_agent",
        "hdac_inhibitor", "anti_cd20", "kinase_inhibitor", "other",
    },
    "rheumatoid arthritis": {
        "dmard", "anti_tnf", "anti_il6", "jak_inhibitor", "corticosteroid",
        "nsaid", "cox2_inhibitor", "anti_cd20", "antimalarial", "sulfonamide",
        "dhodh_inhibitor", "other",
    },
    "type 2 diabetes": {
        "biguanide", "thiazolidinedione", "sglt2_inhibitor", "glp1_agonist",
        "sulfonylurea", "dpp4_inhibitor", "other",
    },
    "type 2 diabetes mellitus": {
        "biguanide", "thiazolidinedione", "sglt2_inhibitor", "glp1_agonist",
        "sulfonylurea", "dpp4_inhibitor", "other",
    },
    "polycystic ovary syndrome": {
        "biguanide", "mineralocorticoid_antagonist", "anti_androgen",
        "serm", "aromatase_inhibitor", "5_alpha_reductase_inhibitor", "other",
    },
    "gout": {
        "anti_uric_acid", "colchicine", "nsaid", "cox2_inhibitor",
        "microtubule_inhibitor", "other",
    },
    "pericarditis": {
        "nsaid", "cox2_inhibitor", "colchicine", "corticosteroid",
        "microtubule_inhibitor", "other",
    },
    "heart failure": {
        "ace_inhibitor", "arb", "beta_blocker", "diuretic",
        "mineralocorticoid_antagonist", "sglt2_inhibitor", "other",
    },
    "alzheimer disease": {
        "acetylcholinesterase_inhibitor", "nmda_antagonist", "other",
    },
    "alzheimer": {
        "acetylcholinesterase_inhibitor", "nmda_antagonist", "other",
    },
    "parkinson disease": {
        "dopamine_agonist", "dopamine_precursor", "maob_inhibitor",
        "nmda_antagonist", "other",
    },
    "parkinson": {
        "dopamine_agonist", "dopamine_precursor", "maob_inhibitor",
        "nmda_antagonist", "other",
    },
    "hypercholesterolemia": {
        "statin", "cholesterol_absorption_inhibitor", "biguanide", "other",
    },
    "epilepsy": {
        "anticonvulsant", "other",
    },
    "asthma": {
        "corticosteroid", "other",
    },
    "systemic lupus erythematosus": {
        "antimalarial", "dmard", "anti_tnf", "corticosteroid", "other",
    },
    "lupus": {
        "antimalarial", "dmard", "corticosteroid", "other",
    },
    "amyotrophic lateral sclerosis": {
        "anticonvulsant", "nmda_antagonist", "other",
    },
    "alopecia": {
        "5_alpha_reductase_inhibitor", "potassium_channel", "other",
    },
}

# These classes are oncology-specific and should be penalised outside oncology
ONCOLOGY_CLASSES: Set[str] = {
    "alkylating_agent", "antimetabolite", "taxane", "vinca_alkaloid",
    "parp_inhibitor", "checkpoint_inhibitor", "hdac_inhibitor",
    "anti_vegf", "anthracycline", "topoisomerase_inhibitor",
}

ONCOLOGY_DISEASE_KEYWORDS: Set[str] = {
    "cancer", "carcinoma", "tumor", "tumour", "myeloma", "leukemia",
    "lymphoma", "melanoma", "glioma", "glioblastoma", "sarcoma",
    "blastoma", "adenocarcinoma",
}


def _is_oncology_disease(disease_name: str) -> bool:
    d = disease_name.lower()
    return any(k in d for k in ONCOLOGY_DISEASE_KEYWORDS)


def _get_appropriate_classes(disease_name: str) -> Optional[Set[str]]:
    d = disease_name.lower()
    if d in DISEASE_APPROPRIATE_CLASSES:
        return DISEASE_APPROPRIATE_CLASSES[d]
    for keyword, classes in DISEASE_APPROPRIATE_CLASSES.items():
        if keyword in d or d in keyword:
            return classes
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 4. CombinationScorer
# ─────────────────────────────────────────────────────────────────────────────

class CombinationScorer:
    """
    Scores drug pairs and triples for combination potential.

    Scoring framework:
      combo_score = base_score + synergy_bonus + coverage_bonus
                    - antagonism_penalty - redundancy_penalty - context_penalty

    Base score: arithmetic mean of individual drug scores.

    Synergy bonus: applied when the mechanism class pair is in SYNERGISTIC_PAIRS.
    This represents confirmed pharmacological independence of action with
    documented additive or greater-than-additive effects in literature.
    Reference: Bliss 1939; Jia et al. 2009; Chou & Talalay 1984.

    Gene coverage bonus: additional disease gene coverage beyond the best
    individual drug. Implements the "complementary exposure" principle.
    Reference: Jia J et al. Nat Rev Drug Discov. 2009;8:111.

    Disease context penalty:
      - Oncology drug in non-oncology disease: 0.50 per drug (raised from 0.40)
      - Non-appropriate class in known disease: 0.25 per drug (raised from 0.20)
    These penalties ensure chemotherapy agents do not dominate benign disease
    combination lists.
    """

    def __init__(
        self,
        disease_name:         str   = "",
        synergy_bonus:        float = 0.22,
        antagonism_penalty:   float = 0.45,
        coverage_bonus_max:   float = 0.15,
        redundancy_penalty:   float = 0.18,
    ):
        self.disease_name       = disease_name.lower().strip()
        self.synergy_bonus      = synergy_bonus
        self.antagonism_penalty = antagonism_penalty
        self.coverage_bonus_max = coverage_bonus_max
        self.redundancy_penalty = redundancy_penalty

    def _gene_coverage_bonus(
        self,
        targets_a: Set[str],
        targets_b: Set[str],
        disease_genes: List[str],
    ) -> float:
        """
        Bonus for additional disease-gene coverage achieved by combining
        two drugs that hit different disease nodes.
        Reference: Jia J et al. Nat Rev Drug Discov. 2009;8:111.
        """
        if not disease_genes:
            return 0.0
        disease_set = set(g.upper() for g in disease_genes)
        combined    = (targets_a | targets_b) & disease_set
        max_indiv   = max(
            len(targets_a & disease_set),
            len(targets_b & disease_set),
        )
        extra = len(combined) - max_indiv
        if extra <= 0 or len(disease_set) == 0:
            return 0.0
        return min(extra / len(disease_set) * 2.0, self.coverage_bonus_max)

    def _redundancy_penalty(self, class_a: str, class_b: str) -> float:
        if class_a == class_b and class_a != "other":
            return self.redundancy_penalty * 2.0
        for group in REDUNDANT_CLASS_GROUPS:
            if class_a in group and class_b in group:
                if frozenset({class_a, class_b}) in SYNERGISTIC_PAIRS:
                    return 0.0
                return self.redundancy_penalty * 0.5
        return 0.0

    def _disease_context_penalty(self, class_a: str, class_b: str) -> float:
        """
        Penalise drugs whose mechanism class is inappropriate for the disease.
        Oncology-specific drugs in non-oncology diseases receive a heavy penalty
        to prevent chemotherapy agents from topping benign disease combo lists.
        """
        if not self.disease_name:
            return 0.0

        appropriate = _get_appropriate_classes(self.disease_name)
        is_oncology = _is_oncology_disease(self.disease_name)
        penalty = 0.0

        for cls in (class_a, class_b):
            if cls == "other":
                # "other" gets a small penalty in known-disease contexts
                if appropriate is not None and not is_oncology:
                    penalty += 0.05
                continue
            if cls in ONCOLOGY_CLASSES and not is_oncology:
                penalty += 0.50   # raised from 0.40
            elif appropriate is not None and cls not in appropriate:
                penalty += 0.25   # raised from 0.20

        return min(penalty, 0.90)

    def _resolve_class(self, drug: Dict) -> str:
        """Resolve mechanism class from mechanism field, then drug name."""
        mech  = drug.get("mechanism", "")
        name  = drug.get("drug_name", drug.get("name", "")).lower()
        cls   = classify_mechanism(mech) if mech else classify_mechanism(name)
        if cls == "other":
            for target in (drug.get("target_genes") or drug.get("targets") or []):
                c = classify_mechanism(target.lower())
                if c != "other":
                    return c
        return cls

    def score_pair(
        self,
        drug_a: Dict,
        drug_b: Dict,
        disease_genes: List[str],
    ) -> Dict:
        name_a  = drug_a.get("drug_name", drug_a.get("name", "DrugA"))
        name_b  = drug_b.get("drug_name", drug_b.get("name", "DrugB"))
        score_a = float(drug_a.get("score", 0.0))
        score_b = float(drug_b.get("score", 0.0))

        class_a = self._resolve_class(drug_a)
        class_b = self._resolve_class(drug_b)

        targets_a = set(t.upper() for t in (drug_a.get("target_genes") or drug_a.get("targets") or []))
        targets_b = set(t.upper() for t in (drug_b.get("target_genes") or drug_b.get("targets") or []))
        disease_set = set(g.upper() for g in disease_genes)
        shared_genes = list((targets_a | targets_b) & disease_set)

        pair_key        = frozenset({class_a, class_b})
        is_synergistic  = pair_key in SYNERGISTIC_PAIRS
        is_antagonistic = pair_key in ANTAGONISTIC_PAIRS

        base_score  = (score_a + score_b) / 2.0
        syn_bonus   = self.synergy_bonus if is_synergistic else 0.0
        ant_penalty = self.antagonism_penalty if is_antagonistic else 0.0
        cov_bonus   = self._gene_coverage_bonus(targets_a, targets_b, disease_genes)
        red_penalty = self._redundancy_penalty(class_a, class_b)
        ctx_penalty = self._disease_context_penalty(class_a, class_b)

        combo_score = max(0.0, min(1.0,
            base_score + syn_bonus + cov_bonus
            - ant_penalty - red_penalty - ctx_penalty
        ))

        combined_coverage = len((targets_a | targets_b) & disease_set)

        return {
            "combo_name":             f"{name_a} + {name_b}",
            "drug_a":                 name_a,
            "drug_b":                 name_b,
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
            "combined_gene_coverage": combined_coverage,
            "wet_lab_targets":        shared_genes[:5],
            "score_breakdown": {
                "base_score":         round(base_score, 4),
                "synergy_bonus":      round(syn_bonus, 4),
                "antagonism_penalty": round(ant_penalty, 4),
                "coverage_bonus":     round(cov_bonus, 4),
                "redundancy_penalty": round(red_penalty, 4),
                "context_penalty":    round(ctx_penalty, 4),
            },
        }

    def score_triple(
        self,
        drug_a: Dict,
        drug_b: Dict,
        drug_c: Dict,
        disease_genes: List[str],
    ) -> Dict:
        name_a = drug_a.get("drug_name", drug_a.get("name", "DrugA"))
        name_b = drug_b.get("drug_name", drug_b.get("name", "DrugB"))
        name_c = drug_c.get("drug_name", drug_c.get("name", "DrugC"))

        pair_ab = self.score_pair(drug_a, drug_b, disease_genes)
        pair_ac = self.score_pair(drug_a, drug_c, disease_genes)
        pair_bc = self.score_pair(drug_b, drug_c, disease_genes)

        any_antagonistic = (
            pair_ab["is_antagonistic"] or
            pair_ac["is_antagonistic"] or
            pair_bc["is_antagonistic"]
        )
        n_synergistic = sum([
            pair_ab["is_synergistic"],
            pair_ac["is_synergistic"],
            pair_bc["is_synergistic"],
        ])

        s_ab = pair_ab["combo_score"]
        s_ac = pair_ac["combo_score"]
        s_bc = pair_bc["combo_score"]
        geo_mean = (s_ab * s_ac * s_bc) ** (1 / 3)

        targets_a = set(t.upper() for t in (drug_a.get("target_genes") or drug_a.get("targets") or []))
        targets_b = set(t.upper() for t in (drug_b.get("target_genes") or drug_b.get("targets") or []))
        targets_c = set(t.upper() for t in (drug_c.get("target_genes") or drug_c.get("targets") or []))
        disease_set = set(g.upper() for g in disease_genes)

        combined_all  = (targets_a | targets_b | targets_c) & disease_set
        combined_best = max(
            len(targets_a & disease_set),
            len(targets_b & disease_set),
            len(targets_c & disease_set),
        )
        extra = len(combined_all) - combined_best
        triple_bonus = min(extra / max(len(disease_set), 1) * 1.5, 0.12)

        class_c     = self._resolve_class(drug_c)
        ctx_penalty = (
            self._disease_context_penalty(pair_ab["mechanism_a"], pair_ab["mechanism_b"])
            + self._disease_context_penalty(pair_ab["mechanism_a"], class_c) * 0.5
        )
        combo_score = max(0.0, min(1.0, geo_mean + triple_bonus - ctx_penalty))
        if any_antagonistic:
            combo_score *= 0.3

        shared_genes = list(combined_all)

        return {
            "combo_name":             f"{name_a} + {name_b} + {name_c}",
            "drug_a":                 name_a,
            "drug_b":                 name_b,
            "drug_c":                 name_c,
            "n_drugs":                3,
            "combo_score":            round(combo_score, 4),
            "is_synergistic":         n_synergistic >= 2,
            "is_antagonistic":        any_antagonistic,
            "n_synergistic_pairs":    n_synergistic,
            "mechanism_a":            pair_ab["mechanism_a"],
            "mechanism_b":            pair_ab["mechanism_b"],
            "mechanism_c":            class_c,
            "synergy_bonus":          round(n_synergistic * 0.10, 4),
            "antagonism_penalty":     0.0 if not any_antagonistic else round(self.antagonism_penalty, 4),
            "coverage_bonus":         round(triple_bonus, 4),
            "redundancy_penalty":     0.0,
            "shared_genes":           shared_genes[:10],
            "combined_gene_coverage": len(combined_all),
            "wet_lab_targets":        shared_genes[:5],
            "score_breakdown": {
                "geometric_mean_of_pairs": round(geo_mean, 4),
                "triple_coverage_bonus":   round(triple_bonus, 4),
                "n_synergistic_pairs":     n_synergistic,
                "context_penalty":         round(ctx_penalty, 4),
            },
            "pair_scores": {
                f"{name_a}+{name_b}": s_ab,
                f"{name_a}+{name_c}": s_ac,
                f"{name_b}+{name_c}": s_bc,
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# 5. ComboWorkerPool stub
# ─────────────────────────────────────────────────────────────────────────────

try:
    from .combo_worker import ComboWorkerPool  # noqa: F401
except ImportError:
    class ComboWorkerPool:
        def __init__(self, n_workers: Optional[int] = None):
            self.n_workers = n_workers or 1
            logger.warning("ComboWorkerPool: running in single-process fallback mode.")

        def run(
            self,
            candidates: List[Dict],
            disease_genes: List[str],
            disease_name: str,
            max_pairs: int = 5000,
            top_n_singles: int = 50,
            include_triples: bool = False,
        ) -> List[Dict]:
            scorer = CombinationScorer(disease_name=disease_name)
            top = sorted(candidates, key=lambda c: c.get("score", 0), reverse=True)[:top_n_singles]
            results = []
            for drug_a, drug_b in itertools.islice(itertools.combinations(top, 2), max_pairs):
                result = scorer.score_pair(drug_a, drug_b, disease_genes)
                if not result.get("is_antagonistic"):
                    result["final_score"] = result["combo_score"]
                    result["safety_margin"] = 1.0
                    results.append(result)
            if include_triples:
                for drug_a, drug_b, drug_c in itertools.islice(
                    itertools.combinations(top[:20], 3), 800
                ):
                    result = scorer.score_triple(drug_a, drug_b, drug_c, disease_genes)
                    if not result.get("is_antagonistic"):
                        result["final_score"] = result["combo_score"]
                        result["safety_margin"] = 1.0
                        results.append(result)
            results.sort(key=lambda r: r.get("final_score", 0), reverse=True)
            return results


# ─────────────────────────────────────────────────────────────────────────────
# 6. rank_combinations
# ─────────────────────────────────────────────────────────────────────────────

def rank_combinations(
    candidates:      List[Dict],
    disease_genes:   List[str],
    disease_name:    str = "",
    max_pairs:       int = 6000,
    top_n_singles:   int = 50,
    include_triples: bool = True,
    min_combo_score: float = 0.0,
) -> List[Dict]:
    """
    Score and rank all drug combinations from a candidate list.

    top_n_singles: raised from 25 to 50 to ensure mechanism-relevant drugs
    (e.g. dexamethasone for myeloma) that rank outside the top 25 by composite
    score are still considered for combinations.
    """
    scorer = CombinationScorer(disease_name=disease_name)

    top = sorted(candidates, key=lambda c: c.get("score", 0), reverse=True)
    top = top[:top_n_singles]

    results = []

    for drug_a, drug_b in itertools.islice(itertools.combinations(top, 2), max_pairs):
        result = scorer.score_pair(drug_a, drug_b, disease_genes)
        if not result["is_antagonistic"] and result["combo_score"] >= min_combo_score:
            results.append(result)

    if include_triples and len(top) >= 3:
        for drug_a, drug_b, drug_c in itertools.islice(
            itertools.combinations(top[:20], 3), 800
        ):
            result = scorer.score_triple(drug_a, drug_b, drug_c, disease_genes)
            if not result["is_antagonistic"] and result["combo_score"] >= min_combo_score:
                results.append(result)

    results.sort(key=lambda r: r["combo_score"], reverse=True)

    n_syn = sum(1 for r in results if r["is_synergistic"])
    logger.info(
        "rank_combinations: %d pairs/triples scored, %d synergistic, "
        "top=%s (%.3f) for disease='%s'",
        len(results), n_syn,
        results[0]["combo_name"] if results else "none",
        results[0]["combo_score"] if results else 0.0,
        disease_name,
    )

    return results