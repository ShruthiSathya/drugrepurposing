"""
disease_ranker.py — Disease Opportunity Ranker
===============================================
TwinTrial Analytics business strategy module.

Scores diseases by total opportunity value so the team knows which
disease to build the next treatment plan for.

Opportunity score formula
-------------------------
  score = weights["market"]          × log_normalised_market_size
        + weights["feasibility"]     × pipeline_feasibility
        + weights["ip_safety"]       × ip_risk_score
        + weights["pag_strength"]    × pag_funding_score
        + weights["combo_potential"] × combo_signal_richness

All component scores are in [0, 1]. Final score is in [0, 1].

Tiers
-----
  A (≥ 0.78): Build treatment plan now. Clear business case + pipeline signal.
  B (≥ 0.68): Queue for next quarter. Good signal, needs more pipeline work.
  C (<  0.68): Watch list. Technically hard or small market or IP risk.

Usage
-----
    from backend.pipeline.disease_ranker import rank_diseases_by_opportunity

    ranking = rank_diseases_by_opportunity()
    for d in ranking[:5]:
        print(f"{d['tier']} {d['disease']:40s} score={d['opportunity_score']:.3f}")
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Disease opportunity database
#
# Schema per entry:
#   market_size_B      : addressable global market in $USD billions
#   feasibility        : 0-1, how well our pipeline scores this disease
#                        (gene annotation depth, pathway richness, API data quality)
#   ip_safety          : 0-1, how safe generic combos are from IP claims
#                        (1.0 = all drugs clearly off-patent, 0.5 = some grey zones)
#   pag_strength       : 0-1, strength of patient advocacy group + research funding
#   combo_potential    : 0-1, how likely that multi-drug synergy adds value
#   orphan_flag        : bool, qualifies for orphan drug designation incentives
#   pag_examples       : example patient advocacy groups that could pay for our report
#   known_generics     : example generic combos that anchor our treatment plan
# ─────────────────────────────────────────────────────────────────────────────

DISEASE_OPPORTUNITY_DB: Dict[str, Dict] = {

    "pulmonary arterial hypertension": {
        "market_size_B":   8,
        "feasibility":     0.92,
        "ip_safety":       0.96,
        "pag_strength":    0.90,
        "combo_potential": 0.96,
        "orphan_flag":     True,
        "pag_examples":    ["Pulmonary Hypertension Association", "Pulmonary Hypertension News"],
        "known_generics":  ["sildenafil", "bosentan", "ambrisentan", "iloprost", "treprostinil"],
        "notes":           "Triple therapy (PDE5i+ERA+prostacyclin) is established. Simulation of which combo works best per genotype is the value add.",
    },

    "multiple myeloma": {
        "market_size_B":   22,
        "feasibility":     0.88,
        "ip_safety":       0.88,
        "pag_strength":    0.88,
        "combo_potential": 0.90,
        "orphan_flag":     True,
        "pag_examples":    ["International Myeloma Foundation", "Multiple Myeloma Research Foundation"],
        "known_generics":  ["thalidomide", "dexamethasone", "melphalan", "cyclophosphamide", "bortezomib"],
        "notes":           "VTd and MPT regimens fully generic. Novel biomarker enrichment to predict responders is the product.",
    },

    "rheumatoid arthritis": {
        "market_size_B":   28,
        "feasibility":     0.92,
        "ip_safety":       0.86,
        "pag_strength":    0.82,
        "combo_potential": 0.92,
        "orphan_flag":     False,
        "pag_examples":    ["Arthritis Foundation", "Rheumatology Research Foundation"],
        "known_generics":  ["methotrexate", "hydroxychloroquine", "sulfasalazine", "leflunomide"],
        "notes":           "Triple DMARD (MTX+HCQ+SSZ) is fully generic and guideline-recommended. Which patients need which combo is the question.",
    },

    "polycystic ovary syndrome": {
        "market_size_B":   4,
        "feasibility":     0.88,
        "ip_safety":       0.97,
        "pag_strength":    0.76,
        "combo_potential": 0.88,
        "orphan_flag":     False,
        "pag_examples":    ["PCOS Challenge", "PCOS Awareness Association"],
        "known_generics":  ["metformin", "spironolactone", "clomiphene", "letrozole"],
        "notes":           "All key PCOS drugs are generic. Combination simulation for insulin resistance + androgen excess is a clear gap.",
    },

    "type 2 diabetes": {
        "market_size_B":   55,
        "feasibility":     0.92,
        "ip_safety":       0.84,
        "pag_strength":    0.62,
        "combo_potential": 0.92,
        "orphan_flag":     False,
        "pag_examples":    ["American Diabetes Association", "JDRF"],
        "known_generics":  ["metformin", "pioglitazone", "glipizide", "sitagliptin", "empagliflozin"],
        "notes":           "Massive market. Metformin + pioglitazone + older SU all generic. Biomarker stratification for combo selection is high value.",
    },

    "pericarditis": {
        "market_size_B":   1.2,
        "feasibility":     0.92,
        "ip_safety":       0.98,
        "pag_strength":    0.65,
        "combo_potential": 0.85,
        "orphan_flag":     False,
        "pag_examples":    ["Myositis Association (adjacent)", "Cardiology patient networks"],
        "known_generics":  ["colchicine", "aspirin", "ibuprofen", "anakinra"],
        "notes":           "Colchicine + aspirin is guideline standard. IL-1 inhibition combo simulation for recurrent pericarditis is a niche product.",
    },

    "gout": {
        "market_size_B":   3,
        "feasibility":     0.90,
        "ip_safety":       0.98,
        "pag_strength":    0.55,
        "combo_potential": 0.80,
        "orphan_flag":     False,
        "pag_examples":    ["Gout & Uric Acid Education Society"],
        "known_generics":  ["colchicine", "allopurinol", "febuxostat", "probenecid"],
        "notes":           "Fully generic landscape. Combo for flare prevention vs uric acid lowering is a clear product.",
    },

    "idiopathic pulmonary fibrosis": {
        "market_size_B":   3.2,
        "feasibility":     0.72,
        "ip_safety":       0.86,
        "pag_strength":    0.82,
        "combo_potential": 0.75,
        "orphan_flag":     True,
        "pag_examples":    ["Pulmonary Fibrosis Foundation", "Coalition for Pulmonary Fibrosis"],
        "known_generics":  ["pirfenidone", "acetylcysteine", "colchicine", "sildenafil"],
        "notes":           "Pirfenidone now generic. Anti-fibrotic combo simulation is nascent area.",
    },

    "parkinson disease": {
        "market_size_B":   6.5,
        "feasibility":     0.74,
        "ip_safety":       0.88,
        "pag_strength":    0.88,
        "combo_potential": 0.76,
        "orphan_flag":     False,
        "pag_examples":    ["Parkinson's Foundation", "Michael J. Fox Foundation"],
        "known_generics":  ["levodopa", "carbidopa", "ropinirole", "pramipexole", "rasagiline", "amantadine"],
        "notes":           "Rich generic landscape. Neuroprotection combo (MAO-B + dopamine agonist + amantadine) simulation is the angle.",
    },

    "systemic lupus erythematosus": {
        "market_size_B":   4.8,
        "feasibility":     0.70,
        "ip_safety":       0.86,
        "pag_strength":    0.80,
        "combo_potential": 0.74,
        "orphan_flag":     False,
        "pag_examples":    ["Lupus Foundation of America", "Lupus Research Alliance"],
        "known_generics":  ["hydroxychloroquine", "mycophenolate", "azathioprine", "belimumab"],
        "notes":           "HCQ is cornerstone generic. Combo with mycophenolate vs azathioprine for different organ involvement is the product.",
    },

    "alzheimer disease": {
        "market_size_B":   14,
        "feasibility":     0.58,
        "ip_safety":       0.87,
        "pag_strength":    0.88,
        "combo_potential": 0.68,
        "orphan_flag":     False,
        "pag_examples":    ["Alzheimer's Association", "Alzheimer's Research UK"],
        "known_generics":  ["donepezil", "memantine", "rivastigmine", "galantamine"],
        "notes":           "All approved symptomatic drugs are generic. Combo + anti-inflammatory (minocycline, simvastatin) simulation is emerging.",
    },

    "epilepsy": {
        "market_size_B":   6.8,
        "feasibility":     0.74,
        "ip_safety":       0.86,
        "pag_strength":    0.78,
        "combo_potential": 0.78,
        "orphan_flag":     False,
        "pag_examples":    ["Epilepsy Foundation", "Dravet Syndrome Foundation"],
        "known_generics":  ["valproate", "lamotrigine", "levetiracetam", "carbamazepine", "gabapentin"],
        "notes":           "Polytherapy is standard of care. Which combo works for which seizure subtype is an unmet need.",
    },

    "amyotrophic lateral sclerosis": {
        "market_size_B":   1.8,
        "feasibility":     0.54,
        "ip_safety":       0.95,
        "pag_strength":    0.85,
        "combo_potential": 0.62,
        "orphan_flag":     True,
        "pag_examples":    ["ALS Association", "Prize4Life"],
        "known_generics":  ["riluzole", "edaravone", "lithium carbonate"],
        "notes":           "Very small generic pool. Riluzole + lithium + melatonin combo is an active research area.",
    },

    "huntington disease": {
        "market_size_B":   1.5,
        "feasibility":     0.56,
        "ip_safety":       0.96,
        "pag_strength":    0.82,
        "combo_potential": 0.64,
        "orphan_flag":     True,
        "pag_examples":    ["HDSA", "Cure HD Initiative"],
        "known_generics":  ["tetrabenazine", "riluzole", "creatine", "coenzyme q10"],
        "notes":           "Tetrabenazine now generic. Neuroprotection combo simulation is the angle.",
    },

    "hypercholesterolemia": {
        "market_size_B":   18,
        "feasibility":     0.88,
        "ip_safety":       0.90,
        "pag_strength":    0.58,
        "combo_potential": 0.82,
        "orphan_flag":     False,
        "pag_examples":    ["American Heart Association", "Family Heart Foundation"],
        "known_generics":  ["atorvastatin", "rosuvastatin", "ezetimibe", "fenofibrate"],
        "notes":           "Statin + ezetimibe combo is generic standard. Biomarker-stratified combo for FH vs polygenic is the product.",
    },

    "glioblastoma": {
        "market_size_B":   3.8,
        "feasibility":     0.46,
        "ip_safety":       0.90,
        "pag_strength":    0.74,
        "combo_potential": 0.58,
        "orphan_flag":     True,
        "pag_examples":    ["National Brain Tumor Society", "Glioblastoma Foundation"],
        "known_generics":  ["temozolomide", "lomustine", "bevacizumab", "chloroquine", "metformin"],
        "notes":           "BBB penetration is the key challenge. Temozolomide + chloroquine + metformin combo has preclinical rationale.",
    },

    "pancreatic cancer": {
        "market_size_B":   4.5,
        "feasibility":     0.44,
        "ip_safety":       0.91,
        "pag_strength":    0.68,
        "combo_potential": 0.55,
        "orphan_flag":     False,
        "pag_examples":    ["Pancreatic Cancer Action Network", "Lustgarten Foundation"],
        "known_generics":  ["gemcitabine", "capecitabine", "oxaliplatin", "hydroxychloroquine"],
        "notes":           "Dense stroma makes this technically hard. Gemcitabine + HCQ autophagy inhibition has phase 2 data.",
    },

    "cystic fibrosis non-f508del": {
        "market_size_B":   7.5,
        "feasibility":     0.68,
        "ip_safety":       0.55,
        "pag_strength":    0.88,
        "combo_potential": 0.62,
        "orphan_flag":     True,
        "pag_examples":    ["Cystic Fibrosis Foundation"],
        "known_generics":  ["azithromycin", "ibuprofen", "dornase alfa", "ursodiol"],
        "notes":           "IP wall from Vertex is the problem. Only worth pursuing for non-F508del mutations not covered by Trikafta.",
    },

    "spinal muscular atrophy type 3-4": {
        "market_size_B":   2.2,
        "feasibility":     0.62,
        "ip_safety":       0.70,
        "pag_strength":    0.82,
        "combo_potential": 0.65,
        "orphan_flag":     True,
        "pag_examples":    ["Cure SMA", "SMA Foundation"],
        "known_generics":  ["riluzole", "salbutamol", "valproic acid"],
        "notes":           "Nusinersen is still patented but types 3/4 may respond to generic combo. Valproate + salbutamol is active area.",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_WEIGHTS: Dict[str, float] = {
    "market":          0.22,   # bigger market = bigger fee potential
    "feasibility":     0.30,   # can our pipeline actually find signal here?
    "ip_safety":       0.20,   # legal risk on generic combos
    "pag_strength":    0.12,   # who will pay us and how quickly?
    "combo_potential": 0.16,   # does multi-drug add genuine value here?
}

TIER_A_THRESHOLD = 0.78
TIER_B_THRESHOLD = 0.68


def rank_diseases_by_opportunity(
    weights: Optional[Dict[str, float]] = None,
    include_orphan_bonus: bool = True,
) -> List[Dict]:
    """
    Score and rank all diseases by total opportunity value.

    Parameters
    ----------
    weights : dict, optional
        Custom weights for the five scoring components.
        Must sum to ~1.0. Defaults to DEFAULT_WEIGHTS.
    include_orphan_bonus : bool
        Add +0.03 bonus for orphan diseases (FDA/EMA designation incentives
        make these more attractive for PAG partners).

    Returns
    -------
    list of dicts, sorted by opportunity_score descending.
    Each dict includes tier, all component scores, and business context.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    assert abs(sum(weights.values()) - 1.0) < 0.01, "Weights must sum to 1.0"

    # Log-normalise market size
    max_market = max(v["market_size_B"] for v in DISEASE_OPPORTUNITY_DB.values())

    results = []
    for disease, data in DISEASE_OPPORTUNITY_DB.items():
        market_norm = math.log10(data["market_size_B"] + 1) / math.log10(max_market + 1)

        score = (
            weights["market"]          * market_norm
            + weights["feasibility"]   * data["feasibility"]
            + weights["ip_safety"]     * data["ip_safety"]
            + weights["pag_strength"]  * data["pag_strength"]
            + weights["combo_potential"] * data["combo_potential"]
        )

        if include_orphan_bonus and data.get("orphan_flag"):
            score += 0.03  # FDA/EMA orphan designation bonus

        score = round(min(score, 1.0), 4)

        tier = "A" if score >= TIER_A_THRESHOLD else "B" if score >= TIER_B_THRESHOLD else "C"

        results.append({
            "disease":             disease,
            "opportunity_score":   score,
            "tier":                tier,
            "market_size_B":       data["market_size_B"],
            "market_norm":         round(market_norm, 4),
            "feasibility":         data["feasibility"],
            "ip_safety":           data["ip_safety"],
            "pag_strength":        data["pag_strength"],
            "combo_potential":     data["combo_potential"],
            "orphan_flag":         data.get("orphan_flag", False),
            "pag_examples":        data.get("pag_examples", []),
            "known_generics":      data.get("known_generics", []),
            "notes":               data.get("notes", ""),
        })

    results.sort(key=lambda x: x["opportunity_score"], reverse=True)

    # Log summary
    tier_a = [r for r in results if r["tier"] == "A"]
    tier_b = [r for r in results if r["tier"] == "B"]
    tier_c = [r for r in results if r["tier"] == "C"]

    logger.info("=" * 60)
    logger.info("DISEASE OPPORTUNITY RANKING")
    logger.info(f"  Tier A (build now):   {len(tier_a)} diseases")
    logger.info(f"  Tier B (pipeline):    {len(tier_b)} diseases")
    logger.info(f"  Tier C (watch):       {len(tier_c)} diseases")
    logger.info("  Top 5:")
    for i, r in enumerate(results[:5], 1):
        logger.info(
            f"    #{i} [{r['tier']}] {r['disease']:42s} "
            f"score={r['opportunity_score']:.3f} "
            f"market=${r['market_size_B']}B"
        )
    logger.info("=" * 60)

    return results


def get_tier_a_diseases() -> List[str]:
    """Return disease names for all Tier A opportunities."""
    return [r["disease"] for r in rank_diseases_by_opportunity() if r["tier"] == "A"]


def get_disease_brief(disease_name: str) -> Optional[Dict]:
    """Return opportunity data for a specific disease."""
    disease_lower = disease_name.lower().strip()
    for key, data in DISEASE_OPPORTUNITY_DB.items():
        if key in disease_lower or disease_lower in key:
            result = rank_diseases_by_opportunity()
            for r in result:
                if r["disease"] == key:
                    return r
    return None


def prioritise_pipeline(n: int = 5) -> List[Dict]:
    """
    Return the top N diseases to work on, with build-order rationale.
    Each entry includes the PAG contact angle and known generic anchors.
    """
    ranked = rank_diseases_by_opportunity()
    top = [r for r in ranked if r["tier"] == "A"][:n]

    for i, r in enumerate(top):
        r["build_priority"] = i + 1
        r["first_contact"] = r["pag_examples"][0] if r["pag_examples"] else "TBD"
        r["anchor_combo"]  = (
            " + ".join(r["known_generics"][:3])
            if r["known_generics"] else "TBD"
        )

    return top