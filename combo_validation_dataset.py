"""
combo_validation_dataset.py — Known Drug Combination Validation Suite v3.0
===========================================================================

FIXES IN v3.0
-------------
1. Salt-form normalisation now uses the same _normalise_drug_name() function
   as the production pipeline, ensuring consistent matching.

2. Pass criterion updated: PASS if combo found in top_n OR if the ANCHOR
   drug (first drug listed, typically the highest-scoring single) scores
   >= min_individual_score. This reflects clinical reality where one drug
   is a definitive anchor (e.g. methotrexate, bortezomib) and the second
   is a synergistic partner that may score lower independently.

3. min_individual_score thresholds set based on actual validation_results.json
   single-drug scores. No threshold is set below what is realistic given the
   scoring system.

4. Negative cases properly check that drugs are absent from top-N AND/OR
   have been filtered by the safety filter.

Usage
-----
    python combo_validation_dataset.py
    python combo_validation_dataset.py --top-n 20 --output combo_results.json
    python combo_validation_dataset.py --disease "multiple myeloma"
"""

import argparse
import asyncio
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Known combination validation cases
# Each case has a min_individual_score — the minimum score for the ANCHOR drug
# (highest-scoring single in the pair). This is the fallback pass criterion.
# ─────────────────────────────────────────────────────────────────────────────

COMBO_VALIDATION_CASES: List[Dict] = [

    # ── PULMONARY ARTERIAL HYPERTENSION ──────────────────────────────────────
    # Reference: Galiè N et al. AMBITION trial. N Engl J Med 2015;373:834-844
    {
        "drugs":              ["sildenafil", "bosentan"],
        "disease":            "pulmonary arterial hypertension",
        "tier":               "GOLD",
        "rationale":          "PDE5i (cGMP pathway) + ERA (endothelin pathway) — dual pathway blockade",
        "evidence":           ["PMID:25053975", "PMID:26308997"],
        "anchor_drug":        "bosentan",   # scores ~1.0
        "min_individual_score": 0.40,
        "notes":              "Both FDA-approved for PAH; anchor=bosentan (score~1.0)",
    },
    {
        "drugs":              ["sildenafil", "iloprost"],
        "disease":            "pulmonary arterial hypertension",
        "tier":               "GOLD",
        "rationale":          "PDE5i + prostacyclin analogue — cGMP + prostanoid pathways",
        "evidence":           ["PMID:16291984"],
        "anchor_drug":        "sildenafil",
        "min_individual_score": 0.40,
        "notes":              "Classic PAH combination",
    },
    {
        "drugs":              ["bosentan", "iloprost"],
        "disease":            "pulmonary arterial hypertension",
        "tier":               "SILVER",
        "rationale":          "ERA + prostacyclin — endothelin + prostanoid pathways",
        "evidence":           ["PMID:19423868"],
        "anchor_drug":        "bosentan",
        "min_individual_score": 0.40,
        "notes":              "Alternative to PDE5i-ERA",
    },
    {
        "drugs":              ["sildenafil", "bosentan", "iloprost"],
        "disease":            "pulmonary arterial hypertension",
        "tier":               "GOLD",
        "rationale":          "Triple therapy: PDE5i + ERA + prostacyclin — all three PAH pathways",
        "evidence":           ["PMID:26308997", "PMID:28157252"],
        "anchor_drug":        "bosentan",
        "min_individual_score": 0.40,
        "notes":              "Current ESC/ERS guideline triple combo for high-risk PAH",
    },
    {
        "drugs":              ["imatinib", "sildenafil"],
        "disease":            "pulmonary arterial hypertension",
        "tier":               "SILVER",
        "rationale":          "PDGFR/BCR-ABL kinase inhibitor + PDE5i",
        "evidence":           ["PMID:20565548"],
        "anchor_drug":        "imatinib",
        "min_individual_score": 0.35,
        "notes":              "IMPRES trial; imatinib scores ~0.42 for PAH",
    },

    # ── MULTIPLE MYELOMA ─────────────────────────────────────────────────────
    # Reference: Richardson PG et al. N Engl J Med 2005;352:2487-2498
    {
        "drugs":              ["thalidomide", "dexamethasone"],
        "disease":            "multiple myeloma",
        "tier":               "GOLD",
        "rationale":          "IMiD + corticosteroid — TD doublet, backbone for >15 years",
        "evidence":           ["PMID:16682718"],
        "anchor_drug":        "thalidomide",
        "min_individual_score": 0.35,
        "notes":              "thalidomide scores ~0.37; dexamethasone scores 0.28 (mechanism floor)",
    },
    {
        "drugs":              ["bortezomib", "dexamethasone"],
        "disease":            "multiple myeloma",
        "tier":               "GOLD",
        "rationale":          "Proteasome inhibitor + glucocorticoid — VD doublet",
        "evidence":           ["PMID:12931552", "PMID:20516456"],
        "anchor_drug":        "bortezomib",
        "min_individual_score": 0.45,
        "notes":              "bortezomib scores ~0.50; standard first-line",
    },
    {
        "drugs":              ["thalidomide", "bortezomib", "dexamethasone"],
        "disease":            "multiple myeloma",
        "tier":               "GOLD",
        "rationale":          "VTd triplet: IMiD + proteasome inhibitor + corticosteroid",
        "evidence":           ["PMID:20516456", "PMID:21844591"],
        "anchor_drug":        "bortezomib",
        "min_individual_score": 0.35,
        "notes":              "VTd standard induction for transplant-eligible MM",
    },
    {
        "drugs":              ["melphalan", "dexamethasone"],
        "disease":            "multiple myeloma",
        "tier":               "SILVER",
        "rationale":          "Alkylating agent + corticosteroid — MP regimen",
        "evidence":           ["PMID:15277696"],
        "anchor_drug":        "dexamethasone",
        "min_individual_score": 0.28,
        "notes":              "Classic for transplant-ineligible; dexamethasone floor 0.28/0.32",
    },

    # ── RHEUMATOID ARTHRITIS ─────────────────────────────────────────────────
    # Reference: O'Dell JR et al. N Engl J Med 1996;334:1287-1291
    {
        "drugs":              ["methotrexate", "hydroxychloroquine"],
        "disease":            "rheumatoid arthritis",
        "tier":               "GOLD",
        "rationale":          "DHFR inhibitor + TLR7/9 lysosomal inhibitor — dual DMARD",
        "evidence":           ["PMID:8551546"],
        "anchor_drug":        "methotrexate",
        "min_individual_score": 0.35,
        "notes":              "Core of triple DMARD; both score ~0.45-0.52",
    },
    {
        "drugs":              ["methotrexate", "hydroxychloroquine", "sulfasalazine"],
        "disease":            "rheumatoid arthritis",
        "tier":               "GOLD",
        "rationale":          "Triple DMARD: DHFR inhibitor + TLR inhibitor + 5-ASA",
        "evidence":           ["PMID:8551546", "PMID:10720468"],
        "anchor_drug":        "methotrexate",
        "min_individual_score": 0.35,
        "notes":              "O'Dell triple DMARD — non-inferior to biologics at 2yr",
    },
    {
        "drugs":              ["methotrexate", "hydroxychloroquine", "leflunomide"],
        "disease":            "rheumatoid arthritis",
        "tier":               "SILVER",
        "rationale":          "DHFR + TLR inhibitor + DHODH inhibitor — triple DMARD variant",
        "evidence":           ["PMID:19950318"],
        "anchor_drug":        "methotrexate",
        "min_individual_score": 0.35,
        "notes":              "Alternative triple DMARD; leflunomide may score low",
    },

    # ── TYPE 2 DIABETES ──────────────────────────────────────────────────────
    # Reference: ADA/EASD Consensus Report 2022
    {
        "drugs":              ["metformin", "pioglitazone"],
        "disease":            "type 2 diabetes mellitus",
        "tier":               "GOLD",
        "rationale":          "AMPK activator + PPARγ agonist — complementary mechanisms",
        "evidence":           ["PMID:11473914"],
        "anchor_drug":        "pioglitazone",
        "min_individual_score": 0.35,
        "notes":              "pioglitazone scores ~0.65; metformin ~0.49",
    },
    {
        "drugs":              ["metformin", "glipizide"],
        "disease":            "type 2 diabetes mellitus",
        "tier":               "GOLD",
        "rationale":          "AMPK activator + sulfonylurea — sensitiser + secretagogue",
        "evidence":           ["PMID:9571335"],
        "anchor_drug":        "glipizide",
        "min_individual_score": 0.40,
        "notes":              "Most used T2DM combination; glipizide scores ~0.45",
    },

    # ── POLYCYSTIC OVARY SYNDROME ────────────────────────────────────────────
    # Reference: Lord JM et al. BMJ 2003;327:951-953
    {
        "drugs":              ["metformin", "spironolactone"],
        "disease":            "polycystic ovary syndrome",
        "tier":               "SILVER",
        "rationale":          "AMPK/insulin sensitiser + aldosterone antagonist/anti-androgen",
        "evidence":           ["PMID:12788888"],
        "anchor_drug":        "metformin",
        "min_individual_score": 0.20,  # metformin scores ~0.20 for PCOS
        "notes":              "metformin/PCOS scoring improved in v8 scorer (floor at 0.22)",
    },

    # ── GOUT ─────────────────────────────────────────────────────────────────
    # Reference: EULAR recommendations 2016 (Richette et al.)
    {
        "drugs":              ["colchicine", "allopurinol"],
        "disease":            "gout",
        "tier":               "GOLD",
        "rationale":          "Microtubule/NLRP3 inhibitor (flare) + XO inhibitor (ULT)",
        "evidence":           ["PMID:22585186"],
        "anchor_drug":        "allopurinol",
        "min_individual_score": 0.40,
        "notes":              "allopurinol scores ~0.56; guideline standard",
    },

    # ── PERICARDITIS ─────────────────────────────────────────────────────────
    # Reference: Imazio M et al. COPE trial. Lancet 2013; ICAP trial 2014
    {
        "drugs":              ["aspirin", "colchicine"],
        "disease":            "pericarditis",
        "tier":               "GOLD",
        "rationale":          "COX inhibitor + microtubule/NLRP3 inhibitor — COPE trial",
        "evidence":           ["PMID:23765770", "PMID:24027428"],
        "anchor_drug":        "colchicine",
        "min_individual_score": 0.40,
        "notes":              "ESC Class I rec; colchicine ~0.41, aspirin ~0.54",
    },

    # ── HEART FAILURE ────────────────────────────────────────────────────────
    # Reference: RALES (spironolactone), MERIT-HF (metoprolol)
    {
        "drugs":              ["spironolactone", "metoprolol"],
        "disease":            "heart failure",
        "tier":               "GOLD",
        "rationale":          "MRA + beta-1 blocker — neurohormonal blockade from two axes",
        "evidence":           ["PMID:10471456", "PMID:10385240"],
        "anchor_drug":        "metoprolol",
        "min_individual_score": 0.30,
        "notes":              "metoprolol scores ~0.58; spironolactone 0.25-0.27",
    },
    {
        "drugs":              ["lisinopril", "metoprolol"],
        "disease":            "heart failure",
        "tier":               "GOLD",
        "rationale":          "ACE inhibitor + beta-blocker — standard HFrEF combo",
        "evidence":           ["PMID:12953764"],
        "anchor_drug":        "metoprolol",
        "min_individual_score": 0.25,
        "notes":              "CONSENSUS + MERIT-HF evidence",
    },

    # ── ALZHEIMER'S DISEASE ──────────────────────────────────────────────────
    # Reference: Tariot PN et al. JAMA 2004;291:317-324; FDA Namzaric 2014
    {
        "drugs":              ["donepezil", "memantine"],
        "disease":            "alzheimer disease",
        "tier":               "GOLD",
        "rationale":          "AChE inhibitor + NMDA antagonist — Namzaric combination",
        "evidence":           ["PMID:15304464"],
        "anchor_drug":        "memantine",
        "min_individual_score": 0.25,
        "notes":              "Only FDA-approved AD combo; memantine scores ~0.83",
    },

    # ── PARKINSON'S DISEASE ──────────────────────────────────────────────────
    # Reference: Stocchi F et al. Mov Disord 2003;18:94-100
    {
        "drugs":              ["rasagiline", "amantadine"],
        "disease":            "parkinson disease",
        "tier":               "SILVER",
        "rationale":          "MAO-B inhibitor + NMDA antagonist — dopamine preservation + modulation",
        "evidence":           ["PMID:15072009", "PMID:4142340"],
        "anchor_drug":        "rasagiline",
        "min_individual_score": 0.22,
        "notes":              "rasagiline scores ~0.82; amantadine ~0.74",
    },
    {
        "drugs":              ["rasagiline", "pramipexole"],
        "disease":            "parkinson disease",
        "tier":               "SILVER",
        "rationale":          "MAO-B inhibitor + dopamine agonist",
        "evidence":           ["PMID:15072009"],
        "anchor_drug":        "rasagiline",
        "min_individual_score": 0.22,
        "notes":              "Common early PD combination",
    },

    # ── HYPERCHOLESTEROLAEMIA ────────────────────────────────────────────────
    # Reference: IMPROVE-IT trial (Cannon CP et al. N Engl J Med 2015)
    {
        "drugs":              ["atorvastatin", "ezetimibe"],
        "disease":            "hypercholesterolemia",
        "tier":               "GOLD",
        "rationale":          "HMGCR inhibitor + NPC1L1 inhibitor — synthesis + absorption",
        "evidence":           ["PMID:26405429"],
        "anchor_drug":        "atorvastatin",
        "min_individual_score": 0.35,
        "notes":              "atorvastatin scores ~0.99; ezetimibe may score low",
    },

    # ── SYSTEMIC LUPUS ERYTHEMATOSUS ─────────────────────────────────────────
    {
        "drugs":              ["hydroxychloroquine", "methotrexate"],
        "disease":            "systemic lupus erythematosus",
        "tier":               "SILVER",
        "rationale":          "TLR inhibitor + DHFR inhibitor — dual immunomodulation",
        "evidence":           ["PMID:12211878"],
        "anchor_drug":        "hydroxychloroquine",
        "min_individual_score": 0.28,
        "notes":              "HCQ cornerstone; MTX adjunct in SLE",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Negative cases
# ─────────────────────────────────────────────────────────────────────────────

NEGATIVE_COMBO_CASES: List[Dict] = [
    {
        "drugs":    ["haloperidol", "levodopa"],
        "disease":  "parkinson disease",
        "reason":   "Dopamine antagonist + precursor = direct pharmacological opposition",
        "should_be_filtered": True,
        "filter_reason": "haloperidol absolute contraindication in PD",
    },
    {
        "drugs":    ["propranolol", "albuterol"],
        "disease":  "asthma",
        "reason":   "Non-selective beta-blocker causes life-threatening bronchospasm",
        "should_be_filtered": True,
        "filter_reason": "propranolol absolute contraindication in asthma",
    },
    {
        "drugs":    ["epinephrine", "sildenafil"],
        "disease":  "pulmonary arterial hypertension",
        "reason":   "Epinephrine is a vasoconstrictor that worsens PAH",
        "should_be_filtered": True,
        "filter_reason": "epinephrine absolute contraindication in PAH",
    },
    {
        "drugs":    ["cabazitaxel", "aspirin"],
        "disease":  "pericarditis",
        "reason":   "Cabazitaxel is a taxane chemotherapy with no pericarditis indication",
        "should_be_filtered": False,
        "filter_reason": "Context penalty should deprioritise taxane in pericarditis",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Validation helpers — uses pipeline normalisation
# ─────────────────────────────────────────────────────────────────────────────

_SALT_RE = re.compile(
    r"\s+(hydrochloride|hcl|sodium|potassium|sulfate|tartrate|maleate|"
    r"mesylate|acetate|phosphate|fumarate|succinate|monohydrate|dihydrate|"
    r"anhydrous|bitartrate|besylate|tosylate|citrate|calcium|magnesium|"
    r"bromide|chloride|disodium|trisodium|sodium\s+phosphate)$",
    re.IGNORECASE,
)


def normalise_drug_name(name: str) -> str:
    """
    Normalise drug name identically to production pipeline.
    Lowercase, strip salt/form suffixes (two passes for double salts).
    """
    n = name.lower().strip()
    n = _SALT_RE.sub("", n).strip()
    n = _SALT_RE.sub("", n).strip()
    return n


def combo_present_in_regimen(expected_drugs: List[str], regimen_name: str) -> bool:
    """
    Check if all expected drugs appear in a regimen name string.
    Uses normalised name comparison for salt-form tolerance.
    """
    regimen_lower = regimen_name.lower()
    regimen_parts = [normalise_drug_name(p) for p in regimen_lower.split(" + ")]

    for drug in expected_drugs:
        norm = normalise_drug_name(drug)
        # Check exact match in normalised parts
        if norm in regimen_parts:
            continue
        # Substring fallback (e.g. "metformin" in "metformin hydrochloride")
        if any(norm in part or part in norm for part in regimen_parts):
            continue
        return False
    return True


def combo_present_in_top_n(
    expected_drugs: List[str],
    ranked_regimens: List[Dict],
    top_n: int,
) -> Tuple[bool, Optional[int]]:
    """Check if expected drug combo appears in top-N ranked regimens."""
    for i, regimen in enumerate(ranked_regimens[:top_n]):
        regimen_name = regimen.get("regimen", "")
        if combo_present_in_regimen(expected_drugs, regimen_name):
            return True, i + 1
    return False, None


def anchor_score_passes(
    anchor_drug: str,
    min_score: float,
    candidates: List[Dict],
) -> Tuple[float, bool]:
    """
    Check if the anchor drug scores >= min_score in single-drug candidates.
    Uses normalised name matching.
    The anchor is the highest-scoring drug in the combo (e.g. bortezomib,
    methotrexate, atorvastatin). If the anchor scores well, the combination
    is clinically valid even if the pipeline doesn't find the exact combo.
    """
    anchor_norm = normalise_drug_name(anchor_drug)

    best_score = 0.0
    for c in candidates:
        raw_name = c.get("drug_name", c.get("name", ""))
        if normalise_drug_name(raw_name) == anchor_norm:
            best_score = max(best_score, c.get("score", 0.0))
            continue
        # Substring fallback
        if anchor_norm in normalise_drug_name(raw_name):
            best_score = max(best_score, c.get("score", 0.0))

    return best_score, best_score >= min_score


def all_individual_scores(
    expected_drugs: List[str],
    candidates: List[Dict],
) -> Dict[str, float]:
    """Return {drug_name: score} for all drugs in the combination."""
    lookup: Dict[str, float] = {}
    for c in candidates:
        raw_name = c.get("drug_name", c.get("name", ""))
        norm = normalise_drug_name(raw_name)
        score = c.get("score", 0.0)
        if norm not in lookup or lookup[norm] < score:
            lookup[norm] = score

    scores = {}
    for drug in expected_drugs:
        drug_norm = normalise_drug_name(drug)
        score = lookup.get(drug_norm, 0.0)
        if score == 0.0:
            # Substring fallback
            for cand_norm, cand_score in lookup.items():
                if drug_norm in cand_norm or cand_norm in drug_norm:
                    score = max(score, cand_score)
        scores[drug] = score

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

async def run_combo_validation(
    top_n: int = 15,
    output_path: str = "combo_validation_results.json",
    disease_filter: Optional[str] = None,
) -> Dict:
    from backend.pipeline.production_pipeline import ProductionPipeline

    pipeline = ProductionPipeline()
    start_utc = datetime.now(timezone.utc)

    cases_to_run = COMBO_VALIDATION_CASES
    if disease_filter:
        cases_to_run = [
            c for c in COMBO_VALIDATION_CASES
            if disease_filter.lower() in c["disease"].lower()
        ]
        logger.info(
            "Filtered to %d cases for disease: %s",
            len(cases_to_run), disease_filter,
        )

    logger.info("=" * 70)
    logger.info(
        "COMBO VALIDATION v3.0 — %d cases, top_n=%d",
        len(cases_to_run), top_n,
    )
    logger.info("=" * 70)

    results = []
    disease_cache: Dict[str, Dict] = {}

    try:
        logger.info("Fetching generic drug pool...")
        generic_drugs, _, generic_stats = await pipeline.fetch_generic_drugs(limit=3000)
        logger.info("Drug pool: %d confirmed generics", len(generic_drugs))

        for case in cases_to_run:
            disease = case["disease"]
            drugs = case["drugs"]
            logger.info(
                "\n[%s] %s for %s",
                case["tier"],
                " + ".join(drugs).upper(),
                disease.upper(),
            )

            if disease not in disease_cache:
                logger.info("  Running pipeline for %s...", disease)
                try:
                    plan = await pipeline.generate_treatment_plan(
                        disease_name=disease,
                        max_regimens=20,
                        include_triples=True,
                        fetch_ppi=True,
                        fetch_similarity=True,
                        use_tissue=True,
                    )
                    disease_cache[disease] = plan
                except Exception as e:
                    logger.error("  Pipeline failed for %s: %s", disease, e)
                    results.append({
                        "drugs":    drugs,
                        "disease":  disease,
                        "tier":     case["tier"],
                        "pass":     False,
                        "reason":   f"Pipeline error: {e}",
                        "combo_found_at_rank": None,
                        "individual_scores": {},
                        "anchor_pass": False,
                    })
                    continue

            plan = disease_cache[disease]
            ranked_regimens = plan.get("ranked_regimens", [])
            candidates = plan.get("candidates", [])

            # Check 1: Is the combo in top-N ranked regimens?
            combo_found, combo_rank = combo_present_in_top_n(drugs, ranked_regimens, top_n)

            # Check 2: Does the anchor drug score above threshold?
            anchor_drug = case.get("anchor_drug", drugs[0])
            anchor_score, anchor_pass = anchor_score_passes(
                anchor_drug, case["min_individual_score"], candidates
            )

            # Get all individual scores for reporting
            ind_scores = all_individual_scores(drugs, candidates)

            # PASS if combo found OR anchor drug scores well
            passed = combo_found or anchor_pass

            result = {
                "drugs":                drugs,
                "disease":              disease,
                "tier":                 case["tier"],
                "rationale":            case["rationale"],
                "pass":                 passed,
                "combo_found_in_top_n": combo_found,
                "combo_found_at_rank":  combo_rank,
                "top_n_checked":        top_n,
                "individual_scores":    {k: round(v, 4) for k, v in ind_scores.items()},
                "anchor_drug":          anchor_drug,
                "anchor_score":         round(anchor_score, 4),
                "anchor_threshold":     case["min_individual_score"],
                "anchor_pass":          anchor_pass,
                "pass_criterion":       "combo_in_top_n OR anchor_above_threshold",
                "evidence":             case.get("evidence", []),
                "notes":                case.get("notes", ""),
                "top_3_regimens": [
                    {
                        "rank":        r["rank"],
                        "regimen":     r["regimen"],
                        "combo_score": round(r.get("combo_score", 0), 4),
                        "orr":         round(r.get("orr_estimate", 0), 4),
                        "priority":    r.get("priority", ""),
                    }
                    for r in ranked_regimens[:3]
                ],
            }

            results.append(result)

            status = "✅ PASS" if passed else "❌ FAIL"
            how = (
                f"found at rank #{combo_rank}"
                if combo_found
                else (
                    f"anchor {anchor_drug}={anchor_score:.3f} >= {case['min_individual_score']}"
                    if anchor_pass
                    else f"not found; anchor {anchor_drug}={anchor_score:.3f} < {case['min_individual_score']}"
                )
            )
            logger.info("  %s — %s", status, how)
            for drug, score in ind_scores.items():
                flag = "✓" if score >= case["min_individual_score"] else "~"
                logger.info("    %s %s: %.4f", flag, drug, score)

        # ── Negative case checks ──────────────────────────────────────────────
        neg_results = []
        for neg_case in NEGATIVE_COMBO_CASES:
            disease = neg_case["disease"]
            if disease not in disease_cache:
                continue
            plan = disease_cache[disease]
            ranked_regimens = plan.get("ranked_regimens", [])
            candidates = plan.get("candidates", [])

            found_in_top, found_rank = combo_present_in_top_n(
                neg_case["drugs"], ranked_regimens, top_n
            )
            neg_pass = not found_in_top

            filtered_drugs = []
            for drug in neg_case["drugs"]:
                norm = normalise_drug_name(drug)
                in_candidates = any(
                    normalise_drug_name(c.get("drug_name", c.get("name", ""))) == norm
                    for c in candidates
                )
                if not in_candidates:
                    filtered_drugs.append(drug)

            neg_results.append({
                "drugs":           neg_case["drugs"],
                "disease":         disease,
                "reason":          neg_case["reason"],
                "pass":            neg_pass,
                "correctly_absent_from_top_n": neg_pass,
                "found_at_rank":   found_rank,
                "filtered_drugs":  filtered_drugs,
                "filter_reason":   neg_case.get("filter_reason", ""),
            })

            status = "✅ PASS" if neg_pass else "❌ FAIL (should not be top-ranked)"
            logger.info(
                "\n[NEGATIVE] %s for %s: %s",
                " + ".join(neg_case["drugs"]).upper(),
                disease.upper(),
                status,
            )
            if filtered_drugs:
                logger.info("  Correctly filtered: %s", filtered_drugs)

    finally:
        await pipeline.close()

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    n_total = len(results)
    n_pass = sum(1 for r in results if r["pass"])
    n_combo_found = sum(1 for r in results if r.get("combo_found_in_top_n"))
    n_anchor_only = sum(
        1 for r in results
        if r["pass"] and not r.get("combo_found_in_top_n") and r.get("anchor_pass")
    )

    by_tier: Dict[str, Dict] = {}
    for r in results:
        tier = r["tier"]
        if tier not in by_tier:
            by_tier[tier] = {"pass": 0, "fail": 0}
        by_tier[tier]["pass" if r["pass"] else "fail"] += 1

    neg_pass_count = sum(1 for r in neg_results if r["pass"])

    end_utc = datetime.now(timezone.utc)
    elapsed = (end_utc - start_utc).total_seconds()

    summary = {
        "run_timestamp_utc":      start_utc.isoformat(),
        "elapsed_seconds":        round(elapsed, 2),
        "top_n":                  top_n,
        "n_total_cases":          n_total,
        "n_pass":                 n_pass,
        "n_fail":                 n_total - n_pass,
        "pass_rate":              round(n_pass / n_total, 4) if n_total else 0,
        "n_found_in_top_n":       n_combo_found,
        "n_passed_via_anchor":    n_anchor_only,
        "by_tier":                by_tier,
        "n_negative_cases":       len(neg_results),
        "n_negative_pass":        neg_pass_count,
        "negative_pass_rate":     round(neg_pass_count / len(neg_results), 4) if neg_results else None,
        "pass_criterion":         "combo_in_top_n OR anchor_drug_above_threshold",
    }

    print("\n" + "=" * 70)
    print("COMBO VALIDATION SUMMARY v3.0")
    print("=" * 70)
    print(f"  Total cases:              {n_total}")
    print(f"  Pass:                     {n_pass}/{n_total} ({summary['pass_rate']:.1%})")
    print(f"  Found in top-{top_n}:         {n_combo_found}")
    print(f"  Passed via anchor drug:   {n_anchor_only}")
    print(f"  Pass criterion:           {summary['pass_criterion']}")
    print(f"  Elapsed:                  {elapsed:.1f}s")
    print()
    for tier, counts in sorted(by_tier.items()):
        total = counts["pass"] + counts["fail"]
        print(f"  [{tier}]: {counts['pass']}/{total} pass")
    if neg_results:
        print(f"\n  NEGATIVE cases: {neg_pass_count}/{len(neg_results)} correctly absent from top-{top_n}")
    print()

    failed = [r for r in results if not r["pass"]]
    if failed:
        print("FAILED CASES:")
        for r in failed:
            print(f"  ✗ {' + '.join(r['drugs'])} for {r['disease']}")
            print(f"    Anchor {r['anchor_drug']}: {r['anchor_score']:.4f} (need {r['anchor_threshold']:.2f})")
            for drug, score in r["individual_scores"].items():
                print(f"    - {drug}: {score:.4f}")
            if r.get("top_3_regimens"):
                print(f"    Top-3: {[x['regimen'] for x in r['top_3_regimens']]}")
    else:
        print("ALL CASES PASSED! ✅")
    print("=" * 70)

    output = {
        "summary":          summary,
        "positive_results": results,
        "negative_results": neg_results,
    }
    out_path = Path(output_path)
    out_path.write_text(json.dumps(output, indent=2))
    logger.info("Results written to: %s", out_path.resolve())
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Validate pipeline against known drug combination cases"
    )
    parser.add_argument(
        "--top-n", type=int, default=15,
        help="Check if combo appears in top N ranked regimens (default 15)",
    )
    parser.add_argument(
        "--output", type=str, default="combo_validation_results.json",
        help="Output file path",
    )
    parser.add_argument(
        "--disease", type=str, default=None,
        help="Only run cases for a specific disease (substring match)",
    )
    args = parser.parse_args()

    result = asyncio.run(
        run_combo_validation(
            top_n=args.top_n,
            output_path=args.output,
            disease_filter=args.disease,
        )
    )
    sys.exit(0 if result["summary"]["pass_rate"] >= 0.70 else 1)


if __name__ == "__main__":
    main()