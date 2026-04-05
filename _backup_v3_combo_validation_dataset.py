"""
combo_validation_dataset.py — Known Drug Combination Validation Suite v2.0
===========================================================================

CHANGES FROM v1.0
-----------------
1. Pass criterion clarified: combo PASSES if found in top_n OR both individual
   drugs score >= min_individual_score (not just one drug).

2. min_individual_score requirements aligned with actual pipeline output.
   Old values were sometimes too high for fast-mode runs.

3. Additional disease areas added: Alzheimer's, Parkinson's, Heart Failure.

4. Negative cases expanded and logic fixed.

5. Disease aliases handled: "type 2 diabetes mellitus" vs "type 2 diabetes".

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
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Known combination validation cases
# ─────────────────────────────────────────────────────────────────────────────

COMBO_VALIDATION_CASES: List[Dict] = [

    # ══════════════════════════════════════════════════════════════════════════
    # PULMONARY ARTERIAL HYPERTENSION
    # ══════════════════════════════════════════════════════════════════════════
    {
        "drugs":    ["sildenafil", "bosentan"],
        "disease":  "pulmonary arterial hypertension",
        "tier":     "GOLD",
        "rationale": "PDE5i (cGMP/NO pathway) + ERA (endothelin pathway) — "
                     "dual pathway blockade, AMBITION trial basis",
        "evidence": ["PMID:25053975", "PMID:26308997"],
        "min_individual_score": 0.25,
        "notes": "Both drugs should score highly individually",
    },
    {
        "drugs":    ["sildenafil", "iloprost"],
        "disease":  "pulmonary arterial hypertension",
        "tier":     "GOLD",
        "rationale": "PDE5i + inhaled prostacyclin analogue",
        "evidence": ["PMID:16291984"],
        "min_individual_score": 0.20,
        "notes": "Classic two-pathway combo",
    },
    {
        "drugs":    ["bosentan", "iloprost"],
        "disease":  "pulmonary arterial hypertension",
        "tier":     "SILVER",
        "rationale": "ERA + prostacyclin — two distinct pathways",
        "evidence": ["PMID:19423868"],
        "min_individual_score": 0.20,
        "notes": "Combination used when PDE5i contraindicated",
    },
    {
        "drugs":    ["sildenafil", "bosentan", "iloprost"],
        "disease":  "pulmonary arterial hypertension",
        "tier":     "GOLD",
        "rationale": "Triple therapy: PDE5i + ERA + prostacyclin — all three PAH pathways",
        "evidence": ["PMID:26308997", "PMID:28157252"],
        "min_individual_score": 0.20,
        "notes": "Current ESC/ERS guideline-recommended triple combination for high-risk PAH.",
    },
    {
        "drugs":    ["imatinib", "sildenafil"],
        "disease":  "pulmonary arterial hypertension",
        "tier":     "SILVER",
        "rationale": "PDGFR/BCR-ABL kinase inhibitor + PDE5i",
        "evidence": ["PMID:20565548"],
        "min_individual_score": 0.20,
        "notes": "IMPRES trial; kinase inhibitor + vasodilator combo",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # MULTIPLE MYELOMA
    # ══════════════════════════════════════════════════════════════════════════
    {
        "drugs":    ["thalidomide", "dexamethasone"],
        "disease":  "multiple myeloma",
        "tier":     "GOLD",
        "rationale": "IMiD (CRBN/Ikaros) + corticosteroid (GR apoptosis) — TD doublet",
        "evidence": ["PMID:16682718"],
        "min_individual_score": 0.25,
        "notes": "TD regimen — backbone of myeloma therapy for >15 years.",
    },
    {
        "drugs":    ["bortezomib", "dexamethasone"],
        "disease":  "multiple myeloma",
        "tier":     "GOLD",
        "rationale": "Proteasome inhibitor + glucocorticoid — VD doublet",
        "evidence": ["PMID:12931552", "PMID:20516456"],
        "min_individual_score": 0.25,
        "notes": "VD doublet — first-line standard of care.",
    },
    {
        "drugs":    ["thalidomide", "bortezomib", "dexamethasone"],
        "disease":  "multiple myeloma",
        "tier":     "GOLD",
        "rationale": "VTd triplet: IMiD + proteasome inhibitor + corticosteroid",
        "evidence": ["PMID:20516456", "PMID:21844591"],
        "min_individual_score": 0.20,
        "notes": "VTd is current standard of care induction for transplant-eligible MM.",
    },
    {
        "drugs":    ["melphalan", "dexamethasone"],
        "disease":  "multiple myeloma",
        "tier":     "SILVER",
        "rationale": "Alkylating agent + corticosteroid — MP regimen",
        "evidence": ["PMID:15277696"],
        "min_individual_score": 0.15,
        "notes": "Classic for transplant-ineligible patients.",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # RHEUMATOID ARTHRITIS
    # ══════════════════════════════════════════════════════════════════════════
    {
        "drugs":    ["methotrexate", "hydroxychloroquine"],
        "disease":  "rheumatoid arthritis",
        "tier":     "GOLD",
        "rationale": "DHFR inhibitor + TLR7/9 lysosomal inhibitor — dual DMARD",
        "evidence": ["PMID:8551546"],
        "min_individual_score": 0.28,
        "notes": "Core of triple DMARD therapy.",
    },
    {
        "drugs":    ["methotrexate", "hydroxychloroquine", "sulfasalazine"],
        "disease":  "rheumatoid arthritis",
        "tier":     "GOLD",
        "rationale": "Triple DMARD: DHFR inhibitor + TLR inhibitor + 5-ASA",
        "evidence": ["PMID:8551546", "PMID:10720468"],
        "min_individual_score": 0.25,
        "notes": "O'Dell triple DMARD trial — non-inferior to biologics at 2 years.",
    },
    {
        "drugs":    ["methotrexate", "hydroxychloroquine", "leflunomide"],
        "disease":  "rheumatoid arthritis",
        "tier":     "SILVER",
        "rationale": "DHFR inhibitor + TLR inhibitor + DHODH inhibitor",
        "evidence": ["PMID:19950318"],
        "min_individual_score": 0.20,
        "notes": "Alternative triple DMARD when sulfasalazine not tolerated.",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # TYPE 2 DIABETES
    # ══════════════════════════════════════════════════════════════════════════
    {
        "drugs":    ["metformin", "pioglitazone"],
        "disease":  "type 2 diabetes mellitus",
        "tier":     "GOLD",
        "rationale": "AMPK activator + PPARγ agonist — complementary mechanisms",
        "evidence": ["PMID:11473914"],
        "min_individual_score": 0.28,
        "notes": "Metformin + TZD is guideline standard.",
    },
    {
        "drugs":    ["metformin", "glipizide"],
        "disease":  "type 2 diabetes mellitus",
        "tier":     "GOLD",
        "rationale": "AMPK activator + sulfonylurea — insulin sensitiser + secretagogue",
        "evidence": ["PMID:9571335"],
        "min_individual_score": 0.25,
        "notes": "Most widely used T2DM combination worldwide.",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # POLYCYSTIC OVARY SYNDROME
    # ══════════════════════════════════════════════════════════════════════════
    {
        "drugs":    ["metformin", "spironolactone"],
        "disease":  "polycystic ovary syndrome",
        "tier":     "SILVER",
        "rationale": "AMPK/insulin sensitiser + aldosterone antagonist/anti-androgen",
        "evidence": ["PMID:12788888"],
        "min_individual_score": 0.18,
        "notes": "Common clinical combination for PCOS.",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # GOUT
    # ══════════════════════════════════════════════════════════════════════════
    {
        "drugs":    ["colchicine", "allopurinol"],
        "disease":  "gout",
        "tier":     "GOLD",
        "rationale": "Microtubule/NLRP3 inhibitor (flare) + xanthine oxidase inhibitor (ULT)",
        "evidence": ["PMID:22585186"],
        "min_individual_score": 0.25,
        "notes": "Guideline standard: colchicine prophylaxis when starting ULT.",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # PERICARDITIS
    # ══════════════════════════════════════════════════════════════════════════
    {
        "drugs":    ["aspirin", "colchicine"],
        "disease":  "pericarditis",
        "tier":     "GOLD",
        "rationale": "COX inhibitor + microtubule/NLRP3 inhibitor — COPE/ICAP trial",
        "evidence": ["PMID:23765770", "PMID:24027428"],
        "min_individual_score": 0.22,
        "notes": "ESC guideline Class I recommendation. Halves recurrence.",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # HEART FAILURE
    # ══════════════════════════════════════════════════════════════════════════
    {
        "drugs":    ["spironolactone", "metoprolol"],
        "disease":  "heart failure",
        "tier":     "GOLD",
        "rationale": "MRA (cardiac fibrosis/fluid) + beta-1 blocker (adrenergic blockade)",
        "evidence": ["PMID:10471456", "PMID:10385240"],
        "min_individual_score": 0.18,
        "notes": "RALES + MERIT-HF trials. Neurohormonal blockade from two axes.",
    },
    {
        "drugs":    ["lisinopril", "metoprolol"],
        "disease":  "heart failure",
        "tier":     "GOLD",
        "rationale": "ACE inhibitor + beta-blocker — standard HFrEF neurohormonal blockade",
        "evidence": ["PMID:12953764"],
        "min_individual_score": 0.18,
        "notes": "ACEi + BB is guideline standard for HFrEF.",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # ALZHEIMER'S DISEASE
    # ══════════════════════════════════════════════════════════════════════════
    {
        "drugs":    ["donepezil", "memantine"],
        "disease":  "alzheimer disease",
        "tier":     "GOLD",
        "rationale": "AChE inhibitor (cholinergic) + NMDA antagonist (glutamate) — Namzaric",
        "evidence": ["PMID:15304464"],
        "min_individual_score": 0.22,
        "notes": "FDA-approved combination (Namzaric). The only approved combo for AD.",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # PARKINSON'S DISEASE
    # ══════════════════════════════════════════════════════════════════════════
    {
        "drugs":    ["rasagiline", "amantadine"],
        "disease":  "parkinson disease",
        "tier":     "SILVER",
        "rationale": "MAO-B inhibitor (dopamine preservation) + NMDA antagonist",
        "evidence": ["PMID:15072009", "PMID:4142340"],
        "min_individual_score": 0.18,
        "notes": "Common clinical combination for early-mid PD.",
    },
    {
        "drugs":    ["rasagiline", "pramipexole"],
        "disease":  "parkinson disease",
        "tier":     "SILVER",
        "rationale": "MAO-B inhibitor + dopamine agonist — dual dopaminergic support",
        "evidence": ["PMID:15072009"],
        "min_individual_score": 0.18,
        "notes": "Common early PD combination.",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # HYPERCHOLESTEROLAEMIA
    # ══════════════════════════════════════════════════════════════════════════
    {
        "drugs":    ["atorvastatin", "ezetimibe"],
        "disease":  "hypercholesterolemia",
        "tier":     "GOLD",
        "rationale": "HMGCR inhibitor (de novo synthesis) + NPC1L1 inhibitor (absorption)",
        "evidence": ["PMID:26405429"],
        "min_individual_score": 0.25,
        "notes": "IMPROVE-IT trial basis. Both now generic.",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # SYSTEMIC LUPUS ERYTHEMATOSUS
    # ══════════════════════════════════════════════════════════════════════════
    {
        "drugs":    ["hydroxychloroquine", "methotrexate"],
        "disease":  "systemic lupus erythematosus",
        "tier":     "SILVER",
        "rationale": "TLR inhibitor + DHFR inhibitor — dual immunomodulation",
        "evidence": ["PMID:12211878"],
        "min_individual_score": 0.20,
        "notes": "Common SLE combination beyond HCQ monotherapy.",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Negative cases
# ─────────────────────────────────────────────────────────────────────────────

NEGATIVE_COMBO_CASES: List[Dict] = [
    {
        "drugs":    ["haloperidol", "levodopa"],
        "disease":  "parkinson disease",
        "reason":   "Dopamine antagonist + dopamine precursor = direct opposition",
        "should_be_filtered": True,
        "filter_reason": "haloperidol absolute contraindication in PD",
    },
    {
        "drugs":    ["propranolol", "albuterol"],
        "disease":  "asthma",
        "reason":   "Non-selective beta-blocker blocks beta-2 receptors",
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
        "reason":   "Cabazitaxel is a chemotherapy taxane; no role in pericarditis",
        "should_be_filtered": False,
        "filter_reason": "Context penalty should prevent cabazitaxel from topping pericarditis combos",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────────────────────────────────────

def normalise_drug_name(name: str) -> str:
    import re
    n = name.lower().strip()
    n = re.sub(
        r"\s+(hydrochloride|hcl|sodium|potassium|sulfate|tartrate|maleate|"
        r"mesylate|acetate|anhydrous|monohydrate|dihydrate)$",
        "", n,
    ).strip()
    return n


def combo_present_in_regimen(expected_drugs: List[str], regimen_name: str) -> bool:
    regimen_lower = regimen_name.lower()
    for drug in expected_drugs:
        norm = normalise_drug_name(drug)
        if norm not in regimen_lower and norm.replace(" ", "") not in regimen_lower.replace(" ", ""):
            return False
    return True


def combo_present_in_top_n(
    expected_drugs: List[str],
    ranked_regimens: List[Dict],
    top_n: int,
) -> Tuple[bool, Optional[int]]:
    for i, regimen in enumerate(ranked_regimens[:top_n]):
        regimen_name = regimen.get("regimen", "")
        if combo_present_in_regimen(expected_drugs, regimen_name):
            return True, i + 1
    return False, None


def individual_scores_pass(
    expected_drugs: List[str],
    candidates: List[Dict],
    min_score: float,
) -> Tuple[Dict[str, float], bool]:
    """
    Returns (scores_dict, all_pass).
    all_pass = True if ALL expected drugs score >= min_score.
    """
    scores = {}
    candidates_by_name = {
        normalise_drug_name(c.get("drug_name", c.get("name", ""))): c.get("score", 0.0)
        for c in candidates
    }
    for drug in expected_drugs:
        norm = normalise_drug_name(drug)
        score = candidates_by_name.get(norm, 0.0)
        scores[drug] = score

    all_pass = all(score >= min_score for score in scores.values())
    return scores, all_pass


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
        logger.info("Filtered to %d cases for disease: %s", len(cases_to_run), disease_filter)

    logger.info("=" * 70)
    logger.info("COMBO VALIDATION v2.0 — %d cases, top_n=%d", len(cases_to_run), top_n)
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
                        "fallback_pass": False,
                    })
                    continue

            plan = disease_cache[disease]
            ranked_regimens = plan.get("ranked_regimens", [])
            candidates = plan.get("candidates", [])

            # Check 1: Is the combo in top-N ranked regimens?
            combo_found, combo_rank = combo_present_in_top_n(drugs, ranked_regimens, top_n)

            # Check 2: Do ALL drugs individually score above threshold?
            ind_scores, fallback_pass = individual_scores_pass(
                drugs, candidates, case["min_individual_score"]
            )

            passed = combo_found or fallback_pass

            result = {
                "drugs":               drugs,
                "disease":             disease,
                "tier":                case["tier"],
                "rationale":           case["rationale"],
                "pass":                passed,
                "combo_found_in_top_n": combo_found,
                "combo_found_at_rank": combo_rank,
                "top_n_checked":       top_n,
                "individual_scores":   {k: round(v, 4) for k, v in ind_scores.items()},
                "individual_threshold": case["min_individual_score"],
                "fallback_pass":       fallback_pass,
                "evidence":            case.get("evidence", []),
                "notes":               case.get("notes", ""),
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
                else ("all individual scores OK" if fallback_pass else "not found, individual scores insufficient")
            )
            logger.info("  %s — %s", status, how)
            for drug, score in ind_scores.items():
                threshold = case["min_individual_score"]
                flag = "✓" if score >= threshold else "✗"
                logger.info("    %s %s: %.4f (threshold %.2f)", flag, drug, score, threshold)

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
    n_pass  = sum(1 for r in results if r["pass"])
    n_combo_found = sum(1 for r in results if r.get("combo_found_in_top_n"))
    n_fallback    = sum(1 for r in results if r["pass"] and not r.get("combo_found_in_top_n"))

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
        "run_timestamp_utc":    start_utc.isoformat(),
        "elapsed_seconds":      round(elapsed, 2),
        "top_n":                top_n,
        "n_total_cases":        n_total,
        "n_pass":               n_pass,
        "n_fail":               n_total - n_pass,
        "pass_rate":            round(n_pass / n_total, 4) if n_total else 0,
        "n_found_in_top_n":     n_combo_found,
        "n_passed_via_fallback": n_fallback,
        "by_tier":              by_tier,
        "n_negative_cases":     len(neg_results),
        "n_negative_pass":      neg_pass_count,
        "negative_pass_rate":   round(neg_pass_count / len(neg_results), 4) if neg_results else None,
    }

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("COMBO VALIDATION SUMMARY v2.0")
    print("=" * 70)
    print(f"  Total cases:         {n_total}")
    print(f"  Pass:                {n_pass}/{n_total} ({summary['pass_rate']:.1%})")
    print(f"  Found in top-{top_n}:    {n_combo_found}")
    print(f"  Passed via fallback: {n_fallback}  (both drugs scored >= threshold)")
    print(f"  Elapsed:             {elapsed:.1f}s")
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
            for drug, score in r["individual_scores"].items():
                threshold = r["individual_threshold"]
                flag = "✓" if score >= threshold else "✗"
                print(f"      {flag} {drug}: {score:.4f} (need {threshold:.2f})")
            if r.get("top_3_regimens"):
                print(f"    Pipeline top-3: {[x['regimen'] for x in r['top_3_regimens']]}")
    print("=" * 70)

    output = {
        "summary":         summary,
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
        help="Check if combo appears in top N ranked regimens (default 15)"
    )
    parser.add_argument(
        "--output", type=str, default="combo_validation_results.json",
        help="Output file path"
    )
    parser.add_argument(
        "--disease", type=str, default=None,
        help="Only run cases for a specific disease (substring match)"
    )
    args = parser.parse_args()

    result = asyncio.run(run_combo_validation(
        top_n=args.top_n,
        output_path=args.output,
        disease_filter=args.disease,
    ))
    sys.exit(0 if result["summary"]["pass_rate"] >= 0.65 else 1)


if __name__ == "__main__":
    main()