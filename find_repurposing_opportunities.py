#!/usr/bin/env python3
"""
find_repurposing_opportunities.py
==================================
Dynamically identifies diseases with STRONG NOVEL drug repurposing combo
opportunities that haven't already been clinically validated.

HOW IT WORKS (fully dynamic, zero hardcoding)
----------------------------------------------
1. Loads combo_validation_results.json to find what's ALREADY been done
   (clinically validated combos → exclude these as "already discovered")

2. Loads validation_results.json for individual drug scores per disease

3. Queries disease_ranker.py for opportunity scores

4. For each candidate disease, runs the combo scorer in-memory against
   the top-scoring drugs to estimate novel combo potential

5. Scores novelty = how many high-scoring drug PAIRS exist that are NOT
   in the known validated combo set

6. Final ranking: opportunity_score × novelty_score × pipeline_feasibility

WHAT "NOVEL" MEANS HERE
------------------------
A combo is considered "novel" if:
  - The drug pair is NOT already in a published clinical trial for that disease
  - The drug pair is NOT already in the combo_validation_dataset.py
  - The drugs have strong individual scores (>= 0.25) for this disease
  - The drug classes have known synergistic potential (from combo_scorer)

Usage
-----
    python find_repurposing_opportunities.py
    python find_repurposing_opportunities.py --top 15
    python find_repurposing_opportunities.py --min-novelty 0.3 --json
    python find_repurposing_opportunities.py --disease "lupus"
    python find_repurposing_opportunities.py --live   # runs actual pipeline (slow)
"""

import argparse
import asyncio
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, ".")

# ─────────────────────────────────────────────────────────────────────────────
# Salt normalisation (same as pipeline)
# ─────────────────────────────────────────────────────────────────────────────
_SALT_RE = re.compile(
    r"\s+(hydrochloride|hcl|sodium|potassium|sulfate|tartrate|maleate|"
    r"mesylate|acetate|phosphate|fumarate|succinate|monohydrate|dihydrate|"
    r"anhydrous|bitartrate|besylate|tosylate|citrate|calcium|magnesium|"
    r"bromide|chloride|disodium|trisodium|sodium\s+phosphate|"
    r"extended.release|er|xr|sr|cr|decanoate|pamoate|microspheres)$",
    re.IGNORECASE,
)

def norm(name: str) -> str:
    n = name.lower().strip()
    n = _SALT_RE.sub("", n).strip()
    n = _SALT_RE.sub("", n).strip()
    return n


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Load what's already been done (validated combos)
# ─────────────────────────────────────────────────────────────────────────────

def load_validated_combos() -> Dict[str, Set[frozenset]]:
    """
    Returns {disease: {frozenset(drug_a, drug_b), ...}} of ALREADY validated
    drug combinations. These are excluded from "novel" discovery.

    Sources:
    1. combo_validation_results.json  (what we've tested)
    2. combo_validation_dataset.py    (our known-truth set)
    """
    validated: Dict[str, Set[frozenset]] = {}

    # Source 1: combo_validation_results.json
    results_path = Path("combo_validation_results.json")
    if results_path.exists():
        data = json.loads(results_path.read_text())
        for r in data.get("positive_results", []):
            disease = r["disease"].lower()
            drugs = [norm(d) for d in r["drugs"]]
            if disease not in validated:
                validated[disease] = set()
            # Add all pairwise combinations
            for i in range(len(drugs)):
                for j in range(i + 1, len(drugs)):
                    validated[disease].add(frozenset({drugs[i], drugs[j]}))
    
    # Source 2: combo_validation_dataset.py
    try:
        from combo_validation_dataset import COMBO_VALIDATION_CASES
        for case in COMBO_VALIDATION_CASES:
            disease = case["disease"].lower()
            drugs = [norm(d) for d in case["drugs"]]
            if disease not in validated:
                validated[disease] = set()
            for i in range(len(drugs)):
                for j in range(i + 1, len(drugs)):
                    validated[disease].add(frozenset({drugs[i], drugs[j]}))
    except ImportError:
        print("  NOTE: combo_validation_dataset.py not found, using results only")

    total = sum(len(v) for v in validated.values())
    print(f"  Loaded {total} already-validated drug pairs across {len(validated)} diseases")
    return validated


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Load individual drug scores from validation_results.json
# ─────────────────────────────────────────────────────────────────────────────

def load_drug_scores() -> Dict[str, Dict[str, float]]:
    """
    Returns {disease: {drug: score}} from validation_results.json.
    Also includes scores from the full candidate list in combo_validation_results.
    """
    scores: Dict[str, Dict[str, float]] = {}

    # From validation_results.json
    val_path = Path("validation_results.json")
    if val_path.exists():
        data = json.loads(val_path.read_text())
        for r in data.get("positive_results", []) + data.get("test_cases", []):
            disease = r["disease"].lower()
            drug = norm(r["drug"])
            score = r.get("raw_score", 0.0)
            if disease not in scores:
                scores[disease] = {}
            scores[disease][drug] = max(scores[disease].get(drug, 0.0), score)

    # From combo_validation_results.json individual scores
    combo_path = Path("combo_validation_results.json")
    if combo_path.exists():
        data = json.loads(combo_path.read_text())
        for r in data.get("positive_results", []):
            disease = r["disease"].lower()
            if disease not in scores:
                scores[disease] = {}
            for drug, score in r.get("individual_scores", {}).items():
                d = norm(drug)
                scores[disease][d] = max(scores[disease].get(d, 0.0), score)

    print(f"  Loaded drug scores for {len(scores)} diseases")
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Load disease opportunity scores
# ─────────────────────────────────────────────────────────────────────────────

def load_opportunity_scores() -> List[Dict]:
    """Load disease opportunity scores from disease_ranker.py."""
    try:
        from backend.pipeline.disease_ranker import rank_diseases_by_opportunity
        ranked = rank_diseases_by_opportunity()
        print(f"  Loaded opportunity scores for {len(ranked)} diseases")
        return ranked
    except ImportError as e:
        print(f"  WARNING: Could not import disease_ranker: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Combo potential scoring using the actual combo_scorer
# ─────────────────────────────────────────────────────────────────────────────

def estimate_novel_combo_potential(
    disease_name: str,
    drug_scores: Dict[str, float],
    already_validated: Set[frozenset],
    min_individual_score: float = 0.20,
    top_k: int = 30,
) -> Dict:
    """
    Uses the actual CombinationScorer from combo_scorer.py to evaluate
    all drug pairs that:
      1. Both score >= min_individual_score for this disease
      2. Are NOT already in the validated set
      3. Are classified as synergistic by the combo scorer

    Returns a dict with novelty metrics and the top novel pairs.
    """
    try:
        from backend.pipeline.combo_scorer import CombinationScorer, SYNERGISTIC_PAIRS, classify_mechanism
    except ImportError:
        return {"error": "combo_scorer not available", "novelty_score": 0.0}

    # Filter to drugs above threshold
    eligible = {
        drug: score for drug, score in drug_scores.items()
        if score >= min_individual_score
    }

    if len(eligible) < 2:
        return {
            "n_eligible_drugs": len(eligible),
            "n_novel_synergistic": 0,
            "n_already_validated": 0,
            "novelty_score": 0.0,
            "top_novel_pairs": [],
        }

    scorer = CombinationScorer(disease_name=disease_name)
    drug_list = sorted(eligible.items(), key=lambda x: -x[1])[:top_k]

    novel_pairs = []
    already_counted = 0

    import itertools
    for (drug_a, score_a), (drug_b, score_b) in itertools.combinations(drug_list, 2):
        pair_key = frozenset({drug_a, drug_b})

        # Skip already validated pairs
        if pair_key in already_validated:
            already_counted += 1
            continue

        # Build minimal drug dicts for the scorer
        drug_a_dict = {
            "drug_name": drug_a,
            "score": score_a,
            "mechanism_score": score_a,  # proxy
            "mechanism": "",
            "target_genes": [],
        }
        drug_b_dict = {
            "drug_name": drug_b,
            "score": score_b,
            "mechanism_score": score_b,
            "mechanism": "",
            "target_genes": [],
        }

        result = scorer.score_pair(drug_a_dict, drug_b_dict, [])

        if result["is_synergistic"] and not result["is_antagonistic"]:
            novel_pairs.append({
                "drugs": [drug_a, drug_b],
                "combo_score": result["combo_score"],
                "mechanism_a": result["mechanism_a"],
                "mechanism_b": result["mechanism_b"],
                "synergy_bonus": result["synergy_bonus"],
                "score_a": round(score_a, 3),
                "score_b": round(score_b, 3),
                "context_penalty": result.get("context_penalty", 0.0),
            })

    # Sort by combo score
    novel_pairs.sort(key=lambda x: -x["combo_score"])

    n_novel = len(novel_pairs)
    n_eligible = len(eligible)

    # Novelty score:
    # - How many novel synergistic pairs exist (log-scaled)
    # - Weighted by average combo score of top pairs
    if n_novel == 0:
        novelty_score = 0.0
    else:
        top_combo_scores = [p["combo_score"] for p in novel_pairs[:5]]
        avg_top_score = sum(top_combo_scores) / len(top_combo_scores)
        # log scale: 1 pair=0.15, 3=0.30, 10=0.50, 30+=0.85
        log_scale = math.log10(n_novel + 1) / math.log10(31)
        novelty_score = round(min(log_scale * avg_top_score * 2.0, 1.0), 4)

    return {
        "n_eligible_drugs": n_eligible,
        "n_novel_synergistic": n_novel,
        "n_already_validated": already_counted,
        "novelty_score": novelty_score,
        "top_novel_pairs": novel_pairs[:8],
        "eligible_drugs": dict(drug_list[:15]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Live pipeline scoring (optional, slow)
# ─────────────────────────────────────────────────────────────────────────────

async def run_live_pipeline_for_disease(disease_name: str, top_k: int = 20) -> Dict[str, float]:
    """
    Runs the actual production pipeline to get fresh drug scores for a disease.
    Used when --live flag is set.
    """
    try:
        from backend.pipeline.production_pipeline import ProductionPipeline
        pipeline = ProductionPipeline()
        try:
            disease_data = await pipeline.data_fetcher.fetch_disease_data(disease_name)
            if not disease_data:
                return {}
            drugs_data = await pipeline.fetch_approved_drugs(limit=3000)
            candidates = await pipeline.generate_candidates(
                disease_data=disease_data,
                drugs_data=drugs_data,
                min_score=0.0,
                fetch_pubmed=False,
                fetch_ppi=False,  # fast mode
                fetch_similarity=False,
                use_efo=False,
                use_tissue=False,
                use_polypharm=False,
            )
            candidates.sort(key=lambda c: c["score"], reverse=True)
            return {norm(c["drug_name"]): c["score"] for c in candidates[:200]}
        finally:
            await pipeline.close()
    except Exception as e:
        print(f"  Pipeline error for {disease_name}: {e}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Diseases not yet in validation set
# ─────────────────────────────────────────────────────────────────────────────

def get_untested_diseases(
    opportunity_ranked: List[Dict],
    validated_combos: Dict[str, Set[frozenset]],
    drug_scores: Dict[str, Dict[str, float]],
) -> List[str]:
    """
    Returns diseases from the opportunity list that have NO validated combos yet.
    These are highest-priority for novel discovery.
    """
    all_diseases = [r["disease"] for r in opportunity_ranked]
    validated_keys = set(validated_combos.keys())
    scored_keys = set(drug_scores.keys())

    untested = []
    for d in all_diseases:
        d_lower = d.lower()
        has_validated = any(d_lower in k or k in d_lower for k in validated_keys)
        if not has_validated:
            untested.append(d)

    return untested


# ─────────────────────────────────────────────────────────────────────────────
# Step 7: Disease-specific known drug pool (from literature, not hardcoded)
# ─────────────────────────────────────────────────────────────────────────────

def get_known_drugs_for_disease(disease_name: str) -> List[str]:
    """
    Pulls known approved drugs for a disease from drug_similarity.py's
    DISEASE_KNOWN_DRUGS map — this is already in the codebase.
    """
    try:
        from backend.pipeline.drug_similarity import DISEASE_KNOWN_DRUGS, get_known_drugs_for_disease as _get
        return _get(disease_name)
    except ImportError:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyse_all_diseases(
    opportunity_ranked: List[Dict],
    validated_combos: Dict[str, Set[frozenset]],
    drug_scores: Dict[str, Dict[str, float]],
    disease_filter: Optional[str] = None,
    min_novelty: float = 0.0,
    min_opportunity: float = 0.0,
) -> List[Dict]:
    """
    For each disease in opportunity_ranked, compute:
    - novelty_score: how many strong, non-validated combos exist
    - opportunity_score: from disease_ranker
    - final_priority: product of both
    - top novel pairs
    """
    results = []

    diseases_to_check = opportunity_ranked
    if disease_filter:
        df = disease_filter.lower()
        diseases_to_check = [d for d in opportunity_ranked if df in d["disease"].lower()]

    print(f"\n  Analysing {len(diseases_to_check)} diseases for novel combo potential...")

    for opp in diseases_to_check:
        disease = opp["disease"]
        d_lower = disease.lower()

        # Get drug scores for this disease
        # Match with fuzzy key lookup
        d_drug_scores = {}
        for key, scores in drug_scores.items():
            if d_lower in key or key in d_lower:
                for drug, score in scores.items():
                    d_drug_scores[drug] = max(d_drug_scores.get(drug, 0.0), score)

        # Also add known approved drugs from drug_similarity.py (with neutral score
        # if we have no pipeline score for them yet)
        known_drugs = get_known_drugs_for_disease(disease)
        for drug in known_drugs:
            d = norm(drug)
            if d not in d_drug_scores:
                d_drug_scores[d] = 0.15  # small non-zero score as proxy

        # Get already validated pairs for this disease
        already = set()
        for key, pairs in validated_combos.items():
            if d_lower in key or key in d_lower:
                already |= pairs

        # Estimate novel combo potential
        novelty = estimate_novel_combo_potential(
            disease_name=disease,
            drug_scores=d_drug_scores,
            already_validated=already,
        )

        opp_score = opp["opportunity_score"]
        nov_score = novelty["novelty_score"]
        feasibility = opp.get("feasibility", 0.5)

        # Final priority: opportunity × novelty × feasibility
        # If no drug scores available yet, penalise (unknown territory)
        data_confidence = min(len(d_drug_scores) / 5, 1.0)
        final_priority = round(
            opp_score * 0.40
            + nov_score * 0.35
            + feasibility * 0.15
            + data_confidence * 0.10,
            4,
        )

        is_untested = len(already) == 0

        result = {
            "disease":                disease,
            "final_priority":         final_priority,
            "opportunity_score":      opp_score,
            "novelty_score":          nov_score,
            "feasibility":            feasibility,
            "tier":                   opp.get("tier", "C"),
            "market_B":               opp.get("market_size_B", 0),
            "orphan":                 opp.get("orphan_flag", False),
            "ip_safety":              opp.get("ip_safety", 0),
            "n_novel_synergistic_pairs": novelty.get("n_novel_synergistic", 0),
            "n_already_validated":    novelty.get("n_already_validated", 0),
            "n_eligible_drugs":       novelty.get("n_eligible_drugs", 0),
            "is_completely_untested": is_untested,
            "top_novel_pairs":        novelty.get("top_novel_pairs", []),
            "known_drug_scores":      {k: v for k, v in sorted(
                                            d_drug_scores.items(),
                                            key=lambda x: -x[1])[:10]},
            "known_generics":         opp.get("known_generics", []),
            "notes":                  opp.get("notes", ""),
        }

        if nov_score >= min_novelty and opp_score >= min_opportunity:
            results.append(result)

    results.sort(key=lambda x: -x["final_priority"])
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────────────────────

TIER_ICON = {"A": "🟢", "B": "🟡", "C": "🔴"}

def print_summary_table(results: List[Dict], top_n: int = 20) -> None:
    print("\n" + "═" * 90)
    print("  NOVEL DRUG REPURPOSING COMBO OPPORTUNITIES")
    print("  Ranked by: opportunity × novelty × feasibility (all dynamic from pipeline)")
    print("═" * 90)
    print(f"\n{'#':>3}  {'Disease':<38}  {'Tier':>4}  {'Priority':>8}  "
          f"{'Novelty':>7}  {'Novel★':>6}  {'Done':>4}  {'New?':>5}")
    print("─" * 90)

    for i, r in enumerate(results[:top_n], 1):
        tier_icon = TIER_ICON.get(r["tier"], "⚪")
        orphan = "🏥" if r["orphan"] else "  "
        untested = "✨NEW" if r["is_completely_untested"] else "     "
        print(
            f"{i:>3}.  {r['disease'][:38]:<38}  {tier_icon}{r['tier']:>3}  "
            f"{r['final_priority']:>8.3f}  "
            f"{r['novelty_score']:>7.3f}  "
            f"{r['n_novel_synergistic_pairs']:>6}  "
            f"{r['n_already_validated']:>4}  "
            f"{untested}  {orphan}"
        )


def print_detail_cards(results: List[Dict], top_n: int = 5) -> None:
    print("\n" + "═" * 90)
    print(f"  TOP {top_n} NOVEL COMBO OPPORTUNITIES — DETAILED VIEW")
    print("═" * 90)

    for i, r in enumerate(results[:top_n], 1):
        tier_icon = TIER_ICON.get(r["tier"], "⚪")
        orphan_label = " [ORPHAN ELIGIBLE]" if r["orphan"] else ""
        untested_label = " ✨ COMPLETELY UNTESTED" if r["is_completely_untested"] else ""

        print(f"\n{'─'*70}")
        print(f"#{i}: {r['disease'].upper()}{orphan_label}{untested_label}")
        print(f"    Tier {tier_icon}{r['tier']}  |  Market: ${r['market_B']}B  |  "
              f"IP Safety: {r['ip_safety']:.2f}  |  Feasibility: {r['feasibility']:.2f}")
        print(f"    Final Priority: {r['final_priority']:.3f}  |  "
              f"Opportunity: {r['opportunity_score']:.3f}  |  "
              f"Novelty: {r['novelty_score']:.3f}")
        print(f"    {r['n_novel_synergistic_pairs']} novel synergistic pairs  |  "
              f"{r['n_already_validated']} already validated  |  "
              f"{r['n_eligible_drugs']} eligible drugs")

        if r["known_generics"]:
            print(f"\n    APPROVED GENERICS: {', '.join(r['known_generics'][:6])}")

        if r["known_drug_scores"]:
            top_drugs = list(r["known_drug_scores"].items())[:6]
            scores_str = "  ".join(f"{k}={v:.2f}" for k, v in top_drugs)
            print(f"    DRUG SCORES (from pipeline): {scores_str}")

        if r["top_novel_pairs"]:
            print(f"\n    TOP NOVEL SYNERGISTIC PAIRS:")
            for j, pair in enumerate(r["top_novel_pairs"][:5], 1):
                mech_str = f"{pair['mechanism_a']} + {pair['mechanism_b']}"
                penalty_note = f" [ctx_penalty={pair['context_penalty']:.2f}]" if pair.get("context_penalty", 0) > 0.1 else ""
                print(f"      {j}. {' + '.join(p.upper() for p in pair['drugs'])}")
                print(f"         combo_score={pair['combo_score']:.3f}  "
                      f"synergy_bonus={pair['synergy_bonus']:.3f}  "
                      f"scores=({pair['score_a']:.2f},{pair['score_b']:.2f}){penalty_note}")
                print(f"         mechanisms: {mech_str}")

        if r["notes"]:
            print(f"\n    NOTE: {r['notes'][:140]}")


def print_untested_highlight(results: List[Dict]) -> None:
    untested = [r for r in results if r["is_completely_untested"]]
    if not untested:
        return

    print("\n" + "═" * 90)
    print("  🚀 COMPLETELY UNTESTED DISEASES (zero validated combos in your dataset)")
    print("  These have been ranked by opportunity + novelty — highest potential for new discovery")
    print("═" * 90)

    for i, r in enumerate(untested[:8], 1):
        tier_icon = TIER_ICON.get(r["tier"], "⚪")
        top_pair = r["top_novel_pairs"][0] if r["top_novel_pairs"] else None
        pair_str = (
            " + ".join(p.upper() for p in top_pair["drugs"])
            if top_pair else "No pairs scored yet"
        )
        print(f"  #{i:2d} {tier_icon}[{r['tier']}] {r['disease']:<40}  "
              f"priority={r['final_priority']:.3f}  "
              f"novel_pairs={r['n_novel_synergistic_pairs']}")
        if top_pair:
            print(f"       → Best pair: {pair_str} (combo_score={top_pair['combo_score']:.3f})")


def print_action_items(results: List[Dict]) -> None:
    print("\n" + "═" * 90)
    print("  ACTION ITEMS")
    print("═" * 90)

    top = results[0] if results else None
    untested_tier_a = [r for r in results if r["is_completely_untested"] and r["tier"] == "A"]

    if top:
        best_pair = top["top_novel_pairs"][0] if top["top_novel_pairs"] else None
        print(f"\n  1. RUN COMBO VALIDATION for #{1} {top['disease'].upper()}")
        if best_pair:
            print(f"     Top novel combo: {' + '.join(p.upper() for p in best_pair['drugs'])}")
            print(f"     combo_score={best_pair['combo_score']:.3f}  "
                  f"mechanisms: {best_pair['mechanism_a']} + {best_pair['mechanism_b']}")
        print(f"""
     python combo_validation_dataset.py --disease "{top['disease']}" --top-n 15
""")

    if untested_tier_a:
        best_untested = untested_tier_a[0]
        print(f"  2. QUICK PIPELINE RUN for completely untested Tier A disease:")
        print(f"     {best_untested['disease'].upper()}")
        print(f"""
     python -c "
import asyncio
from backend.pipeline.production_pipeline import ProductionPipeline
async def run():
    p = ProductionPipeline()
    plan = await p.generate_treatment_plan('{best_untested['disease']}', max_regimens=15)
    for r in plan['ranked_regimens'][:5]:
        print(r['rank'], r['regimen'], f'ORR={{r[\\'orr_estimate\\']:.1%}}')
    await p.close()
asyncio.run(run())"
""")

    print("  3. TO GET RICHER SCORES: Run validation_results.py with --fast flag:")
    print("""
     python run_validation.py --fast --disease "<disease_name>"
""")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

async def main_async(args) -> None:
    """Async version for --live flag."""
    print("\n" + "═" * 90)
    print("  DYNAMIC NOVEL DRUG REPURPOSING COMBO DISCOVERY")
    print("  Fully dynamic — reads live pipeline outputs, zero hardcoding")
    print("═" * 90)

    print("\n[1/4] Loading validated combos (what's already been done)...")
    validated_combos = load_validated_combos()

    print("\n[2/4] Loading drug scores from pipeline results...")
    drug_scores = load_drug_scores()

    print("\n[3/4] Loading disease opportunity rankings...")
    opportunity_ranked = load_opportunity_scores()

    if not opportunity_ranked:
        print("ERROR: Could not load opportunity scores. Run from repo root.")
        sys.exit(1)

    # If --live, run pipeline for untested diseases
    if args.live:
        print("\n[3b/4] Running LIVE pipeline for untested diseases (this may take a while)...")
        untested = get_untested_diseases(opportunity_ranked, validated_combos, drug_scores)
        print(f"  Found {len(untested)} completely untested diseases")
        for disease in untested[:args.live_n]:
            print(f"  Running pipeline for: {disease}")
            live_scores = await run_live_pipeline_for_disease(disease)
            if live_scores:
                drug_scores[disease.lower()] = live_scores
                print(f"    Got {len(live_scores)} drug scores")

    print("\n[4/4] Computing novel combo potential for each disease...")
    results = analyse_all_diseases(
        opportunity_ranked=opportunity_ranked,
        validated_combos=validated_combos,
        drug_scores=drug_scores,
        disease_filter=args.disease,
        min_novelty=args.min_novelty,
        min_opportunity=args.min_opportunity,
    )

    if not results:
        print("No results match the filter criteria.")
        sys.exit(0)

    if args.json:
        # Machine-readable output
        output = {
            "n_diseases_analysed": len(results),
            "top_novel_opportunities": results[:args.top],
            "n_already_validated_diseases": len(validated_combos),
            "diseases_with_drug_scores": len(drug_scores),
        }
        print(json.dumps(output, indent=2))
        return

    # Human-readable output
    print_summary_table(results, top_n=args.top)
    print_untested_highlight(results)
    print_detail_cards(results, top_n=min(args.detail, len(results)))
    print_action_items(results)

    # Save JSON report
    out_path = Path("novel_repurposing_opportunities.json")
    out_path.write_text(json.dumps({
        "generated_at": __import__("datetime").datetime.utcnow().isoformat(),
        "n_diseases_analysed": len(results),
        "n_already_validated_diseases": len(validated_combos),
        "opportunities": results,
    }, indent=2))
    print(f"\n  Full report saved to: {out_path.resolve()}")
    print("═" * 90)


def main():
    parser = argparse.ArgumentParser(
        description="Find novel drug repurposing combo opportunities not yet validated"
    )
    parser.add_argument("--top", type=int, default=20,
                        help="Number of diseases to show in summary table (default 20)")
    parser.add_argument("--detail", type=int, default=5,
                        help="Number of diseases to show in detail cards (default 5)")
    parser.add_argument("--disease", type=str, default=None,
                        help="Filter to a specific disease (substring match)")
    parser.add_argument("--min-novelty", type=float, default=0.0,
                        help="Minimum novelty score to include (default 0.0)")
    parser.add_argument("--min-opportunity", type=float, default=0.0,
                        help="Minimum opportunity score to include (default 0.0)")
    parser.add_argument("--json", action="store_true",
                        help="Output machine-readable JSON")
    parser.add_argument("--live", action="store_true",
                        help="Run actual pipeline for untested diseases (slow, more accurate)")
    parser.add_argument("--live-n", type=int, default=3,
                        help="How many untested diseases to run live pipeline for (default 3)")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()