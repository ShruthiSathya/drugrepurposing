#!/usr/bin/env python3
"""
rank_top_candidates.py
======================
Produces a ranked view of:
  1. Top diseases by opportunity score (from disease_ranker.py)
  2. Best known drug combinations per disease (from validation data)
  3. Single-drug scores for anchor drugs (from validation_results.json)
  4. A final priority matrix to identify the BEST disease to run a full
     in-depth pipeline on.

Usage
-----
    python rank_top_candidates.py                  # full report
    python rank_top_candidates.py --disease gout   # single disease detail
    python rank_top_candidates.py --json           # machine-readable output
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, ".")


# ─────────────────────────────────────────────────────────────────────────────
# Load existing run results (avoid re-running the expensive pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def load_validation_scores() -> Dict[str, float]:
    """Load drug scores from last validation run."""
    path = Path("validation_results.json")
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    scores: Dict[str, float] = {}
    for r in data.get("positive_results", []) + data.get("test_cases", []):
        key = f"{r['drug'].lower()}|{r['disease'].lower()}"
        scores[key] = r.get("raw_score", 0.0)
    return scores


def load_combo_results() -> Dict[str, dict]:
    """Load combo validation results from last run."""
    path = Path("combo_validation_results.json")
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    results: Dict[str, dict] = {}
    for r in data.get("positive_results", []):
        key = r["disease"].lower()
        if key not in results:
            results[key] = {"cases": [], "pass_count": 0, "total": 0}
        results[key]["cases"].append(r)
        results[key]["total"] += 1
        if r.get("pass"):
            results[key]["pass_count"] += 1
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Disease opportunity ranking (from disease_ranker.py)
# ─────────────────────────────────────────────────────────────────────────────

def get_disease_ranking() -> List[dict]:
    try:
        from backend.pipeline.disease_ranker import rank_diseases_by_opportunity
        return rank_diseases_by_opportunity()
    except ImportError:
        print("WARNING: Could not import disease_ranker. Using embedded data.")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Known best combos per disease (from combo_validation_dataset)
# ─────────────────────────────────────────────────────────────────────────────

def get_known_combos() -> Dict[str, List[dict]]:
    try:
        from combo_validation_dataset import COMBO_VALIDATION_CASES
        by_disease: Dict[str, List[dict]] = {}
        for c in COMBO_VALIDATION_CASES:
            key = c["disease"].lower()
            by_disease.setdefault(key, []).append(c)
        return by_disease
    except ImportError:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Composite priority score for pipeline targeting
# ─────────────────────────────────────────────────────────────────────────────

def compute_pipeline_priority(
    disease_row: dict,
    validation_scores: Dict[str, float],
    combo_results: Dict[str, dict],
    known_combos: Dict[str, List[dict]],
) -> dict:
    """
    Combines:
    - Opportunity score (market, feasibility, IP, PAG strength)
    - Pipeline validation accuracy (from validation_results.json)
    - Combo pass rate (from combo_validation_results.json)
    - Number of validated gold-standard combos available
    
    Returns a dict with priority score and breakdown.
    """
    disease = disease_row["disease"]
    key = disease.lower()

    # 1. Opportunity score (already 0-1)
    opp_score = disease_row["opportunity_score"]

    # 2. Pipeline accuracy for this disease (from known drug scores)
    known_drug_scores = []
    for drug in disease_row.get("known_generics", []):
        sk = f"{drug.lower()}|{key}"
        if sk in validation_scores:
            known_drug_scores.append(validation_scores[sk])
    pipeline_accuracy = (
        sum(1 for s in known_drug_scores if s >= 0.25) / len(known_drug_scores)
        if known_drug_scores else 0.5  # unknown → neutral
    )
    pipeline_avg_score = (
        sum(known_drug_scores) / len(known_drug_scores)
        if known_drug_scores else 0.0
    )

    # 3. Combo validation pass rate
    cr = combo_results.get(key, {})
    combo_pass_rate = (
        cr["pass_count"] / cr["total"] if cr.get("total", 0) > 0 else None
    )

    # 4. Number of gold-standard combos (GOLD tier)
    gold_combos = [
        c for c in known_combos.get(key, [])
        if c.get("tier") == "GOLD"
    ]

    # Composite score
    # Weights: opportunity 40%, pipeline accuracy 30%, combo pass 20%, gold count 10%
    combo_component = (combo_pass_rate * 0.20) if combo_pass_rate is not None else 0.10
    gold_component = min(len(gold_combos) / 5, 1.0) * 0.10

    priority = (
        opp_score * 0.40
        + pipeline_accuracy * 0.30
        + combo_component
        + gold_component
    )

    return {
        "disease": disease,
        "priority_score": round(priority, 4),
        "opportunity_score": opp_score,
        "pipeline_accuracy": round(pipeline_accuracy, 3),
        "pipeline_avg_drug_score": round(pipeline_avg_score, 3),
        "combo_pass_rate": round(combo_pass_rate, 3) if combo_pass_rate is not None else None,
        "n_gold_combos": len(gold_combos),
        "n_combo_cases": cr.get("total", 0),
        "known_drug_scores": {
            drug: round(validation_scores.get(f"{drug.lower()}|{key}", 0.0), 3)
            for drug in disease_row.get("known_generics", [])
        },
        "gold_combos": [" + ".join(c["drugs"]) for c in gold_combos],
        "tier": disease_row["tier"],
        "market_B": disease_row["market_size_B"],
        "feasibility": disease_row["feasibility"],
        "ip_safety": disease_row["ip_safety"],
        "orphan": disease_row.get("orphan_flag", False),
        "known_generics": disease_row.get("known_generics", []),
        "notes": disease_row.get("notes", ""),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────────────────────

TIER_COLORS = {"A": "🟢", "B": "🟡", "C": "🔴"}

def print_header(title: str) -> None:
    print("\n" + "═" * 72)
    print(f"  {title}")
    print("═" * 72)


def print_disease_card(p: dict, rank: int, verbose: bool = False) -> None:
    tier_icon = TIER_COLORS.get(p["tier"], "⚪")
    orphan = " 🏥ORPHAN" if p["orphan"] else ""
    print(f"\n#{rank:2d} {tier_icon}[{p['tier']}] {p['disease'].upper()}{orphan}")
    print(f"     Priority:   {p['priority_score']:.3f}  │  "
          f"Opportunity: {p['opportunity_score']:.3f}  │  "
          f"Market: ${p['market_B']}B")
    print(f"     Pipeline accuracy: {p['pipeline_accuracy']:.0%}  │  "
          f"Avg drug score: {p['pipeline_avg_drug_score']:.3f}  │  "
          f"Feasibility: {p['feasibility']:.2f}  │  IP safety: {p['ip_safety']:.2f}")

    if p["combo_pass_rate"] is not None:
        star = "★" if p["combo_pass_rate"] >= 0.80 else ("◑" if p["combo_pass_rate"] >= 0.50 else "○")
        print(f"     Combo validation: {star} {p['combo_pass_rate']:.0%} pass "
              f"({p['n_combo_cases']} cases, {p['n_gold_combos']} GOLD combos)")
    else:
        print("     Combo validation: ─ not yet run")

    if verbose:
        if p["gold_combos"]:
            print("     GOLD combos:")
            for combo in p["gold_combos"]:
                print(f"       • {combo}")
        if p["known_drug_scores"]:
            scored = {k: v for k, v in p["known_drug_scores"].items() if v > 0}
            if scored:
                drugs_str = "  ".join(f"{k}={v:.2f}" for k, v in sorted(
                    scored.items(), key=lambda x: -x[1])[:5])
                print(f"     Drug scores: {drugs_str}")
        if p["notes"]:
            print(f"     Note: {p['notes'][:100]}")


def print_matrix(ranked: List[dict]) -> None:
    """Compact comparison matrix."""
    print_header("PRIORITY MATRIX — all diseases ranked for pipeline targeting")
    print(f"{'#':>3}  {'Disease':<38}  {'Tier':>4}  {'Priority':>8}  "
          f"{'Opp':>6}  {'PipeAcc':>7}  {'Combo%':>6}  {'Market':>7}")
    print("─" * 85)
    for i, p in enumerate(ranked, 1):
        combo_str = (f"{p['combo_pass_rate']:.0%}" if p["combo_pass_rate"] is not None
                     else "  ─")
        tier_icon = TIER_COLORS.get(p["tier"], "⚪")
        print(f"{i:>3}.  {p['disease'][:38]:<38}  {tier_icon}{p['tier']:>3}  "
              f"{p['priority_score']:>8.3f}  "
              f"{p['opportunity_score']:>6.3f}  "
              f"{p['pipeline_accuracy']:>7.0%}  "
              f"{combo_str:>6}  "
              f"${p['market_B']:>5.1f}B")


def print_recommendation(ranked: List[dict]) -> None:
    print_header("RECOMMENDATION — Best disease for full pipeline run")
    top = ranked[0]
    tier_a = [p for p in ranked if p["tier"] == "A"]
    best_combo = max(ranked, key=lambda p: (p["combo_pass_rate"] or 0, p["priority_score"]))

    print(f"\n  🏆 TOP BY COMPOSITE PRIORITY SCORE: {top['disease'].upper()}")
    print(f"     Priority {top['priority_score']:.3f}  |  "
          f"${top['market_B']}B market  |  "
          f"{top['pipeline_accuracy']:.0%} pipeline accuracy")
    print(f"     GOLD combos: {', '.join(top['gold_combos'][:3]) or 'none yet validated'}")

    if best_combo["disease"] != top["disease"] and best_combo["combo_pass_rate"]:
        print(f"\n  🎯 TOP BY COMBO VALIDATION RATE: {best_combo['disease'].upper()}")
        print(f"     Combo pass rate: {best_combo['combo_pass_rate']:.0%}  "
              f"({best_combo['n_gold_combos']} GOLD combos found)")

    print(f"\n  📋 Tier A diseases ({len(tier_a)} total):")
    for p in tier_a[:6]:
        combo_str = f"combo={p['combo_pass_rate']:.0%}" if p["combo_pass_rate"] is not None else "no combo data"
        print(f"     • {p['disease']}: priority={p['priority_score']:.3f}, {combo_str}")

    print(f"\n  ⚡ Quick start command:")
    print(f"     python -c \"")
    print(f"     import asyncio")
    print(f"     from backend.pipeline.production_pipeline import ProductionPipeline")
    print(f"     async def run():")
    print(f"         p = ProductionPipeline()")
    print(f"         plan = await p.generate_treatment_plan('{top['disease']}', max_regimens=15)")
    print(f"         for r in plan['ranked_regimens'][:5]:")
    print(f"             print(r['rank'], r['regimen'], f\"ORR={{r['orr_estimate']:.1%}}\")")
    print(f"         await p.close()")
    print(f"     asyncio.run(run())\"")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Rank disease-drug combo candidates for TwinTrial pipeline"
    )
    parser.add_argument("--disease", type=str, default=None,
                        help="Show detailed view for one disease (substring match)")
    parser.add_argument("--json", action="store_true",
                        help="Output machine-readable JSON")
    parser.add_argument("--top", type=int, default=10,
                        help="How many top diseases to display in detail (default 10)")
    args = parser.parse_args()

    print("Loading data...", end=" ", flush=True)
    validation_scores = load_validation_scores()
    combo_results = load_combo_results()
    known_combos = get_known_combos()
    disease_ranking = get_disease_ranking()

    if not disease_ranking:
        print("ERROR: Could not load disease ranking. Ensure backend/ is in PYTHONPATH.")
        sys.exit(1)
    print(f"✓ ({len(disease_ranking)} diseases, "
          f"{len(validation_scores)} drug scores, "
          f"{len(combo_results)} disease combo sets)")

    # Compute priority for all diseases
    priorities = [
        compute_pipeline_priority(row, validation_scores, combo_results, known_combos)
        for row in disease_ranking
    ]
    priorities.sort(key=lambda p: p["priority_score"], reverse=True)

    # Single disease mode
    if args.disease:
        target = args.disease.lower()
        matches = [p for p in priorities if target in p["disease"].lower()]
        if not matches:
            print(f"No disease matching '{args.disease}'")
            sys.exit(1)
        print_header(f"DISEASE DETAIL: {matches[0]['disease'].upper()}")
        for m in matches:
            print_disease_card(m, priorities.index(m) + 1, verbose=True)
        return

    # JSON mode
    if args.json:
        print(json.dumps(priorities, indent=2))
        return

    # Full report
    print_header("TWINTRIAL ANALYTICS — Disease-Drug Combo Candidate Rankings")
    print(f"  Based on: opportunity scores, pipeline validation, combo evidence")
    print(f"  Validation data: {'✓ loaded' if validation_scores else '✗ run run_validation.py first'}")
    print(f"  Combo data:      {'✓ loaded' if combo_results else '✗ run combo_validation_dataset.py first'}")

    print_matrix(priorities)

    print_header(f"TOP {min(args.top, len(priorities))} DISEASES — DETAILED VIEW")
    for i, p in enumerate(priorities[:args.top], 1):
        print_disease_card(p, i, verbose=True)

    print_recommendation(priorities)

    print("\n")


if __name__ == "__main__":
    main()