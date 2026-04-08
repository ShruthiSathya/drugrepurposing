#!/usr/bin/env python3
"""
find_novel_repurposing.py
==========================
Discovers GENUINELY NOVEL drug repurposing combo opportunities.

KEY DIFFERENCE FROM find_repurposing_opportunities.py
------------------------------------------------------
The old script used "anchor_score_passes" as a pass criterion — meaning it
flagged diseases where a drug ALREADY scores high for that disease. That's
not discovery; that's confirming known uses.

THIS SCRIPT:
1. Looks at drug scores from validation_results.json
2. For each drug, finds diseases where it scores MODERATELY (0.20–0.55)
   — too low to be a known treatment, but high enough to suggest biological
   relevance via gene/pathway overlap
3. Finds drug PAIRS where both drugs score in this "off-label signal" range
   for the SAME disease, AND the combo_scorer calls them synergistic
4. Excludes combos already in the validated set (known uses)
5. Ranks by: synergy_score × avg_signal × opportunity_score

WHY 0.20–0.55?
--------------
- < 0.20: likely noise (PPI inflation, no real mechanism)
- 0.20–0.55: off-label biological signal — mechanistic connection exists
  but drug not indicated for that disease
- > 0.55: likely already used or well-known for that disease (exclude!)

Usage
-----
    python find_novel_repurposing.py
    python find_novel_repurposing.py --min-signal 0.22 --max-signal 0.60
    python find_novel_repurposing.py --disease "epilepsy"
    python find_novel_repurposing.py --json
"""

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

sys.path.insert(0, ".")

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
# Step 1: Load ALL drug-disease scores from validation run
# We want the full matrix, not just TP/TN cases
# ─────────────────────────────────────────────────────────────────────────────

def load_all_drug_disease_scores() -> Dict[str, Dict[str, float]]:
    """
    Returns {disease: {drug: score}} from validation_results.json.
    
    Critically: we load ALL cases, including the ones where drugs scored
    moderately — these are the off-label signal we're hunting.
    """
    scores: Dict[str, Dict[str, float]] = {}
    
    val_path = Path("validation_results.json")
    if not val_path.exists():
        print("  WARNING: validation_results.json not found")
        return scores
    
    data = json.loads(val_path.read_text())
    
    # Load from positive AND negative results — we want all scores
    all_results = data.get("positive_results", []) + data.get("negative_results", [])
    for r in all_results:
        disease = r["disease"].lower()
        drug = norm(r["drug"])
        score = r.get("raw_score", 0.0)
        if disease not in scores:
            scores[disease] = {}
        scores[disease][drug] = max(scores[disease].get(drug, 0.0), score)
    
    # Also pull from combo validation individual scores
    combo_path = Path("combo_validation_results.json")
    if combo_path.exists():
        combo_data = json.loads(combo_path.read_text())
        for r in combo_data.get("positive_results", []):
            disease = r["disease"].lower()
            if disease not in scores:
                scores[disease] = {}
            for drug_name, drug_score in r.get("individual_scores", {}).items():
                d = norm(drug_name)
                scores[disease][d] = max(scores[disease].get(d, 0.0), drug_score)
    
    print(f"  Loaded drug-disease scores: {sum(len(v) for v in scores.values())} "
          f"pairs across {len(scores)} diseases")
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Load known/validated drug-disease pairs to EXCLUDE
# These are drugs that are ALREADY used for the disease
# ─────────────────────────────────────────────────────────────────────────────

def load_known_indications() -> Dict[str, Set[str]]:
    """
    Returns {disease: {drug, drug, ...}} of KNOWN drug-disease pairs.
    A drug in this set for a disease is ALREADY indicated → not novel.
    
    Sources:
    - validation_dataset.py TRUE_POSITIVE cases (anchor drugs)
    - disease_ranker.py known_generics
    - drug_similarity.py DISEASE_KNOWN_DRUGS
    """
    known: Dict[str, Set[str]] = defaultdict(set)
    
    # From validation dataset TRUE_POSITIVE cases
    try:
        from backend.pipeline.validation_dataset import get_positive_cases
        for case in get_positive_cases():
            disease = case["disease"].lower()
            known[disease].add(norm(case["drug"]))
    except ImportError:
        pass
    
    # From drug_similarity DISEASE_KNOWN_DRUGS
    try:
        from backend.pipeline.drug_similarity import DISEASE_KNOWN_DRUGS
        for disease, drugs in DISEASE_KNOWN_DRUGS.items():
            for drug in drugs:
                known[disease.lower()].add(norm(drug))
    except ImportError:
        pass
    
    # From disease_ranker known_generics
    try:
        from backend.pipeline.disease_ranker import DISEASE_OPPORTUNITY_DB
        for disease, data in DISEASE_OPPORTUNITY_DB.items():
            for drug in data.get("known_generics", []):
                known[disease.lower()].add(norm(drug))
    except ImportError:
        pass
    
    total = sum(len(v) for v in known.values())
    print(f"  Loaded {total} known drug-disease indications across {len(known)} diseases")
    return known


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Load already-validated combo pairs to EXCLUDE
# ─────────────────────────────────────────────────────────────────────────────

def load_validated_combos() -> Dict[str, Set[FrozenSet]]:
    """Already clinically validated combos — exclude from novel candidates."""
    validated: Dict[str, Set[FrozenSet]] = defaultdict(set)
    
    results_path = Path("combo_validation_results.json")
    if results_path.exists():
        data = json.loads(results_path.read_text())
        for r in data.get("positive_results", []):
            disease = r["disease"].lower()
            drugs = [norm(d) for d in r["drugs"]]
            for i in range(len(drugs)):
                for j in range(i + 1, len(drugs)):
                    validated[disease].add(frozenset({drugs[i], drugs[j]}))
    
    try:
        from combo_validation_dataset import COMBO_VALIDATION_CASES
        for case in COMBO_VALIDATION_CASES:
            disease = case["disease"].lower()
            drugs = [norm(d) for d in case["drugs"]]
            for i in range(len(drugs)):
                for j in range(i + 1, len(drugs)):
                    validated[disease].add(frozenset({drugs[i], drugs[j]}))
    except ImportError:
        pass
    
    total = sum(len(v) for v in validated.values())
    print(f"  Loaded {total} already-validated combos to exclude")
    return validated


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Load opportunity scores
# ─────────────────────────────────────────────────────────────────────────────

def load_opportunity_scores() -> Dict[str, Dict]:
    try:
        from backend.pipeline.disease_ranker import rank_diseases_by_opportunity
        ranked = rank_diseases_by_opportunity()
        return {r["disease"].lower(): r for r in ranked}
    except ImportError:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: For each disease, find OFF-LABEL drug signal
# Off-label = scores in [min_signal, max_signal] AND not a known indication
# ─────────────────────────────────────────────────────────────────────────────

def find_off_label_signals(
    disease: str,
    drug_scores: Dict[str, float],
    known_for_disease: Set[str],
    min_signal: float = 0.20,
    max_signal: float = 0.55,
) -> Dict[str, float]:
    """
    Returns drugs with off-label signal for this disease:
    - Score in [min_signal, max_signal] (moderate — not known treatment)
    - NOT a known indication for this disease
    """
    off_label = {}
    for drug, score in drug_scores.items():
        if score < min_signal:
            continue  # too weak — likely noise
        if score > max_signal:
            continue  # too strong — likely already used for this disease
        if drug in known_for_disease:
            continue  # known indication — not novel
        # Additional check: if drug name appears in known list with minor variation
        is_known = any(
            norm(drug) == norm(k) or 
            norm(drug) in norm(k) or 
            norm(k) in norm(drug)
            for k in known_for_disease
        )
        if is_known:
            continue
        off_label[drug] = score
    return off_label


# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Score novel combo pairs using combo_scorer
# ─────────────────────────────────────────────────────────────────────────────

def find_novel_synergistic_pairs(
    disease_name: str,
    off_label_drugs: Dict[str, float],
    already_validated: Set[FrozenSet],
    top_k: int = 20,
) -> List[Dict]:
    """
    For all pairs of off-label drugs, check synergy via combo_scorer.
    Returns pairs that are:
    - Synergistic (not antagonistic)
    - NOT already validated
    - Ranked by combo_score
    """
    try:
        from backend.pipeline.combo_scorer import CombinationScorer
    except ImportError:
        return []
    
    if len(off_label_drugs) < 2:
        return []
    
    scorer = CombinationScorer(disease_name=disease_name)
    drug_list = sorted(off_label_drugs.items(), key=lambda x: -x[1])[:top_k]
    
    novel_pairs = []
    import itertools
    
    for (drug_a, score_a), (drug_b, score_b) in itertools.combinations(drug_list, 2):
        pair_key = frozenset({drug_a, drug_b})
        
        if pair_key in already_validated:
            continue  # already known combo for this disease
        
        drug_a_dict = {
            "drug_name": drug_a,
            "score": score_a,
            "mechanism_score": score_a,
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
            # Novelty bonus: higher if mechanisms are from different classes
            mech_diversity = 1.0 if result["mechanism_a"] != result["mechanism_b"] else 0.7
            
            novel_pairs.append({
                "drugs": [drug_a, drug_b],
                "combo_score": result["combo_score"],
                "mechanism_a": result["mechanism_a"],
                "mechanism_b": result["mechanism_b"],
                "synergy_bonus": result["synergy_bonus"],
                "context_penalty": result.get("context_penalty", 0.0),
                "score_a": round(score_a, 3),
                "score_b": round(score_b, 3),
                "avg_signal": round((score_a + score_b) / 2, 3),
                "mech_diversity": mech_diversity,
                # Final novelty score: synergy × signal strength × mechanism diversity
                "novelty_discovery_score": round(
                    result["combo_score"] * ((score_a + score_b) / 2) * mech_diversity, 4
                ),
            })
    
    novel_pairs.sort(key=lambda x: -x["novelty_discovery_score"])
    return novel_pairs


# ─────────────────────────────────────────────────────────────────────────────
# Step 7: Main analysis across all diseases
# ─────────────────────────────────────────────────────────────────────────────

def analyse_novel_opportunities(
    all_drug_scores: Dict[str, Dict[str, float]],
    known_indications: Dict[str, Set[str]],
    validated_combos: Dict[str, Set[FrozenSet]],
    opportunity_scores: Dict[str, Dict],
    min_signal: float = 0.20,
    max_signal: float = 0.55,
    min_pairs: int = 1,
    disease_filter: Optional[str] = None,
) -> List[Dict]:
    
    results = []
    diseases = list(all_drug_scores.keys())
    
    if disease_filter:
        diseases = [d for d in diseases if disease_filter.lower() in d]
    
    print(f"\n  Analysing {len(diseases)} diseases for off-label synergistic pairs...")
    print(f"  Signal window: {min_signal:.2f} ≤ score ≤ {max_signal:.2f} (off-label range)")
    print(f"  (Drugs scoring > {max_signal:.2f} are excluded as likely already-indicated)")
    
    for disease in diseases:
        drug_scores = all_drug_scores[disease]
        known_drugs = known_indications.get(disease, set())
        already_validated = validated_combos.get(disease, set())
        opp_data = opportunity_scores.get(disease, {})
        
        # Find off-label signal drugs
        off_label = find_off_label_signals(
            disease, drug_scores, known_drugs, min_signal, max_signal
        )
        
        if len(off_label) < 2:
            continue  # need at least 2 drugs to form a pair
        
        # Find novel synergistic pairs among off-label drugs
        novel_pairs = find_novel_synergistic_pairs(
            disease, off_label, already_validated
        )
        
        if len(novel_pairs) < min_pairs:
            continue
        
        opp_score = opp_data.get("opportunity_score", 0.5)
        feasibility = opp_data.get("feasibility", 0.5)
        
        # Final priority:
        # opportunity × top_novelty × n_pairs_factor
        top_novelty = novel_pairs[0]["novelty_discovery_score"] if novel_pairs else 0.0
        n_pairs_factor = min(math.log10(len(novel_pairs) + 1) / math.log10(20), 1.0)
        
        final_priority = round(
            opp_score * 0.35
            + top_novelty * 0.40
            + feasibility * 0.15
            + n_pairs_factor * 0.10,
            4
        )
        
        results.append({
            "disease": disease,
            "final_priority": final_priority,
            "opportunity_score": opp_score,
            "feasibility": feasibility,
            "tier": opp_data.get("tier", "?"),
            "market_B": opp_data.get("market_size_B", 0),
            "orphan": opp_data.get("orphan_flag", False),
            "n_off_label_drugs": len(off_label),
            "n_novel_synergistic_pairs": len(novel_pairs),
            "top_novelty_score": round(top_novelty, 4),
            "n_known_excluded": len(known_drugs),
            "n_combos_excluded": len(already_validated),
            "top_novel_pairs": novel_pairs[:6],
            "off_label_drugs_sampled": dict(
                sorted(off_label.items(), key=lambda x: -x[1])[:8]
            ),
            "notes": opp_data.get("notes", ""),
        })
    
    results.sort(key=lambda x: -x["final_priority"])
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────────────────────

TIER_ICON = {"A": "🟢", "B": "🟡", "C": "🔴"}

def print_summary(results: List[Dict], top_n: int = 20) -> None:
    print("\n" + "═" * 95)
    print("  NOVEL DRUG REPURPOSING DISCOVERY")
    print("  Drugs scoring 0.20–0.55 for a disease they're NOT indicated for,")
    print("  forming synergistic pairs never before validated.")
    print("═" * 95)
    print(f"\n{'#':>3}  {'Disease':<40}  {'Tier':>4}  {'Priority':>8}  "
          f"{'Pairs':>5}  {'TopNovel':>8}  {'OffLabel':>8}  {'Excl':>5}")
    print("─" * 95)
    
    for i, r in enumerate(results[:top_n], 1):
        tier_icon = TIER_ICON.get(r["tier"], "⚪")
        orphan = "🏥" if r["orphan"] else "  "
        print(
            f"{i:>3}.  {r['disease'][:40]:<40}  {tier_icon}{r['tier']:>3}  "
            f"{r['final_priority']:>8.3f}  "
            f"{r['n_novel_synergistic_pairs']:>5}  "
            f"{r['top_novelty_score']:>8.4f}  "
            f"{r['n_off_label_drugs']:>8}  "
            f"{r['n_known_excluded']:>5}  {orphan}"
        )


def print_detail_cards(results: List[Dict], top_n: int = 5) -> None:
    print("\n" + "═" * 95)
    print(f"  TOP {top_n} NOVEL REPURPOSING OPPORTUNITIES — DETAILED VIEW")
    print("  These are combos where NEITHER drug is a known treatment for the disease.")
    print("═" * 95)
    
    for i, r in enumerate(results[:top_n], 1):
        tier_icon = TIER_ICON.get(r["tier"], "⚪")
        orphan_label = " [ORPHAN]" if r["orphan"] else ""
        
        print(f"\n{'─' * 75}")
        print(f"#{i}: {r['disease'].upper()}{orphan_label}")
        print(f"    Tier {tier_icon}{r['tier']}  |  Market: ${r['market_B']}B  |  "
              f"Opportunity: {r['opportunity_score']:.3f}  |  Feasibility: {r['feasibility']:.2f}")
        print(f"    Priority: {r['final_priority']:.3f}  |  "
              f"{r['n_novel_synergistic_pairs']} novel synergistic pairs  |  "
              f"{r['n_off_label_drugs']} off-label drugs with signal")
        print(f"    Excluded: {r['n_known_excluded']} known drugs + "
              f"{r['n_combos_excluded']} validated combos")
        
        if r["off_label_drugs_sampled"]:
            drugs_str = "  ".join(
                f"{k}={v:.2f}" 
                for k, v in list(r["off_label_drugs_sampled"].items())[:6]
            )
            print(f"\n    OFF-LABEL SIGNALS (not currently used for this disease):")
            print(f"    {drugs_str}")
        
        if r["top_novel_pairs"]:
            print(f"\n    TOP NOVEL SYNERGISTIC PAIRS (neither drug is indicated for this disease):")
            for j, pair in enumerate(r["top_novel_pairs"][:4], 1):
                drugs_str = " + ".join(p.upper() for p in pair["drugs"])
                mech_str = f"{pair['mechanism_a']} × {pair['mechanism_b']}"
                ctx = f" [ctx_pen={pair['context_penalty']:.2f}]" if pair.get("context_penalty", 0) > 0.15 else ""
                print(f"      {j}. {drugs_str}")
                print(f"         discovery_score={pair['novelty_discovery_score']:.4f}  "
                      f"combo={pair['combo_score']:.3f}  "
                      f"signals=({pair['score_a']:.2f}, {pair['score_b']:.2f}){ctx}")
                print(f"         mechanisms: {mech_str}")
        
        if r["notes"]:
            print(f"\n    DISEASE NOTE: {r['notes'][:130]}")


def print_action_items(results: List[Dict]) -> None:
    print("\n" + "═" * 95)
    print("  DISCOVERY ACTION ITEMS")
    print("═" * 95)
    
    if not results:
        print("\n  No novel opportunities found with current signal window.")
        return
    
    top = results[0]
    best_pair = top["top_novel_pairs"][0] if top["top_novel_pairs"] else None
    
    print(f"\n  🔬 TOP DISCOVERY CANDIDATE: {top['disease'].upper()}")
    if best_pair:
        print(f"     Novel combo: {' + '.join(p.upper() for p in best_pair['drugs'])}")
        print(f"     Mechanisms:  {best_pair['mechanism_a']} × {best_pair['mechanism_b']}")
        print(f"     Both drugs score {best_pair['avg_signal']:.2f} on average for this disease")
        print(f"     (off-label signal — neither is currently indicated)")
        print(f"\n     Next step: Run full pipeline to get richer gene/pathway evidence")
        print(f"     python -c \"")
        print(f"     import asyncio")
        print(f"     from backend.pipeline.production_pipeline import ProductionPipeline")
        print(f"     async def run():")
        print(f"         p = ProductionPipeline()")
        print(f"         plan = await p.generate_treatment_plan('{top['disease']}', max_regimens=20)")
        print(f"         # Look for {' + '.join(p.upper() for p in best_pair['drugs'])} in combos")
        print(f"         for r in plan['ranked_regimens'][:10]:")
        print(f"             print(r['rank'], r['regimen'], f\\\"ORR={{r['orr_estimate']:.1%}}\\\")")
        print(f"         await p.close()")
        print(f"     asyncio.run(run())\"")
    
    # Show the top cross-disease discovery (drug repurposed to totally different area)
    print(f"\n  📊 TO GET RICHER SIGNALS — run validation on new diseases:")
    print(f"     python run_validation.py --fast --disease \"<disease_name>\"")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Find genuinely novel drug repurposing combos (off-label signal only)"
    )
    parser.add_argument("--min-signal", type=float, default=0.20,
                        help="Min drug-disease score to be considered off-label signal (default 0.20)")
    parser.add_argument("--max-signal", type=float, default=0.55,
                        help="Max score — above this, drug likely already used for disease (default 0.55)")
    parser.add_argument("--top", type=int, default=20,
                        help="Diseases to show in summary table (default 20)")
    parser.add_argument("--detail", type=int, default=5,
                        help="Diseases to show in detail cards (default 5)")
    parser.add_argument("--min-pairs", type=int, default=1,
                        help="Min novel synergistic pairs required (default 1)")
    parser.add_argument("--disease", type=str, default=None,
                        help="Filter to a specific disease (substring match)")
    parser.add_argument("--json", action="store_true",
                        help="Output machine-readable JSON")
    args = parser.parse_args()
    
    print("\n" + "═" * 95)
    print("  NOVEL DRUG REPURPOSING DISCOVERY ENGINE")
    print("  Finding off-label synergistic pairs — drugs not currently used for the disease")
    print("═" * 95)
    
    print("\n[1/5] Loading drug-disease scores...")
    all_scores = load_all_drug_disease_scores()
    
    print("\n[2/5] Loading known indications (to exclude)...")
    known_indications = load_known_indications()
    
    print("\n[3/5] Loading validated combos (to exclude)...")
    validated_combos = load_validated_combos()
    
    print("\n[4/5] Loading disease opportunity scores...")
    opportunity_scores = load_opportunity_scores()
    
    print("\n[5/5] Finding novel synergistic pairs in off-label signal window...")
    print(f"       Signal window: [{args.min_signal:.2f}, {args.max_signal:.2f}]")
    
    results = analyse_novel_opportunities(
        all_drug_scores=all_scores,
        known_indications=known_indications,
        validated_combos=validated_combos,
        opportunity_scores=opportunity_scores,
        min_signal=args.min_signal,
        max_signal=args.max_signal,
        min_pairs=args.min_pairs,
        disease_filter=args.disease,
    )
    
    if not results:
        print("\nNo novel opportunities found. Try widening the signal window:")
        print("  --min-signal 0.15 --max-signal 0.65")
        return
    
    print(f"\n  Found {len(results)} diseases with novel off-label synergistic pairs")
    
    if args.json:
        output = {
            "signal_window": [args.min_signal, args.max_signal],
            "n_diseases_analysed": len(all_scores),
            "n_with_novel_pairs": len(results),
            "results": results[:args.top],
        }
        print(json.dumps(output, indent=2))
        return
    
    print_summary(results, top_n=args.top)
    print_detail_cards(results, top_n=args.detail)
    print_action_items(results)
    
    out_path = Path("novel_repurposing_candidates.json")
    out_path.write_text(json.dumps({
        "generated_at": __import__("datetime").datetime.utcnow().isoformat(),
        "signal_window": {"min": args.min_signal, "max": args.max_signal},
        "methodology": (
            "Off-label discovery: drugs scoring in [min_signal, max_signal] for a disease "
            "they are NOT currently indicated for. Both drugs in each pair are off-label. "
            "Pairs scored using CombinationScorer synergy model from combo_scorer.py. "
            "Known indications excluded via validation_dataset.py, drug_similarity.py, "
            "and disease_ranker.py known_generics."
        ),
        "n_diseases_with_novel_pairs": len(results),
        "results": results,
    }, indent=2))
    print(f"\n  Full report saved to: {out_path.resolve()}")
    print("═" * 95)


if __name__ == "__main__":
    main()