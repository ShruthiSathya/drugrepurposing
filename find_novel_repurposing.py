#!/usr/bin/env python3
"""
find_novel_repurposing.py
=========================
Discovers more trustworthy novel drug repurposing combo opportunities.

This version adds guardrails:
1. Refuses to use a filtered validation_results.json as if it were a full matrix.
2. Refuses to run when the drug-disease matrix is too small to support discovery.
3. Requires stronger off-label evidence than the old permissive defaults.
4. Excludes context-free / weakly grounded pairs from novel discovery.

Usage
-----
    python find_novel_repurposing.py
    python find_novel_repurposing.py --min-signal 0.24 --max-signal 0.55
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

MIN_DISEASES_FOR_DISCOVERY = 5
MIN_DRUG_DISEASE_PAIRS_FOR_DISCOVERY = 40


def norm(name: str) -> str:
    n = (name or "").lower().strip()
    n = _SALT_RE.sub("", n).strip()
    n = _SALT_RE.sub("", n).strip()
    return n


def _read_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def _validation_summary(data: Dict) -> Dict[str, object]:
    positive = data.get("positive_results", []) or []
    negative = data.get("negative_results", []) or []
    all_results = positive + negative

    diseases = {str(r.get("disease", "")).lower() for r in all_results if r.get("disease")}
    disease_filter = data.get("disease_filter")

    return {
        "disease_filter": disease_filter,
        "n_diseases": len(diseases),
        "n_pairs": len(all_results),
    }


def validate_input_matrix(strict: bool = True) -> Tuple[bool, str]:
    """
    Ensure validation_results.json represents a full enough matrix for discovery.

    Returns (ok, message). In strict mode, filtered or tiny matrices are rejected.
    """
    val_path = Path("validation_results.json")
    if not val_path.exists():
        return False, "validation_results.json not found. Run a full validation first."

    try:
        data = _read_json(val_path)
    except Exception as exc:
        return False, f"Could not read validation_results.json: {exc}"

    summary = _validation_summary(data)

    if summary["disease_filter"]:
        return (
            False,
            "validation_results.json was generated with a disease_filter. "
            "Run validation again without --disease so discovery uses a full matrix."
        )

    if strict:
        if summary["n_diseases"] < MIN_DISEASES_FOR_DISCOVERY:
            return (
                False,
                f"Validation matrix too small for discovery: only {summary['n_diseases']} diseases found. "
                f"Need at least {MIN_DISEASES_FOR_DISCOVERY}."
            )
        if summary["n_pairs"] < MIN_DRUG_DISEASE_PAIRS_FOR_DISCOVERY:
            return (
                False,
                f"Validation matrix too small for discovery: only {summary['n_pairs']} drug-disease pairs found. "
                f"Need at least {MIN_DRUG_DISEASE_PAIRS_FOR_DISCOVERY}."
            )

    return True, (
        f"Validation matrix OK: {summary['n_pairs']} drug-disease pairs "
        f"across {summary['n_diseases']} diseases."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Load ALL drug-disease scores from validation run
# ─────────────────────────────────────────────────────────────────────────────

def load_all_drug_disease_scores() -> Dict[str, Dict[str, float]]:
    """
    Returns {disease: {drug: score}} from validation_results.json.

    This function assumes validate_input_matrix() was called first.
    """
    scores: Dict[str, Dict[str, float]] = {}

    val_path = Path("validation_results.json")
    if not val_path.exists():
        print("  WARNING: validation_results.json not found")
        return scores

    data = _read_json(val_path)
    all_results = (data.get("positive_results", []) or []) + (data.get("negative_results", []) or [])

    for r in all_results:
        disease = str(r.get("disease", "")).lower()
        drug = norm(str(r.get("drug", "")))
        score = float(r.get("raw_score", 0.0) or 0.0)

        if not disease or not drug:
            continue
        scores.setdefault(disease, {})
        scores[disease][drug] = max(scores[disease].get(drug, 0.0), score)

    combo_path = Path("combo_validation_results.json")
    if combo_path.exists():
        combo_data = _read_json(combo_path)
        for r in combo_data.get("positive_results", []) or []:
            disease = str(r.get("disease", "")).lower()
            if not disease:
                continue
            scores.setdefault(disease, {})
            for drug_name, drug_score in (r.get("individual_scores", {}) or {}).items():
                d = norm(drug_name)
                scores[disease][d] = max(scores[disease].get(d, 0.0), float(drug_score or 0.0))

    print(
        f"  Loaded drug-disease scores: {sum(len(v) for v in scores.values())} "
        f"pairs across {len(scores)} diseases"
    )
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Load known/validated drug-disease pairs to EXCLUDE
# ─────────────────────────────────────────────────────────────────────────────

def load_known_indications() -> Dict[str, Set[str]]:
    known: Dict[str, Set[str]] = defaultdict(set)

    try:
        from backend.pipeline.validation_dataset import get_positive_cases
        for case in get_positive_cases():
            disease = case["disease"].lower()
            known[disease].add(norm(case["drug"]))
    except ImportError:
        pass

    try:
        from backend.pipeline.drug_similarity import DISEASE_KNOWN_DRUGS
        for disease, drugs in DISEASE_KNOWN_DRUGS.items():
            for drug in drugs:
                known[disease.lower()].add(norm(drug))
    except ImportError:
        pass

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
    validated: Dict[str, Set[FrozenSet]] = defaultdict(set)

    results_path = Path("combo_validation_results.json")
    if results_path.exists():
        data = _read_json(results_path)
        for r in data.get("positive_results", []) or []:
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
# ─────────────────────────────────────────────────────────────────────────────

def find_off_label_signals(
    disease: str,
    drug_scores: Dict[str, float],
    known_for_disease: Set[str],
    min_signal: float = 0.24,
    max_signal: float = 0.55,
) -> Dict[str, float]:
    """
    Returns drugs with off-label signal for this disease:
    - Score in [min_signal, max_signal]
    - NOT a known indication for this disease
    """
    off_label = {}
    for drug, score in drug_scores.items():
        if score < min_signal:
            continue
        if score > max_signal:
            continue
        if drug in known_for_disease:
            continue

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
            continue

        # Deliberately sparse dicts are okay, but the scorer will now penalise
        # context-free / weakly grounded candidates instead of rewarding them.
        drug_a_dict = {
            "drug_name": drug_a,
            "score": score_a,
            "mechanism_score": score_a,
            "mechanism": "",
            "target_genes": [],
            "targets": [],
            "pathways": [],
        }
        drug_b_dict = {
            "drug_name": drug_b,
            "score": score_b,
            "mechanism_score": score_b,
            "mechanism": "",
            "target_genes": [],
            "targets": [],
            "pathways": [],
        }

        result = scorer.score_pair(drug_a_dict, drug_b_dict, [])

        if not result["is_synergistic"] or result["is_antagonistic"]:
            continue

        # Reject pairs that remain too weakly grounded after scoring.
        prof_a = result.get("evidence_profile_a", {})
        prof_b = result.get("evidence_profile_b", {})
        if prof_a.get("weakly_grounded") and prof_b.get("weakly_grounded"):
            continue

        if result.get("context_penalty", 0.0) >= 0.40:
            continue

        mech_diversity = 1.0 if result["mechanism_a"] != result["mechanism_b"] else 0.7

        novelty_discovery_score = round(
            result["combo_score"] * ((score_a + score_b) / 2.0) * mech_diversity,
            4,
        )
        if novelty_discovery_score <= 0:
            continue

        novel_pairs.append({
            "drugs": [drug_a, drug_b],
            "combo_score": result["combo_score"],
            "mechanism_a": result["mechanism_a"],
            "mechanism_b": result["mechanism_b"],
            "synergy_bonus": result["synergy_bonus"],
            "context_penalty": result.get("context_penalty", 0.0),
            "evidence_penalty": result.get("evidence_penalty", 0.0),
            "score_a": round(score_a, 3),
            "score_b": round(score_b, 3),
            "avg_signal": round((score_a + score_b) / 2.0, 3),
            "mech_diversity": mech_diversity,
            "novelty_discovery_score": novelty_discovery_score,
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
    min_signal: float = 0.24,
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

        off_label = find_off_label_signals(
            disease, drug_scores, known_drugs, min_signal, max_signal
        )
        if len(off_label) < 2:
            continue

        novel_pairs = find_novel_synergistic_pairs(
            disease, off_label, already_validated
        )
        if len(novel_pairs) < min_pairs:
            continue

        opp_score = opp_data.get("opportunity_score", 0.5)
        feasibility = opp_data.get("feasibility", 0.5)

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
    print("  Drugs scoring in a moderate off-label range for a disease they're NOT")
    print("  indicated for, forming synergistic pairs never before validated.")
    print("═" * 95)
    print(
        f"\n{'#':>3}  {'Disease':<40}  {'Tier':>4}  {'Priority':>8}  "
        f"{'Pairs':>5}  {'TopNovel':>8}  {'OffLabel':>8}  {'Excl':>5}"
    )
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
    print("  These are combos where neither drug is a known treatment for the disease.")
    print("═" * 95)

    for i, r in enumerate(results[:top_n], 1):
        tier_icon = TIER_ICON.get(r["tier"], "⚪")
        orphan_label = " [ORPHAN]" if r["orphan"] else ""

        print(f"\n{'─' * 75}")
        print(f"#{i}: {r['disease'].upper()}{orphan_label}")
        print(
            f"    Tier {tier_icon}{r['tier']}  |  Market: ${r['market_B']}B  |  "
            f"Opportunity: {r['opportunity_score']:.3f}  |  Feasibility: {r['feasibility']:.2f}"
        )
        print(
            f"    Priority: {r['final_priority']:.3f}  |  "
            f"{r['n_novel_synergistic_pairs']} novel synergistic pairs  |  "
            f"{r['n_off_label_drugs']} off-label drugs with signal"
        )
        print(
            f"    Excluded: {r['n_known_excluded']} known drugs + "
            f"{r['n_combos_excluded']} validated combos"
        )

        if r["off_label_drugs_sampled"]:
            drugs_str = "  ".join(
                f"{k}={v:.2f}"
                for k, v in list(r["off_label_drugs_sampled"].items())[:6]
            )
            print("\n    OFF-LABEL SIGNALS (not currently used for this disease):")
            print(f"    {drugs_str}")

        if r["top_novel_pairs"]:
            print("\n    TOP NOVEL SYNERGISTIC PAIRS:")
            for j, pair in enumerate(r["top_novel_pairs"][:4], 1):
                drugs_str = " + ".join(p.upper() for p in pair["drugs"])
                mech_str = f"{pair['mechanism_a']} × {pair['mechanism_b']}"
                ctx = (
                    f" [ctx_pen={pair['context_penalty']:.2f}]"
                    if pair.get("context_penalty", 0) > 0.15 else ""
                )
                evp = (
                    f" [ev_pen={pair['evidence_penalty']:.2f}]"
                    if pair.get("evidence_penalty", 0) > 0.0 else ""
                )
                print(f"      {j}. {drugs_str}")
                print(
                    f"         discovery_score={pair['novelty_discovery_score']:.4f}  "
                    f"combo={pair['combo_score']:.3f}  "
                    f"signals=({pair['score_a']:.2f}, {pair['score_b']:.2f}){ctx}{evp}"
                )
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
        print("     (moderate off-label signal — neither is currently indicated)")
        print("\n     Next step: run the full treatment pipeline and inspect gene/pathway evidence")
        print("     for this pair before treating it as a serious candidate.")

    print("\n  📊 To strengthen discovery quality, rerun validation on a broader set of diseases.")
    print("     Avoid filtered matrices for discovery.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Find more trustworthy novel drug repurposing combos (off-label signal only)"
    )
    parser.add_argument(
        "--min-signal",
        type=float,
        default=0.24,
        help="Min drug-disease score to be considered off-label signal (default 0.24)",
    )
    parser.add_argument(
        "--max-signal",
        type=float,
        default=0.55,
        help="Max score — above this, drug likely already used for disease (default 0.55)",
    )
    parser.add_argument("--top", type=int, default=20, help="Diseases to show in summary table (default 20)")
    parser.add_argument("--detail", type=int, default=5, help="Diseases to show in detail cards (default 5)")
    parser.add_argument("--min-pairs", type=int, default=1, help="Min novel synergistic pairs required (default 1)")
    parser.add_argument("--disease", type=str, default=None, help="Filter to a specific disease (substring match)")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args()

    print("\n" + "═" * 95)
    print("  NOVEL DRUG REPURPOSING DISCOVERY ENGINE")
    print("  Finding off-label synergistic pairs with stricter evidence guardrails")
    print("═" * 95)

    ok, msg = validate_input_matrix(strict=True)
    print(f"\n[0/5] Matrix check: {msg}")
    if not ok:
        print("\n  Stop here and rerun validation without a disease filter.")
        print("  Example:")
        print("    python run_validation.py --output validation_results.json")
        return

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
        print("\nNo novel opportunities found. Try widening the signal window slightly:")
        print("  --min-signal 0.20 --max-signal 0.60")
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
    out_path.write_text(
        json.dumps(
            {
                "generated_at": __import__("datetime").datetime.utcnow().isoformat(),
                "signal_window": {"min": args.min_signal, "max": args.max_signal},
                "methodology": (
                    "Off-label discovery: drugs scoring in [min_signal, max_signal] for a disease "
                    "they are not currently indicated for. Both drugs in each pair are off-label. "
                    "Pairs scored using CombinationScorer with added evidence-grounding penalties. "
                    "Filtered or undersized validation matrices are rejected."
                ),
                "n_diseases_with_novel_pairs": len(results),
                "results": results,
            },
            indent=2,
        )
    )
    print(f"\n  Full report saved to: {out_path.resolve()}")
    print("═" * 95)


if __name__ == "__main__":
    main()
