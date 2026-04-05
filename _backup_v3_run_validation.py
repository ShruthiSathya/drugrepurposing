"""
run_validation.py — Curated Validation Runner
==============================================
Runs the production pipeline against the curated 55-case validation dataset
and writes results to validation_results.json.

Usage
-----
    # Full run (accurate but slow ~45-90 min):
    python run_validation.py

    # Fast run (skips PPI/tissue/EFO APIs — ~5-10 min):
    python run_validation.py --fast

    # Test a single disease quickly:
    python run_validation.py --fast --disease "rheumatoid arthritis"

    # Other options:
    python run_validation.py [--min-score 0.0] [--output validation_results.json]
                             [--skip-baselines] [--skip-sensitivity] [--fast]
                             [--disease DISEASE_FILTER]
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.pipeline.production_pipeline import ProductionPipeline
from backend.pipeline.calibration import load_calibrator
from backend.pipeline.baselines import (
    JaccardOverlapBaseline,
    GeneCountBaseline,
    CosineSimilarityBaseline,
)
from backend.pipeline.scorer import (
    sensitivity_analysis,
    WEIGHT_GENE, WEIGHT_PATHWAY, WEIGHT_PPI,
    WEIGHT_SIMILARITY, WEIGHT_MECHANISM, WEIGHT_LITERATURE,
)
from backend.pipeline.validation_dataset import (
    VALIDATION_CASES,
    DATASET_VERSION,
    N_TEST_CASES,
    get_positive_cases,
    get_negative_cases,
    OUT_OF_SCOPE_CASES,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Disease area stratification
# ─────────────────────────────────────────────────────────────────────────────
DISEASE_AREA_MAP: Dict[str, List[str]] = {
    "oncology": [
        "multiple myeloma", "chronic myelogenous leukemia",
        "gastrointestinal stromal tumor", "breast cancer", "ovarian cancer",
        "non-small cell lung carcinoma", "melanoma", "colorectal cancer",
    ],
    "autoimmune_inflammatory": [
        "rheumatoid arthritis", "systemic lupus erythematosus",
        "inflammatory bowel disease", "pericarditis",
    ],
    "cardiovascular": [
        "pulmonary arterial hypertension", "heart failure",
        "coronary artery disease",
    ],
    "neurological": [
        "epilepsy", "parkinson disease", "alzheimer disease",
        "multiple sclerosis", "amyotrophic lateral sclerosis",
        "essential tremor",
    ],
    "metabolic": [
        "type 2 diabetes mellitus", "polycystic ovary syndrome",
        "hypercholesterolemia", "gout",
    ],
    "rare_disease": [
        "tuberous sclerosis", "paroxysmal nocturnal hemoglobinuria",
        "cystic fibrosis", "infantile hemangioma",
        "benign prostatic hyperplasia", "alopecia",
    ],
}


def classify_disease_area(disease_name: str) -> str:
    name_lower = disease_name.lower()
    for area, diseases in DISEASE_AREA_MAP.items():
        for d in diseases:
            if d in name_lower or name_lower in d:
                return area
    return "other"


def compute_stratified_metrics(all_results: List[Dict]) -> Dict:
    area_counts: Dict[str, Dict] = {}
    for r in all_results:
        area = classify_disease_area(r["disease"])
        if area not in area_counts:
            area_counts[area] = {"tp": 0, "fn": 0, "tn": 0, "fp": 0}
        expected = r.get("expected_status", "")
        passed = r.get("pass", False)
        if expected == "TRUE_POSITIVE":
            area_counts[area]["tp" if passed else "fn"] += 1
        elif expected == "TRUE_NEGATIVE":
            area_counts[area]["tn" if passed else "fp"] += 1

    stratified = {}
    for area, c in area_counts.items():
        n_pos = c["tp"] + c["fn"]
        n_neg = c["tn"] + c["fp"]
        sens = c["tp"] / n_pos if n_pos else None
        spec = c["tn"] / n_neg if n_neg else None
        stratified[area] = {
            "tp": c["tp"], "fn": c["fn"], "tn": c["tn"], "fp": c["fp"],
            "sensitivity": round(sens, 4) if sens is not None else None,
            "specificity": round(spec, 4) if spec is not None else None,
            "n_positive_cases": n_pos,
            "n_negative_cases": n_neg,
        }
    return stratified


# ─────────────────────────────────────────────────────────────────────────────
# Data integrity checks
# ─────────────────────────────────────────────────────────────────────────────
def check_data_integrity(drugs_data: List[Dict]) -> Dict:
    issues, warnings = [], []

    drugs_with_targets = sum(1 for d in drugs_data if d.get("targets"))
    coverage = drugs_with_targets / len(drugs_data) if drugs_data else 0
    if coverage < 0.50:
        issues.append(f"CRITICAL: Only {coverage:.1%} drugs have gene targets (expected >50%)")
    elif coverage < 0.65:
        warnings.append(f"Low target coverage: {coverage:.1%} (expected >65%)")

    names = [d["name"].lower() for d in drugs_data]
    n_dupes = len(names) - len(set(names))
    if n_dupes > 0:
        issues.append(f"Found {n_dupes} duplicate drug names in pool")

    if len(drugs_data) < 1500:
        issues.append(f"Drug pool too small: {len(drugs_data)} (expected >1500)")
    elif len(drugs_data) < 2000:
        warnings.append(f"Drug pool smaller than expected: {len(drugs_data)}")

    essential = ["imatinib", "sildenafil", "metformin", "thalidomide", "rituximab",
                 "bosentan", "dexamethasone", "spironolactone", "atorvastatin"]
    drug_names_lower = {d["name"].lower() for d in drugs_data}
    missing = [e for e in essential if e not in drug_names_lower]
    if missing:
        warnings.append(f"Missing essential drugs: {missing}")

    # Check that key drugs have targets
    no_targets = []
    for name in ["sildenafil", "bosentan", "dexamethasone", "spironolactone",
                 "atorvastatin", "memantine", "rasagiline"]:
        drug = next((d for d in drugs_data if d["name"].lower() == name), None)
        if drug and not drug.get("targets"):
            no_targets.append(name)
    if no_targets:
        warnings.append(
            f"Key drugs have no targets (will rely on mechanism scoring only): {no_targets}\n"
            f"  → Fix: delete /tmp/drug_repurposing_cache/chembl_approved_drugs.json and re-run"
        )

    return {
        "passed": len(issues) == 0,
        "n_drugs": len(drugs_data),
        "target_coverage": round(coverage, 4),
        "issues": issues,
        "warnings": warnings,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline fingerprint
# ─────────────────────────────────────────────────────────────────────────────
def get_pipeline_fingerprint(drugs_data: List[Dict], fast_mode: bool = False) -> Dict:
    target_sources: Dict[str, int] = {}
    for d in drugs_data:
        src = d.get("target_source", "none")
        target_sources[src] = target_sources.get(src, 0) + 1

    return {
        "dataset_version": DATASET_VERSION,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "fast_mode": fast_mode,
        "n_drugs_in_pool": len(drugs_data),
        "target_source_breakdown": target_sources,
        "scoring_weights": {
            "gene": WEIGHT_GENE,
            "pathway": WEIGHT_PATHWAY,
            "ppi": WEIGHT_PPI,
            "similarity": WEIGHT_SIMILARITY,
            "mechanism": WEIGHT_MECHANISM,
            "literature": WEIGHT_LITERATURE,
        },
        "data_sources": [
            "ChEMBL (max_phase=4)",
            "OpenTargets Platform",
            "DGIdb 4.0",
            "STRING v12 (PPI)" + (" [SKIPPED in fast mode]" if fast_mode else ""),
            "Reactome v88 + KEGG PATHWAY (pathway annotations)",
            "ClinicalTrials.gov v2",
            "OpenFDA",
        ],
        "fast_mode_note": (
            "Fast mode skips STRING PPI, tissue expression, drug similarity, and EFO "
            "expansion to reduce runtime from ~60-90min to ~5-10min. "
            "PPI/tissue/similarity scores will be 0 in fast mode."
        ) if fast_mode else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Baseline comparison
# ─────────────────────────────────────────────────────────────────────────────
async def run_baseline_comparison(
    positive_cases: List[Dict],
    negative_cases: List[Dict],
    pipeline: ProductionPipeline,
    drugs_data: List[Dict],
    top_k: int = 100,
    disease_cache: Optional[Dict] = None,
) -> Dict:
    logger.info("\n" + "=" * 60)
    logger.info("BASELINE COMPARISON (Hit@%d)", top_k)
    logger.info("=" * 60)

    if disease_cache is None:
        disease_cache = {}

    baselines = {
        "jaccard":    JaccardOverlapBaseline(),
        "gene_count": GeneCountBaseline(),
        "cosine":     CosineSimilarityBaseline(use_tfidf=True),
    }
    all_gene_lists = [d.get("targets", []) for d in drugs_data]
    baselines["cosine"].fit(all_gene_lists)

    counts = {name: {"tp": 0, "fn": 0, "tn": 0, "fp": 0} for name in baselines}

    for case in positive_cases + negative_cases:
        disease = case["disease"]
        if disease not in disease_cache:
            disease_cache[disease] = await pipeline.data_fetcher.fetch_disease_data(disease)
        disease_data = disease_cache[disease]
        if not disease_data:
            continue

        for name, baseline in baselines.items():
            results = baseline.score_all(drugs_data, disease_data, min_score=0.0)
            top_names = {r["drug_name"].lower() for r in results[:top_k]}
            drug_lower = case["drug"].lower()
            if case["status"] == "TRUE_POSITIVE":
                counts[name]["tp" if drug_lower in top_names else "fn"] += 1
            else:
                counts[name]["tn" if drug_lower not in top_names else "fp"] += 1

    metrics = {}
    for name, c in counts.items():
        n_pos = c["tp"] + c["fn"]
        n_neg = c["tn"] + c["fp"]
        sens = c["tp"] / n_pos if n_pos else 0
        spec = c["tn"] / n_neg if n_neg else 0
        prec = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0
        f1 = (
            2 * prec * sens / (prec + sens)
            if (prec + sens) else 0
        )
        metrics[name] = {
            "sensitivity": round(sens, 4), "specificity": round(spec, 4),
            "precision": round(prec, 4), "f1": round(f1, 4), **c,
        }
        logger.info(
            "  %-15s: sens=%.3f spec=%.3f f1=%.3f  (TP=%d FN=%d TN=%d FP=%d)",
            name, sens, spec, f1, c["tp"], c["fn"], c["tn"], c["fp"],
        )

    return {"top_k": top_k, "baselines": metrics}


# ─────────────────────────────────────────────────────────────────────────────
# Single case runner
# ─────────────────────────────────────────────────────────────────────────────
async def run_single_validation_case(
    pipeline: ProductionPipeline,
    drugs_data: List[Dict],
    case: Dict,
    calibrator: Any,
    fast_mode: bool = False,
    disease_cache: Optional[Dict] = None,
) -> Dict:
    """
    Run one validation case.

    fast_mode=True skips PPI (STRING API), tissue expression (OT API),
    drug similarity, and EFO expansion. This reduces per-case time from
    ~60-120s to ~3-8s — critical for iterating on the pipeline quickly.
    """
    if disease_cache is None:
        disease_cache = {}

    drug_name    = case["drug"]
    disease_name = case["disease"]
    logger.info("  Testing: %s vs %s %s...",
                drug_name, disease_name,
                "[FAST]" if fast_mode else "")

    # Use cached disease data when available (avoids repeated OpenTargets calls)
    if disease_name not in disease_cache:
        disease_cache[disease_name] = await pipeline.data_fetcher.fetch_disease_data(disease_name)
    disease_data = disease_cache[disease_name]

    if not disease_data:
        logger.warning("    Disease not found in OpenTargets: %s", disease_name)
        return {
            "drug":                drug_name,
            "disease":             disease_name,
            "disease_area":        classify_disease_area(disease_name),
            "status":              "analysis_failed",
            "reason":              "Disease not found in OpenTargets",
            "raw_score":           0.0,
            "calibrated_score":    calibrator.transform(0.0),
            "rank":                None,
            "expected_rank_top_n": case["expected_rank_top_n"],
            "rank_pass":           False,
            "score_pass":          False,
            "expected_status":     case["status"],
            "pass":                False,
            "notes":               case.get("notes", ""),
            "score_components":    {},
        }

    candidates = await pipeline.generate_candidates(
        disease_data=disease_data,
        drugs_data=drugs_data,
        min_score=0.0,
        fetch_pubmed=False,
        # Fast mode: skip all external API calls (STRING, OT tissue, EFO)
        # This is the main speedup — each of these can add 30-90s per case
        fetch_ppi=not fast_mode,
        fetch_similarity=not fast_mode,
        use_efo=not fast_mode,
        use_tissue=False,        # Always off — too slow even in full mode
        use_polypharm=True,      # In-memory, always fast
    )

    candidates_sorted = sorted(candidates, key=lambda x: x["score"], reverse=True)
    drug_lower = drug_name.lower()
    found_candidate = None
    found_rank = None

    for rank, cand in enumerate(candidates_sorted, 1):
        if cand["name"].lower() == drug_lower:
            found_candidate = cand
            found_rank = rank
            break

    if found_candidate is None:
        raw_score = 0.0
        cal_score = calibrator.transform(0.0)
        rank_ok   = False
        score_ok  = case["status"] == "TRUE_NEGATIVE"
        passed    = score_ok
        result_status = (
            "false_negative" if case["status"] == "TRUE_POSITIVE"
            else "true_negative_not_found"
        )
        logger.info("    %s not in candidates — status: %s", drug_name, result_status)
        score_components = {}
    else:
        raw_score = found_candidate["score"]
        cal_score = calibrator.transform(raw_score)

        if case["status"] == "TRUE_POSITIVE":
            rank_ok  = (
                found_rank is not None
                and case["expected_rank_top_n"] > 0
                and found_rank <= case["expected_rank_top_n"]
            )
            score_ok = raw_score >= case["min_score"]
            passed   = rank_ok or score_ok
            result_status = "found" if passed else "false_negative"
            if not passed:
                logger.warning(
                    "    FAIL: %s score %.3f < %.3f AND rank %s > %s",
                    drug_name, raw_score, case["min_score"],
                    found_rank, case["expected_rank_top_n"],
                )
        else:  # TRUE_NEGATIVE
            rank_ok  = False
            score_ok = raw_score < case["min_score"]
            passed   = score_ok
            result_status = (
                "true_negative_low_score" if passed else "false_positive"
            )
            if not passed:
                logger.warning(
                    "    FAIL (FP): %s scored %.3f (threshold %.3f)",
                    drug_name, raw_score, case["min_score"],
                )

        score_components = {
            "gene_score":       found_candidate.get("gene_score", 0.0),
            "pathway_score":    found_candidate.get("pathway_score", 0.0),
            "ppi_score":        found_candidate.get("ppi_score", 0.0),
            "similarity_score": found_candidate.get("similarity_score", 0.0),
            "mechanism_score":  found_candidate.get("mechanism_score", 0.0),
            "literature_score": found_candidate.get("literature_score", 0.0),
            "shared_genes":     found_candidate.get("shared_genes", []),
            "shared_pathways":  found_candidate.get("shared_pathways", []),
        }

    result = {
        "drug":                  drug_name,
        "disease":               disease_name,
        "disease_area":          classify_disease_area(disease_name),
        "status":                result_status,
        "expected_status":       case["status"],
        "raw_score":             round(raw_score, 4),
        "calibrated_score":      round(cal_score, 4),
        "rank":                  found_rank,
        "total_candidates":      len(candidates_sorted),
        "expected_rank_top_n":   case["expected_rank_top_n"],
        "rank_pass":             rank_ok,
        "score_pass":            score_ok,
        "pass":                  passed,
        "min_score_threshold":   case["min_score"],
        "mechanism":             case["mechanism"],
        "sources":               case["sources"],
        "notes":                 case.get("notes", ""),
        "score_components":      score_components,
        "fast_mode":             fast_mode,
    }

    logger.info(
        "    %s — raw=%.3f cal=%.3f rank=%s status=%s area=%s",
        "PASS ✓" if passed else "FAIL ✗",
        raw_score, cal_score, found_rank, result_status,
        result["disease_area"],
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────
async def run_all_validations(
    min_score:       float = 0.0,
    output_path:     str   = "validation_results.json",
    run_baselines:   bool  = True,
    run_sensitivity: bool  = True,
    fast_mode:       bool  = False,
    disease_filter:  Optional[str] = None,
) -> Dict:
    pipeline   = ProductionPipeline()
    calibrator = load_calibrator()
    start_utc  = datetime.now(timezone.utc)

    mode_label = "FAST MODE" if fast_mode else "FULL MODE"
    logger.info("=" * 70)
    logger.info(
        "VALIDATION RUN [%s] — Dataset %s — %d cases",
        mode_label, DATASET_VERSION, N_TEST_CASES,
    )
    if fast_mode:
        logger.info(
            "  Fast mode: PPI (STRING API), EFO expansion, and drug similarity "
            "scoring are DISABLED. Runtime: ~5-10 min instead of ~60-90 min."
        )
    logger.info("=" * 70)

    # Shared disease data cache — avoids re-fetching the same disease
    # (e.g. "rheumatoid arthritis" appears in multiple test cases)
    disease_cache: Dict = {}

    try:
        logger.info("\nFetching approved drugs (shared across all test cases)...")
        drugs_data = await pipeline.fetch_approved_drugs(limit=3000)
        logger.info("Using %d drugs for all tests\n", len(drugs_data))

        integrity   = check_data_integrity(drugs_data)
        if not integrity["passed"]:
            for issue in integrity["issues"]:
                logger.error("DATA INTEGRITY: %s", issue)
            logger.warning("Continuing despite integrity issues — results may be unreliable")
        for w in integrity.get("warnings", []):
            logger.warning("DATA INTEGRITY WARNING: %s", w)

        fingerprint = get_pipeline_fingerprint(drugs_data, fast_mode=fast_mode)

        positive_cases = get_positive_cases()
        negative_cases = get_negative_cases()

        # Apply disease filter if specified
        if disease_filter:
            df = disease_filter.lower()
            positive_cases = [c for c in positive_cases if df in c["disease"].lower()]
            negative_cases = [c for c in negative_cases if df in c["disease"].lower()]
            logger.info(
                "Disease filter '%s': %d positive, %d negative cases",
                disease_filter, len(positive_cases), len(negative_cases),
            )

        logger.info("Running %d TRUE_POSITIVE cases...", len(positive_cases))
        positive_results = []
        for i, case in enumerate(positive_cases, 1):
            logger.info("  [%d/%d]", i, len(positive_cases))
            result = await run_single_validation_case(
                pipeline, drugs_data, case, calibrator,
                fast_mode=fast_mode,
                disease_cache=disease_cache,
            )
            positive_results.append(result)

        logger.info("\nRunning %d TRUE_NEGATIVE cases...", len(negative_cases))
        negative_results = []
        for i, case in enumerate(negative_cases, 1):
            logger.info("  [%d/%d]", i, len(negative_cases))
            result = await run_single_validation_case(
                pipeline, drugs_data, case, calibrator,
                fast_mode=fast_mode,
                disease_cache=disease_cache,
            )
            negative_results.append(result)

        baseline_comparison: Dict = {}
        if run_baselines:
            try:
                baseline_comparison = await run_baseline_comparison(
                    positive_cases, negative_cases, pipeline, drugs_data,
                    disease_cache=disease_cache,
                )
            except Exception as e:
                logger.warning("Baseline comparison failed (non-fatal): %s", e)

        sensitivity_result: Dict = {}
        if run_sensitivity:
            try:
                candidates_with_components = [
                    r["score_components"] for r in positive_results + negative_results
                    if r.get("score_components")
                ]
                if candidates_with_components:
                    sensitivity_result = sensitivity_analysis(
                        candidates=candidates_with_components,
                        perturbation=0.10,
                    )
                    logger.info(
                        "\nSensitivity analysis: ρ_min=%.3f  stable=%s",
                        sensitivity_result.get("rank_correlation_min", 0),
                        sensitivity_result.get("stable"),
                    )
            except Exception as e:
                logger.warning("Sensitivity analysis failed (non-fatal): %s", e)

    finally:
        await pipeline.close()

    # ── Metrics ───────────────────────────────────────────────────────────────
    n_pos = len(positive_results)
    n_neg = len(negative_results)
    tp    = sum(1 for r in positive_results if r["pass"])
    fn    = n_pos - tp
    tn    = sum(1 for r in negative_results if r["pass"])
    fp    = n_neg - tn

    sensitivity = tp / n_pos if n_pos else 0.0
    specificity = tn / n_neg if n_neg else 0.0
    precision   = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if (precision + sensitivity) else 0.0
    )

    rank_only_passes  = sum(1 for r in positive_results
                            if r.get("rank_pass") and not r.get("score_pass") and r["pass"])
    score_only_passes = sum(1 for r in positive_results
                            if r.get("score_pass") and not r.get("rank_pass") and r["pass"])
    both_passes       = sum(1 for r in positive_results
                            if r.get("rank_pass") and r.get("score_pass") and r["pass"])

    all_results = positive_results + negative_results
    stratified  = compute_stratified_metrics(all_results)

    baseline_delta: Dict = {}
    if baseline_comparison.get("baselines"):
        for bl_name, bl_m in baseline_comparison["baselines"].items():
            baseline_delta[bl_name] = {
                "main_f1":     round(f1, 4),
                "baseline_f1": bl_m["f1"],
                "f1_delta":    round(f1 - bl_m["f1"], 4),
                "main_better": f1 > bl_m["f1"],
            }

    end_utc = datetime.now(timezone.utc)
    elapsed = (end_utc - start_utc).total_seconds()

    # ── Failed case summary ───────────────────────────────────────────────────
    failed_positive = [r for r in positive_results if not r["pass"]]
    failed_negative = [r for r in negative_results if not r["pass"]]

    logger.info("\n" + "=" * 70)
    logger.info("VALIDATION SUMMARY [%s]", mode_label)
    logger.info("=" * 70)
    logger.info("  True Positives:  %d/%d (%.1f%%)", tp, n_pos, sensitivity * 100)
    logger.info("  True Negatives:  %d/%d (%.1f%%)", tn, n_neg, specificity * 100)
    logger.info("  Sensitivity:     %.3f", sensitivity)
    logger.info("  Specificity:     %.3f", specificity)
    logger.info("  Precision:       %.3f", precision)
    logger.info("  F1:              %.3f", f1)
    logger.info("  Elapsed:         %.1fs (%.1f min)", elapsed, elapsed / 60)
    logger.info("  Pass breakdown:  rank-only=%d  score-only=%d  both=%d",
                rank_only_passes, score_only_passes, both_passes)

    if failed_positive:
        logger.info("\n  FAILED TRUE_POSITIVE cases:")
        for r in failed_positive:
            sc = r.get("score_components", {})
            logger.info(
                "    ✗ %-25s  score=%.3f  "
                "(gene=%.2f path=%.2f mech=%.2f ppi=%.2f)",
                f"{r['drug']} / {r['disease'][:30]}",
                r["raw_score"],
                sc.get("gene_score", 0), sc.get("pathway_score", 0),
                sc.get("mechanism_score", 0), sc.get("ppi_score", 0),
            )

    if failed_negative:
        logger.info("\n  FALSE POSITIVE cases (should score low but didn't):")
        for r in failed_negative:
            logger.info("    ✗ %s for %s  score=%.3f",
                        r["drug"], r["disease"], r["raw_score"])

    output = {
        "header": {
            "dataset_version":   DATASET_VERSION,
            "n_test_cases":      N_TEST_CASES,
            "n_positive_cases":  n_pos,
            "n_negative_cases":  n_neg,
            "run_timestamp_utc": start_utc.isoformat(),
            "elapsed_seconds":   round(elapsed, 2),
            "elapsed_minutes":   round(elapsed / 60, 1),
            "min_score_used":    min_score,
            "fast_mode":         fast_mode,
            "disease_filter":    disease_filter,
            "pass_criterion":    "rank_ok OR score_ok (v3.1)",
        },
        "pipeline_fingerprint":  fingerprint,
        "data_integrity":        integrity,
        "metrics": {
            "tp":                   tp,
            "fn":                   fn,
            "tn":                   tn,
            "fp":                   fp,
            "sensitivity":          round(sensitivity, 4),
            "specificity":          round(specificity, 4),
            "precision":            round(precision, 4),
            "f1":                   round(f1, 4),
            "rank_only_passes":     rank_only_passes,
            "score_only_passes":    score_only_passes,
            "both_criteria_passes": both_passes,
        },
        "stratified_metrics":    stratified,
        "baseline_comparison":   baseline_comparison,
        "baseline_f1_delta":     baseline_delta,
        "sensitivity_analysis":  sensitivity_result,
        "positive_results":      positive_results,
        "negative_results":      negative_results,
        "test_cases":            positive_results,
        "negative_cases":        negative_results,
        "out_of_scope_cases": [
            {
                "drug":            c["drug"],
                "disease":         c["disease"],
                "reason":          c["reason"],
                "version_removed": c["removed_in_version"],
            }
            for c in OUT_OF_SCOPE_CASES
        ],
    }

    out_path = Path(output_path)
    out_path.write_text(json.dumps(output, indent=2))
    logger.info("\nResults written to: %s", out_path.resolve())
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Run curated validation suite against the drug repurposing pipeline"
    )
    parser.add_argument("--min-score",        type=float, default=0.0,
                        help="Minimum score threshold")
    parser.add_argument("--output",           type=str,   default="validation_results.json",
                        help="Output JSON file path")
    parser.add_argument("--skip-baselines",   action="store_true",
                        help="Skip Jaccard/GeneCount/Cosine baseline comparison")
    parser.add_argument("--skip-sensitivity", action="store_true",
                        help="Skip weight sensitivity analysis")
    parser.add_argument("--fast",             action="store_true",
                        help=(
                            "Fast mode: skip PPI (STRING), EFO expansion, and drug "
                            "similarity scoring. Reduces runtime from ~60-90min to "
                            "~5-10min. Gene/pathway/mechanism scores are unaffected. "
                            "Recommended for iterative development."
                        ))
    parser.add_argument("--disease",          type=str, default=None,
                        help="Only run cases matching this disease substring (e.g. 'parkinson')")
    args = parser.parse_args()

    if args.fast:
        print("\n⚡ FAST MODE enabled — skipping PPI, EFO, and similarity scoring")
        print("   Expected runtime: 5-10 minutes\n")

    try:
        result  = asyncio.run(run_all_validations(
            min_score=args.min_score,
            output_path=args.output,
            run_baselines=not args.skip_baselines,
            run_sensitivity=not args.skip_sensitivity,
            fast_mode=args.fast,
            disease_filter=args.disease,
        ))
        metrics = result["metrics"]
        print("\n" + "=" * 50)
        print("FINAL METRICS")
        print("=" * 50)
        print(f"  Sensitivity: {metrics['sensitivity']:.3f}  ({metrics['tp']}/{metrics['tp']+metrics['fn']} TP)")
        print(f"  Specificity: {metrics['specificity']:.3f}  ({metrics['tn']}/{metrics['tn']+metrics['fp']} TN)")
        print(f"  Precision:   {metrics['precision']:.3f}")
        print(f"  F1:          {metrics['f1']:.3f}")
        print(f"  Time:        {result['header']['elapsed_minutes']:.1f} min")

        if result.get("baseline_f1_delta"):
            print("\nBASELINE F1 COMPARISON:")
            for bl_name, d in result["baseline_f1_delta"].items():
                better = "✓" if d["main_better"] else "✗"
                print(f"  {better} vs {bl_name:15s}: main={d['main_f1']:.3f}  "
                      f"baseline={d['baseline_f1']:.3f}  Δ={d['f1_delta']:+.3f}")
        print("=" * 50)

        sys.exit(0)
    except Exception as e:
        logger.error("Validation failed: %s", e)
        raise


if __name__ == "__main__":
    main()