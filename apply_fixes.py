#!/usr/bin/env python3
"""
apply_fixes.py — Apply all pipeline fixes and validate they work
================================================================
Run this script from the repo root AFTER copying the fixed files:

    cp fixed_pipeline/data_fetcher.py backend/pipeline/data_fetcher.py
    cp fixed_pipeline/combo_scorer.py backend/pipeline/combo_scorer.py
    cp fixed_pipeline/insilico_trial.py backend/pipeline/insilico_trial.py
    python apply_fixes.py

This script:
1. Deletes stale drug and trial caches
2. Verifies the key drugs now have targets in KNOWN_SMALL_MOLECULE_TARGETS
3. Verifies the disease context penalty correctly kills oncology combos for non-oncology
4. Runs a quick smoke test on a few key drug-disease pairs
"""

import sys
import json
import asyncio
from pathlib import Path

CACHE_DIR = Path("/tmp/drug_repurposing_cache")

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Delete stale caches
# ─────────────────────────────────────────────────────────────────────────────
def clear_caches():
    print("\n" + "="*60)
    print("STEP 1: Clearing stale caches")
    print("="*60)

    files_to_delete = [
        CACHE_DIR / "chembl_approved_drugs.json",
        CACHE_DIR / "insilico_trial_cache.json",
        # Keep disease cache — those are fine
    ]

    for f in files_to_delete:
        if f.exists():
            f.unlink()
            print(f"  ✅ Deleted: {f}")
        else:
            print(f"  ⏭  Not found (OK): {f}")

    print("  Caches cleared. Next pipeline run will re-fetch drug data.")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Verify KNOWN_SMALL_MOLECULE_TARGETS
# ─────────────────────────────────────────────────────────────────────────────
def verify_targets():
    print("\n" + "="*60)
    print("STEP 2: Verifying target annotations")
    print("="*60)

    sys.path.insert(0, ".")
    from backend.pipeline.data_fetcher import KNOWN_SMALL_MOLECULE_TARGETS

    required = {
        # PAH drugs
        "bosentan":         ["EDNRA", "EDNRB"],
        "iloprost":         ["PTGIR"],
        "sildenafil":       ["PDE5A"],
        # Myeloma
        "dexamethasone":    ["NR3C1"],
        "bortezomib":       ["PSMB5"],
        "thalidomide":      ["CRBN"],
        "melphalan":        ["MGMT"],
        # RA
        "hydroxychloroquine": ["TLR7", "TLR9"],
        "sulfasalazine":    ["DHODH"],
        "leflunomide":      ["DHODH"],
        # Heart failure
        "spironolactone":   ["NR3C2"],
        "metoprolol":       ["ADRB1"],
        # Alzheimer
        "memantine":        ["GRIN1"],
        "donepezil":        ["ACHE"],
        # Parkinson
        "rasagiline":       ["MAOB"],
        "amantadine":       ["GRIN1"],
        # T2DM
        "glipizide":        ["ABCC8"],
        "pioglitazone":     ["PPARG"],
        # Hypercholesterolemia
        "atorvastatin":     ["HMGCR"],
        "ezetimibe":        ["NPC1L1"],
        # Gout
        "allopurinol":      ["XDH"],
        "colchicine":       ["TUBB", "NLRP3"],
    }

    all_ok = True
    for drug, expected_targets in required.items():
        actual = KNOWN_SMALL_MOLECULE_TARGETS.get(drug, [])
        missing = [t for t in expected_targets if t not in actual]
        if missing:
            print(f"  ❌ {drug:30s}  missing targets: {missing}")
            all_ok = False
        else:
            print(f"  ✅ {drug:30s}  targets: {actual[:4]}")

    if all_ok:
        print("\n  ✅ All required targets present!")
    else:
        print("\n  ❌ Some targets missing — re-check data_fetcher.py")

    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Verify disease context penalties
# ─────────────────────────────────────────────────────────────────────────────
def verify_context_penalties():
    print("\n" + "="*60)
    print("STEP 3: Verifying disease context penalties")
    print("="*60)

    from backend.pipeline.combo_scorer import CombinationScorer

    # These oncology combos should score NEAR ZERO for non-oncology diseases
    bad_combos = [
        ("cisplatin",   "doxorubicin",  "polycystic ovary syndrome"),
        ("cisplatin",   "gemcitabine",  "gout"),
        ("paclitaxel",  "cisplatin",    "alzheimer disease"),
        ("doxorubicin", "gemcitabine",  "heart failure"),
        ("cisplatin",   "metformin",    "polycystic ovary syndrome"),  # cisplatin should kill this
    ]

    # These disease-appropriate combos should score HIGH
    good_combos = [
        ("sildenafil",    "bosentan",       "pulmonary arterial hypertension"),
        ("metformin",     "pioglitazone",   "type 2 diabetes mellitus"),
        ("metformin",     "spironolactone", "polycystic ovary syndrome"),
        ("colchicine",    "allopurinol",    "gout"),
        ("donepezil",     "memantine",      "alzheimer disease"),
        ("thalidomide",   "dexamethasone",  "multiple myeloma"),
        ("bortezomib",    "dexamethasone",  "multiple myeloma"),
        ("methotrexate",  "hydroxychloroquine", "rheumatoid arthritis"),
    ]

    all_ok = True

    print("\n  Bad combos (should score < 0.25):")
    for drug_a, drug_b, disease in bad_combos:
        scorer = CombinationScorer(disease_name=disease)
        da = {"drug_name": drug_a, "name": drug_a, "score": 0.6, "mechanism": drug_a, "targets": []}
        db = {"drug_name": drug_b, "name": drug_b, "score": 0.6, "mechanism": drug_b, "targets": []}
        result = scorer.score_pair(da, db, [])
        score = result["combo_score"]
        ctx   = result.get("context_penalty", 0)
        ok = score < 0.25
        status = "✅" if ok else "❌"
        print(f"  {status}  {drug_a:15s} + {drug_b:15s} for {disease:40s}  score={score:.3f}  ctx_pen={ctx:.3f}")
        if not ok:
            all_ok = False

    print("\n  Good combos (should score > 0.30):")
    for drug_a, drug_b, disease in good_combos:
        scorer = CombinationScorer(disease_name=disease)
        da = {"drug_name": drug_a, "name": drug_a, "score": 0.55, "mechanism": drug_a, "targets": []}
        db = {"drug_name": drug_b, "name": drug_b, "score": 0.50, "mechanism": drug_b, "targets": []}
        result = scorer.score_pair(da, db, [])
        score = result["combo_score"]
        syn   = result.get("is_synergistic", False)
        ok = score > 0.30
        status = "✅" if ok else "❌"
        syn_str = "SYNERGISTIC" if syn else ""
        print(f"  {status}  {drug_a:15s} + {drug_b:15s} for {disease:40s}  score={score:.3f}  {syn_str}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\n  ✅ All context penalties working correctly!")
    else:
        print("\n  ❌ Some combos mis-scored — re-check combo_scorer.py")

    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Verify priority thresholds
# ─────────────────────────────────────────────────────────────────────────────
def verify_priority_thresholds():
    print("\n" + "="*60)
    print("STEP 4: Verifying priority thresholds")
    print("="*60)

    from backend.pipeline.insilico_trial import InSilicoTrialSimulator

    # Manually call _compute_trial_summary with mocked outcomes
    import random
    from backend.pipeline.insilico_trial import PatientOutcome, PKPDProfile

    sim = InSilicoTrialSimulator(disease="pulmonary arterial hypertension")

    # Mock outcomes with ~40% ORR (should be HIGH)
    rng = random.Random(42)
    outcomes = []
    for i in range(200):
        r = rng.random()
        if r < 0.15:
            resp = "CR"
        elif r < 0.40:
            resp = "PR"
        elif r < 0.65:
            resp = "SD"
        else:
            resp = "PD"
        outcomes.append(PatientOutcome(
            patient_id=i, recist_response=resp,
            tumor_reduction=rng.uniform(0, 80) if resp in ("CR","PR") else rng.uniform(0, 20),
            pfs_weeks=rng.uniform(8, 48),
            treatment_stopped=False, biomarkers={}
        ))

    pkpd = PKPDProfile(0.6, 8.0, 0.5, 0.6, 0.7, 0.5)
    candidate = {"drug_name": "test_combo", "score": 0.65}
    result = sim._compute_trial_summary(outcomes, candidate, pkpd)

    orr = result["orr"]
    p2  = result["phase2_success_probability"]
    pri = result["priority"]
    print(f"  Mock trial (40% ORR): ORR={orr:.1%}  P2={p2:.2f}  Priority={pri}")

    ok = pri in ("HIGH", "MEDIUM")
    if ok:
        print("  ✅ Priority threshold correct — not FAILED for reasonable ORR")
    else:
        print("  ❌ Priority should be HIGH or MEDIUM for 40% ORR, got:", pri)

    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("TwinTrial Pipeline Fix Validator")
    print("="*60)

    clear_caches()

    results = []
    results.append(("Target annotations",   verify_targets()))
    results.append(("Context penalties",    verify_context_penalties()))
    results.append(("Priority thresholds",  verify_priority_thresholds()))

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    all_passed = True
    for name, ok in results:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  {name}")
        if not ok:
            all_passed = False

    if all_passed:
        print("""
✅ ALL CHECKS PASSED

Next steps:
  1. Run the combo validation:
       python combo_validation_dataset.py --top-n 15
  2. Expected pass rate: ~60-75% (up from 24%)
  3. Key improvements:
     - bosentan, iloprost, dexamethasone, spironolactone: now score correctly
     - Oncology drugs no longer dominate non-oncology top combos
     - Priorities will be HIGH/MEDIUM/LOW (not FAILED) for good combos
""")
    else:
        print("""
❌ SOME CHECKS FAILED

Make sure you copied all three files:
  cp fixed_pipeline/data_fetcher.py  backend/pipeline/data_fetcher.py
  cp fixed_pipeline/combo_scorer.py  backend/pipeline/combo_scorer.py
  cp fixed_pipeline/insilico_trial.py backend/pipeline/insilico_trial.py

Then re-run: python apply_fixes.py
""")
    sys.exit(0 if all_passed else 1)