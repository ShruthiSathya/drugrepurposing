#!/usr/bin/env python3
"""
deploy_transcriptomics_upgrade.py
===================================
Deployment script for the LiveDRP v5.0 transcriptomics upgrade.

WHAT THIS DOES
--------------
1. Backs up existing production_pipeline.py
2. Copies the two new files into backend/pipeline/
3. Runs offline unit tests for all four new capabilities
4. Prints integration instructions

Run from repo root:
    python deploy_transcriptomics_upgrade.py
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).parent
PIPELINE_DIR = REPO_ROOT / "backend" / "pipeline"
BACKUP_DIR = PIPELINE_DIR / "_backup_v4"


def step(msg: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {msg}")
    print('─'*60)


def check(ok: bool, label: str) -> None:
    print(f"  {'✅' if ok else '❌'} {label}")
    if not ok:
        sys.exit(1)


def main():
    print("=" * 60)
    print("  LiveDRP v5.0 — Transcriptomics Upgrade Deployment")
    print("=" * 60)

    # ── Step 1: Validate repo structure ──────────────────────────────
    step("1/5  Validating repository structure")
    check(PIPELINE_DIR.exists(), f"Pipeline dir exists: {PIPELINE_DIR}")
    check((PIPELINE_DIR / "scorer.py").exists(), "scorer.py present")
    check((PIPELINE_DIR / "combo_scorer.py").exists(), "combo_scorer.py present")
    check((PIPELINE_DIR / "data_fetcher.py").exists(), "data_fetcher.py present")

    # ── Step 2: Backup existing files ────────────────────────────────
    step("2/5  Backing up existing production_pipeline.py")
    BACKUP_DIR.mkdir(exist_ok=True)
    src = PIPELINE_DIR / "production_pipeline.py"
    if src.exists():
        dst = BACKUP_DIR / "production_pipeline_v4.py.bak"
        shutil.copy2(src, dst)
        print(f"  Backed up to: {dst}")
    else:
        print("  No existing production_pipeline.py found (fresh install)")

    # ── Step 3: Deploy new files ──────────────────────────────────────
    step("3/5  Deploying new files to backend/pipeline/")

    files_to_deploy = [
        ("transcriptomics_engine.py", "transcriptomics_engine.py"),
        ("production_pipeline.py", "production_pipeline.py"),
    ]

    upgrade_dir = REPO_ROOT / "upgrade_files"
    if not upgrade_dir.exists():
        # Files are next to this script
        upgrade_dir = REPO_ROOT

    for src_name, dst_name in files_to_deploy:
        src_path = upgrade_dir / src_name
        dst_path = PIPELINE_DIR / dst_name
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
            print(f"  ✅ Deployed: {dst_name}")
        else:
            print(f"  ⚠️  Source not found: {src_path}")
            print(f"      Please manually copy {src_name} to {dst_path}")

    # ── Step 4: Run offline unit tests ───────────────────────────────
    step("4/5  Running offline unit tests")

    test_code = '''
import sys
sys.path.insert(0, ".")

FAILURES = []

def check(ok, label, detail=""):
    status = "✅" if ok else "❌"
    print(f"  {status} {label}" + (f" ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(label)

# ── Test 1: TranscriptomicSignature (offline — no API) ────────────────
print("\\n[1] TranscriptomicSignature structure")
try:
    from backend.pipeline.transcriptomics_engine import TranscriptomicSignature, _resolve_tissues
    sig = TranscriptomicSignature("parkinson disease")
    tissues = _resolve_tissues("parkinson disease")
    check("substantia nigra" in tissues, "Parkinson disease maps to substantia nigra")
    check("alzheimer" not in " ".join(tissues).lower(), "No cross-contamination")

    tissues2 = _resolve_tissues("pulmonary arterial hypertension")
    check("lung" in tissues2, "PAH maps to lung")
    check(len(tissues2) > 0, "PAH tissue list non-empty")

    tissues3 = _resolve_tissues("gout")
    check("kidney" in tissues3, "Gout maps to kidney")
    print("  OK: Tissue mapping works for all tested diseases")
except Exception as e:
    print(f"  ❌ ERROR: {e}")
    FAILURES.append(f"TxnSig import: {e}")

# ── Test 2: TissueExpressionGate (offline) ────────────────────────────
print("\\n[2] TissueExpressionGate multiplier logic")
try:
    from backend.pipeline.transcriptomics_engine import TissueExpressionGate

    gate = TissueExpressionGate("parkinson disease")
    check(gate.compute_gate_multiplier(0.0) == 0.30, "gate_score=0.0 → multiplier=0.30")
    check(gate.compute_gate_multiplier(1.0) == 1.00, "gate_score=1.0 → multiplier=1.00")
    check(abs(gate.compute_gate_multiplier(0.5) - 0.65) < 0.01, "gate_score=0.5 → ~0.65")

    # Simulate drug with good tissue expression
    import asyncio
    txn_sig_mock = {
        "upregulated": {"SNCA": 0.8, "LRRK2": 0.75, "DRD2": 0.65},
        "tissue_anchored": {"MAOB": 0.9, "TH": 0.85},
        "downregulated": {},
    }
    score, detail = asyncio.get_event_loop().run_until_complete(
        gate.score_drug("rasagiline", ["MAOB", "CYP1A2"], txn_signature=txn_sig_mock)
    )
    check(score > 0.5, f"rasagiline gate_score > 0.5 (MAOB expressed)", f"got {score:.3f}")
    check(detail.get("multiplier", 0) >= 0.65, "multiplier >= 0.65 for expressed drug")

    # Drug with no tissue expression → neutral, not zero
    score2, detail2 = asyncio.get_event_loop().run_until_complete(
        gate.score_drug("metformin", ["PRKAA1", "PRKAA2"], txn_signature=txn_sig_mock)
    )
    check(score2 >= 0.30, f"metformin gate_score neutral (not zero) in PD", f"got {score2:.3f}")
    print("  OK: Gate multipliers work correctly, no drug gets zeroed out")
except Exception as e:
    print(f"  ❌ ERROR: {e}")
    import traceback; traceback.print_exc()
    FAILURES.append(f"TissueGate: {e}")

# ── Test 3: DysregulationPathwayScorer ────────────────────────────────
print("\\n[3] DysregulationPathwayScorer")
try:
    from backend.pipeline.transcriptomics_engine import DysregulationPathwayScorer

    scorer = DysregulationPathwayScorer("parkinson disease")

    txn_mock = {
        "upregulated":  {"SNCA": 0.8, "LRRK2": 0.75},
        "downregulated": {"TH": 0.4},
        "tissue_anchored": {"MAOB": 0.9},
    }
    pathways = [
        "Alpha-synuclein aggregation",
        "Dopamine metabolism",
        "Mitophagy",
        "Insulin signaling pathway",
    ]
    dysreg = scorer.identify_dysregulated_pathways(txn_mock, pathways)
    check(isinstance(dysreg["overactive"], list), "identify_dysregulated_pathways returns lists")
    check("overactive" in dysreg and "silenced" in dysreg, "Both overactive/silenced keys present")

    # Test combo repair scoring
    combo = {
        "combo_name": "RASAGILINE + PRAMIPEXOLE",
        "combo_score": 0.75,
        "shared_pathways": ["Dopamine metabolism", "Alpha-synuclein aggregation"],
        "mechanism_a": "maob_inhibitor",
        "mechanism_b": "dopamine_agonist",
    }
    repair = scorer.score_combo_pathway_repair(combo, dysreg)
    check(0.0 <= repair <= 1.0, f"repair score in [0,1]: {repair:.3f}")

    # Rescore list
    combos = [combo]
    rescored = scorer.rescore_combos(combos, txn_mock, pathways, repair_weight=0.20)
    check(len(rescored) == 1, "rescored list has same length")
    check("txn_rescored_combo_score" in rescored[0], "txn_rescored_combo_score added")
    check("pathway_repair_score" in rescored[0], "pathway_repair_score added")
    print("  OK: DysregulationPathwayScorer works correctly")
except Exception as e:
    print(f"  ❌ ERROR: {e}")
    import traceback; traceback.print_exc()
    FAILURES.append(f"DysregulationPathwayScorer: {e}")

# ── Test 4: TranscriptomicSubtypeSimulator ────────────────────────────
print("\\n[4] TranscriptomicSubtypeSimulator")
try:
    from backend.pipeline.transcriptomics_engine import TranscriptomicSubtypeSimulator

    sim = TranscriptomicSubtypeSimulator()

    txn_mock = {
        "disease": "parkinson disease",
        "upregulated":  {"SNCA": 0.9, "LRRK2": 0.7},
        "downregulated": {},
        "tissue_anchored": {"MAOB": 0.85, "TH": 0.8, "MAPT": 0.3},
        "n_extended": 25,
    }
    subtypes = sim.classify_disease_subtypes(txn_mock)
    check(isinstance(subtypes, dict), "classify_disease_subtypes returns dict")
    total = sum(subtypes.values())
    check(abs(total - 1.0) < 0.01, f"subtype fractions sum to 1.0: {total:.3f}")
    check("mixed" in subtypes, "mixed subtype present")

    combos = [
        {
            "combo_name": "RASAGILINE + PRAMIPEXOLE",
            "combo_score": 0.80,
            "txn_rescored_combo_score": 0.82,
            "mechanism_a": "maob_inhibitor",
            "mechanism_b": "dopamine_agonist",
        },
        {
            "combo_name": "RASAGILINE + AMANTADINE",
            "combo_score": 0.72,
            "txn_rescored_combo_score": 0.74,
            "mechanism_a": "maob_inhibitor",
            "mechanism_b": "nmda_antagonist",
        },
    ]
    result = sim.simulate_subtype_trials(combos, txn_mock, n_top_combos=2)
    check("subtype_results" in result, "subtype_results key present")
    check("overall_population_orr" in result, "overall_population_orr present")
    check(0.0 < result["overall_population_orr"] < 1.0,
          f"overall ORR in range: {result['overall_population_orr']:.3f}")
    check("recommended_combo_overall" in result, "recommended_combo present")
    check("precision_medicine_note" in result, "precision_medicine_note present")
    print(f"  Subtypes: {dict((k, f\'{v:.1%}\') for k,v in subtypes.items())}")
    print(f"  Overall ORR: {result['overall_population_orr']:.1%}")
    print(f"  Recommended: {result['recommended_combo_overall']}")
    print("  OK: TranscriptomicSubtypeSimulator works correctly")
except Exception as e:
    print(f"  ❌ ERROR: {e}")
    import traceback; traceback.print_exc()
    FAILURES.append(f"SubtypeSimulator: {e}")

# ── Test 5: Production pipeline import ───────────────────────────────
print("\\n[5] ProductionPipeline v5.0 import")
try:
    from backend.pipeline.production_pipeline import ProductionPipeline
    p = ProductionPipeline(enable_transcriptomics=True)
    check(p.enable_transcriptomics is True, "enable_transcriptomics=True")
    p2 = ProductionPipeline(enable_transcriptomics=False)
    check(p2.enable_transcriptomics is False, "enable_transcriptomics=False (opt-out)")
    print("  OK: ProductionPipeline v5.0 imports correctly")
except Exception as e:
    print(f"  ❌ ERROR: {e}")
    FAILURES.append(f"ProductionPipeline import: {e}")

# ── Summary ────────────────────────────────────────────────────────────
print("\\n" + "=" * 60)
if FAILURES:
    print(f"  ❌ FAILED: {len(FAILURES)} test(s)")
    for f in FAILURES:
        print(f"    • {f}")
    sys.exit(1)
else:
    print("  ✅ ALL OFFLINE TESTS PASSED")
print("=" * 60)
'''

    # Write and run the test
    test_file = REPO_ROOT / "_test_txn_upgrade.py"
    test_file.write_text(test_code)
    result = subprocess.run(
        [sys.executable, str(test_file)],
        capture_output=False,
        cwd=str(REPO_ROOT),
    )
    test_file.unlink()  # clean up

    if result.returncode != 0:
        print("\n  ❌ Unit tests failed. Deployment aborted.")
        sys.exit(1)

    # ── Step 5: Print integration instructions ────────────────────────
    step("5/5  Integration instructions")
    print("""
  Files deployed:
    backend/pipeline/transcriptomics_engine.py   (NEW — 4 capability classes)
    backend/pipeline/production_pipeline.py      (UPDATED — v5.0)

  ─────────────────────────────────────────────────────────────
  HOW TO USE
  ─────────────────────────────────────────────────────────────

  Default (full transcriptomics ON):
    pipeline = ProductionPipeline()                 # enable_transcriptomics=True
    plan = await pipeline.generate_treatment_plan("parkinson disease")

  Opt out of transcriptomics (v4.1 behaviour):
    pipeline = ProductionPipeline(enable_transcriptomics=False)

  Disable tissue gate only:
    plan = await pipeline.generate_treatment_plan(
        "parkinson disease",
        use_tissue=False          # skips all transcriptomics
    )

  ─────────────────────────────────────────────────────────────
  NEW FIELDS IN TREATMENT PLAN RESPONSE
  ─────────────────────────────────────────────────────────────

  plan["transcriptomics_context"]
    .signature_summary
      .n_upregulated            # genes upregulated + expressed in disease tissue
      .n_tissue_anchored        # genes strongly expressed even without OT proof
      .n_extended_genes_added   # new genes added to disease gene set
      .target_tissues           # tissues checked for expression

    .subtype_simulation
      .subtype_fractions         # {subtype: fraction_of_patients}
      .subtype_results           # per-subtype ORR + best combo
      .overall_population_orr    # weighted population average
      .recommended_combo_overall # best combo for majority subtype
      .precision_medicine_note   # plain-language stratification insight

  plan["ranked_regimens"][N]
    .pathway_repair_score        # 0-1: how well combo fixes broken pathways
    .txn_rescored_combo_score    # combo_score blended with pathway repair signal
    .dysregulation_context
      .overactive_pathways_count
      .silenced_pathways_count

  plan["candidates"][N]
    .tissue_gate_score           # 0-1: how well drug targets expressed in tissue
    .tissue_gate_multiplier      # 0.30-1.00: score multiplier applied
    .tissue_gated_score          # score after tissue gating
    .score_before_tissue_gate    # original pre-gate score

  ─────────────────────────────────────────────────────────────
  VALIDATION
  ─────────────────────────────────────────────────────────────

  Run smoke test (5 diseases, ~2 min):
    python smoke_test.py

  Verify combo validation still passes (target: 24/24 = 100%):
    python combo_validation_dataset.py --top-n 15

  Check transcriptomics doesn't regress single-drug F1:
    python run_validation.py --disease "parkinson"
    python run_validation.py --disease "pulmonary arterial hypertension"

  Full validation:
    python run_validation.py --output validation_results_v5.json

  ─────────────────────────────────────────────────────────────
  EXPECTED IMPROVEMENTS
  ─────────────────────────────────────────────────────────────

  1. Parkinson disease — rasagiline/amantadine should improve:
     - MAOB is highly expressed in substantia nigra → gate multiplier ~0.95
     - Dopamine metabolism pathway overactive → repair score boosts combo

  2. Heart failure — spiro+metoprolol should improve:
     - Cardiac genes well-expressed in heart muscle
     - Neurohormonal pathways identified as overactive

  3. Novel candidates may emerge for diseases with sparse OT annotations:
     - Tissue-anchored genes extend the target set
     - Drugs hitting those targets now score higher

  4. Patient stratification available for all diseases:
     - Subtype fractions reveal heterogeneity level
     - Use this to pitch "precision repurposing" to PAGs

  ─────────────────────────────────────────────────────────────
""")

    print("  ✅ Deployment complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()