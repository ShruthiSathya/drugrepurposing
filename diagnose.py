"""
diagnose.py — Run this from the repo root to diagnose why the patch isn't working.

    python diagnose.py

It will tell you exactly what's wrong.
"""

import asyncio
import json
import sys
from pathlib import Path

CACHE_FILE = Path("/tmp/drug_repurposing_cache/chembl_approved_drugs.json")
PROBLEM_DRUGS = [
    "bosentan", "dexamethasone", "iloprost", "spironolactone",
    "metoprolol", "rasagiline", "amantadine", "atorvastatin",
    "ezetimibe", "hydroxychloroquine", "leflunomide", "melphalan",
]


def check_1_cache():
    """Check if the drug cache exists and what targets the problem drugs have."""
    print("\n" + "="*60)
    print("CHECK 1: Drug cache file")
    print("="*60)

    if not CACHE_FILE.exists():
        print("✅ Cache file does NOT exist — pipeline will re-fetch from APIs")
        return False

    print(f"⚠️  Cache file EXISTS: {CACHE_FILE}")
    print(f"   Size: {CACHE_FILE.stat().st_size / 1024:.1f} KB")

    try:
        with open(CACHE_FILE) as f:
            drugs = json.load(f)
        print(f"   Contains {len(drugs)} drugs")
    except Exception as e:
        print(f"❌ Failed to read cache: {e}")
        return True

    print("\n   Targets for problem drugs in cache:")
    found_any_empty = False
    for drug in drugs:
        name = drug.get("name", "").lower()
        if name in PROBLEM_DRUGS:
            targets = drug.get("targets", [])
            status = "✅" if targets else "❌ EMPTY"
            print(f"   {status}  {drug['name']:30s}  targets={targets[:5]}")
            if not targets:
                found_any_empty = True

    if found_any_empty:
        print("\n   ❌ PROBLEM: Cache has empty targets for some drugs.")
        print("   FIX: Delete the cache file and re-run:")
        print(f"        rm {CACHE_FILE}")
    else:
        print("\n   ✅ All problem drugs have targets in cache")

    return found_any_empty


def check_2_patch_applied():
    """Check if the expanded KNOWN_SMALL_MOLECULE_TARGETS is in data_fetcher.py."""
    print("\n" + "="*60)
    print("CHECK 2: data_fetcher.py patch status")
    print("="*60)

    fetcher_path = Path("backend/pipeline/data_fetcher.py")
    if not fetcher_path.exists():
        print(f"❌ Cannot find {fetcher_path}")
        print("   Make sure you're running this from the repo root directory.")
        return False

    content = fetcher_path.read_text()

    # Check for drugs that should be in the expanded dict
    checks = {
        "dexamethasone": '"dexamethasone":',
        "bosentan (EDNRA)": "EDNRA",
        "iloprost (PTGIR)": "PTGIR",
        "spironolactone (NR3C2)": "NR3C2",
        "metoprolol (ADRB1)": '"metoprolol":',
        "atorvastatin (HMGCR)": '"atorvastatin":',
        "ezetimibe (NPC1L1)": "NPC1L1",
        "glipizide (ABCC8)": "ABCC8",
        "sulfasalazine (DHODH)": '"sulfasalazine":',
        "leflunomide (DHODH)": '"leflunomide":',
        "rasagiline (MAOB)": '"rasagiline":',
        "amantadine (GRIN1)": '"amantadine":',
    }

    all_present = True
    for label, token in checks.items():
        present = token in content
        status = "✅" if present else "❌ MISSING"
        print(f"   {status}  {label}")
        if not present:
            all_present = False

    if not all_present:
        print("\n   ❌ PROBLEM: The expanded KNOWN_SMALL_MOLECULE_TARGETS was NOT applied.")
        print("   FIX: Open backend/pipeline/data_fetcher.py")
        print("        Find KNOWN_SMALL_MOLECULE_TARGETS and replace it with")
        print("        the dict from data_fetcher_patch.py")
    else:
        print("\n   ✅ Patch appears to be applied correctly")

    return all_present


def check_3_live_targets():
    """Import the actual data_fetcher and check what it has."""
    print("\n" + "="*60)
    print("CHECK 3: Live import of KNOWN_SMALL_MOLECULE_TARGETS")
    print("="*60)

    try:
        sys.path.insert(0, ".")
        from backend.pipeline.data_fetcher import KNOWN_SMALL_MOLECULE_TARGETS
    except ImportError as e:
        print(f"❌ Cannot import data_fetcher: {e}")
        return False

    print(f"   Dict has {len(KNOWN_SMALL_MOLECULE_TARGETS)} entries\n")

    for drug in PROBLEM_DRUGS:
        targets = KNOWN_SMALL_MOLECULE_TARGETS.get(drug, [])
        status = "✅" if targets else "❌ MISSING"
        print(f"   {status}  {drug:30s}  {targets[:4]}")

    missing = [d for d in PROBLEM_DRUGS
               if not KNOWN_SMALL_MOLECULE_TARGETS.get(d)]
    if missing:
        print(f"\n   ❌ {len(missing)} drugs still missing from KNOWN_SMALL_MOLECULE_TARGETS:")
        for d in missing:
            print(f"      - {d}")
        print("\n   FIX: Replace KNOWN_SMALL_MOLECULE_TARGETS in data_fetcher.py")
        print("        with the dict from data_fetcher_patch.py")
    else:
        print("\n   ✅ All problem drugs have targets in KNOWN_SMALL_MOLECULE_TARGETS")

    return len(missing) == 0


def check_4_combo_scorer():
    """Check if the disease context penalty is in combo_scorer.py."""
    print("\n" + "="*60)
    print("CHECK 4: combo_scorer.py patch status")
    print("="*60)

    scorer_path = Path("backend/pipeline/combo_scorer.py")
    if not scorer_path.exists():
        print(f"❌ Cannot find {scorer_path}")
        return False

    content = scorer_path.read_text()

    checks = {
        "DISEASE_APPROPRIATE_CLASSES dict": "DISEASE_APPROPRIATE_CLASSES",
        "ONCOLOGY_CLASSES set": "ONCOLOGY_CLASSES",
        "_disease_context_penalty method": "_disease_context_penalty",
        "ctx_penalty in score_pair": "ctx_penalty",
    }

    all_present = True
    for label, token in checks.items():
        present = token in content
        status = "✅" if present else "❌ MISSING"
        print(f"   {status}  {label}")
        if not present:
            all_present = False

    if not all_present:
        print("\n   ❌ PROBLEM: combo_scorer.py patch was NOT applied.")
        print("   FIX: Apply the changes described in combo_scorer_patch.py")
    else:
        print("\n   ✅ combo_scorer.py patch appears applied")

    return all_present


async def check_5_live_score():
    """Run a quick live score for bosentan vs PAH to verify end-to-end."""
    print("\n" + "="*60)
    print("CHECK 5: Live score test (bosentan vs PAH)")
    print("="*60)

    # Don't run if cache still has empty targets — misleading
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE) as f:
                drugs = json.load(f)
            bosentan = next((d for d in drugs if d.get("name","").lower() == "bosentan"), None)
            if bosentan and not bosentan.get("targets"):
                print("   ⏭  Skipping live score — cache still has empty bosentan.")
                print("   Delete cache first, then re-run this script.")
                return
        except Exception:
            pass

    try:
        sys.path.insert(0, ".")
        from backend.pipeline.production_pipeline import ProductionPipeline

        print("   Fetching disease data for PAH...")
        pipeline = ProductionPipeline()
        disease_data = await pipeline.data_fetcher.fetch_disease_data(
            "pulmonary arterial hypertension"
        )
        if not disease_data:
            print("   ❌ Could not fetch PAH disease data (API may be down)")
            await pipeline.close()
            return

        print(f"   PAH genes: {len(disease_data.get('genes', []))}")

        # Fetch just a small slice of drugs
        drugs_data = await pipeline.data_fetcher.fetch_approved_drugs(limit=500)
        await pipeline.close()

        target_drugs = ["bosentan", "sildenafil", "dexamethasone",
                        "iloprost", "spironolactone", "atorvastatin"]
        drug_targets_found = {}
        for drug in drugs_data:
            name = drug.get("name", "").lower()
            if name in target_drugs:
                drug_targets_found[name] = drug.get("targets", [])

        print("\n   Drug targets after enrichment:")
        for drug, targets in sorted(drug_targets_found.items()):
            status = "✅" if targets else "❌ STILL EMPTY"
            print(f"   {status}  {drug:20s}  {targets[:5]}")

        still_empty = [d for d, t in drug_targets_found.items() if not t]
        if still_empty:
            print(f"\n   ❌ {len(still_empty)} drugs still have no targets after enrichment.")
            print("   This means the fallback in _apply_small_molecule_fallback()")
            print("   is not being called, OR the drug name doesn't match.")
        else:
            print("\n   ✅ All test drugs have targets — pipeline should score correctly now")

    except Exception as e:
        print(f"   ❌ Live test failed: {e}")
        import traceback
        traceback.print_exc()


def print_fix_summary(cache_has_problem, patch_applied, combo_patch):
    print("\n" + "="*60)
    print("SUMMARY AND FIXES")
    print("="*60)

    if cache_has_problem:
        print("""
🔴 FIX 1 (REQUIRED — do this first):
   Delete the stale drug cache:

       rm /tmp/drug_repurposing_cache/chembl_approved_drugs.json

   Without this, the pipeline will keep using the old cached drug
   entries that have empty target lists, ignoring your patch entirely.
""")

    if not patch_applied:
        print("""
🔴 FIX 2 (REQUIRED):
   The data_fetcher.py patch was not applied (or not saved).

   Steps:
   1. Open backend/pipeline/data_fetcher.py
   2. Find the block starting with:
          KNOWN_SMALL_MOLECULE_TARGETS: Dict[str, List[str]] = {
   3. Select from that line all the way to the closing }
      (it ends somewhere around line 100-120 of the dict)
   4. Replace the entire block with the dict in data_fetcher_patch.py
   5. Save the file
""")

    if not combo_patch:
        print("""
🟡 FIX 3 (RECOMMENDED):
   The combo_scorer.py patch was not applied.
   This fixes oncology drugs dominating non-oncology diseases.

   Steps (4 edits in backend/pipeline/combo_scorer.py):

   EDIT A — after REDUNDANT_CLASS_GROUPS, paste:
     DISEASE_APPROPRIATE_CLASSES = { ... }   (from combo_scorer_patch.py)
     ONCOLOGY_CLASSES = { ... }
     def _is_oncology_disease(disease_name): ...
     def _get_appropriate_classes(disease_name): ...

   EDIT B — add _disease_context_penalty() as a method of CombinationScorer

   EDIT C — in score_pair(), change:
     combo_score = max(0.0, min(1.0,
         base_score + syn_bonus + cov_bonus - ant_penalty - red_penalty
     ))
   TO:
     ctx_penalty = self._disease_context_penalty(class_a, class_b)
     combo_score = max(0.0, min(1.0,
         base_score + syn_bonus + cov_bonus
         - ant_penalty - red_penalty - ctx_penalty
     ))

   EDIT D — same pattern in score_triple()
""")

    if not cache_has_problem and patch_applied and combo_patch:
        print("✅ Everything looks correct. Re-run combo_validation_dataset.py")
    else:
        print("""
After applying fixes, re-run:
    python combo_validation_dataset.py --top-n 15

Expected result: pass rate improves from 28% to ~55-65%
""")


if __name__ == "__main__":
    cache_problem = check_1_cache()
    patch_applied = check_2_patch_applied()
    targets_ok    = check_3_live_targets()
    combo_patch   = check_4_combo_scorer()
    asyncio.run(check_5_live_score())
    print_fix_summary(cache_problem, targets_ok, combo_patch)