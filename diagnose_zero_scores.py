"""
diagnose_zero_scores.py
=======================
Diagnostic script to understand exactly WHY specific drugs score 0.0.
Checks each stage of the pipeline for the failing drugs.

Run with:
    python diagnose_zero_scores.py

Output tells you:
  1. Is the drug in the pool at all?
  2. Does it have targets annotated?
  3. Do those targets overlap with the disease genes?
  4. Is it being filtered by safety filter?
  5. What does OpenTargets return for disease genes?
"""

import asyncio
import logging
from typing import Dict, List

logging.basicConfig(level=logging.WARNING)  # suppress pipeline noise

# Drugs scoring 0.0 that should score well
DIAGNOSE_CASES = [
    {"drug": "Bosentan",          "disease": "pulmonary arterial hypertension"},
    {"drug": "Iloprost",          "disease": "pulmonary arterial hypertension"},
    {"drug": "Dexamethasone",     "disease": "multiple myeloma"},
    {"drug": "Bortezomib",        "disease": "multiple myeloma"},
    {"drug": "Hydroxychloroquine","disease": "rheumatoid arthritis"},
    {"drug": "Spironolactone",    "disease": "heart failure"},
    {"drug": "Metoprolol",        "disease": "heart failure"},
    {"drug": "Atorvastatin",      "disease": "hypercholesterolemia"},
    {"drug": "Ezetimibe",         "disease": "hypercholesterolemia"},
    {"drug": "Rasagiline",        "disease": "parkinson disease"},
    {"drug": "Amantadine",        "disease": "parkinson disease"},
]

async def diagnose():
    from backend.pipeline.production_pipeline import ProductionPipeline
    from backend.pipeline.drug_filter import DrugSafetyFilter

    pipeline = ProductionPipeline()

    print("=" * 70)
    print("DRUG POOL DIAGNOSTIC")
    print("=" * 70)

    try:
        # Fetch all drugs
        print("\n[1] Fetching drug pool...")
        all_drugs = await pipeline.fetch_approved_drugs(limit=3000)
        generic_drugs, _, _ = await pipeline.fetch_generic_drugs(limit=3000)

        drug_lookup_all     = {d["name"].lower(): d for d in all_drugs}
        drug_lookup_generic = {d["name"].lower(): d for d in generic_drugs}

        print(f"    Total drugs: {len(all_drugs)}")
        print(f"    Generic pool: {len(generic_drugs)}")

        # Check each failing drug
        for case in DIAGNOSE_CASES:
            drug_name = case["drug"]
            disease   = case["disease"]
            drug_lower = drug_name.lower()

            print(f"\n{'─'*60}")
            print(f"DRUG: {drug_name.upper()}  |  DISEASE: {disease}")
            print(f"{'─'*60}")

            # Stage 1: Is it in the pool at all?
            in_all     = drug_lower in drug_lookup_all
            in_generic = drug_lower in drug_lookup_generic

            # Check for salt variants
            salt_match_all = None
            salt_match_generic = None
            for name in drug_lookup_all:
                if drug_lower in name or name in drug_lower:
                    salt_match_all = name
                    break
            for name in drug_lookup_generic:
                if drug_lower in name or name in drug_lower:
                    salt_match_generic = name
                    break

            if in_all:
                print(f"  ✓ In full drug pool (exact match)")
            elif salt_match_all:
                print(f"  ~ In full drug pool as: '{salt_match_all}'")
                drug_lower = salt_match_all
                in_all = True
            else:
                print(f"  ✗ NOT IN DRUG POOL — drug is absent entirely")
                print(f"    → Fix: Add to ESSENTIAL_DRUGS with correct ChEMBL ID")
                print(f"    → Check: https://www.ebi.ac.uk/chembl/compound_report_card/search/{drug_name}/")
                continue

            if in_generic:
                print(f"  ✓ In generic pool")
            elif salt_match_generic:
                print(f"  ~ In generic pool as: '{salt_match_generic}'")
            else:
                print(f"  ✗ NOT in generic pool (filtered as patented)")
                drug_obj = drug_lookup_all.get(drug_lower, {})
                print(f"    first_approval_year: {drug_obj.get('first_approval_year')}")
                print(f"    patent_status: {drug_obj.get('patent_status')}")
                print(f"    patent_reason: {drug_obj.get('patent_reason')}")
                print(f"    → Fix: Add to KNOWN_APPROVAL_YEARS or CONFIRMED_GENERIC_OVERRIDES")

            # Stage 2: Does it have targets?
            drug_obj = drug_lookup_generic.get(drug_lower) or drug_lookup_all.get(drug_lower, {})
            targets = drug_obj.get("targets", [])
            target_source = drug_obj.get("target_source", "unknown")

            if targets:
                print(f"  ✓ Has {len(targets)} targets (source: {target_source})")
                print(f"    Targets: {targets[:10]}")
            else:
                print(f"  ✗ NO TARGETS annotated (source: {target_source})")
                print(f"    → Fix: Add to KNOWN_BIOLOGIC_TARGETS or KNOWN_SMALL_MOLECULE_TARGETS")
                print(f"    → Known targets for {drug_name}:")
                _print_known_targets(drug_name)

            # Stage 3: Do targets overlap with disease genes?
            disease_data = await pipeline.data_fetcher.fetch_disease_data(disease)
            if not disease_data:
                print(f"  ✗ Disease '{disease}' not found in OpenTargets")
                continue

            disease_genes = set(disease_data.get("genes", []))
            drug_targets  = set(targets)
            overlap       = drug_targets & disease_genes

            print(f"  Disease genes: {len(disease_genes)} total")
            if overlap:
                print(f"  ✓ Gene overlap: {sorted(overlap)}")
            else:
                print(f"  ✗ ZERO gene overlap with disease genes")
                print(f"    Drug targets:    {sorted(drug_targets)[:10]}")
                print(f"    Disease genes:   {sorted(list(disease_genes))[:15]}")
                print(f"    → This is a PPI proximity problem — targets are")
                print(f"      real but not in OpenTargets disease gene list")

            # Stage 4: Safety filter check
            safety = DrugSafetyFilter()
            safe, filtered = await safety.filter_candidates(
                candidates=[{"drug_name": drug_name, "mechanism": drug_obj.get("mechanism", ""), "score": 0.5}],
                disease_name=disease,
                remove_absolute=True,
                remove_relative=True,
            )
            if filtered:
                ci = filtered[0].get("contraindication", {})
                print(f"  ✗ FILTERED BY SAFETY: severity={ci.get('severity')} reason={ci.get('reason','')[:60]}")
            else:
                print(f"  ✓ Passes safety filter")

    finally:
        await pipeline.close()

    print(f"\n{'='*70}")
    print("SUMMARY: Fix priority order based on above")
    print("1. Drugs absent from pool → add ChEMBL IDs to ESSENTIAL_DRUGS")
    print("2. Drugs with no targets → add to KNOWN_SMALL_MOLECULE_TARGETS")
    print("3. Drugs filtered by safety → fix drug_filter.py")
    print("4. Drugs with zero overlap → need PPI proximity working (STRING)")
    print(f"{'='*70}")


def _print_known_targets(drug_name: str):
    """Print the known correct targets for each failing drug."""
    CORRECT_TARGETS = {
        "bosentan":           ["EDNRA", "EDNRB"],
        "iloprost":           ["PTGIR", "PTGDR2", "PTGER2"],
        "dexamethasone":      ["NR3C1", "NR3C2", "AR"],
        "bortezomib":         [
            "PSMB5", "PSMB1", "PSMB2", "PSMB3", "PSMB7",
            "PSMA1", "PSMA2", "PSMA3", "PSMA4", "PSMA5",
        ],
        "hydroxychloroquine": ["TLR7", "TLR9", "SQSTM1"],
        "spironolactone":     ["NR3C2", "AR", "NR3C1"],
        "metoprolol":         ["ADRB1", "ADRB2"],
        "atorvastatin":       ["HMGCR", "LDLR"],
        "ezetimibe":          ["NPC1L1"],
        "rasagiline":         ["MAOB", "MAOA"],
        "amantadine":         ["GRIN1", "GRIN2A", "GRIN2B", "SLC22A2"],
    }
    targets = CORRECT_TARGETS.get(drug_name.lower(), ["Unknown — check literature"])
    print(f"      Should be: {targets}")


if __name__ == "__main__":
    asyncio.run(diagnose())