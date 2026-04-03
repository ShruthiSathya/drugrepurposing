"""
check_drug_names_fast.py
========================
Verifies the salt-stripping fix and mechanism scoring WITHOUT any API calls.
Runs in ~5 seconds.

    python check_drug_names_fast.py
"""

import sys
sys.path.insert(0, ".")

# ── 1. Test salt stripping ────────────────────────────────────────────────────
from backend.pipeline.data_fetcher import (
    ProductionDataFetcher,
    KNOWN_SMALL_MOLECULE_TARGETS,
)

# These are actual names ChEMBL returns — with salt/form suffixes
CHEMBL_ACTUAL_NAMES = {
    "Sildenafil citrate":              "sildenafil",
    "Bosentan monohydrate":            "bosentan",
    "Iloprost":                        "iloprost",         # no suffix
    "Dexamethasone sodium phosphate":  "dexamethasone",
    "Metformin hydrochloride":         "metformin",
    "Spironolactone":                  "spironolactone",   # no suffix
    "Metoprolol tartrate":             "metoprolol",
    "Atorvastatin calcium":            "atorvastatin",
    "Rasagiline mesylate":             "rasagiline",
    "Amantadine hydrochloride":        "amantadine",
    "Memantine hydrochloride":         "memantine",
    "Donepezil hydrochloride":         "donepezil",
    "Hydroxychloroquine sulfate":      "hydroxychloroquine",
    "Leflunomide":                     "leflunomide",      # no suffix
    "Sulfasalazine":                   "sulfasalazine",    # no suffix
    "Ezetimibe":                       "ezetimibe",        # no suffix
    "Melphalan":                       "melphalan",        # no suffix
    "Colchicine":                      "colchicine",       # no suffix
    "Allopurinol":                     "allopurinol",      # no suffix
    "Aspirin":                         "aspirin",          # no suffix
    "Bortezomib":                      "bortezomib",       # no suffix
    "Thalidomide":                     "thalidomide",      # no suffix
}

print("=" * 65)
print("TEST 1: Salt stripping + fallback dict lookup")
print("=" * 65)
print(f"{'ChEMBL name':<40} {'Stripped':<22} {'In fallback?'}")
print("─" * 65)

all_pass = True
for chembl_name, expected_stripped in CHEMBL_ACTUAL_NAMES.items():
    stripped  = ProductionDataFetcher._strip_salt(chembl_name)
    in_dict   = stripped in KNOWN_SMALL_MOLECULE_TARGETS
    correct   = stripped == expected_stripped
    status    = "✓" if (in_dict and correct) else "✗"
    if not (in_dict and correct):
        all_pass = False
    print(f"{status} {chembl_name:<38} → {stripped:<20} {'✓ found' if in_dict else '✗ MISSING'}")

print()
if all_pass:
    print("✅ All salt stripping + fallback lookups PASS")
else:
    print("❌ Some lookups failed — check KNOWN_SMALL_MOLECULE_TARGETS keys")

# ── 2. Simulate the fallback on fake drug objects ─────────────────────────────
print()
print("=" * 65)
print("TEST 2: _apply_small_molecule_fallback with salt names")
print("=" * 65)

fetcher = ProductionDataFetcher.__new__(ProductionDataFetcher)
fetcher._pathway_mapper = None

fake_drugs = [
    {"name": "Sildenafil citrate",             "targets": [], "pathways": []},
    {"name": "Bosentan monohydrate",            "targets": [], "pathways": []},
    {"name": "Dexamethasone sodium phosphate",  "targets": [], "pathways": []},
    {"name": "Metformin hydrochloride",         "targets": [], "pathways": []},
    {"name": "Spironolactone",                  "targets": [], "pathways": []},
    {"name": "Metoprolol tartrate",             "targets": [], "pathways": []},
    {"name": "Atorvastatin calcium",            "targets": [], "pathways": []},
    {"name": "Rasagiline mesylate",             "targets": [], "pathways": []},
    {"name": "Amantadine hydrochloride",        "targets": [], "pathways": []},
    {"name": "Memantine hydrochloride",         "targets": [], "pathways": []},
    {"name": "Donepezil hydrochloride",         "targets": [], "pathways": []},
    {"name": "Hydroxychloroquine sulfate",      "targets": [], "pathways": []},
    {"name": "Ezetimibe",                       "targets": [], "pathways": []},
    {"name": "Bortezomib",                      "targets": [], "pathways": []},
    {"name": "Colchicine",                      "targets": [], "pathways": []},
    {"name": "Allopurinol",                     "targets": [], "pathways": []},
    {"name": "Aspirin",                         "targets": [], "pathways": []},
    {"name": "Thalidomide",                     "targets": [], "pathways": []},
]

result = fetcher._apply_small_molecule_fallback(fake_drugs)

print(f"{'Drug name':<40} {'Targets found'}")
print("─" * 65)
all_have_targets = True
for drug in result:
    targets = drug.get("targets", [])
    status  = "✓" if targets else "✗"
    if not targets:
        all_have_targets = False
    target_str = ", ".join(targets[:4]) + ("..." if len(targets) > 4 else "")
    print(f"{status} {drug['name']:<38} {target_str or 'NONE'}")

print()
if all_have_targets:
    print("✅ All drugs got targets from fallback — salt stripping is working")
else:
    print("❌ Some drugs still have no targets after fallback")
    print("   Check that _strip_salt is defined and KNOWN_SMALL_MOLECULE_TARGETS has the right keys")

# ── 3. Test mechanism scoring ─────────────────────────────────────────────────
print()
print("=" * 65)
print("TEST 3: Mechanism scoring (scorer.py DRUG_NAME_MECHANISM_HINTS)")
print("=" * 65)

try:
    import networkx as nx
    from backend.pipeline.scorer import ProductionScorer, DRUG_NAME_MECHANISM_HINTS

    scorer = ProductionScorer(nx.Graph())

    test_cases = [
        # (drug_name, chembl_mechanism, disease, expected_score_gt)
        ("sildenafil",       "",                     "pulmonary arterial hypertension", 0.0),
        ("bosentan",         "",                     "pulmonary arterial hypertension", 0.0),
        ("iloprost",         "",                     "pulmonary arterial hypertension", 0.0),
        ("dexamethasone",    "",                     "multiple myeloma",                0.0),
        ("bortezomib",       "proteasome inhibitor", "multiple myeloma",                0.0),
        ("thalidomide",      "",                     "multiple myeloma",                0.0),
        ("metformin",        "",                     "type 2 diabetes mellitus",        0.0),
        ("spironolactone",   "",                     "heart failure",                   0.0),
        ("metoprolol",       "",                     "heart failure",                   0.0),
        ("donepezil",        "",                     "alzheimer disease",               0.0),
        ("memantine",        "",                     "alzheimer disease",               0.0),
        ("rasagiline",       "",                     "parkinson disease",               0.0),
        ("amantadine",       "",                     "parkinson disease",               0.0),
        ("colchicine",       "",                     "gout",                            0.0),
        ("aspirin",          "",                     "pericarditis",                    0.0),
        ("atorvastatin",     "",                     "hypercholesterolemia",            0.0),
        ("ezetimibe",        "",                     "hypercholesterolemia",            0.0),
        ("hydroxychloroquine","",                    "rheumatoid arthritis",            0.0),
        ("methotrexate",     "antifolate",           "rheumatoid arthritis",            0.0),
        # These should score LOW (negative controls)
        ("aspirin",          "",                     "parkinson disease",               None),
        ("metformin",        "",                     "multiple myeloma",                None),
        ("haloperidol",      "dopamine antagonist",  "parkinson disease",               None),
    ]

    print(f"{'Drug':<22} {'Disease':<36} {'Hint?':<8} {'Score':<8} {'Result'}")
    print("─" * 85)

    positive_failures = []
    for drug_name, mech, disease, expected in test_cases:
        has_hint = drug_name.lower() in DRUG_NAME_MECHANISM_HINTS
        disease_data = {
            "name": disease, "description": "", "genes": [], "gene_scores": {}
        }
        drug_data = {
            "name": drug_name, "mechanism": mech, "targets": [], "pathways": []
        }
        score = scorer._score_mechanism_similarity(drug_data, disease_data)

        if expected is None:
            # Negative control — just show, don't fail
            result_str = f"(neg ctrl: {score:.2f})"
        elif score > expected:
            result_str = f"✓ PASS ({score:.2f} > {expected})"
        else:
            result_str = f"✗ FAIL ({score:.2f} should be > {expected})"
            positive_failures.append(f"{drug_name} / {disease}: score={score:.2f}")

        hint_str = "✓" if has_hint else "✗"
        print(f"  {drug_name:<20} {disease:<35} {hint_str:<8} {score:<8.2f} {result_str}")

    print()
    if not positive_failures:
        print("✅ All mechanism scores PASS — DRUG_NAME_MECHANISM_HINTS is working")
    else:
        print(f"❌ {len(positive_failures)} mechanism score failures:")
        for f in positive_failures:
            print(f"   - {f}")
        print("   → Make sure scorer.py was replaced with the patched version")

except ImportError as e:
    print(f"❌ Could not import scorer: {e}")
    print("   Replace backend/pipeline/scorer.py with the patched version first")

# ── 4. Summary ────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("SUMMARY")
print("=" * 65)
print("If all tests pass, run:")
print("  python run_validation.py --fast")
print()
print("Expected validation results after fixes:")
print("  sildenafil / PAH:        score > 0.10  (was 0.0)")
print("  bosentan / PAH:          score > 0.10  (was 0.0)")
print("  dexamethasone / myeloma: score > 0.10  (was 0.0)")
print("  spironolactone / HF:     score > 0.10  (was 0.0)")
print("  donepezil+memantine:     appear in AD top combos")
print("  sildenafil+bosentan:     appear in PAH top combos")