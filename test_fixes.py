#!/usr/bin/env python3
"""
test_fixes.py — Quick offline tests for scorer v9 and combo_scorer v6
Run from repo root: python test_fixes.py

Tests:
  1. Mechanism floors (dexamethasone/myeloma, sirolimus/TSC, valproic/epilepsy, 
     metformin/PCOS, spironolactone/HF)
  2. Synergy pairs (all missing pairs now present)
  3. Context penalty (dex+bortezomib in myeloma, sildenafil+bosentan in PAH)
  4. Salt-form normalisation (Memantine hydrochloride → memantine)
"""

import sys
sys.path.insert(0, ".")

print("=" * 65)
print("TwinTrial Pipeline Fix Tests v6.0")
print("=" * 65)

FAILURES = []

def check(condition: bool, label: str, detail: str = "") -> None:
    status = "✅ PASS" if condition else "❌ FAIL"
    print(f"  {status}: {label}" + (f" ({detail})" if detail else ""))
    if not condition:
        FAILURES.append(f"{label}" + (f" — {detail}" if detail else ""))


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1: Mechanism floors in scorer v9
# ─────────────────────────────────────────────────────────────────────────────

print("\n[1] Mechanism floor tests (scorer.py v9)")
print("─" * 50)

try:
    import networkx as nx
    from backend.pipeline.scorer import (
        ProductionScorer,
        DRUG_NAME_MECHANISM_HINTS,
        MECHANISM_FLOOR_VALUE,
        MECHANISM_FLOOR2_VALUE,
        MECHANISM_FLOOR3_VALUE,
        MECHANISM_FLOOR4_VALUE,
    )

    scorer = ProductionScorer(nx.Graph())

    def make_disease(name, genes=None, pathways=None):
        return {
            "name": name, "description": name,
            "genes": genes or [], "gene_scores": {},
            "pathways": pathways or [], "is_rare": False,
        }

    def make_drug(name, mechanism="", targets=None, pathways=None):
        return {
            "name": name, "drug_name": name,
            "mechanism": mechanism,
            "targets": targets or [],
            "pathways": pathways or [],
        }

    def score(drug_name, disease_name, mechanism="", targets=None, disease_genes=None, is_rare=False):
        d_data = make_disease(disease_name, genes=disease_genes or [])
        d_data["is_rare"] = is_rare
        dr_data = make_drug(drug_name, mechanism=mechanism, targets=targets or [])
        s, ev = scorer.score_drug_disease_match(drug_name, disease_name, d_data, dr_data)
        return s, ev

    # PRIMARY FLOOR (0.32): high mechanism, near-zero gene overlap
    s, ev = score("dexamethasone", "multiple myeloma",
                  mechanism="glucocorticoid", 
                  targets=["NR3C1", "CD86"],
                  disease_genes=["CRBN", "IRF4", "PSMB5", "IKZF1", "TP53", "FLT3", "KRAS", "NRAS", "FGFR3", "BCL2"])
    check(s >= 0.30, f"dexamethasone/myeloma >= 0.30", f"got {s:.3f}")
    check(s >= MECHANISM_FLOOR_VALUE or s >= 0.30, 
          f"dexamethasone/myeloma floor applied or score high", f"got {s:.3f}, floor={MECHANISM_FLOOR_VALUE}")

    # SECONDARY FLOOR (0.27): moderate mechanism + some gene overlap
    s, ev = score("spironolactone", "heart failure",
                  mechanism="aldosterone antagonist mineralocorticoid",
                  targets=["NR3C2", "NR3C1", "NOS3"],
                  disease_genes=["ADRB1", "NPPA", "NPPB", "NR3C2", "ACE", "MYH7", "TNNT2"])
    check(s >= 0.25, f"spironolactone/heart failure >= 0.25", f"got {s:.3f}")

    # TERTIARY FLOOR (0.30): anticonvulsant with gene signal
    s, ev = score("valproic acid", "epilepsy",
                  mechanism="anticonvulsant antiepileptic sodium channel",
                  targets=["SCN1A", "GRIN2B", "CACNA1C", "ABAT", "ALDH5A1"],
                  disease_genes=["SCN1A", "KCNQ2", "GRIN2B", "CACNA1C", "CDKL5", "DEPDC5"])
    check(s >= 0.30, f"valproic acid/epilepsy >= 0.30", f"got {s:.3f}")

    # QUATERNARY FLOOR (0.35): mTOR in TSC (rare disease)
    s, ev = score("sirolimus", "tuberous sclerosis",
                  mechanism="mtor inhibitor tuberous sclerosis tsc fkbp1a",
                  targets=["MTOR", "TSC1", "TSC2", "FKBP1A"],
                  disease_genes=["TSC1", "TSC2", "MTOR", "AKT1", "RHEB"],
                  is_rare=True)
    check(s >= 0.35, f"sirolimus/tuberous sclerosis >= 0.35", f"got {s:.3f}")

    # PCOS FLOOR (0.28): metformin in PCOS
    s, ev = score("metformin", "polycystic ovary syndrome",
                  mechanism="biguanide ampk",
                  targets=["PRKAA1", "PRKAA2", "PPARG"],
                  disease_genes=["INSR", "CYP11A1", "LHCGR", "PTEN", "PPARG"])
    check(s >= 0.28, f"metformin/PCOS >= 0.28", f"got {s:.3f}")

    # Mechanism floor constant check
    check(MECHANISM_FLOOR_VALUE >= 0.32, f"Primary floor constant >= 0.32", f"is {MECHANISM_FLOOR_VALUE}")
    check(MECHANISM_FLOOR2_VALUE >= 0.27, f"Secondary floor constant >= 0.27", f"is {MECHANISM_FLOOR2_VALUE}")
    check(MECHANISM_FLOOR3_VALUE >= 0.30, f"Tertiary floor constant >= 0.30", f"is {MECHANISM_FLOOR3_VALUE}")
    check(MECHANISM_FLOOR4_VALUE >= 0.35, f"Quaternary floor constant >= 0.35", f"is {MECHANISM_FLOOR4_VALUE}")

    # Negative controls — should NOT get high scores
    s, _ = score("metformin", "multiple myeloma",
                 mechanism="biguanide ampk", targets=["PRKAA1"])
    check(s < 0.20, f"metformin/myeloma (negative control) < 0.20", f"got {s:.3f}")

    s, _ = score("haloperidol", "parkinson disease",
                 mechanism="dopamine antagonist", targets=["DRD2"])
    check(s == 0.0, f"haloperidol/parkinson = 0.0 (hard contraindication)", f"got {s:.3f}")

except Exception as e:
    print(f"  ❌ ERROR in test 1: {e}")
    import traceback; traceback.print_exc()
    FAILURES.append(f"Test 1 crashed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2: Synergy pairs in combo_scorer v6
# ─────────────────────────────────────────────────────────────────────────────

print("\n[2] Synergy pair tests (combo_scorer.py v6)")
print("─" * 50)

try:
    from backend.pipeline.combo_scorer import (
        CombinationScorer, SYNERGISTIC_PAIRS, classify_mechanism, _normalise_name
    )

    # PAH synergies
    check(frozenset({"pde5_inhibitor", "endothelin_antagonist"}) in SYNERGISTIC_PAIRS,
          "PDE5i + ERA synergistic (sildenafil+bosentan PAH)")
    check(frozenset({"pde5_inhibitor", "prostacyclin"}) in SYNERGISTIC_PAIRS,
          "PDE5i + prostacyclin synergistic (sildenafil+iloprost PAH)")
    check(frozenset({"endothelin_antagonist", "prostacyclin"}) in SYNERGISTIC_PAIRS,
          "ERA + prostacyclin synergistic (bosentan+iloprost PAH)")

    # Myeloma synergies
    check(frozenset({"imid", "corticosteroid"}) in SYNERGISTIC_PAIRS,
          "IMiD + corticosteroid synergistic (thalidomide+dex TD regimen)")
    check(frozenset({"proteasome_inhibitor", "corticosteroid"}) in SYNERGISTIC_PAIRS,
          "proteasome_inhibitor + corticosteroid synergistic (bortezomib+dex VD)")
    check(frozenset({"alkylating_agent", "corticosteroid"}) in SYNERGISTIC_PAIRS,
          "alkylating_agent + corticosteroid synergistic (melphalan+dex MPD)")
    check(frozenset({"imid", "proteasome_inhibitor"}) in SYNERGISTIC_PAIRS,
          "IMiD + proteasome_inhibitor synergistic (thalidomide+bortezomib VTd)")

    # Heart failure
    check(frozenset({"mineralocorticoid_antagonist", "beta_blocker"}) in SYNERGISTIC_PAIRS,
          "MRA + beta_blocker synergistic (spironolactone+metoprolol RALES+MERIT)")
    check(frozenset({"ace_inhibitor", "beta_blocker"}) in SYNERGISTIC_PAIRS,
          "ACEi + beta_blocker synergistic (lisinopril+metoprolol)")

    # Neurology
    check(frozenset({"acetylcholinesterase_inhibitor", "nmda_antagonist"}) in SYNERGISTIC_PAIRS,
          "AChEI + NMDA synergistic (Namzaric - donepezil+memantine)")
    check(frozenset({"maob_inhibitor", "nmda_antagonist"}) in SYNERGISTIC_PAIRS,
          "MAO-B + NMDA synergistic (rasagiline+amantadine PD)")

    # RA
    check(frozenset({"dmard", "antimalarial"}) in SYNERGISTIC_PAIRS,
          "DMARD + antimalarial synergistic (MTX+HCQ)")
    check(frozenset({"dmard", "sulfonamide"}) in SYNERGISTIC_PAIRS,
          "DMARD + sulfonamide synergistic (MTX+SSZ)")
    check(frozenset({"antimalarial", "sulfonamide"}) in SYNERGISTIC_PAIRS,
          "antimalarial + sulfonamide synergistic (HCQ+SSZ)")

    # Pericarditis
    check(frozenset({"colchicine", "nsaid"}) in SYNERGISTIC_PAIRS,
          "colchicine + NSAID synergistic (COPE trial aspirin+colchicine)")

    # Hypercholesterolaemia
    check(frozenset({"statin", "cholesterol_absorption_inhibitor"}) in SYNERGISTIC_PAIRS,
          "statin + cholesterol_absorption_inhibitor (atorvastatin+ezetimibe IMPROVE-IT)")

except Exception as e:
    print(f"  ❌ ERROR in test 2: {e}")
    import traceback; traceback.print_exc()
    FAILURES.append(f"Test 2 crashed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3: Context penalty tests
# ─────────────────────────────────────────────────────────────────────────────

print("\n[3] Context penalty tests (combo_scorer.py v6)")
print("─" * 50)

try:
    from backend.pipeline.combo_scorer import CombinationScorer

    # PAH: sildenafil + bosentan should have near-zero context penalty
    scorer_pah = CombinationScorer(disease_name="pulmonary arterial hypertension")
    sild = {"drug_name": "sildenafil", "mechanism": "pde5 inhibitor", "mechanism_score": 1.0, "score": 0.55, "target_genes": ["PDE5A", "NOS3"]}
    bosen = {"drug_name": "bosentan", "mechanism": "endothelin antagonist", "mechanism_score": 1.0, "score": 0.60, "target_genes": ["EDNRA", "EDNRB"]}
    result = scorer_pah.score_pair(sild, bosen, ["PDE5A", "EDNRA", "EDNRB", "NOS3", "BMPR2"])
    check(result["context_penalty"] < 0.05, 
          f"sildenafil+bosentan context_penalty near-zero in PAH", 
          f"got {result['context_penalty']:.3f}")
    check(result["is_synergistic"], "sildenafil+bosentan is_synergistic=True in PAH")
    check(result["combo_score"] > 0.5, f"sildenafil+bosentan combo_score > 0.5", 
          f"got {result['combo_score']:.3f}")

    # PAH: iloprost + sildenafil
    ilo = {"drug_name": "iloprost", "mechanism": "prostacyclin", "mechanism_score": 1.0, "score": 0.50, "target_genes": ["PTGIR", "PTGIS"]}
    result2 = scorer_pah.score_pair(ilo, sild, ["PDE5A", "EDNRA", "PTGIR", "NOS3", "BMPR2"])
    check(result2["is_synergistic"], "iloprost+sildenafil is_synergistic=True in PAH")
    check(result2["context_penalty"] < 0.05, 
          f"iloprost+sildenafil near-zero context_penalty in PAH",
          f"got {result2['context_penalty']:.3f}")

    # Myeloma: bortezomib + dexamethasone — CRITICAL FIX
    scorer_mm = CombinationScorer(disease_name="multiple myeloma")
    bort = {"drug_name": "bortezomib", "mechanism": "proteasome inhibitor", "mechanism_score": 0.3, "score": 0.50, "target_genes": ["PSMB5", "PSMB6"]}
    dex = {"drug_name": "dexamethasone", "mechanism": "glucocorticoid corticosteroid", "mechanism_score": 1.0, "score": 0.32, "target_genes": ["NR3C1"]}
    result3 = scorer_mm.score_pair(bort, dex, ["CRBN", "PSMB5", "NR3C1", "IRF4", "IKZF1"])
    check(result3["is_synergistic"], 
          "bortezomib+dexamethasone is_synergistic=True in myeloma (VD regimen)")
    check(result3["context_penalty"] < 0.15, 
          f"bortezomib+dexamethasone low context_penalty in myeloma",
          f"got {result3['context_penalty']:.3f}")
    check(result3["combo_score"] > 0.40, 
          f"bortezomib+dexamethasone combo_score > 0.40",
          f"got {result3['combo_score']:.3f}")

    # Myeloma: thalidomide + dexamethasone (TD regimen)
    thal = {"drug_name": "thalidomide", "mechanism": "cereblon imid", "mechanism_score": 0.6, "score": 0.37, "target_genes": ["CRBN", "IRF4"]}
    result4 = scorer_mm.score_pair(thal, dex, ["CRBN", "NR3C1", "IRF4", "IKZF1"])
    check(result4["is_synergistic"],
          "thalidomide+dexamethasone is_synergistic=True in myeloma (TD regimen)")
    check(result4["combo_score"] > 0.40,
          f"thalidomide+dexamethasone combo_score > 0.40",
          f"got {result4['combo_score']:.3f}")

    # Alzheimer: donepezil + memantine (Namzaric)
    scorer_ad = CombinationScorer(disease_name="alzheimer disease")
    done = {"drug_name": "donepezil", "mechanism": "acetylcholinesterase inhibitor", "mechanism_score": 0.6, "score": 0.50, "target_genes": ["ACHE", "BCHE"]}
    mema = {"drug_name": "memantine", "mechanism": "nmda antagonist", "mechanism_score": 0.6, "score": 0.83, "target_genes": ["GRIN1", "GRIN2A", "GRIN2B"]}
    result5 = scorer_ad.score_pair(done, mema, ["ACHE", "APP", "PSEN1", "MAPT", "GRIN1", "GRIN2B"])
    check(result5["is_synergistic"],
          "donepezil+memantine is_synergistic=True in Alzheimer (Namzaric)")

    # Heart failure: spironolactone + metoprolol
    scorer_hf = CombinationScorer(disease_name="heart failure")
    spiro = {"drug_name": "spironolactone", "mechanism": "aldosterone antagonist mineralocorticoid", "mechanism_score": 0.6, "score": 0.26, "target_genes": ["NR3C2"]}
    metro = {"drug_name": "metoprolol", "mechanism": "beta blocker", "mechanism_score": 0.6, "score": 0.58, "target_genes": ["ADRB1"]}
    result6 = scorer_hf.score_pair(spiro, metro, ["ADRB1", "NR3C2", "ACE", "NPPA", "NPPB"])
    check(result6["is_synergistic"],
          "spironolactone+metoprolol is_synergistic=True in HF (RALES+MERIT)")

    # Taxane should be heavily penalised in PAH (negative control)
    taxane = {"drug_name": "cabazitaxel", "mechanism": "taxane", "mechanism_score": 0.0, "score": 0.30, "target_genes": ["TUBB"]}
    result7 = scorer_pah.score_pair(taxane, sild, [])
    check(result7["context_penalty"] > 0.30,
          f"taxane+PDE5i heavy context_penalty in PAH (negative control)",
          f"got {result7['context_penalty']:.3f}")

except Exception as e:
    print(f"  ❌ ERROR in test 3: {e}")
    import traceback; traceback.print_exc()
    FAILURES.append(f"Test 3 crashed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4: Salt-form normalisation
# ─────────────────────────────────────────────────────────────────────────────

print("\n[4] Salt-form normalisation tests")
print("─" * 50)

try:
    from backend.pipeline.combo_scorer import _normalise_name, classify_mechanism

    pairs = [
        ("Memantine hydrochloride", "memantine"),
        ("Metformin hydrochloride", "metformin"),
        ("Dexamethasone sodium phosphate", "dexamethasone"),
        ("Bosentan monohydrate", "bosentan"),
        ("Donepezil hydrochloride", "donepezil"),
        ("Rasagiline mesylate", "rasagiline"),
        ("Atorvastatin calcium", "atorvastatin"),
        ("Hydroxychloroquine sulfate", "hydroxychloroquine"),
    ]

    for raw, expected in pairs:
        result = _normalise_name(raw)
        check(result == expected, f"_normalise_name('{raw}')", f"got '{result}', expected '{expected}'")

    # Verify mechanism classification for salt-form names
    check(classify_mechanism("memantine hydrochloride") == "nmda_antagonist",
          "classify_mechanism('memantine hydrochloride') = nmda_antagonist")
    check(classify_mechanism("bosentan monohydrate") == "endothelin_antagonist",
          "classify_mechanism('bosentan monohydrate') = endothelin_antagonist")
    check(classify_mechanism("iloprost") == "prostacyclin",
          "classify_mechanism('iloprost') = prostacyclin")
    check(classify_mechanism("dexamethasone") == "corticosteroid",
          "classify_mechanism('dexamethasone') = corticosteroid")

except Exception as e:
    print(f"  ❌ ERROR in test 4: {e}")
    import traceback; traceback.print_exc()
    FAILURES.append(f"Test 4 crashed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5: Production pipeline name normalisation (offline)
# ─────────────────────────────────────────────────────────────────────────────

print("\n[5] Production pipeline normalisation tests")
print("─" * 50)

try:
    from backend.pipeline.production_pipeline import (
        _normalise_drug_name, _build_canonical_lookup
    )

    pairs = [
        ("Memantine hydrochloride", "memantine"),
        ("Metformin hydrochloride", "metformin"),
        ("Dexamethasone sodium phosphate", "dexamethasone"),
    ]
    for raw, expected in pairs:
        result = _normalise_drug_name(raw)
        check(result == expected, f"_normalise_drug_name('{raw}')", f"got '{result}'")

    # Canonical lookup with salt forms
    candidates = [
        {"drug_name": "Memantine hydrochloride", "score": 0.83},
        {"drug_name": "Donepezil", "score": 0.50},
        {"drug_name": "Metformin hydrochloride", "score": 0.49},
    ]
    lookup = _build_canonical_lookup(candidates)
    check("memantine" in lookup, "canonical lookup has 'memantine' from 'Memantine hydrochloride'")
    check("donepezil" in lookup, "canonical lookup has 'donepezil'")
    check("metformin" in lookup, "canonical lookup has 'metformin' from 'Metformin hydrochloride'")

except Exception as e:
    print(f"  ❌ ERROR in test 5: {e}")
    import traceback; traceback.print_exc()
    FAILURES.append(f"Test 5 crashed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
if FAILURES:
    print(f"❌ FAILED: {len(FAILURES)} test(s)")
    for f in FAILURES:
        print(f"   • {f}")
    print()
    print("Fix: Make sure scorer.py and combo_scorer.py were copied correctly.")
    sys.exit(1)
else:
    print("✅ ALL TESTS PASSED")
    print()
    print("Next steps:")
    print("  1. Delete drug cache if it exists:")
    print("     rm -f /tmp/drug_repurposing_cache/chembl_approved_drugs.json")
    print()
    print("  2. Run fast validation (5-10 min):")
    print("     python run_validation.py --fast")
    print()
    print("  3. Run combo validation (60-90 min):")
    print("     python combo_validation_dataset.py --top-n 15")
    print()
    print("Expected results:")
    print("  Validation F1: ≥ 0.95 (was 0.947)")
    print("  dexamethasone/myeloma: ≥ 0.30 (was 0.28)")
    print("  sirolimus/TSC:         ≥ 0.35 (was 0.28)")
    print("  valproic acid/epilepsy:≥ 0.30 (was 0.28)")
    print("  metformin/PCOS:        ≥ 0.28 (was 0.204)")
    print("  Combo pass rate:       ≥ 60%  (was 28.6%)")
    print("=" * 65)