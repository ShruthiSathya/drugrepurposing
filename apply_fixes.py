"""
PATCHES FOR data_fetcher.py AND combo_scorer.py
================================================
Apply these changes to fix:
  1. data_fetcher.py  — fallback now SUPPLEMENTS existing targets (not just fills empty)
  2. combo_scorer.py  — wrong combos at top (IMATINIB+OLANZAPINE for PAH, etc.)
                        fixed by adding drug class overrides + increasing "other" penalty

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATCH 1: data_fetcher.py
File: backend/pipeline/data_fetcher.py

Find the method `_apply_small_molecule_fallback` and REPLACE IT with:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# ── REPLACEMENT for data_fetcher.py::_apply_small_molecule_fallback ──────────
#
# WHY: The old version only filled drugs with EMPTY target lists.
# If the cache had a drug with partial/wrong targets from a previous API run
# (e.g. spironolactone with targets from DGIdb that don't include NR3C2),
# the curated fallback would be silently skipped.
#
# The new version ALWAYS supplements: it adds any curated targets that are
# missing from the drug's current target list. Existing targets are preserved.

def _apply_small_molecule_fallback_REPLACEMENT(self, drugs):
    """
    PATCHED v2: Supplements existing targets with curated known targets.
    Always runs (not just for drugs with empty targets).
    Called on every cache load — cheap dict lookups, no API calls.
    """
    filled = 0
    supplemented = 0
    for drug in drugs:
        name_lower = drug["name"].lower()
        if name_lower not in KNOWN_SMALL_MOLECULE_TARGETS:
            continue

        known = KNOWN_SMALL_MOLECULE_TARGETS[name_lower]
        existing = drug.get("targets") or []

        if not existing:
            # Drug has no targets at all — fill completely
            drug["targets"]       = known
            drug["target_source"] = "small_molecule_lit_fallback"
            drug["pathways"]      = self._infer_pathways_from_targets_fallback(known)
            filled += 1
        else:
            # Drug already has targets — add any missing curated targets
            existing_set = set(existing)
            new_targets  = [t for t in known if t not in existing_set]
            if new_targets:
                drug["targets"] = existing + new_targets
                drug["pathways"] = self._infer_pathways_from_targets_fallback(drug["targets"])
                supplemented += 1

    if filled or supplemented:
        import logging
        logging.getLogger(__name__).info(
            f"Small molecule fallback: {filled} drugs filled, "
            f"{supplemented} drugs supplemented with additional targets"
        )
    return drugs


"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATCH 2: combo_scorer.py
File: backend/pipeline/combo_scorer.py

Step A: Add this dict RIGHT AFTER the MECHANISM_KEYWORD_MAP list (before classify_mechanism).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# ── ADD this dict to combo_scorer.py after MECHANISM_KEYWORD_MAP ─────────────
KNOWN_DRUG_CLASS_OVERRIDES = {
    # PAH / pulmonary vascular
    "sildenafil":          "pde5_inhibitor",
    "tadalafil":           "pde5_inhibitor",
    "vardenafil":          "pde5_inhibitor",
    "bosentan":            "endothelin_antagonist",
    "ambrisentan":         "endothelin_antagonist",
    "macitentan":          "endothelin_antagonist",
    "sitaxentan":          "endothelin_antagonist",
    "iloprost":            "prostacyclin",
    "treprostinil":        "prostacyclin",
    "epoprostenol":        "prostacyclin",
    "beraprost":           "prostacyclin",
    "selexipag":           "prostacyclin",
    "riociguat":           "sgc_stimulator",
    # Cardiovascular
    "spironolactone":      "mineralocorticoid_antagonist",
    "eplerenone":          "mineralocorticoid_antagonist",
    "finerenone":          "mineralocorticoid_antagonist",
    "metoprolol":          "beta_blocker",
    "carvedilol":          "beta_blocker",
    "atenolol":            "beta_blocker",
    "bisoprolol":          "beta_blocker",
    "nebivolol":           "beta_blocker",
    "propranolol":         "beta_blocker",
    "labetalol":           "beta_blocker",
    "lisinopril":          "ace_inhibitor",
    "enalapril":           "ace_inhibitor",
    "ramipril":            "ace_inhibitor",
    "captopril":           "ace_inhibitor",
    "losartan":            "arb",
    "valsartan":           "arb",
    "candesartan":         "arb",
    "irbesartan":          "arb",
    "furosemide":          "diuretic",
    "hydrochlorothiazide": "diuretic",
    "torsemide":           "diuretic",
    # Lipid
    "atorvastatin":        "statin",
    "rosuvastatin":        "statin",
    "simvastatin":         "statin",
    "lovastatin":          "statin",
    "pravastatin":         "statin",
    "fluvastatin":         "statin",
    "pitavastatin":        "statin",
    "ezetimibe":           "cholesterol_absorption_inhibitor",
    # Metabolic
    "metformin":           "biguanide",
    "metformin hydrochloride": "biguanide",
    "pioglitazone":        "thiazolidinedione",
    "rosiglitazone":       "thiazolidinedione",
    "glipizide":           "sulfonylurea",
    "glimepiride":         "sulfonylurea",
    "glyburide":           "sulfonylurea",
    "glibenclamide":       "sulfonylurea",
    "empagliflozin":       "sglt2_inhibitor",
    "dapagliflozin":       "sglt2_inhibitor",
    "canagliflozin":       "sglt2_inhibitor",
    # Anti-inflammatory / Immunology
    "dexamethasone":       "corticosteroid",
    "prednisone":          "corticosteroid",
    "prednisolone":        "corticosteroid",
    "methylprednisolone":  "corticosteroid",
    "hydrocortisone":      "corticosteroid",
    "hydroxychloroquine":  "antimalarial",
    "chloroquine":         "antimalarial",
    "methotrexate":        "dmard",
    "sulfasalazine":       "sulfonamide",
    "leflunomide":         "dhodh_inhibitor",
    "thalidomide":         "imid",
    "lenalidomide":        "imid",
    "pomalidomide":        "imid",
    "bortezomib":          "proteasome_inhibitor",
    "melphalan":           "alkylating_agent",
    "cyclophosphamide":    "alkylating_agent",
    # Neurology
    "donepezil":           "acetylcholinesterase_inhibitor",
    "rivastigmine":        "acetylcholinesterase_inhibitor",
    "galantamine":         "acetylcholinesterase_inhibitor",
    "memantine":           "nmda_antagonist",
    "amantadine":          "nmda_antagonist",
    "rasagiline":          "maob_inhibitor",
    "selegiline":          "maob_inhibitor",
    "pramipexole":         "dopamine_agonist",
    "ropinirole":          "dopamine_agonist",
    "levodopa":            "dopamine_precursor",
    "carbidopa":           "dopamine_precursor",
    # Pain / inflammation
    "aspirin":             "nsaid",
    "ibuprofen":           "nsaid",
    "naproxen":            "nsaid",
    "celecoxib":           "cox2_inhibitor",
    "indomethacin":        "nsaid",
    "colchicine":          "colchicine",
    "allopurinol":         "anti_uric_acid",
    "febuxostat":          "anti_uric_acid",
    # Oncology
    "tamoxifen":           "serm",
    "raloxifene":          "serm",
    "letrozole":           "aromatase_inhibitor",
    "anastrozole":         "aromatase_inhibitor",
    "finasteride":         "5_alpha_reductase_inhibitor",
    "minoxidil":           "potassium_channel",
    "imatinib":            "kinase_inhibitor",
    "sirolimus":           "mtor_inhibitor",
    "olaparib":            "parp_inhibitor",
}


"""
Step B: REPLACE the classify_mechanism function with this version that
        checks KNOWN_DRUG_CLASS_OVERRIDES first:
"""

def classify_mechanism_REPLACEMENT(mechanism: str, drug_name: str = "") -> str:
    """
    PATCHED: Checks KNOWN_DRUG_CLASS_OVERRIDES by drug name first,
    then falls back to keyword matching on mechanism text.
    This ensures sildenafil → pde5_inhibitor, dexamethasone → corticosteroid, etc.
    even when ChEMBL mechanism strings are empty or don't match keywords.
    """
    # Check drug name override first (highest priority)
    if drug_name:
        override = KNOWN_DRUG_CLASS_OVERRIDES.get(drug_name.lower().strip())
        if override:
            return override

    if not mechanism:
        return "other"

    mech_lower = mechanism.lower()
    for keyword, cls in MECHANISM_KEYWORD_MAP:
        if keyword in mech_lower:
            return cls
    return "other"


"""
Step C: In the CombinationScorer.score_pair() method, update the two
        classify_mechanism calls to pass the drug name:

FIND (appears twice in score_pair, once for each drug):
    class_a = classify_mechanism(mech_a) if mech_a else classify_mechanism(name_a.lower())
    class_b = classify_mechanism(mech_b) if mech_b else classify_mechanism(name_b.lower())

REPLACE WITH:
    class_a = classify_mechanism(mech_a, drug_name=name_a) if mech_a else classify_mechanism("", drug_name=name_a)
    class_b = classify_mechanism(mech_b, drug_name=name_b) if mech_b else classify_mechanism("", drug_name=name_b)

Also update the target-based fallback section (the "if class_a == 'other'" block)
to pass drug_name so the override check happens first:
    if class_a == "other":
        class_a = classify_mechanism("", drug_name=name_a)   # re-check with name
        if class_a == "other":
            for target in (drug_a.get("target_genes") or drug_a.get("targets") or []):
                c = classify_mechanism(target.lower())
                if c != "other":
                    class_a = c
                    break

    if class_b == "other":
        class_b = classify_mechanism("", drug_name=name_b)   # re-check with name
        if class_b == "other":
            for target in (drug_b.get("target_genes") or drug_b.get("targets") or []):
                c = classify_mechanism(target.lower())
                if c != "other":
                    class_b = c
                    break
"""

"""
Step D: REPLACE the _disease_context_penalty method with this version
        (increases "other" penalty from 0.05 to 0.20):
"""

def _disease_context_penalty_REPLACEMENT(self, class_a: str, class_b: str) -> float:
    """
    PATCHED: "other" class penalty increased from 0.05 to 0.20.

    This stops unrelated drugs (antipsychotics, glaucoma drops, antiemetics)
    from dominating combo rankings for well-defined diseases like PAH and AD.

    With the KNOWN_DRUG_CLASS_OVERRIDES dict, legitimate drugs are now
    correctly classified BEFORE reaching the "other" path, so this
    heavier penalty only hits truly unrelated drugs.

    Old: other_penalty = 0.05  → IMATINIB + OLANZAPINE ranked #1 for PAH
    New: other_penalty = 0.20  → unrelated drugs get penalized out of top results
    """
    if not self.disease_name:
        return 0.0

    from .combo_scorer import _get_appropriate_classes, _is_oncology_disease, ONCOLOGY_CLASSES

    appropriate = _get_appropriate_classes(self.disease_name)
    is_oncology = _is_oncology_disease(self.disease_name)
    penalty = 0.0

    OTHER_PENALTY = 0.20  # was 0.05 — increased 4x

    for cls in (class_a, class_b):
        if cls == "other":
            # "other" gets a meaningful penalty in any known-disease context
            if appropriate is not None:
                penalty += OTHER_PENALTY
            continue
        # Heavy penalty for oncology drugs in non-oncology diseases
        if cls in ONCOLOGY_CLASSES and not is_oncology:
            penalty += 0.40
        elif appropriate is not None and cls not in appropriate:
            penalty += 0.20

    return min(penalty, 0.80)


"""
Step E: In DISEASE_APPROPRIATE_CLASSES dict, REMOVE "other" from the
        entries for diseases where the drug classes are well-known.
        This forces non-PAH, non-myeloma etc. drugs to get the context penalty.

FIND:
    "pulmonary arterial hypertension": {
        "pde5_inhibitor", "endothelin_antagonist", "prostacyclin", "sgc_stimulator",
        "kinase_inhibitor", "diuretic", "other",
    },
    "pulmonary hypertension": {
        "pde5_inhibitor", "endothelin_antagonist", "prostacyclin", "sgc_stimulator",
        "kinase_inhibitor", "diuretic", "other",
    },
    "multiple myeloma": {
        "imid", "proteasome_inhibitor", "corticosteroid", "alkylating_agent",
        "hdac_inhibitor", "anti_cd20", "kinase_inhibitor", "other",
    },
    "alzheimer disease": {
        "acetylcholinesterase_inhibitor", "nmda_antagonist", "other",
    },
    "alzheimer": {
        "acetylcholinesterase_inhibitor", "nmda_antagonist", "other",
    },
    "parkinson disease": {
        "dopamine_agonist", "dopamine_precursor", "maob_inhibitor",
        "nmda_antagonist", "other",
    },
    "parkinson": {
        "dopamine_agonist", "dopamine_precursor", "maob_inhibitor",
        "nmda_antagonist", "other",
    },
    "heart failure": {
        "ace_inhibitor", "arb", "beta_blocker", "diuretic",
        "mineralocorticoid_antagonist", "sglt2_inhibitor", "other",
    },
    "hypercholesterolemia": {
        "statin", "cholesterol_absorption_inhibitor", "biguanide", "other",
    },

REPLACE WITH (remove "other" from all of these):
    "pulmonary arterial hypertension": {
        "pde5_inhibitor", "endothelin_antagonist", "prostacyclin", "sgc_stimulator",
        "kinase_inhibitor", "diuretic",
    },
    "pulmonary hypertension": {
        "pde5_inhibitor", "endothelin_antagonist", "prostacyclin", "sgc_stimulator",
        "kinase_inhibitor", "diuretic",
    },
    "multiple myeloma": {
        "imid", "proteasome_inhibitor", "corticosteroid", "alkylating_agent",
        "hdac_inhibitor", "anti_cd20", "kinase_inhibitor",
    },
    "alzheimer disease": {
        "acetylcholinesterase_inhibitor", "nmda_antagonist",
    },
    "alzheimer": {
        "acetylcholinesterase_inhibitor", "nmda_antagonist",
    },
    "parkinson disease": {
        "dopamine_agonist", "dopamine_precursor", "maob_inhibitor", "nmda_antagonist",
    },
    "parkinson": {
        "dopamine_agonist", "dopamine_precursor", "maob_inhibitor", "nmda_antagonist",
    },
    "heart failure": {
        "ace_inhibitor", "arb", "beta_blocker", "diuretic",
        "mineralocorticoid_antagonist", "sglt2_inhibitor",
    },
    "hypercholesterolemia": {
        "statin", "cholesterol_absorption_inhibitor", "biguanide",
    },
"""

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SUMMARY OF ALL CHANGES TO MAKE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

backend/pipeline/scorer.py
  → REPLACE ENTIRE FILE with the provided scorer.py
    (adds DRUG_NAME_MECHANISM_HINTS + updated _score_mechanism_similarity)

run_validation.py
  → REPLACE ENTIRE FILE with the provided run_validation.py
    (adds --fast flag, disease caching, --disease filter)

backend/pipeline/data_fetcher.py
  → FIND method _apply_small_molecule_fallback
  → REPLACE BODY with the patched version above (supplements existing targets)

backend/pipeline/combo_scorer.py
  → ADD KNOWN_DRUG_CLASS_OVERRIDES dict after MECHANISM_KEYWORD_MAP
  → REPLACE classify_mechanism function (add drug_name param + override check)
  → UPDATE score_pair to pass drug_name to classify_mechanism
  → REPLACE _disease_context_penalty method (increase "other" penalty to 0.20)
  → REMOVE "other" from PAH, myeloma, AD, PD, HF, hypercholesterolemia
    in DISEASE_APPROPRIATE_CLASSES dict

ALSO (one-time fix):
  → DELETE /tmp/drug_repurposing_cache/chembl_approved_drugs.json
    This forces the pipeline to re-fetch and re-apply all fallbacks.
    The stale cache is why sildenafil/dexamethasone/spironolactone score 0.

EXPECTED IMPROVEMENT after fixes:
  combo_validation pass rate: 6/21 (28%) → ~16/21 (75%+)
  PAH top combos: IMATINIB+OLANZAPINE → sildenafil+bosentan+iloprost
  Myeloma: bortezomib+dexamethasone, thalidomide+dexamethasone appear
  AD: donepezil+memantine appears
  PD: rasagiline+amantadine appears

RUNTIME:
  python run_validation.py --fast   →  ~5-10 min  (was ~60-90 min)
  python run_validation.py          →  ~45-90 min  (full PPI + tissue)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")