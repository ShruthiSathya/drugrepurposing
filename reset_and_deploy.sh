#!/bin/bash
# reset_and_deploy.sh — Apply TwinTrial v4.0 fixes
# Run from repo root: bash reset_and_deploy.sh

set -e

echo "============================================================"
echo "TwinTrial Analytics Pipeline Fix Deployment"
echo "============================================================"

# Step 1: Backup old files
echo ""
echo "[1/4] Backing up existing files..."
mkdir -p backend/pipeline/_backup_v3
cp backend/pipeline/combo_scorer.py       backend/pipeline/_backup_v3/combo_scorer.py       2>/dev/null || true
cp backend/pipeline/production_pipeline.py backend/pipeline/_backup_v3/production_pipeline.py 2>/dev/null || true
cp combo_validation_dataset.py             _backup_v3_combo_validation_dataset.py             2>/dev/null || true
cp run_validation.py                       _backup_v3_run_validation.py                       2>/dev/null || true
echo "   Backed up to backend/pipeline/_backup_v3/"

# Step 2: Copy new files
echo ""
echo "[2/4] Deploying fixed files..."
cp twintrial_fixes/combo_scorer.py          backend/pipeline/combo_scorer.py
cp twintrial_fixes/production_pipeline.py   backend/pipeline/production_pipeline.py
cp twintrial_fixes/combo_validation_dataset.py  combo_validation_dataset.py
cp twintrial_fixes/run_validation.py        run_validation.py
echo "   Files deployed."

# Step 3: Clear stale drug cache (REQUIRED - old cache has empty targets)
echo ""
echo "[3/4] Clearing stale caches..."
CACHE_DIR="/tmp/drug_repurposing_cache"

# Keep disease cache (expensive to rebuild) but clear drug cache
# The drug cache needs rebuilding because we now pass mechanism_score to combo scorer
if [ -f "$CACHE_DIR/chembl_approved_drugs.json" ]; then
    DRUG_COUNT=$(python3 -c "import json; d=json.load(open('$CACHE_DIR/chembl_approved_drugs.json')); print(len(d))" 2>/dev/null || echo "unknown")
    echo "   Removing drug cache ($DRUG_COUNT drugs) — will be rebuilt on next run"
    rm "$CACHE_DIR/chembl_approved_drugs.json"
fi

# Clear insilico trial cache (may have stale combo priorities)
if [ -f "$CACHE_DIR/insilico_trial_cache.json" ]; then
    echo "   Removing stale trial cache..."
    rm "$CACHE_DIR/insilico_trial_cache.json"
fi

echo "   Caches cleared. Disease cache preserved (saves ~15 min)."

# Step 4: Quick sanity check
echo ""
echo "[4/4] Running quick sanity check..."
python3 -c "
import sys
sys.path.insert(0, '.')
from backend.pipeline.combo_scorer import CombinationScorer, SYNERGISTIC_PAIRS, classify_mechanism

# Test 1: PDE5i + ERA is synergistic
pair = frozenset({'pde5_inhibitor', 'endothelin_antagonist'})
assert pair in SYNERGISTIC_PAIRS, 'PDE5i+ERA not synergistic!'

# Test 2: AChEI + NMDA is synergistic (donepezil + memantine)
pair2 = frozenset({'acetylcholinesterase_inhibitor', 'nmda_antagonist'})
assert pair2 in SYNERGISTIC_PAIRS, 'AChEI+NMDA not synergistic!'

# Test 3: MAO-B + NMDA is synergistic (rasagiline + amantadine)
pair3 = frozenset({'maob_inhibitor', 'nmda_antagonist'})
assert pair3 in SYNERGISTIC_PAIRS, 'MAO-B+NMDA not synergistic!'

# Test 4: MRA + beta-blocker is synergistic (spironolactone + metoprolol)
pair4 = frozenset({'mineralocorticoid_antagonist', 'beta_blocker'})
assert pair4 in SYNERGISTIC_PAIRS, 'MRA+BB not synergistic!'

# Test 5: Context penalty for PAH — taxane should be penalised
scorer = CombinationScorer(disease_name='pulmonary arterial hypertension')
# Taxane (chemo) + PDE5i in PAH should get heavy penalty
taxane_drug = {'drug_name': 'cabazitaxel', 'mechanism': 'taxane microtubule', 'mechanism_score': 0.0}
pde5_drug   = {'drug_name': 'sildenafil',  'mechanism': 'pde5 inhibitor',     'mechanism_score': 1.0}
result = scorer.score_pair(taxane_drug, pde5_drug, [])
assert result['context_penalty'] > 0.20, f'Expected context penalty > 0.20, got {result[\"context_penalty\"]}'

# Test 6: VTd synergy — imid + proteasome + corticosteroid
imid_drug  = {'drug_name': 'thalidomide', 'mechanism': 'cereblon imid',     'mechanism_score': 0.6}
prot_drug  = {'drug_name': 'bortezomib',  'mechanism': 'proteasome',        'mechanism_score': 0.3}
dexa_drug  = {'drug_name': 'dexamethasone','mechanism': 'glucocorticoid',   'mechanism_score': 1.0}
scorer_mm  = CombinationScorer(disease_name='multiple myeloma')
result_td  = scorer_mm.score_pair(imid_drug, dexa_drug, [])
assert result_td['is_synergistic'], 'IMiD+dex should be synergistic!'
assert result_td['context_penalty'] < 0.10, f'IMiD+dex should have low context penalty, got {result_td[\"context_penalty\"]}'

print('All sanity checks PASSED!')
print(f'  ✓ PDE5i+ERA synergistic in PAH')
print(f'  ✓ AChEI+NMDA synergistic (Namzaric)')
print(f'  ✓ MAO-B+NMDA synergistic (PD standard)')
print(f'  ✓ MRA+BB synergistic (HF standard)')
print(f'  ✓ Taxane penalised in PAH context')
print(f'  ✓ IMiD+dex synergistic in myeloma (no penalty)')
"
echo ""
echo "============================================================"
echo "Deployment complete!"
echo ""
echo "Next steps:"
echo "  1. Run combo validation:"
echo "     python combo_validation_dataset.py --top-n 15"
echo ""
echo "  2. Run full single-drug validation:"
echo "     python run_validation.py --fast"
echo ""
echo "  3. Expected improvement:"
echo "     Combo pass rate: 28% → 65%+"
echo "     Single-drug F1:  0.947 (should stay the same)"
echo "============================================================"