#!/usr/bin/env python3
"""
apply_pipeline_patch.py
========================
Applies the two surgical fixes to production_pipeline.py.
Run from repo root: python apply_pipeline_patch.py

Changes applied:
  1. plan["candidates"] = safe_candidates[:20]  →  safe_candidates[:100]
  2. max_pool=70  →  max_pool=80  (in the _build_combo_pool call inside generate_treatment_plan)
  3. max_pool: int = 70,  →  max_pool: int = 80,  (in _build_combo_pool function signature)
"""

import re
from pathlib import Path

PIPELINE_FILE = Path("backend/pipeline/production_pipeline.py")

if not PIPELINE_FILE.exists():
    print(f"ERROR: {PIPELINE_FILE} not found. Run from repo root.")
    exit(1)

content = PIPELINE_FILE.read_text()
original = content

# FIX 1: candidates[:20] → candidates[:100]
content = content.replace(
    "plan[\"candidates\"] = safe_candidates[:20]",
    "plan[\"candidates\"] = safe_candidates[:100]  # v4.1: was [:20], increased for combo validation lookup"
)

# FIX 2: max_pool=70 → max_pool=80 in the combo_pool call
content = content.replace(
    "        max_pool=70,\n    )",
    "        max_pool=80,  # v4.1: increased from 70\n    )"
)

# FIX 3: function signature default max_pool: int = 70 → 80
content = content.replace(
    "    max_pool: int = 70,\n) -> List[Dict]:",
    "    max_pool: int = 80,  # v4.1: increased from 70\n) -> List[Dict]:"
)

if content == original:
    print("WARNING: No changes applied — the target strings were not found.")
    print("The file may already be patched, or line formatting differs.")
    print("Please apply the following changes manually:")
    print()
    print("  1. Find:    plan[\"candidates\"] = safe_candidates[:20]")
    print("     Replace: plan[\"candidates\"] = safe_candidates[:100]")
    print()
    print("  2. Find:    max_pool=70,  (in _build_combo_pool call)")
    print("     Replace: max_pool=80,")
    print()
    print("  3. Find:    max_pool: int = 70,  (in function signature)")
    print("     Replace: max_pool: int = 80,")
else:
    n_changes = 3
    PIPELINE_FILE.write_text(content)
    print(f"✅ Applied {n_changes} changes to {PIPELINE_FILE}")
    print()
    print("Changes applied:")
    print("  1. plan['candidates'] = safe_candidates[:20] → [:100]")
    print("  2. _build_combo_pool call: max_pool=70 → max_pool=80")
    print("  3. _build_combo_pool signature: max_pool: int = 70 → 80")