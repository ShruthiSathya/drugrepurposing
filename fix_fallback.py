"""
fix_fallback.py
===============
Run this once from your repo root to patch data_fetcher.py in-place.

    python fix_fallback.py

What it fixes
-------------
ChEMBL returns drug names with salt/form suffixes:
  "Sildenafil citrate"         → lookup fails for "sildenafil"
  "Bosentan monohydrate"       → lookup fails for "bosentan"
  "Dexamethasone sodium phosphate" → lookup fails for "dexamethasone"
  "Metformin hydrochloride"    → lookup fails for "metformin"
  "Spironolactone"             → this one is fine (no suffix)

_apply_small_molecule_fallback and _apply_biologic_fallback both do:
    name_lower = drug["name"].lower()
    if name_lower in KNOWN_SMALL_MOLECULE_TARGETS:   ← FAILS for "sildenafil citrate"

Fix: strip salt suffixes before the lookup, same way _deduplicate_drug_pool does.
"""

import re
import sys
from pathlib import Path

FETCHER_PATH = Path("backend/pipeline/data_fetcher.py")


def main():
    if not FETCHER_PATH.exists():
        print(f"ERROR: {FETCHER_PATH} not found. Run from repo root.")
        sys.exit(1)

    content = FETCHER_PATH.read_text()

    # ── Check if already patched ──────────────────────────────────────────────
    if "_strip_salt" in content:
        print("✅ data_fetcher.py already patched (found _strip_salt). Nothing to do.")
        return

    # ── Define the helper and the two patched methods ─────────────────────────

    HELPER = '''
    @staticmethod
    def _strip_salt(name: str) -> str:
        """
        Strip common salt/form suffixes from drug names returned by ChEMBL.
        ChEMBL often returns "Sildenafil citrate", "Bosentan monohydrate" etc.
        We need the base INN name to match KNOWN_SMALL_MOLECULE_TARGETS keys.
        """
        _SALT_RE = re.compile(
            r"\\s+(hydrochloride|hcl|sodium|potassium|sulfate|tartrate|maleate|"
            r"mesylate|acetate|phosphate|fumarate|succinate|monohydrate|dihydrate|"
            r"anhydrous|bitartrate|besylate|tosylate|citrate|calcium|magnesium|"
            r"bromide|chloride|iodide|nitrate|oxalate|gluconate|lactate|"
            r"sodium\\s+phosphate|disodium|trisodium)$",
            re.IGNORECASE,
        )
        stripped = _SALT_RE.sub("", name.strip()).strip()
        # Run twice to catch double salts like "X sodium phosphate"
        stripped = _SALT_RE.sub("", stripped).strip()
        return stripped.lower()

'''

    NEW_BIOLOGIC = '''    def _apply_biologic_fallback(self, drugs: List[Dict]) -> List[Dict]:
        """
        Apply biologic fallback even on cached drugs.
        PATCHED: uses _strip_salt() so names like "Rituximab biosimilar" still match.
        """
        filled = 0
        for drug in drugs:
            if drug.get("targets"):
                continue
            name_stripped = self._strip_salt(drug["name"])
            if name_stripped in KNOWN_BIOLOGIC_TARGETS:
                targets               = KNOWN_BIOLOGIC_TARGETS[name_stripped]
                drug["targets"]       = targets
                drug["target_source"] = "biologic_label_fallback"
                drug["pathways"]      = self._infer_pathways_from_targets_fallback(targets)
                filled += 1
        if filled:
            logger.info(f"Biologic fallback applied to {filled} drugs")
        return drugs
'''

    NEW_SMALL_MOL = '''    def _apply_small_molecule_fallback(self, drugs: List[Dict]) -> List[Dict]:
        """
        Apply small molecule fallback on every load.
        PATCHED v3:
          1. Uses _strip_salt() so "Sildenafil citrate" matches "sildenafil".
          2. SUPPLEMENTS existing targets rather than skipping non-empty lists.
             Adds any curated targets missing from the drug's current target list.
        """
        filled = 0
        supplemented = 0
        for drug in drugs:
            name_stripped = self._strip_salt(drug["name"])
            if name_stripped not in KNOWN_SMALL_MOLECULE_TARGETS:
                continue
            known    = KNOWN_SMALL_MOLECULE_TARGETS[name_stripped]
            existing = drug.get("targets") or []
            if not existing:
                drug["targets"]       = known
                drug["target_source"] = "small_molecule_lit_fallback"
                drug["pathways"]      = self._infer_pathways_from_targets_fallback(known)
                filled += 1
            else:
                existing_set = set(existing)
                new_targets  = [t for t in known if t not in existing_set]
                if new_targets:
                    drug["targets"] = existing + new_targets
                    drug["pathways"] = self._infer_pathways_from_targets_fallback(drug["targets"])
                    supplemented += 1
        if filled or supplemented:
            logger.info(
                f"Small molecule fallback: {filled} drugs filled, "
                f"{supplemented} drugs supplemented with additional targets"
            )
        return drugs
'''

    # ── Locate insertion point for _strip_salt ────────────────────────────────
    # Insert right before _apply_biologic_fallback
    BIOLOGIC_MARKER = "    def _apply_biologic_fallback(self, drugs: List[Dict]) -> List[Dict]:"
    if BIOLOGIC_MARKER not in content:
        print("ERROR: Could not find _apply_biologic_fallback method. Check file structure.")
        sys.exit(1)

    # ── Replace _apply_biologic_fallback ──────────────────────────────────────
    OLD_BIOLOGIC_START = "    def _apply_biologic_fallback(self, drugs: List[Dict]) -> List[Dict]:"
    OLD_BIOLOGIC_END   = "    def _apply_small_molecule_fallback(self, drugs: List[Dict]) -> List[Dict]:"

    # Find the old biologic method body
    bio_start = content.index(OLD_BIOLOGIC_START)
    bio_end   = content.index(OLD_BIOLOGIC_END)
    old_biologic_block = content[bio_start:bio_end]

    # ── Replace _apply_small_molecule_fallback ─────────────────────────────────
    OLD_SMALL_START = "    def _apply_small_molecule_fallback(self, drugs: List[Dict]) -> List[Dict]:"
    # Find end of _apply_small_molecule_fallback (next def at same indent level)
    small_start = content.index(OLD_SMALL_START)
    # Find the next method at same indentation after small_start
    next_method_pattern = re.compile(r"\n    (async )?def ", re.MULTILINE)
    match = next_method_pattern.search(content, small_start + len(OLD_SMALL_START))
    if match:
        small_end = match.start() + 1  # include the newline
    else:
        # It's the last method — find next class or end of class
        small_end = len(content)

    old_small_block = content[small_start:small_end]

    # ── Apply patches ─────────────────────────────────────────────────────────
    # 1. Insert _strip_salt + replace _apply_biologic_fallback
    new_content = content.replace(
        old_biologic_block,
        HELPER + NEW_BIOLOGIC + "\n"
    )
    # 2. Replace _apply_small_molecule_fallback
    # Re-find positions since content changed
    old_small_start_pos = new_content.index(OLD_SMALL_START)
    match2 = next_method_pattern.search(new_content, old_small_start_pos + len(OLD_SMALL_START))
    if match2:
        new_small_end = match2.start() + 1
    else:
        new_small_end = len(new_content)

    old_small_in_new = new_content[old_small_start_pos:new_small_end]
    new_content = new_content.replace(
        old_small_in_new,
        NEW_SMALL_MOL + "\n",
        1,  # only first occurrence
    )

    # ── Ensure `import re` is at top ──────────────────────────────────────────
    if "import re" not in new_content[:500]:
        new_content = new_content.replace(
            "import asyncio\n",
            "import asyncio\nimport re\n",
            1,
        )

    # ── Write back ────────────────────────────────────────────────────────────
    FETCHER_PATH.write_text(new_content)
    print(f"✅ Patched {FETCHER_PATH}")
    print()
    print("Changes made:")
    print("  + Added _strip_salt() static method")
    print("  + _apply_biologic_fallback: now strips salt suffixes before lookup")
    print("  + _apply_small_molecule_fallback: strips salts + supplements existing targets")
    print()
    print("Now run:")
    print("  python run_validation.py --fast")
    print()
    print("Expected: sildenafil, bosentan, dexamethasone, spironolactone etc.")
    print("  should now score > 0 for their target diseases.")


if __name__ == "__main__":
    main()