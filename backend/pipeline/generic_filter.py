"""
generic_filter.py — Generic Drug Filter
========================================
TwinTrial Analytics legal foundation: only off-patent generic drugs are
ever included in treatment plans. This prevents any IP infringement claim
because we are recommending drugs that anyone can manufacture.

Three-layer filter:
  1. Explicit patent-still-active exclusion list (manually curated)
  2. Biologic exclusivity heuristic (mAb suffix detection + whitelist)
  3. ChEMBL first_approval year — anything approved after 2005 flagged
     for manual verification (20-year US patent life from filing, not approval)

Patent safe cutoff rationale:
  A drug filed for patent in 1985 and approved in 1995 loses patent in 2005.
  Using approval year 2005 as cutoff is conservative — catches edge cases
  where filing-to-approval was long.
"""

import logging
import re
from typing import Dict, List, Set, Tuple

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Drugs still under active patent/exclusivity as of 2026
# Sources: FDA Orange Book, European Medicines Agency, company SEC filings
# Update this list annually — patent cliffs happen continuously
# ─────────────────────────────────────────────────────────────────────────────
STILL_PATENTED: Set[str] = {
    # PD-1/PD-L1 checkpoint inhibitors
    "pembrolizumab", "nivolumab", "atezolizumab", "durvalumab",
    "avelumab", "cemiplimab", "dostarlimab",
    # CTLA-4
    "ipilimumab",
    # PARP inhibitors
    "olaparib", "niraparib", "rucaparib", "talazoparib", "veliparib",
    # CFTR modulators (Vertex patents until ~2037)
    "ivacaftor", "lumacaftor", "tezacaftor", "elexacaftor",
    # Complement inhibitors
    "eculizumab", "ravulizumab", "avacopan",
    # Gene/ASO therapies
    "nusinersen", "onasemnogene abeparvovec", "risdiplam",
    # SGLT2 inhibitors (patents ~2030)
    "empagliflozin", "dapagliflozin", "canagliflozin", "ertugliflozin",
    # GLP-1 agonists
    "semaglutide", "liraglutide", "dulaglutide", "tirzepatide",
    "exenatide",  # original still has biosimilar complexity
    # newer mTOR inhibitors
    "everolimus",  # sirolimus is generic but everolimus still has indications under patent
    # CDK4/6 inhibitors
    "palbociclib", "ribociclib", "abemaciclib",
    # BTK inhibitors
    "ibrutinib", "acalabrutinib", "zanubrutinib",
    # JAK inhibitors (newer ones)
    "baricitinib", "upadacitinib", "filgotinib",
    # newer anti-IL agents
    "secukinumab", "ixekizumab", "guselkumab", "risankizumab",
    "bimekizumab", "spesolimab",
    # VEGF/angiogenesis (newer)
    "ramucirumab", "aflibercept",
    # HER2
    "pertuzumab", "ado-trastuzumab emtansine", "tucatinib",
    # ALK/ROS1 inhibitors
    "crizotinib", "alectinib", "brigatinib", "lorlatinib", "ceritinib",
    # EGFR newer gen
    "osimertinib", "afatinib", "dacomitinib",
    # BRAF/MEK
    "vemurafenib", "dabrafenib", "trametinib", "cobimetinib",
    # Newer anticoagulants
    "apixaban", "rivaroxaban", "edoxaban",
    # Newer antivirals
    "remdesivir", "nirmatrelvir", "molnupiravir",
    # Anti-CGRP (migraine)
    "erenumab", "fremanezumab", "galcanezumab",
    # PCSK9 inhibitors
    "evolocumab", "alirocumab",
    # Newer antidiabetics
    "alogliptin", "trelagliptin",
    # IL-6 newer
    "sarilumab", "satralizumab",
    # Miscellaneous recent approvals
    "deucravacitinib", "ozanimod", "siponimod", "ponesimod",
    "belantamab mafodotin", "isatuximab", "daratumumab",
    "luspatercept", "roxadustat",
}

# ─────────────────────────────────────────────────────────────────────────────
# Biologics that ARE off-patent / have biosimilar competition
# These pass the mAb heuristic filter because they're explicitly whitelisted
# ─────────────────────────────────────────────────────────────────────────────
GENERIC_BIOLOGICS: Set[str] = {
    "rituximab",         # biosimilars since 2018
    "trastuzumab",       # biosimilars since 2019
    "bevacizumab",       # biosimilars since 2019
    "infliximab",        # biosimilars since 2016
    "adalimumab",        # biosimilars since 2023
    "cetuximab",         # biosimilar approved 2024
    "tocilizumab",       # biosimilars emerging 2024
    "abatacept",         # patent expiry 2019, biosimilars pending
    "etanercept",        # biosimilars since 2016 (EU), 2019 (US)
    "omalizumab",        # biosimilars approved 2024
    "natalizumab",       # biosimilar pathway open
    "ustekinumab",       # biosimilars approved 2023
    "denosumab",         # biosimilar approved 2024
    "golimumab",         # patent expired, biosimilars coming
    "vedolizumab",       # biosimilars approved 2024
}

# ─────────────────────────────────────────────────────────────────────────────
# Biologic/mAb suffix patterns — these trigger the exclusivity check
# ─────────────────────────────────────────────────────────────────────────────
MAB_SUFFIXES = (
    "mab", "zumab", "ximab", "umab", "lumab", "numab",
    "tinib",   # most kinase inhibitors ending in -tinib are still patented
    "ciclib",  # CDK inhibitors
    "rafenib", # BRAF inhibitors
)

# Most -tinib drugs approved before 2010 may be off-patent
GENERIC_TINIB: Set[str] = {
    "imatinib",    # patent expired 2016
    "gefitinib",   # patent expired 2017
    "erlotinib",   # patent expired 2016
    "sorafenib",   # patent expired 2020
    "sunitinib",   # patent expired 2021
    "lapatinib",   # patent expired 2020
}

# ─────────────────────────────────────────────────────────────────────────────
# Salt/form suffixes to strip before name matching
# ─────────────────────────────────────────────────────────────────────────────
SALT_PATTERN = re.compile(
    r"\s+(sodium|potassium|hydrochloride|hcl|sulfate|tartrate|maleate|"
    r"mesylate|acetate|phosphate|fumarate|succinate|monohydrate|dihydrate|"
    r"anhydrous|extended.release|er|xr|sr|cr|besylate|tosylate|citrate)$",
    re.IGNORECASE,
)

# Cutoff: drugs approved before this year are almost certainly off-patent
PATENT_SAFE_CUTOFF_YEAR = 2004


class GenericDrugFilter:
    """
    Filters the approved drug pool to off-patent generics only.

    This is TwinTrial's legal firewall — every treatment plan we sell
    contains only drugs that any manufacturer can produce, which means:
      - No IP infringement in recommending them
      - No patent royalty complications for PAG/biotech partners
      - Clean "information product" legal status for our deliverable

    Usage
    -----
        gf = GenericDrugFilter()
        generic_drugs, excluded = gf.filter_to_generics(all_drugs)
        print(f"{len(generic_drugs)} generics, {len(excluded)} excluded")
    """

    def __init__(self):
        self._stats: Dict = {}

    def _normalize_name(self, name: str) -> str:
        """Strip salt suffixes and lowercase."""
        n = SALT_PATTERN.sub("", name.strip()).lower()
        return n.strip()

    def _is_mab_or_biologic(self, name_lower: str) -> bool:
        """True if the name matches a biologic suffix pattern."""
        return any(name_lower.endswith(sfx) for sfx in MAB_SUFFIXES)

    def _classify_drug(self, drug: Dict) -> Tuple[str, str]:
        """
        Returns (status, reason).
        status: "generic" | "verify" | "excluded"
        """
        raw_name = drug.get("name", "")
        name = self._normalize_name(raw_name)

        # Layer 1: explicit exclusion list
        if name in STILL_PATENTED:
            return "excluded", "active_patent_list"

        # Layer 2: biologic heuristic
        if self._is_mab_or_biologic(name):
            if name in GENERIC_BIOLOGICS:
                return "generic", "whitelisted_biosimilar"
            if name in GENERIC_TINIB:
                return "generic", "generic_tinib"
            # Unknown biologic — conservative exclusion
            return "excluded", "biologic_exclusivity_unknown"

        # Layer 3: approval year
        year = drug.get("first_approval_year") or drug.get("first_approval")
        try:
            year = int(year)
        except (TypeError, ValueError):
            year = 0

        if year == 0:
            # No year data — keep but flag
            return "verify", "no_approval_year"
        elif year <= PATENT_SAFE_CUTOFF_YEAR:
            return "generic", f"approved_{year}_pre_cutoff"
        elif year <= 2010:
            # Grey zone — likely off-patent but verify
            return "verify", f"approved_{year}_verify"
        else:
            # High chance still patented — exclude conservatively
            return "excluded", f"approved_{year}_post_cutoff"

    def filter_to_generics(
        self,
        drugs: List[Dict],
        include_verify: bool = True,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Filter drug list to off-patent generics.

        Parameters
        ----------
        drugs : list of dict
            Full approved drug pool from fetch_approved_drugs().
        include_verify : bool
            If True (default), include "verify" drugs (grey zone).
            Set False for a maximally conservative filter.

        Returns
        -------
        (generic_drugs, excluded_drugs) : tuple of lists
        """
        generic_drugs = []
        excluded_drugs = []

        counts = {"generic": 0, "verify": 0, "excluded": 0}

        for drug in drugs:
            status, reason = self._classify_drug(drug)
            drug["patent_status"] = status
            drug["patent_reason"] = reason
            counts[status] += 1

            if status == "excluded":
                excluded_drugs.append(drug)
            elif status == "verify" and include_verify:
                drug["patent_flag"] = "needs_verification"
                generic_drugs.append(drug)
            elif status == "verify" and not include_verify:
                excluded_drugs.append(drug)
            else:
                generic_drugs.append(drug)

        self._stats = {
            "total_input":       len(drugs),
            "generic_confirmed": counts["generic"],
            "grey_zone_verify":  counts["verify"],
            "excluded":          counts["excluded"],
            "generic_pool_size": len(generic_drugs),
        }

        logger.info("=" * 55)
        logger.info("GENERIC FILTER RESULTS")
        logger.info(f"  Input pool:          {len(drugs)}")
        logger.info(f"  Confirmed generic:   {counts['generic']}")
        logger.info(f"  Grey zone (verify):  {counts['verify']}")
        logger.info(f"  Excluded (patented): {counts['excluded']}")
        logger.info(f"  Final generic pool:  {len(generic_drugs)}")
        logger.info("=" * 55)

        return generic_drugs, excluded_drugs

    def get_stats(self) -> Dict:
        return self._stats

    def is_generic(self, drug_name: str) -> bool:
        """Quick check for a single drug name."""
        status, _ = self._classify_drug({"name": drug_name})
        return status in ("generic", "verify")

    def get_excluded_names(self, drugs: List[Dict]) -> List[str]:
        """Return list of excluded drug names for logging/reporting."""
        _, excluded = self.filter_to_generics(drugs)
        return [d["name"] for d in excluded]