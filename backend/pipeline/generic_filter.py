"""
generic_filter.py — Generic Drug Filter (v2 — Orange Book + Purple Book integrated)
======================================================================================
TwinTrial Analytics legal foundation: only off-patent generic drugs and
biosimilar-eligible biologics are ever included in treatment plans.

Three-layer filter (small molecules):
  1. Explicit patent-still-active exclusion list (manually curated)
  2. Biologic exclusivity heuristic (mAb suffix detection + whitelist)
  3. ChEMBL first_approval year — conservative 20-year patent window

Purple Book integration (biologics):
  4. PurpleBookFilter.is_biosimilar_available() — checks curated FDA biosimilar table
  5. Falls back to heuristic (BLA year + 12-year exclusivity) if not in curated table

Patent safe cutoff rationale:
  A drug filed for patent in 1985 and approved in 1995 loses US patent in 2005.
  Using approval year 2004 as cutoff is conservative — catches edge cases where
  filing-to-approval was long.

Biologic safe cutoff rationale:
  US Biologics Price Competition and Innovation Act (BPCIA) grants 12 years of
  reference product exclusivity. A biologic approved in 2008 has exclusivity
  until 2020. Using 2010 as our cutoff (12 years back from current ~2022+)
  is conservative.
"""

import logging
import re
from typing import Dict, List, Set, Tuple

from .purple_book_filter import (
    PURPLE_BOOK_CURATED,
    NOT_A_BIOLOGIC,
    biosimilar_available,
    is_biologic,
)

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
    # Complement inhibitors (new generation)
    "ravulizumab", "avacopan",
    # Gene/ASO therapies
    "nusinersen", "onasemnogene abeparvovec", "risdiplam",
    # SGLT2 inhibitors (patents ~2030)
    "empagliflozin", "dapagliflozin", "canagliflozin", "ertugliflozin",
    # GLP-1 agonists
    "semaglutide", "liraglutide", "dulaglutide", "tirzepatide",
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
    # newer anti-IL-17/23
    "sarilumab", "satralizumab",
    # Miscellaneous recent approvals
    "deucravacitinib", "ozanimod", "siponimod", "ponesimod",
    "belantamab mafodotin", "isatuximab", "daratumumab",
    "luspatercept", "roxadustat",
    # Newer mTOR indications still patented
    "everolimus",
}

# ─────────────────────────────────────────────────────────────────────────────
# Biologics confirmed eligible for TwinTrial (biosimilars available / off-patent)
# Sourced from purple_book_filter.py PURPLE_BOOK_CURATED
# ─────────────────────────────────────────────────────────────────────────────
GENERIC_BIOLOGICS: Set[str] = {
    name for name, data in PURPLE_BOOK_CURATED.items()
    if data.get("biosimilar_available", False)
} | {
    # Explicit additions not in Purple Book curated table
    "rituximab",
    "trastuzumab",
    "bevacizumab",
    "infliximab",
    "adalimumab",
    "etanercept",
    "cetuximab",
    "tocilizumab",
    "abatacept",
    "omalizumab",
    "ustekinumab",
    "denosumab",
    "natalizumab",    # no biosimilar but exclusivity expired → can recommend reference
    "golimumab",      # patent expired EU; US biosimilar path open
    "vedolizumab",
}

# ─────────────────────────────────────────────────────────────────────────────
# Biologic/mAb suffix patterns — these trigger the biologics check path
# ─────────────────────────────────────────────────────────────────────────────
MAB_SUFFIXES = (
    "mab", "zumab", "ximab", "umab", "lumab", "numab",
    "cept",       # fusion proteins: etanercept, abatacept
    "alfa", "beta", "gamma",  # cytokines: epoetin alfa
    "kinase",     # but most kinase inhibitors are small molecules
    "tinib",      # most -tinib are small molecules; handled by GENERIC_TINIB
    "ciclib",     # CDK inhibitors
    "rafenib",    # BRAF inhibitors
    "grastim",    # G-CSF
    "plase",      # plasminogen activators
)

# Generic (off-patent) kinase inhibitors and targeted therapy small molecules
GENERIC_TINIB: Set[str] = {
    "imatinib",    # patent expired 2016
    "gefitinib",   # patent expired 2017
    "erlotinib",   # patent expired 2016
    "sorafenib",   # patent expired 2020
    "sunitinib",   # patent expired 2021
    "lapatinib",   # patent expired 2020
    "dasatinib",   # patent expired 2024
}

# ─────────────────────────────────────────────────────────────────────────────
# Salt/form suffixes to strip before name matching
# ─────────────────────────────────────────────────────────────────────────────
SALT_PATTERN = re.compile(
    r"\s+(sodium|potassium|hydrochloride|hcl|sulfate|tartrate|maleate|"
    r"mesylate|acetate|phosphate|fumarate|succinate|monohydrate|dihydrate|"
    r"anhydrous|extended.release|er|xr|sr|cr|besylate|tosylate|citrate|"
    r"alfa|beta|gamma)$",
    re.IGNORECASE,
)

# Small molecules approved before this year are almost certainly off-patent
PATENT_SAFE_CUTOFF_YEAR = 2004

# Biologics approved before this year are almost certainly past 12-yr exclusivity
BIOLOGIC_SAFE_CUTOFF_YEAR = 2010


class GenericDrugFilter:
    """
    Filters the approved drug pool to off-patent generics and biosimilar-eligible
    biologics only.

    This is TwinTrial's legal firewall — every treatment plan contains only drugs
    that any manufacturer can produce (generics) or that have biosimilar
    competition (biologics), which means:
      - No IP infringement in recommending them
      - No patent royalty complications for PAG/biotech partners
      - Clean "information product" legal status for our deliverable

    The filter now integrates with PurpleBookFilter so biologics are correctly
    classified using the FDA Purple Book rather than pure name heuristics.

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
        name     = self._normalize_name(raw_name)

        # ── Layer 0: NOT_A_BIOLOGIC override ─────────────────────────────────
        # Small molecules that superficially match biologic suffixes
        if name in NOT_A_BIOLOGIC:
            return "generic", "confirmed_small_molecule"

        # ── Layer 1: Explicit exclusion list ─────────────────────────────────
        if name in STILL_PATENTED:
            return "excluded", "active_patent_list"

        # ── Layer 2: Biologics — Purple Book integration ─────────────────────
        if self._is_mab_or_biologic(name):

            # Check generic biologics list (biosimilar confirmed)
            if name in GENERIC_BIOLOGICS:
                return "generic", "whitelisted_biosimilar_purple_book"

            # Check generic tinib (small molecule kinase inhibitors)
            if name in GENERIC_TINIB:
                return "generic", "generic_tinib"

            # Try Purple Book curated lookup
            pb_data = PURPLE_BOOK_CURATED.get(name, {})
            if pb_data:
                if pb_data.get("biosimilar_available"):
                    return "generic", "purple_book_biosimilar_available"
                elif pb_data.get("reference_exclusivity_expired"):
                    # No biosimilar but exclusivity expired → borderline
                    return "verify", "purple_book_exclusivity_expired_no_biosimilar"
                else:
                    return "excluded", "purple_book_under_exclusivity"

            # Unknown biologic — check BLA year from drug data
            year = drug.get("first_approval_year") or drug.get("first_approval")
            try:
                year = int(year)
            except (TypeError, ValueError):
                year = 0

            if year > 0 and year <= BIOLOGIC_SAFE_CUTOFF_YEAR:
                # Old enough that 12-year exclusivity has expired
                return "verify", f"biologic_approved_{year}_exclusivity_likely_expired"
            elif year > BIOLOGIC_SAFE_CUTOFF_YEAR:
                return "excluded", f"biologic_approved_{year}_may_be_under_exclusivity"
            else:
                # No year data — conservative exclusion
                return "excluded", "biologic_exclusivity_unknown"

        # ── Layer 3: Approval year (small molecules) ──────────────────────────
        year = drug.get("first_approval_year") or drug.get("first_approval")
        try:
            year = int(year)
        except (TypeError, ValueError):
            year = 0

        if year == 0:
            return "verify", "no_approval_year"
        elif year <= PATENT_SAFE_CUTOFF_YEAR:
            return "generic", f"approved_{year}_pre_cutoff"
        elif year <= 2010:
            return "verify", f"approved_{year}_verify"
        else:
            return "excluded", f"approved_{year}_post_cutoff"

    def filter_to_generics(
        self,
        drugs: List[Dict],
        include_verify: bool = True,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Filter drug list to off-patent generics and biosimilar-eligible biologics.

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
        generic_drugs: List[Dict] = []
        excluded_drugs: List[Dict] = []
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

        logger.info("=" * 60)
        logger.info("GENERIC FILTER RESULTS (Orange Book + Purple Book)")
        logger.info(f"  Input pool:          {len(drugs)}")
        logger.info(f"  Confirmed generic:   {counts['generic']}")
        logger.info(f"  Grey zone (verify):  {counts['verify']}")
        logger.info(f"  Excluded (patented): {counts['excluded']}")
        logger.info(f"  Final generic pool:  {len(generic_drugs)}")
        logger.info("=" * 60)

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

    def biosimilar_available(self, drug_name: str) -> bool:
        """Check if a biosimilar is available for a reference biologic."""
        from .purple_book_filter import biosimilar_available as pb_check
        return pb_check(drug_name)

    def get_biosimilar_info(self, drug_name: str) -> Dict:
        """Return Purple Book data for a biologic."""
        from .purple_book_filter import PURPLE_BOOK_CURATED
        name_lower = self._normalize_name(drug_name)
        return PURPLE_BOOK_CURATED.get(name_lower, {})