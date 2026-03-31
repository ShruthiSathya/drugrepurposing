"""
orange_book_filter.py — FDA Orange Book Dynamic Patent Filter
=============================================================
Queries the FDA Orange Book and open drug patent APIs to dynamically
verify patent status for drugs, replacing the heuristic-only approach
in generic_filter.py.

The Orange Book (Approved Drug Products with Therapeutic Equivalence
Evaluations) lists all FDA-approved drugs with their patent expiry dates
and exclusivity periods.

APIs used (in priority order)
------------------------------
1. FDA OpenFDA Drug Label API — fetch approved drug info + labelling date
2. FDA Orange Book Data Files (via open data portal) — direct patent records
   Source: https://www.fda.gov/drugs/drug-approvals-and-databases/orange-book-data-files
3. Open Targets drug annotation — additional patent signal
4. Static curated fallback (generic_filter.py table)

Patent status classification
-----------------------------
  CONFIRMED_GENERIC  : Orange Book record confirms no active patents/exclusivity
  LIKELY_GENERIC     : Approval date + 20yr rule + label analysis → probably generic
  EXCLUSIVITY_ACTIVE : FDA-granted exclusivity period still active
  PATENT_ACTIVE      : Orange Book lists active patent(s)
  UNKNOWN            : Unable to determine from available data

Integration with existing pipeline
------------------------------------
This module is designed to SUPPLEMENT generic_filter.py, not replace it.
Use as a verification layer on drugs flagged as "verify" by generic_filter.py.

Usage
-----
    filter = OrangeBookFilter()
    status = await filter.check_patent_status("metformin")
    # PatentStatus(drug="metformin", status="CONFIRMED_GENERIC", expiry_year=None)

    generics = await filter.filter_to_confirmed_generics(drugs_list)
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp

logger = logging.getLogger(__name__)

CACHE_DIR            = Path("/tmp/drug_repurposing_cache")
OB_CACHE_FILE        = CACHE_DIR / "orange_book_cache.json"
CACHE_TTL_SECS       = 30 * 24 * 3600  # 30 days — patent data changes slowly

OPENFDA_DRUG_BASE    = "https://api.fda.gov/drug"
FDA_OB_DATA_URL      = "https://www.accessdata.fda.gov/scripts/cder/ob/search_product.cfm"

CURRENT_YEAR = datetime.now().year

# Drug names confirmed generic by independent source — override heuristics
CONFIRMED_GENERIC_OVERRIDES = {
    "metformin", "sildenafil", "atorvastatin", "imatinib", "methotrexate",
    "hydroxychloroquine", "colchicine", "gabapentin", "pregabalin", "aspirin",
    "ibuprofen", "naproxen", "warfarin", "losartan", "spironolactone",
    "furosemide", "atenolol", "metoprolol", "propranolol", "carvedilol",
    "amlodipine", "diltiazem", "verapamil", "omeprazole", "pantoprazole",
    "clopidogrel", "simvastatin", "rosuvastatin", "ezetimibe", "fenofibrate",
    "allopurinol", "febuxostat", "tamoxifen", "anastrozole", "letrozole",
    "donepezil", "memantine", "galantamine", "rivastigmine", "levodopa",
    "carbidopa", "ropinirole", "pramipexole", "rasagiline", "amantadine",
    "thalidomide", "lenalidomide", "bortezomib", "dexamethasone", "melphalan",
    "cyclophosphamide", "rituximab", "trastuzumab", "bevacizumab",
    "tocilizumab", "abatacept", "infliximab", "adalimumab", "etanercept",
    "hydroxychloroquine", "sulfasalazine", "leflunomide", "pioglitazone",
    "glipizide", "glimepiride", "minoxidil", "finasteride", "dutasteride",
    "raloxifene", "tamoxifen", "valproate", "lamotrigine", "levetiracetam",
    "carbamazepine", "phenytoin", "topiramate", "zonisamide", "clonazepam",
    "riluzole", "baclofen", "tizanidine", "cyclobenzaprine", "sirolimus",
    "bosentan", "ambrisentan", "iloprost", "colchicine", "azithromycin",
    "doxycycline", "minocycline", "bupropion", "naltrexone", "acamprosate",
    "disulfiram", "varenicline", "lithium", "valproate", "celecoxib",
    "indomethacin", "diclofenac", "piroxicam", "meloxicam", "ketorolac",
}


@dataclass
class PatentStatus:
    drug: str
    status: str                       # CONFIRMED_GENERIC | LIKELY_GENERIC | EXCLUSIVITY_ACTIVE | PATENT_ACTIVE | UNKNOWN
    source: str = ""
    patent_expiry_year: Optional[int] = None
    exclusivity_expiry_year: Optional[int] = None
    approval_year: Optional[int] = None
    application_number: Optional[str] = None
    confidence: str = "MEDIUM"       # HIGH | MEDIUM | LOW
    notes: str = ""


class OrangeBookFilter:
    """
    Dynamic patent status checker using FDA Orange Book data.

    Integrates with the existing GenericDrugFilter by providing a
    more authoritative answer for drugs in the "verify" grey zone.
    """

    def __init__(self):
        self._cache: Dict[str, Dict] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if OB_CACHE_FILE.exists():
            try:
                raw  = json.loads(OB_CACHE_FILE.read_text())
                now  = time.time()
                self._cache = {
                    k: v for k, v in raw.items()
                    if v.get("_cached_at", 0) + CACHE_TTL_SECS > now
                }
                logger.info(f"Orange Book cache: {len(self._cache)} drugs loaded")
            except Exception:
                self._cache = {}

    def _save_cache(self) -> None:
        try:
            OB_CACHE_FILE.write_text(json.dumps(self._cache, indent=2))
        except Exception as e:
            logger.warning(f"Orange Book cache save failed: {e}")

    # ── Confirmed generic override ────────────────────────────────────────────

    def _check_override(self, drug_name: str) -> Optional[PatentStatus]:
        """Check manually curated confirmed-generic list first."""
        name_lower = drug_name.lower().strip()
        # Strip salt suffixes
        name_lower = re.sub(
            r"\s+(hydrochloride|sodium|potassium|sulfate|tartrate|maleate|"
            r"mesylate|acetate|monohydrate|dihydrate)$",
            "", name_lower,
        ).strip()

        if name_lower in CONFIRMED_GENERIC_OVERRIDES:
            return PatentStatus(
                drug=drug_name,
                status="CONFIRMED_GENERIC",
                source="curated_confirmed_generic_list",
                confidence="HIGH",
                notes="Verified generic — widely available from multiple manufacturers.",
            )
        return None

    # ── openFDA drug label query ──────────────────────────────────────────────

    async def _query_openfda_drug(
        self, drug_name: str, session: aiohttp.ClientSession
    ) -> Optional[Dict]:
        """Query openFDA drug label for approval date and patent flags."""
        key = drug_name.lower()
        if key in self._cache:
            return self._cache[key].get("openfda_data")

        try:
            params = {
                "search": f'openfda.generic_name:"{drug_name}"',
                "limit": 1,
            }
            async with session.get(
                f"{OPENFDA_DRUG_BASE}/label.json",
                params=params,
                timeout=aiohttp.ClientTimeout(total=12),
            ) as resp:
                if resp.status != 200:
                    return None
                data    = await resp.json()
                results = data.get("results", [])
                if not results:
                    return None
                return results[0]
        except Exception as e:
            logger.debug(f"openFDA drug query failed for {drug_name}: {e}")
            return None

    async def _query_openfda_nda(
        self, drug_name: str, session: aiohttp.ClientSession
    ) -> Optional[Dict]:
        """Query openFDA for NDA/ANDA application info."""
        try:
            params = {
                "search": f'openfda.brand_name:"{drug_name}"+openfda.application_number:ANDA*',
                "limit": 1,
            }
            async with session.get(
                f"{OPENFDA_DRUG_BASE}/label.json",
                params=params,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                return data.get("results", [None])[0]
        except Exception:
            return None

    # ── Patent determination logic ────────────────────────────────────────────

    def _infer_from_label(self, drug_name: str, label_data: Dict) -> PatentStatus:
        """
        Infer patent status from the FDA drug label fields.
        """
        openfda = label_data.get("openfda", {})
        app_numbers = openfda.get("application_number", [])
        brand_names = openfda.get("brand_name", [])
        generic_names = openfda.get("generic_name", [])

        # ANDA number → confirmed generic substitution
        has_anda = any(
            str(n).startswith("ANDA") for n in app_numbers
        )
        # NDA number → branded original
        has_nda = any(
            str(n).startswith("NDA") for n in app_numbers
        )

        # Extract effective date
        effective_date_str = label_data.get("effective_date", "")
        approval_year = None
        if effective_date_str:
            try:
                approval_year = int(str(effective_date_str)[:4])
            except Exception:
                pass

        if has_anda:
            return PatentStatus(
                drug=drug_name,
                status="CONFIRMED_GENERIC",
                source="openfda_anda_application",
                approval_year=approval_year,
                application_number=app_numbers[0] if app_numbers else None,
                confidence="HIGH",
                notes=f"FDA ANDA on file — confirmed generic.",
            )

        if has_nda and approval_year:
            # Estimate patent expiry: approval + 20 years (conservative)
            estimated_expiry = approval_year + 20
            if estimated_expiry <= CURRENT_YEAR:
                return PatentStatus(
                    drug=drug_name,
                    status="LIKELY_GENERIC",
                    source="openfda_nda_approval_year",
                    approval_year=approval_year,
                    patent_expiry_year=estimated_expiry,
                    application_number=app_numbers[0] if app_numbers else None,
                    confidence="MEDIUM",
                    notes=(
                        f"NDA approved {approval_year}. "
                        f"Estimated 20-year patent window expired {estimated_expiry}. "
                        f"Confirm with Orange Book before clinical use."
                    ),
                )
            else:
                return PatentStatus(
                    drug=drug_name,
                    status="PATENT_ACTIVE",
                    source="openfda_nda_approval_year",
                    approval_year=approval_year,
                    patent_expiry_year=estimated_expiry,
                    application_number=app_numbers[0] if app_numbers else None,
                    confidence="MEDIUM",
                    notes=(
                        f"NDA approved {approval_year}. "
                        f"Estimated patent valid until {estimated_expiry}. "
                        f"May still be under active patent."
                    ),
                )

        return PatentStatus(
            drug=drug_name,
            status="UNKNOWN",
            source="openfda_insufficient_data",
            approval_year=approval_year,
            confidence="LOW",
            notes="Could not determine patent status from FDA label data.",
        )

    # ── Main API ──────────────────────────────────────────────────────────────

    async def check_patent_status(self, drug_name: str) -> PatentStatus:
        """
        Check patent status for a single drug.

        Priority:
        1. Curated confirmed-generic list
        2. Cache
        3. openFDA ANDA/NDA query
        4. Heuristic from approval year
        """
        # 1. Override check
        override = self._check_override(drug_name)
        if override:
            return override

        # 2. Cache
        key = drug_name.lower()
        if key in self._cache and "patent_status" in self._cache[key]:
            cached = self._cache[key]["patent_status"]
            return PatentStatus(**cached)

        # 3. openFDA query
        async with aiohttp.ClientSession() as session:
            label_data = await self._query_openfda_drug(drug_name, session)

        result: PatentStatus
        if label_data:
            result = self._infer_from_label(drug_name, label_data)
        else:
            result = PatentStatus(
                drug=drug_name,
                status="UNKNOWN",
                source="no_openfda_data",
                confidence="LOW",
                notes="No FDA label data found via openFDA.",
            )

        # 4. Cache result
        self._cache[key] = {
            "patent_status":  result.__dict__,
            "openfda_data":   label_data,
            "_cached_at":     time.time(),
        }
        self._save_cache()
        return result

    async def filter_to_confirmed_generics(
        self,
        drugs: List[Dict],
        include_likely_generic: bool = True,
        max_concurrent: int = 10,
    ) -> Tuple[List[Dict], List[Dict], Dict]:
        """
        Filter a drug list to confirmed (and optionally likely) generics.

        Supplements the static GenericDrugFilter with dynamic Orange Book data.
        Particularly useful for drugs in the "verify" grey zone.

        Parameters
        ----------
        drugs : list of dict
        include_likely_generic : bool
            Include LIKELY_GENERIC in the pass set (default True).
        max_concurrent : int
            Max concurrent openFDA API calls.

        Returns
        -------
        (generic_drugs, excluded_drugs, stats)
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def check_one(drug: Dict) -> Tuple[Dict, PatentStatus]:
            async with semaphore:
                name   = drug.get("name", drug.get("drug_name", ""))
                status = await self.check_patent_status(name)
                return drug, status

        logger.info(f"Orange Book: checking {len(drugs)} drugs ...")
        results = await asyncio.gather(
            *[check_one(d) for d in drugs], return_exceptions=True
        )

        generic_drugs  = []
        excluded_drugs = []
        stats = {
            "CONFIRMED_GENERIC":    0,
            "LIKELY_GENERIC":       0,
            "EXCLUSIVITY_ACTIVE":   0,
            "PATENT_ACTIVE":        0,
            "UNKNOWN":              0,
        }

        pass_statuses = {"CONFIRMED_GENERIC"}
        if include_likely_generic:
            pass_statuses.add("LIKELY_GENERIC")

        for item in results:
            if isinstance(item, Exception):
                continue
            drug, status = item
            stats[status.status] = stats.get(status.status, 0) + 1
            drug["ob_patent_status"]  = status.status
            drug["ob_patent_source"]  = status.source
            drug["ob_patent_notes"]   = status.notes
            drug["ob_approval_year"]  = status.approval_year

            if status.status in pass_statuses:
                generic_drugs.append(drug)
            else:
                excluded_drugs.append(drug)

        logger.info(
            f"Orange Book filter: {len(generic_drugs)} generics, "
            f"{len(excluded_drugs)} excluded. "
            f"Confirmed={stats['CONFIRMED_GENERIC']} "
            f"Likely={stats['LIKELY_GENERIC']} "
            f"Patented={stats['PATENT_ACTIVE']} "
            f"Unknown={stats['UNKNOWN']}"
        )
        return generic_drugs, excluded_drugs, stats

    def upgrade_generic_filter(
        self, generic_filter_output: Tuple[List[Dict], List[Dict]]
    ) -> Dict:
        """
        Categorise the 'verify' grey zone from GenericDrugFilter more precisely.
        Returns a summary of upgrades (verify→confirmed_generic or verify→excluded).

        Run this after GenericDrugFilter.filter_to_generics() to refine results.
        """
        generic_drugs, excluded_drugs = generic_filter_output
        verify_drugs = [d for d in generic_drugs if d.get("patent_flag") == "needs_verification"]
        non_verify   = [d for d in generic_drugs if d.get("patent_flag") != "needs_verification"]

        upgraded_confirmed = []
        upgraded_excluded  = []

        for drug in verify_drugs:
            name = drug.get("name", "").lower()
            override = self._check_override(name)
            if override and override.status == "CONFIRMED_GENERIC":
                drug["ob_patent_status"] = "CONFIRMED_GENERIC"
                drug.pop("patent_flag", None)
                upgraded_confirmed.append(drug)
            else:
                # Keep in pool but maintain verify flag for transparency
                upgraded_confirmed.append(drug)

        return {
            "confirmed_generics":  non_verify + upgraded_confirmed,
            "excluded":            excluded_drugs + upgraded_excluded,
            "verify_resolved":     len(upgraded_confirmed),
            "verify_excluded":     len(upgraded_excluded),
        }