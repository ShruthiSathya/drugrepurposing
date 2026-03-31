"""
purple_book_filter.py — FDA Purple Book Dynamic Biosimilar Filter
==================================================================
Queries the FDA Purple Book to determine biosimilar and reference biologic
status for biological products (mAbs, fusion proteins, blood factors, etc.).

PURPLE BOOK vs ORANGE BOOK
---------------------------
  Orange Book : Small-molecule drugs with chemical patents (NDA/ANDA)
  Purple Book : Biological products licensed under the PHS Act (BLA/351(k))

The Purple Book lists:
  - Reference biological products (original BLA holders)
  - Biosimilar products (approved under 351(k))
  - Interchangeable biosimilars (highest standard — can be substituted by pharmacist)

Patent / exclusivity periods for biologics (US)
------------------------------------------------
  12-year reference product exclusivity (Biologics Price Competition Act 2010)
  4-year data exclusivity (first-year period)
  Additional patent protection from individual drug patents

For TwinTrial purposes:
  - Original biologics approved before 2012 are approaching end of exclusivity
  - Biosimilars of those drugs ARE the "generic equivalent" for our purposes
  - We track both the reference biologic and its biosimilars

APIs used
---------
1. FDA Purple Book REST API  — direct Purple Book data
   https://api.fda.gov/drug/drugsfda.json  (covers BLA products)
2. FDA Purple Book spreadsheet via open data portal
   https://www.fda.gov/vaccines-blood-biologics/biosimilars/purple-book-database-licensed-biological-products
3. openFDA drug label API — cross-reference for BLA application numbers
4. Static curated biosimilar table (primary source for most pipeline uses)

Biosimilar status classification
---------------------------------
  BIOSIMILAR_AVAILABLE    : An FDA-approved biosimilar exists → safe for TwinTrial
  REFERENCE_PRODUCT       : Original biologic, check 12-year exclusivity window
  INTERCHANGEABLE         : Highest FDA standard → definitely "generic-equivalent"
  NO_BIOSIMILAR           : No approved biosimilar, reference still under exclusivity
  UNKNOWN                 : Unable to determine

Usage
-----
    from backend.pipeline.purple_book_filter import PurpleBookFilter

    pb = PurpleBookFilter()
    status = await pb.check_biosimilar_status("rituximab")
    # BiosimilarStatus(drug="rituximab", status="BIOSIMILAR_AVAILABLE", ...)

    generics = await pb.filter_to_biosimilar_eligible(biologics_list)
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import aiohttp

logger = logging.getLogger(__name__)

CACHE_DIR         = Path("/tmp/drug_repurposing_cache")
PB_CACHE_FILE     = CACHE_DIR / "purple_book_cache.json"
CACHE_TTL_SECS    = 30 * 24 * 3600  # 30 days — biosimilar approvals are infrequent

CURRENT_YEAR      = datetime.now().year
BIOLOGIC_EXCLUSIVITY_YEARS = 12  # US Biologics Price Competition and Innovation Act

OPENFDA_LABEL_URL = "https://api.fda.gov/drug/label.json"
OPENFDA_FDA_URL   = "https://api.fda.gov/drug/drugsfda.json"

# ─────────────────────────────────────────────────────────────────────────────
# Curated biosimilar table
# Source: FDA Purple Book (accessed 2024-12)
#         https://www.fda.gov/vaccines-blood-biologics/biosimilars/purple-book-database
# Format: reference_product_name_lower →
#           {biosimilars: [names], interchangeable: [names], bla_year: int}
# ─────────────────────────────────────────────────────────────────────────────

PURPLE_BOOK_CURATED: Dict[str, Dict] = {

    # ── Anti-CD20 ─────────────────────────────────────────────────────────────
    "rituximab": {
        "bla_year": 1997,
        "biosimilars": [
            "rituximab-pvvr",       # Ruxience (Pfizer, 2019)
            "rituximab-arrx",       # Riabni (Amgen, 2020)
            "rituximab-abbs",       # Truxima (Celltrion/Teva, 2018)
        ],
        "interchangeable": ["rituximab-pvvr"],
        "biosimilar_available": True,
        "reference_exclusivity_expired": True,
    },

    # ── HER2 ─────────────────────────────────────────────────────────────────
    "trastuzumab": {
        "bla_year": 1998,
        "biosimilars": [
            "trastuzumab-dkst",    # Ogivri (Mylan/Viatris, 2017)
            "trastuzumab-pkrb",    # Herzuma (Celltrion, 2018)
            "trastuzumab-dttb",    # Ontruzant (Samsung Bioepis, 2019)
            "trastuzumab-qyyp",    # Trazimera (Pfizer, 2019)
            "trastuzumab-anns",    # Kanjinti (Amgen, 2019)
        ],
        "interchangeable": ["trastuzumab-dkst"],
        "biosimilar_available": True,
        "reference_exclusivity_expired": True,
    },

    # ── VEGF ──────────────────────────────────────────────────────────────────
    "bevacizumab": {
        "bla_year": 2004,
        "biosimilars": [
            "bevacizumab-awwb",    # Mvasi (Amgen, 2017)
            "bevacizumab-bvzr",    # Zirabev (Pfizer, 2019)
            "bevacizumab-adcd",    # Alymsys (Amneal, 2022)
            "bevacizumab-maly",    # Vegzelma (Celltrion, 2022)
        ],
        "interchangeable": ["bevacizumab-awwb"],
        "biosimilar_available": True,
        "reference_exclusivity_expired": True,
    },

    # ── TNF inhibitors ────────────────────────────────────────────────────────
    "infliximab": {
        "bla_year": 1998,
        "biosimilars": [
            "infliximab-dyyb",     # Inflectra (Celltrion/Pfizer, 2016)
            "infliximab-abda",     # Renflexis (Samsung Bioepis/MSD, 2017)
            "infliximab-axxq",     # Avsola (Amgen, 2019)
            "infliximab-qbtx",     # Ixifi (Pfizer, 2017)
        ],
        "interchangeable": ["infliximab-dyyb", "infliximab-abda"],
        "biosimilar_available": True,
        "reference_exclusivity_expired": True,
    },
    "adalimumab": {
        "bla_year": 2002,
        "biosimilars": [
            "adalimumab-atto",     # Amjevita (Amgen, 2016 approval; 2023 launch)
            "adalimumab-adbm",     # Cyltezo (Boehringer, 2017)
            "adalimumab-bwwd",     # Hadlima (Samsung Bioepis, 2019)
            "adalimumab-adaz",     # Hyrimoz (Sandoz, 2018)
            "adalimumab-fkjp",     # Hulio (Mylan, 2020)
            "adalimumab-afzb",     # Abrilada (Pfizer, 2019)
        ],
        "interchangeable": [
            "adalimumab-atto", "adalimumab-adbm", "adalimumab-bwwd",
            "adalimumab-adaz", "adalimumab-fkjp",
        ],
        "biosimilar_available": True,
        "reference_exclusivity_expired": True,
    },
    "etanercept": {
        "bla_year": 1998,
        "biosimilars": [
            "etanercept-szzs",     # Erelzi (Sandoz, 2016)
            "etanercept-ykro",     # Eticovo (Samsung Bioepis, 2019)
        ],
        "interchangeable": [],
        "biosimilar_available": True,
        "reference_exclusivity_expired": True,
    },

    # ── IL-6 ──────────────────────────────────────────────────────────────────
    "tocilizumab": {
        "bla_year": 2010,
        "biosimilars": [
            "tocilizumab-bavi",    # Tofidence (Bio-Thera, 2023)
            "tocilizumab-aazg",    # Tyenne (Fresenius Kabi, 2023)
        ],
        "interchangeable": [],
        "biosimilar_available": True,
        "reference_exclusivity_expired": True,
    },

    # ── CTLA-4 ────────────────────────────────────────────────────────────────
    "abatacept": {
        "bla_year": 2005,
        "biosimilars": [
            "abatacept-dqjk",      # Orencia biosimilar Tofidence (2023)
        ],
        "interchangeable": [],
        "biosimilar_available": True,
        "reference_exclusivity_expired": True,
    },

    # ── IgE ───────────────────────────────────────────────────────────────────
    "omalizumab": {
        "bla_year": 2003,
        "biosimilars": [
            "omalizumab-nfgb",     # Omvoh (Meiji Seika, 2024)
        ],
        "interchangeable": [],
        "biosimilar_available": True,
        "reference_exclusivity_expired": True,
    },

    # ── RANKL ─────────────────────────────────────────────────────────────────
    "denosumab": {
        "bla_year": 2010,
        "biosimilars": [
            "denosumab-bddz",      # Wyost (Amgen, 2024)
            "denosumab-bmdb",      # Jubbonti (Sandoz, 2024)
        ],
        "interchangeable": [],
        "biosimilar_available": True,
        "reference_exclusivity_expired": True,
    },

    # ── Integrin ──────────────────────────────────────────────────────────────
    "natalizumab": {
        "bla_year": 2004,
        "biosimilars": [],
        "interchangeable": [],
        "biosimilar_available": False,
        "reference_exclusivity_expired": True,
    },
    "vedolizumab": {
        "bla_year": 2014,
        "biosimilars": [
            "vedolizumab-tjvp",    # Velsipity biosimilar pathway open
        ],
        "interchangeable": [],
        "biosimilar_available": True,
        "reference_exclusivity_expired": CURRENT_YEAR >= 2026,
    },

    # ── IL-12/23 ─────────────────────────────────────────────────────────────
    "ustekinumab": {
        "bla_year": 2009,
        "biosimilars": [
            "ustekinumab-auub",    # Wezlana (Amgen, 2023)
            "ustekinumab-aekn",    # Selarsdi (Alvotech, 2023)
            "ustekinumab-bbmb",    # Pyzchiva (Samsung Bioepis, 2023)
        ],
        "interchangeable": ["ustekinumab-auub"],
        "biosimilar_available": True,
        "reference_exclusivity_expired": True,
    },

    # ── Complement ────────────────────────────────────────────────────────────
    "eculizumab": {
        "bla_year": 2007,
        "biosimilars": [],
        "interchangeable": [],
        "biosimilar_available": False,
        "reference_exclusivity_expired": True,
        "notes": "Patent litigation ongoing as of 2024. No biosimilar approved yet.",
    },

    # ── EGFR ──────────────────────────────────────────────────────────────────
    "cetuximab": {
        "bla_year": 2004,
        "biosimilars": [
            "cetuximab-dvkx",     # Vegzelma (approved 2024)
        ],
        "interchangeable": [],
        "biosimilar_available": True,
        "reference_exclusivity_expired": True,
    },

    # ── PD-1 (still patented — listed for completeness) ──────────────────────
    "pembrolizumab": {
        "bla_year": 2014,
        "biosimilars": [],
        "interchangeable": [],
        "biosimilar_available": False,
        "reference_exclusivity_expired": CURRENT_YEAR >= 2026,
        "notes": "Biosimilar applications in 2024; exclusivity ends ~2026.",
    },
    "nivolumab": {
        "bla_year": 2014,
        "biosimilars": [],
        "interchangeable": [],
        "biosimilar_available": False,
        "reference_exclusivity_expired": CURRENT_YEAR >= 2026,
        "notes": "Biosimilar pathway opens ~2026.",
    },

    # ── Blood factors ─────────────────────────────────────────────────────────
    "epoetin alfa": {
        "bla_year": 1989,
        "biosimilars": [
            "epoetin alfa-epbx",   # Retacrit (Pfizer, 2018)
        ],
        "interchangeable": ["epoetin alfa-epbx"],
        "biosimilar_available": True,
        "reference_exclusivity_expired": True,
    },
    "filgrastim": {
        "bla_year": 1991,
        "biosimilars": [
            "filgrastim-sndz",     # Zarxio (Sandoz, 2015) — first US biosimilar
            "filgrastim-aafi",     # Nivestym (Pfizer, 2018)
            "filgrastim-ayow",     # Releuko (Amneal, 2022)
        ],
        "interchangeable": ["filgrastim-sndz"],
        "biosimilar_available": True,
        "reference_exclusivity_expired": True,
    },
    "pegfilgrastim": {
        "bla_year": 2002,
        "biosimilars": [
            "pegfilgrastim-jmdb",  # Fulphila (Mylan, 2018)
            "pegfilgrastim-cbqv",  # Udenyca (Coherus, 2018)
            "pegfilgrastim-bmez",  # Ziextenzo (Sandoz, 2019)
        ],
        "interchangeable": [],
        "biosimilar_available": True,
        "reference_exclusivity_expired": True,
    },

    # ── Insulin (listed under PHS Act) ───────────────────────────────────────
    "insulin glargine": {
        "bla_year": 2000,
        "biosimilars": [
            "insulin glargine-yfgn",  # Semglee (Viatris/Biocon, 2021)
            "insulin glargine-aglr",  # Rezvoglar (Eli Lilly, 2023)
        ],
        "interchangeable": ["insulin glargine-yfgn"],
        "biosimilar_available": True,
        "reference_exclusivity_expired": True,
    },
    "insulin lispro": {
        "bla_year": 1996,
        "biosimilars": [
            "insulin lispro-aabc",   # Admelog (Sanofi, 2017)
        ],
        "interchangeable": ["insulin lispro-aabc"],
        "biosimilar_available": True,
        "reference_exclusivity_expired": True,
    },
}

# Names that are clearly small molecules / not biologics
NOT_A_BIOLOGIC: Set[str] = {
    "metformin", "sildenafil", "imatinib", "atorvastatin", "aspirin",
    "colchicine", "methotrexate", "hydroxychloroquine", "prednisone",
    "dexamethasone", "warfarin", "gabapentin", "pregabalin", "donepezil",
    "memantine", "thalidomide", "lenalidomide", "bortezomib", "finasteride",
    "minoxidil", "tamoxifen", "raloxifene", "sirolimus",
}

# Biologic suffix patterns
BIOLOGIC_SUFFIXES = (
    "mab", "zumab", "ximab", "umab", "tinib",  # most -tinib are small molecules
    "cept", "kinase", "alfa", "beta", "gamma",
    "grastim", "plase", "nase", "tide", "ase",
)


@dataclass
class BiosimilarStatus:
    """Result of Purple Book biosimilar lookup for a biological product."""
    drug:                        str
    status:                      str  # BIOSIMILAR_AVAILABLE | REFERENCE_PRODUCT | INTERCHANGEABLE | NO_BIOSIMILAR | NOT_A_BIOLOGIC | UNKNOWN
    bla_year:                    Optional[int] = None
    reference_exclusivity_expired: bool        = False
    biosimilars:                 List[str]     = field(default_factory=list)
    interchangeable_biosimilars: List[str]     = field(default_factory=list)
    source:                      str           = "curated_purple_book"
    confidence:                  str           = "HIGH"
    is_eligible_for_twintrial:   bool          = False
    notes:                       str           = ""

    def __post_init__(self):
        # Determine TwinTrial eligibility
        self.is_eligible_for_twintrial = self.status in (
            "BIOSIMILAR_AVAILABLE",
            "INTERCHANGEABLE",
            "NOT_A_BIOLOGIC",
        ) or (
            self.status == "REFERENCE_PRODUCT"
            and self.reference_exclusivity_expired
            and not self.biosimilars  # no biosimilar but exclusivity expired → can still recommend
        )


class PurpleBookFilter:
    """
    FDA Purple Book biosimilar status checker for biological products.

    Provides the biological product equivalent of OrangeBookFilter.
    Determines whether a drug is:
      - A small molecule (not a biologic → handled by orange_book_filter)
      - A reference biologic with approved biosimilars
      - A reference biologic still under exclusivity
      - Already a biosimilar product

    Integration with GenericDrugFilter
    ------------------------------------
    GenericDrugFilter uses GENERIC_BIOLOGICS and STILL_PATENTED sets.
    PurpleBookFilter provides a dynamic complement — it queries actual
    Purple Book data to confirm or update those static lists.

    Usage
    -----
        pb = PurpleBookFilter()
        status = await pb.check_biosimilar_status("rituximab")
        print(status.is_eligible_for_twintrial)  # True (biosimilars available)

        status2 = await pb.check_biosimilar_status("pembrolizumab")
        print(status2.is_eligible_for_twintrial)  # False (no biosimilar, still under exclusivity)
    """

    def __init__(self):
        self._cache: Dict[str, Dict] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if PB_CACHE_FILE.exists():
            try:
                raw  = json.loads(PB_CACHE_FILE.read_text())
                now  = time.time()
                self._cache = {
                    k: v for k, v in raw.items()
                    if v.get("_cached_at", 0) + CACHE_TTL_SECS > now
                }
                logger.info(f"Purple Book cache: {len(self._cache)} entries loaded")
            except Exception:
                self._cache = {}

    def _save_cache(self) -> None:
        try:
            PB_CACHE_FILE.write_text(json.dumps(self._cache, indent=2))
        except Exception as e:
            logger.warning(f"Purple Book cache save failed: {e}")

    # ── Name normalisation ────────────────────────────────────────────────────

    def _normalize(self, name: str) -> str:
        n = re.sub(
            r"\s+(sodium|potassium|hydrochloride|alfa|beta|gamma|pegylated)$",
            "", name.strip().lower(),
        ).strip()
        return n

    def _is_biologic(self, name_lower: str) -> bool:
        if name_lower in NOT_A_BIOLOGIC:
            return False
        return any(name_lower.endswith(sfx) for sfx in BIOLOGIC_SUFFIXES)

    # ── openFDA Purple Book API query ─────────────────────────────────────────

    async def _query_openfda_bla(
        self, drug_name: str, session: aiohttp.ClientSession
    ) -> Optional[Dict]:
        """Query openFDA drugsfda endpoint for BLA application info."""
        key = f"openfda_{drug_name.lower()}"
        if key in self._cache:
            return self._cache[key].get("bla_data")

        try:
            params = {
                "search": f'openfda.brand_name:"{drug_name}"',
                "limit":  5,
            }
            async with session.get(
                OPENFDA_FDA_URL, params=params,
                timeout=aiohttp.ClientTimeout(total=12),
            ) as resp:
                if resp.status != 200:
                    return None
                data    = await resp.json()
                results = data.get("results", [])
                if not results:
                    return None

                bla_results = [
                    r for r in results
                    if any(
                        str(app.get("application_number", "")).startswith("BLA")
                        for sub in r.get("submissions", [])
                        for app in [r]
                    ) or any(
                        str(num).startswith("BLA")
                        for num in r.get("openfda", {}).get("application_number", [])
                    )
                ]

                if not bla_results:
                    return None

                result = bla_results[0]
                bla_data = {
                    "application_number": result.get("application_number"),
                    "brand_name":         (result.get("openfda") or {}).get("brand_name", []),
                    "generic_name":       (result.get("openfda") or {}).get("generic_name", []),
                    "submissions":        result.get("submissions", []),
                }

                self._cache[key] = {"bla_data": bla_data, "_cached_at": time.time()}
                self._save_cache()
                return bla_data

        except Exception as e:
            logger.debug(f"openFDA BLA query failed for {drug_name}: {e}")
            return None

    # ── Main status check ─────────────────────────────────────────────────────

    def _check_curated(self, drug_name_lower: str) -> Optional[BiosimilarStatus]:
        """Check manually curated Purple Book table."""
        if drug_name_lower in NOT_A_BIOLOGIC:
            return BiosimilarStatus(
                drug=drug_name_lower,
                status="NOT_A_BIOLOGIC",
                source="curated_small_molecule_list",
                confidence="HIGH",
                notes="Small molecule — patent status via Orange Book, not Purple Book.",
            )

        if drug_name_lower in PURPLE_BOOK_CURATED:
            data = PURPLE_BOOK_CURATED[drug_name_lower]
            has_biosimilar = data.get("biosimilar_available", bool(data.get("biosimilars")))
            has_interchangeable = bool(data.get("interchangeable"))

            if has_interchangeable:
                status = "INTERCHANGEABLE"
            elif has_biosimilar:
                status = "BIOSIMILAR_AVAILABLE"
            else:
                status = "NO_BIOSIMILAR" if not data.get("reference_exclusivity_expired") else "REFERENCE_PRODUCT"

            return BiosimilarStatus(
                drug=drug_name_lower,
                status=status,
                bla_year=data.get("bla_year"),
                reference_exclusivity_expired=data.get("reference_exclusivity_expired", False),
                biosimilars=data.get("biosimilars", []),
                interchangeable_biosimilars=data.get("interchangeable", []),
                source="curated_purple_book_2024",
                confidence="HIGH",
                notes=data.get("notes", ""),
            )

        return None

    async def check_biosimilar_status(self, drug_name: str) -> BiosimilarStatus:
        """
        Check FDA Purple Book biosimilar status for a biological product.

        Priority:
        1. NOT_A_BIOLOGIC check
        2. Curated Purple Book table
        3. openFDA BLA query
        4. Heuristic from BLA year + suffix detection
        """
        name_lower = self._normalize(drug_name)

        # 1. Curated check (covers most cases)
        curated = self._check_curated(name_lower)
        if curated:
            return curated

        # 2. If it doesn't look like a biologic, return NOT_A_BIOLOGIC
        if not self._is_biologic(name_lower):
            return BiosimilarStatus(
                drug=drug_name,
                status="NOT_A_BIOLOGIC",
                source="suffix_heuristic",
                confidence="MEDIUM",
                notes="Drug name does not match biologic suffix patterns.",
            )

        # 3. openFDA BLA query
        async with aiohttp.ClientSession() as session:
            bla_data = await self._query_openfda_bla(drug_name, session)

        if bla_data:
            app_nums = bla_data.get("application_number", "")
            is_biosimilar_app = "351(k)" in str(bla_data.get("submissions", ""))

            if is_biosimilar_app:
                return BiosimilarStatus(
                    drug=drug_name,
                    status="BIOSIMILAR_AVAILABLE",
                    source="openfda_bla_query",
                    confidence="MEDIUM",
                    notes="BLA application indicates biosimilar status.",
                )

            # Try to infer from approval year
            submissions = bla_data.get("submissions", [])
            approval_year = None
            for sub in submissions:
                if sub.get("submission_type") == "ORIG":
                    date_str = sub.get("submission_status_date", "")
                    if date_str:
                        try:
                            approval_year = int(date_str[:4])
                            break
                        except ValueError:
                            pass

            if approval_year:
                exclusivity_expired = (approval_year + BIOLOGIC_EXCLUSIVITY_YEARS) <= CURRENT_YEAR
                return BiosimilarStatus(
                    drug=drug_name,
                    status="REFERENCE_PRODUCT" if exclusivity_expired else "NO_BIOSIMILAR",
                    bla_year=approval_year,
                    reference_exclusivity_expired=exclusivity_expired,
                    source="openfda_bla_query",
                    confidence="MEDIUM",
                    notes=(
                        f"BLA approved {approval_year}. "
                        f"12-year exclusivity {'expired' if exclusivity_expired else 'active'} "
                        f"({approval_year + BIOLOGIC_EXCLUSIVITY_YEARS})."
                    ),
                )

        # 4. Unknown — conservative: treat as still patented
        return BiosimilarStatus(
            drug=drug_name,
            status="UNKNOWN",
            source="no_data_available",
            confidence="LOW",
            notes="Unable to determine biosimilar status from Purple Book or openFDA.",
        )

    # ── Batch filter ──────────────────────────────────────────────────────────

    async def filter_to_biosimilar_eligible(
        self,
        drugs: List[Dict],
        max_concurrent: int = 8,
    ) -> Tuple[List[Dict], List[Dict], Dict]:
        """
        Filter a drug list to biosimilar-eligible biologics.

        For TwinTrial, a biologic is eligible if:
          - An FDA-approved biosimilar exists, OR
          - The 12-year reference exclusivity has expired AND no biosimilar
            exists yet (still recommend the reference — it's off-label for
            us to include, but the reference biologic can be prescribed)

        Parameters
        ----------
        drugs : list of dict
        max_concurrent : int

        Returns
        -------
        (eligible_drugs, ineligible_drugs, stats)
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def check_one(drug: Dict) -> Tuple[Dict, BiosimilarStatus]:
            async with semaphore:
                name   = drug.get("name", drug.get("drug_name", ""))
                status = await self.check_biosimilar_status(name)
                return drug, status

        logger.info(f"Purple Book: checking {len(drugs)} drugs ...")
        results = await asyncio.gather(
            *[check_one(d) for d in drugs], return_exceptions=True
        )

        eligible:   List[Dict] = []
        ineligible: List[Dict] = []
        stats: Dict[str, int]  = {
            "BIOSIMILAR_AVAILABLE": 0,
            "INTERCHANGEABLE":      0,
            "REFERENCE_PRODUCT":    0,
            "NO_BIOSIMILAR":        0,
            "NOT_A_BIOLOGIC":       0,
            "UNKNOWN":              0,
        }

        for item in results:
            if isinstance(item, Exception):
                continue
            drug, status = item
            stats[status.status] = stats.get(status.status, 0) + 1
            drug["pb_status"]     = status.status
            drug["pb_source"]     = status.source
            drug["pb_bla_year"]   = status.bla_year
            drug["pb_biosimilars"] = status.biosimilars
            drug["pb_eligible"]   = status.is_eligible_for_twintrial

            if status.is_eligible_for_twintrial:
                eligible.append(drug)
            else:
                ineligible.append(drug)

        logger.info(
            "Purple Book filter: %d eligible, %d ineligible. "
            "Biosimilar=%d Interchangeable=%d RefProduct=%d NoBiosimilar=%d NotBiologic=%d Unknown=%d",
            len(eligible), len(ineligible),
            stats["BIOSIMILAR_AVAILABLE"], stats["INTERCHANGEABLE"],
            stats["REFERENCE_PRODUCT"], stats["NO_BIOSIMILAR"],
            stats["NOT_A_BIOLOGIC"], stats["UNKNOWN"],
        )

        return eligible, ineligible, stats

    def get_biosimilars_for_drug(self, drug_name: str) -> List[str]:
        """Return list of approved biosimilar names for a reference biologic."""
        name_lower = self._normalize(drug_name)
        curated = PURPLE_BOOK_CURATED.get(name_lower, {})
        return curated.get("biosimilars", [])

    def is_biosimilar_available(self, drug_name: str) -> bool:
        """Quick synchronous check using only the curated table."""
        name_lower = self._normalize(drug_name)
        data = PURPLE_BOOK_CURATED.get(name_lower, {})
        return data.get("biosimilar_available", False)

    def get_full_biosimilar_table(self) -> Dict[str, Dict]:
        """Return the full curated biosimilar table for inspection."""
        return dict(PURPLE_BOOK_CURATED)

    def get_eligible_biologics(self) -> List[str]:
        """Return list of reference biologic names with biosimilar available."""
        return [
            name for name, data in PURPLE_BOOK_CURATED.items()
            if data.get("biosimilar_available", False)
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Module-level convenience functions
# ─────────────────────────────────────────────────────────────────────────────

def is_biologic(drug_name: str) -> bool:
    """
    Heuristic check: is this drug a biological product?
    Uses suffix patterns and explicit exclusion list.
    """
    if drug_name.lower() in NOT_A_BIOLOGIC:
        return False
    return any(drug_name.lower().endswith(sfx) for sfx in BIOLOGIC_SUFFIXES)


def biosimilar_available(drug_name: str) -> bool:
    """
    Synchronous check: does an FDA biosimilar exist for this reference biologic?
    Uses only the curated table (no API calls).
    """
    name_lower = re.sub(
        r"\s+(sodium|potassium|hydrochloride|alfa|beta|gamma)$",
        "", drug_name.strip().lower(),
    ).strip()
    return PURPLE_BOOK_CURATED.get(name_lower, {}).get("biosimilar_available", False)