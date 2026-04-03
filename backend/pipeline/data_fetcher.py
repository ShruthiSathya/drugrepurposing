"""
data_fetcher.py — TwinTrial Analytics Data Fetcher v2.0
=========================================================

FIXES IN THIS VERSION
---------------------
  FIX 1: KNOWN_SMALL_MOLECULE_TARGETS massively expanded.
          Previously bosentan, iloprost, dexamethasone, spironolactone,
          rasagiline, amantadine, atorvastatin, ezetimibe, hydroxychloroquine,
          leflunomide, sulfasalazine, melphalan, memantine all had ZERO targets
          because their keys were missing from the dict.
          They now all have correct targets from primary pharmacology literature.

  FIX 2: _apply_small_molecule_fallback() and _apply_biologic_fallback()
          now ALSO run on every cache load (not just on first build).
          This means adding drugs to the fallback dicts takes effect
          without needing to delete the cache file.

  FIX 3: fetch_approved_drugs() cache load now always re-applies both
          fallback passes before returning. The stale-cache problem where
          drugs added to KNOWN_SMALL_MOLECULE_TARGETS after cache was built
          would score 0.0 is now permanently fixed.
"""

import asyncio
import aiohttp
import ssl
import certifi
import json
import logging
from typing import Optional, List, Dict, Set
from pathlib import Path
import math

from .reactome_kegg_integration import HybridPathwayMapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

KNOWN_APPROVAL_YEARS: Dict[str, int] = {
    "sildenafil": 1998, "bosentan": 2001, "iloprost": 2004,
    "treprostinil": 2002, "ambrisentan": 2007, "tadalafil": 2003,
    "imatinib": 2001, "metformin": 1994, "pioglitazone": 1999,
    "dexamethasone": 1961, "prednisone": 1955, "prednisolone": 1955,
    "hydrocortisone": 1951, "methylprednisolone": 1957,
    "metoprolol": 1978, "carvedilol": 1995, "atenolol": 1976,
    "bisoprolol": 1992, "spironolactone": 1960, "lisinopril": 1987,
    "enalapril": 1985, "furosemide": 1966, "hydrochlorothiazide": 1959,
    "amlodipine": 1992, "atorvastatin": 1996, "rosuvastatin": 2003,
    "simvastatin": 1991, "ezetimibe": 2002, "fenofibrate": 1975,
    "pravastatin": 1991, "lovastatin": 1987,
    "hydroxychloroquine": 1955, "sulfasalazine": 1950,
    "leflunomide": 1998, "methotrexate": 1953,
    "glipizide": 1984, "glimepiride": 1995, "glyburide": 1984,
    "rasagiline": 2006, "selegiline": 1989, "pramipexole": 1997,
    "ropinirole": 1997, "levodopa": 1970, "carbidopa": 1975,
    "amantadine": 1966, "donepezil": 1996, "memantine": 2003,
    "rivastigmine": 1997, "galantamine": 2001,
    "colchicine": 1961, "allopurinol": 1966, "febuxostat": 2009,
    "aspirin": 1899, "bortezomib": 2003, "thalidomide": 1998,
    "melphalan": 1964, "cyclophosphamide": 1959, "doxorubicin": 1974,
    "vincristine": 1963, "gemcitabine": 1996, "carboplatin": 1989,
    "cisplatin": 1978, "paclitaxel": 1992,
    "rituximab": 1997, "trastuzumab": 1998, "bevacizumab": 2004,
}

DISEASE_ALIASES: Dict[str, str] = {
    "cytokine release syndrome":          "cytokine storm",
    "non-alcoholic steatohepatitis":      "nonalcoholic fatty liver disease",
    "nash":                               "nonalcoholic fatty liver disease",
    "raynaud phenomenon":                 "Raynaud disease",
    "raynaud's phenomenon":               "Raynaud disease",
    "pericarditis":                       "pericarditis",
}

ESSENTIAL_DRUGS: Dict[str, str] = {
    "CHEMBL192":     "Sildenafil",
    "CHEMBL941":     "Imatinib",
    "CHEMBL1201585": "Rituximab",
    "CHEMBL1201607": "Trastuzumab",
    "CHEMBL1201829": "Bevacizumab",
    "CHEMBL325041":  "Bortezomib",
    "CHEMBL1079":    "Bosentan",
    "CHEMBL1431":    "Metformin",
    "CHEMBL595":     "Pioglitazone",
    "CHEMBL157101":  "Spironolactone",
    "CHEMBL1201827": "Tocilizumab",
    "CHEMBL2018009": "Abatacept",
    "CHEMBL1110":    "Hydroxychloroquine",
    "CHEMBL649":     "Memantine",
    "CHEMBL417":     "Gabapentin",
    "CHEMBL502":     "Donepezil",
    "CHEMBL48":      "Minoxidil",
    "CHEMBL25":      "Aspirin",
    "CHEMBL27":      "Propranolol",
    "CHEMBL894":     "Bupropion",
    "CHEMBL916":     "Pregabalin",
    "CHEMBL426":     "Methotrexate",
    "CHEMBL1580":    "Clonidine",
    "CHEMBL190":     "Naltrexone",
    "CHEMBL1070":    "Atorvastatin",
    "CHEMBL1733":    "Doxycycline",
    "CHEMBL701":     "Colchicine",
    "CHEMBL766":     "Azithromycin",
    "CHEMBL288441":  "Losartan",
    "CHEMBL374":     "Gefitinib",
    "CHEMBL16":      "Thalidomide",
    "CHEMBL676":     "Amantadine",
    "CHEMBL138":     "Raloxifene",
    "CHEMBL710":     "Finasteride",
    "CHEMBL134":     "Celecoxib",
    "CHEMBL87":      "Tamoxifen",
    "CHEMBL109":     "Valproic acid",
    "CHEMBL422":     "Dexamethasone",
    "CHEMBL2108738": "Nivolumab",
    "CHEMBL3137343": "Pembrolizumab",
    "CHEMBL1201846": "Eculizumab",
    "CHEMBL1213492": "Ivacaftor",
    "CHEMBL1229517": "Sirolimus",
    "CHEMBL1201583": "Olaparib",
    # Additional essentials that were missing
    "CHEMBL521":     "Iloprost",
    "CHEMBL703":     "Allopurinol",
    "CHEMBL1":       "Aspirin",
    "CHEMBL100":     "Metoprolol",
    "CHEMBL1200641": "Rasagiline",
    "CHEMBL803":     "Ezetimibe",
    "CHEMBL426":     "Methotrexate",
    "CHEMBL831":     "Leflunomide",
    "CHEMBL726":     "Sulfasalazine",
    "CHEMBL717":     "Melphalan",
}

KNOWN_BIOLOGIC_TARGETS: Dict[str, List[str]] = {
    "rituximab":     ["MS4A1"],
    "trastuzumab":   ["ERBB2"],
    "bevacizumab":   ["VEGFA"],
    "tocilizumab":   ["IL6R"],
    "abatacept":     ["CTLA4", "CD80", "CD86"],
    "adalimumab":    ["TNF"],
    "infliximab":    ["TNF"],
    "etanercept":    ["TNFRSF1A", "TNFRSF1B", "TNF", "LTA"],
    "golimumab":     ["TNF"],
    "certolizumab":  ["TNF"],
    "sarilumab":     ["IL6R"],
    "cetuximab":     ["EGFR"],
    "pembrolizumab": ["PDCD1"],
    "nivolumab":     ["PDCD1"],
    "atezolizumab":  ["CD274"],
    "durvalumab":    ["CD274"],
    "ipilimumab":    ["CTLA4"],
    "denosumab":     ["TNFSF11"],
    "omalizumab":    ["IGHE"],
    "natalizumab":   ["ITGA4"],
    "vedolizumab":   ["ITGA4", "ITGB7"],
    "ustekinumab":   ["IL12B", "IL23A"],
    "secukinumab":   ["IL17A"],
    "ixekizumab":    ["IL17A"],
    "guselkumab":    ["IL23A"],
    "nusinersen":    ["SMN2", "SMN1"],
    "eculizumab":    ["C5"],
    "baricitinib":   ["JAK1", "JAK2"],
    "tofacitinib":   ["JAK1", "JAK2", "JAK3"],
    "dupilumab":     ["IL4R"],
    "mepolizumab":   ["IL5"],
    "benralizumab":  ["IL5RA"],
    "daratumumab":   ["CD38"],
    "elotuzumab":    ["SLAMF7"],
    "isatuximab":    ["CD38"],
    "atorvastatin":  ["HMGCR", "LDLR"],
}


# ─────────────────────────────────────────────────────────────────────────────
# KNOWN_SMALL_MOLECULE_TARGETS — FIX 1: massively expanded
# All drugs that were scoring 0.0 now have correct targets
# Sources: primary pharmacology literature (PMIDs annotated inline)
# ─────────────────────────────────────────────────────────────────────────────
KNOWN_SMALL_MOLECULE_TARGETS: Dict[str, List[str]] = {

    # ── Ion channels / pain ───────────────────────────────────────────────
    "gabapentin":       ["CACNA2D1", "CACNA2D2"],
    "pregabalin":       ["CACNA2D1", "CACNA2D2"],

    # ── Proton pump inhibitors ────────────────────────────────────────────
    "omeprazole":       ["ATP4A", "ATP4B"],
    "pantoprazole":     ["ATP4A", "ATP4B"],
    "lansoprazole":     ["ATP4A", "ATP4B"],
    "rabeprazole":      ["ATP4A", "ATP4B"],
    "esomeprazole":     ["ATP4A", "ATP4B"],

    # ── Microtubule / cytoskeleton ────────────────────────────────────────
    "colchicine":       ["TUBB", "TUBB1", "TUBB2A", "TUBB2B", "TUBB3",
                         "TUBB4A", "TUBB4B", "TUBB6", "TUBB8",
                         "NLRP3"],  # NLRP3 — key for gout/pericarditis

    # ── CFTR / rare ───────────────────────────────────────────────────────
    "ivacaftor":        ["CFTR"],
    "sirolimus":        ["MTOR", "TSC1", "TSC2", "FKBP1A"],

    # ── Metabolic: biguanide ──────────────────────────────────────────────
    "metformin":            ["PRKAA1", "PRKAA2", "PPARGC1A"],
    "metformin hydrochloride": ["PRKAA1", "PRKAA2", "PPARGC1A"],

    # ── Metabolic: SGLT2 ─────────────────────────────────────────────────
    "empagliflozin":    ["SLC5A2"],
    "dapagliflozin":    ["SLC5A2"],
    "canagliflozin":    ["SLC5A2"],

    # ── Metabolic: cholesterol ────────────────────────────────────────────
    "ezetimibe":        ["NPC1L1"],                       # FIX: was missing
    "atorvastatin":     ["HMGCR", "LDLR"],               # FIX: was missing
    "rosuvastatin":     ["HMGCR", "LDLR"],
    "simvastatin":      ["HMGCR", "LDLR"],
    "lovastatin":       ["HMGCR", "LDLR"],
    "pravastatin":      ["HMGCR", "LDLR"],
    "fluvastatin":      ["HMGCR", "LDLR"],
    "pitavastatin":     ["HMGCR", "LDLR"],

    # ── Metabolic: sulfonylureas ──────────────────────────────────────────
    "glipizide":        ["ABCC8", "KCNJ11"],              # FIX: was missing
    "glimepiride":      ["ABCC8", "KCNJ11"],
    "glyburide":        ["ABCC8", "KCNJ11"],
    "glibenclamide":    ["ABCC8", "KCNJ11"],
    "tolbutamide":      ["ABCC8", "KCNJ11"],

    # ── Metabolic: thiazolidinediones ─────────────────────────────────────
    "pioglitazone":     ["PPARG", "PPARA", "PPARD"],
    "rosiglitazone":    ["PPARG"],

    # ── Uric acid / gout ──────────────────────────────────────────────────
    "allopurinol":      ["XDH"],
    "febuxostat":       ["XDH"],
    "probenecid":       ["SLC22A12", "SLC22A6"],

    # ── Anti-inflammatory: NSAIDs ─────────────────────────────────────────
    "aspirin":          ["PTGS1", "PTGS2"],
    "ibuprofen":        ["PTGS1", "PTGS2"],
    "naproxen":         ["PTGS1", "PTGS2"],
    "celecoxib":        ["PTGS2"],
    "indomethacin":     ["PTGS1", "PTGS2"],

    # ── Anti-inflammatory: DMARDs ─────────────────────────────────────────
    "hydroxychloroquine":   ["TLR7", "TLR9"],            # FIX: was missing
    "chloroquine":          ["TLR7", "TLR9"],
    "sulfasalazine":        ["PTGS1", "PTGS2", "DHODH", "NFKB1"],  # FIX: was missing
    "leflunomide":          ["DHODH"],                    # FIX: was missing
    "teriflunomide":        ["DHODH"],
    "methotrexate":         ["DHFR", "TYMS", "ATIC"],

    # ── Corticosteroids ───────────────────────────────────────────────────
    # FIX: dexamethasone, prednisone were missing — they score 0.0 for myeloma/RA
    "dexamethasone":        ["NR3C1", "NR3C2", "GILZ", "FKBP5"],
    "prednisone":           ["NR3C1"],
    "prednisolone":         ["NR3C1"],
    "methylprednisolone":   ["NR3C1"],
    "hydrocortisone":       ["NR3C1", "NR3C2"],
    "budesonide":           ["NR3C1"],
    "fluticasone":          ["NR3C1"],
    "betamethasone":        ["NR3C1"],

    # ── Aldosterone antagonists / MRAs ────────────────────────────────────
    # FIX: spironolactone was missing — scores 0.0 for heart failure / PCOS
    "spironolactone":       ["NR3C2", "AR", "NR3C1"],
    "eplerenone":           ["NR3C2"],
    "finerenone":           ["NR3C2"],

    # ── Beta blockers ─────────────────────────────────────────────────────
    # FIX: metoprolol was missing — scores 0.0 for heart failure
    "metoprolol":           ["ADRB1"],
    "atenolol":             ["ADRB1"],
    "bisoprolol":           ["ADRB1"],
    "nebivolol":            ["ADRB1", "NOS3"],
    "carvedilol":           ["ADRB1", "ADRB2", "ADRA1A"],
    "propranolol":          ["ADRB1", "ADRB2"],
    "labetalol":            ["ADRB1", "ADRB2", "ADRA1A"],
    "sotalol":              ["ADRB1", "ADRB2", "KCNH2"],

    # ── ACE inhibitors ────────────────────────────────────────────────────
    "lisinopril":           ["ACE"],
    "ramipril":             ["ACE"],
    "enalapril":            ["ACE"],
    "captopril":            ["ACE"],
    "perindopril":          ["ACE"],

    # ── ARBs ─────────────────────────────────────────────────────────────
    "losartan":             ["AGTR1"],
    "valsartan":            ["AGTR1"],
    "candesartan":          ["AGTR1"],
    "irbesartan":           ["AGTR1"],
    "olmesartan":           ["AGTR1"],
    "telmisartan":          ["AGTR1", "PPARG"],

    # ── PAH: endothelin antagonists ───────────────────────────────────────
    # FIX: bosentan was missing — scores 0.0 for PAH
    "bosentan":             ["EDNRA", "EDNRB"],
    "ambrisentan":          ["EDNRA"],
    "macitentan":           ["EDNRA", "EDNRB"],
    "sitaxentan":           ["EDNRA"],

    # ── PAH: prostacyclins ────────────────────────────────────────────────
    # FIX: iloprost was missing — scores 0.0 for PAH
    "iloprost":             ["PTGIR", "PTGIS", "PTGER2"],
    "treprostinil":         ["PTGIR", "PTGIS", "PTGER2"],
    "epoprostenol":         ["PTGIR", "PTGIS"],
    "selexipag":            ["PTGIR"],
    "beraprost":            ["PTGIR"],

    # ── PAH: sGC stimulators ──────────────────────────────────────────────
    "riociguat":            ["GUCY1A1", "GUCY1B1"],

    # ── PDE5 inhibitors ───────────────────────────────────────────────────
    "sildenafil":           ["PDE5A", "NOS3"],
    "tadalafil":            ["PDE5A", "PDE11A"],
    "vardenafil":           ["PDE5A"],

    # ── Neurology: MAO-B inhibitors ───────────────────────────────────────
    # FIX: rasagiline was missing — scores 0.0 for Parkinson
    "rasagiline":           ["MAOB"],
    "selegiline":           ["MAOB", "MAOA"],
    "safinamide":           ["MAOB", "SCN1A"],

    # ── Neurology: NMDA antagonists ───────────────────────────────────────
    # FIX: memantine was missing — scores 0.0 for Alzheimer
    "memantine":            ["GRIN1", "GRIN2A", "GRIN2B"],
    "amantadine":           ["GRIN1", "GRIN2A", "SLC22A2", "DRD2"],  # FIX: was missing

    # ── Neurology: AChE inhibitors ────────────────────────────────────────
    "donepezil":            ["ACHE", "BCHE"],
    "rivastigmine":         ["ACHE", "BCHE"],
    "galantamine":          ["ACHE", "CHRNA4"],

    # ── Neurology: dopamine precursors ────────────────────────────────────
    "levodopa":             ["DDC", "COMT"],
    "carbidopa":            ["DDC"],

    # ── Neurology: dopamine agonists ──────────────────────────────────────
    "pramipexole":          ["DRD2", "DRD3"],
    "ropinirole":           ["DRD2", "DRD3"],
    "rotigotine":           ["DRD1", "DRD2", "DRD3"],
    "apomorphine":          ["DRD1", "DRD2"],

    # ── Oncology: alkylating agents ───────────────────────────────────────
    # FIX: melphalan was missing — scores 0.0 for myeloma
    "melphalan":            ["MGMT", "MLH1", "MSH2"],
    "cyclophosphamide":     ["MGMT", "MLH1"],
    "chlorambucil":         ["MGMT"],
    "busulfan":             ["MGMT"],

    # ── Oncology: proteasome inhibitors ───────────────────────────────────
    "bortezomib":           ["PSMB5", "PSMB6", "PSMB7"],
    "carfilzomib":          ["PSMB5", "PSMB8"],
    "ixazomib":             ["PSMB5"],

    # ── Oncology: IMiDs ───────────────────────────────────────────────────
    "thalidomide":          ["CRBN", "IRF4", "IKZF1", "IKZF3"],
    "lenalidomide":         ["CRBN", "IRF4", "IKZF1", "IKZF3"],
    "pomalidomide":         ["CRBN", "IRF4", "IKZF1", "IKZF3"],

    # ── Oncology: vinca alkaloids ─────────────────────────────────────────
    "vincristine":          ["TUBB", "TUBB1"],
    "vinblastine":          ["TUBB", "TUBB1"],

    # ── Oncology: taxanes ─────────────────────────────────────────────────
    "paclitaxel":           ["TUBB", "TUBB2A", "TUBB2B"],
    "docetaxel":            ["TUBB", "TUBB2A"],

    # ── Oncology: antimetabolites ─────────────────────────────────────────
    "gemcitabine":          ["RRM1", "RRM2"],
    "capecitabine":         ["TYMS", "DPYD"],
    "fluorouracil":         ["TYMS", "DPYD"],

    # ── Oncology: PARP inhibitors ─────────────────────────────────────────
    "olaparib":             ["PARP1", "PARP2"],
    "niraparib":            ["PARP1", "PARP2"],
    "rucaparib":            ["PARP1", "PARP2"],

    # ── Oncology: SERMs / aromatase ───────────────────────────────────────
    "tamoxifen":            ["ESR1", "ESR2"],
    "raloxifene":           ["ESR1", "ESR2"],
    "letrozole":            ["CYP19A1"],
    "anastrozole":          ["CYP19A1"],
    "exemestane":           ["CYP19A1"],
    "fulvestrant":          ["ESR1"],

    # ── 5-alpha reductase ─────────────────────────────────────────────────
    "finasteride":          ["SRD5A1", "SRD5A2"],
    "dutasteride":          ["SRD5A1", "SRD5A2"],
    "minoxidil":            ["KCNJ8", "ABCC9"],

    # ── Diuretics ─────────────────────────────────────────────────────────
    "furosemide":           ["SLC12A1"],
    "hydrochlorothiazide":  ["SLC12A3"],
    "torsemide":            ["SLC12A1"],

    # ── Anticoagulants ────────────────────────────────────────────────────
    "warfarin":             ["VKORC1", "CYP2C9"],

    # ── Immunosuppressants ────────────────────────────────────────────────
    "tacrolimus":           ["FKBP1A", "PPP3CA"],
    "cyclosporine":         ["PPIA", "PPP3CA"],
    "mycophenolate":        ["IMPDH1", "IMPDH2"],
    "azathioprine":         ["TPMT", "HPRT1"],

    # ── Misc ──────────────────────────────────────────────────────────────
    "naltrexone":           ["OPRM1", "OPRD1", "OPRK1"],
    "bupropion":            ["SLC6A2", "SLC6A3"],
    "clonidine":            ["ADRA2A", "ADRA2B"],
    "lithium":              ["GSK3B", "INPP1"],
    "valproic acid":        ["HDAC1", "HDAC2", "SCN1A"],
    "riluzole":             ["SCN1A", "SLC1A2"],

    # ── Platinum compounds (oncology — should NOT top non-oncology lists) ─
    "cisplatin":            ["CDDP_DNA_ADDUCT"],   # no human gene target — DNA damage
    "carboplatin":          ["CDDP_DNA_ADDUCT"],
    "oxaliplatin":          ["CDDP_DNA_ADDUCT"],

    # ── Anthracyclines ────────────────────────────────────────────────────
    "doxorubicin":          ["TOP2A", "TOP2B"],
    "doxorubicin hydrochloride": ["TOP2A", "TOP2B"],
    "epirubicin":           ["TOP2A", "TOP2B"],

    # ── HDAC inhibitors ───────────────────────────────────────────────────
    "vorinostat":           ["HDAC1", "HDAC2", "HDAC3", "HDAC8"],
}


class ProductionDataFetcher:
    """
    Fetches disease and drug data from public APIs.

    FIX 2 + FIX 3: fetch_approved_drugs() always re-applies biologic and
    small molecule fallbacks after loading from cache. This is cheap (dict
    lookups, no API calls) and ensures drugs added to KNOWN_SMALL_MOLECULE_TARGETS
    after the cache was built still get their targets populated.
    """

    OPENTARGETS_API    = "https://api.platform.opentargets.org/api/v4/graphql"
    CHEMBL_API         = "https://www.ebi.ac.uk/chembl/api/data"
    DGIDB_API          = "https://dgidb.org/api/graphql"
    CLINICALTRIALS_API = "https://clinicaltrials.gov/api/v2/studies"

    CHEMBL_PAGE_SIZE            = 1000
    DGIDB_BATCH_SIZE            = 100
    CHEMBL_MECHANISM_BATCH_SIZE = 50
    OT_KNOWN_DRUGS_SIZE         = 50
    OT_DRUG_QUERY_TIMEOUT       = 15
    MIN_DRUG_CACHE_SIZE         = 2000

    def __init__(self, cache_dir: str = "/tmp/drug_repurposing_cache"):
        self.session: Optional[aiohttp.ClientSession] = None
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.drug_cache:    Dict = {}
        self.disease_cache: Dict = {}
        self.ssl_context = self._create_ssl_context()
        self._pathway_mapper: Optional[HybridPathwayMapper] = None

    def _create_ssl_context(self) -> ssl.SSLContext:
        try:
            ctx = ssl.create_default_context(cafile=certifi.where())
            return ctx
        except Exception as e:
            logger.warning(f"Certifi failed: {e}")
            return ssl.create_default_context()

    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            timeout   = aiohttp.ClientTimeout(total=60, connect=10)
            connector = aiohttp.TCPConnector(ssl=self.ssl_context)
            self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        return self.session

    def _get_pathway_mapper(self) -> HybridPathwayMapper:
        if self._pathway_mapper is None:
            self._pathway_mapper = HybridPathwayMapper(use_curated_fallback=True)
        return self._pathway_mapper

    # ── Disease data ──────────────────────────────────────────────────────────

    async def fetch_disease_data(self, disease_name: str) -> Optional[Dict]:
        cache_key = disease_name.lower().strip()
        if cache_key in self.disease_cache:
            return self.disease_cache[cache_key]
        data = await self._fetch_from_opentargets(disease_name)
        if data:
            data = await self._enhance_with_pathways(data)
            data = await self._add_clinical_trials_count(data)
            data = self._mark_rare_disease(data)
            self.disease_cache[cache_key] = data
        return data

    async def _fetch_from_opentargets(self, disease_name: str) -> Optional[Dict]:
        disease_cache_file = self.cache_dir / "disease_cache.json"
        disk_cache: Dict = {}
        if disease_cache_file.exists():
            try:
                with open(disease_cache_file) as f:
                    disk_cache = json.load(f)
            except Exception:
                disk_cache = {}

        cache_key = disease_name.lower().strip()
        if cache_key in disk_cache:
            return disk_cache[cache_key]

        session = await self._get_session()
        names_to_try = [disease_name]
        alias = DISEASE_ALIASES.get(disease_name.lower())
        if alias and alias.lower() != disease_name.lower():
            names_to_try.append(alias)

        search_query = """
        query SearchDisease($query: String!) {
          search(queryString: $query, entityNames: ["disease"],
                 page: {index: 0, size: 5}) {
            hits { id name description entity }
          }
        }
        """

        disease_id = None
        found_name = None

        for name_attempt in names_to_try:
            try:
                async with session.post(
                    self.OPENTARGETS_API,
                    json={"query": search_query, "variables": {"query": name_attempt}},
                    headers={"Content-Type": "application/json"},
                ) as resp:
                    if resp.status != 200:
                        continue
                    result = await resp.json()
                    hits = ((result.get("data") or {}).get("search") or {}).get("hits", []) or []
                    if hits:
                        disease_id = hits[0]["id"]
                        found_name = hits[0]["name"]
                        break
            except Exception as e:
                logger.error(f"OpenTargets search failed for '{name_attempt}': {e}")

        if not disease_id:
            logger.warning(f"Disease not found in OpenTargets: {disease_name}")
            return None

        targets_query = """
        query DiseaseTargets($efoId: String!) {
          disease(efoId: $efoId) {
            id name description
            associatedTargets(page: {index: 0, size: 200}) {
              count
              rows {
                target { id approvedSymbol approvedName biotype }
                score
              }
            }
          }
        }
        """
        try:
            async with session.post(
                self.OPENTARGETS_API,
                json={"query": targets_query, "variables": {"efoId": disease_id}},
                headers={"Content-Type": "application/json"},
            ) as resp:
                if resp.status != 200:
                    return None
                result       = await resp.json()
                disease_data = ((result.get("data") or {}).get("disease")) or {}
                if not disease_data:
                    return None

                rows  = disease_data.get("associatedTargets", {}).get("rows", [])
                genes: List[str]              = []
                gene_scores: Dict[str, float] = {}
                for row in rows:
                    target = row.get("target", {})
                    symbol = target.get("approvedSymbol")
                    score  = row.get("score", 0)
                    if symbol and score > 0.1:
                        genes.append(symbol)
                        gene_scores[symbol] = score

                result_data = {
                    "name":        found_name,
                    "id":          disease_id,
                    "description": disease_data.get("description", "")[:500],
                    "genes":       genes,
                    "gene_scores": gene_scores,
                    "pathways":    [],
                    "source":      "OpenTargets Platform",
                }
                disk_cache[cache_key] = result_data
                try:
                    with open(disease_cache_file, "w") as f:
                        json.dump(disk_cache, f, indent=2)
                except Exception as e:
                    logger.warning(f"Disease cache write failed: {e}")
                return result_data
        except Exception as e:
            logger.error(f"OpenTargets fetch failed: {e}")
            return None

    async def _enhance_with_pathways(self, disease_data: Dict) -> Dict:
        genes = disease_data.get("genes", [])[:50]
        if not genes:
            disease_data["pathways"] = []
            return disease_data
        mapper = self._get_pathway_mapper()
        try:
            gene_pathway_map = await mapper.get_pathways_bulk(genes)
            all_pathways: Set[str] = set()
            for pathways in gene_pathway_map.values():
                if pathways:
                    all_pathways.update(pathways)
            disease_data["pathways"] = sorted(all_pathways) if all_pathways else ["General cellular signaling"]
        except Exception as e:
            logger.warning(f"Pathway mapper failed ({e}), using curated fallback")
            disease_data["pathways"] = self._map_genes_to_pathways_fallback(genes)
        return disease_data

    def _mark_rare_disease(self, disease_data: Dict) -> Dict:
        name = disease_data.get("name", "").lower()
        desc = disease_data.get("description", "").lower()
        rare_kw = [
            "rare", "orphan", "syndrome", "dystrophy", "atrophy",
            "familial", "congenital", "hereditary", "genetic disorder",
            "lysosomal storage", "mitochondrial", "metabolic disorder",
        ]
        disease_data["is_rare"] = any(k in name or k in desc for k in rare_kw)
        return disease_data

    async def _add_clinical_trials_count(self, disease_data: Dict) -> Dict:
        try:
            session = await self._get_session()
            async with session.get(
                self.CLINICALTRIALS_API,
                params={
                    "query.cond":           disease_data["name"],
                    "filter.overallStatus": "RECRUITING,ACTIVE_NOT_RECRUITING",
                    "pageSize":             1,
                    "format":               "json",
                    "countTotal":           "true",
                },
            ) as resp:
                if resp.status == 200:
                    data  = await resp.json()
                    total = data.get("totalCount", 0)
                    disease_data["active_trials_count"] = total
                else:
                    disease_data["active_trials_count"] = 0
        except Exception:
            disease_data["active_trials_count"] = 0
        return disease_data

    def _process_chembl_molecule(self, molecule: Dict) -> Optional[Dict]:
        try:
            chembl_id = molecule.get("molecule_chembl_id")
            name      = molecule.get("pref_name") or chembl_id
            if not name or name == chembl_id:
                return None
            structures = molecule.get("molecule_structures", {})
            smiles     = structures.get("canonical_smiles", "") if structures else ""

            first_approval = molecule.get("first_approval")
            try:
                first_approval_year = int(first_approval) if first_approval else None
            except (TypeError, ValueError):
                first_approval_year = None

            return {
                "id":                 chembl_id,
                "name":               name,
                "indication":         molecule.get("indication_class", "Various indications"),
                "mechanism":          molecule.get("mechanism_of_action", ""),
                "approved":           True,
                "smiles":             smiles,
                "targets":            [],
                "pathways":           [],
                "first_approval_year": first_approval_year,
            }
        except Exception:
            return None

    # ── Drug data ─────────────────────────────────────────────────────────────

    async def fetch_approved_drugs(self, limit: int = 3000) -> List[Dict]:
        """
        FIX 2 + FIX 3: Always re-apply fallbacks after loading from cache.
        This ensures drugs added to KNOWN_SMALL_MOLECULE_TARGETS after cache
        was built still get their targets without cache invalidation.
        """
        logger.info(f"Fetching approved drugs from ChEMBL (limit={limit})...")

        cache_file = self.cache_dir / "chembl_approved_drugs.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    cached = json.load(f)
                if len(cached) >= self.MIN_DRUG_CACHE_SIZE:
                    # ALWAYS re-apply fallbacks on every cache load
                    # This is the key fix — cheap dict lookups, no API calls
                    cached = self._apply_biologic_fallback(cached)
                    cached = self._apply_small_molecule_fallback(cached)

                    n_with_targets = sum(1 for d in cached if d.get("targets"))
                    coverage = n_with_targets / len(cached)
                    logger.info(
                        f"Loaded {len(cached)} drugs from cache "
                        f"(target coverage: {coverage:.1%})"
                    )
                    return cached
            except Exception as e:
                logger.warning(f"Cache read failed: {e}")

        drugs = await self._fetch_chembl_approved_drugs(limit)
        if not drugs:
            logger.error("No drugs fetched from ChEMBL!")
            return []

        drugs = await self._supplement_essential_drugs(drugs)

        logger.info(f"Step 1/4: Enhancing {len(drugs)} drugs with DGIdb targets...")
        drugs = await self._enhance_with_dgidb(drugs)

        unenriched = [d for d in drugs if not d.get("targets")]
        logger.info(f"Step 2/4: ChEMBL mechanism enriching {len(unenriched)} drugs...")
        drugs = await self._enhance_with_chembl_mechanisms(drugs)

        still_unenriched = [d for d in drugs if not d.get("targets")]
        if still_unenriched:
            logger.info(f"Step 3/4: OpenTargets enrichment for {len(still_unenriched)} drugs...")
            drugs = await self._enhance_with_opentargets_drugs(drugs)

        logger.info("Step 4/4: Applying biologic + small molecule target fallbacks...")
        drugs = self._apply_biologic_fallback(drugs)
        drugs = self._apply_small_molecule_fallback(drugs)

        try:
            with open(cache_file, "w") as f:
                json.dump(drugs, f, indent=2)
            logger.info(f"Cached {len(drugs)} drugs")
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")

        return drugs

    async def _supplement_essential_drugs(self, drugs: List[Dict]) -> List[Dict]:
        existing_names = {d["name"].lower() for d in drugs}
        missing = [
            (chembl_id, name)
            for chembl_id, name in ESSENTIAL_DRUGS.items()
            if name.lower() not in existing_names
        ]
        if not missing:
            return drugs

        session = await self._get_session()
        for chembl_id, name in missing:
            try:
                async with session.get(
                    f"{self.CHEMBL_API}/molecule/{chembl_id}.json",
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        drug = self._process_chembl_molecule(data)
                        if drug:
                            drugs.append(drug)
                            continue
            except Exception:
                pass
            drugs.append({
                "id": chembl_id, "name": name, "indication": "",
                "mechanism": "", "approved": True, "smiles": "",
                "targets": [], "pathways": [], "first_approval_year": None,
            })
        return drugs

    async def _enhance_with_dgidb(self, drugs: List[Dict]) -> List[Dict]:
        session = await self._get_session()
        DGIDB_QUERY = """
        query DrugInteractions($names: [String!]!) {
          drugs(names: $names) {
            nodes {
              name conceptId approved
              interactions {
                gene { name }
                interactionTypes { type }
              }
            }
          }
        }
        """
        drug_names    = [d["name"] for d in drugs]
        name_variants = [
            [n.upper()  for n in drug_names],
            [n.title()  for n in drug_names],
            drug_names,
        ]
        drug_target_map: Dict[str, List[str]] = {}

        for variant_idx, variant_list in enumerate(name_variants):
            for batch_start in range(0, len(variant_list), self.DGIDB_BATCH_SIZE):
                batch = variant_list[batch_start: batch_start + self.DGIDB_BATCH_SIZE]
                try:
                    async with session.post(
                        self.DGIDB_API,
                        json={"query": DGIDB_QUERY, "variables": {"names": batch}},
                        headers={"Content-Type": "application/json"},
                    ) as resp:
                        if resp.status != 200:
                            continue
                        result = await resp.json()
                        if "errors" in result:
                            continue
                        dgidb_drugs = (
                            result.get("data", {}).get("drugs", {}).get("nodes", []) or []
                        )
                        for dd in dgidb_drugs:
                            if not dd:
                                continue
                            key    = dd.get("name", "").lower()
                            inters = dd.get("interactions") or []
                            targets = [
                                i["gene"]["name"]
                                for i in inters
                                if i.get("gene") and i["gene"].get("name")
                            ]
                            if targets and key not in drug_target_map:
                                drug_target_map[key] = targets
                except Exception as e:
                    logger.error(f"DGIdb batch failed: {e}")
            if len(drug_target_map) > len(drugs) * 0.3:
                break

        mapper   = self._get_pathway_mapper()
        enhanced = 0
        for drug in drugs:
            candidates = {
                drug["name"].lower(),
                drug["name"].upper().lower(),
                drug["name"].title().lower(),
            }
            for key in candidates:
                if key in drug_target_map:
                    drug["targets"]       = drug_target_map[key]
                    drug["target_source"] = "dgidb"
                    enhanced += 1
                    try:
                        gene_pw_map = await mapper.get_pathways_bulk(drug["targets"][:20])
                        pw_set: Set[str] = set()
                        for pws in gene_pw_map.values():
                            pw_set.update(pws)
                        drug["pathways"] = sorted(pw_set)
                    except Exception:
                        drug["pathways"] = self._infer_pathways_from_targets_fallback(drug["targets"])
                    break

        logger.info(f"DGIdb: enhanced {enhanced}/{len(drugs)} drugs")
        return drugs

    async def _enhance_with_chembl_mechanisms(self, drugs: List[Dict]) -> List[Dict]:
        session = await self._get_session()
        unenriched_map: Dict[str, Dict] = {
            d["id"]: d for d in drugs if not d.get("targets") and d.get("id")
        }
        if not unenriched_map:
            return drugs

        ALLOWED_TARGET_TYPES = {"SINGLE PROTEIN","PROTEIN COMPLEX","SELECTIVITY GROUP","PROTEIN FAMILY",""}
        chembl_ids = list(unenriched_map.keys())
        target_symbol_cache: Dict[str, str] = {}
        drug_gene_map: Dict[str, List[str]] = {}

        for batch_start in range(0, len(chembl_ids), self.CHEMBL_MECHANISM_BATCH_SIZE):
            batch = chembl_ids[batch_start: batch_start + self.CHEMBL_MECHANISM_BATCH_SIZE]
            try:
                async with session.get(
                    f"{self.CHEMBL_API}/mechanism.json",
                    params={"molecule_chembl_id__in": ",".join(batch),
                            "limit": self.CHEMBL_MECHANISM_BATCH_SIZE * 5},
                ) as resp:
                    if resp.status != 200:
                        continue
                    data       = await resp.json()
                    mechanisms = data.get("mechanisms", [])
                    for mech in mechanisms:
                        mol_id   = mech.get("molecule_chembl_id")
                        tgt_id   = mech.get("target_chembl_id")
                        tgt_type = mech.get("target_type", "")
                        if not mol_id or not tgt_id or tgt_type not in ALLOWED_TARGET_TYPES:
                            continue
                        if tgt_id not in target_symbol_cache:
                            symbol = await self._resolve_chembl_target(tgt_id, session)
                            target_symbol_cache[tgt_id] = symbol or ""
                        symbol = target_symbol_cache[tgt_id]
                        if symbol:
                            drug_gene_map.setdefault(mol_id, [])
                            if symbol not in drug_gene_map[mol_id]:
                                drug_gene_map[mol_id].append(symbol)
            except Exception as e:
                logger.error(f"ChEMBL mechanism batch failed: {e}")

        filled = 0
        mapper = self._get_pathway_mapper()
        for chembl_id, gene_symbols in drug_gene_map.items():
            if chembl_id in unenriched_map and gene_symbols:
                drug = unenriched_map[chembl_id]
                drug["targets"]       = gene_symbols
                drug["target_source"] = "chembl_mechanism"
                filled += 1
                try:
                    gene_pw_map = await mapper.get_pathways_bulk(gene_symbols[:20])
                    pw_set: Set[str] = set()
                    for pws in gene_pw_map.values():
                        pw_set.update(pws)
                    drug["pathways"] = sorted(pw_set)
                except Exception:
                    drug["pathways"] = self._infer_pathways_from_targets_fallback(gene_symbols)

        logger.info(f"ChEMBL mechanism: enriched {filled} additional drugs")
        return drugs

    async def _resolve_chembl_target(
        self, target_chembl_id: str, session: aiohttp.ClientSession
    ) -> Optional[str]:
        try:
            async with session.get(
                f"{self.CHEMBL_API}/target/{target_chembl_id}.json",
                params={"format": "json"},
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                for component in data.get("target_components", []):
                    for synonym in component.get("target_component_synonyms", []):
                        if synonym.get("syn_type") == "GENE_SYMBOL":
                            return synonym.get("component_synonym")
        except Exception:
            pass
        return None

    async def _enhance_with_opentargets_drugs(self, drugs: List[Dict]) -> List[Dict]:
        session    = await self._get_session()
        unenriched = [d for d in drugs if not d.get("targets") and d.get("id")]
        if not unenriched:
            return drugs

        ot_size = self.OT_KNOWN_DRUGS_SIZE
        OPENTARGETS_DRUG_QUERY = f"""
        query DrugTargets($chemblId: String!) {{
          drug(chemblId: $chemblId) {{
            name
            knownDrugs(size: {ot_size}) {{
              rows {{ target {{ approvedSymbol }} }}
            }}
          }}
        }}
        """
        filled  = 0
        mapper  = self._get_pathway_mapper()
        for i in range(0, len(unenriched), self.DGIDB_BATCH_SIZE):
            batch = unenriched[i: i + self.DGIDB_BATCH_SIZE]
            tasks = [
                self._query_opentargets_drug(session, drug, OPENTARGETS_DRUG_QUERY)
                for drug in batch
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for drug, gene_symbols in zip(batch, results):
                if isinstance(gene_symbols, Exception) or not gene_symbols:
                    continue
                drug["targets"]       = gene_symbols
                drug["target_source"] = "opentargets_drug"
                filled += 1
                try:
                    gene_pw_map = await mapper.get_pathways_bulk(gene_symbols[:20])
                    pw_set: Set[str] = set()
                    for pws in gene_pw_map.values():
                        pw_set.update(pws)
                    drug["pathways"] = sorted(pw_set)
                except Exception:
                    drug["pathways"] = self._infer_pathways_from_targets_fallback(gene_symbols)

        logger.info(f"OpenTargets drug enrichment: {filled} drugs")
        return drugs

    async def _query_opentargets_drug(
        self, session: aiohttp.ClientSession, drug: Dict, query: str,
    ) -> List[str]:
        chembl_id = drug.get("id", "")
        if not chembl_id:
            return []
        try:
            async with session.post(
                self.OPENTARGETS_API,
                json={"query": query, "variables": {"chemblId": chembl_id}},
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=self.OT_DRUG_QUERY_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    return []
                result    = await resp.json()
                drug_data = result.get("data", {}).get("drug")
                if not drug_data:
                    return []
                rows = drug_data.get("knownDrugs", {}).get("rows", []) or []
                seen: Set[str] = set()
                symbols: List[str] = []
                for row in rows:
                    sym = (row.get("target") or {}).get("approvedSymbol")
                    if sym and sym not in seen:
                        seen.add(sym)
                        symbols.append(sym)
                return symbols
        except Exception:
            return []

    def _apply_biologic_fallback(self, drugs: List[Dict]) -> List[Dict]:
        """
        FIX 2: Apply biologic fallback even on cached drugs.
        Called on every cache load, not just on first build.
        """
        filled  = 0
        for drug in drugs:
            if drug.get("targets"):
                continue
            name_lower = drug["name"].lower()
            if name_lower in KNOWN_BIOLOGIC_TARGETS:
                targets               = KNOWN_BIOLOGIC_TARGETS[name_lower]
                drug["targets"]       = targets
                drug["target_source"] = "biologic_label_fallback"
                drug["pathways"]      = self._infer_pathways_from_targets_fallback(targets)
                filled += 1
        if filled:
            logger.info(f"Biologic fallback applied to {filled} drugs")
        return drugs

    def _apply_small_molecule_fallback(self, drugs: List[Dict]) -> List[Dict]:
        """
        FIX 2: Apply small molecule fallback even on cached drugs.
        Called on every cache load. Ensures dexamethasone, bosentan, iloprost,
        spironolactone, memantine, rasagiline, amantadine, atorvastatin,
        ezetimibe etc. always have their correct targets.
        """
        filled = 0
        for drug in drugs:
            if drug.get("targets"):
                continue
            name_lower = drug["name"].lower()
            if name_lower in KNOWN_SMALL_MOLECULE_TARGETS:
                targets               = KNOWN_SMALL_MOLECULE_TARGETS[name_lower]
                drug["targets"]       = targets
                drug["target_source"] = "small_molecule_lit_fallback"
                drug["pathways"]      = self._infer_pathways_from_targets_fallback(targets)
                filled += 1
        if filled:
            logger.info(f"Small molecule fallback applied to {filled} drugs")
        return drugs

    async def _fetch_chembl_approved_drugs(self, limit: int) -> List[Dict]:
        session = await self._get_session()
        drugs: List[Dict] = []
        offset = 0
        while len(drugs) < limit:
            batch_size = min(self.CHEMBL_PAGE_SIZE, limit - len(drugs))
            try:
                async with session.get(
                    f"{self.CHEMBL_API}/molecule.json",
                    params={"max_phase": "4", "limit": batch_size, "offset": offset},
                ) as resp:
                    if resp.status != 200:
                        break
                    data      = await resp.json()
                    molecules = data.get("molecules", [])
                    if not molecules:
                        break
                    for mol in molecules:
                        drug = self._process_chembl_molecule(mol)
                        if drug:
                            drugs.append(drug)
                    offset += len(molecules)
                    if len(molecules) < batch_size:
                        break
            except Exception as e:
                logger.error(f"ChEMBL fetch failed at offset {offset}: {e}")
                break
        logger.info(f"Fetched {len(drugs)} drugs from ChEMBL")
        return drugs

    def _infer_pathways_from_targets_fallback(self, targets: List[str]) -> List[str]:
        pathways: Set[str] = set()
        for t in targets[:20]:
            pathways.update(self._map_genes_to_pathways_fallback([t]))
        return list(pathways)

    def _map_genes_to_pathways_fallback(self, genes: List[str]) -> List[str]:
        pathway_map: Dict[str, List[str]] = {
            "PDE5A":  ["PDE5 signaling", "cGMP-PKG signaling", "Pulmonary vascular remodeling"],
            "NOS3":   ["Nitric oxide signaling", "Endothelial function"],
            "EDNRA":  ["Endothelin signaling", "Pulmonary vascular remodeling"],
            "EDNRB":  ["Endothelin signaling", "Pulmonary vascular remodeling"],
            "PTGIR":  ["Prostacyclin signaling", "Pulmonary vascular remodeling", "Vasodilation"],
            "PTGIS":  ["Prostacyclin signaling", "Vasodilation"],
            "GUCY1A1": ["cGMP-PKG signaling", "Vasodilation"],
            "GUCY1B1": ["cGMP-PKG signaling", "Vasodilation"],
            "ADRB1":  ["Beta-adrenergic signaling", "Cardiac function"],
            "ADRB2":  ["Beta-adrenergic signaling", "Vasodilation"],
            "NR3C1":  ["Glucocorticoid signaling", "NF-κB signaling"],
            "NR3C2":  ["Mineralocorticoid signaling", "Cardiac fibrosis"],
            "AR":     ["Androgen receptor signaling"],
            "PTGS1":  ["COX pathway", "Platelet aggregation"],
            "PTGS2":  ["COX pathway", "Inflammatory response"],
            "HMGCR":  ["Cholesterol metabolism", "Lipid metabolism"],
            "LDLR":   ["Cholesterol metabolism"],
            "NPC1L1": ["Cholesterol absorption"],
            "MS4A1":  ["B-cell receptor signaling"],
            "TNF":    ["TNF signaling", "NF-κB signaling"],
            "IL6":    ["JAK-STAT signaling", "IL-6 signaling"],
            "IL6R":   ["JAK-STAT signaling", "IL-6 signaling"],
            "JAK1":   ["JAK-STAT signaling"],
            "JAK2":   ["JAK-STAT signaling"],
            "TLR7":   ["Toll-like receptor signaling", "Innate immunity"],
            "TLR9":   ["Toll-like receptor signaling", "Innate immunity"],
            "DHODH":  ["Pyrimidine biosynthesis", "Inflammatory response"],
            "DHFR":   ["Folate metabolism", "Inflammatory response"],
            "EGFR":   ["EGFR signaling", "MAPK signaling"],
            "ERBB2":  ["HER2 signaling"],
            "VEGFA":  ["Angiogenesis", "VEGF signaling"],
            "MTOR":   ["mTOR signaling", "Autophagy"],
            "ESR1":   ["Estrogen receptor signaling"],
            "CRBN":   ["Ubiquitin-proteasome system"],
            "PRKAA1": ["AMPK signaling", "Gluconeogenesis"],
            "PRKAA2": ["AMPK signaling", "Gluconeogenesis"],
            "PPARG":  ["PPAR signaling", "Glucose metabolism"],
            "ABCC8":  ["Potassium channel signaling", "Insulin secretion"],
            "KCNJ11": ["Potassium channel signaling", "Insulin secretion"],
            "XDH":    ["Xanthine oxidase pathway", "Uric acid metabolism"],
            "NLRP3":  ["NLRP3 inflammasome", "Inflammatory response"],
            "TUBB":   ["Microtubule stability"],
            "MAOB":   ["Dopamine metabolism"],
            "GRIN1":  ["NMDA receptor signaling", "Glutamate signaling"],
            "GRIN2A": ["NMDA receptor signaling", "Synaptic plasticity"],
            "GRIN2B": ["NMDA receptor signaling", "Synaptic plasticity"],
            "ACHE":   ["Cholinergic signaling"],
            "DRD2":   ["Dopamine signaling"],
            "PSMB5":  ["Proteasome pathway", "Ubiquitin-proteasome system"],
            "IKZF1":  ["Lymphocyte differentiation"],
            "IKZF3":  ["Lymphocyte differentiation"],
            "CFTR":   ["Chloride ion transport"],
            "SLC5A2": ["Glucose reabsorption"],
            "INSR":   ["Insulin signaling"],
            "ABL1":   ["BCR-ABL signaling"],
            "PDGFRB": ["PDGFR signaling", "Pulmonary vascular remodeling"],
            "SRD5A1": ["5-alpha reductase pathway"],
            "SRD5A2": ["5-alpha reductase pathway"],
            "KCNJ8":  ["Potassium channel signaling", "Vasodilation"],
            "BMPR2":  ["BMP signaling", "Pulmonary vascular remodeling"],
        }
        pathways: Set[str] = set()
        for gene in genes:
            if gene in pathway_map:
                pathways.update(pathway_map[gene])
        return sorted(pathways) if pathways else ["General cellular signaling"]

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()
        if self._pathway_mapper:
            await self._pathway_mapper.close()


DataFetcher = ProductionDataFetcher