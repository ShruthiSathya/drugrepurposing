import asyncio
import json
import logging
import math
import random
import hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

CACHE_DIR        = Path("/tmp/drug_repurposing_cache")
TRIAL_CACHE_FILE = CACHE_DIR / "insilico_trial_cache.json"


# ─────────────────────────────────────────────────────────────────────────────
# Disease-specific simulation parameters
# ─────────────────────────────────────────────────────────────────────────────

DISEASE_PARAMS: Dict[str, Dict] = {

    "pancreatic": {
        "baseline_orr":                  0.12,
        "baseline_pfs6":                 0.24,
        "stroma_barrier":                0.65,
        "mutation_heterogeneity":        0.82,
        "kras_prevalence":               0.93,
        "tp53_prevalence":               0.72,
        "immune_desert_fraction":        0.75,
        "phase2_success_threshold_orr":  0.20,
        "resistance_genes":              {"KRAS","YAP1","MCL1","ABCB1","FGF2","IL6"},
        "description": "PDAC — dense stroma, near-universal KRAS mutation, cold tumors.",
    },
    "glioblastoma": {
        "baseline_orr":                  0.05,
        "baseline_pfs6":                 0.15,
        "stroma_barrier":                0.40,
        "bbb_barrier":                   0.70,
        "mutation_heterogeneity":        0.88,
        "egfr_prevalence":               0.50,
        "immune_desert_fraction":        0.80,
        "phase2_success_threshold_orr":  0.15,
        "resistance_genes":              {"EGFR","PTEN","IDH1","MGMT","PDGFRA"},
        "description": "GBM — blood-brain barrier, high heterogeneity, cold TME.",
    },
    "lung": {
        "baseline_orr":                  0.20,
        "baseline_pfs6":                 0.40,
        "stroma_barrier":                0.25,
        "mutation_heterogeneity":        0.70,
        "kras_prevalence":               0.30,
        "egfr_prevalence":               0.15,
        "pdl1_high_fraction":            0.30,
        "immune_desert_fraction":        0.40,
        "phase2_success_threshold_orr":  0.25,
        "resistance_genes":              {"KRAS","MET","EGFR","ALK","RET"},
        "description": "NSCLC — heterogeneous driver mutations, IO-responsive subsets.",
    },
    "breast": {
        "baseline_orr":                  0.25,
        "baseline_pfs6":                 0.50,
        "stroma_barrier":                0.30,
        "mutation_heterogeneity":        0.60,
        "her2_positive_fraction":        0.20,
        "hr_positive_fraction":          0.70,
        "tnbc_fraction":                 0.15,
        "brca_mutant_fraction":          0.10,
        "immune_desert_fraction":        0.50,
        "phase2_success_threshold_orr":  0.25,
        "resistance_genes":              {"ESR1","ERBB2","PIK3CA","AKT1","CDK4"},
        "description": "Breast cancer — HR+, HER2+, TNBC subtypes with distinct biology.",
    },
    "colorectal": {
        "baseline_orr":                  0.10,
        "baseline_pfs6":                 0.25,
        "stroma_barrier":                0.35,
        "mutation_heterogeneity":        0.65,
        "kras_prevalence":               0.45,
        "braf_prevalence":               0.10,
        "msi_high_fraction":             0.15,
        "immune_desert_fraction":        0.55,
        "phase2_success_threshold_orr":  0.20,
        "resistance_genes":              {"KRAS","BRAF","PIK3CA","SMAD4"},
        "description": "CRC — KRAS/BRAF driven, MSI-H subset is IO-responsive.",
    },
    "ovarian": {
        "baseline_orr":                  0.20,
        "baseline_pfs6":                 0.45,
        "stroma_barrier":                0.30,
        "mutation_heterogeneity":        0.65,
        "brca_mutant_fraction":          0.20,
        "immune_desert_fraction":        0.50,
        "phase2_success_threshold_orr":  0.25,
        "resistance_genes":              {"BRCA1","BRCA2","CCNE1","RB1"},
        "description": "Ovarian cancer — BRCA-driven subset benefits from PARP inhibitors.",
    },
    "melanoma": {
        "baseline_orr":                  0.35,
        "baseline_pfs6":                 0.55,
        "stroma_barrier":                0.20,
        "mutation_heterogeneity":        0.70,
        "braf_prevalence":               0.50,
        "immune_desert_fraction":        0.30,
        "phase2_success_threshold_orr":  0.30,
        "resistance_genes":              {"BRAF","NRAS","MAP2K1","CDKN2A"},
        "description": "Melanoma — BRAF-driven, generally immunogenic TME.",
    },
    "multiple myeloma": {
        "baseline_orr":                  0.20,   # R/R 3L+ benchmark; IMiDs show 25-35% in trials
        "baseline_pfs6":                 0.50,
        "stroma_barrier":                0.20,
        "mutation_heterogeneity":        0.60,
        "crbn_wild_type_fraction":       0.85,
        "immune_desert_fraction":        0.40,
        "phase2_success_threshold_orr":  0.20,   # ≥20% ORR = positive signal in R/R MM
        "resistance_genes":              {"CRBN","TP53","RAS","FGFR3"},
        "description": "Multiple myeloma — CRBN-dependent IMiD sensitivity.",
    },
    "leukemia": {
        "baseline_orr":                  0.40,
        "baseline_pfs6":                 0.70,
        "stroma_barrier":                0.10,
        "mutation_heterogeneity":        0.50,
        "bcr_abl_prevalence":            0.95,
        "immune_desert_fraction":        0.30,
        "phase2_success_threshold_orr":  0.40,
        "resistance_genes":              {"ABL1","BCR","T315I"},
        "description": "CML — BCR-ABL driven, excellent TKI responses.",
    },
    "rheumatoid arthritis": {
        "baseline_orr":                  0.40,
        "baseline_pfs6":                 0.60,
        "stroma_barrier":                0.15,
        "mutation_heterogeneity":        0.30,
        "seropositive_fraction":         0.70,
        "immune_desert_fraction":        0.10,
        "phase2_success_threshold_orr":  0.40,
        "resistance_genes":              {"TNF","IL6","STAT3","JAK1","JAK2"},
        "description": "RA — TNF/IL-6/JAK driven inflammation.",
        "outcome_metric": "ACR20 response rate",
    },
    "lupus": {
        "baseline_orr":                  0.30,
        "baseline_pfs6":                 0.55,
        "stroma_barrier":                0.10,
        "mutation_heterogeneity":        0.40,
        "immune_desert_fraction":        0.05,
        "phase2_success_threshold_orr":  0.30,
        "resistance_genes":              {"IFNA1","BLK","IRF5","STAT4"},
        "description": "SLE — type I interferon-driven.",
        "outcome_metric": "SRI-4 response rate",
    },
    "inflammatory bowel disease": {
        "baseline_orr":                  0.35,
        "baseline_pfs6":                 0.55,
        "stroma_barrier":                0.20,
        "mutation_heterogeneity":        0.40,
        "immune_desert_fraction":        0.10,
        "phase2_success_threshold_orr":  0.30,
        "resistance_genes":              {"TNF","IL12B","IL23A","JAK1"},
        "description": "IBD — mucosal inflammation; anti-TNF/IL-12/23 responsive.",
        "outcome_metric": "Clinical remission rate",
    },
    "multiple sclerosis": {
        "baseline_orr":                  0.35,
        "baseline_pfs6":                 0.60,
        "stroma_barrier":                0.60,
        "mutation_heterogeneity":        0.40,
        "immune_desert_fraction":        0.20,
        "phase2_success_threshold_orr":  0.30,
        "resistance_genes":              {"HLA-DRB1","IL7R","CD58"},
        "description": "MS — BBB limits CNS drug access.",
        "outcome_metric": "Annualised relapse rate reduction",
    },
    "heart failure": {
        "baseline_orr":                  0.25,
        "baseline_pfs6":                 0.50,
        "stroma_barrier":                0.10,
        "mutation_heterogeneity":        0.35,
        "hfref_fraction":                0.50,
        "immune_desert_fraction":        0.30,
        "phase2_success_threshold_orr":  0.25,
        "resistance_genes":              {"NPPA","NPPB","MYH7","TNNT2"},
        "description": "Heart failure — HFrEF/HFpEF distinction matters.",
        "outcome_metric": "LVEF improvement",
    },
    "pulmonary arterial hypertension": {
        "baseline_orr":                  0.40,
        "baseline_pfs6":                 0.65,
        "stroma_barrier":                0.20,
        "mutation_heterogeneity":        0.35,
        "bmpr2_mutation_fraction":       0.25,
        "immune_desert_fraction":        0.40,
        "phase2_success_threshold_orr":  0.35,
        "resistance_genes":              {"BMPR2","KCNK3","EIF2AK4"},
        "description": "PAH — BMPR2 mutations; endothelin/PDE5/sGC pathways.",
        "outcome_metric": "6-minute walk distance improvement",
    },
    "pericarditis": {
        "baseline_orr":                  0.70,
        "baseline_pfs6":                 0.80,
        "stroma_barrier":                0.10,
        "mutation_heterogeneity":        0.20,
        "immune_desert_fraction":        0.10,
        "phase2_success_threshold_orr":  0.60,
        "resistance_genes":              {"IL1B","MEFV","NLRP3"},
        "description": "Pericarditis — NLRP3/IL-1B driven.",
        "outcome_metric": "Symptom resolution rate",
    },
    "parkinson": {
        "baseline_orr":                  0.50,
        "baseline_pfs6":                 0.65,
        "stroma_barrier":                0.55,
        "mutation_heterogeneity":        0.45,
        "lrrk2_prevalence":              0.05,
        "gba_prevalence":                0.10,
        "immune_desert_fraction":        0.50,
        "phase2_success_threshold_orr":  0.40,
        "resistance_genes":              {"SNCA","LRRK2","GBA","PINK1"},
        "description": "PD — BBB limits drug access; alpha-synuclein aggregation.",
        "outcome_metric": "UPDRS motor score improvement",
    },
    "alzheimer": {
        "baseline_orr":                  0.20,
        "baseline_pfs6":                 0.35,
        "stroma_barrier":                0.55,
        "mutation_heterogeneity":        0.50,
        "apoe4_prevalence":              0.40,
        "amyloid_positive_fraction":     0.75,
        "immune_desert_fraction":        0.60,
        "phase2_success_threshold_orr":  0.20,
        "resistance_genes":              {"APP","PSEN1","PSEN2","APOE","TREM2"},
        "description": "AD — amyloid/tau driven.",
        "outcome_metric": "CDR-SB or ADAS-Cog stabilisation",
    },
    "epilepsy": {
        "baseline_orr":                  0.40,
        "baseline_pfs6":                 0.55,
        "stroma_barrier":                0.40,
        "mutation_heterogeneity":        0.50,
        "drug_resistant_fraction":       0.30,
        "immune_desert_fraction":        0.50,
        "phase2_success_threshold_orr":  0.35,
        "resistance_genes":              {"SCN1A","KCNQ2","CDKL5","DEPDC5"},
        "description": "Epilepsy — ~30% drug-resistant; ion channel mutations.",
        "outcome_metric": "Seizure-free rate",
    },
    "amyotrophic lateral sclerosis": {
        "baseline_orr":                  0.10,
        "baseline_pfs6":                 0.30,
        "stroma_barrier":                0.55,
        "mutation_heterogeneity":        0.55,
        "sod1_prevalence":               0.20,
        "tdp43_prevalence":              0.97,
        "immune_desert_fraction":        0.60,
        "phase2_success_threshold_orr":  0.15,
        "resistance_genes":              {"SOD1","TARDBP","FUS","C9orf72"},
        "description": "ALS — TDP-43 pathology universal.",
        "outcome_metric": "ALSFRS-R slope reduction",
    },
    "huntington": {
        "baseline_orr":                  0.20,
        "baseline_pfs6":                 0.40,
        "stroma_barrier":                0.55,
        "mutation_heterogeneity":        0.30,
        "immune_desert_fraction":        0.55,
        "phase2_success_threshold_orr":  0.20,
        "resistance_genes":              {"HTT","ADORA2A"},
        "description": "HD — mHTT aggregation; BBB challenge.",
        "outcome_metric": "cUHDRS or mHTT reduction",
    },
    "cystic fibrosis": {
        "baseline_orr":                  0.50,
        "baseline_pfs6":                 0.70,
        "stroma_barrier":                0.20,
        "mutation_heterogeneity":        0.40,
        "f508del_prevalence":            0.85,
        "immune_desert_fraction":        0.30,
        "phase2_success_threshold_orr":  0.40,
        "resistance_genes":              {"CFTR"},
        "description": "CF — CFTR mutation class determines modulator response.",
        "outcome_metric": "FEV1% predicted improvement",
    },
    "tuberous sclerosis": {
        "baseline_orr":                  0.50,
        "baseline_pfs6":                 0.70,
        "stroma_barrier":                0.30,
        "mutation_heterogeneity":        0.35,
        "tsc1_prevalence":               0.30,
        "tsc2_prevalence":               0.70,
        "immune_desert_fraction":        0.40,
        "phase2_success_threshold_orr":  0.40,
        "resistance_genes":              {"TSC1","TSC2","MTOR"},
        "description": "TSC — mTOR pathway; sirolimus/everolimus responsive.",
        "outcome_metric": "Lesion volume reduction",
    },
    "spinal muscular atrophy": {
        "baseline_orr":                  0.60,
        "baseline_pfs6":                 0.75,
        "stroma_barrier":                0.50,
        "mutation_heterogeneity":        0.20,
        "smn2_copy_number_median":       2.0,
        "immune_desert_fraction":        0.40,
        "phase2_success_threshold_orr":  0.50,
        "resistance_genes":              {"SMN1","SMN2"},
        "description": "SMA — SMN2 copy number predicts response.",
        "outcome_metric": "Motor milestone achievement",
    },
    "type 2 diabetes": {
        "baseline_orr":                  0.50,
        "baseline_pfs6":                 0.65,
        "stroma_barrier":                0.10,
        "mutation_heterogeneity":        0.35,
        "insulin_resistant_fraction":    0.70,
        "immune_desert_fraction":        0.40,
        "phase2_success_threshold_orr":  0.40,
        "resistance_genes":              {"INSR","PRKAA1","PPARG","GLP1R"},
        "description": "T2DM — AMPK/GLP-1/SGLT2 pathways.",
        "outcome_metric": "HbA1c reduction ≥ 0.5%",
    },
    "polycystic ovary syndrome": {
        "baseline_orr":                  0.45,
        "baseline_pfs6":                 0.65,
        "stroma_barrier":                0.10,
        "mutation_heterogeneity":        0.30,
        "immune_desert_fraction":        0.40,
        "phase2_success_threshold_orr":  0.35,
        "resistance_genes":              {"INSR","LHCGR","CYP11A1"},
        "description": "PCOS — insulin resistance and androgen excess.",
        "outcome_metric": "Ovulation rate or testosterone normalisation",
    },
    "hypercholesterolemia": {
        "baseline_orr":                  0.60,
        "baseline_pfs6":                 0.80,
        "stroma_barrier":                0.10,
        "mutation_heterogeneity":        0.25,
        "familial_fraction":             0.10,
        "immune_desert_fraction":        0.40,
        "phase2_success_threshold_orr":  0.50,
        "resistance_genes":              {"LDLR","PCSK9","APOB"},
        "description": "Hypercholesterolemia — HMGCR/PCSK9/NPC1L1 pathways.",
        "outcome_metric": "LDL-C reduction ≥ 20%",
    },
    "gout": {
        "baseline_orr":                  0.65,
        "baseline_pfs6":                 0.80,
        "stroma_barrier":                0.10,
        "mutation_heterogeneity":        0.20,
        "hyperuricemia_fraction":        1.0,
        "immune_desert_fraction":        0.30,
        "phase2_success_threshold_orr":  0.55,
        "resistance_genes":              {"ABCG2","SLC22A12","NLRP3"},
        "description": "Gout — uric acid deposition; NLRP3 inflammasome.",
        "outcome_metric": "Gout flare rate reduction",
    },
    "paroxysmal nocturnal hemoglobinuria": {
        "baseline_orr":                  0.80,
        "baseline_pfs6":                 0.90,
        "stroma_barrier":                0.05,
        "mutation_heterogeneity":        0.20,
        "immune_desert_fraction":        0.20,
        "phase2_success_threshold_orr":  0.70,
        "resistance_genes":              {"C5","C3","CD55","CD59"},
        "description": "PNH — complement pathway; eculizumab highly effective.",
        "outcome_metric": "LDH normalisation or transfusion independence",
    },
    "default": {
        "baseline_orr":                  0.22,
        "baseline_pfs6":                 0.40,
        "stroma_barrier":                0.25,
        "mutation_heterogeneity":        0.50,
        "immune_desert_fraction":        0.45,
        "phase2_success_threshold_orr":  0.25,
        "resistance_genes":              set(),
        "description": "Generic — used when no disease-specific params are available.",
        "outcome_metric": "Primary endpoint response rate",
    },
}

DISEASE_KEYWORD_MAP: List[Tuple[str, str]] = [
    ("pancreatic",                          "pancreatic"),
    ("pdac",                                "pancreatic"),
    ("glioblastoma",                        "glioblastoma"),
    ("gbm",                                 "glioblastoma"),
    ("glioma",                              "glioblastoma"),
    ("non-small cell lung",                 "lung"),
    ("nsclc",                               "lung"),
    ("lung carcinoma",                      "lung"),
    ("lung cancer",                         "lung"),
    ("breast cancer",                       "breast"),
    ("breast carcinoma",                    "breast"),
    ("colorectal",                          "colorectal"),
    ("colon cancer",                        "colorectal"),
    ("rectal cancer",                       "colorectal"),
    ("ovarian",                             "ovarian"),
    ("melanoma",                            "melanoma"),
    ("multiple myeloma",                    "multiple myeloma"),
    ("myeloma",                             "multiple myeloma"),
    ("chronic myelogenous leukemia",        "leukemia"),
    ("cml",                                 "leukemia"),
    ("leukemia",                            "leukemia"),
    ("lymphoma",                            "leukemia"),
    ("rheumatoid arthritis",                "rheumatoid arthritis"),
    ("systemic lupus",                      "lupus"),
    ("lupus",                               "lupus"),
    ("inflammatory bowel",                  "inflammatory bowel disease"),
    ("crohn",                               "inflammatory bowel disease"),
    ("ulcerative colitis",                  "inflammatory bowel disease"),
    ("multiple sclerosis",                  "multiple sclerosis"),
    ("heart failure",                       "heart failure"),
    ("pulmonary arterial hypertension",     "pulmonary arterial hypertension"),
    ("pah",                                 "pulmonary arterial hypertension"),
    ("pulmonary hypertension",              "pulmonary arterial hypertension"),
    ("pericarditis",                        "pericarditis"),
    ("parkinson",                           "parkinson"),
    ("alzheimer",                           "alzheimer"),
    ("epilepsy",                            "epilepsy"),
    ("amyotrophic lateral sclerosis",       "amyotrophic lateral sclerosis"),
    ("als",                                 "amyotrophic lateral sclerosis"),
    ("huntington",                          "huntington"),
    ("cystic fibrosis",                     "cystic fibrosis"),
    ("tuberous sclerosis",                  "tuberous sclerosis"),
    ("spinal muscular atrophy",             "spinal muscular atrophy"),
    ("sma",                                 "spinal muscular atrophy"),
    ("type 2 diabetes",                     "type 2 diabetes"),
    ("polycystic ovary",                    "polycystic ovary syndrome"),
    ("pcos",                                "polycystic ovary syndrome"),
    ("hypercholesterolemia",                "hypercholesterolemia"),
    ("gout",                                "gout"),
    ("paroxysmal nocturnal hemoglobinuria", "paroxysmal nocturnal hemoglobinuria"),
    ("pnh",                                 "paroxysmal nocturnal hemoglobinuria"),
]

from typing import Tuple, List  # noqa: E402


def _resolve_disease_params(disease_name: str) -> Dict:
    name_lower = disease_name.lower()
    for keyword, params_key in DISEASE_KEYWORD_MAP:
        if keyword in name_lower:
            params = DISEASE_PARAMS.get(params_key, DISEASE_PARAMS["default"]).copy()
            logger.info(f"🧬 Disease params resolved: '{disease_name}' → '{params_key}'")
            return params
    logger.warning(f"⚠️  No specific disease params for '{disease_name}' — using default.")
    return DISEASE_PARAMS["default"].copy()


@dataclass
class VirtualPatient:
    patient_id:           int
    ethnicity: str
    age_group: str
    genomic_subgroup: str
    mutation_burden:      float = 0.5
    barrier_sensitivity:  float = 0.5
    stroma_density:       float = 0.3
    tumor_volume_cm3:     float = 10.0
    immune_infiltration:  float = 0.3
    drug_sensitivity:     float = 0.5
    pk_variability:       float = 1.0
    performance_status:   int   = 1

    @property
    def effective_drug_exposure(self) -> float:
        return self.pk_variability * (1.0 - self.stroma_density * 0.4)


@dataclass
class PKPDProfile:
    bioavailability:    float
    half_life_hours:    float
    cmax_relative:      float
    tissue_penetration: float
    target_occupancy:   float
    pd_effect_size:     float

    @classmethod
    def from_chembl_properties(cls, properties: Dict) -> "PKPDProfile":
        mw   = properties.get("full_mwt", 400)
        logp = properties.get("alogp", 2.5)
        tpsa = properties.get("psa", 80)
        ro5  = properties.get("num_ro5_violations", 0)

        if ro5 == 0:
            bioavail = max(0.6 - (tpsa / 200.0), 0.3)
        elif ro5 == 1:
            bioavail = max(0.4 - (tpsa / 250.0), 0.15)
        else:
            bioavail = 0.1

        logp_norm   = max(0, min(abs(logp - 2.0) / 4.0, 1.0))
        half_life   = 8.0 + (1.0 - logp_norm) * 16.0
        tissue_pen  = 1.0 / (1.0 + math.exp(-(logp - 1.5)))
        mw_pen      = max(0, 1.0 - (mw - 300) / 500.0)
        cmax_rel    = bioavail * mw_pen
        target_occ  = min(cmax_rel * 2.0, 0.95)
        pd_effect   = 0.3 + (1.0 - tpsa / 200.0) * 0.5

        return cls(
            bioavailability    = round(bioavail, 3),
            half_life_hours    = round(half_life, 1),
            cmax_relative      = round(cmax_rel, 3),
            tissue_penetration = round(tissue_pen, 3),
            target_occupancy   = round(target_occ, 3),
            pd_effect_size     = round(max(pd_effect, 0.1), 3),
        )


@dataclass
class TumorDynamics:
    initial_volume:     float
    sensitive_fraction: float
    resistant_fraction: float
    cycles:             List[Dict] = field(default_factory=list)

    @property
    def current_volume(self) -> float:
        return self.cycles[-1]["volume"] if self.cycles else self.initial_volume

    @property
    def best_response_pct(self) -> float:
        if not self.cycles:
            return 0.0
        min_vol = min(c["volume"] for c in self.cycles)
        return (self.initial_volume - min_vol) / self.initial_volume * 100.0


@dataclass
class PatientOutcome:
    patient_id:        int
    recist_response:   str
    tumor_reduction:   float
    pfs_weeks:         float
    treatment_stopped: bool
    biomarkers:        Dict = field(default_factory=dict)


class InSilicoTrialSimulator:
    """
    Simulates virtual Phase 2 clinical trials for drug repurposing candidates.
    Works across disease types — not limited to oncology.
    """

    def __init__(
        self,
        disease:     str = "unknown disease",
        n_patients:  int = 200,
        n_cycles:    int = 6,
        random_seed: Optional[int] = 42,
    ):
        self.disease        = disease.lower()
        self.n_patients     = n_patients
        self.n_cycles       = n_cycles
        self.random_seed    = random_seed
        self._disk_cache    = self._load_disk_cache()
        self.disease_params = _resolve_disease_params(disease)

    def _load_disk_cache(self) -> Dict:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if TRIAL_CACHE_FILE.exists():
            try:
                with open(TRIAL_CACHE_FILE) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_disk_cache(self) -> None:
        try:
            with open(TRIAL_CACHE_FILE, "w") as f:
                json.dump(self._disk_cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Trial cache save failed: {e}")

    def _cache_key(self, candidate: Dict) -> str:
        key_data = {
            "drug":    candidate.get("drug_name", ""),
            "chembl":  candidate.get("chembl_id", ""),
            "score":   round(candidate.get("score", 0), 3),
            "disease": self.disease,
            "n":       self.n_patients,
            "sim_ver": "2.3",   # bump to invalidate old cached results
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()[:12]

    def _generate_patient_cohort(self, rng: random.Random) -> List[VirtualPatient]:
        params      = self.disease_params
        het         = params.get("mutation_heterogeneity", 0.5)
        imm         = params.get("immune_desert_fraction", 0.45)
        str_barrier = params.get("stroma_barrier", 0.25)
        bbb_barrier = params.get("bbb_barrier", 0.0)
        patients    = []

        for i in range(self.n_patients):
            effective_barrier = max(str_barrier, bbb_barrier)
            patient = VirtualPatient(
                patient_id          = i,
                mutation_burden     = rng.betavariate(2 + het * 3, 2),
                barrier_sensitivity = rng.betavariate(2, 2 + effective_barrier * 4),
                stroma_density      = rng.betavariate(2 + effective_barrier * 2, 2),
                tumor_volume_cm3    = rng.lognormvariate(2.2, 0.6),
                immune_infiltration = (
                    rng.betavariate(1.5, 5)
                    if rng.random() < imm
                    else rng.betavariate(3, 2)
                ),
                drug_sensitivity    = rng.betavariate(2, 2),          # mean=0.50, symmetric
                pk_variability      = rng.lognormvariate(0, 0.3),
                performance_status  = rng.choices([0, 1, 2], weights=[0.3, 0.5, 0.2])[0],
            )
            patients.append(patient)
        return patients

    def _calculate_network_effect(
        self,
        target_genes:           List[str],
        disease_genes:          List[str],
        polypharmacology_score: float,
        composite_score:        float = 0.5,
    ) -> float:
        """
        FIX v2.1: Blend network topology signal with the candidate's composite
        ML score (which already encodes PPI + gene overlap + pathway proximity).

        The topology signal alone was producing near-zero values (~0.14) when
        drug targets don't directly overlap with the disease gene set, causing
        ORR=0.0% for all candidates. The composite score is a much richer signal
        and should dominate the network effect calculation.

        Weights: topology 40%, composite ML score 60%.
        Floor: 0.15 (all approved drugs have some baseline biological activity).
        """
        if not target_genes or not disease_genes:
            # No gene data — fall back to composite score entirely
            return max(composite_score * 0.5, 0.15)

        disease_gene_set  = set(disease_genes)
        direct_hits       = len(set(target_genes) & disease_gene_set)
        total_disease     = len(disease_gene_set)

        if total_disease == 0:
            return max(composite_score * 0.5, 0.15)

        direct_fraction   = direct_hits / total_disease
        indirect_fraction = polypharmacology_score * 0.3

        # Resistance penalty: if drug doesn't hit known resistance genes
        resistance_genes = self.disease_params.get("resistance_genes", set())
        resistance_penalty = 0.0
        if resistance_genes:
            has_resistance_target = any(g in resistance_genes for g in target_genes)
            if not has_resistance_target:
                resistance_penalty = 0.15

        topology_score = min(
            direct_fraction * 0.5 + indirect_fraction + 0.2, 1.0
        ) * (1.0 - resistance_penalty)

        # Composite score carries PPI + gene overlap + pathway signal already.
        # Use it as a strong prior, especially when topology signal is weak.
        blended = topology_score * 0.4 + composite_score * 0.6

        return max(round(blended, 4), 0.15)

    def _simulate_tumor_dynamics(
        self,
        patient:        VirtualPatient,
        pkpd:           PKPDProfile,
        network_effect: float,
        rng:            random.Random,
    ) -> TumorDynamics:
        """
        Simulate disease burden dynamics over treatment cycles.

        v2.2 CALIBRATION FIX: Replaced the 4-way PKPD multiplication chain
        (effective_exposure * barrier * network * sensitivity) which produced
        kill rates of ~0.05 — far below the ~0.13 threshold needed for any
        tumor reduction. The chain had too many sub-1 dampeners with no
        end-to-end calibration against real ORR benchmarks.

        New formula: kill_rate = (network_effect^1.5) * sensitivity * pk_noise * 0.35
        The ^1.5 exponent creates steeper differentiation between strong/weak matches.
        Scale=0.35 calibrated against real-world myeloma ORR benchmarks:
          network=0.15 (floor / wrong disease) → ORR ~0%
          network=0.40 (weak signal)           → ORR ~8-12%
          network=0.57 (Lenalidomide in MM)    → ORR ~30-35%
          network=0.80+ (strong match)         → ORR ~60-70%
        """
        het            = self.disease_params.get("mutation_heterogeneity", 0.5)
        resistant_init = rng.uniform(0.03, het * 0.20)
        sensitive_init = 1.0 - resistant_init

        dynamics = TumorDynamics(
            initial_volume     = patient.tumor_volume_cm3,
            sensitive_fraction = sensitive_init,
            resistant_fraction = resistant_init,
        )

        # Calibrated kill rate using network_effect^1.5 for steeper differentiation.
        # Scale=0.35 chosen so network=0.57 (Lenalidomide/MM) → ORR≈30-35%,
        # network=0.15 (floor/wrong disease) → ORR≈0%.
        # patient.drug_sensitivity ~ betavariate(2,2) → mean=0.50, range 0.05-0.95
        # pk_noise: lognormal(0, 0.25) → ±25% PK variability across patients
        pk_noise  = rng.lognormvariate(0, 0.25)
        kill_rate = (network_effect ** 1.5) * patient.drug_sensitivity * pk_noise * 0.35

        # Slower growth rate ensures kill_rate 0.13+ yields meaningful reduction
        growth_rate = rng.uniform(0.03, 0.08)

        v_sensitive = dynamics.initial_volume * sensitive_init
        v_resistant = dynamics.initial_volume * resistant_init

        for cycle in range(self.n_cycles):
            k = kill_rate * rng.uniform(0.85, 1.15)
            g = growth_rate

            # Sensitive compartment: net effect of kill minus residual growth
            v_sensitive = max(v_sensitive * (1 - k + g * (1 - k)), 0.001)
            # Resistant compartment: grows unchecked with mild stochasticity
            v_resistant = v_resistant * (1 + g) * rng.uniform(0.95, 1.10)
            # Gradual adaptation: small fraction of sensitive cells become resistant
            adapt        = v_sensitive * k * 0.05 * (cycle / max(self.n_cycles, 1))
            v_sensitive  = max(v_sensitive - adapt, 0.001)
            v_resistant += adapt

            total_volume = v_sensitive + v_resistant
            dynamics.cycles.append({
                "cycle":              cycle + 1,
                "volume":             round(total_volume, 4),
                "sensitive_fraction": round(v_sensitive / total_volume, 3),
                "kill_rate":          round(k, 3),
            })

        return dynamics

    def _classify_outcome(
        self,
        patient:  VirtualPatient,
        dynamics: TumorDynamics,
        rng:      random.Random,
    ) -> PatientOutcome:
        best_reduction = dynamics.best_response_pct
        final_volume   = dynamics.current_volume

        if best_reduction >= 20:
            recist = "CR" if best_reduction >= 70 else "PR"
        elif final_volume <= dynamics.initial_volume * 1.20:
            recist = "SD"
        else:
            recist = "PD"

        nadir     = min(c["volume"] for c in dynamics.cycles) if dynamics.cycles else dynamics.initial_volume
        pfs_weeks = 24.0
        for i, cycle in enumerate(dynamics.cycles):
            if cycle["volume"] > nadir * 1.20 and i > 0:
                pfs_weeks = i * 4.0
                break

        toxicity_stop = rng.random() < 0.07 * patient.performance_status

        biomarkers = {
            "high_mutation_burden": patient.mutation_burden > 0.6,
            "high_stroma":          patient.stroma_density > 0.6,
            "immune_hot":           patient.immune_infiltration > 0.4,
            "high_pk_variability":  patient.pk_variability > 1.3,
        }

        return PatientOutcome(
            patient_id        = patient.patient_id,
            recist_response   = recist,
            tumor_reduction   = round(best_reduction, 2),
            pfs_weeks         = round(pfs_weeks, 1),
            treatment_stopped = toxicity_stop,
            biomarkers        = biomarkers,
        )

    def _analyze_biomarkers(self, outcomes: List[PatientOutcome]) -> Dict:
        responders     = [o for o in outcomes if o.recist_response in ("CR", "PR")]
        non_responders = [o for o in outcomes if o.recist_response == "PD"]

        if len(responders) < 5 or len(non_responders) < 5:
            return {"insufficient_responders": True}

        result: Dict = {}
        for bm in list(responders[0].biomarkers.keys()) if responders else []:
            rr_pos = (
                sum(1 for o in outcomes if o.biomarkers.get(bm) and o.recist_response in ("CR", "PR"))
                / max(sum(1 for o in outcomes if o.biomarkers.get(bm)), 1)
            )
            rr_neg = (
                sum(1 for o in outcomes if not o.biomarkers.get(bm) and o.recist_response in ("CR", "PR"))
                / max(sum(1 for o in outcomes if not o.biomarkers.get(bm)), 1)
            )
            enrichment = rr_pos / max(rr_neg, 0.001)
            result[bm] = {
                "response_rate_positive": round(rr_pos, 3),
                "response_rate_negative": round(rr_neg, 3),
                "enrichment_ratio":       round(enrichment, 2),
                "predictive":             enrichment > 1.5 or enrichment < 0.5,
            }
        return result

    def _compute_trial_summary(
        self,
        outcomes:  List[PatientOutcome],
        candidate: Dict,
        pkpd:      PKPDProfile,
    ) -> Dict:
        n = len(outcomes)
        if n == 0:
            return {"error": "No outcomes"}

        cr  = sum(1 for o in outcomes if o.recist_response == "CR")
        pr  = sum(1 for o in outcomes if o.recist_response == "PR")
        sd  = sum(1 for o in outcomes if o.recist_response == "SD")
        pd  = sum(1 for o in outcomes if o.recist_response == "PD")

        orr       = (cr + pr) / n
        dcr       = (cr + pr + sd) / n
        disc_rate = sum(1 for o in outcomes if o.treatment_stopped) / n

        pfs_times  = [o.pfs_weeks for o in outcomes]
        median_pfs = sorted(pfs_times)[n // 2]
        pfs6_rate  = sum(1 for p in pfs_times if p >= 24) / n

        recist_dist = {
            "CR": round(cr / n, 3),
            "PR": round(pr / n, 3),
            "SD": round(sd / n, 3),
            "PD": round(pd / n, 3),
        }

        p2_threshold   = self.disease_params.get("phase2_success_threshold_orr", 0.25)
        baseline_orr   = self.disease_params.get("baseline_orr", 0.20)
        outcome_metric = self.disease_params.get("outcome_metric", "Response rate")

        z = 1.645
        wilson_lower = (
            orr + z**2 / (2 * n) -
            z * math.sqrt(orr * (1 - orr) / n + z**2 / (4 * n**2))
        ) / (1 + z**2 / n)
        wilson_upper = (
            orr + z**2 / (2 * n) +
            z * math.sqrt(orr * (1 - orr) / n + z**2 / (4 * n**2))
        ) / (1 + z**2 / n)

        if wilson_lower >= p2_threshold:
            p2_success_prob = min(0.95, 0.6 + (orr - p2_threshold) * 2)
        elif orr >= p2_threshold:
            p2_success_prob = 0.4 + (orr - p2_threshold) * 1.5
        else:
            p2_success_prob = max(0.05, (orr / p2_threshold) * 0.35)

        relative_improvement = (orr - baseline_orr) / max(baseline_orr, 0.01)

        if p2_success_prob >= 0.55 and relative_improvement >= 0.50:
            recommendation = "ADVANCE TO WET LAB VALIDATION"
            priority       = "HIGH"
        elif p2_success_prob >= 0.35 or relative_improvement >= 0.25:
            recommendation = "CONSIDER WITH BIOMARKER ENRICHMENT"
            priority       = "MEDIUM"
        else:
            recommendation = "DEPRIORITIZE — INSUFFICIENT SIGNAL"
            priority       = "LOW"

        biomarker_analysis = self._analyze_biomarkers(outcomes)

        return {
            "drug_name":                 candidate.get("drug_name", "Unknown"),
            "chembl_id":                 candidate.get("chembl_id", ""),
            "disease":                   self.disease,
            "disease_params_used":       self.disease_params.get("description", ""),
            "outcome_metric":            outcome_metric,
            "n_patients":                n,
            "orr":                       round(orr, 4),
            "orr_ci_90":                 [round(wilson_lower, 4), round(wilson_upper, 4)],
            "dcr":                       round(dcr, 4),
            "median_pfs_weeks":          round(median_pfs, 1),
            "pfs6_rate":                 round(pfs6_rate, 4),
            "discontinuation_rate":      round(disc_rate, 3),
            "recist_distribution":       recist_dist,
            "baseline_orr_comparison": {
                "baseline_orr":          baseline_orr,
                "simulated_orr":         round(orr, 4),
                "relative_improvement":  round(relative_improvement * 100, 1),
            },
            "phase2_success_probability": round(p2_success_prob, 4),
            "phase2_threshold_orr":       p2_threshold,
            "recommendation":             recommendation,
            "priority":                   priority,
            "pkpd_profile":               asdict(pkpd),
            "biomarker_analysis":         biomarker_analysis,
        }
    def calculate_dosage_sparing(self, kill_rate_combo, kill_rate_singles):
        # If combo kill rate is achieved at 30% of single dose = 70% sparing
        sparing_potential = (kill_rate_combo / max(kill_rate_singles)) - 1.0
        return min(sparing_potential, 0.9) # Cap at 90% sparing

    async def run_virtual_trial(self, candidate: Dict) -> Dict:
        """Run a full virtual Phase 2 trial for one drug candidate."""
        cache_key = self._cache_key(candidate)
        if cache_key in self._disk_cache:
            logger.info(f"   💾 Trial cached: {candidate.get('drug_name')}")
            return self._disk_cache[cache_key]

        drug_name = candidate.get("drug_name", "Unknown")
        logger.info(f"🧪 Virtual trial: {drug_name} for {self.disease}")

        seed = self.random_seed
        rng  = (
            random.Random(seed ^ (hash(drug_name) % (2**31)))
            if seed is not None
            else random.Random()
        )

        properties     = candidate.get("chembl_properties", {})
        pkpd           = PKPDProfile.from_chembl_properties(properties)

        target_genes   = candidate.get("target_genes", [])
        disease_genes  = candidate.get("disease_genes", [])
        poly_score     = candidate.get("polypharmacology_score", 0.3)
        # FIX v2.1: pass composite score so network effect reflects ML signal
        composite_score = candidate.get("score", 0.5)
        network_effect  = self._calculate_network_effect(
            target_genes, disease_genes, poly_score, composite_score
        )

        patients = self._generate_patient_cohort(rng)
        outcomes: List[PatientOutcome] = []
        for patient in patients:
            dynamics = self._simulate_tumor_dynamics(patient, pkpd, network_effect, rng)
            outcome  = self._classify_outcome(patient, dynamics, rng)
            outcomes.append(outcome)

        trial_result = self._compute_trial_summary(outcomes, candidate, pkpd)
        trial_result["network_effect"] = round(network_effect, 4)

        logger.info(
            f"   ✅ {drug_name}: ORR={trial_result['orr']:.1%}, "
            f"P2 prob={trial_result['phase2_success_probability']:.2f} "
            f"→ {trial_result['priority']}"
        )

        self._disk_cache[cache_key] = trial_result
        self._save_disk_cache()
        return trial_result

    async def run_batch(
        self,
        candidates:     List[Dict],
        max_concurrent: int = 5,
    ) -> List[Dict]:
        logger.info(
            f"🧬 Virtual trial batch: {len(candidates)} candidates × "
            f"{self.n_patients} virtual patients for '{self.disease}'"
        )

        semaphore = asyncio.Semaphore(max_concurrent)

        async def run_one(c: Dict) -> Dict:
            async with semaphore:
                return await self.run_virtual_trial(c)

        results = await asyncio.gather(
            *[run_one(c) for c in candidates],
            return_exceptions=True,
        )

        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    f"Trial failed for {candidates[i].get('drug_name', '?')}: {result}"
                )
                valid_results.append({
                    "drug_name": candidates[i].get("drug_name", "Unknown"),
                    "error":     str(result),
                    "phase2_success_probability": 0.0,
                    "priority":  "FAILED",
                })
            else:
                valid_results.append(result)

        valid_results.sort(
            key=lambda r: r.get("phase2_success_probability", 0),
            reverse=True,
        )

        high   = sum(1 for r in valid_results if r.get("priority") == "HIGH")
        medium = sum(1 for r in valid_results if r.get("priority") == "MEDIUM")
        low    = sum(1 for r in valid_results if r.get("priority") == "LOW")
        logger.info(f"   🏁 Batch: {high} HIGH, {medium} MEDIUM, {low} LOW priority")

        return valid_results

    def generate_trial_report(self, trial_results: List[Dict]) -> str:
        lines = [
            "# In-Silico Virtual Trial Results",
            f"**Disease**: {self.disease.title()}  ",
            f"**Virtual Cohort**: {self.n_patients} patients per candidate  ",
            f"**Treatment Cycles**: {self.n_cycles} × 4-week cycles  ",
            f"**Disease Params**: {self.disease_params.get('description', 'generic')}  ",
            f"**Primary Endpoint**: {self.disease_params.get('outcome_metric', 'Response rate')}  ",
            "",
            "---",
            "",
            "## Summary Rankings",
            "",
            "| Rank | Drug | ORR | PFS-6 | P2 Prob | Priority |",
            "|------|------|-----|-------|---------|----------|",
        ]

        for i, r in enumerate(trial_results[:10], 1):
            lines.append(
                f"| {i} | {r.get('drug_name','?')} | "
                f"{r.get('orr',0):.1%} | "
                f"{r.get('pfs6_rate',0):.1%} | "
                f"{r.get('phase2_success_probability',0):.2f} | "
                f"**{r.get('priority','?')}** |"
            )

        lines += ["", "---", "", "## Detailed Results", ""]

        for r in trial_results:
            if r.get("priority") not in ("HIGH", "MEDIUM"):
                continue
            lines += [
                f"### {r.get('drug_name', 'Unknown')}",
                f"**Recommendation**: {r.get('recommendation', '')}",
                f"**ORR**: {r.get('orr',0):.1%} "
                f"(90% CI: {r.get('orr_ci_90',[0,0])[0]:.1%}–{r.get('orr_ci_90',[0,0])[1]:.1%})",
                f"**Disease Control Rate**: {r.get('dcr',0):.1%}",
                f"**Median PFS**: {r.get('median_pfs_weeks',0):.0f} weeks",
                f"**PFS-6 Rate**: {r.get('pfs6_rate',0):.1%}",
                "",
                "**Response Distribution:**",
                "```",
                f"  CR: {r.get('recist_distribution',{}).get('CR',0):.1%}",
                f"  PR: {r.get('recist_distribution',{}).get('PR',0):.1%}",
                f"  SD: {r.get('recist_distribution',{}).get('SD',0):.1%}",
                f"  PD: {r.get('recist_distribution',{}).get('PD',0):.1%}",
                "```",
                "",
                "---",
                "",
            ]

        return "\n".join(lines)