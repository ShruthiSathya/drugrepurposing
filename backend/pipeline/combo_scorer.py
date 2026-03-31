"""
combo_scorer.py — Drug Combination Scorer
==========================================
Scores drug pairs and triples using mechanism synergy/antagonism rules,
gene coverage bonuses, and redundancy penalties.

Provides:
  CombinationScorer      — scores pairs and triples, used by ComboWorkerPool
  classify_mechanism     — maps a drug mechanism string to a class string
  rank_combinations      — ranks a pre-scored combo list (used in treatment plan)

Architecture
------------
  ComboWorkerPool (combo_worker.py) calls CombinationScorer.score_pair()
  inside subprocess workers. All I/O is done BEFORE workers are spawned;
  workers only use in-memory data and this module's static tables.

  score_pair()  → mechanism synergy/antagonism + gene coverage + redundancy
  score_triple() → three-way extension of score_pair()
  rank_combinations() → post-hoc ranking of pre-scored combos for treatment plan

Mechanism classes
-----------------
  kinase_inhibitor, endothelin_antagonist, pde5_inhibitor, prostacyclin,
  sgc_stimulator, beta_blocker, ace_inhibitor, arb, statin, biguanide,
  immunomodulator, corticosteroid, anti_tnf, anti_il6, jak_inhibitor,
  dmard, anti_cd20, parp_inhibitor, checkpoint_inhibitor, alkylating_agent,
  antimetabolite, taxane, vinca_alkaloid, hdac_inhibitor, proteasome_inhibitor,
  imid, anti_vegf, aromatase_inhibitor, serm, anticonvulsant, dopamine_agonist,
  maob_inhibitor, nmda_antagonist, acetylcholinesterase_inhibitor,
  alpha2_agonist, opioid_antagonist, nsaid, cox2_inhibitor, colchicine,
  anti_uric_acid, diuretic, potassium_channel, microtubule_inhibitor,
  mtor_inhibitor, cftr_modulator, complement_inhibitor, other
"""

import itertools
import logging
import math
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Mechanism classification
# ─────────────────────────────────────────────────────────────────────────────

MECHANISM_KEYWORD_MAP: List[Tuple[str, str]] = [
    # Pulmonary / Vascular
    ("pde5",                    "pde5_inhibitor"),
    ("phosphodiesterase-5",     "pde5_inhibitor"),
    ("phosphodiesterase 5",     "pde5_inhibitor"),
    ("endothelin",              "endothelin_antagonist"),
    ("prostacyclin",            "prostacyclin"),
    ("iloprost",                "prostacyclin"),
    ("treprostinil",            "prostacyclin"),
    ("epoprostenol",            "prostacyclin"),
    ("soluble guanylate",       "sgc_stimulator"),
    ("riociguat",               "sgc_stimulator"),
    # Cardiovascular
    ("beta.adrenergic blocker", "beta_blocker"),
    ("beta blocker",            "beta_blocker"),
    ("beta-blocker",            "beta_blocker"),
    ("propranolol",             "beta_blocker"),
    ("metoprolol",              "beta_blocker"),
    ("ace inhibitor",           "ace_inhibitor"),
    ("angiotensin.converting",  "ace_inhibitor"),
    ("angiotensin receptor",    "arb"),
    ("arb",                     "arb"),
    ("losartan",                "arb"),
    ("statin",                  "statin"),
    ("hmgcr",                   "statin"),
    ("hmg-coa",                 "statin"),
    # Metabolic
    ("biguanide",               "biguanide"),
    ("metformin",               "biguanide"),
    ("ampk",                    "biguanide"),
    ("thiazolidinedione",       "thiazolidinedione"),
    ("ppar.gamma",              "thiazolidinedione"),
    ("sglt2",                   "sglt2_inhibitor"),
    ("glp-1",                   "glp1_agonist"),
    ("glucagon-like",           "glp1_agonist"),
    # Immunology / Inflammation
    ("immunomodulat",           "immunomodulator"),
    ("thalidomide",             "imid"),
    ("lenalidomide",            "imid"),
    ("cereblon",                "imid"),
    ("crbn",                    "imid"),
    ("corticosteroid",          "corticosteroid"),
    ("glucocorticoid",          "corticosteroid"),
    ("dexamethasone",           "corticosteroid"),
    ("prednisone",              "corticosteroid"),
    ("anti-tnf",                "anti_tnf"),
    ("tumor necrosis factor",   "anti_tnf"),
    ("infliximab",              "anti_tnf"),
    ("adalimumab",              "anti_tnf"),
    ("anti-il",                 "anti_il6"),
    ("interleukin",             "anti_il6"),
    ("tocilizumab",             "anti_il6"),
    ("jak inhibitor",           "jak_inhibitor"),
    ("jak-stat",                "jak_inhibitor"),
    ("janus kinase",            "jak_inhibitor"),
    ("dmard",                   "dmard"),
    ("methotrexate",            "dmard"),
    ("hydroxychloroquine",      "dmard"),
    ("anti-cd20",               "anti_cd20"),
    ("cd20",                    "anti_cd20"),
    ("rituximab",               "anti_cd20"),
    # Oncology
    ("parp",                    "parp_inhibitor"),
    ("pd-1",                    "checkpoint_inhibitor"),
    ("pd-l1",                   "checkpoint_inhibitor"),
    ("ctla-4",                  "checkpoint_inhibitor"),
    ("checkpoint",              "checkpoint_inhibitor"),
    ("alkylat",                 "alkylating_agent"),
    ("cyclophosphamide",        "alkylating_agent"),
    ("melphalan",               "alkylating_agent"),
    ("antimetabolite",          "antimetabolite"),
    ("gemcitabine",             "antimetabolite"),
    ("capecitabine",            "antimetabolite"),
    ("taxane",                  "taxane"),
    ("paclitaxel",              "taxane"),
    ("docetaxel",               "taxane"),
    ("vinca",                   "vinca_alkaloid"),
    ("vincristine",             "vinca_alkaloid"),
    ("hdac",                    "hdac_inhibitor"),
    ("histone deacetylase",     "hdac_inhibitor"),
    ("proteasome",              "proteasome_inhibitor"),
    ("bortezomib",              "proteasome_inhibitor"),
    ("anti-vegf",               "anti_vegf"),
    ("vegf",                    "anti_vegf"),
    ("bevacizumab",             "anti_vegf"),
    ("aromatase",               "aromatase_inhibitor"),
    ("letrozole",               "aromatase_inhibitor"),
    ("anastrozole",             "aromatase_inhibitor"),
    ("serm",                    "serm"),
    ("tamoxifen",               "serm"),
    ("raloxifene",              "serm"),
    ("kinase inhibitor",        "kinase_inhibitor"),
    ("tyrosine kinase",         "kinase_inhibitor"),
    ("imatinib",                "kinase_inhibitor"),
    ("mtor",                    "mtor_inhibitor"),
    ("sirolimus",               "mtor_inhibitor"),
    ("everolimus",              "mtor_inhibitor"),
    # Neurology
    ("anticonvulsant",          "anticonvulsant"),
    ("antiepileptic",           "anticonvulsant"),
    ("sodium channel",          "anticonvulsant"),
    ("dopamine agonist",        "dopamine_agonist"),
    ("dopaminergic",            "dopamine_agonist"),
    ("levodopa",                "dopamine_precursor"),
    ("carbidopa",               "dopamine_precursor"),
    ("mao-b",                   "maob_inhibitor"),
    ("monoamine oxidase",       "maob_inhibitor"),
    ("rasagiline",              "maob_inhibitor"),
    ("nmda",                    "nmda_antagonist"),
    ("memantine",               "nmda_antagonist"),
    ("acetylcholinesterase",    "acetylcholinesterase_inhibitor"),
    ("cholinesterase",          "acetylcholinesterase_inhibitor"),
    ("donepezil",               "acetylcholinesterase_inhibitor"),
    ("alpha-2",                 "alpha2_agonist"),
    ("alpha2",                  "alpha2_agonist"),
    ("clonidine",               "alpha2_agonist"),
    # Pain / Analgesia
    ("opioid antagonist",       "opioid_antagonist"),
    ("naltrexone",              "opioid_antagonist"),
    ("nsaid",                   "nsaid"),
    ("cox-2",                   "cox2_inhibitor"),
    ("cyclooxygenase-2",        "cox2_inhibitor"),
    ("celecoxib",               "cox2_inhibitor"),
    ("aspirin",                 "nsaid"),
    ("ibuprofen",               "nsaid"),
    # Rare / other
    ("colchicine",              "colchicine"),
    ("microtubule",             "microtubule_inhibitor"),
    ("tubulin",                 "microtubule_inhibitor"),
    ("xanthine oxidase",        "anti_uric_acid"),
    ("allopurinol",             "anti_uric_acid"),
    ("febuxostat",              "anti_uric_acid"),
    ("diuretic",                "diuretic"),
    ("furosemide",              "diuretic"),
    ("spironolactone",          "diuretic"),
    ("potassium channel",       "potassium_channel"),
    ("minoxidil",               "potassium_channel"),
    ("cftr",                    "cftr_modulator"),
    ("ivacaftor",               "cftr_modulator"),
    ("complement",              "complement_inhibitor"),
    ("eculizumab",              "complement_inhibitor"),
]


def classify_mechanism(mechanism: str) -> str:
    """
    Map a free-text mechanism string to a standardised mechanism class.
    Returns "other" if no match found.
    """
    if not mechanism:
        return "other"
    mech_lower = mechanism.lower()
    for keyword, cls in MECHANISM_KEYWORD_MAP:
        if keyword in mech_lower:
            return cls
    return "other"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Known synergistic and antagonistic pairs (mechanism class level)
# ─────────────────────────────────────────────────────────────────────────────

# Pairs of mechanism classes that are clinically validated as synergistic.
# Order-independent: (A, B) == (B, A).
SYNERGISTIC_PAIRS: Set[frozenset] = {
    # PAH triple therapy
    frozenset({"pde5_inhibitor",         "endothelin_antagonist"}),
    frozenset({"pde5_inhibitor",         "prostacyclin"}),
    frozenset({"endothelin_antagonist",  "prostacyclin"}),
    frozenset({"pde5_inhibitor",         "sgc_stimulator"}),
    # Myeloma
    frozenset({"imid",                   "proteasome_inhibitor"}),
    frozenset({"imid",                   "corticosteroid"}),
    frozenset({"proteasome_inhibitor",   "corticosteroid"}),
    frozenset({"imid",                   "anti_cd20"}),
    # RA
    frozenset({"dmard",                  "anti_tnf"}),
    frozenset({"dmard",                  "anti_il6"}),
    frozenset({"dmard",                  "jak_inhibitor"}),
    # Oncology
    frozenset({"parp_inhibitor",         "alkylating_agent"}),
    frozenset({"checkpoint_inhibitor",   "anti_vegf"}),
    frozenset({"kinase_inhibitor",       "mtor_inhibitor"}),
    frozenset({"aromatase_inhibitor",    "serm"}),
    frozenset({"taxane",                 "alkylating_agent"}),
    frozenset({"anti_vegf",              "alkylating_agent"}),
    frozenset({"hdac_inhibitor",         "imid"}),
    frozenset({"hdac_inhibitor",         "proteasome_inhibitor"}),
    # CV
    frozenset({"beta_blocker",           "ace_inhibitor"}),
    frozenset({"beta_blocker",           "arb"}),
    frozenset({"ace_inhibitor",          "arb"}),
    frozenset({"statin",                 "biguanide"}),
    # Metabolic
    frozenset({"biguanide",              "thiazolidinedione"}),
    frozenset({"biguanide",              "sglt2_inhibitor"}),
    # Neurology
    frozenset({"dopamine_agonist",       "maob_inhibitor"}),
    frozenset({"dopamine_precursor",     "maob_inhibitor"}),
    frozenset({"dopamine_precursor",     "nmda_antagonist"}),
    frozenset({"acetylcholinesterase_inhibitor", "nmda_antagonist"}),
    # Gout
    frozenset({"anti_uric_acid",         "colchicine"}),
    frozenset({"nsaid",                  "colchicine"}),
    frozenset({"cox2_inhibitor",         "colchicine"}),
}

# Pairs that are clinically antagonistic or dangerous.
ANTAGONISTIC_PAIRS: Set[frozenset] = {
    # Hypotension risk
    frozenset({"pde5_inhibitor",     "nsaid"}),       # renal / CV risk
    frozenset({"beta_blocker",       "dopamine_agonist"}),  # PD — worsens motor
    frozenset({"beta_blocker",       "dopamine_precursor"}),
    # Myelosuppression stacking
    frozenset({"alkylating_agent",   "antimetabolite"}),  # additive myelosuppression
    # RA
    frozenset({"anti_tnf",           "jak_inhibitor"}),   # immunosuppression risk
    frozenset({"anti_tnf",           "anti_il6"}),        # dual biologics
    # Renal
    frozenset({"nsaid",              "ace_inhibitor"}),   # AKI triad
    frozenset({"nsaid",              "diuretic"}),
    frozenset({"nsaid",              "arb"}),
}

# Mechanism classes that target the same pathway — penalise for redundancy
REDUNDANT_CLASS_GROUPS: List[Set[str]] = [
    {"ace_inhibitor", "arb"},                                  # both RAAS
    {"pde5_inhibitor", "sgc_stimulator"},                      # both cGMP
    {"anti_tnf", "anti_il6", "jak_inhibitor"},                 # all immunosuppressive
    {"aromatase_inhibitor", "serm"},                           # both ER-related
    {"proteasome_inhibitor", "imid"},                          # both myeloma (keep together for synergy check)
    {"alkylating_agent", "antimetabolite"},                    # same DNA-damage pathway
    {"acetylcholinesterase_inhibitor", "nmda_antagonist"},     # both AD (synergistic exception)
    {"dopamine_agonist", "dopamine_precursor", "maob_inhibitor"},  # all dopaminergic
    {"hdac_inhibitor", "dnmt_inhibitor"},                      # both epigenetic
    {"statin", "statin"},                                      # same class (always redundant)
]


# ─────────────────────────────────────────────────────────────────────────────
# 3. CombinationScorer
# ─────────────────────────────────────────────────────────────────────────────

class CombinationScorer:
    """
    Scores drug pairs and triples for combination potential.

    Scoring formula (pair)
    ----------------------
        base_score = mean(score_a, score_b)
        + synergy_bonus     (+0.20 if known synergistic pair)
        + coverage_bonus    (+0–0.15 based on unique gene coverage)
        - antagonism_penalty (−0.40 if known antagonistic pair)
        - redundancy_penalty (−0.05–0.10 for same-class drugs)
        combo_score = clip(result, 0, 1)

    Parameters
    ----------
    disease_name : str
        Used for disease-specific scoring context (future extension).
    synergy_bonus : float
        Score boost for known synergistic mechanism pairs.
    antagonism_penalty : float
        Score reduction for known antagonistic pairs.
    coverage_bonus_max : float
        Maximum bonus for complementary gene coverage.
    redundancy_penalty : float
        Penalty per degree of mechanism redundancy.
    """

    def __init__(
        self,
        disease_name:         str   = "",
        synergy_bonus:        float = 0.20,
        antagonism_penalty:   float = 0.40,
        coverage_bonus_max:   float = 0.15,
        redundancy_penalty:   float = 0.08,
    ):
        self.disease_name       = disease_name.lower().strip()
        self.synergy_bonus      = synergy_bonus
        self.antagonism_penalty = antagonism_penalty
        self.coverage_bonus_max = coverage_bonus_max
        self.redundancy_penalty = redundancy_penalty

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _gene_coverage_bonus(
        self,
        targets_a: Set[str],
        targets_b: Set[str],
        disease_genes: List[str],
    ) -> float:
        """
        Bonus for covering more unique disease genes together than either alone.
        """
        if not disease_genes:
            return 0.0
        disease_set    = set(g.upper() for g in disease_genes)
        combined       = (targets_a | targets_b) & disease_set
        max_individual = max(
            len(targets_a & disease_set),
            len(targets_b & disease_set),
        )
        extra = len(combined) - max_individual
        if extra <= 0 or len(disease_set) == 0:
            return 0.0
        return min(extra / len(disease_set) * 2.0, self.coverage_bonus_max)

    def _redundancy_penalty(
        self,
        class_a: str,
        class_b: str,
    ) -> float:
        """
        Penalty when two drugs are in the same mechanism class or class group.
        """
        if class_a == class_b and class_a != "other":
            return self.redundancy_penalty

        for group in REDUNDANT_CLASS_GROUPS:
            if class_a in group and class_b in group:
                # Check it's not a known synergistic pair first
                if frozenset({class_a, class_b}) in SYNERGISTIC_PAIRS:
                    return 0.0
                return self.redundancy_penalty * 0.5

        return 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    def score_pair(
        self,
        drug_a: Dict,
        drug_b: Dict,
        disease_genes: List[str],
    ) -> Dict:
        """
        Score a drug pair for combination potential.

        Parameters
        ----------
        drug_a, drug_b : dict
            Drug candidate dicts. Required keys: drug_name (or name), score.
            Optional: target_genes (or targets), mechanism, pathways.
        disease_genes : list of str

        Returns
        -------
        dict with:
            combo_score, combo_name, is_synergistic, is_antagonistic,
            mechanism_a, mechanism_b, synergy_bonus, antagonism_penalty,
            coverage_bonus, redundancy_penalty, shared_genes,
            combined_gene_coverage, score_breakdown
        """
        name_a    = drug_a.get("drug_name", drug_a.get("name", "DrugA"))
        name_b    = drug_b.get("drug_name", drug_b.get("name", "DrugB"))
        score_a   = float(drug_a.get("score", 0.0))
        score_b   = float(drug_b.get("score", 0.0))
        mech_a    = drug_a.get("mechanism", "")
        mech_b    = drug_b.get("mechanism", "")
        class_a   = classify_mechanism(mech_a)
        class_b   = classify_mechanism(mech_b)

        # Gene sets
        targets_a = set(
            t.upper() for t in (drug_a.get("target_genes") or drug_a.get("targets") or [])
        )
        targets_b = set(
            t.upper() for t in (drug_b.get("target_genes") or drug_b.get("targets") or [])
        )
        disease_set = set(g.upper() for g in disease_genes)

        # Shared genes
        shared_genes = list((targets_a | targets_b) & disease_set)

        # Synergy / antagonism
        pair_key        = frozenset({class_a, class_b})
        is_synergistic  = pair_key in SYNERGISTIC_PAIRS
        is_antagonistic = pair_key in ANTAGONISTIC_PAIRS

        # Score components
        base_score      = (score_a + score_b) / 2.0
        syn_bonus       = self.synergy_bonus if is_synergistic else 0.0
        ant_penalty     = self.antagonism_penalty if is_antagonistic else 0.0
        cov_bonus       = self._gene_coverage_bonus(targets_a, targets_b, disease_genes)
        red_penalty     = self._redundancy_penalty(class_a, class_b)

        combo_score = max(0.0, min(1.0,
            base_score + syn_bonus + cov_bonus - ant_penalty - red_penalty
        ))

        combined_coverage = len((targets_a | targets_b) & disease_set)

        return {
            "combo_name":             f"{name_a} + {name_b}",
            "drug_a":                 name_a,
            "drug_b":                 name_b,
            "n_drugs":                2,
            "combo_score":            round(combo_score, 4),
            "base_score":             round(base_score, 4),
            "is_synergistic":         is_synergistic,
            "is_antagonistic":        is_antagonistic,
            "mechanism_a":            class_a,
            "mechanism_b":            class_b,
            "synergy_bonus":          round(syn_bonus, 4),
            "antagonism_penalty":     round(ant_penalty, 4),
            "coverage_bonus":         round(cov_bonus, 4),
            "redundancy_penalty":     round(red_penalty, 4),
            "shared_genes":           shared_genes[:10],
            "combined_gene_coverage": combined_coverage,
            "wet_lab_targets":        shared_genes[:5],
            "score_breakdown": {
                "base_score":          round(base_score, 4),
                "synergy_bonus":       round(syn_bonus, 4),
                "antagonism_penalty":  round(ant_penalty, 4),
                "coverage_bonus":      round(cov_bonus, 4),
                "redundancy_penalty":  round(red_penalty, 4),
            },
        }

    def score_triple(
        self,
        drug_a: Dict,
        drug_b: Dict,
        drug_c: Dict,
        disease_genes: List[str],
    ) -> Dict:
        """
        Score a 3-drug combination.

        Approach: score all three pairs, take the weighted geometric mean,
        apply an additional coverage bonus for the third drug, and penalise
        if any pair is antagonistic.
        """
        name_a = drug_a.get("drug_name", drug_a.get("name", "DrugA"))
        name_b = drug_b.get("drug_name", drug_b.get("name", "DrugB"))
        name_c = drug_c.get("drug_name", drug_c.get("name", "DrugC"))

        pair_ab = self.score_pair(drug_a, drug_b, disease_genes)
        pair_ac = self.score_pair(drug_a, drug_c, disease_genes)
        pair_bc = self.score_pair(drug_b, drug_c, disease_genes)

        any_antagonistic = (
            pair_ab["is_antagonistic"] or
            pair_ac["is_antagonistic"] or
            pair_bc["is_antagonistic"]
        )
        n_synergistic = sum([
            pair_ab["is_synergistic"],
            pair_ac["is_synergistic"],
            pair_bc["is_synergistic"],
        ])

        # Geometric mean of pair scores
        s_ab = pair_ab["combo_score"]
        s_ac = pair_ac["combo_score"]
        s_bc = pair_bc["combo_score"]
        geo_mean = (s_ab * s_ac * s_bc) ** (1 / 3)

        # Triple coverage bonus
        targets_a = set(t.upper() for t in (drug_a.get("target_genes") or drug_a.get("targets") or []))
        targets_b = set(t.upper() for t in (drug_b.get("target_genes") or drug_b.get("targets") or []))
        targets_c = set(t.upper() for t in (drug_c.get("target_genes") or drug_c.get("targets") or []))
        disease_set = set(g.upper() for g in disease_genes)

        combined_all  = (targets_a | targets_b | targets_c) & disease_set
        combined_best = max(
            len(targets_a & disease_set),
            len(targets_b & disease_set),
            len(targets_c & disease_set),
        )
        extra = len(combined_all) - combined_best
        triple_bonus = min(extra / max(len(disease_set), 1) * 1.5, 0.12)

        combo_score = max(0.0, min(1.0, geo_mean + triple_bonus))
        if any_antagonistic:
            combo_score *= 0.3  # heavy penalty but don't discard here (worker will)

        shared_genes = list(combined_all)

        return {
            "combo_name":             f"{name_a} + {name_b} + {name_c}",
            "drug_a":                 name_a,
            "drug_b":                 name_b,
            "drug_c":                 name_c,
            "n_drugs":                3,
            "combo_score":            round(combo_score, 4),
            "is_synergistic":         n_synergistic >= 2,
            "is_antagonistic":        any_antagonistic,
            "n_synergistic_pairs":    n_synergistic,
            "mechanism_a":            pair_ab["mechanism_a"],
            "mechanism_b":            pair_ab["mechanism_b"],
            "mechanism_c":            classify_mechanism(drug_c.get("mechanism", "")),
            "synergy_bonus":          round(n_synergistic * 0.10, 4),
            "antagonism_penalty":     0.0 if not any_antagonistic else round(self.antagonism_penalty, 4),
            "coverage_bonus":         round(triple_bonus, 4),
            "redundancy_penalty":     0.0,
            "shared_genes":           shared_genes[:10],
            "combined_gene_coverage": len(combined_all),
            "wet_lab_targets":        shared_genes[:5],
            "score_breakdown": {
                "geometric_mean_of_pairs": round(geo_mean, 4),
                "triple_coverage_bonus":   round(triple_bonus, 4),
                "n_synergistic_pairs":     n_synergistic,
            },
            "pair_scores": {
                f"{name_a}+{name_b}": s_ab,
                f"{name_a}+{name_c}": s_ac,
                f"{name_b}+{name_c}": s_bc,
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# 4. ComboWorkerPool (re-export from combo_worker — avoids circular import)
# ─────────────────────────────────────────────────────────────────────────────
# combo_worker.py imports CombinationScorer from this module.
# production_pipeline.py imports ComboWorkerPool from combo_worker.
# We re-expose ComboWorkerPool here for convenience so callers only need one import.

try:
    from .combo_worker import ComboWorkerPool  # noqa: F401
except ImportError:
    # If combo_worker isn't available (e.g. test environment) expose a minimal stub
    class ComboWorkerPool:  # type: ignore
        """
        Minimal fallback ComboWorkerPool.
        Used when multiprocessing workers are not available (tests, notebook).
        Runs combination scoring synchronously in the main process.
        """

        def __init__(self, n_workers: Optional[int] = None):
            self.n_workers = n_workers or 1
            logger.warning(
                "ComboWorkerPool: running in single-process fallback mode "
                "(combo_worker.py not importable)."
            )

        def run(
            self,
            candidates: List[Dict],
            disease_genes: List[str],
            disease_name: str,
            max_pairs: int = 5000,
            top_n_singles: int = 30,
            include_triples: bool = False,
        ) -> List[Dict]:
            scorer = CombinationScorer(disease_name=disease_name)
            top = sorted(candidates, key=lambda c: c.get("score", 0), reverse=True)[:top_n_singles]
            results = []
            for drug_a, drug_b in itertools.islice(itertools.combinations(top, 2), max_pairs):
                result = scorer.score_pair(drug_a, drug_b, disease_genes)
                if not result.get("is_antagonistic"):
                    result["final_score"] = result["combo_score"]
                    result["safety_margin"] = 1.0
                    results.append(result)
            if include_triples:
                for drug_a, drug_b, drug_c in itertools.islice(
                    itertools.combinations(top[:15], 3), 500
                ):
                    result = scorer.score_triple(drug_a, drug_b, drug_c, disease_genes)
                    if not result.get("is_antagonistic"):
                        result["final_score"] = result["combo_score"]
                        result["safety_margin"] = 1.0
                        results.append(result)
            results.sort(key=lambda r: r.get("final_score", 0), reverse=True)
            return results

        def write_report(self, proofs, output_path="simulation_proof.json", top_n=20):
            import json
            from pathlib import Path
            out = Path(output_path)
            out.write_text(json.dumps({"top_combinations": proofs[:top_n]}, indent=2))
            return out


# ─────────────────────────────────────────────────────────────────────────────
# 5. rank_combinations — used by TreatmentPlanAssembler
# ─────────────────────────────────────────────────────────────────────────────

def rank_combinations(
    candidates:    List[Dict],
    disease_genes: List[str],
    disease_name:  str = "",
    max_pairs:     int = 5000,
    top_n_singles: int = 30,
    include_triples: bool = True,
    min_combo_score: float = 0.0,
) -> List[Dict]:
    """
    Score and rank all drug combinations from a candidate list.

    Used by the treatment plan assembler when the full ComboWorkerPool is
    not available or when a lighter-weight in-process scorer is preferred.

    Parameters
    ----------
    candidates : list of dict
        Single-drug scored candidates (must have 'score', 'drug_name' or 'name').
    disease_genes : list of str
    disease_name : str
    max_pairs : int
        Maximum number of pairs to evaluate.
    top_n_singles : int
        Only combine the top N singles by score (limits combinatorial explosion).
    include_triples : bool
        Whether to include 3-drug combinations.
    min_combo_score : float
        Minimum combo_score to include in results.

    Returns
    -------
    list of combo dicts sorted by combo_score descending.
    """
    scorer = CombinationScorer(disease_name=disease_name)

    top = sorted(candidates, key=lambda c: c.get("score", 0), reverse=True)
    top = top[:top_n_singles]

    results = []

    # Pairs
    for drug_a, drug_b in itertools.islice(itertools.combinations(top, 2), max_pairs):
        result = scorer.score_pair(drug_a, drug_b, disease_genes)
        if not result["is_antagonistic"] and result["combo_score"] >= min_combo_score:
            results.append(result)

    # Triples
    if include_triples and len(top) >= 3:
        for drug_a, drug_b, drug_c in itertools.islice(
            itertools.combinations(top[:15], 3), 500
        ):
            result = scorer.score_triple(drug_a, drug_b, drug_c, disease_genes)
            if not result["is_antagonistic"] and result["combo_score"] >= min_combo_score:
                results.append(result)

    results.sort(key=lambda r: r["combo_score"], reverse=True)

    n_syn = sum(1 for r in results if r["is_synergistic"])
    logger.info(
        "rank_combinations: %d pairs/triples scored, %d synergistic, "
        "top=%s (%.3f) for disease='%s'",
        len(results),
        n_syn,
        results[0]["combo_name"] if results else "none",
        results[0]["combo_score"] if results else 0.0,
        disease_name,
    )

    return results