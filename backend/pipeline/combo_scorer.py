"""
combo_scorer.py — Combination Therapy Scorer
=============================================
TwinTrial Analytics core IP: scores 2- and 3-drug combinations from the
generic drug pool.

Combo score formula
-------------------
  combo_score = mean(individual_scores)
              + synergy_bonus        (+0.10 if known synergistic mechanism pair)
              + coverage_bonus       (up to +0.15 if combo covers more genes than best single)
              - antagonism_penalty   (-0.20 if mechanisms antagonise each other)
              - overlap_penalty      (-0.08 if both drugs hit same mechanism class)
              - redundancy_penalty   (-0.10 × jaccard if same gene targets)

All values clamped to [0, 1].

Design principles
-----------------
- Synergy table is based on published clinical combination regimens,
  not modelled from scratch — if it works in the clinic, it's synergistic.
- Coverage bonus rewards combinations that expand the disease gene set
  covered vs the best individual drug alone.
- Redundancy penalty prevents recommending two drugs that do exactly
  the same thing (same target set = no combination benefit).
- Antagonism penalty is hard — recommending a dopamine agonist +
  antagonist combo would be clinically dangerous.

Triple scoring
--------------
Triples are scored as: best_pair_score × 0.80 + third_drug_score × 0.20 + 0.05
with a -0.15 penalty if any sub-pair is antagonistic.
This is conservative — triples carry more safety complexity.
"""

import itertools
import logging
import math
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Mechanism classification
# Maps mechanism text → a class key used for synergy/antagonism lookup
# ─────────────────────────────────────────────────────────────────────────────

MECHANISM_KEYWORDS: List[Tuple[str, str]] = [
    # Cardiovascular / pulmonary
    ("pde5",                        "pde5_inhibitor"),
    ("phosphodiesterase 5",         "pde5_inhibitor"),
    ("endothelin receptor",         "endothelin_antagonist"),
    ("endothelin",                  "endothelin_antagonist"),
    ("prostacyclin",                "prostacyclin_analogue"),
    ("prostaglandin i2",            "prostacyclin_analogue"),
    ("epoprostenol",                "prostacyclin_analogue"),
    ("iloprost",                    "prostacyclin_analogue"),
    ("treprostinil",                "prostacyclin_analogue"),
    ("nitrate",                     "nitrate"),
    ("soluable guanylate cyclase",  "sgc_stimulator"),
    ("riociguat",                   "sgc_stimulator"),
    ("beta adrenergic blocker",     "beta_blocker"),
    ("beta blocker",                "beta_blocker"),
    ("beta-blocker",                "beta_blocker"),
    ("ace inhibitor",               "ace_inhibitor"),
    ("angiotensin-converting enzyme","ace_inhibitor"),
    ("angiotensin receptor",        "arb"),
    ("arb",                         "arb"),
    ("diuretic",                    "diuretic"),
    ("aldosterone",                 "aldosterone_antagonist"),
    ("spironolactone",              "aldosterone_antagonist"),
    ("calcium channel blocker",     "calcium_channel_blocker"),
    ("calcium channel antagonist",  "calcium_channel_blocker"),
    ("anticoagulant",               "anticoagulant"),
    ("warfarin",                    "anticoagulant"),
    ("heparin",                     "anticoagulant"),
    ("antiplatelet",                "antiplatelet"),
    ("platelet aggregation",        "antiplatelet"),
    ("aspirin",                     "antiplatelet"),
    ("statin",                      "statin"),
    ("hmg-coa reductase",           "statin"),
    ("hmgcr",                       "statin"),
    # Metabolic
    ("biguanide",                   "biguanide"),
    ("metformin",                   "biguanide"),
    ("thiazolidinedione",           "thiazolidinedione"),
    ("pparg",                       "thiazolidinedione"),
    ("pioglitazone",                "thiazolidinedione"),
    ("sglt2",                       "sglt2_inhibitor"),
    ("glp-1",                       "glp1_agonist"),
    ("glucagon-like peptide",       "glp1_agonist"),
    ("dpp-4",                       "dpp4_inhibitor"),
    ("dipeptidyl peptidase",        "dpp4_inhibitor"),
    ("sulfonylurea",                "sulfonylurea"),
    ("insulin",                     "insulin"),
    ("ampk",                        "ampk_activator"),
    # Rheumatology / immunology
    ("dmard",                       "dmard"),
    ("methotrexate",                "dmard"),
    ("hydroxychloroquine",          "antimalarial_dmard"),
    ("antimalarial",                "antimalarial_dmard"),
    ("sulfasalazine",               "dmard"),
    ("anti-tnf",                    "anti_tnf"),
    ("tnf",                         "anti_tnf"),
    ("anti-il-6",                   "anti_il6"),
    ("il-6",                        "anti_il6"),
    ("il6",                         "anti_il6"),
    ("jak inhibitor",               "jak_inhibitor"),
    ("jak1",                        "jak_inhibitor"),
    ("jak2",                        "jak_inhibitor"),
    ("tofacitinib",                 "jak_inhibitor"),
    ("corticosteroid",              "corticosteroid"),
    ("glucocorticoid",              "corticosteroid"),
    ("prednisone",                  "corticosteroid"),
    ("dexamethasone",               "corticosteroid"),
    ("nsaid",                       "nsaid"),
    ("cox inhibitor",               "nsaid"),
    ("cox-2",                       "nsaid"),
    ("ibuprofen",                   "nsaid"),
    ("colchicine",                  "colchicine"),
    ("microtubule inhibitor",       "colchicine"),
    ("anti-il-1",                   "anti_il1"),
    ("il-1",                        "anti_il1"),
    ("anakinra",                    "anti_il1"),
    ("mycophenolate",               "mycophenolate"),
    ("azathioprine",                "azathioprine"),
    ("cyclophosphamide",            "alkylating_agent"),
    # Oncology
    ("imid",                        "imid"),
    ("thalidomide",                 "imid"),
    ("lenalidomide",                "imid"),
    ("pomalidomide",                "imid"),
    ("proteasome inhibitor",        "proteasome_inhibitor"),
    ("bortezomib",                  "proteasome_inhibitor"),
    ("carfilzomib",                 "proteasome_inhibitor"),
    ("alkylating",                  "alkylating_agent"),
    ("melphalan",                   "alkylating_agent"),
    ("cyclophosphamide",            "alkylating_agent"),
    ("anthracycline",               "anthracycline"),
    ("doxorubicin",                 "anthracycline"),
    ("taxane",                      "taxane"),
    ("paclitaxel",                  "taxane"),
    ("docetaxel",                   "taxane"),
    ("platinum",                    "platinum"),
    ("cisplatin",                   "platinum"),
    ("carboplatin",                 "platinum"),
    ("antimetabolite",              "antimetabolite"),
    ("gemcitabine",                 "antimetabolite"),
    ("5-fluorouracil",              "antimetabolite"),
    ("capecitabine",                "antimetabolite"),
    ("parp inhibitor",              "parp_inhibitor"),
    ("parp",                        "parp_inhibitor"),
    ("aromatase inhibitor",         "aromatase_inhibitor"),
    ("letrozole",                   "aromatase_inhibitor"),
    ("anastrozole",                 "aromatase_inhibitor"),
    ("serm",                        "serm"),
    ("tamoxifen",                   "serm"),
    ("raloxifene",                  "serm"),
    ("androgen receptor",           "antiandrogen"),
    ("5-alpha reductase",           "5ar_inhibitor"),
    ("5ar",                         "5ar_inhibitor"),
    ("finasteride",                 "5ar_inhibitor"),
    ("dutasteride",                 "5ar_inhibitor"),
    ("tyrosine kinase inhibitor",   "tki"),
    ("imatinib",                    "tki"),
    ("dasatinib",                   "tki"),
    ("mtor inhibitor",              "mtor_inhibitor"),
    ("sirolimus",                   "mtor_inhibitor"),
    ("chloroquine",                 "autophagy_inhibitor"),
    ("hydroxychloroquine",          "autophagy_inhibitor"),
    # Neurology
    ("acetylcholinesterase",        "acetylcholinesterase_inhibitor"),
    ("ache inhibitor",              "acetylcholinesterase_inhibitor"),
    ("donepezil",                   "acetylcholinesterase_inhibitor"),
    ("rivastigmine",                "acetylcholinesterase_inhibitor"),
    ("nmda",                        "nmda_antagonist"),
    ("memantine",                   "nmda_antagonist"),
    ("dopamine agonist",            "dopamine_agonist"),
    ("levodopa",                    "dopamine_precursor"),
    ("carbidopa",                   "dopamine_precursor"),
    ("dopamine antagonist",         "dopamine_antagonist"),
    ("antipsychotic",               "dopamine_antagonist"),
    ("maob inhibitor",              "maob_inhibitor"),
    ("rasagiline",                  "maob_inhibitor"),
    ("selegiline",                  "maob_inhibitor"),
    ("anticonvulsant",              "anticonvulsant"),
    ("antiepileptic",               "anticonvulsant"),
    ("valproate",                   "anticonvulsant"),
    ("lamotrigine",                 "anticonvulsant"),
    ("gabapentin",                  "anticonvulsant"),
    ("pregabalin",                  "anticonvulsant"),
    ("riluzole",                    "riluzole"),
    ("lithium",                     "lithium"),
    ("amantadine",                  "amantadine"),
    # Hormonal / PCOS
    ("gonadotropin",                "gonadotropin"),
    ("clomiphene",                  "serm"),
    ("lhcgr",                       "gonadotropin"),
    ("antiandrogen",                "antiandrogen"),
    # Antimicrobial
    ("antibiotic",                  "antibiotic"),
    ("macrolide",                   "macrolide"),
    ("azithromycin",                "macrolide"),
    ("doxycycline",                 "antibiotic"),
    ("tetracycline",                "antibiotic"),
    # Opioid / addiction
    ("opioid antagonist",           "opioid_antagonist"),
    ("naltrexone",                  "opioid_antagonist"),
    ("naloxone",                    "opioid_antagonist"),
    ("mu-opioid",                   "opioid_antagonist"),
    # Potassium channel / hair
    ("potassium channel",           "potassium_channel_opener"),
    ("minoxidil",                   "potassium_channel_opener"),
]


def classify_mechanism(mechanism_text: str) -> str:
    """Map free-text mechanism to a standardised class key."""
    if not mechanism_text:
        return "other"
    m = mechanism_text.lower()
    for keyword, class_key in MECHANISM_KEYWORDS:
        if keyword in m:
            return class_key
    return "other"


# ─────────────────────────────────────────────────────────────────────────────
# Synergy table
# Based on published, guideline-recommended combination regimens.
# Stored as frozensets so order doesn't matter.
# ─────────────────────────────────────────────────────────────────────────────

SYNERGISTIC_PAIRS: Set[frozenset] = {
    # PAH triple therapy (ESC/ERS guidelines)
    frozenset({"pde5_inhibitor", "endothelin_antagonist"}),
    frozenset({"pde5_inhibitor", "prostacyclin_analogue"}),
    frozenset({"endothelin_antagonist", "prostacyclin_analogue"}),
    frozenset({"pde5_inhibitor", "sgc_stimulator"}),
    # Heart failure (HFrEF quadruple therapy)
    frozenset({"ace_inhibitor", "beta_blocker"}),
    frozenset({"arb", "beta_blocker"}),
    frozenset({"ace_inhibitor", "aldosterone_antagonist"}),
    frozenset({"arb", "aldosterone_antagonist"}),
    frozenset({"ace_inhibitor", "diuretic"}),
    # RA triple DMARD (ACR guidelines)
    frozenset({"dmard", "antimalarial_dmard"}),
    frozenset({"dmard", "nsaid"}),
    frozenset({"anti_tnf", "dmard"}),
    frozenset({"anti_il6", "dmard"}),
    frozenset({"jak_inhibitor", "dmard"}),
    # Pericarditis (colchicine + anti-inflammatory)
    frozenset({"colchicine", "nsaid"}),
    frozenset({"colchicine", "anti_il1"}),
    frozenset({"colchicine", "corticosteroid"}),
    # Multiple myeloma combos
    frozenset({"imid", "proteasome_inhibitor"}),
    frozenset({"imid", "corticosteroid"}),
    frozenset({"proteasome_inhibitor", "corticosteroid"}),
    frozenset({"imid", "alkylating_agent"}),
    # Diabetes metabolic combinations
    frozenset({"biguanide", "thiazolidinedione"}),
    frozenset({"biguanide", "sglt2_inhibitor"}),
    frozenset({"biguanide", "dpp4_inhibitor"}),
    frozenset({"biguanide", "glp1_agonist"}),
    frozenset({"sglt2_inhibitor", "dpp4_inhibitor"}),
    # PCOS
    frozenset({"biguanide", "antiandrogen"}),
    frozenset({"biguanide", "serm"}),
    frozenset({"antiandrogen", "serm"}),
    frozenset({"biguanide", "5ar_inhibitor"}),
    # Alzheimer
    frozenset({"acetylcholinesterase_inhibitor", "nmda_antagonist"}),
    # Parkinson
    frozenset({"dopamine_precursor", "maob_inhibitor"}),
    frozenset({"dopamine_precursor", "dopamine_agonist"}),
    frozenset({"dopamine_precursor", "amantadine"}),
    # Epilepsy
    frozenset({"anticonvulsant", "anticonvulsant"}),  # polytherapy is standard
    # Oncology combos
    frozenset({"taxane", "platinum"}),
    frozenset({"antimetabolite", "platinum"}),
    frozenset({"anthracycline", "alkylating_agent"}),
    frozenset({"aromatase_inhibitor", "parp_inhibitor"}),
    frozenset({"tki", "mtor_inhibitor"}),
    frozenset({"taxane", "anthracycline"}),
    frozenset({"antimetabolite", "autophagy_inhibitor"}),
    frozenset({"mtor_inhibitor", "autophagy_inhibitor"}),
    # Gout
    frozenset({"colchicine", "nsaid"}),
    # ALS
    frozenset({"riluzole", "lithium"}),
    # Hypercholesterolemia
    frozenset({"statin", "antiplatelet"}),
    frozenset({"statin", "ace_inhibitor"}),
    # Addiction
    frozenset({"opioid_antagonist", "biguanide"}),
    # Rosacea / anti-microbial combos
    frozenset({"antibiotic", "nsaid"}),
    frozenset({"macrolide", "nsaid"}),
}

# ─────────────────────────────────────────────────────────────────────────────
# Antagonism table — DO NOT combine these
# ─────────────────────────────────────────────────────────────────────────────

ANTAGONISTIC_PAIRS: Set[frozenset] = {
    frozenset({"ace_inhibitor", "arb"}),           # dual RAAS — dangerous
    frozenset({"dopamine_agonist", "dopamine_antagonist"}),  # cancel out
    frozenset({"dopamine_precursor", "dopamine_antagonist"}),
    frozenset({"nsaid", "anticoagulant"}),          # bleeding risk
    frozenset({"corticosteroid", "nsaid"}),         # GI bleed
    frozenset({"beta_blocker", "calcium_channel_blocker"}),  # bradycardia/hypotension
    frozenset({"opioid_antagonist", "opioid_antagonist"}),  # additive reversal
    frozenset({"nitrate", "pde5_inhibitor"}),       # severe hypotension
    frozenset({"sgc_stimulator", "pde5_inhibitor"}),  # hypotension (riociguat + sildenafil CI)
}


class CombinationScorer:
    """
    Scores 2- and 3-drug combinations from the generic drug pool.

    The combo_score is TwinTrial's core differentiator: it captures
    whether two drugs together are better than either alone, not just
    whether both score well individually.

    Usage
    -----
        scorer = CombinationScorer(disease_name="pulmonary arterial hypertension")
        combos = scorer.rank_combinations(
            candidates=top_candidates,
            disease_genes=disease_genes,
        )
        # combos[0] is the best regimen to put in the treatment plan
    """

    def __init__(self, disease_name: str = ""):
        self.disease_name = disease_name.lower()
        logger.info(
            f"CombinationScorer initialised for '{disease_name}'"
        )

    # ── Sub-scorers ───────────────────────────────────────────────────────────

    def _gene_coverage_bonus(
        self,
        targets_a: Set[str],
        targets_b: Set[str],
        disease_genes: List[str],
    ) -> float:
        """
        Bonus if the combo covers more disease genes than the best single drug.
        Max +0.15. Rewards complementary rather than redundant drug pairs.
        """
        disease_set = set(disease_genes)
        if not disease_set:
            return 0.0
        combined = (targets_a | targets_b) & disease_set
        best_solo = max(
            len(targets_a & disease_set),
            len(targets_b & disease_set),
            1,
        )
        gain = (len(combined) - best_solo) / len(disease_set)
        return round(min(gain * 0.4, 0.15), 4)

    def _redundancy_penalty(
        self,
        targets_a: Set[str],
        targets_b: Set[str],
    ) -> float:
        """
        Penalise if the two drugs hit nearly the same gene set.
        Uses Jaccard similarity × -0.10.
        """
        if not targets_a or not targets_b:
            return 0.0
        jaccard = len(targets_a & targets_b) / len(targets_a | targets_b)
        return round(-0.10 * jaccard, 4)

    def _mechanism_overlap_penalty(self, mech_a: str, mech_b: str) -> float:
        """
        Small penalty if both drugs belong to the same mechanism class.
        Exception: anticonvulsant + anticonvulsant is standard polytherapy.
        """
        if mech_a == mech_b and mech_a not in ("other", "anticonvulsant"):
            return -0.08
        return 0.0

    # ── Pair scoring ──────────────────────────────────────────────────────────

    def score_pair(
        self,
        drug_a: Dict,
        drug_b: Dict,
        disease_genes: List[str],
    ) -> Dict:
        """
        Score a 2-drug combination.

        Returns a dict with combo_score and full evidence breakdown.
        """
        score_a = drug_a.get("score", 0.0)
        score_b = drug_b.get("score", 0.0)
        base = (score_a + score_b) / 2.0

        targets_a = set(drug_a.get("target_genes", []))
        targets_b = set(drug_b.get("target_genes", []))

        mech_a = classify_mechanism(drug_a.get("mechanism", ""))
        mech_b = classify_mechanism(drug_b.get("mechanism", ""))
        pair_key = frozenset({mech_a, mech_b})

        synergy_bonus       = 0.10  if pair_key in SYNERGISTIC_PAIRS  else 0.0
        antagonism_penalty  = -0.20 if pair_key in ANTAGONISTIC_PAIRS else 0.0
        overlap_penalty     = self._mechanism_overlap_penalty(mech_a, mech_b)
        redundancy_penalty  = self._redundancy_penalty(targets_a, targets_b)
        coverage_bonus      = self._gene_coverage_bonus(targets_a, targets_b, disease_genes)

        combo_score = max(min(
            base
            + synergy_bonus
            + antagonism_penalty
            + overlap_penalty
            + redundancy_penalty
            + coverage_bonus,
            1.0,
        ), 0.0)

        disease_set = set(disease_genes)
        shared_genes = list((targets_a | targets_b) & disease_set)

        return {
            "n_drugs":              2,
            "drug_a":               drug_a.get("drug_name", drug_a.get("name", "")),
            "drug_b":               drug_b.get("drug_name", drug_b.get("name", "")),
            "drug_c":               None,
            "combo_name":           f"{drug_a.get('drug_name', drug_a.get('name', ''))} + {drug_b.get('drug_name', drug_b.get('name', ''))}",
            "combo_score":          round(combo_score, 4),
            "base_score":           round(base, 4),
            "score_a":              round(score_a, 4),
            "score_b":              round(score_b, 4),
            "synergy_bonus":        round(synergy_bonus, 4),
            "antagonism_penalty":   round(antagonism_penalty, 4),
            "overlap_penalty":      round(overlap_penalty, 4),
            "redundancy_penalty":   round(redundancy_penalty, 4),
            "coverage_bonus":       round(coverage_bonus, 4),
            "mechanism_a":          mech_a,
            "mechanism_b":          mech_b,
            "mechanism_c":          None,
            "is_synergistic":       pair_key in SYNERGISTIC_PAIRS,
            "is_antagonistic":      pair_key in ANTAGONISTIC_PAIRS,
            "combined_gene_coverage": len(shared_genes),
            "shared_genes":         shared_genes[:15],
            "target_genes_a":       list(targets_a)[:10],
            "target_genes_b":       list(targets_b)[:10],
            "target_genes_c":       [],
            "wet_lab_targets":      shared_genes[:5],
        }

    # ── Triple scoring ────────────────────────────────────────────────────────

    def score_triple(
        self,
        drug_a: Dict,
        drug_b: Dict,
        drug_c: Dict,
        disease_genes: List[str],
    ) -> Dict:
        """
        Score a 3-drug combination.
        Conservative: based on best pair + marginal third drug contribution.
        """
        pair_ab = self.score_pair(drug_a, drug_b, disease_genes)
        pair_ac = self.score_pair(drug_a, drug_c, disease_genes)
        pair_bc = self.score_pair(drug_b, drug_c, disease_genes)

        best_pair_score = max(
            pair_ab["combo_score"],
            pair_ac["combo_score"],
            pair_bc["combo_score"],
        )
        # Third drug contributes its individual score × 0.20
        score_c = drug_c.get("score", 0.0)
        triple_base = best_pair_score * 0.80 + score_c * 0.20 + 0.05

        # Hard penalty if any sub-pair is antagonistic
        any_antagonistic = any(
            p["is_antagonistic"] for p in [pair_ab, pair_ac, pair_bc]
        )
        if any_antagonistic:
            triple_base -= 0.15

        triple_score = max(min(triple_base, 1.0), 0.0)

        targets_a = set(drug_a.get("target_genes", []))
        targets_b = set(drug_b.get("target_genes", []))
        targets_c = set(drug_c.get("target_genes", []))
        disease_set = set(disease_genes)
        shared_genes = list((targets_a | targets_b | targets_c) & disease_set)

        mech_c = classify_mechanism(drug_c.get("mechanism", ""))

        name_a = drug_a.get("drug_name", drug_a.get("name", ""))
        name_b = drug_b.get("drug_name", drug_b.get("name", ""))
        name_c = drug_c.get("drug_name", drug_c.get("name", ""))

        return {
            "n_drugs":              3,
            "drug_a":               name_a,
            "drug_b":               name_b,
            "drug_c":               name_c,
            "combo_name":           f"{name_a} + {name_b} + {name_c}",
            "combo_score":          round(triple_score, 4),
            "base_score":           round(best_pair_score, 4),
            "score_a":              round(drug_a.get("score", 0.0), 4),
            "score_b":              round(drug_b.get("score", 0.0), 4),
            "score_c":              round(score_c, 4),
            "synergy_bonus":        0.05,   # fixed bonus for any triple
            "antagonism_penalty":   -0.15 if any_antagonistic else 0.0,
            "overlap_penalty":      0.0,
            "redundancy_penalty":   0.0,
            "coverage_bonus":       0.0,
            "mechanism_a":          pair_ab["mechanism_a"],
            "mechanism_b":          pair_ab["mechanism_b"],
            "mechanism_c":          mech_c,
            "is_synergistic":       not any_antagonistic,
            "is_antagonistic":      any_antagonistic,
            "combined_gene_coverage": len(shared_genes),
            "shared_genes":         shared_genes[:15],
            "target_genes_a":       list(targets_a)[:10],
            "target_genes_b":       list(targets_b)[:10],
            "target_genes_c":       list(targets_c)[:10],
            "wet_lab_targets":      shared_genes[:5],
            "pair_scores": {
                "ab": pair_ab["combo_score"],
                "ac": pair_ac["combo_score"],
                "bc": pair_bc["combo_score"],
            },
        }

    # ── Main ranking method ───────────────────────────────────────────────────

    def rank_combinations(
        self,
        candidates:      List[Dict],
        disease_genes:   List[str],
        top_n_singles:   int  = 20,
        max_combos:      int  = 50,
        include_triples: bool = True,
        triple_from_top: int  = 10,
    ) -> List[Dict]:
        """
        Generate and rank all pairs (and optionally triples) from
        the top N single-drug candidates.

        Parameters
        ----------
        candidates : list of dict
            Single-drug scored candidates (must have 'score' and 'target_genes').
        disease_genes : list of str
            Disease-associated genes for coverage scoring.
        top_n_singles : int
            How many top single drugs to combine (default 20 → 190 pairs).
        max_combos : int
            Cap on returned combinations (default 50).
        include_triples : bool
            Whether to include 3-drug combinations.
        triple_from_top : int
            Use only this many top drugs for triples (limits compute).

        Returns
        -------
        list of combo dicts, sorted by combo_score descending
        """
        top = sorted(candidates, key=lambda c: c.get("score", 0), reverse=True)
        top = top[:top_n_singles]

        combos: List[Dict] = []

        logger.info(
            f"Generating combinations from {len(top)} top singles "
            f"({'pairs + triples' if include_triples else 'pairs only'})"
        )

        # All pairs
        pair_count = 0
        for drug_a, drug_b in itertools.combinations(top, 2):
            combo = self.score_pair(drug_a, drug_b, disease_genes)
            combos.append(combo)
            pair_count += 1

        logger.info(f"  Generated {pair_count} pairs")

        # Triples (from smaller top set — O(n^3) grows fast)
        if include_triples and len(top) >= 3:
            top_for_triples = top[:triple_from_top]
            triple_count = 0
            for drug_a, drug_b, drug_c in itertools.combinations(top_for_triples, 3):
                combo = self.score_triple(drug_a, drug_b, drug_c, disease_genes)
                combos.append(combo)
                triple_count += 1
            logger.info(f"  Generated {triple_count} triples")

        combos.sort(key=lambda x: x["combo_score"], reverse=True)

        # Log top 5
        logger.info("Top 5 combinations:")
        for i, c in enumerate(combos[:5], 1):
            logger.info(
                f"  #{i}: {c['combo_name']} — "
                f"score={c['combo_score']:.3f} "
                f"synergy={c['is_synergistic']} "
                f"genes={c['combined_gene_coverage']}"
            )

        return combos[:max_combos]

    # ── Utility ───────────────────────────────────────────────────────────────

    def explain_combo(self, combo: Dict) -> str:
        """Generate a human-readable explanation for a combo (for PAG briefs)."""
        lines = [f"Regimen: {combo['combo_name']}"]
        lines.append(f"Combination score: {combo['combo_score']:.3f}")

        if combo.get("is_synergistic"):
            lines.append(
                f"Mechanism synergy: {combo['mechanism_a']} + "
                f"{combo.get('mechanism_b', '')} is a known synergistic pair."
            )
        if combo.get("is_antagonistic"):
            lines.append("WARNING: These mechanisms may antagonise each other.")

        if combo["combined_gene_coverage"] > 0:
            lines.append(
                f"Combined disease gene coverage: "
                f"{combo['combined_gene_coverage']} genes "
                f"({', '.join(combo['shared_genes'][:5])}...)"
            )

        if combo.get("wet_lab_targets"):
            lines.append(
                f"Recommended wet-lab validation targets: "
                f"{', '.join(combo['wet_lab_targets'])}"
            )

        score_breakdown = (
            f"Score breakdown: "
            f"base={combo['base_score']:.3f} "
            f"synergy={combo['synergy_bonus']:+.3f} "
            f"antagonism={combo['antagonism_penalty']:+.3f} "
            f"coverage={combo['coverage_bonus']:+.3f} "
            f"redundancy={combo['redundancy_penalty']:+.3f}"
        )
        lines.append(score_breakdown)
        return "\n".join(lines)
    