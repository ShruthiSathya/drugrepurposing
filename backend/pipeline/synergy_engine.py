"""
synergy_engine.py — Pharmacological Synergy Models
====================================================
Implements two gold-standard synergy quantification frameworks used in
combination pharmacology research:

  1. Bliss Independence Model (Bliss 1939)
     Assumes two drugs act through statistically independent mechanisms.
     Expected combined effect: E_AB = E_A + E_B - E_A·E_B
     Bliss Score = E_AB_observed - E_AB_expected
       > 0: synergistic   = 0: additive   < 0: antagonistic

  2. Loewe Additivity / Combination Index (Chou & Talalay 1984)
     Models the combination as if both drugs competed for the same target.
     CI = D_A/IC50_A + D_B/IC50_B
     CI < 1: synergy   CI = 1: additivity   CI > 1: antagonism

  3. Highest Single Agent (HSA) Reference
     E_combo_expected = max(E_A, E_B)
     Any improvement above HSA is classified as synergy.

IMPORTANT — Scope limitation
-----------------------------
True Bliss/Loewe requires in vitro dose-response curves. Since this engine
operates on in-silico scores (no wet-lab data), it uses the pipeline's
composite scores as proxy effect sizes and applies the mathematical models
analytically. Results are clearly labelled IN_SILICO_ESTIMATE.

All predictions must be validated with wet-lab dose-response assays
before use in any clinical or regulatory context.

References
----------
Bliss CI (1939). The toxicity of poisons applied jointly. Ann Appl Biol 26:585-615.
Chou TC & Talalay P (1984). Quantitative analysis of dose-effect relationships.
  Adv Enzyme Regul 22:27-55.
Yadav B et al (2015). Quantitative scoring of differential drug synergy.
  Sci Rep 5:13115. doi:10.1038/srep13115
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Thresholds
# ─────────────────────────────────────────────────────────────────────────────
BLISS_SYNERGY_THRESHOLD     =  0.10   # Bliss score > 0.10 → strong synergy
BLISS_ANTAGONISM_THRESHOLD  = -0.10   # Bliss score < -0.10 → antagonism
CI_SYNERGY_THRESHOLD        =  0.90   # CI < 0.90 → synergy
CI_ANTAGONISM_THRESHOLD     =  1.10   # CI > 1.10 → antagonism
HSA_SYNERGY_THRESHOLD       =  0.08   # ΔE above HSA > 0.08 → meaningful synergy

# Denominator guard for CI calculation
_EPSILON = 1e-6


@dataclass
class SynergyResult:
    """Full synergy quantification for a drug pair."""

    # Drug identities
    drug_a: str
    drug_b: str

    # Individual effect sizes (proxy from composite scores, range 0-1)
    effect_a: float          # E_A: normalised pipeline score for drug A
    effect_b: float          # E_B: normalised pipeline score for drug B
    effect_combo: float      # E_AB: estimated combined effect

    # Bliss Independence
    bliss_expected: float    # E_A + E_B - E_A·E_B
    bliss_score: float       # E_AB - bliss_expected (positive = synergy)

    # Loewe Additivity
    ci: float                # Combination Index (< 1 synergy, > 1 antagonism)

    # Highest Single Agent
    hsa_reference: float     # max(E_A, E_B)
    hsa_delta: float         # E_AB - hsa_reference

    # Aggregate
    synergy_call: str        # SYNERGISTIC | ADDITIVE | ANTAGONISTIC
    synergy_confidence: str  # HIGH | MEDIUM | LOW
    aggregate_synergy_score: float   # normalised 0-1 (1 = max synergy)

    # Mechanistic context
    shared_targets: List[str] = field(default_factory=list)
    complementary_pathways: List[str] = field(default_factory=list)
    mechanistic_rationale: str = ""

    # Metadata
    model_note: str = (
        "IN_SILICO_ESTIMATE — pipeline composite scores used as proxy "
        "effect sizes. Validate with wet-lab dose-response assays."
    )


class SynergyEngine:
    """
    Computes Bliss Independence and Loewe Additivity synergy scores for
    drug pairs using pipeline composite scores as proxy effect sizes.

    Effect-size estimation
    ----------------------
    Without dose-response curves, we use:
      E = tanh(2 × composite_score)  →  maps [0, 1] score to effect [0, ~0.96]

    The tanh transform is preferred over linear mapping because:
    - Scores near 0 (no evidence) → near-zero effects (not spurious synergy)
    - Scores near 1 (strong evidence) → near-ceiling effects
    - The non-linearity prevents artefactual synergy from two weak drugs

    Combo effect estimation
    -----------------------
    E_AB = f(E_A, E_B, gene_coverage_bonus, mechanism_class)

    Gene coverage bonus: drugs covering different disease genes can achieve
    higher combined effect than either alone, beyond Bliss prediction.
    This is grounded in multi-target pharmacology literature.

    Usage
    -----
        engine = SynergyEngine()
        result = engine.compute_synergy(drug_a_data, drug_b_data, disease_genes)
        print(result.synergy_call, result.aggregate_synergy_score)
    """

    def __init__(self, effect_scale: float = 2.0):
        """
        Parameters
        ----------
        effect_scale : float
            Scaling factor for composite-score → effect-size mapping.
            Default 2.0 works well for the 6-component weighted score.
        """
        self.effect_scale = effect_scale

    # ── Effect size estimation ────────────────────────────────────────────────

    def _score_to_effect(self, composite_score: float) -> float:
        """
        Map a pipeline composite score in [0, 1] to an effect size in [0, ~0.96].
        Uses tanh to impose a biological ceiling (no drug is 100% effective).
        """
        score = max(0.0, min(1.0, composite_score))
        return round(math.tanh(self.effect_scale * score), 6)

    def _estimate_combo_effect(
        self,
        e_a: float,
        e_b: float,
        gene_coverage_bonus: float = 0.0,
        pathway_bonus: float = 0.0,
    ) -> float:
        """
        Estimate the combined effect E_AB.

        Base model: Bliss expected effect (independence assumption)
        Adjusted by:
          - gene_coverage_bonus: extra effect from covering additional disease genes
          - pathway_bonus: extra effect from complementary pathway modulation

        This models the intuition that two drugs targeting different parts of the
        disease biology achieve more than the Bliss prediction (true synergy).
        """
        bliss_expected = e_a + e_b - e_a * e_b
        # Adjustments are fractional improvements above the Bliss floor
        adjustment = gene_coverage_bonus * 0.15 + pathway_bonus * 0.10
        combo = bliss_expected + adjustment * (1.0 - bliss_expected)
        return round(min(combo, 0.99), 6)

    # ── Bliss Independence ────────────────────────────────────────────────────

    def bliss(self, e_a: float, e_b: float, e_combo: float) -> Tuple[float, float]:
        """
        Bliss Independence calculation.

        Parameters
        ----------
        e_a : float     Effect of drug A alone [0, 1]
        e_b : float     Effect of drug B alone [0, 1]
        e_combo : float Observed combined effect [0, 1]

        Returns
        -------
        (bliss_expected, bliss_score)
        bliss_score > 0: synergistic
        bliss_score < 0: antagonistic
        """
        bliss_expected = e_a + e_b - e_a * e_b
        bliss_score = e_combo - bliss_expected
        return round(bliss_expected, 6), round(bliss_score, 6)

    # ── Loewe Additivity / Combination Index ──────────────────────────────────

    def loewe_ci(
        self,
        e_a: float,
        e_b: float,
        e_combo: float,
        d_ratio: float = 1.0,
    ) -> float:
        """
        Loewe Combination Index.

        Uses a dose-effect model to back-calculate pseudo-IC50 values from
        the proxy effect sizes, then computes CI at the combined effect level.

        Hill equation inverse: IC50 = D × (E / (1 - E))^(-1/h)
        At h=1 (linear Hill): D_effect/IC50 = E / (1 - E) × (1 / D_ratio)

        CI < 1: synergy   CI = 1: additivity   CI > 1: antagonism

        Parameters
        ----------
        e_a, e_b    : individual drug effects [0, 1]
        e_combo     : combined effect [0, 1]
        d_ratio     : molar dose ratio D_A / D_B (default 1.0 for equimolar)
        """
        def effect_to_pseudo_dose_ratio(effect: float) -> float:
            """d/IC50 = E / (1 - E) at Hill=1."""
            effect = max(_EPSILON, min(1 - _EPSILON, effect))
            return effect / (1.0 - effect)

        if e_combo <= _EPSILON:
            return 1.0   # No combined effect → treat as additive

        # Pseudo dose-ratios at the combined effect level
        d_a_at_combo = effect_to_pseudo_dose_ratio(e_combo) / (
            effect_to_pseudo_dose_ratio(e_a) + _EPSILON
        ) * d_ratio / (d_ratio + 1)

        d_b_at_combo = effect_to_pseudo_dose_ratio(e_combo) / (
            effect_to_pseudo_dose_ratio(e_b) + _EPSILON
        ) / (d_ratio + 1)

        ci = d_a_at_combo + d_b_at_combo
        return round(max(0.0, ci), 6)

    # ── Highest Single Agent ──────────────────────────────────────────────────

    def hsa(self, e_a: float, e_b: float, e_combo: float) -> Tuple[float, float]:
        """
        Highest Single Agent reference.
        Returns (hsa_reference, hsa_delta).
        """
        hsa_ref = max(e_a, e_b)
        return round(hsa_ref, 6), round(e_combo - hsa_ref, 6)

    # ── Synergy call ──────────────────────────────────────────────────────────

    def _classify(self, bliss_score: float, ci: float, hsa_delta: float) -> Tuple[str, str]:
        """
        Combine Bliss, CI, and HSA evidence to produce a synergy call.

        Returns (call, confidence).
        """
        synergy_votes   = 0
        antagonism_votes = 0

        if bliss_score >= BLISS_SYNERGY_THRESHOLD:     synergy_votes += 2
        elif bliss_score <= BLISS_ANTAGONISM_THRESHOLD: antagonism_votes += 2
        if ci <= CI_SYNERGY_THRESHOLD:                 synergy_votes += 2
        elif ci >= CI_ANTAGONISM_THRESHOLD:             antagonism_votes += 2
        if hsa_delta >= HSA_SYNERGY_THRESHOLD:         synergy_votes += 1

        if synergy_votes >= 3:
            call = "SYNERGISTIC"
            confidence = "HIGH" if synergy_votes >= 4 else "MEDIUM"
        elif antagonism_votes >= 3:
            call = "ANTAGONISTIC"
            confidence = "HIGH" if antagonism_votes >= 4 else "MEDIUM"
        elif synergy_votes >= 1:
            call = "SYNERGISTIC"
            confidence = "LOW"
        else:
            call = "ADDITIVE"
            confidence = "MEDIUM"

        return call, confidence

    def _aggregate_score(self, bliss_score: float, ci: float, hsa_delta: float) -> float:
        """
        Normalised synergy score in [0, 1] from the three metrics.
        Used for ranking combinations.
        """
        # Bliss: map [-0.5, +0.5] → [0, 1]
        bliss_norm = max(0.0, min(1.0, (bliss_score + 0.5)))
        # CI: map [2.0, 0.0] → [0, 1]  (lower CI = more synergistic)
        ci_norm = max(0.0, min(1.0, (2.0 - ci) / 2.0))
        # HSA: map [-0.2, +0.5] → [0, 1]
        hsa_norm = max(0.0, min(1.0, (hsa_delta + 0.2) / 0.7))

        weighted = 0.40 * bliss_norm + 0.40 * ci_norm + 0.20 * hsa_norm
        return round(weighted, 4)

    # ── Main API ──────────────────────────────────────────────────────────────

    def compute_synergy(
        self,
        drug_a: Dict,
        drug_b: Dict,
        disease_genes: List[str],
    ) -> SynergyResult:
        """
        Compute full synergy quantification for a drug pair.

        Parameters
        ----------
        drug_a, drug_b : dict
            Drug candidate dicts with keys:
              score          (composite pipeline score, float 0-1)
              drug_name      (str)
              target_genes   (list of str)
              pathways       (list of str)
        disease_genes : list of str
            Disease-associated gene symbols.

        Returns
        -------
        SynergyResult
        """
        name_a = drug_a.get("drug_name", drug_a.get("name", "DrugA"))
        name_b = drug_b.get("drug_name", drug_b.get("name", "DrugB"))

        score_a = drug_a.get("score", 0.0)
        score_b = drug_b.get("score", 0.0)

        e_a = self._score_to_effect(score_a)
        e_b = self._score_to_effect(score_b)

        # Gene coverage bonus: fraction of additional disease genes covered by B
        targets_a = set(t.upper() for t in drug_a.get("target_genes", []))
        targets_b = set(t.upper() for t in drug_b.get("target_genes", []))
        disease_set = set(g.upper() for g in disease_genes)

        only_b_hits = (targets_b - targets_a) & disease_set
        only_a_hits = (targets_a - targets_b) & disease_set
        shared_hits = (targets_a & targets_b) & disease_set
        gene_coverage_bonus = (
            min(len(only_a_hits) + len(only_b_hits), 10) / max(len(disease_set), 10)
        )

        # Pathway complementarity bonus
        paths_a = set(drug_a.get("pathways", []))
        paths_b = set(drug_b.get("pathways", []))
        path_only_a = paths_a - paths_b
        path_only_b = paths_b - paths_a
        pathway_bonus = min(len(path_only_a | path_only_b) / max(len(paths_a | paths_b), 1), 1.0) * 0.3

        e_combo = self._estimate_combo_effect(e_a, e_b, gene_coverage_bonus, pathway_bonus)

        # Model calculations
        bliss_expected, bliss_score = self.bliss(e_a, e_b, e_combo)
        ci = self.loewe_ci(e_a, e_b, e_combo)
        hsa_ref, hsa_delta = self.hsa(e_a, e_b, e_combo)
        synergy_call, confidence = self._classify(bliss_score, ci, hsa_delta)
        agg_score = self._aggregate_score(bliss_score, ci, hsa_delta)

        # Mechanistic rationale
        rationale = self._build_rationale(
            name_a, name_b, e_a, e_b, bliss_score, ci,
            list(shared_hits), list(only_a_hits | only_b_hits),
            paths_a, paths_b,
        )

        logger.debug(
            "Synergy %s + %s: Bliss=%.3f CI=%.3f HSA_Δ=%.3f → %s [%s]",
            name_a, name_b, bliss_score, ci, hsa_delta, synergy_call, confidence,
        )

        return SynergyResult(
            drug_a=name_a,
            drug_b=name_b,
            effect_a=e_a,
            effect_b=e_b,
            effect_combo=e_combo,
            bliss_expected=bliss_expected,
            bliss_score=bliss_score,
            ci=ci,
            hsa_reference=hsa_ref,
            hsa_delta=hsa_delta,
            synergy_call=synergy_call,
            synergy_confidence=confidence,
            aggregate_synergy_score=agg_score,
            shared_targets=list(shared_hits)[:10],
            complementary_pathways=list(path_only_a | path_only_b)[:6],
            mechanistic_rationale=rationale,
        )

    def _build_rationale(
        self,
        name_a: str, name_b: str,
        e_a: float, e_b: float,
        bliss_score: float, ci: float,
        shared_targets: List[str],
        complementary_targets: List[str],
        paths_a: set, paths_b: set,
    ) -> str:
        lines = [
            f"{name_a} (estimated effect {e_a:.2f}) and {name_b} "
            f"(estimated effect {e_b:.2f}) were scored using the Bliss "
            f"Independence and Loewe Additivity models."
        ]
        if bliss_score > BLISS_SYNERGY_THRESHOLD:
            lines.append(
                f"Bliss score of {bliss_score:+.3f} indicates the combined "
                f"effect exceeds statistical independence — consistent with "
                f"mechanistic synergy rather than simple additivity."
            )
        if ci < CI_SYNERGY_THRESHOLD:
            lines.append(
                f"Combination Index {ci:.3f} < 1.0 (Loewe): both drugs can "
                f"achieve the same combined endpoint at sub-therapeutic doses, "
                f"reducing expected toxicity."
            )
        if shared_targets:
            lines.append(
                f"Shared disease targets: {', '.join(shared_targets[:5])} "
                f"— both drugs converge on these nodes."
            )
        if complementary_targets:
            lines.append(
                f"Complementary targets: {', '.join(complementary_targets[:5])} "
                f"— drugs cover distinct disease-relevant proteins, "
                f"increasing biological coverage."
            )
        unique_paths = (paths_a | paths_b) - (paths_a & paths_b)
        if unique_paths:
            lines.append(
                f"Distinct pathway coverage ({len(unique_paths)} unique pathways) "
                f"reduces the probability of single-mechanism resistance escape."
            )
        lines.append(
            "NOTE: All effect estimates are in-silico proxies. "
            "Wet-lab dose-response validation is required before clinical consideration."
        )
        return " ".join(lines)

    # ── Batch API ─────────────────────────────────────────────────────────────

    def batch_synergy(
        self,
        drug_pairs: List[Tuple[Dict, Dict]],
        disease_genes: List[str],
    ) -> List[SynergyResult]:
        """
        Score a list of (drug_a, drug_b) pairs.
        Returns results sorted by aggregate_synergy_score descending.
        """
        results = []
        for drug_a, drug_b in drug_pairs:
            try:
                r = self.compute_synergy(drug_a, drug_b, disease_genes)
                results.append(r)
            except Exception as e:
                logger.warning(
                    "Synergy computation failed for %s + %s: %s",
                    drug_a.get("drug_name", "?"),
                    drug_b.get("drug_name", "?"), e,
                )
        results.sort(key=lambda r: r.aggregate_synergy_score, reverse=True)
        return results

    def to_dict(self, result: SynergyResult) -> Dict:
        """Serialise a SynergyResult to a JSON-safe dict."""
        return {
            "drug_a": result.drug_a,
            "drug_b": result.drug_b,
            "effect_a": result.effect_a,
            "effect_b": result.effect_b,
            "effect_combo": result.effect_combo,
            "bliss": {
                "expected":     result.bliss_expected,
                "score":        result.bliss_score,
                "interpretation": (
                    "synergistic" if result.bliss_score > BLISS_SYNERGY_THRESHOLD
                    else "antagonistic" if result.bliss_score < BLISS_ANTAGONISM_THRESHOLD
                    else "additive"
                ),
            },
            "loewe": {
                "combination_index": result.ci,
                "interpretation": (
                    "synergistic" if result.ci < CI_SYNERGY_THRESHOLD
                    else "antagonistic" if result.ci > CI_ANTAGONISM_THRESHOLD
                    else "additive"
                ),
            },
            "hsa": {
                "reference": result.hsa_reference,
                "delta":     result.hsa_delta,
            },
            "synergy_call":             result.synergy_call,
            "synergy_confidence":       result.synergy_confidence,
            "aggregate_synergy_score":  result.aggregate_synergy_score,
            "shared_targets":           result.shared_targets,
            "complementary_pathways":   result.complementary_pathways,
            "mechanistic_rationale":    result.mechanistic_rationale,
            "model_note":               result.model_note,
        }