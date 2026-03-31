"""
combo_worker.py — Parallel Combination Scoring Engine
======================================================
Implements the "Combination Loop" using multiprocessing to score thousands
of drug pairs efficiently. Integrates:

  1. Combination enumeration via itertools.combinations
  2. Bliss Independence + Loewe Additivity synergy models (SynergyEngine)
  3. CYP450 metabolic overlap check (CYP450Checker — cached static table)
  4. Toxicity Ω penalty (ToxicityEngine — curated table, no async in worker)
  5. Existing combo_scorer mechanism synergy/antagonism table

Architecture
------------
  ComboWorkerPool: spawns N worker processes
  Each worker: receives a batch of (drug_a, drug_b, disease_genes) tuples
               returns a scored SimulationProof dict per pair

  Main process: collects results, sorts by final_score, writes JSON report

Why multiprocessing vs asyncio
-------------------------------
  asyncio is ideal for I/O bound work (API calls).
  Combination scoring is CPU-bound (thousands of math operations).
  multiprocessing.Pool distributes CPU work across physical cores.
  I/O (openFDA, STRING) is done BEFORE launching workers — workers use
  only in-memory data and the curated static tables.

Usage
-----
    from backend.pipeline.combo_worker import ComboWorkerPool

    pool = ComboWorkerPool(n_workers=4)
    proofs = pool.run(
        candidates=scored_drugs,
        disease_genes=disease_data["genes"],
        disease_name="pulmonary arterial hypertension",
        max_pairs=5000,
    )
    # proofs: list of SimulationProof dicts, sorted by final_score desc
    pool.write_report(proofs, output_path="simulation_proof.json")
"""

import itertools
import json
import logging
import math
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Worker-side imports are done inside worker functions to avoid fork issues ──


def _score_pair_worker(args: Tuple) -> Optional[Dict]:
    """
    Worker function executed in a subprocess.
    Receives a serialisable args tuple; returns a SimulationProof dict or None.

    args = (drug_a_dict, drug_b_dict, disease_genes, disease_name)
    """
    # Deferred imports: safe inside subprocess
    from backend.pipeline.synergy_engine import SynergyEngine
    from backend.pipeline.toxicity_engine import ToxicityEngine
    from backend.pipeline.cyp450_checker import CYP450Checker
    from backend.pipeline.combo_scorer import CombinationScorer, classify_mechanism

    drug_a, drug_b, disease_genes, disease_name = args

    name_a = drug_a.get("drug_name", drug_a.get("name", "DrugA"))
    name_b = drug_b.get("drug_name", drug_b.get("name", "DrugB"))

    try:
        # ── 1. Mechanism synergy/antagonism (fast, no I/O) ────────────────
        scorer = CombinationScorer(disease_name=disease_name)
        pair_result = scorer.score_pair(drug_a, drug_b, disease_genes)

        # Skip antagonistic pairs immediately
        if pair_result.get("is_antagonistic"):
            return None

        # ── 2. CYP450 check (static table, synchronous) ───────────────────
        cyp_checker = CYP450Checker()
        cyp_result = cyp_checker._check_pair_static(name_a, name_b)

        # Skip CRITICAL CYP450 interactions
        if cyp_result.severity == "CRITICAL":
            return None

        # ── 3. Toxicity Ω penalty (curated table, synchronous) ────────────
        tox_engine = ToxicityEngine()
        tox_result = tox_engine.assess_combination_sync([name_a, name_b])

        # Skip CRITICAL toxicity combos
        if tox_result.is_critical:
            return None

        # ── 4. Bliss + Loewe synergy ──────────────────────────────────────
        syn_engine = SynergyEngine()
        syn_result = syn_engine.compute_synergy(drug_a, drug_b, disease_genes)

        # ── 5. Final score composition ────────────────────────────────────
        base_score       = pair_result.get("combo_score", 0.0)
        synergy_bonus    = syn_result.aggregate_synergy_score * 0.15
        cyp_penalty      = cyp_result.omega_penalty   # in [-0.40, 0]
        tox_penalty      = tox_result.omega_penalty if tox_result.omega_penalty != float("-inf") else -1.0
        cumulative_penalty = cyp_penalty + tox_penalty

        final_score = max(
            min(base_score + synergy_bonus + cumulative_penalty, 1.0),
            0.0
        )

        # ── 6. Assemble SimulationProof ───────────────────────────────────
        proof = _build_simulation_proof(
            drug_a=drug_a,
            drug_b=drug_b,
            disease_name=disease_name,
            disease_genes=disease_genes,
            pair_result=pair_result,
            syn_engine=syn_engine,
            syn_result=syn_result,
            cyp_result=cyp_result,
            tox_result=tox_result,
            final_score=final_score,
            synergy_bonus=synergy_bonus,
            cyp_penalty=cyp_penalty,
            tox_penalty=tox_penalty,
        )
        return proof

    except Exception as e:
        logger.warning(f"Worker failed for {name_a} + {name_b}: {e}")
        return None


def _build_simulation_proof(
    drug_a: Dict,
    drug_b: Dict,
    disease_name: str,
    disease_genes: List[str],
    pair_result: Dict,
    syn_engine: Any,
    syn_result: Any,
    cyp_result: Any,
    tox_result: Any,
    final_score: float,
    synergy_bonus: float,
    cyp_penalty: float,
    tox_penalty: float,
) -> Dict:
    """Assemble the structured SimulationProof JSON for a drug pair."""
    from backend.pipeline.synergy_engine import SynergyEngine
    from backend.pipeline.cyp450_checker import CYP450Checker
    from backend.pipeline.toxicity_engine import ToxicityEngine

    name_a = drug_a.get("drug_name", drug_a.get("name", "DrugA"))
    name_b = drug_b.get("drug_name", drug_b.get("name", "DrugB"))

    targets_a = set(t.upper() for t in drug_a.get("target_genes", []) or drug_a.get("targets", []))
    targets_b = set(t.upper() for t in drug_b.get("target_genes", []) or drug_b.get("targets", []))
    disease_set = set(g.upper() for g in disease_genes)
    combined_targets = (targets_a | targets_b) & disease_set

    return {
        # ── Identity ──────────────────────────────────────────────────────
        "regimen":      f"{name_a} + {name_b}",
        "drug_a":       name_a,
        "drug_b":       name_b,
        "disease":      disease_name,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_version": "TwinTrial v1.1 (Bliss+Loewe+Ω)",

        # ── Final score ───────────────────────────────────────────────────
        "final_score":  round(final_score, 4),
        "score_breakdown": {
            "mechanism_combo_score": round(pair_result.get("combo_score", 0), 4),
            "bliss_loewe_synergy_bonus": round(synergy_bonus, 4),
            "cyp450_omega_penalty": round(cyp_penalty, 4),
            "toxicity_omega_penalty": round(tox_penalty, 4),
            "final_score": round(final_score, 4),
        },

        # ── Target pathway nodes ──────────────────────────────────────────
        "target_pathway_nodes": {
            "combined_disease_targets": sorted(combined_targets),
            "targets_drug_a_only": sorted(targets_a - targets_b - disease_set),
            "targets_drug_b_only": sorted(targets_b - targets_a - disease_set),
            "targets_shared_with_disease": sorted(combined_targets),
            "shared_pathways": pair_result.get("shared_genes", [])[:10],
            "total_disease_coverage": f"{len(combined_targets)}/{len(disease_set)} disease genes",
        },

        # ── Synergy models ─────────────────────────────────────────────────
        "synergy_analysis": {
            "synergy_call":              syn_result.synergy_call,
            "synergy_confidence":        syn_result.synergy_confidence,
            "aggregate_synergy_score":   syn_result.aggregate_synergy_score,
            "bliss_independence": {
                "drug_a_effect_proxy":  syn_result.effect_a,
                "drug_b_effect_proxy":  syn_result.effect_b,
                "expected_combined":    syn_result.bliss_expected,
                "estimated_combined":   syn_result.effect_combo,
                "bliss_score":          syn_result.bliss_score,
                "interpretation": (
                    "Synergistic — combined effect exceeds Bliss prediction"
                    if syn_result.bliss_score > 0.10
                    else "Antagonistic" if syn_result.bliss_score < -0.10
                    else "Additive"
                ),
            },
            "loewe_additivity": {
                "combination_index": syn_result.ci,
                "interpretation": (
                    "Synergistic (CI < 1.0)"
                    if syn_result.ci < 0.90
                    else "Antagonistic (CI > 1.0)" if syn_result.ci > 1.10
                    else "Additive (CI ≈ 1.0)"
                ),
            },
            "highest_single_agent": {
                "hsa_reference": syn_result.hsa_reference,
                "delta_above_hsa": syn_result.hsa_delta,
            },
        },

        # ── Mechanistic rationale ──────────────────────────────────────────
        "mechanistic_rationale": {
            "summary": syn_result.mechanistic_rationale,
            "mechanism_a":    drug_a.get("mechanism", ""),
            "mechanism_b":    drug_b.get("mechanism", ""),
            "mechanism_class_a": pair_result.get("mechanism_a", ""),
            "mechanism_class_b": pair_result.get("mechanism_b", ""),
            "is_known_synergistic_pair": pair_result.get("is_synergistic", False),
            "complementary_pathways": syn_result.complementary_pathways,
            "gene_coverage_bonus": pair_result.get("coverage_bonus", 0),
            "redundancy_penalty": pair_result.get("redundancy_penalty", 0),
        },

        # ── Safety: CYP450 ─────────────────────────────────────────────────
        "metabolic_safety": {
            "cyp450_severity":      cyp_result.severity,
            "omega_penalty_cyp":    cyp_result.omega_penalty,
            "enzymes_affected":     cyp_result.enzymes_affected,
            "risk_description":     cyp_result.risk_description,
            "recommendation":       cyp_result.recommendation,
            "interactions": cyp_result.interactions,
        },

        # ── Safety: Adverse Events + Ω ─────────────────────────────────────
        "adverse_event_safety": {
            "is_critical":          tox_result.is_critical,
            "omega_penalty_ae":     (
                tox_result.omega_penalty
                if tox_result.omega_penalty != float("-inf")
                else "NEGATIVE_INFINITY"
            ),
            "critical_flags":       tox_result.critical_flags,
            "cumulative_burden":    tox_result.cumulative_burden,
            "safety_margin":        tox_result.safety_margin,
            "shared_ae_categories": tox_result.shared_ae_categories,
            "per_drug_profiles":    tox_result.per_drug_profiles,
            "recommendation":       tox_result.recommendation,
            "ae_data_source":       tox_result.ae_source,
        },

        # ── Overall safety margin (0-1) ────────────────────────────────────
        "safety_margin": round(
            tox_result.safety_margin * (1.0 - abs(min(cyp_penalty, 0))),
            4
        ),

        # ── Disclaimers ────────────────────────────────────────────────────
        "limitations": [
            "All synergy scores are in-silico estimates using pipeline composite "
            "scores as proxy effect sizes. Bliss/Loewe require wet-lab dose-response "
            "curves for validation.",
            "CYP450 interaction data derived from curated table + openFDA labels. "
            "Not a substitute for full PK/PD modelling.",
            "Adverse event profiles from openFDA FAERS and OFFSIDES curated dataset. "
            "FAERS data is not randomised — causality not established.",
            "This output is a research tool, not medical advice.",
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main pool orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class ComboWorkerPool:
    """
    Parallel combination scoring engine.
    Enumerates drug pairs via itertools.combinations and distributes
    scoring across multiple CPU cores using multiprocessing.Pool.
    """

    def __init__(self, n_workers: Optional[int] = None):
        self.n_workers = n_workers or max(1, mp.cpu_count() - 1)
        logger.info(f"ComboWorkerPool: {self.n_workers} workers")

    def run(
        self,
        candidates: List[Dict],
        disease_genes: List[str],
        disease_name: str,
        max_pairs: int = 5000,
        top_n_singles: int = 30,
        include_triples: bool = False,
    ) -> List[Dict]:
        """
        Run parallel combination scoring.

        Parameters
        ----------
        candidates : list of dict
            Single-drug scored candidates from generate_candidates().
        disease_genes : list of str
        disease_name : str
        max_pairs : int
            Cap on total pairs to evaluate (default 5000).
        top_n_singles : int
            Restrict combinations to top-N singles by score (default 30 → 435 pairs).
        include_triples : bool
            Include 3-drug combinations (expensive).

        Returns
        -------
        list of SimulationProof dicts, sorted by final_score descending.
        Non-scoring pairs (antagonistic, critical toxicity) are excluded.
        """
        top = sorted(candidates, key=lambda c: c.get("score", 0), reverse=True)
        top = top[:top_n_singles]

        # Build argument tuples (serialisable for multiprocessing)
        pair_args = [
            (drug_a, drug_b, disease_genes, disease_name)
            for drug_a, drug_b in itertools.combinations(top, 2)
        ][:max_pairs]

        if include_triples and len(top) >= 3:
            triple_cap = min(max_pairs // 2, 500)
            triple_args = [
                (drug_a, drug_b, drug_c, disease_genes, disease_name)
                for drug_a, drug_b, drug_c in itertools.combinations(top[:15], 3)
            ][:triple_cap]
        else:
            triple_args = []

        n_total = len(pair_args) + len(triple_args)
        logger.info(
            f"Combination loop: {len(pair_args)} pairs + {len(triple_args)} triples "
            f"= {n_total} evaluations on {self.n_workers} workers"
        )

        t0 = time.time()
        results: List[Dict] = []

        # Pairs via Pool
        if pair_args:
            with mp.Pool(processes=self.n_workers) as pool:
                raw = pool.map(_score_pair_worker, pair_args, chunksize=20)
            results = [r for r in raw if r is not None]

        logger.info(
            f"Pairs: {len(results)}/{len(pair_args)} passed filters "
            f"in {time.time() - t0:.1f}s"
        )

        # Triples: run sequentially to avoid excessive memory (O(n³))
        if triple_args:
            from backend.pipeline.combo_scorer import CombinationScorer
            from backend.pipeline.synergy_engine import SynergyEngine
            from backend.pipeline.toxicity_engine import ToxicityEngine
            t_scorer = CombinationScorer(disease_name=disease_name)
            t_synergy = SynergyEngine()
            t_tox = ToxicityEngine()

            for triple_arg in triple_args:
                drug_a, drug_b, drug_c = triple_arg[0], triple_arg[1], triple_arg[2]
                triple_r = t_scorer.score_triple(drug_a, drug_b, drug_c, disease_genes)
                if triple_r.get("is_antagonistic"):
                    continue
                tox_r = t_tox.assess_combination_sync([
                    drug_a.get("drug_name", ""),
                    drug_b.get("drug_name", ""),
                    drug_c.get("drug_name", ""),
                ])
                if tox_r.is_critical:
                    continue
                triple_r["final_score"] = max(
                    triple_r.get("combo_score", 0) + tox_r.omega_penalty, 0.0
                )
                triple_r["safety_margin"] = tox_r.safety_margin
                triple_r["tox_result"] = t_tox.to_dict(tox_r)
                results.append(triple_r)

        results.sort(key=lambda r: r.get("final_score", 0), reverse=True)
        logger.info(
            f"Combination loop complete: {len(results)} viable combinations "
            f"in {time.time() - t0:.1f}s total"
        )
        return results

    def write_report(
        self,
        proofs: List[Dict],
        output_path: str = "simulation_proof.json",
        top_n: int = 20,
    ) -> Path:
        """
        Write structured SimulationProof JSON report.

        Parameters
        ----------
        proofs : list of SimulationProof dicts
        output_path : str
        top_n : int
            Include only top N results in the report.

        Returns
        -------
        Path to the written file.
        """
        report = {
            "report_type": "TwinTrial SimulationProof",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": "Bliss Independence + Loewe Additivity + CYP450-Ω + AE-Ω",
            "n_combinations_evaluated": len(proofs),
            "top_n_shown": top_n,
            "summary": {
                "top_regimen":          proofs[0]["regimen"] if proofs else None,
                "top_final_score":      proofs[0].get("final_score", 0) if proofs else 0,
                "top_safety_margin":    proofs[0].get("safety_margin", 0) if proofs else 0,
                "n_synergistic":        sum(
                    1 for p in proofs
                    if p.get("synergy_analysis", {}).get("synergy_call") == "SYNERGISTIC"
                ),
            },
            "top_combinations": proofs[:top_n],
        }

        out = Path(output_path)
        out.write_text(json.dumps(report, indent=2, default=str))
        logger.info(f"SimulationProof written: {out.resolve()}")
        return out