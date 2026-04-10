"""
production_pipeline.py  (v5.0 — Transcriptomics-Enabled)
=========================================================
Key changes vs v4.1
--------------------
1. `use_tissue` is NO LONGER deleted / bypassed.
   TissueExpressionGate applies a continuous multiplier to candidate scores,
   penalising drugs whose targets aren't expressed in the disease-relevant
   tissue without completely eliminating them.

2. TranscriptomicSignature extends the disease gene set beyond OpenTargets
   GWAS associations to include tissue-anchored expression data.

3. DysregulationPathwayScorer reweights combo scores based on how precisely
   they repair the disease's specifically broken pathways.

4. TranscriptomicSubtypeSimulator runs per-subtype precision in-silico trials,
   producing patient-stratified ORR estimates rather than population averages.

All four improvements degrade gracefully — if any API fails, the pipeline
falls back to the v4.1 behaviour with a warning.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import networkx as nx

from .combo_scorer import rank_combinations
from .data_fetcher import ProductionDataFetcher
from .drug_similarity import DrugSimilarityScorer, batch_similarity_scores, build_reference_smiles
from .generic_filter import GenericDrugFilter
from .insilico_trial import InSilicoTrialSimulator
from .ppi_network import PPINetworkScorer, batch_ppi_scores
from .scorer import DEFAULT_COMPONENT_WEIGHTS, ProductionScorer
from .treatment_plan import TreatmentPlanAssembler

logger = logging.getLogger(__name__)


class ProductionPipeline:
    """
    Main entry point for disease → ranked repurposing hypotheses.
    v5.0: Transcriptomics-enabled.
    """

    def __init__(
        self,
        *,
        cache_dir: str = "/tmp/drug_repurposing_cache",
        component_weights: Optional[Dict[str, float]] = None,
        enable_transcriptomics: bool = True,
    ) -> None:
        self.data_fetcher = ProductionDataFetcher(cache_dir=cache_dir)
        self.generic_filter = GenericDrugFilter()
        self.component_weights = component_weights or DEFAULT_COMPONENT_WEIGHTS.copy()
        self.enable_transcriptomics = enable_transcriptomics

        self._ppi_scorer: Optional[PPINetworkScorer] = None
        self._similarity_scorer: Optional[DrugSimilarityScorer] = None

        self.drugs_cache: Optional[List[Dict]] = None
        self.generic_drugs_cache: Optional[List[Dict]] = None

    # ------------------------------------------------------------------
    # Drug pool management
    # ------------------------------------------------------------------

    async def fetch_approved_drugs(self, limit: int = 3000) -> List[Dict]:
        if self.drugs_cache is None:
            self.drugs_cache = await self.data_fetcher.fetch_approved_drugs(limit=limit)
            logger.info("Fetched %d approved drugs", len(self.drugs_cache))
        return self.drugs_cache[:limit]

    async def fetch_generic_drugs(self, limit: int = 3000) -> Tuple[List[Dict], List[Dict], Dict]:
        if self.generic_drugs_cache is not None:
            return self.generic_drugs_cache[:limit], [], self.generic_filter.get_stats()

        all_drugs = await self.fetch_approved_drugs(limit=limit)
        generics, excluded = self.generic_filter.filter_to_generics(all_drugs)
        self.generic_drugs_cache = generics
        logger.info("Generic pool: %d kept, %d excluded", len(generics), len(excluded))
        return generics[:limit], excluded, self.generic_filter.get_stats()

    # ------------------------------------------------------------------
    # Core scoring path
    # ------------------------------------------------------------------

    async def generate_candidates(
        self,
        disease_data: Dict,
        drugs_data: List[Dict],
        *,
        min_score: float = 0.0,
        fetch_pubmed: bool = False,
        fetch_ppi: bool = True,
        fetch_similarity: bool = True,
        use_efo: bool = False,
        use_tissue: bool = True,          # ← NO LONGER DELETED
        use_polypharm: bool = False,
    ) -> List[Dict]:
        """
        Score all drugs for one disease using the cleaned scorer.

        v5.0: use_tissue parameter is now honoured. When True, a tissue
        expression gate is applied AFTER initial scoring (in generate_treatment_plan).
        The parameter is retained here for API compatibility.
        """
        del fetch_pubmed, use_polypharm  # intentionally disabled in the clean path

        disease_data_local = dict(disease_data)
        if use_efo:
            try:
                from .efo_ontology import EFOOntologyExpander
                expander = EFOOntologyExpander(
                    session=self.data_fetcher.session, descendant_depth=2
                )
                try:
                    disease_data_local = await expander.expand_disease_genes(disease_data_local)
                finally:
                    await expander.close()
            except Exception as exc:
                logger.warning("EFO expansion unavailable: %s", exc)

        disease_name = disease_data_local["name"]
        disease_genes = disease_data_local.get("genes", []) or []

        ppi_map: Dict[str, float] = {}
        if fetch_ppi and disease_genes:
            if self._ppi_scorer is None:
                self._ppi_scorer = PPINetworkScorer()
            try:
                ppi_results = await batch_ppi_scores(
                    drugs_data=drugs_data,
                    disease_genes=disease_genes,
                    scorer=self._ppi_scorer,
                )
                ppi_map = {name: score for name, (score, _) in ppi_results.items()}
            except Exception as exc:
                logger.warning("PPI scoring unavailable: %s", exc)

        similarity_map: Dict[str, float] = {}
        if fetch_similarity:
            try:
                if self._similarity_scorer is None:
                    self._similarity_scorer = DrugSimilarityScorer()
                ref_smiles, ref_names = build_reference_smiles(disease_name, drugs_data)
                if ref_smiles:
                    similarity_results = batch_similarity_scores(
                        drugs_data=drugs_data,
                        reference_smiles=ref_smiles,
                        reference_names=ref_names,
                        scorer=self._similarity_scorer,
                    )
                    similarity_map = {
                        name: score for name, (score, _) in similarity_results.items()
                    }
            except Exception as exc:
                logger.warning("Similarity scoring unavailable: %s", exc)

        scorer = ProductionScorer(nx.Graph(), component_weights=self.component_weights)
        candidates: List[Dict] = []

        for drug in drugs_data:
            drug_name = drug["name"]
            ppi_score = float(ppi_map.get(drug_name, 0.0))
            similarity_score = float(similarity_map.get(drug_name, 0.0))

            score, evidence = scorer.score_drug_disease_match(
                drug_name=drug_name,
                disease_name=disease_name,
                disease_data=disease_data_local,
                drug_data=drug,
                external_literature_score=0.0,
                ppi_score=ppi_score,
                similarity_score=similarity_score,
            )

            if score < min_score:
                continue

            candidates.append({
                "name": drug_name,
                "drug_name": drug_name,
                "drug_id": drug.get("id", ""),
                "score": score,
                "confidence": evidence.get("confidence", "low"),
                "shared_genes": evidence.get("shared_genes", []),
                "shared_pathways": evidence.get("shared_pathways", []),
                "explanation": evidence.get("explanation", ""),
                "indication": drug.get("indication", ""),
                "mechanism": drug.get("mechanism", ""),
                "gene_score": evidence.get("gene_score", 0.0),
                "pathway_score": evidence.get("pathway_score", 0.0),
                "ppi_score": evidence.get("ppi_score", 0.0),
                "similarity_score": evidence.get("similarity_score", 0.0),
                "mechanism_score": evidence.get("mechanism_score", 0.0),
                "literature_score": evidence.get("literature_score", 0.0),
                "target_quality_score": evidence.get("target_quality_score", 0.0),
                "target_genes": [t.upper() for t in (drug.get("targets") or [])],
                "targets": list(drug.get("targets", []) or []),
                "pathways": list(drug.get("pathways", []) or []),
                "target_source": drug.get("target_source"),
                "pathway_source": drug.get("pathway_source"),
                "target_provenance": list(drug.get("target_provenance", []) or []),
                "score_components": {
                    "gene_score": evidence.get("gene_score", 0.0),
                    "pathway_score": evidence.get("pathway_score", 0.0),
                    "ppi_score": evidence.get("ppi_score", 0.0),
                    "similarity_score": evidence.get("similarity_score", 0.0),
                    "mechanism_score": evidence.get("mechanism_score", 0.0),
                    "literature_score": evidence.get("literature_score", 0.0),
                    "target_quality_score": evidence.get("target_quality_score", 0.0),
                },
                "feature_trace": evidence.get("feature_trace", []),
            })

        candidates.sort(key=lambda c: c["score"], reverse=True)
        return candidates

    # ------------------------------------------------------------------
    # Combination helpers
    # ------------------------------------------------------------------

    def rank_combinations_for_validation(self, candidates: List[Dict], disease_data: Dict) -> List[Dict]:
        return rank_combinations(
            candidates=candidates,
            disease_genes=disease_data.get("genes", []),
            disease_name=disease_data.get("name", ""),
            max_pairs=3000,
            top_n_singles=60,
            include_triples=False,
            min_combo_score=0.0,
        )

    def rank_combinations_for_plan(
        self,
        candidates: List[Dict],
        disease_data: Dict,
        *,
        include_triples: bool = True,
        min_combo_score: float = 0.10,
    ) -> List[Dict]:
        return rank_combinations(
            candidates=candidates,
            disease_genes=disease_data.get("genes", []),
            disease_name=disease_data.get("name", ""),
            max_pairs=5000,
            top_n_singles=60,
            include_triples=include_triples,
            min_combo_score=min_combo_score,
        )

    # ------------------------------------------------------------------
    # Main pipeline  (v5.0 — transcriptomics integrated)
    # ------------------------------------------------------------------

    async def generate_treatment_plan(
        self,
        disease_name: str,
        max_regimens: int = 10,
        include_triples: bool = True,
        fetch_ppi: bool = True,
        fetch_similarity: bool = True,
        use_tissue: bool = True,          # ← NOW FUNCTIONAL (not deleted)
        min_single_score: float = 0.05,
        min_combo_score: float = 0.10,
        use_transcriptomics: Optional[bool] = None,
    ) -> Dict:
        """
        Generate a ranked generic combination therapy plan.

        v5.0 additions
        --------------
        use_tissue=True:   Applies continuous tissue expression gating.
                           Drugs whose targets aren't expressed in the disease
                           tissue receive a score multiplier ∈ [0.30, 1.00].

        use_transcriptomics=True (default when self.enable_transcriptomics):
            1. Extends disease gene set with tissue-anchored expression data.
            2. Applies tissue expression gate to candidate scores.
            3. Rescores combo list using dysregulation pathway repair signal.
            4. Runs per-subtype precision in-silico trials.
        """
        _use_txn = (
            use_transcriptomics
            if use_transcriptomics is not None
            else self.enable_transcriptomics
        ) and use_tissue  # only run transcriptomics when tissue mode is on

        logger.info(
            "Generating treatment plan for %s (v5.0, transcriptomics=%s)",
            disease_name, _use_txn
        )

        disease_data = await self.data_fetcher.fetch_disease_data(disease_name)
        if disease_data is None:
            return {
                "success": False,
                "error": f"Disease not found: {disease_name}",
                "disease": disease_name,
            }

        generic_drugs, excluded, generic_stats = await self.fetch_generic_drugs(limit=3000)

        candidates = await self.generate_candidates(
            disease_data=disease_data,
            drugs_data=generic_drugs,
            min_score=min_single_score,
            fetch_ppi=fetch_ppi,
            fetch_similarity=fetch_similarity,
            use_efo=False,
            use_tissue=use_tissue,
            use_polypharm=False,
        )

        from .drug_filter import DrugSafetyFilter
        safety_filter = DrugSafetyFilter()
        candidates, filtered_out = await safety_filter.filter_candidates(
            candidates=candidates,
            disease_name=disease_name,
            remove_absolute=True,
            remove_relative=True,
        )
        logger.info(
            "Safety filter: %d safe candidates, %d filtered out",
            len(candidates), len(filtered_out)
        )

        # ── v5.0: Transcriptomics enrichment ──────────────────────────────
        txn_context: Dict = {"enabled": False}
        if _use_txn:
            try:
                # Import here to avoid circular imports
                import sys
                import os
                # Try importing from same package first, then from file
                try:
                    from .transcriptomics_engine import enrich_pipeline_with_transcriptomics
                except ImportError:
                    # Fallback: look for it next to this file
                    import importlib.util
                    spec = importlib.util.spec_from_file_location(
                        "transcriptomics_engine",
                        os.path.join(os.path.dirname(__file__), "transcriptomics_engine.py")
                    )
                    txn_mod = importlib.util.load_from_spec(spec)
                    spec.loader.exec_module(txn_mod)
                    enrich_pipeline_with_transcriptomics = txn_mod.enrich_pipeline_with_transcriptomics

                candidates, combos_pre, txn_context = await enrich_pipeline_with_transcriptomics(
                    disease_name=disease_name,
                    disease_data=disease_data,
                    candidates=candidates,
                    combos=[],  # combos not yet ranked; we enrich after
                    enable_tissue_gate=True,
                    enable_pathway_repair=True,
                    enable_subtype_sim=False,  # run after combos are ready
                )

                # Use tissue-gated scores for combo ranking
                # (tissue_gated_score is set by TissueExpressionGate)
                for c in candidates:
                    if "tissue_gated_score" in c:
                        c["_base_score"] = c["score"]  # preserve original
                        c["score"] = c["tissue_gated_score"]  # use gated score for ranking

                logger.info(
                    "Transcriptomics: tissue gate applied, gene set extended by %d genes",
                    txn_context.get("signature_summary", {}).get("n_extended_genes_added", 0)
                )
            except Exception as exc:
                logger.warning(
                    "Transcriptomics enrichment failed (falling back to v4.1): %s", exc
                )
                txn_context = {"enabled": False, "error": str(exc)}
        # ── end transcriptomics ────────────────────────────────────────────

        combos = self.rank_combinations_for_plan(
            candidates=candidates,
            disease_data=disease_data,
            include_triples=include_triples,
            min_combo_score=min_combo_score,
        )

        # ── v5.0: Pathway repair rescore + subtype simulation ──────────────
        if _use_txn and combos and txn_context.get("enabled"):
            try:
                from .transcriptomics_engine import (
                    DysregulationPathwayScorer,
                    TranscriptomicSubtypeSimulator,
                )
                # Rescore combos with pathway repair
                repair_scorer = DysregulationPathwayScorer(disease_name)

                # Reconstruct txn_signature from context for pathway repair
                # (We use the cached version rather than re-fetching)
                from .transcriptomics_engine import TranscriptomicSignature
                txn_sig_engine = TranscriptomicSignature(disease_name)
                txn_signature_cached = txn_sig_engine._load_cache() or {}

                if txn_signature_cached:
                    combos = repair_scorer.rescore_combos(
                        combos=combos,
                        txn_signature=txn_signature_cached,
                        disease_pathways=disease_data.get("pathways", []),
                        repair_weight=0.20,
                    )

                    # Run subtype simulation on top combos
                    subtype_sim = TranscriptomicSubtypeSimulator()
                    subtype_context = subtype_sim.simulate_subtype_trials(
                        combos, txn_signature_cached, n_top_combos=5
                    )
                    txn_context["subtype_simulation"] = subtype_context

            except Exception as exc:
                logger.warning("Post-combo transcriptomics failed: %s", exc)
        # ── end post-combo transcriptomics ─────────────────────────────────

        top_sim_inputs = []
        for combo in combos[:max_regimens * 2]:
            top_sim_inputs.append({
                "candidate_name": combo["combo_name"],
                "drug_name": combo["combo_name"],
                # Prefer txn-rescored score if available
                "score": combo.get("txn_rescored_combo_score", combo.get("combo_score", 0.0)),
                "target_genes": combo.get("shared_genes", []),
                "disease_genes": disease_data.get("genes", []),
                "mechanism_score": max(
                    combo.get("score_breakdown", {}).get("base_score", 0.0),
                    combo.get("combo_score", 0.0) * 0.5,
                ),
            })

        trial_results: List[Dict] = []
        if top_sim_inputs:
            try:
                simulator = InSilicoTrialSimulator(disease=disease_name, n_patients=200)
                trial_results = await simulator.run_batch(top_sim_inputs)
            except Exception as exc:
                logger.warning("Exploratory simulation unavailable: %s", exc)

        assembler = TreatmentPlanAssembler(disease_name=disease_name)
        plan = await assembler.build(
            disease_data=disease_data,
            candidates=candidates,
            combos=combos,
            trial_results=trial_results,
            generic_stats=generic_stats,
            top_n=max_regimens,
        )

        # Restore original scores in candidate list (for display)
        for c in candidates:
            if "_base_score" in c:
                c["score_before_tissue_gate"] = c["_base_score"]
                del c["_base_score"]

        plan["success"] = True
        plan["disease"] = disease_data["name"]
        plan["candidates"] = candidates[:400]
        plan["pipeline_stats"] = {
            "total_drugs_evaluated": len(generic_drugs),
            "excluded_non_generic": len(excluded),
            "n_candidates_scored": len(candidates),
            "n_combos_scored": len(combos),
            "n_exploratory_simulations": len(trial_results),
        }
        plan["transcriptomics_context"] = txn_context
        plan["scoring_policy"] = {
            "version": "v5.0",
            "component_weights": self.component_weights,
            "transcriptomics_enabled": _use_txn,
            "principles": [
                "Tissue expression gate applied as continuous multiplier [0.30, 1.00]",
                "Gene set extended with tissue-anchored expression data",
                "Combo scores blended with dysregulation pathway repair signal (weight=0.20)",
                "Per-subtype precision virtual trials for patient stratification",
                "Graceful degradation: all txn features fall back to v4.1 on API failure",
            ],
        }
        return plan

    async def analyze_disease(
        self,
        disease_name: str,
        min_score: float = 0.05,
        max_results: int = 10,
    ) -> Dict:
        return await self.generate_treatment_plan(
            disease_name=disease_name,
            max_regimens=max_results,
            min_single_score=min_score,
        )

    async def close(self) -> None:
        await self.data_fetcher.close()
        if self._ppi_scorer is not None:
            await self._ppi_scorer.close()


RepurposingPipeline = ProductionPipeline


async def analyze(disease_name: str, min_score: float = 0.05, max_results: int = 10) -> Dict:
    pipeline = ProductionPipeline()
    try:
        return await pipeline.generate_treatment_plan(
            disease_name=disease_name,
            max_regimens=max_results,
            min_single_score=min_score,
        )
    finally:
        await pipeline.close()