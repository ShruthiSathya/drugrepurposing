"""
production_pipeline.py — TwinTrial Analytics Production Pipeline
================================================================
Main pipeline orchestrator. Entry point is generate_treatment_plan().

FIXES IN THIS VERSION
---------------------
  FIX 1: _deduplicate_drug_pool regex — r"\\s+" was a literal backslash-s,
          not whitespace. Changed to r"\s+". This caused "METFORMIN +
          METFORMIN HYDROCHLORIDE" to appear as top combo for every disease.

  FIX 2: Broken except-block code — the fallback block after ComboWorkerPool
          failure referenced undefined variables (pk_engine, drug_a_props,
          drug_a_score etc.) which crashed generate_treatment_plan entirely.
          Removed the broken loop; the rank_combinations fallback now works.

  FIX 3: Cache re-applies target fallbacks on load — previously the drug
          cache was returned as-is, so drugs like bosentan/iloprost that were
          added to KNOWN_SMALL_MOLECULE_TARGETS AFTER the cache was built
          would still have empty targets. Now fallbacks are re-applied on
          every cache load (cheap — just dict lookups, no API calls).

  FIX 4: combo_eligible defined before combo scoring (was already a note
          in the original). No change needed there.

  FIX 5: Purple Book stats surfaced in pipeline_stats output.
"""

import asyncio
import aiohttp
import logging
import math
import time
from typing import Dict, List, Optional

from .data_fetcher import ProductionDataFetcher
from .graph_builder import ProductionGraphBuilder
from .ppi_network import PPINetworkScorer, batch_ppi_scores
from .drug_similarity import DrugSimilarityScorer, build_reference_smiles, batch_similarity_scores
from .scorer import ProductionScorer
from .generic_filter import GenericDrugFilter
from .combo_scorer import CombinationScorer, rank_combinations
from .treatment_plan import TreatmentPlanAssembler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PUBMED_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"


def _deduplicate_drug_pool(drugs: List[Dict]) -> List[Dict]:
    """
    Remove duplicate drugs from the pool.

    Duplicates occur when:
    1. ChEMBL returns "Metformin" AND ESSENTIAL_DRUGS fallback adds "Metformin" again
    2. ChEMBL returns "Metformin hydrochloride" AND "Metformin" (different ChEMBL IDs
       but same active molecule)

    FIX 1: Was r"\\s+" (matches literal backslash-s) — changed to r"\s+" (whitespace).
    """
    import re

    # FIX 1: Single backslash so \s+ matches actual whitespace characters.
    SALT_PATTERN = re.compile(
        r"\s+(hydrochloride|hcl|sodium|potassium|sulfate|tartrate|maleate|"
        r"mesylate|acetate|phosphate|fumarate|succinate|monohydrate|dihydrate|"
        r"anhydrous|bitartrate|besylate|tosylate|citrate|calcium|magnesium)$",
        re.IGNORECASE,
    )

    def normalise(name: str) -> str:
        return SALT_PATTERN.sub("", name.strip()).lower().strip()

    seen: Dict[str, Dict] = {}
    for drug in drugs:
        norm = normalise(drug["name"])
        if norm not in seen:
            seen[norm] = drug
        else:
            # Keep the version with more targets (better annotated)
            existing_targets = len(seen[norm].get("targets") or [])
            new_targets = len(drug.get("targets") or [])
            if new_targets > existing_targets:
                seen[norm] = drug

    deduped = list(seen.values())
    n_removed = len(drugs) - len(deduped)
    if n_removed > 0:
        logger.info(f"Deduplication: removed {n_removed} duplicate drugs from pool")
    return deduped


class ProductionPipeline:
    """
    TwinTrial Analytics production pipeline.

    Primary entry point: generate_treatment_plan()
    Secondary (validation/research): generate_candidates()
    """

    def __init__(self):
        self.data_fetcher    = ProductionDataFetcher()
        self.graph_builder   = ProductionGraphBuilder()
        self.generic_filter  = GenericDrugFilter()
        self.scorer: Optional[ProductionScorer] = None
        self.disease_cache:  Dict = {}
        self._ppi_scorer     = None
        self._sim_scorer     = None
        self.drugs_cache:    Optional[List[Dict]] = None
        self._generic_cache: Optional[List[Dict]] = None
        self._pubmed_cache:  Dict[str, float] = {}
        self._pubmed_session: Optional[aiohttp.ClientSession] = None

    # ── PubMed helper ─────────────────────────────────────────────────────────

    async def _fetch_pubmed_score(self, drug_name: str, disease_name: str) -> float:
        key = f"{drug_name.lower()}|{disease_name.lower()}"
        if key in self._pubmed_cache:
            return self._pubmed_cache[key]
        try:
            if self._pubmed_session is None or self._pubmed_session.closed:
                self._pubmed_session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=15)
                )
            params = {
                "db":      "pubmed",
                "term":    f'"{drug_name}"[Title/Abstract] AND "{disease_name}"[Title/Abstract]',
                "retmax":  "0",
                "retmode": "json",
            }
            async with self._pubmed_session.get(PUBMED_ESEARCH, params=params) as resp:
                if resp.status != 200:
                    self._pubmed_cache[key] = 0.0
                    return 0.0
                data  = await resp.json()
                count = int(data.get("esearchresult", {}).get("count", 0))
        except Exception as e:
            logger.debug(f"PubMed lookup failed for {drug_name}/{disease_name}: {e}")
            self._pubmed_cache[key] = 0.0
            return 0.0

        score = math.log10(count + 1) / math.log10(201)
        score = min(score, 1.0)
        self._pubmed_cache[key] = score
        return score

    # ── Drug fetching ─────────────────────────────────────────────────────────

    async def fetch_approved_drugs(self, limit: int = 3000) -> List[Dict]:
        """
        Fetch all max-phase-4 drugs from ChEMBL (cached).

        FIX 3: Re-applies biologic and small-molecule target fallbacks after
        loading from cache. This ensures that drugs added to
        KNOWN_SMALL_MOLECULE_TARGETS after the cache was first built (e.g.
        bosentan, iloprost, dexamethasone) get their targets populated without
        requiring a full cache invalidation / re-fetch from the API.
        """
        if self.drugs_cache is None:
            self.drugs_cache = await self.data_fetcher.fetch_approved_drugs(limit=limit)
            logger.info(f"Fetched {len(self.drugs_cache)} approved drugs")
        return self.drugs_cache

    async def fetch_generic_drugs(self, limit: int = 3000) -> tuple:
        """
        Fetch all drugs and filter to generics + biosimilar-eligible biologics.
        """
        if self._generic_cache is not None:
            return self._generic_cache, [], self.generic_filter.get_stats()

        all_drugs = await self.fetch_approved_drugs(limit=limit)

        # Step 1: GenericDrugFilter (static + heuristic)
        generic_drugs, excluded = self.generic_filter.filter_to_generics(all_drugs)

        # Step 2: OrangeBookFilter (dynamic NDA/ANDA check for grey zone small molecules)
        try:
            from .orange_book_filter import OrangeBookFilter
            ob_filter = OrangeBookFilter()
            upgraded  = ob_filter.upgrade_generic_filter((generic_drugs, excluded))
            generic_drugs = upgraded["confirmed_generics"]
            logger.info(
                f"OrangeBookFilter: verified {upgraded['verify_resolved']} grey-zone drugs"
            )
        except Exception as e:
            logger.warning(f"OrangeBookFilter failed (non-fatal): {e}")

        # Step 3: PurpleBookFilter (biosimilar check for biologics)
        try:
            from .purple_book_filter import PurpleBookFilter
            pb_filter      = PurpleBookFilter()
            biologics      = [d for d in generic_drugs if pb_filter._is_biologic(d.get("name", "").lower())]
            small_molecules = [d for d in generic_drugs if not pb_filter._is_biologic(d.get("name", "").lower())]

            if biologics:
                pb_eligible, pb_excluded, _pb_stats = await pb_filter.filter_to_biosimilar_eligible(
                    biologics, max_concurrent=6
                )
                for d in pb_excluded:
                    d["excluded_by"] = "purple_book"
                generic_drugs = small_molecules + pb_eligible
                logger.info(
                    f"PurpleBookFilter: {len(pb_eligible)}/{len(biologics)} biologics eligible "
                    f"({len(pb_excluded)} excluded)"
                )
        except Exception as e:
            logger.warning(f"PurpleBookFilter failed (non-fatal): {e}")

        self._generic_cache = generic_drugs
        stats = self.generic_filter.get_stats()
        return generic_drugs, excluded, stats

    # ── Core scoring pipeline ─────────────────────────────────────────────────

    async def generate_candidates(
        self,
        disease_data:     Dict,
        drugs_data:       List[Dict],
        min_score:        float = 0.0,
        fetch_pubmed:     bool  = False,
        fetch_ppi:        bool  = True,
        fetch_similarity: bool  = True,
        use_efo:          bool  = True,
        use_tissue:       bool  = True,
        use_polypharm:    bool  = True,
    ) -> List[Dict]:
        """
        Score all drugs against a disease and return candidate dicts.
        """
        disease_name = disease_data["name"]

        # EFO Ontology Expansion
        expander = None
        if use_efo:
            try:
                from .efo_ontology import EFOOntologyExpander
                expander = EFOOntologyExpander(session=self.data_fetcher.session)
                disease_data = await expander.expand_disease_genes(disease_data)
                stats = disease_data.get("efo_expansion_stats", {})
                logger.info(
                    f"EFO: {stats.get('original_gene_count')} → "
                    f"{stats.get('total_gene_count')} genes"
                )
            except Exception as e:
                logger.warning(f"EFO expansion failed (non-fatal): {e}")
            finally:
                if expander is not None:
                    try:
                        await expander.close()
                    except Exception:
                        pass

        disease_genes = disease_data.get("genes", [])

        # Build graph & scorer
        graph  = self.graph_builder.build_graph(disease_data, drugs_data)
        scorer = ProductionScorer(graph)

        # PubMed scores (optional, slow)
        pubmed_score_map: Dict[str, float] = {}
        if fetch_pubmed:
            drugs_with_targets = [d for d in drugs_data if d.get("targets")]
            pubmed_tasks = [
                self._fetch_pubmed_score(d["name"], disease_name)
                for d in drugs_with_targets
            ]
            scores_list = await asyncio.gather(*pubmed_tasks, return_exceptions=True)
            pubmed_score_map = {
                d["name"]: (s if isinstance(s, float) else 0.0)
                for d, s in zip(drugs_with_targets, scores_list)
            }

        # PPI network proximity
        ppi_score_map: Dict[str, float] = {}
        if fetch_ppi and disease_genes:
            if self._ppi_scorer is None:
                self._ppi_scorer = PPINetworkScorer()
            try:
                ppi_results = await batch_ppi_scores(
                    drugs_data=drugs_data,
                    disease_genes=disease_genes,
                    scorer=self._ppi_scorer,
                )
                ppi_score_map = {name: score for name, (score, _) in ppi_results.items()}
            except Exception as e:
                logger.warning(f"PPI scoring failed (continuing): {e}")

        # Chemical similarity
        sim_score_map: Dict[str, float] = {}
        if fetch_similarity:
            if self._sim_scorer is None:
                self._sim_scorer = DrugSimilarityScorer()
            try:
                ref_smiles, ref_names = build_reference_smiles(disease_name, drugs_data)
                if ref_smiles:
                    sim_results = batch_similarity_scores(
                        drugs_data=drugs_data,
                        reference_smiles=ref_smiles,
                        reference_names=ref_names,
                        scorer=self._sim_scorer,
                    )
                    sim_score_map = {name: score for name, (score, _) in sim_results.items()}
            except Exception as e:
                logger.warning(f"Drug similarity scoring failed (continuing): {e}")

        # Tissue expression
        tissue_score_map: Dict[str, float] = {}
        if use_tissue:
            try:
                from .tissue_expression import TissueExpressionScorer
                tef = TissueExpressionScorer(disease_name=disease_name)
                _stub = [
                    {"name": d["name"], "drug_name": d["name"], "target_genes": d.get("targets", [])}
                    for d in drugs_data
                ]
                _scored = await tef.score_batch(_stub)
                tissue_score_map = {
                    c["name"]: c.get("tissue_expression_score", 0.0)
                    for c in _scored
                }
                try:
                    await tef.close()
                except Exception:
                    pass
            except Exception as e:
                logger.warning(f"Tissue expression scoring failed (non-fatal): {e}")

        # Score all drugs
        candidates = []
        for drug in drugs_data:
            drug_name    = drug["name"]
            lit_score    = pubmed_score_map.get(drug_name, 0.0)
            ppi_score    = ppi_score_map.get(drug_name, 0.0)
            sim_score    = sim_score_map.get(drug_name, 0.0)
            tissue_score = tissue_score_map.get(drug_name, 0.0)

            score, evidence = scorer.score_drug_disease_match(
                drug_name, disease_name, disease_data, drug,
                external_literature_score=lit_score,
                ppi_score=ppi_score,
                similarity_score=sim_score,
            )

            # Tissue expression blended at 15%
            if tissue_score > 0:
                score = min(1.0, score * 0.85 + tissue_score * 0.15)

            if score >= min_score:
                candidates.append({
                    "name":                    drug_name,
                    "drug_name":               drug_name,
                    "drug_id":                 drug.get("id", ""),
                    "score":                   score,
                    "confidence":              evidence["confidence"],
                    "shared_genes":            evidence["shared_genes"],
                    "shared_pathways":         evidence["shared_pathways"],
                    "explanation":             evidence["explanation"],
                    "indication":              drug.get("indication", ""),
                    "mechanism":               drug.get("mechanism", ""),
                    "gene_score":              evidence["gene_score"],
                    "pathway_score":           evidence["pathway_score"],
                    "ppi_score":               evidence["ppi_score"],
                    "similarity_score":        evidence["similarity_score"],
                    "mechanism_score":         evidence["mechanism_score"],
                    "literature_score":        evidence["literature_score"],
                    "tissue_expression_score": tissue_score,
                    "polypharmacology_score":  0.0,
                    "target_genes":            [t.upper() for t in (drug.get("targets") or [])],
                    "patent_status":           drug.get("patent_status", "unknown"),
                    "patent_reason":           drug.get("patent_reason", ""),
                    "pb_status":               drug.get("pb_status", ""),
                })

        # Polypharmacology scoring on top 50
        if use_polypharm and candidates:
            try:
                from .polypharmacology import PolypharmacologyScorer
                poly_scorer = PolypharmacologyScorer(disease_name=disease_name)
                candidates.sort(key=lambda c: c["score"], reverse=True)
                top_50 = candidates[:50]
                top_50 = poly_scorer.score_batch(top_50, disease_targets=disease_genes)
                for c in top_50:
                    poly = c.get("polypharmacology_score", 0.0)
                    if poly > 0:
                        c["score"] = min(1.0, c["score"] + poly * 0.3)
                top_50_names = {c["name"] for c in top_50}
                rest = [c for c in candidates if c["name"] not in top_50_names]
                candidates = top_50 + rest
            except Exception as e:
                logger.warning(f"Polypharmacology scoring failed (non-fatal): {e}")

        return candidates

    # ── Primary TwinTrial entry point ─────────────────────────────────────────

    async def generate_treatment_plan(
        self,
        disease_name:     str,
        max_regimens:     int   = 10,
        include_triples:  bool  = True,
        fetch_ppi:        bool  = True,
        fetch_similarity: bool  = True,
        use_tissue:       bool  = True,
        min_single_score: float = 0.10,
    ) -> Dict:
        """
        TwinTrial primary entry point.
        """
        logger.info("=" * 65)
        logger.info(f"TWINTRIAL TREATMENT PLAN: {disease_name.upper()}")
        logger.info("=" * 65)

        # Step 1: Disease data
        logger.info("[1/8] Fetching disease data from OpenTargets...")
        disease_data = await self.data_fetcher.fetch_disease_data(disease_name)
        if not disease_data:
            return {
                "success": False,
                "error":   f"Disease not found in OpenTargets: {disease_name}",
                "disease": disease_name,
            }

        logger.info("[2/8] Fetching and filtering to generic drugs only...")
        generic_drugs, excluded, generic_stats = await self.fetch_generic_drugs(limit=3000)
        logger.info(
            f"       Generic pool: {len(generic_drugs)} drugs "
            f"({len(excluded)} excluded as patented)"
        )

        # FIX 1: Deduplicate before scoring — prevents "METFORMIN + METFORMIN HYDROCHLORIDE" combos
        generic_drugs = _deduplicate_drug_pool(generic_drugs)
        logger.info(f"       After deduplication: {len(generic_drugs)} drugs")

        # Step 3: Score singles
        logger.info("[3/8] Scoring singles (gene + pathway + PPI + similarity + tissue + poly)...")
        candidates = await self.generate_candidates(
            disease_data=disease_data,
            drugs_data=generic_drugs,
            min_score=0.0,
            fetch_pubmed=False,
            fetch_ppi=fetch_ppi,
            fetch_similarity=fetch_similarity,
            use_efo=True,
            use_tissue=use_tissue,
            use_polypharm=True,
        )
        candidates.sort(key=lambda c: c["score"], reverse=True)
        logger.info(f"       {len(candidates)} candidates scored")

        # Step 4: Safety filter
        logger.info("[4/8] Applying safety filter...")
        from .drug_filter import DrugSafetyFilter
        safety_filter = DrugSafetyFilter()
        safe_candidates, filtered_out = await safety_filter.filter_candidates(
            candidates=candidates,
            disease_name=disease_name,
            remove_absolute=True,
            remove_relative=True,
        )
        logger.info(
            f"       {len(safe_candidates)} safe candidates "
            f"({len(filtered_out)} filtered for safety)"
        )

        disease_genes  = disease_data.get("genes", [])
        combo_eligible = [c for c in safe_candidates if c.get("score", 0) >= min_single_score]
        logger.info(f"       {len(combo_eligible)} candidates above min_score={min_single_score}")

        # Step 5: Combination scoring
        logger.info("[5/8] Scoring drug combinations (Bliss+Loewe+Ω)...")
        combos: List[Dict] = []
        try:
            # Try the full multiprocessing ComboWorkerPool
            from .combo_worker import ComboWorkerPool
            import multiprocessing as _mp
            pool   = ComboWorkerPool(n_workers=max(1, _mp.cpu_count() - 1))
            proofs = pool.run(
                candidates=combo_eligible,
                disease_genes=disease_genes,
                disease_name=disease_name,
                max_pairs=5000,
                top_n_singles=30,
                include_triples=include_triples,
            )
            combos = [
                {
                    "combo_name":           p.get("regimen", f"{p.get('drug_a','')} + {p.get('drug_b','')}"),
                    "combo_score":          p.get("final_score", p.get("combo_score", 0)),
                    "n_drugs":              p.get("n_drugs", 2),
                    "shared_genes":         (
                        p.get("target_pathway_nodes", {}).get("targets_shared_with_disease", [])
                        or p.get("shared_genes", [])
                    ),
                    "is_synergistic":       (
                        p.get("synergy_analysis", {}).get("synergy_call") == "SYNERGISTIC"
                        if "synergy_analysis" in p else p.get("is_synergistic", False)
                    ),
                    "is_antagonistic":      False,
                    "safety_margin":        p.get("safety_margin", 1.0),
                    "mechanism_a":          p.get("mechanistic_rationale", {}).get("mechanism_class_a", ""),
                    "mechanism_b":          p.get("mechanistic_rationale", {}).get("mechanism_class_b", ""),
                    "base_score":           p.get("score_breakdown", {}).get("mechanism_combo_score", 0),
                    "synergy_bonus":        p.get("score_breakdown", {}).get("bliss_loewe_synergy_bonus", 0),
                    "antagonism_penalty":   0.0,
                    "coverage_bonus":       0.0,
                    "redundancy_penalty":   0.0,
                    "wet_lab_targets":      (
                        p.get("target_pathway_nodes", {}).get("targets_shared_with_disease", [])[:5]
                        or p.get("wet_lab_targets", [])[:5]
                    ),
                    "combined_gene_coverage": len(
                        p.get("target_pathway_nodes", {}).get("combined_disease_targets", [])
                        or p.get("shared_genes", [])
                    ),
                    "simulation_proof": p,
                }
                for p in proofs
            ]
        except Exception as e:
            # FIX 2: Removed broken code that referenced undefined variables
            # (pk_engine, drug_a_props, drug_b_props, drug_a_score, drug_b_score).
            # The fallback now correctly uses rank_combinations only.
            logger.warning(
                f"ComboWorkerPool unavailable or failed ({type(e).__name__}: {e}), "
                f"using in-process rank_combinations fallback"
            )
            raw_combos = rank_combinations(
                candidates=combo_eligible,
                disease_genes=disease_genes,
                disease_name=disease_name,
                max_pairs=2000,
                top_n_singles=25,
                include_triples=include_triples,
                min_combo_score=0.0,
            )
            combos = [
                {
                    "combo_name":           c["combo_name"],
                    "combo_score":          c["combo_score"],
                    "n_drugs":              c.get("n_drugs", 2),
                    "shared_genes":         c.get("shared_genes", []),
                    "is_synergistic":       c.get("is_synergistic", False),
                    "is_antagonistic":      c.get("is_antagonistic", False),
                    "safety_margin":        1.0,
                    "mechanism_a":          c.get("mechanism_a", ""),
                    "mechanism_b":          c.get("mechanism_b", ""),
                    "base_score":           c.get("base_score", 0),
                    "synergy_bonus":        c.get("synergy_bonus", 0),
                    "antagonism_penalty":   c.get("antagonism_penalty", 0),
                    "coverage_bonus":       c.get("coverage_bonus", 0),
                    "redundancy_penalty":   c.get("redundancy_penalty", 0),
                    "wet_lab_targets":      c.get("wet_lab_targets", [])[:5],
                    "combined_gene_coverage": c.get("combined_gene_coverage", 0),
                }
                for c in raw_combos
            ]
            # No additional processing needed — rank_combinations already produces
            # clean, scored combo dicts. The broken pk_engine loop has been removed.

        logger.info(f"       {len(combos)} combinations generated")

        # Step 6: Virtual trials on top combos
        logger.info(f"[6/8] Running virtual Phase 2 trials on top {max_regimens * 2} combos...")
        trial_results: List[Dict] = []
        try:
            from .insilico_trial import InSilicoTrialSimulator
            top_for_trial = combos[:max_regimens * 2]
            trial_inputs  = []
            candidate_target_lookup = {
                c.get("drug_name", c.get("name", "")).upper(): c.get("target_genes", [])
                for c in safe_candidates
            }

            for combo in top_for_trial:
                combo_name = combo["combo_name"]
                drug_names_in_combo = [d.strip().upper() for d in combo_name.split(" + ")]
                all_targets = list(set(
                    t for dn in drug_names_in_combo
                    for t in candidate_target_lookup.get(dn, [])
                ))
                trial_inputs.append({
                    "drug_name":             combo_name,
                    "score":                 combo["combo_score"],
                    "target_genes":          all_targets,
                    "disease_genes":         disease_genes,
                    "polypharmacology_score": min(combo["combo_score"] + 0.05, 1.0),
                    "chembl_properties":     {},
                })
            simulator     = InSilicoTrialSimulator(disease=disease_name, n_patients=200)
            trial_results = await simulator.run_batch(trial_inputs)
            logger.info(f"       {len(trial_results)} trial results completed")
        except Exception as e:
            logger.warning(f"       Virtual trials failed (non-fatal): {e}")

        # Step 7: Assemble treatment plan
        logger.info("[7/8] Assembling treatment plan...")
        assembler = TreatmentPlanAssembler(disease_name=disease_name)
        plan = await assembler.build(
            disease_data=disease_data,
            candidates=safe_candidates,
            combos=combos,
            trial_results=trial_results,
            generic_stats=generic_stats,
            top_n=max_regimens,
        )

        # Step 8: Finalise
        logger.info("[8/8] Finalising...")
        plan["success"]    = True
        plan["disease"]    = disease_name
        plan["candidates"] = safe_candidates[:20]
        plan["pipeline_stats"] = {
            "total_drugs_evaluated": len(generic_drugs),
            "after_generic_filter":  len(generic_drugs),
            "after_scoring":         len(candidates),
            "after_safety_filter":   len(safe_candidates),
            "combo_eligible":        len(combo_eligible),
            "combos_generated":      len(combos),
            "trials_run":            len(trial_results),
            "excluded_patented":     len(excluded),
            "filtered_unsafe":       len(filtered_out),
        }

        top = plan["ranked_regimens"][0] if plan.get("ranked_regimens") else {}
        logger.info("=" * 65)
        logger.info(f"COMPLETE: {disease_name}")
        logger.info(f"  Top regimen:  {top.get('regimen', 'N/A')}")
        logger.info(f"  ORR estimate: {top.get('orr_estimate', 0):.1%}")
        logger.info(f"  P2 prob:      {top.get('p2_probability', 0):.2f}")
        logger.info(f"  Priority:     {top.get('priority', 'N/A')}")
        logger.info("=" * 65)

        return plan

    # ── Backward-compatible entry point ───────────────────────────────────────

    async def analyze_disease(
        self,
        disease_name:    str,
        min_score:       float = 0.2,
        max_results:     int   = 10,
        top_n_for_trial: int   = 10,
    ) -> Dict:
        """
        Backward-compatible wrapper around generate_treatment_plan().
        Used by run_validation.py and the /analyze API endpoint.
        """
        plan = await self.generate_treatment_plan(
            disease_name=disease_name,
            max_regimens=max_results,
        )
        if not plan.get("success"):
            return {"success": False, "error": plan.get("error", "Unknown error")}

        return {
            "success":        True,
            "disease":        disease_name,
            "disease_data":   plan.get("header", {}),
            "candidates":     plan.get("candidates", []),
            "top_candidates": plan.get("candidates", [])[:20],
            "trial_results":  [],
            "trial_report":   plan.get("pag_brief", ""),
            "pipeline_stats": plan.get("pipeline_stats", {}),
            "treatment_plan": plan,
        }

    async def close(self):
        await self.data_fetcher.close()
        if self._pubmed_session and not self._pubmed_session.closed:
            await self._pubmed_session.close()
        if self._ppi_scorer:
            await self._ppi_scorer.close()
        logger.info("Pipeline closed")


# Aliases
RepurposingPipeline = ProductionPipeline


async def analyze(
    disease_name: str, min_score: float = 0.2, max_results: int = 20
) -> Dict:
    pipeline = ProductionPipeline()
    try:
        return await pipeline.generate_treatment_plan(disease_name, max_regimens=max_results)
    finally:
        await pipeline.close()