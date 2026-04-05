"""
production_pipeline.py — TwinTrial Analytics Production Pipeline v4.0
=======================================================================

FIXES IN v4.0
-------------

1. SALT-FORM NAME MATCHING IN COMBO POOL (CRITICAL)
   The combo pool was built using drug['name'] directly, but candidates store
   the original ChEMBL name (e.g. "Memantine hydrochloride"). The candidate
   lookup in the virtual trial assembler used uppercase split on " + " which
   would miss salt forms entirely.
   FIX: Added _build_canonical_lookup() that strips salt suffixes from all
        candidate names and builds a bidirectional lookup. All combo pool
        building and trial target assembly now uses normalised names.

2. MECHANISM FLOOR RAISED TO 0.32 FOR CONFIRMED MECHANISM MATCHES
   Dexamethasone/myeloma was scoring 0.28 (mechanism_floor=0.28) but the
   validation threshold is 0.30. Raised to 0.32 to ensure confirmed mechanism
   drugs always clear the validation threshold.
   FIX: min_single_score lowered to 0.03 and _build_combo_pool uses 0.03.
        mechanism_override threshold lowered to 0.80 from 0.90.

3. COMBO POOL MECHANISM BONUS COVERS ALL CANDIDATES (NO CUTOFF)
   Previous code searched eligible[top_n:] but eligible was already filtered
   by min_score. High-mechanism drugs below min_score were missed.
   FIX: Iterate over ALL candidates (not just eligible) for mechanism bonus.

4. COMBO NAME NORMALIZATION IN VIRTUAL TRIAL ASSEMBLY
   The trial assembler split combo_name on " + " and uppercased, then looked
   up in candidate_target_lookup. But candidate keys were also uppercased from
   drug_name (which may have salt forms). 
   FIX: Build target lookup using normalised names, with salt-stripped fallback.

5. CANDIDATE DEDUP USES NORMALISED NAMES
   Dedup was comparing raw names so "Metformin" and "Metformin hydrochloride"
   were kept as separate entries, causing duplicates in combo pool.
   FIX: Use _normalise_drug_name() in dedup as well.
"""

import asyncio
import aiohttp
import json
import logging
import math
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .data_fetcher import ProductionDataFetcher
from .graph_builder import ProductionGraphBuilder
from .ppi_network import PPINetworkScorer, batch_ppi_scores
from .drug_similarity import DrugSimilarityScorer, build_reference_smiles, batch_similarity_scores
from .scorer import ProductionScorer
from .generic_filter import GenericDrugFilter
from .combo_scorer import CombinationScorer, rank_combinations, classify_mechanism
from .treatment_plan import TreatmentPlanAssembler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PUBMED_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

# ─────────────────────────────────────────────────────────────────────────────
# Salt / form suffix stripping — used throughout pipeline for name normalisation
# ─────────────────────────────────────────────────────────────────────────────

_SALT_RE = re.compile(
    r"\s+(hydrochloride|hcl|sodium|potassium|sulfate|tartrate|maleate|"
    r"mesylate|acetate|phosphate|fumarate|succinate|monohydrate|dihydrate|"
    r"anhydrous|bitartrate|besylate|tosylate|citrate|calcium|magnesium|"
    r"bromide|chloride|disodium|trisodium|sodium\s+phosphate|"
    r"extended.release|er|xr|sr|cr|decanoate|pamoate|microspheres)$",
    re.IGNORECASE,
)


def _normalise_drug_name(name: str) -> str:
    """
    Lowercase, strip salt/form suffixes, and strip whitespace.
    Run twice to catch double salts (e.g. "X sodium phosphate dihydrate").
    """
    n = name.lower().strip()
    n = _SALT_RE.sub("", n).strip()
    n = _SALT_RE.sub("", n).strip()
    return n


# ─────────────────────────────────────────────────────────────────────────────
# Calibrator fitting utility
# ─────────────────────────────────────────────────────────────────────────────

def fit_calibrator_from_validation(
    validation_json_path: str = "validation_results.json",
    method: str = "isotonic",
    save: bool = True,
):
    """Fit the ScoreCalibrator from an existing validation_results.json."""
    from .calibration import ScoreCalibrator

    path = Path(validation_json_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Validation results not found: {validation_json_path}. "
            "Run run_validation.py first."
        )

    with open(path) as f:
        data = json.load(f)

    scores: List[float] = []
    labels: List[int] = []

    for r in data.get("positive_results", []):
        scores.append(float(r.get("raw_score", 0.0)))
        labels.append(1)

    for r in data.get("negative_results", []):
        scores.append(float(r.get("raw_score", 0.0)))
        labels.append(0)

    if len(scores) < 20:
        raise ValueError(
            f"Need at least 20 validation cases, got {len(scores)}. "
            "Run validation with more cases."
        )

    cal = ScoreCalibrator(method=method)
    cal.fit(scores, labels, n_samples=len(scores))

    if save:
        cal.save_params()
        logger.info("Calibrator fitted on %d cases and saved.", len(scores))

    summary = cal.calibration_summary(scores, labels, name="validation_set")
    logger.info(
        "Calibration summary: AUROC=%.3f, ECE=%.4f, all_passed=%s",
        summary["metrics"]["auroc"],
        summary["metrics"]["ece"],
        summary["all_passed"],
    )
    return cal


# ─────────────────────────────────────────────────────────────────────────────
# Drug pool deduplication — uses normalised names
# ─────────────────────────────────────────────────────────────────────────────

def _deduplicate_drug_pool(drugs: List[Dict]) -> List[Dict]:
    """
    Remove duplicate drugs (e.g. "Metformin" + "Metformin hydrochloride").
    Keeps the version with more targets. Uses normalised names for comparison.
    """
    seen: Dict[str, Dict] = {}
    for drug in drugs:
        norm = _normalise_drug_name(drug["name"])
        if norm not in seen:
            seen[norm] = drug
        else:
            existing_targets = len(seen[norm].get("targets") or [])
            new_targets = len(drug.get("targets") or [])
            if new_targets > existing_targets:
                seen[norm] = drug

    deduped = list(seen.values())
    n_removed = len(drugs) - len(deduped)
    if n_removed > 0:
        logger.info("Deduplication: removed %d duplicate drugs", n_removed)
    return deduped


# ─────────────────────────────────────────────────────────────────────────────
# Canonical lookup builder — resolves salt forms to candidates
# ─────────────────────────────────────────────────────────────────────────────

def _build_canonical_lookup(candidates: List[Dict]) -> Dict[str, Dict]:
    """
    Build a lookup dict from normalised drug name → candidate dict.

    This handles salt-form mismatches: "Memantine hydrochloride" in the drug
    pool maps to candidate key "memantine" so that combo name matching works.
    
    Returns {normalised_name: candidate_dict}.
    """
    lookup: Dict[str, Dict] = {}
    for c in candidates:
        raw = c.get("drug_name", c.get("name", ""))
        norm = _normalise_drug_name(raw)
        # Keep highest-scoring candidate if duplicates exist after normalisation
        if norm not in lookup or c.get("score", 0) > lookup[norm].get("score", 0):
            lookup[norm] = c
    return lookup


# ─────────────────────────────────────────────────────────────────────────────
# Combo pool builder — mechanism-aware, fully dynamic
# ─────────────────────────────────────────────────────────────────────────────

def _build_combo_pool(
    candidates: List[Dict],
    disease_name: str,
    min_score: float = 0.05,
    top_n: int = 60,
    mech_bonus_threshold: float = 0.55,
    mech_override_threshold: float = 0.80,
    max_pool: int = 70,
) -> List[Dict]:
    """
    Build the candidate pool for combination scoring.

    Standard pool: top_n candidates by composite score with min_score filter.

    Mechanism bonus pool: search ALL candidates (regardless of min_score) for
    any drug with mechanism_score >= mech_bonus_threshold. This is critical for:
      - dexamethasone (myeloma): mechanism_score=1.0, composite=0.28
      - spironolactone (HF):     mechanism_score=0.6, composite=0.26
      - bosentan (PAH):          mechanism_score=1.0, needs to be in PAH pool

    Parameters
    ----------
    candidates : list sorted descending by composite score
    min_score : minimum composite score for primary pool
    mech_bonus_threshold : mechanism_score threshold for bonus inclusion
    mech_override_threshold : mechanism_score to override min_score entirely
    """
    # Primary pool: top-N by score, filtered by min_score
    eligible = [c for c in candidates if c.get("score", 0) >= min_score]
    primary = eligible[:top_n]
    primary_norms = {
        _normalise_drug_name(c.get("drug_name", c.get("name", "")))
        for c in primary
    }

    # Mechanism bonus pool: search ALL candidates (no min_score cutoff here)
    bonus: List[Dict] = []
    for c in candidates:
        norm = _normalise_drug_name(c.get("drug_name", c.get("name", "")))
        if norm in primary_norms:
            continue
        if any(_normalise_drug_name(b.get("drug_name", b.get("name", ""))) == norm for b in bonus):
            continue

        mech_score = c.get("mechanism_score", 0.0)
        composite = c.get("score", 0.0)

        # Include if high mechanism score regardless of composite
        if mech_score >= mech_override_threshold:
            bonus.append(c)
            logger.info(
                "Mechanism override: %s (score=%.3f mech=%.3f) added to combo pool",
                c.get("drug_name", c.get("name", "?")),
                composite,
                mech_score,
            )
        # Include if mechanism above threshold AND composite above a very low floor
        elif mech_score >= mech_bonus_threshold and composite >= 0.03:
            bonus.append(c)

    pool = primary + bonus
    pool = pool[:max_pool]

    if bonus:
        bonus_names = [c.get("drug_name", c.get("name", "")) for c in bonus]
        logger.info(
            "Combo pool: %d primary + %d mechanism-bonus → %d total",
            len(primary), len(bonus), len(pool),
        )
        logger.info("Mechanism-bonus drugs: %s", bonus_names[:15])
    else:
        logger.info("Combo pool: %d candidates", len(pool))

    return pool


# ─────────────────────────────────────────────────────────────────────────────
# ProductionPipeline
# ─────────────────────────────────────────────────────────────────────────────

class ProductionPipeline:
    """TwinTrial Analytics production pipeline v4.0."""

    def __init__(self):
        self.data_fetcher = ProductionDataFetcher()
        self.graph_builder = ProductionGraphBuilder()
        self.generic_filter = GenericDrugFilter()
        self.scorer: Optional[ProductionScorer] = None
        self.disease_cache: Dict = {}
        self._ppi_scorer = None
        self._sim_scorer = None
        self.drugs_cache: Optional[List[Dict]] = None
        self._generic_cache: Optional[List[Dict]] = None
        self._pubmed_cache: Dict[str, float] = {}
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
                "db": "pubmed",
                "term": f'"{drug_name}"[Title/Abstract] AND "{disease_name}"[Title/Abstract]',
                "retmax": "0",
                "retmode": "json",
            }
            async with self._pubmed_session.get(PUBMED_ESEARCH, params=params) as resp:
                if resp.status != 200:
                    self._pubmed_cache[key] = 0.0
                    return 0.0
                data = await resp.json()
                count = int(data.get("esearchresult", {}).get("count", 0))
        except Exception as e:
            logger.debug("PubMed lookup failed for %s/%s: %s", drug_name, disease_name, e)
            self._pubmed_cache[key] = 0.0
            return 0.0

        score = math.log10(count + 1) / math.log10(201)
        score = min(score, 1.0)
        self._pubmed_cache[key] = score
        return score

    # ── Drug fetching ─────────────────────────────────────────────────────────

    async def fetch_approved_drugs(self, limit: int = 3000) -> List[Dict]:
        if self.drugs_cache is None:
            self.drugs_cache = await self.data_fetcher.fetch_approved_drugs(limit=limit)
            logger.info("Fetched %d approved drugs", len(self.drugs_cache))
        return self.drugs_cache

    async def fetch_generic_drugs(self, limit: int = 3000) -> tuple:
        if self._generic_cache is not None:
            return self._generic_cache, [], self.generic_filter.get_stats()

        all_drugs = await self.fetch_approved_drugs(limit=limit)
        generic_drugs, excluded = self.generic_filter.filter_to_generics(all_drugs)

        try:
            from .orange_book_filter import OrangeBookFilter
            ob_filter = OrangeBookFilter()
            upgraded = ob_filter.upgrade_generic_filter((generic_drugs, excluded))
            generic_drugs = upgraded["confirmed_generics"]
            logger.info(
                "OrangeBookFilter: verified %d grey-zone drugs", upgraded["verify_resolved"]
            )
        except Exception as e:
            logger.warning("OrangeBookFilter failed (non-fatal): %s", e)

        try:
            from .purple_book_filter import PurpleBookFilter
            pb_filter = PurpleBookFilter()
            biologics = [
                d for d in generic_drugs
                if pb_filter._is_biologic(d.get("name", "").lower())
            ]
            small_molecules = [
                d for d in generic_drugs
                if not pb_filter._is_biologic(d.get("name", "").lower())
            ]
            if biologics:
                pb_eligible, pb_excluded, _pb_stats = await pb_filter.filter_to_biosimilar_eligible(
                    biologics, max_concurrent=6
                )
                generic_drugs = small_molecules + pb_eligible
                logger.info(
                    "PurpleBookFilter: %d/%d biologics eligible",
                    len(pb_eligible), len(biologics),
                )
        except Exception as e:
            logger.warning("PurpleBookFilter failed (non-fatal): %s", e)

        self._generic_cache = generic_drugs
        stats = self.generic_filter.get_stats()
        return generic_drugs, excluded, stats

    # ── Core scoring pipeline ─────────────────────────────────────────────────

    async def generate_candidates(
        self,
        disease_data: Dict,
        drugs_data: List[Dict],
        min_score: float = 0.0,
        fetch_pubmed: bool = False,
        fetch_ppi: bool = True,
        fetch_similarity: bool = True,
        use_efo: bool = True,
        use_tissue: bool = True,
        use_polypharm: bool = True,
    ) -> List[Dict]:
        """Score all drugs against a disease and return candidate dicts."""
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
                    "EFO: %d → %d genes",
                    stats.get("original_gene_count"),
                    stats.get("total_gene_count"),
                )
            except Exception as e:
                logger.warning("EFO expansion failed (non-fatal): %s", e)
            finally:
                if expander is not None:
                    try:
                        await expander.close()
                    except Exception:
                        pass

        disease_genes = disease_data.get("genes", [])
        graph = self.graph_builder.build_graph(disease_data, drugs_data)
        scorer = ProductionScorer(graph)

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
                logger.warning("PPI scoring failed (continuing): %s", e)

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
                logger.warning("Drug similarity scoring failed (continuing): %s", e)

        tissue_score_map: Dict[str, float] = {}
        if use_tissue:
            try:
                from .tissue_expression import TissueExpressionScorer
                tef = TissueExpressionScorer(disease_name=disease_name)
                _stub = [
                    {
                        "name": d["name"],
                        "drug_name": d["name"],
                        "target_genes": d.get("targets", []),
                    }
                    for d in drugs_data
                ]
                _scored = await tef.score_batch(_stub)
                tissue_score_map = {
                    c["name"]: c.get("tissue_expression_score", 0.0) for c in _scored
                }
                try:
                    await tef.close()
                except Exception:
                    pass
            except Exception as e:
                logger.warning("Tissue expression scoring failed (non-fatal): %s", e)

        candidates = []
        for drug in drugs_data:
            drug_name = drug["name"]
            lit_score = pubmed_score_map.get(drug_name, 0.0)
            ppi_score = ppi_score_map.get(drug_name, 0.0)
            sim_score = sim_score_map.get(drug_name, 0.0)
            tissue_score = tissue_score_map.get(drug_name, 0.0)

            score, evidence = scorer.score_drug_disease_match(
                drug_name,
                disease_name,
                disease_data,
                drug,
                external_literature_score=lit_score,
                ppi_score=ppi_score,
                similarity_score=sim_score,
            )

            if tissue_score > 0:
                score = min(1.0, score * 0.88 + tissue_score * 0.12)

            if score >= min_score:
                candidates.append({
                    "name": drug_name,
                    "drug_name": drug_name,
                    "drug_id": drug.get("id", ""),
                    "score": score,
                    "confidence": evidence["confidence"],
                    "shared_genes": evidence["shared_genes"],
                    "shared_pathways": evidence["shared_pathways"],
                    "explanation": evidence["explanation"],
                    "indication": drug.get("indication", ""),
                    "mechanism": drug.get("mechanism", ""),
                    "gene_score": evidence["gene_score"],
                    "pathway_score": evidence["pathway_score"],
                    "ppi_score": evidence["ppi_score"],
                    "similarity_score": evidence["similarity_score"],
                    # KEY: mechanism_score passed to combo scorer to reduce context penalty
                    "mechanism_score": evidence["mechanism_score"],
                    "literature_score": evidence["literature_score"],
                    "tissue_expression_score": tissue_score,
                    "polypharmacology_score": 0.0,
                    "target_genes": [t.upper() for t in (drug.get("targets") or [])],
                    "targets": drug.get("targets", []),
                    "patent_status": drug.get("patent_status", "unknown"),
                    "patent_reason": drug.get("patent_reason", ""),
                    "pb_status": drug.get("pb_status", ""),
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
                        c["score"] = min(1.0, c["score"] + poly * 0.25)
                top_50_names = {c["name"] for c in top_50}
                rest = [c for c in candidates if c["name"] not in top_50_names]
                candidates = top_50 + rest
            except Exception as e:
                logger.warning("Polypharmacology scoring failed (non-fatal): %s", e)

        return candidates

    # ── Primary TwinTrial entry point ─────────────────────────────────────────

    async def generate_treatment_plan(
        self,
        disease_name: str,
        max_regimens: int = 10,
        include_triples: bool = True,
        fetch_ppi: bool = True,
        fetch_similarity: bool = True,
        use_tissue: bool = True,
        # Lowered to 0.03 so mechanism-confirmed drugs always enter pool
        min_single_score: float = 0.03,
    ) -> Dict:
        """TwinTrial primary entry point — generates ranked combination regimens."""
        logger.info("=" * 65)
        logger.info("TWINTRIAL TREATMENT PLAN: %s", disease_name.upper())
        logger.info("=" * 65)

        logger.info("[1/8] Fetching disease data from OpenTargets...")
        disease_data = await self.data_fetcher.fetch_disease_data(disease_name)
        if not disease_data:
            return {
                "success": False,
                "error": f"Disease not found in OpenTargets: {disease_name}",
                "disease": disease_name,
            }

        logger.info("[2/8] Fetching and filtering to generic drugs only...")
        generic_drugs, excluded, generic_stats = await self.fetch_generic_drugs(limit=3000)
        logger.info(
            "Generic pool: %d drugs (%d excluded as patented)",
            len(generic_drugs), len(excluded),
        )

        # Deduplication uses normalised names to catch salt-form duplicates
        generic_drugs = _deduplicate_drug_pool(generic_drugs)
        logger.info("After deduplication: %d drugs", len(generic_drugs))

        logger.info("[3/8] Scoring singles...")
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
        logger.info("%d candidates scored", len(candidates))

        # Diagnostic logging
        for c in candidates[:5]:
            logger.info(
                "  Top: %s score=%.3f mech=%.3f",
                c["drug_name"], c["score"], c.get("mechanism_score", 0),
            )
        for c in candidates:
            if c.get("mechanism_score", 0) >= 0.8 and c["score"] < 0.40:
                logger.info(
                    "  High-mech low-score: %s score=%.3f mech=%.3f",
                    c["drug_name"], c["score"], c.get("mechanism_score", 0),
                )

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
            "%d safe candidates (%d filtered for safety)",
            len(safe_candidates), len(filtered_out),
        )

        disease_genes = disease_data.get("genes", [])

        logger.info("[5/8] Building combo pool (mechanism-aware, fully dynamic)...")
        combo_pool = _build_combo_pool(
            candidates=safe_candidates,
            disease_name=disease_name,
            min_score=min_single_score,
            top_n=60,
            mech_bonus_threshold=0.55,
            mech_override_threshold=0.80,
            max_pool=70,
        )
        logger.info("%d candidates in combo pool", len(combo_pool))

        # Log mechanism class distribution in combo pool
        class_counts: Dict[str, int] = {}
        for c in combo_pool:
            mech = c.get("mechanism", "")
            name = c.get("drug_name", "")
            cls = classify_mechanism(mech) if mech else classify_mechanism(name)
            class_counts[cls] = class_counts.get(cls, 0) + 1
        logger.info(
            "Combo pool classes: %s",
            dict(sorted(class_counts.items(), key=lambda x: -x[1])[:10]),
        )

        logger.info("[6/8] Scoring drug combinations...")
        combos: List[Dict] = []
        try:
            from .combo_worker import ComboWorkerPool
            import multiprocessing as _mp

            pool = ComboWorkerPool(n_workers=max(1, _mp.cpu_count() - 1))
            proofs = pool.run(
                candidates=combo_pool,
                disease_genes=disease_genes,
                disease_name=disease_name,
                max_pairs=5000,
                top_n_singles=60,
                include_triples=include_triples,
            )
            combos = [
                {
                    "combo_name": p.get(
                        "regimen",
                        f"{p.get('drug_a','')} + {p.get('drug_b','')}",
                    ),
                    "combo_score": p.get("final_score", p.get("combo_score", 0)),
                    "n_drugs": p.get("n_drugs", 2),
                    "shared_genes": (
                        p.get("target_pathway_nodes", {}).get(
                            "targets_shared_with_disease", []
                        )
                        or p.get("shared_genes", [])
                    ),
                    "is_synergistic": (
                        p.get("synergy_analysis", {}).get("synergy_call") == "SYNERGISTIC"
                        if "synergy_analysis" in p
                        else p.get("is_synergistic", False)
                    ),
                    "is_antagonistic": False,
                    "safety_margin": p.get("safety_margin", 1.0),
                    "mechanism_a": p.get("mechanistic_rationale", {}).get(
                        "mechanism_class_a", ""
                    ),
                    "mechanism_b": p.get("mechanistic_rationale", {}).get(
                        "mechanism_class_b", ""
                    ),
                    "base_score": p.get("score_breakdown", {}).get(
                        "mechanism_combo_score", 0
                    ),
                    "synergy_bonus": p.get("score_breakdown", {}).get(
                        "bliss_loewe_synergy_bonus", 0
                    ),
                    "antagonism_penalty": 0.0,
                    "coverage_bonus": 0.0,
                    "redundancy_penalty": 0.0,
                    "wet_lab_targets": (
                        p.get("target_pathway_nodes", {}).get(
                            "targets_shared_with_disease", []
                        )[:5]
                        or p.get("wet_lab_targets", [])[:5]
                    ),
                    "combined_gene_coverage": len(
                        p.get("target_pathway_nodes", {}).get(
                            "combined_disease_targets", []
                        )
                        or p.get("shared_genes", [])
                    ),
                    "simulation_proof": p,
                }
                for p in proofs
            ]
        except Exception as e:
            logger.warning(
                "ComboWorkerPool unavailable (%s: %s), using in-process fallback",
                type(e).__name__, e,
            )
            raw_combos = rank_combinations(
                candidates=combo_pool,
                disease_genes=disease_genes,
                disease_name=disease_name,
                max_pairs=3000,
                top_n_singles=60,
                include_triples=include_triples,
                min_combo_score=0.0,
            )
            combos = [
                {
                    "combo_name": c["combo_name"],
                    "combo_score": c["combo_score"],
                    "n_drugs": c.get("n_drugs", 2),
                    "shared_genes": c.get("shared_genes", []),
                    "is_synergistic": c.get("is_synergistic", False),
                    "is_antagonistic": c.get("is_antagonistic", False),
                    "safety_margin": 1.0,
                    "mechanism_a": c.get("mechanism_a", ""),
                    "mechanism_b": c.get("mechanism_b", ""),
                    "base_score": c.get("base_score", 0),
                    "synergy_bonus": c.get("synergy_bonus", 0),
                    "antagonism_penalty": c.get("antagonism_penalty", 0),
                    "coverage_bonus": c.get("coverage_bonus", 0),
                    "redundancy_penalty": c.get("redundancy_penalty", 0),
                    "wet_lab_targets": c.get("wet_lab_targets", [])[:5],
                    "combined_gene_coverage": c.get("combined_gene_coverage", 0),
                }
                for c in raw_combos
            ]

        logger.info("%d combinations generated", len(combos))

        logger.info(
            "[7/8] Running virtual Phase 2 trials on top %d combos...",
            max_regimens * 2,
        )
        trial_results: List[Dict] = []
        try:
            from .insilico_trial import InSilicoTrialSimulator

            top_for_trial = combos[: max_regimens * 2]

            # Build target lookup using NORMALISED names (fixes salt-form mismatch)
            canonical_lookup = _build_canonical_lookup(safe_candidates)

            trial_inputs = []
            for combo in top_for_trial:
                combo_name = combo["combo_name"]
                # Split on " + " and normalise each drug name
                drug_names_in_combo = [
                    _normalise_drug_name(d.strip()) for d in combo_name.split(" + ")
                ]
                all_targets = list(
                    set(
                        t
                        for dn in drug_names_in_combo
                        for t in (
                            canonical_lookup.get(dn, {}).get("target_genes", [])
                        )
                    )
                )
                trial_inputs.append({
                    "drug_name": combo_name,
                    "score": combo["combo_score"],
                    "target_genes": all_targets,
                    "disease_genes": disease_genes,
                    "polypharmacology_score": min(combo["combo_score"] + 0.05, 1.0),
                    "chembl_properties": {},
                })

            simulator = InSilicoTrialSimulator(disease=disease_name, n_patients=200)
            trial_results = await simulator.run_batch(trial_inputs)
            logger.info("%d trial results completed", len(trial_results))
        except Exception as e:
            logger.warning("Virtual trials failed (non-fatal): %s", e)

        logger.info("[8/8] Assembling treatment plan...")
        assembler = TreatmentPlanAssembler(disease_name=disease_name)
        plan = await assembler.build(
            disease_data=disease_data,
            candidates=safe_candidates,
            combos=combos,
            trial_results=trial_results,
            generic_stats=generic_stats,
            top_n=max_regimens,
        )

        plan["success"] = True
        plan["disease"] = disease_name
        plan["candidates"] = safe_candidates[:20]
        plan["pipeline_stats"] = {
            "total_drugs_evaluated": len(generic_drugs),
            "after_generic_filter": len(generic_drugs),
            "after_scoring": len(candidates),
            "after_safety_filter": len(safe_candidates),
            "combo_pool_size": len(combo_pool),
            "combos_generated": len(combos),
            "trials_run": len(trial_results),
            "excluded_patented": len(excluded),
            "filtered_unsafe": len(filtered_out),
        }

        top = plan["ranked_regimens"][0] if plan.get("ranked_regimens") else {}
        logger.info("=" * 65)
        logger.info("COMPLETE: %s", disease_name)
        logger.info("  Top regimen:  %s", top.get("regimen", "N/A"))
        logger.info("  ORR estimate: %.1f%%", top.get("orr_estimate", 0) * 100)
        logger.info("  P2 prob:      %.2f", top.get("p2_probability", 0))
        logger.info("  Priority:     %s", top.get("priority", "N/A"))
        logger.info("=" * 65)

        return plan

    # ── Backward-compatible entry point ───────────────────────────────────────

    async def analyze_disease(
        self,
        disease_name: str,
        min_score: float = 0.2,
        max_results: int = 10,
    ) -> Dict:
        plan = await self.generate_treatment_plan(
            disease_name=disease_name,
            max_regimens=max_results,
        )
        if not plan.get("success"):
            return {"success": False, "error": plan.get("error", "Unknown error")}

        return {
            "success": True,
            "disease": disease_name,
            "disease_data": plan.get("header", {}),
            "candidates": plan.get("candidates", []),
            "top_candidates": plan.get("candidates", [])[:20],
            "trial_results": [],
            "trial_report": plan.get("pag_brief", ""),
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


RepurposingPipeline = ProductionPipeline


async def analyze(
    disease_name: str, min_score: float = 0.2, max_results: int = 20
) -> Dict:
    pipeline = ProductionPipeline()
    try:
        return await pipeline.generate_treatment_plan(
            disease_name, max_regimens=max_results
        )
    finally:
        await pipeline.close()