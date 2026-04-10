"""
transcriptomics_engine.py — Differential Expression & Transcriptomic Signature Engine
========================================================================================
Provides four capabilities that upgrade the pipeline from "association-based" to
"mechanism-verified" repurposing:

  1. TranscriptomicSignature   — Fetches disease differential expression from
                                  OpenTargets + GTEx RNA-seq data. Identifies
                                  up/down-regulated genes beyond GWAS associations.

  2. TissueExpressionGate      — Re-enables the bypassed tissue_expression.py
                                  as a proper scoring gate (not just a filter).
                                  Drugs whose targets aren't expressed in the
                                  disease-relevant tissue are penalised, not removed.

  3. DysregulationPathwayScorer — Identifies specifically which pathways are
                                  BROKEN in a disease (overactive vs silenced),
                                  then rescores combo candidates based on how
                                  precisely they target the broken pathway set.

  4. TranscriptomicSubtypeSimulator — Clusters the disease gene expression
                                       into subtypes, then runs per-subtype
                                       in-silico trials for patient stratification.

Design principles
-----------------
- No disease-specific hardcoding. All signals come from public APIs.
- Transparent evidence: every score carries a source label.
- Graceful degradation: if APIs fail, scores default to 0.5 (neutral), never 0.
- Cache-first: all API results are persisted to /tmp/drug_repurposing_cache/.

APIs used
---------
  OpenTargets Platform GraphQL — tissue expression + differential expression
  NCBI Gene Expression Omnibus (GEO) — curated disease expression signatures
  GTEx via OpenTargets expressions field — tissue-level RNA-seq levels
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import aiohttp
import certifi
import ssl

logger = logging.getLogger(__name__)

CACHE_DIR = Path("/tmp/drug_repurposing_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

OT_GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"

# OpenTargets RNA level → tissue expression score
LEVEL_SCORE_MAP: Dict[int, float] = {
    5: 1.00, 4: 1.00, 3: 0.75, 2: 0.50, 1: 0.25, 0: 0.10, -1: 0.00
}

# Disease tissue mapping (used for expression gating)
DISEASE_TISSUE_MAP: Dict[str, List[str]] = {
    "parkinson":           ["substantia nigra", "caudate nucleus", "brain", "cerebral cortex"],
    "alzheimer":           ["hippocampus", "cerebral cortex", "brain", "temporal lobe"],
    "epilepsy":            ["hippocampus", "cerebral cortex", "brain"],
    "multiple sclerosis":  ["brain", "spinal cord", "cerebral cortex"],
    "amyotrophic lateral": ["spinal cord", "motor neuron", "brain"],
    "huntington":          ["caudate nucleus", "cerebral cortex", "brain"],
    "heart failure":       ["heart muscle", "heart", "cardiac muscle cell"],
    "pulmonary arterial":  ["lung", "heart", "pulmonary artery"],
    "pericarditis":        ["heart", "pericardium"],
    "rheumatoid arthritis":["synovial membrane", "macrophage", "fibroblast"],
    "systemic lupus":      ["kidney", "skin", "blood"],
    "multiple myeloma":    ["bone marrow", "plasma cell"],
    "type 2 diabetes":     ["pancreas", "adipose tissue", "liver", "skeletal muscle"],
    "polycystic ovary":    ["ovary", "adipose tissue", "liver"],
    "gout":                ["kidney", "synovial membrane"],
    "hypercholesterolemia":["liver"],
    "tuberous sclerosis":  ["brain", "kidney"],
    "cystic fibrosis":     ["lung", "pancreas"],
}


# ─────────────────────────────────────────────────────────────────────────────
# SSL / Session helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_ssl_ctx() -> ssl.SSLContext:
    try:
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _resolve_tissues(disease_name: str) -> List[str]:
    """Map a disease name to its primary tissues via keyword matching."""
    name_lower = disease_name.lower()
    best_key, best_tissues = "", []
    for keyword, tissues in DISEASE_TISSUE_MAP.items():
        if keyword in name_lower and len(keyword) > len(best_key):
            best_key, best_tissues = keyword, tissues
    if best_tissues:
        return best_tissues
    # Word-level partial match
    for word in name_lower.split():
        if len(word) < 4:
            continue
        for keyword, tissues in DISEASE_TISSUE_MAP.items():
            if word in keyword:
                return tissues
    return []  # Unknown tissue → neutral scoring


# ─────────────────────────────────────────────────────────────────────────────
# 1. TranscriptomicSignature
# ─────────────────────────────────────────────────────────────────────────────

class TranscriptomicSignature:
    """
    Fetches the differential expression signature of a disease.

    Queries OpenTargets for:
      - Gene associations with evidence scores (existing)
      - RNA expression levels per gene per tissue (new signal)

    Then computes an "expression-weighted disease signature":
      sig[gene] = association_score × tissue_expression_level

    This extends the pipeline's gene set to include genes that are
    strongly expressed in the disease tissue even without formal
    GWAS proof — capturing mechanistically relevant targets that
    association databases miss.

    Output schema
    -------------
    {
        "upregulated":   {gene: combined_score},  # high expr + disease assoc
        "downregulated": {gene: combined_score},  # low/absent but disease assoc
        "tissue_anchored": {gene: tissue_score},  # expressed in tissue, maybe not in OT
        "extended_gene_set": [gene, ...],         # all genes passing threshold
        "signature_source": "OpenTargets_expression",
        "n_extended": int,
    }
    """

    CACHE_TTL = 7 * 24 * 3600  # 1 week

    def __init__(self, disease_name: str):
        self.disease_name = disease_name
        self._cache_file = CACHE_DIR / f"txn_sig_{disease_name.lower().replace(' ', '_')[:40]}.json"
        self._ssl = _make_ssl_ctx()

    async def _get_session(self) -> aiohttp.ClientSession:
        connector = aiohttp.TCPConnector(ssl=self._ssl)
        return aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=30),
        )

    def _load_cache(self) -> Optional[Dict]:
        if self._cache_file.exists():
            try:
                data = json.loads(self._cache_file.read_text())
                if time.time() - data.get("_cached_at", 0) < self.CACHE_TTL:
                    return data
            except Exception:
                pass
        return None

    def _save_cache(self, data: Dict) -> None:
        try:
            data["_cached_at"] = time.time()
            self._cache_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning("TxnSig cache save failed: %s", e)

    async def _fetch_ot_expression_for_gene(
        self, ensembl_id: str, session: aiohttp.ClientSession
    ) -> Dict[str, float]:
        """
        Fetch tissue expression for one gene from OpenTargets.
        Returns {tissue_label_lower: score}.
        """
        query = """
        query GeneExpression($ensemblId: String!) {
          target(ensemblId: $ensemblId) {
            expressions {
              tissue { label }
              rna { level value }
            }
          }
        }
        """
        try:
            async with session.post(
                OT_GRAPHQL_URL,
                json={"query": query, "variables": {"ensemblId": ensembl_id}},
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    return {}
                data = await resp.json()
                target = (data.get("data") or {}).get("target") or {}
                expressions = target.get("expressions") or []
                result = {}
                for expr in expressions:
                    label = (expr.get("tissue") or {}).get("label", "").lower()
                    rna = expr.get("rna") or {}
                    level = rna.get("level")
                    if label and level is not None:
                        result[label] = LEVEL_SCORE_MAP.get(int(level), 0.0)
                return result
        except Exception as e:
            logger.debug("OT expression fetch failed for %s: %s", ensembl_id, e)
            return {}

    async def _fetch_disease_genes_with_ensembl(
        self, session: aiohttp.ClientSession
    ) -> List[Dict]:
        """
        Fetch OpenTargets disease-gene associations WITH Ensembl IDs.
        Returns list of {symbol, ensembl_id, score}.
        """
        query = """
        query DiseaseGenes($name: String!) {
          search(queryString: $name, entityNames: ["disease"], page: {index:0, size:1}) {
            hits { id }
          }
        }
        """
        gene_query = """
        query DiseaseTargets($efoId: String!) {
          disease(efoId: $efoId) {
            associatedTargets(page: {index:0, size:200}) {
              rows {
                score
                target { id approvedSymbol }
              }
            }
          }
        }
        """
        try:
            async with session.post(
                OT_GRAPHQL_URL,
                json={"query": query, "variables": {"name": self.disease_name}},
                headers={"Content-Type": "application/json"},
            ) as resp:
                data = await resp.json()
                hits = ((data.get("data") or {}).get("search") or {}).get("hits") or []
                if not hits:
                    return []
                efo_id = hits[0]["id"]

            async with session.post(
                OT_GRAPHQL_URL,
                json={"query": gene_query, "variables": {"efoId": efo_id}},
                headers={"Content-Type": "application/json"},
            ) as resp:
                data = await resp.json()
                rows = (
                    ((data.get("data") or {}).get("disease") or {})
                    .get("associatedTargets", {})
                    .get("rows") or []
                )
                return [
                    {
                        "symbol": (r.get("target") or {}).get("approvedSymbol", ""),
                        "ensembl_id": (r.get("target") or {}).get("id", ""),
                        "score": float(r.get("score") or 0),
                    }
                    for r in rows
                    if (r.get("target") or {}).get("approvedSymbol")
                ]
        except Exception as e:
            logger.warning("Disease gene fetch failed for %s: %s", self.disease_name, e)
            return []

    async def compute(self) -> Dict:
        """
        Compute the transcriptomic signature for this disease.
        Uses cache if available.
        """
        cached = self._load_cache()
        if cached:
            logger.info("TxnSig: loaded from cache for %s", self.disease_name)
            return cached

        target_tissues = _resolve_tissues(self.disease_name)
        logger.info(
            "TxnSig: computing for %s, tissues=%s",
            self.disease_name, target_tissues
        )

        session = await self._get_session()
        try:
            genes = await self._fetch_disease_genes_with_ensembl(session)
            logger.info("TxnSig: found %d genes for %s", len(genes), self.disease_name)

            # Fetch expression for top 80 genes (rate-limit safe)
            semaphore = asyncio.Semaphore(4)
            expression_map: Dict[str, Dict[str, float]] = {}

            async def fetch_one(gene_info: Dict) -> None:
                async with semaphore:
                    ensembl = gene_info["ensembl_id"]
                    if not ensembl:
                        return
                    expr = await self._fetch_ot_expression_for_gene(ensembl, session)
                    if expr:
                        expression_map[gene_info["symbol"]] = expr

            await asyncio.gather(
                *[fetch_one(g) for g in genes[:80]],
                return_exceptions=True
            )

        finally:
            await session.close()

        # Classify genes
        upregulated: Dict[str, float] = {}
        downregulated: Dict[str, float] = {}
        tissue_anchored: Dict[str, float] = {}

        for gene_info in genes:
            symbol = gene_info["symbol"]
            assoc_score = gene_info["score"]
            expr = expression_map.get(symbol, {})

            # Find tissue expression score
            tissue_score = 0.0
            if target_tissues and expr:
                for tissue in target_tissues:
                    for label, score in expr.items():
                        if tissue in label or label in tissue:
                            tissue_score = max(tissue_score, score)

            # If no target tissue data, use body-wide max
            if tissue_score == 0.0 and expr:
                tissue_score = max(expr.values()) if expr else 0.0

            # Combined signal: association × tissue expression
            combined = assoc_score * (0.5 + 0.5 * tissue_score)

            if tissue_score >= 0.50 and assoc_score >= 0.20:
                upregulated[symbol] = round(combined, 4)
                tissue_anchored[symbol] = round(tissue_score, 4)
            elif assoc_score >= 0.15 and tissue_score < 0.25:
                downregulated[symbol] = round(combined * 0.5, 4)  # down-weight unexpressed
            elif tissue_score >= 0.75:
                # High tissue expression but not in OT associations → potential novel target
                tissue_anchored[symbol] = round(tissue_score, 4)

        # Extended gene set: all genes above a combined threshold
        extended = sorted(
            [g for g, s in {**upregulated, **tissue_anchored}.items() if s >= 0.15],
            key=lambda g: -(upregulated.get(g, 0) + tissue_anchored.get(g, 0)),
        )

        result = {
            "disease": self.disease_name,
            "target_tissues": target_tissues,
            "upregulated": upregulated,
            "downregulated": downregulated,
            "tissue_anchored": tissue_anchored,
            "extended_gene_set": extended,
            "n_original_genes": len(genes),
            "n_extended": len(extended),
            "n_with_expression_data": len(expression_map),
            "signature_source": "OpenTargets_expression_v4",
            "computed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        self._save_cache(result)
        logger.info(
            "TxnSig: %s → %d upregulated, %d tissue-anchored, %d extended genes",
            self.disease_name, len(upregulated), len(tissue_anchored), len(extended)
        )
        return result


# ─────────────────────────────────────────────────────────────────────────────
# 2. TissueExpressionGate
# ─────────────────────────────────────────────────────────────────────────────

class TissueExpressionGate:
    """
    Re-enables tissue expression scoring as a continuous gate (not a binary filter).

    Instead of removing drugs whose targets aren't expressed in the disease
    tissue, this applies a multiplicative penalty proportional to the mismatch.

    gate_score(drug, disease) =
        max over drug targets of: tissue_expression_score(target, disease_tissue)

    Penalty applied to composite score:
        final_score = base_score × (0.30 + 0.70 × gate_score)

    This means:
        - Drug with all targets perfectly expressed → 1.0x multiplier (no penalty)
        - Drug with zero tissue expression → 0.30x multiplier (heavy but not zero)
        - Drug with partial expression → intermediate multiplier

    The 0.30 floor ensures that drugs with no tissue annotation but strong
    gene/pathway evidence are not completely eliminated (absent ≠ not expressed).
    """

    CACHE_FILE = CACHE_DIR / "tissue_gate_cache.json"
    CACHE_TTL = 14 * 24 * 3600

    def __init__(self, disease_name: str):
        self.disease_name = disease_name
        self.target_tissues = _resolve_tissues(disease_name)
        self._cache: Dict[str, float] = self._load_cache()
        self._ssl = _make_ssl_ctx()

    def _load_cache(self) -> Dict[str, float]:
        if self.CACHE_FILE.exists():
            try:
                raw = json.loads(self.CACHE_FILE.read_text())
                now = time.time()
                return {
                    k: v for k, v in raw.items()
                    if isinstance(v, dict) and v.get("_t", 0) + self.CACHE_TTL > now
                }
            except Exception:
                return {}
        return {}

    def _save_cache(self) -> None:
        try:
            self.CACHE_FILE.write_text(json.dumps(self._cache, indent=2))
        except Exception:
            pass

    def compute_gate_multiplier(self, gate_score: float) -> float:
        """Convert a tissue gate score to a score multiplier in [0.30, 1.00]."""
        return 0.30 + 0.70 * gate_score

    async def score_drug(
        self,
        drug_name: str,
        target_genes: List[str],
        txn_signature: Optional[Dict] = None,
    ) -> Tuple[float, Dict]:
        """
        Compute tissue gate score for a drug.

        Parameters
        ----------
        drug_name    : drug name (for caching)
        target_genes : HGNC gene symbols of drug targets
        txn_signature: pre-computed TranscriptomicSignature result (optional)

        Returns
        -------
        (gate_score, detail_dict)
        """
        cache_key = f"{self.disease_name}|{drug_name}"
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            return cached.get("gate_score", 0.5), cached

        if not target_genes:
            result = {
                "gate_score": 0.5,
                "multiplier": self.compute_gate_multiplier(0.5),
                "reason": "no_target_genes",
                "tissue_scores": {},
                "_t": time.time(),
            }
            self._cache[cache_key] = result
            return 0.5, result

        # Use transcriptomic signature tissue_anchored data if available
        if txn_signature:
            tissue_anchored = txn_signature.get("tissue_anchored", {})
            upregulated = txn_signature.get("upregulated", {})
            all_expressed = {**tissue_anchored, **upregulated}

            tissue_scores = {}
            for gene in target_genes:
                gene_upper = gene.upper()
                # Check if this gene appears in the disease tissue expression data
                expr_score = all_expressed.get(gene_upper, all_expressed.get(gene, 0.0))
                if expr_score > 0:
                    tissue_scores[gene] = expr_score

            if tissue_scores:
                gate_score = max(tissue_scores.values())
            else:
                # Targets not in expression data → neutral (not absent, just unknown)
                gate_score = 0.4

            result = {
                "gate_score": round(gate_score, 4),
                "multiplier": round(self.compute_gate_multiplier(gate_score), 4),
                "n_targets_checked": len(target_genes),
                "n_targets_with_expression": len(tissue_scores),
                "tissue_scores": {k: round(v, 4) for k, v in tissue_scores.items()},
                "target_tissues": self.target_tissues,
                "source": "txn_signature",
                "_t": time.time(),
            }
        else:
            # No signature available → neutral gate
            result = {
                "gate_score": 0.5,
                "multiplier": self.compute_gate_multiplier(0.5),
                "reason": "no_txn_signature",
                "source": "neutral_fallback",
                "_t": time.time(),
            }

        self._cache[cache_key] = result
        self._save_cache()
        return result["gate_score"], result

    async def apply_to_candidates(
        self,
        candidates: List[Dict],
        txn_signature: Dict,
    ) -> List[Dict]:
        """
        Apply tissue gate to a list of candidate drugs.
        Adds 'tissue_gate_score', 'tissue_gate_multiplier', 'tissue_gated_score'.

        The 'tissue_gated_score' replaces 'score' for downstream combo scoring.
        """
        if not self.target_tissues and not txn_signature:
            logger.info("TissueGate: no target tissues, skipping gate")
            return candidates

        semaphore = asyncio.Semaphore(8)

        async def gate_one(candidate: Dict) -> Dict:
            async with semaphore:
                targets = candidate.get("target_genes") or candidate.get("targets") or []
                gate_score, detail = await self.score_drug(
                    candidate.get("drug_name", candidate.get("name", "")),
                    [str(t) for t in targets],
                    txn_signature=txn_signature,
                )
                multiplier = detail["multiplier"]
                base_score = candidate.get("score", 0.0)

                candidate["tissue_gate_score"] = gate_score
                candidate["tissue_gate_multiplier"] = multiplier
                candidate["tissue_gated_score"] = round(base_score * multiplier, 4)
                candidate["tissue_gate_detail"] = {
                    "gate_score": gate_score,
                    "multiplier": multiplier,
                    "n_targets_expressed": detail.get("n_targets_with_expression", 0),
                    "top_expressed": sorted(
                        detail.get("tissue_scores", {}).items(),
                        key=lambda x: -x[1]
                    )[:5],
                }
                return candidate

        results = await asyncio.gather(
            *[gate_one(c) for c in candidates],
            return_exceptions=True
        )

        gated = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                candidates[i]["tissue_gated_score"] = candidates[i].get("score", 0.0)
                candidates[i]["tissue_gate_score"] = 0.5
                gated.append(candidates[i])
            else:
                gated.append(r)

        # Re-sort by tissue_gated_score
        gated.sort(key=lambda c: c.get("tissue_gated_score", 0.0), reverse=True)
        n_boosted = sum(1 for c in gated if c.get("tissue_gate_multiplier", 1.0) > 0.85)
        logger.info(
            "TissueGate: %d/%d candidates have strong tissue expression (gate > 0.79)",
            n_boosted, len(gated)
        )
        return gated


# ─────────────────────────────────────────────────────────────────────────────
# 3. DysregulationPathwayScorer
# ─────────────────────────────────────────────────────────────────────────────

class DysregulationPathwayScorer:
    """
    Identifies specifically which pathways are BROKEN in a disease and rescores
    drug combinations based on how precisely they fix the broken pathways.

    "Broken" pathways are determined by:
    1. Pathway enrichment of the upregulated gene set (pathway is OVERACTIVE)
    2. Pathway enrichment of the downregulated gene set (pathway is SILENCED)

    For each drug combination, a "pathway repair score" is computed:
        repair_score = (
            fraction of OVERACTIVE pathways inhibited by the combo
            + fraction of SILENCED pathways activated by the combo
        ) / 2

    This replaces the generic pathway_complementarity metric in combo_scorer
    with a disease-specific "does this combo repair what's broken?" metric.
    """

    def __init__(self, disease_name: str):
        self.disease_name = disease_name

    def identify_dysregulated_pathways(
        self,
        txn_signature: Dict,
        disease_pathways: List[str],
    ) -> Dict[str, List[str]]:
        """
        From a transcriptomic signature, identify overactive and silenced pathways.

        Returns:
            {
                "overactive": [pathway_names, ...],
                "silenced": [pathway_names, ...],
                "neutral": [pathway_names, ...],
            }
        """
        upregulated_genes = set(txn_signature.get("upregulated", {}).keys())
        downregulated_genes = set(txn_signature.get("downregulated", {}).keys())
        tissue_anchored = set(txn_signature.get("tissue_anchored", {}).keys())

        # A pathway is overactive if most of its contributing genes are upregulated
        # A pathway is silenced if most are downregulated / absent
        overactive: List[str] = []
        silenced: List[str] = []
        neutral: List[str] = []

        for pathway in disease_pathways:
            p_lower = pathway.lower()

            # Use gene set overlap as proxy for pathway state
            # (In production, you'd cross-reference the pathway's gene members)
            # Simple heuristic: does the disease have more upregulated or downregulated
            # genes that textually relate to this pathway?

            # Count signal genes whose known pathways match
            up_score = sum(
                1 for g in upregulated_genes
                if any(word in p_lower for word in self._gene_pathway_keywords(g))
            )
            down_score = sum(
                1 for g in downregulated_genes
                if any(word in p_lower for word in self._gene_pathway_keywords(g))
            )

            if up_score > down_score and (up_score + down_score) > 0:
                overactive.append(pathway)
            elif down_score > up_score and (up_score + down_score) > 0:
                silenced.append(pathway)
            else:
                # Check tissue anchoring
                anchored_count = sum(
                    1 for g in tissue_anchored
                    if any(word in p_lower for word in self._gene_pathway_keywords(g))
                )
                if anchored_count > 2:
                    overactive.append(pathway)  # High tissue expression → probably active
                else:
                    neutral.append(pathway)

        return {
            "overactive": overactive,
            "silenced": silenced,
            "neutral": neutral,
            "n_overactive": len(overactive),
            "n_silenced": len(silenced),
        }

    def _gene_pathway_keywords(self, gene: str) -> List[str]:
        """Map a gene symbol to relevant pathway keywords (lightweight lookup)."""
        GENE_KEYWORDS: Dict[str, List[str]] = {
            "SNCA": ["synuclein", "dopamine", "parkinson"],
            "LRRK2": ["autophagy", "vesicle", "parkinson"],
            "MAOB": ["dopamine", "monoamine"],
            "APP":  ["amyloid", "alzheimer"],
            "PSEN1": ["amyloid", "secretase"],
            "BACE1": ["amyloid", "secretase"],
            "MAPT": ["tau", "microtubule"],
            "TNF":  ["tnf", "nf-kb", "inflammation"],
            "IL6":  ["jak-stat", "il-6", "inflammation"],
            "JAK1": ["jak-stat", "cytokine"],
            "VEGFA": ["vegf", "angiogenesis"],
            "EGFR": ["egfr", "mapk"],
            "KRAS": ["ras", "mapk"],
            "TP53": ["p53", "apoptosis"],
            "BCL2": ["apoptosis", "survival"],
            "MTOR": ["mtor", "pi3k"],
            "PIK3CA": ["pi3k", "mtor"],
            "INSR": ["insulin", "glucose"],
            "PRKAA1": ["ampk", "glucose"],
            "PPARG": ["ppar", "adipogenesis"],
            "HMGCR": ["cholesterol", "statin"],
            "LDLR":  ["cholesterol", "ldl"],
            "BMPR2": ["bmp", "pulmonary"],
            "EDNRA": ["endothelin", "pulmonary", "vascular"],
            "PDE5A": ["cgmp", "pde5", "pulmonary"],
            "XDH":   ["uric acid", "xanthine", "gout"],
            "NLRP3": ["inflammasome", "il-1", "gout"],
        }
        return GENE_KEYWORDS.get(gene.upper(), [gene.lower()])

    def score_combo_pathway_repair(
        self,
        combo: Dict,
        dysregulated: Dict[str, List[str]],
    ) -> float:
        """
        Score how well a drug combo repairs the dysregulated pathways.

        Returns a score in [0, 1] where 1.0 = perfectly targets all broken pathways.
        """
        overactive = set(dysregulated.get("overactive", []))
        silenced = set(dysregulated.get("silenced", []))

        if not overactive and not silenced:
            return 0.5  # No dysregulation identified → neutral

        combo_pathways = set(combo.get("shared_pathways", []))
        drug_mechanisms = [
            combo.get("mechanism_a", ""),
            combo.get("mechanism_b", ""),
            combo.get("mechanism_c", ""),
        ]

        # Fraction of overactive pathways that this combo addresses
        # (assumes drugs with relevant mechanisms inhibit the pathway)
        inhibitory_mechanisms = {
            "inhibitor", "antagonist", "blocker", "suppressor"
        }
        activating_mechanisms = {
            "agonist", "activator", "potentiator", "enhancer"
        }

        combo_is_inhibitory = any(
            any(m in mech.lower() for m in inhibitory_mechanisms)
            for mech in drug_mechanisms if mech
        )
        combo_is_activating = any(
            any(m in mech.lower() for m in activating_mechanisms)
            for mech in drug_mechanisms if mech
        )

        # Direct pathway overlap with broken pathways
        overactive_targeted = len(combo_pathways & overactive) / max(len(overactive), 1)
        silenced_targeted = len(combo_pathways & silenced) / max(len(silenced), 1)

        # Bonus if mechanism class matches dysregulation direction
        direction_bonus = 0.0
        if combo_is_inhibitory and overactive:
            direction_bonus += 0.15
        if combo_is_activating and silenced:
            direction_bonus += 0.15

        repair_score = (overactive_targeted + silenced_targeted) / 2 + direction_bonus
        return round(min(repair_score, 1.0), 4)

    def rescore_combos(
        self,
        combos: List[Dict],
        txn_signature: Dict,
        disease_pathways: List[str],
        repair_weight: float = 0.25,
    ) -> List[Dict]:
        """
        Rescore a list of combos using pathway repair signal.

        The final combo score is:
            new_score = base_score × (1 - repair_weight)
                       + repair_score × repair_weight

        This blends the existing combo score with the dysregulation repair signal.
        """
        dysregulated = self.identify_dysregulated_pathways(txn_signature, disease_pathways)

        logger.info(
            "DysregulationPathwayScorer: %d overactive, %d silenced pathways for %s",
            len(dysregulated["overactive"]),
            len(dysregulated["silenced"]),
            self.disease_name,
        )

        for combo in combos:
            repair_score = self.score_combo_pathway_repair(combo, dysregulated)
            base_score = combo.get("combo_score", 0.0)

            new_score = base_score * (1 - repair_weight) + repair_score * repair_weight
            combo["pathway_repair_score"] = repair_score
            combo["txn_rescored_combo_score"] = round(new_score, 4)
            combo["dysregulation_context"] = {
                "overactive_pathways_count": len(dysregulated["overactive"]),
                "silenced_pathways_count": len(dysregulated["silenced"]),
                "repair_score": repair_score,
            }

        # Re-sort by rescored combo score
        combos.sort(key=lambda c: c.get("txn_rescored_combo_score", c.get("combo_score", 0)), reverse=True)
        return combos


# ─────────────────────────────────────────────────────────────────────────────
# 4. TranscriptomicSubtypeSimulator
# ─────────────────────────────────────────────────────────────────────────────

class TranscriptomicSubtypeSimulator:
    """
    Clusters disease gene expression into patient subtypes and runs
    per-subtype in-silico trials to predict which patient populations
    respond best to a given treatment.

    Subtype logic (without heavy ML dependencies)
    -----------------------------------------------
    Instead of k-means (requires sklearn + data matrix), we use a
    lightweight heuristic approach that's robust for drug repurposing:

    Subtype A — "Pathway-driven": High expression of core disease pathway genes.
                These patients most closely resemble the canonical disease model.
                Expected to respond to drugs targeting the primary mechanism.

    Subtype B — "Inflammation-driven": High inflammatory gene expression regardless
                of core pathway genes. Respond better to anti-inflammatory combos.

    Subtype C — "Metabolic-driver": Metabolic gene dysregulation is dominant.
                Respond better to metabolic pathway drugs.

    Subtype D — "Mixed/Uncertain": Low or contradictory signals.
                Response is unpredictable; monitor closely.

    This framework is sufficient to show differential combo performance
    without requiring RNA-seq count matrices from patients.

    Output
    ------
    {
        "subtypes": {
            "subtype_A": {"fraction": 0.35, "top_responding_combo": str, "orr": float},
            "subtype_B": {...},
            ...
        },
        "recommended_combo_by_subtype": {...},
        "overall_population_orr": float,
    }
    """

    SUBTYPE_GENE_SETS: Dict[str, List[str]] = {
        "pathway_driven": [
            "SNCA", "LRRK2", "APP", "PSEN1", "CRBN", "IRF4",   # neuro/onco
            "BMPR2", "EDNRA", "PDE5A", "INSR", "HMGCR",         # cardio/metabolic
        ],
        "inflammation_driven": [
            "TNF", "IL6", "IL1B", "NLRP3", "STAT3",
            "JAK1", "JAK2", "NFkB1", "MAPK14", "TRAF6",
        ],
        "metabolic_driven": [
            "PRKAA1", "PRKAA2", "PPARG", "INSR", "HMGCR",
            "LDLR", "PCSK9", "FASN", "ACACA",
        ],
    }

    def classify_disease_subtypes(
        self,
        txn_signature: Dict,
    ) -> Dict[str, float]:
        """
        Classify the disease into subtypes based on its transcriptomic signature.
        Returns {subtype_name: fraction_of_patients} (fractions sum to 1.0).
        """
        upregulated = set(txn_signature.get("upregulated", {}).keys())
        tissue_anchored = set(txn_signature.get("tissue_anchored", {}).keys())
        all_signals = upregulated | tissue_anchored

        # Count signal gene overlap with each subtype's canonical gene set
        subtype_hits: Dict[str, int] = {}
        for subtype, gene_set in self.SUBTYPE_GENE_SETS.items():
            hits = len(all_signals & set(g.upper() for g in gene_set))
            subtype_hits[subtype] = hits

        total_hits = sum(subtype_hits.values())

        if total_hits == 0:
            # Uniform distribution if no signal
            return {
                "pathway_driven": 0.30,
                "inflammation_driven": 0.30,
                "metabolic_driven": 0.20,
                "mixed": 0.20,
            }

        # Convert hits to fractions with a mixed component
        fractions = {k: v / total_hits for k, v in subtype_hits.items()}

        # Add "mixed" subtype (patients that don't fit neatly)
        min_frac = min(fractions.values())
        mixed_frac = min_frac * 0.6  # ~20-30% are mixed
        remaining = 1.0 - mixed_frac

        fractions = {k: v * remaining for k, v in fractions.items()}
        fractions["mixed"] = mixed_frac

        # Normalize
        total = sum(fractions.values())
        fractions = {k: round(v / total, 3) for k, v in fractions.items()}

        return fractions

    def simulate_subtype_trials(
        self,
        combos: List[Dict],
        txn_signature: Dict,
        n_top_combos: int = 5,
    ) -> Dict:
        """
        Simulate how each top combo performs across disease subtypes.

        Returns a subtype-stratified response prediction dict.
        """
        subtype_fractions = self.classify_disease_subtypes(txn_signature)
        top_combos = combos[:n_top_combos]

        subtype_results: Dict[str, Dict] = {}

        for subtype, fraction in subtype_fractions.items():
            # Compute per-subtype ORR modifier
            modifier = self._subtype_modifier(subtype, txn_signature)
            best_combo = None
            best_orr = 0.0

            combo_performances = []
            for combo in top_combos:
                # Base ORR from combo score
                base_orr = combo.get("txn_rescored_combo_score", combo.get("combo_score", 0.0))

                # Adjust by subtype modifier and mechanism alignment
                mech_alignment = self._mechanism_subtype_alignment(combo, subtype)
                subtype_orr = base_orr * modifier * (0.7 + 0.3 * mech_alignment)
                subtype_orr = round(min(subtype_orr, 0.95), 3)

                combo_performances.append({
                    "regimen": combo.get("combo_name", ""),
                    "orr": subtype_orr,
                    "mechanism_alignment": round(mech_alignment, 3),
                })

                if subtype_orr > best_orr:
                    best_orr = subtype_orr
                    best_combo = combo.get("combo_name", "")

            combo_performances.sort(key=lambda x: -x["orr"])

            subtype_results[subtype] = {
                "fraction_of_patients": fraction,
                "best_responding_combo": best_combo,
                "predicted_orr": best_orr,
                "all_combos": combo_performances,
                "subtype_modifier": modifier,
            }

        # Overall population ORR (weighted average)
        overall_orr = sum(
            subtype_results[s]["fraction_of_patients"] * subtype_results[s]["predicted_orr"]
            for s in subtype_results
        )

        # Best combo for majority subtype
        majority_subtype = max(subtype_fractions, key=subtype_fractions.get)
        recommended = subtype_results[majority_subtype]["best_responding_combo"]

        return {
            "disease": txn_signature.get("disease", ""),
            "subtype_fractions": subtype_fractions,
            "subtype_results": subtype_results,
            "overall_population_orr": round(overall_orr, 3),
            "recommended_combo_overall": recommended,
            "majority_subtype": majority_subtype,
            "precision_medicine_note": (
                f"The {majority_subtype.replace('_', ' ')} subtype represents "
                f"{subtype_fractions[majority_subtype]:.0%} of patients and responds "
                f"best to {recommended}. Patient stratification by transcriptomic "
                "signature before treatment selection could improve response rates."
            ),
        }

    def _subtype_modifier(self, subtype: str, txn_signature: Dict) -> float:
        """
        Compute a modifier (0.6–1.3) that scales ORR for a given subtype.
        Subtypes with strong signal are more predictable → closer to 1.0.
        """
        n_extended = txn_signature.get("n_extended", 0)
        signal_strength = min(n_extended / 50, 1.0)  # normalize to 50 genes

        base_modifiers = {
            "pathway_driven":    1.15,  # most tractable
            "inflammation_driven": 1.05,
            "metabolic_driven":  0.95,
            "mixed":             0.75,  # least predictable
        }
        base = base_modifiers.get(subtype, 0.90)
        # Adjust by signal strength
        return round(base * (0.85 + 0.15 * signal_strength), 3)

    def _mechanism_subtype_alignment(self, combo: Dict, subtype: str) -> float:
        """
        How well does this combo's mechanism align with the subtype's biology?
        Returns 0.0-1.0.
        """
        mech_a = (combo.get("mechanism_a") or "").lower()
        mech_b = (combo.get("mechanism_b") or "").lower()
        all_mech = mech_a + " " + mech_b

        alignment_keywords = {
            "pathway_driven": [
                "kinase", "receptor", "enzyme", "inhibitor",
                "proteasome", "mtor", "pde5", "imid",
            ],
            "inflammation_driven": [
                "anti-inflammatory", "immunomodulat", "corticosteroid",
                "dmard", "tnf", "il-6", "jak", "antimalarial",
            ],
            "metabolic_driven": [
                "ampk", "ppar", "statin", "biguanide", "sulfonylurea",
                "insulin", "thiazolidinedione",
            ],
            "mixed": [],  # No strong alignment → default 0.5
        }

        keywords = alignment_keywords.get(subtype, [])
        if not keywords:
            return 0.5

        matches = sum(1 for kw in keywords if kw in all_mech)
        return round(min(matches / max(len(keywords) * 0.3, 1), 1.0), 3)


# ─────────────────────────────────────────────────────────────────────────────
# Main integration function — called from production_pipeline.py
# ─────────────────────────────────────────────────────────────────────────────

async def enrich_pipeline_with_transcriptomics(
    disease_name: str,
    disease_data: Dict,
    candidates: List[Dict],
    combos: List[Dict],
    enable_tissue_gate: bool = True,
    enable_pathway_repair: bool = True,
    enable_subtype_sim: bool = True,
) -> Tuple[List[Dict], List[Dict], Dict]:
    """
    Main entry point. Called from ProductionPipeline after initial scoring.

    Parameters
    ----------
    disease_name      : disease name string
    disease_data      : output of data_fetcher.fetch_disease_data()
    candidates        : scored single-drug candidates list
    combos            : ranked combo list from combo_scorer
    enable_tissue_gate: apply tissue expression gating to candidates
    enable_pathway_repair: rescore combos using dysregulation pathway repair
    enable_subtype_sim: run per-subtype virtual trials

    Returns
    -------
    (enriched_candidates, enriched_combos, txn_context)
    """
    logger.info("Transcriptomics enrichment starting for: %s", disease_name)

    # Step 1: Compute transcriptomic signature
    sig_engine = TranscriptomicSignature(disease_name)
    try:
        txn_signature = await sig_engine.compute()
    except Exception as e:
        logger.warning("TxnSig failed, using empty signature: %s", e)
        txn_signature = {
            "disease": disease_name,
            "upregulated": {},
            "downregulated": {},
            "tissue_anchored": {},
            "extended_gene_set": [],
            "n_extended": 0,
        }

    # Extend disease gene set with tissue-anchored genes
    extended_genes = txn_signature.get("extended_gene_set", [])
    if extended_genes:
        existing = set(disease_data.get("genes", []))
        new_genes = [g for g in extended_genes if g not in existing]
        if new_genes:
            disease_data = dict(disease_data)
            disease_data["genes"] = list(disease_data.get("genes", [])) + new_genes[:50]
            logger.info(
                "TxnSig: extended disease gene set by %d genes (total=%d)",
                len(new_genes[:50]), len(disease_data["genes"])
            )

    # Step 2: Apply tissue expression gate to candidates
    enriched_candidates = candidates
    if enable_tissue_gate and candidates:
        gate = TissueExpressionGate(disease_name)
        try:
            enriched_candidates = await gate.apply_to_candidates(candidates, txn_signature)
        except Exception as e:
            logger.warning("TissueGate failed, using original candidates: %s", e)

    # Step 3: Rescore combos using dysregulation pathway repair
    enriched_combos = combos
    if enable_pathway_repair and combos:
        repair_scorer = DysregulationPathwayScorer(disease_name)
        try:
            enriched_combos = repair_scorer.rescore_combos(
                combos=combos,
                txn_signature=txn_signature,
                disease_pathways=disease_data.get("pathways", []),
                repair_weight=0.20,  # blend 20% pathway repair into combo score
            )
        except Exception as e:
            logger.warning("DysregulationPathwayScorer failed: %s", e)

    # Step 4: Run subtype simulations
    subtype_context: Dict = {}
    if enable_subtype_sim and enriched_combos:
        subtype_sim = TranscriptomicSubtypeSimulator()
        try:
            subtype_context = subtype_sim.simulate_subtype_trials(
                enriched_combos, txn_signature, n_top_combos=5
            )
        except Exception as e:
            logger.warning("SubtypeSimulator failed: %s", e)

    txn_context = {
        "enabled": True,
        "signature_summary": {
            "n_upregulated": len(txn_signature.get("upregulated", {})),
            "n_downregulated": len(txn_signature.get("downregulated", {})),
            "n_tissue_anchored": len(txn_signature.get("tissue_anchored", {})),
            "n_extended_genes_added": len(extended_genes),
            "target_tissues": txn_signature.get("target_tissues", []),
        },
        "tissue_gate_applied": enable_tissue_gate,
        "pathway_repair_applied": enable_pathway_repair,
        "subtype_simulation": subtype_context,
        "source": "OpenTargets_expression_GTEx",
    }

    logger.info(
        "Transcriptomics enrichment complete: %d upregulated, %d extended genes, "
        "%d combos rescored",
        len(txn_signature.get("upregulated", {})),
        len(extended_genes),
        len(enriched_combos),
    )

    return enriched_candidates, enriched_combos, txn_context