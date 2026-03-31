"""
efo_ontology.py — EFO Ontology Expander
========================================
Walks the EFO (Experimental Factor Ontology) disease tree to pull in genes
from related conditions, widening the disease gene set without hardcoding
any drug-disease pairs.

WHY EFO EXPANSION
-----------------
OpenTargets disease gene associations are EFO-term specific. "Pulmonary
arterial hypertension" (EFO:0001361) has its own gene set, but many relevant
genes are only annotated at:
  - parent terms:   "hypertensive disorder" / "vascular disease"
  - sibling terms:  "pulmonary hypertension" / "familial PAH"
  - child terms:    "idiopathic PAH" / "heritable PAH"

Walking the ontology tree ±1–2 levels and merging gene sets substantially
improves recall for rare diseases (often have sparse direct annotations)
and complex diseases with heterogeneous subtypes.

API
---
POST https://api.platform.opentargets.org/api/v4/graphql
  → disease.ancestors (EFO parent chain)
  → disease.descendants (more specific subtypes)
  → disease.associatedTargets (gene associations per node)

Usage
-----
    expander = EFOOntologyExpander(session=aiohttp_session)
    disease_data = await expander.expand_disease_genes(disease_data)
    print(disease_data["efo_expansion_stats"])

    # or standalone:
    expander = EFOOntologyExpander()
    await expander.initialize()
    expanded = await expander.expand_disease_genes(disease_data)
    await expander.close()
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import aiohttp
import ssl
import certifi

logger = logging.getLogger(__name__)

CACHE_DIR      = Path("/tmp/drug_repurposing_cache")
EFO_CACHE_FILE = CACHE_DIR / "efo_ontology_cache.json"
CACHE_TTL_SECS = 7 * 24 * 3600   # 1 week

OT_GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"

# Maximum tree depth to walk (1 = parents + children of the query disease only)
DEFAULT_ANCESTOR_DEPTH   = 1
DEFAULT_DESCENDANT_DEPTH = 2
# Maximum additional genes to add from expansion (prevents score dilution)
MAX_EXPANSION_GENES      = 100
# Minimum OpenTargets association score for expanded genes
MIN_EXPANSION_SCORE      = 0.08


class EFOOntologyExpander:
    """
    Expands a disease's associated gene set by walking the EFO ontology tree.

    The expander queries OpenTargets for:
      1. The disease's ancestors (parent / grandparent EFO terms)
      2. The disease's descendants (child / grandchild EFO terms)

    For each related disease node, it fetches associated genes and merges them
    into the original gene set, down-weighting by ontology distance so distant
    relatives contribute less.

    Distance weighting:
      distance 0 (query disease)    → weight 1.00
      distance 1 (direct parent/child) → weight 0.60
      distance 2 (grandparent/grandchild) → weight 0.35
      distance 3+                    → weight 0.15 (not fetched by default)
    """

    DISTANCE_WEIGHTS = {0: 1.00, 1: 0.60, 2: 0.35, 3: 0.15}

    def __init__(
        self,
        session: Optional[aiohttp.ClientSession] = None,
        ancestor_depth:   int = DEFAULT_ANCESTOR_DEPTH,
        descendant_depth: int = DEFAULT_DESCENDANT_DEPTH,
    ):
        self._external_session = session
        self._session:     Optional[aiohttp.ClientSession] = None
        self._cache:       Dict = {}
        self._ssl_context  = self._make_ssl_context()
        self.ancestor_depth   = ancestor_depth
        self.descendant_depth = descendant_depth
        self._load_cache()

    def _make_ssl_context(self) -> ssl.SSLContext:
        try:
            return ssl.create_default_context(cafile=certifi.where())
        except Exception:
            return ssl.create_default_context()

    def _load_cache(self) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if EFO_CACHE_FILE.exists():
            try:
                raw = json.loads(EFO_CACHE_FILE.read_text())
                now = time.time()
                self._cache = {
                    k: v for k, v in raw.items()
                    if v.get("_cached_at", 0) + CACHE_TTL_SECS > now
                }
                logger.debug(f"EFO cache: {len(self._cache)} disease nodes loaded")
            except Exception as e:
                logger.warning(f"EFO cache load failed: {e}")
                self._cache = {}

    def _save_cache(self) -> None:
        try:
            EFO_CACHE_FILE.write_text(json.dumps(self._cache, indent=2))
        except Exception as e:
            logger.warning(f"EFO cache save failed: {e}")

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._external_session and not self._external_session.closed:
            return self._external_session
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(ssl=self._ssl_context)
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=30),
            )
        return self._session

    # ── GraphQL queries ───────────────────────────────────────────────────────

    async def _fetch_disease_ontology(self, efo_id: str) -> Optional[Dict]:
        """Fetch ancestors and descendants for a disease EFO ID."""
        if efo_id in self._cache:
            return self._cache[efo_id].get("ontology")

        query = """
        query DiseaseOntology($efoId: String!) {
          disease(efoId: $efoId) {
            id
            name
            ancestors
            descendants
          }
        }
        """
        session = await self._get_session()
        try:
            async with session.post(
                OT_GRAPHQL_URL,
                json={"query": query, "variables": {"efoId": efo_id}},
                headers={"Content-Type": "application/json"},
            ) as resp:
                if resp.status != 200:
                    return None
                data    = await resp.json()
                disease = data.get("data", {}).get("disease")
                if not disease:
                    return None

                ontology = {
                    "id":          disease["id"],
                    "name":        disease.get("name", ""),
                    "ancestors":   disease.get("ancestors", []),
                    "descendants": disease.get("descendants", []),
                }
                self._cache[efo_id] = {
                    "ontology":    ontology,
                    "_cached_at":  time.time(),
                }
                self._save_cache()
                return ontology

        except Exception as e:
            logger.debug(f"EFO ontology fetch failed for {efo_id}: {e}")
            return None

    async def _fetch_disease_genes(
        self, efo_id: str, limit: int = 100
    ) -> List[Dict]:
        """Fetch top associated genes for a disease EFO ID."""
        cache_key = f"{efo_id}_genes"
        if cache_key in self._cache:
            return self._cache[cache_key].get("genes", [])

        query = """
        query DiseaseGenes($efoId: String!, $size: Int!) {
          disease(efoId: $efoId) {
            associatedTargets(page: {index: 0, size: $size}) {
              rows {
                target { approvedSymbol }
                score
              }
            }
          }
        }
        """
        session = await self._get_session()
        try:
            async with session.post(
                OT_GRAPHQL_URL,
                json={"query": query, "variables": {"efoId": efo_id, "size": limit}},
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    return []
                data    = await resp.json()
                disease = data.get("data", {}).get("disease")
                if not disease:
                    return []

                rows = disease.get("associatedTargets", {}).get("rows", []) or []
                genes = []
                for row in rows:
                    symbol = (row.get("target") or {}).get("approvedSymbol")
                    score  = row.get("score", 0)
                    if symbol and score >= MIN_EXPANSION_SCORE:
                        genes.append({"symbol": symbol, "score": float(score)})

                self._cache[cache_key] = {"genes": genes, "_cached_at": time.time()}
                self._save_cache()
                return genes

        except Exception as e:
            logger.debug(f"EFO gene fetch failed for {efo_id}: {e}")
            return []

    # ── Main expansion ────────────────────────────────────────────────────────

    async def expand_disease_genes(self, disease_data: Dict) -> Dict:
        """
        Expand a disease's gene set by walking the EFO ontology tree.

        Parameters
        ----------
        disease_data : dict
            Output of data_fetcher.fetch_disease_data(). Must have 'id' and 'genes'.

        Returns
        -------
        disease_data enriched with:
            genes           : expanded gene list (original + neighbours)
            gene_scores     : updated score dict (neighbours down-weighted)
            efo_expansion_stats : summary of expansion
        """
        efo_id = disease_data.get("id", "")
        if not efo_id:
            logger.warning("EFO expansion: no disease ID available")
            return disease_data

        original_genes:  List[str]         = list(disease_data.get("genes", []))
        original_scores: Dict[str, float]  = dict(disease_data.get("gene_scores", {}))
        original_set:    Set[str]          = set(original_genes)

        # Fetch ontology tree
        ontology = await self._fetch_disease_ontology(efo_id)
        if not ontology:
            logger.warning(f"EFO expansion: no ontology data for {efo_id}")
            disease_data["efo_expansion_stats"] = {
                "original_gene_count": len(original_genes),
                "total_gene_count":    len(original_genes),
                "genes_added":         0,
                "related_nodes_fetched": 0,
                "status":              "ontology_unavailable",
            }
            return disease_data

        # Build list of related EFO IDs and their distances
        related: Dict[str, int] = {}  # {efo_id: distance}

        # Ancestors (parents, grandparents)
        for i, ancestor_id in enumerate(ontology["ancestors"][:self.ancestor_depth * 3]):
            dist = min(i + 1, self.ancestor_depth)
            related[ancestor_id] = dist

        # Descendants (children, grandchildren)
        for i, desc_id in enumerate(ontology["descendants"][:self.descendant_depth * 5]):
            dist = min(i // 3 + 1, self.descendant_depth)
            related[desc_id] = min(related.get(desc_id, 99), dist)

        if not related:
            disease_data["efo_expansion_stats"] = {
                "original_gene_count":   len(original_genes),
                "total_gene_count":      len(original_genes),
                "genes_added":           0,
                "related_nodes_fetched": 0,
                "status":                "no_related_nodes",
            }
            return disease_data

        # Fetch genes for related nodes concurrently (max 5 concurrent)
        semaphore = asyncio.Semaphore(5)

        async def fetch_one(node_id: str) -> Dict[str, List]:
            async with semaphore:
                genes = await self._fetch_disease_genes(node_id, limit=80)
                return {"id": node_id, "genes": genes}

        fetched = await asyncio.gather(
            *[fetch_one(nid) for nid in related],
            return_exceptions=True,
        )

        # Merge gene sets
        added_genes: Dict[str, float] = {}  # {symbol: best_weighted_score}

        for result in fetched:
            if isinstance(result, Exception):
                continue
            node_id  = result["id"]
            distance = related.get(node_id, 2)
            weight   = self.DISTANCE_WEIGHTS.get(distance, 0.15)

            for gene_info in result["genes"]:
                symbol = gene_info["symbol"]
                if symbol in original_set:
                    continue  # already in original set
                weighted = gene_info["score"] * weight
                if weighted > added_genes.get(symbol, 0.0):
                    added_genes[symbol] = weighted

        # Sort added genes by weighted score and take top MAX_EXPANSION_GENES
        sorted_additions = sorted(
            added_genes.items(), key=lambda x: x[1], reverse=True
        )[:MAX_EXPANSION_GENES]

        new_genes   = [sym for sym, _ in sorted_additions]
        new_scores  = {sym: round(score, 4) for sym, score in sorted_additions}

        # Merge
        expanded_genes  = original_genes + new_genes
        expanded_scores = {**original_scores, **new_scores}

        disease_data["genes"]       = expanded_genes
        disease_data["gene_scores"] = expanded_scores
        disease_data["efo_expansion_stats"] = {
            "original_gene_count":   len(original_genes),
            "total_gene_count":      len(expanded_genes),
            "genes_added":           len(new_genes),
            "related_nodes_fetched": len([r for r in fetched if not isinstance(r, Exception)]),
            "ancestor_nodes":        len([nid for nid, d in related.items() if d <= self.ancestor_depth]),
            "descendant_nodes":      len([nid for nid, d in related.items() if d > self.ancestor_depth]),
            "status":                "ok",
        }

        logger.info(
            "EFO expansion: %s → %d genes (added %d from %d related nodes)",
            disease_data.get("name", efo_id),
            len(expanded_genes),
            len(new_genes),
            len(related),
        )

        return disease_data

    async def close(self) -> None:
        """Close the internal session if we created it."""
        self._save_cache()
        if self._session and not self._session.closed:
            await self._session.close()