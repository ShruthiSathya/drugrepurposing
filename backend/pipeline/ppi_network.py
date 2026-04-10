"""
ppi_network.py — Protein-Protein Interaction Network Proximity Scoring
=======================================================================
(Unchanged from v9.3 — no bugs found in this file. Included for completeness.)

The one thing clarified: STRING_MIN_SCORE=400 is passed as `required_score`
in the URL params (expects 0–1000 scale) and as `self.min_score / 1000.0` in
the local filter check (converts to 0–1 scale). This is intentional and correct.
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import aiohttp
import certifi
import ssl

logger = logging.getLogger(__name__)

STRING_API_BASE    = "https://string-db.org/api/json"
STRING_SPECIES     = 9606
STRING_MIN_SCORE   = 400
STRING_LIMIT       = 500

CACHE_DIR       = Path("/tmp/drug_repurposing_cache")
PPI_CACHE_FILE  = CACHE_DIR / "string_ppi_cache.json"
PPI_CACHE_TTL_DAYS = 7


class PPINetworkScorer:
    def __init__(self, min_score: int = STRING_MIN_SCORE):
        self.min_score = min_score
        self.graph: Dict[str, Dict[str, float]] = {}
        self._cache: Dict[str, List] = {}
        self._session: Optional[aiohttp.ClientSession] = None
        self._ssl_context = self._create_ssl_context()
        self._load_cache()

    def _create_ssl_context(self) -> ssl.SSLContext:
        try:
            return ssl.create_default_context(cafile=certifi.where())
        except Exception:
            return ssl.create_default_context()

    def _load_cache(self):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if PPI_CACHE_FILE.exists():
            try:
                data     = json.loads(PPI_CACHE_FILE.read_text())
                age_days = (time.time() - data.get("_cached_at", 0)) / 86400
                if age_days < PPI_CACHE_TTL_DAYS:
                    self._cache = data.get("interactions", {})
                    logger.info("Loaded STRING PPI cache (%d proteins, %.1f days old)",
                                len(self._cache), age_days)
                else:
                    logger.info("PPI cache expired (%.1f days) — will refresh", age_days)
                    self._cache = {}
            except Exception as e:
                logger.warning("Could not load PPI cache: %s", e)
                self._cache = {}

    def _save_cache(self):
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            PPI_CACHE_FILE.write_text(
                json.dumps({"_cached_at": time.time(), "interactions": self._cache})
            )
        except Exception as e:
            logger.warning("Could not save PPI cache: %s", e)

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector    = aiohttp.TCPConnector(ssl=self._ssl_context)
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=30),
            )
        return self._session

    async def _fetch_string_interactions(self, proteins: List[str]) -> List[Dict]:
        uncached = [p for p in proteins if p not in self._cache]
        if not uncached:
            interactions = []
            for p in proteins:
                interactions.extend(self._cache.get(p, []))
            return interactions

        session    = await self._get_session()
        batch_size = 100
        all_interactions = []

        for i in range(0, len(uncached), batch_size):
            batch = uncached[i:i + batch_size]
            try:
                url    = f"{STRING_API_BASE}/network"
                params = {
                    "identifiers":    "%0d".join(batch),
                    "species":        STRING_SPECIES,
                    "required_score": self.min_score,
                    "limit":          STRING_LIMIT,
                    "caller_identity": "drug_repurposing_research",
                }
                async with session.get(url, params=params) as resp:
                    if resp.status != 200:
                        logger.warning("STRING API returned %d for batch %d", resp.status, i)
                        continue
                    data = await resp.json(content_type=None)
                    if not isinstance(data, list):
                        continue

                    batch_interactions = [
                        {
                            "protein_a": item.get("preferredName_A", ""),
                            "protein_b": item.get("preferredName_B", ""),
                            "score":     item.get("score", 0),
                        }
                        for item in data
                        if item.get("score", 0) >= self.min_score / 1000.0
                    ]
                    all_interactions.extend(batch_interactions)

                    for protein in batch:
                        self._cache[protein] = [
                            ix for ix in batch_interactions
                            if ix["protein_a"] == protein or ix["protein_b"] == protein
                        ]

                    logger.debug("STRING: %d interactions for %d proteins",
                                 len(batch_interactions), len(batch))
                    await asyncio.sleep(0.2)

            except Exception as e:
                logger.warning("STRING fetch failed for batch %d: %s", i, e)
                for protein in batch:
                    if protein not in self._cache:
                        self._cache[protein] = []

        for p in proteins:
            if p not in uncached:
                all_interactions.extend(self._cache.get(p, []))

        self._save_cache()
        return all_interactions

    def _build_subgraph(self, interactions: List[Dict]):
        for ix in interactions:
            a = ix["protein_a"]
            b = ix["protein_b"]
            s = float(ix["score"])
            if a and b and a != b:
                if a not in self.graph:
                    self.graph[a] = {}
                if b not in self.graph:
                    self.graph[b] = {}
                self.graph[a][b] = max(self.graph[a].get(b, 0), s)
                self.graph[b][a] = max(self.graph[b].get(a, 0), s)

    def _bfs_shortest_paths(
        self, sources: Set[str], targets: Set[str], max_depth: int = 4,
    ) -> Dict[str, int]:
        distances: Dict[str, int] = {}
        visited   = set(sources)
        queue     = [(s, 0) for s in sources if s in self.graph]

        while queue:
            next_queue = []
            for node, depth in queue:
                if depth >= max_depth:
                    continue
                for neighbor in self.graph.get(node, {}):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        new_depth = depth + 1
                        if neighbor in targets:
                            distances[neighbor] = new_depth
                        next_queue.append((neighbor, new_depth))
            queue = next_queue

        return distances

    async def compute_proximity(
        self,
        drug_targets:  List[str],
        disease_genes: List[str],
        max_depth:     int = 4,
    ) -> Tuple[float, Dict]:
        evidence = {
            "direct_hits":        [],
            "nearest_neighbors":  [],
            "mean_shortest_path": None,
            "n_reachable":        0,
            "n_disease_genes":    len(disease_genes),
            "paths_found":        {},
            "ppi_score":          0.0,
            "note":               "",
        }

        if not drug_targets or not disease_genes:
            evidence["note"] = "No drug targets or disease genes provided"
            return 0.0, evidence

        direct_hits = set(drug_targets) & set(disease_genes)
        evidence["direct_hits"] = list(direct_hits)

        if not self.graph:
            all_proteins = list(set(drug_targets) | set(disease_genes))
            interactions = await self._fetch_string_interactions(all_proteins)
            self._build_subgraph(interactions)

            if not self.graph:
                if direct_hits:
                    score = len(direct_hits) / len(disease_genes)
                    evidence["note"]      = "STRING unavailable — using direct overlap only"
                    evidence["ppi_score"] = round(score, 4)
                    return round(score, 4), evidence
                evidence["note"] = "STRING unavailable and no direct overlap"
                return 0.0, evidence

        target_set  = set(drug_targets)
        disease_set = set(disease_genes)

        path_lengths: Dict[str, int] = {}
        for gene in direct_hits:
            path_lengths[gene] = 0

        remaining = disease_set - direct_hits
        if remaining:
            bfs_results = self._bfs_shortest_paths(
                sources=target_set, targets=remaining, max_depth=max_depth,
            )
            path_lengths.update(bfs_results)

        nearest = [g for g, d in path_lengths.items() if d == 1]
        evidence["nearest_neighbors"] = nearest
        evidence["paths_found"]       = path_lengths
        evidence["n_reachable"]       = len(path_lengths)

        if not path_lengths:
            evidence["note"] = "No paths found within max_depth"
            return 0.0, evidence

        total_distance = 0.0
        for gene in disease_genes:
            total_distance += path_lengths.get(gene, max_depth + 1)

        mean_path = total_distance / len(disease_genes)
        evidence["mean_shortest_path"] = round(mean_path, 3)

        score = round(min(1.0 / (1.0 + mean_path), 1.0), 4)
        evidence["ppi_score"] = score

        if direct_hits:
            evidence["note"] = (
                f"{len(direct_hits)} direct target-gene overlaps + "
                f"{len(nearest)} 1-hop neighbors"
            )
        else:
            evidence["note"] = f"No direct overlap; {len(nearest)} 1-hop neighbors found"

        logger.debug(
            "PPI proximity: mean_path=%.2f → score=%.3f "
            "(direct=%d, 1-hop=%d, reachable=%d/%d)",
            mean_path, score, len(direct_hits), len(nearest),
            len(path_lengths), len(disease_genes),
        )
        return score, evidence

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()


async def batch_ppi_scores(
    drugs_data:    List[Dict],
    disease_genes: List[str],
    scorer:        PPINetworkScorer,
) -> Dict[str, Tuple[float, Dict]]:
    all_proteins = set(disease_genes)
    for drug in drugs_data:
        all_proteins.update(drug.get("targets", []))

    interactions = await scorer._fetch_string_interactions(list(all_proteins))
    scorer._build_subgraph(interactions)
    logger.info(
        "STRING PPI graph: %d proteins, %d edges",
        len(scorer.graph),
        sum(len(v) for v in scorer.graph.values()) // 2,
    )

    results = {}
    for drug in drugs_data:
        targets = drug.get("targets", [])
        if targets:
            score, evidence = await scorer.compute_proximity(
                drug_targets=targets, disease_genes=disease_genes,
            )
        else:
            score   = 0.0
            evidence = {"ppi_score": 0.0, "note": "No drug targets"}
        results[drug["name"]] = (score, evidence)

    return results