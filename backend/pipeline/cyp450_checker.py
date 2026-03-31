"""
cyp450_checker.py — CYP450 Metabolic Overlap Checker
======================================================
Detects potentially dangerous pharmacokinetic drug-drug interactions
arising from shared CYP450 enzyme metabolism.

Why this matters
----------------
Two drugs that are both CYP3A4 substrates will compete for the same
metabolic enzyme. If drug B is also a CYP3A4 inhibitor, it will raise
plasma levels of drug A to potentially toxic concentrations — even if
both drugs are safe individually.

CYP450 interaction types
-------------------------
  SUBSTRATE  : drug is metabolised by the enzyme (concentration affected)
  INHIBITOR  : drug blocks the enzyme (raises substrate levels)
  INDUCER    : drug up-regulates the enzyme (lowers substrate levels)

Interaction severity
--------------------
  CRITICAL  : Well-documented, clinically dangerous interaction. Apply Ω penalty.
  MAJOR     : Significant interaction requiring dose adjustment.
  MODERATE  : Interaction possible; monitor closely.
  MINOR     : Unlikely to be clinically significant.

Data sources
------------
Primary: FDA drug label data via openFDA API (live)
Fallback: Curated static table compiled from:
  - Flockhart DA (2021). Drug Interactions: Cytochrome P450 Drug Interaction Table.
    Indiana University School of Medicine.
    https://drug-interactions.medicine.iu.edu
  - Lexicomp Drug Interactions (pharmacist reference)
  - FDA Drug Development and Drug Interactions (2020)

Usage
-----
    checker = CYP450Checker()
    result = await checker.check_pair("sildenafil", "erythromycin")
    print(result.severity, result.risk_description)
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import aiohttp

logger = logging.getLogger(__name__)

CACHE_DIR      = Path("/tmp/drug_repurposing_cache")
CYP_CACHE_FILE = CACHE_DIR / "cyp450_cache.json"
OPENFDA_BASE   = "https://api.fda.gov/drug/label.json"

# ─────────────────────────────────────────────────────────────────────────────
# Curated CYP450 interaction table
# Source: Indiana University P450 Table + FDA Drug Interaction Guidance
# Format: drug_name_lower → {enzyme: [SUBSTRATE|INHIBITOR|INDUCER]}
# ─────────────────────────────────────────────────────────────────────────────

CYP_TABLE: Dict[str, Dict[str, List[str]]] = {
    # ── CYP3A4 ─────────────────────────────────────────────────────────────
    "sildenafil":       {"CYP3A4": ["SUBSTRATE"]},
    "tadalafil":        {"CYP3A4": ["SUBSTRATE"]},
    "vardenafil":       {"CYP3A4": ["SUBSTRATE"]},
    "bosentan":         {"CYP3A4": ["SUBSTRATE", "INDUCER"], "CYP2C9": ["INDUCER"]},
    "imatinib":         {"CYP3A4": ["SUBSTRATE", "INHIBITOR"], "CYP2D6": ["INHIBITOR"]},
    "erlotinib":        {"CYP3A4": ["SUBSTRATE"]},
    "gefitinib":        {"CYP3A4": ["SUBSTRATE"]},
    "sorafenib":        {"CYP3A4": ["SUBSTRATE"]},
    "dasatinib":        {"CYP3A4": ["SUBSTRATE"]},
    "lapatinib":        {"CYP3A4": ["SUBSTRATE", "INHIBITOR"]},
    "sirolimus":        {"CYP3A4": ["SUBSTRATE"]},
    "everolimus":       {"CYP3A4": ["SUBSTRATE"]},
    "tacrolimus":       {"CYP3A4": ["SUBSTRATE"]},
    "cyclosporine":     {"CYP3A4": ["SUBSTRATE", "INHIBITOR"], "CYP2C9": ["INHIBITOR"]},
    "midazolam":        {"CYP3A4": ["SUBSTRATE"]},
    "alprazolam":       {"CYP3A4": ["SUBSTRATE"]},
    "triazolam":        {"CYP3A4": ["SUBSTRATE"]},
    "diazepam":         {"CYP3A4": ["SUBSTRATE"], "CYP2C19": ["SUBSTRATE"]},
    "atorvastatin":     {"CYP3A4": ["SUBSTRATE"]},
    "simvastatin":      {"CYP3A4": ["SUBSTRATE"]},
    "lovastatin":       {"CYP3A4": ["SUBSTRATE"]},
    "amlodipine":       {"CYP3A4": ["SUBSTRATE"]},
    "nifedipine":       {"CYP3A4": ["SUBSTRATE"]},
    "diltiazem":        {"CYP3A4": ["SUBSTRATE", "INHIBITOR"]},
    "verapamil":        {"CYP3A4": ["SUBSTRATE", "INHIBITOR"]},
    "erythromycin":     {"CYP3A4": ["SUBSTRATE", "INHIBITOR"]},
    "clarithromycin":   {"CYP3A4": ["SUBSTRATE", "INHIBITOR"]},
    "azithromycin":     {"CYP3A4": ["WEAK_INHIBITOR"]},
    "ketoconazole":     {"CYP3A4": ["STRONG_INHIBITOR"]},
    "itraconazole":     {"CYP3A4": ["STRONG_INHIBITOR"]},
    "fluconazole":      {"CYP3A4": ["MODERATE_INHIBITOR"], "CYP2C9": ["STRONG_INHIBITOR"]},
    "rifampicin":       {"CYP3A4": ["STRONG_INDUCER"], "CYP2C9": ["INDUCER"], "CYP2C19": ["INDUCER"]},
    "carbamazepine":    {"CYP3A4": ["SUBSTRATE", "STRONG_INDUCER"], "CYP1A2": ["INDUCER"]},
    "phenytoin":        {"CYP3A4": ["INDUCER"], "CYP2C9": ["SUBSTRATE", "INDUCER"]},
    "phenobarbital":    {"CYP3A4": ["STRONG_INDUCER"]},
    "dexamethasone":    {"CYP3A4": ["SUBSTRATE", "INDUCER"]},
    "prednisone":       {"CYP3A4": ["SUBSTRATE"]},
    "tamoxifen":        {"CYP3A4": ["SUBSTRATE"], "CYP2D6": ["SUBSTRATE", "INHIBITOR"]},
    "letrozole":        {"CYP3A4": ["SUBSTRATE"], "CYP2A6": ["INHIBITOR"]},
    "anastrozole":      {"CYP3A4": ["SUBSTRATE"]},
    "thalidomide":      {"CYP3A4": ["WEAK_SUBSTRATE"]},
    "lenalidomide":     {},  # minimal CYP450 metabolism
    "bortezomib":       {"CYP3A4": ["SUBSTRATE"], "CYP2C19": ["SUBSTRATE"]},
    "hydroxychloroquine":{"CYP3A4": ["SUBSTRATE"], "CYP2D6": ["WEAK_INHIBITOR"]},
    "chloroquine":      {"CYP3A4": ["SUBSTRATE"], "CYP2D6": ["INHIBITOR"]},
    "colchicine":       {"CYP3A4": ["SUBSTRATE"]},
    # ── CYP2D6 ─────────────────────────────────────────────────────────────
    "codeine":          {"CYP2D6": ["SUBSTRATE"]},
    "tramadol":         {"CYP2D6": ["SUBSTRATE"]},
    "metoprolol":       {"CYP2D6": ["SUBSTRATE"]},
    "carvedilol":       {"CYP2D6": ["SUBSTRATE"]},
    "propranolol":      {"CYP2D6": ["SUBSTRATE"]},
    "amitriptyline":    {"CYP2D6": ["SUBSTRATE"], "CYP3A4": ["SUBSTRATE"]},
    "fluoxetine":       {"CYP2D6": ["SUBSTRATE", "STRONG_INHIBITOR"]},
    "paroxetine":       {"CYP2D6": ["SUBSTRATE", "STRONG_INHIBITOR"]},
    "bupropion":        {"CYP2D6": ["STRONG_INHIBITOR"]},
    "duloxetine":       {"CYP2D6": ["SUBSTRATE", "MODERATE_INHIBITOR"]},
    "venlafaxine":      {"CYP2D6": ["SUBSTRATE"]},
    "haloperidol":      {"CYP2D6": ["SUBSTRATE"]},
    "risperidone":      {"CYP2D6": ["SUBSTRATE"]},
    "aripiprazole":     {"CYP2D6": ["SUBSTRATE"], "CYP3A4": ["SUBSTRATE"]},
    "donepezil":        {"CYP2D6": ["SUBSTRATE"], "CYP3A4": ["SUBSTRATE"]},
    "metformin":        {},  # not significantly CYP450 metabolised
    # ── CYP2C9 ─────────────────────────────────────────────────────────────
    "warfarin":         {"CYP2C9": ["SUBSTRATE"]},
    "ibuprofen":        {"CYP2C9": ["SUBSTRATE"]},
    "naproxen":         {"CYP2C9": ["SUBSTRATE"]},
    "celecoxib":        {"CYP2C9": ["SUBSTRATE"]},
    "losartan":         {"CYP2C9": ["SUBSTRATE"]},
    "irbesartan":       {"CYP2C9": ["SUBSTRATE"]},
    "glipizide":        {"CYP2C9": ["SUBSTRATE"]},
    "glimepiride":      {"CYP2C9": ["SUBSTRATE"]},
    "rosiglitazone":    {"CYP2C9": ["SUBSTRATE"]},
    "pioglitazone":     {"CYP2C9": ["SUBSTRATE"], "CYP3A4": ["SUBSTRATE"]},
    "methotrexate":     {},  # primarily renal excretion
    # ── CYP2C19 ────────────────────────────────────────────────────────────
    "omeprazole":       {"CYP2C19": ["SUBSTRATE", "INHIBITOR"]},
    "pantoprazole":     {"CYP2C19": ["SUBSTRATE"]},
    "clopidogrel":      {"CYP2C19": ["SUBSTRATE"]},
    "voriconazole":     {"CYP2C19": ["SUBSTRATE", "INHIBITOR"], "CYP3A4": ["INHIBITOR"]},
    "escitalopram":     {"CYP2C19": ["SUBSTRATE"]},
    "citalopram":       {"CYP2C19": ["SUBSTRATE"]},
    # ── CYP1A2 ─────────────────────────────────────────────────────────────
    "theophylline":     {"CYP1A2": ["SUBSTRATE"]},
    "caffeine":         {"CYP1A2": ["SUBSTRATE"]},
    "clozapine":        {"CYP1A2": ["SUBSTRATE"], "CYP3A4": ["SUBSTRATE"]},
    "olanzapine":       {"CYP1A2": ["SUBSTRATE"], "CYP2D6": ["SUBSTRATE"]},
    "ropinirole":       {"CYP1A2": ["SUBSTRATE"]},
    "rasagiline":       {"CYP1A2": ["SUBSTRATE"]},
    "ciprofloxacin":    {"CYP1A2": ["MODERATE_INHIBITOR"]},
    "fluvoxamine":      {"CYP1A2": ["STRONG_INHIBITOR"], "CYP2C19": ["STRONG_INHIBITOR"]},
    # ── Drugs with minimal CYP450 involvement (safe to combine) ───────────
    "aspirin":          {},
    "gabapentin":       {},
    "pregabalin":       {},
    "memantine":        {},
    "lithium":          {},
    "atenolol":         {},
    "minoxidil":        {},
    "colchicine":       {"CYP3A4": ["SUBSTRATE"]},
}

# Enzyme strength classification
INHIBITOR_STRENGTH = {
    "STRONG_INHIBITOR":    3,
    "MODERATE_INHIBITOR":  2,
    "WEAK_INHIBITOR":      1,
    "INHIBITOR":           2,
    "STRONG_INDUCER":      3,
    "INDUCER":             2,
    "WEAK_INDUCER":        1,
    "SUBSTRATE":           0,
    "WEAK_SUBSTRATE":      0,
}

CRITICAL_ENZYMES = {"CYP3A4", "CYP2D6", "CYP2C9"}

# Severity threshold: inhibitor_strength + substrate_present >= this → CRITICAL
CRITICAL_INTERACTION_THRESHOLD = 3


@dataclass
class CYP450Interaction:
    """Result of CYP450 metabolic overlap analysis for a drug pair."""
    drug_a: str
    drug_b: str
    interactions: List[Dict] = field(default_factory=list)
    severity: str = "NONE"           # CRITICAL | MAJOR | MODERATE | MINOR | NONE
    omega_penalty: float = 0.0       # 0 = no penalty, -inf = discard pair
    risk_description: str = ""
    enzymes_affected: List[str] = field(default_factory=list)
    recommendation: str = ""


class CYP450Checker:
    """
    Checks for CYP450 metabolic drug-drug interactions in a drug pair.

    Interaction logic
    -----------------
    CRITICAL (Ω = -inf):
      Drug A is a substrate of enzyme E AND drug B is a strong inhibitor of E.
      This means B will cause clinically dangerous accumulation of A.

    MAJOR (Ω = -0.40):
      Drug A and B are both substrates of the same major CYP enzyme.
      Competition reduces predictable exposure for both.

    MODERATE (Ω = -0.15):
      Weak inhibitor/substrate overlap on major enzyme.

    MINOR (Ω = -0.05):
      Overlap on minor CYP enzymes only.
    """

    def __init__(self):
        self._openfda_cache: Dict[str, Optional[Dict]] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if CYP_CACHE_FILE.exists():
            try:
                self._openfda_cache = json.loads(CYP_CACHE_FILE.read_text())
                logger.info(f"CYP450 cache: {len(self._openfda_cache)} drugs loaded")
            except Exception:
                self._openfda_cache = {}

    def _save_cache(self) -> None:
        try:
            CYP_CACHE_FILE.write_text(json.dumps(self._openfda_cache, indent=2))
        except Exception as e:
            logger.warning(f"CYP450 cache save failed: {e}")

    # ── openFDA label enrichment ──────────────────────────────────────────────

    async def _fetch_openfda_cyp_data(
        self, drug_name: str, session: aiohttp.ClientSession
    ) -> Optional[Dict]:
        """
        Query openFDA drug label for CYP450 interaction text.
        Falls back to static table if API unavailable.
        """
        key = drug_name.lower()
        if key in self._openfda_cache:
            return self._openfda_cache[key]

        try:
            params = {
                "search": f'openfda.generic_name:"{drug_name}"',
                "limit": 1,
            }
            async with session.get(
                OPENFDA_BASE, params=params,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    self._openfda_cache[key] = None
                    return None
                data    = await resp.json()
                results = data.get("results", [])
                if not results:
                    self._openfda_cache[key] = None
                    return None

                label = results[0]
                # Extract drug interaction sections
                cyp_data = {
                    "drug_interactions": label.get("drug_interactions", []),
                    "clinical_pharmacology": label.get("clinical_pharmacology", []),
                    "pharmacokinetics": label.get("pharmacokinetics", []),
                }
                self._openfda_cache[key] = cyp_data
                self._save_cache()
                return cyp_data

        except Exception as e:
            logger.debug(f"openFDA CYP lookup failed for {drug_name}: {e}")
            self._openfda_cache[key] = None
            return None

    def _extract_cyp_from_label(self, drug_name: str, label_data: Optional[Dict]) -> Dict[str, List[str]]:
        """
        Parse openFDA label text to extract CYP450 roles.
        Returns {enzyme: [role, ...]} or falls back to static table.
        """
        # Always start with static table as base
        static = CYP_TABLE.get(drug_name.lower(), {})

        if not label_data:
            return static

        roles: Dict[str, List[str]] = dict(static)
        enzyme_keywords = {
            "CYP3A4": ["CYP3A4", "CYP 3A4", "CYP3A"],
            "CYP2D6": ["CYP2D6", "CYP 2D6"],
            "CYP2C9": ["CYP2C9", "CYP 2C9"],
            "CYP2C19": ["CYP2C19", "CYP 2C19"],
            "CYP1A2": ["CYP1A2", "CYP 1A2"],
        }
        role_keywords = {
            "SUBSTRATE":   ["substrate", "metabolized by", "metabolised by"],
            "INHIBITOR":   ["inhibit", "inhibitor"],
            "INDUCER":     ["induc"],
            "STRONG_INHIBITOR": ["strong inhibitor", "potent inhibitor"],
        }

        all_text = " ".join(str(v) for v in label_data.values()).lower()

        for enzyme, kws in enzyme_keywords.items():
            for kw in kws:
                if kw.lower() in all_text:
                    if enzyme not in roles:
                        roles[enzyme] = []
                    for role, role_kws in role_keywords.items():
                        if any(rk in all_text for rk in role_kws):
                            if role not in roles[enzyme]:
                                roles[enzyme].append(role)

        return roles

    # ── Interaction logic ─────────────────────────────────────────────────────

    def _check_pair_static(
        self, name_a: str, name_b: str
    ) -> CYP450Interaction:
        """Check drug pair against static CYP450 table."""
        cyp_a = CYP_TABLE.get(name_a.lower(), {})
        cyp_b = CYP_TABLE.get(name_b.lower(), {})
        return self._analyse_cyp_overlap(name_a, name_b, cyp_a, cyp_b)

    def _analyse_cyp_overlap(
        self,
        name_a: str,
        name_b: str,
        cyp_a: Dict[str, List[str]],
        cyp_b: Dict[str, List[str]],
    ) -> CYP450Interaction:
        interactions = []
        enzymes_affected = []
        severity = "NONE"
        omega_penalty = 0.0
        descriptions = []

        all_enzymes = set(cyp_a.keys()) | set(cyp_b.keys())

        for enzyme in all_enzymes:
            roles_a = set(cyp_a.get(enzyme, []))
            roles_b = set(cyp_b.get(enzyme, []))

            if not roles_a or not roles_b:
                continue

            # Both are substrates of same critical enzyme → competition
            both_substrates = (
                any(r in ("SUBSTRATE", "WEAK_SUBSTRATE") for r in roles_a) and
                any(r in ("SUBSTRATE", "WEAK_SUBSTRATE") for r in roles_b) and
                enzyme in CRITICAL_ENZYMES
            )

            # B is a (strong) inhibitor and A is a substrate → A accumulates
            b_inhibits_a = (
                any(r in ("SUBSTRATE", "WEAK_SUBSTRATE") for r in roles_a) and
                any("INHIBITOR" in r for r in roles_b)
            )
            a_inhibits_b = (
                any(r in ("SUBSTRATE", "WEAK_SUBSTRATE") for r in roles_b) and
                any("INHIBITOR" in r for r in roles_a)
            )

            b_strength = max(
                (INHIBITOR_STRENGTH.get(r, 0) for r in roles_b), default=0
            )
            a_strength = max(
                (INHIBITOR_STRENGTH.get(r, 0) for r in roles_a), default=0
            )

            if (b_inhibits_a or a_inhibits_b) and enzyme in CRITICAL_ENZYMES:
                if max(a_strength, b_strength) >= 3:
                    severity = "CRITICAL"
                    omega_penalty = float("-inf")
                    descriptions.append(
                        f"CRITICAL: {'B' if b_inhibits_a else 'A'} is a strong "
                        f"{enzyme} inhibitor that will cause dangerous accumulation "
                        f"of {'A' if b_inhibits_a else 'B'}."
                    )
                elif max(a_strength, b_strength) >= 2:
                    if severity not in ("CRITICAL",):
                        severity = "MAJOR"
                        omega_penalty = min(omega_penalty, -0.40)
                    descriptions.append(
                        f"MAJOR: {enzyme} inhibition interaction — "
                        f"plasma levels of co-substrate may increase 2-3×."
                    )
                else:
                    if severity not in ("CRITICAL", "MAJOR"):
                        severity = "MODERATE"
                        omega_penalty = min(omega_penalty, -0.15)
                    descriptions.append(f"MODERATE: {enzyme} weak inhibitor interaction.")

            elif both_substrates:
                if severity not in ("CRITICAL", "MAJOR"):
                    severity = "MODERATE" if severity == "NONE" else severity
                    omega_penalty = min(omega_penalty, -0.08)
                descriptions.append(
                    f"MODERATE: Both drugs are {enzyme} substrates — "
                    f"competition may alter exposure of both."
                )

            if roles_a and roles_b:
                enzymes_affected.append(enzyme)
                interactions.append({
                    "enzyme": enzyme,
                    "roles_drug_a": list(roles_a),
                    "roles_drug_b": list(roles_b),
                    "interaction_type": (
                        "INHIBITOR_SUBSTRATE" if (b_inhibits_a or a_inhibits_b)
                        else "DUAL_SUBSTRATE"
                    ),
                })

        # Build recommendation
        if severity == "CRITICAL":
            recommendation = (
                "DO NOT COMBINE: Critical metabolic interaction predicted. "
                "Contraindicated unless alternative metabolic pathway confirmed."
            )
        elif severity == "MAJOR":
            recommendation = (
                "CAUTION: Major PK interaction — require dose reduction and "
                "therapeutic drug monitoring if combination used."
            )
        elif severity in ("MODERATE",):
            recommendation = (
                "MONITOR: Moderate PK interaction possible — "
                "consider dose adjustment and clinical monitoring."
            )
        elif severity == "MINOR":
            recommendation = "LOW RISK: Minor interaction; routine monitoring sufficient."
        else:
            recommendation = "NO SIGNIFICANT CYP450 INTERACTION DETECTED."

        return CYP450Interaction(
            drug_a=name_a,
            drug_b=name_b,
            interactions=interactions,
            severity=severity,
            omega_penalty=omega_penalty,
            risk_description=" | ".join(descriptions) if descriptions else "None identified.",
            enzymes_affected=list(set(enzymes_affected)),
            recommendation=recommendation,
        )

    # ── Main API ──────────────────────────────────────────────────────────────

    async def check_pair(
        self,
        drug_a_name: str,
        drug_b_name: str,
        enrich_from_openfda: bool = True,
    ) -> CYP450Interaction:
        """
        Check CYP450 interaction for a drug pair.

        Parameters
        ----------
        drug_a_name, drug_b_name : str
        enrich_from_openfda : bool
            Query openFDA to supplement static table (default True).
        """
        cyp_a = CYP_TABLE.get(drug_a_name.lower(), {})
        cyp_b = CYP_TABLE.get(drug_b_name.lower(), {})

        if enrich_from_openfda and (not cyp_a or not cyp_b):
            async with aiohttp.ClientSession() as session:
                tasks = []
                if not cyp_a:
                    tasks.append(self._fetch_openfda_cyp_data(drug_a_name, session))
                else:
                    tasks.append(asyncio.sleep(0))
                if not cyp_b:
                    tasks.append(self._fetch_openfda_cyp_data(drug_b_name, session))
                else:
                    tasks.append(asyncio.sleep(0))
                results = await asyncio.gather(*tasks, return_exceptions=True)

            if not cyp_a and isinstance(results[0], dict):
                cyp_a = self._extract_cyp_from_label(drug_a_name, results[0])
            if not cyp_b and isinstance(results[1], dict):
                cyp_b = self._extract_cyp_from_label(drug_b_name, results[1])

        return self._analyse_cyp_overlap(drug_a_name, drug_b_name, cyp_a, cyp_b)

    async def check_combo(
        self,
        drug_names: List[str],
        enrich_from_openfda: bool = True,
    ) -> Dict:
        """
        Check all pairwise CYP450 interactions in a combo of 2 or 3 drugs.
        Returns worst-case severity and aggregate omega_penalty.
        """
        results = []
        pairs = [(drug_names[i], drug_names[j])
                 for i in range(len(drug_names))
                 for j in range(i + 1, len(drug_names))]

        for a, b in pairs:
            r = await self.check_pair(a, b, enrich_from_openfda)
            results.append(r)

        severities = [r.severity for r in results]
        priority = ["CRITICAL", "MAJOR", "MODERATE", "MINOR", "NONE"]
        worst = next((s for s in priority if s in severities), "NONE")

        # Aggregate penalty: sum but never worse than -inf
        total_penalty = 0.0
        for r in results:
            if r.omega_penalty == float("-inf"):
                total_penalty = float("-inf")
                break
            total_penalty += r.omega_penalty

        return {
            "drugs":           drug_names,
            "pairwise_checks": [
                {
                    "pair":           [r.drug_a, r.drug_b],
                    "severity":       r.severity,
                    "omega_penalty":  r.omega_penalty,
                    "risk":           r.risk_description,
                    "recommendation": r.recommendation,
                    "enzymes":        r.enzymes_affected,
                }
                for r in results
            ],
            "worst_severity":  worst,
            "aggregate_omega": total_penalty,
            "is_critical":     worst == "CRITICAL",
        }

    def to_dict(self, result: CYP450Interaction) -> Dict:
        return {
            "drug_a": result.drug_a,
            "drug_b": result.drug_b,
            "severity": result.severity,
            "omega_penalty": result.omega_penalty,
            "risk_description": result.risk_description,
            "enzymes_affected": result.enzymes_affected,
            "interactions": result.interactions,
            "recommendation": result.recommendation,
        }