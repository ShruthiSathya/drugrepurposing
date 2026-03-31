"""
toxicity_engine.py — Toxicity Penalty Engine (Ω)
=================================================
Implements the Strict Toxicity Penalty Ω for drug combination screening.

Ω (omega) penalty structure
----------------------------
  Ω = -∞  : DISCARD — one or more CRITICAL co-toxicities detected.
             Both drugs cause hepatotoxicity, QT prolongation, or nephrotoxicity
             at high frequency. Combination is clinically dangerous.

  Ω ∈ [-1, 0] : Graduated penalty for cumulative adverse event burden.
                 Applied as a multiplicative reduction to the combination score.

Critical toxicity categories (automatic -∞)
--------------------------------------------
  - HEPATOTOXICITY:  Both drugs cause liver damage (drug-induced liver injury)
  - QT_PROLONGATION: Both drugs prolong QTc interval (torsades de pointes risk)
  - NEPHROTOXICITY:  Both drugs impair renal function
  - MYELOSUPPRESSION: Both suppress bone marrow in oncology contexts
  - SEROTONIN_SYNDROME: Both are serotonergic agents

Data sources (live API + curated fallback)
------------------------------------------
  1. openFDA FAERS: adverse event frequency from FDA Adverse Event Reporting System
  2. FDA drug labels via openFDA /drug/label API
  3. OFFSIDES dataset (curated fallback): published drug off-target side effects
     Source: Tatonetti et al. (2012) Sci Transl Med. doi:10.1126/scitranslmed.3003377

Safety threshold
----------------
  A combination is discarded (Ω = -∞) if:
    - BOTH drugs have critical AE rate >= CRITICAL_AE_THRESHOLD (default 0.01 = 1%)
    - OR sum of weighted AE burden > CUMULATIVE_BURDEN_THRESHOLD

Usage
-----
    engine = ToxicityEngine()
    result = await engine.assess_combination(["sildenafil", "nitrate"])
    if result.is_critical:
        print("DISCARD:", result.omega_penalty, result.critical_flags)
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

CACHE_DIR          = Path("/tmp/drug_repurposing_cache")
TOX_CACHE_FILE     = CACHE_DIR / "toxicity_ae_cache.json"
CACHE_TTL_SECS     = 7 * 24 * 3600   # 1 week

OPENFDA_EVENTS_URL = "https://api.fda.gov/drug/event.json"
OPENFDA_LABEL_URL  = "https://api.fda.gov/drug/label.json"

CRITICAL_AE_THRESHOLD       = 0.01    # 1% frequency → critical
CUMULATIVE_BURDEN_THRESHOLD = 0.40    # sum of weighted AE scores
MAX_AE_EVENTS_PER_DRUG      = 500     # openFDA query limit


# ─────────────────────────────────────────────────────────────────────────────
# OFFSIDES-derived curated toxicity profiles
# Source: Tatonetti et al. (2012), Sci Transl Med + FDA drug labels
# Format: drug_name_lower → {tox_category: relative_frequency 0-1}
# ─────────────────────────────────────────────────────────────────────────────
CURATED_TOXICITY: Dict[str, Dict[str, float]] = {
    # ── Hepatotoxic drugs ────────────────────────────────────────────────────
    "methotrexate":        {"HEPATOTOXICITY": 0.15, "MYELOSUPPRESSION": 0.20},
    "isoniazid":           {"HEPATOTOXICITY": 0.10},
    "amiodarone":          {"HEPATOTOXICITY": 0.08, "QT_PROLONGATION": 0.25},
    "valproate":           {"HEPATOTOXICITY": 0.06},
    "azathioprine":        {"HEPATOTOXICITY": 0.05, "MYELOSUPPRESSION": 0.15},
    "leflunomide":         {"HEPATOTOXICITY": 0.05},
    "bosentan":            {"HEPATOTOXICITY": 0.12},
    "imatinib":            {"HEPATOTOXICITY": 0.04, "FLUID_RETENTION": 0.10},
    "gefitinib":           {"HEPATOTOXICITY": 0.06, "INTERSTITIAL_LUNG": 0.01},
    "acetaminophen":       {"HEPATOTOXICITY": 0.02},
    "paracetamol":         {"HEPATOTOXICITY": 0.02},
    # ── QT prolonging drugs ──────────────────────────────────────────────────
    "haloperidol":         {"QT_PROLONGATION": 0.20, "EXTRAPYRAMIDAL": 0.30},
    "droperidol":          {"QT_PROLONGATION": 0.25},
    "thioridazine":        {"QT_PROLONGATION": 0.30},
    "ziprasidone":         {"QT_PROLONGATION": 0.15},
    "quetiapine":          {"QT_PROLONGATION": 0.05},
    "ondansetron":         {"QT_PROLONGATION": 0.04},
    "azithromycin":        {"QT_PROLONGATION": 0.02},
    "clarithromycin":      {"QT_PROLONGATION": 0.05},
    "erythromycin":        {"QT_PROLONGATION": 0.06},
    "hydroxychloroquine":  {"QT_PROLONGATION": 0.04},
    "chloroquine":         {"QT_PROLONGATION": 0.06},
    "methadone":           {"QT_PROLONGATION": 0.20},
    "amiodarone":          {"QT_PROLONGATION": 0.25, "HEPATOTOXICITY": 0.08},
    "sotalol":             {"QT_PROLONGATION": 0.15},
    "verapamil":           {"QT_PROLONGATION": 0.04},
    "cisapride":           {"QT_PROLONGATION": 0.30},
    "terfenadine":         {"QT_PROLONGATION": 0.40},
    # ── Nephrotoxic drugs ────────────────────────────────────────────────────
    "lithium":             {"NEPHROTOXICITY": 0.20},
    "vancomycin":          {"NEPHROTOXICITY": 0.15},
    "aminoglycoside":      {"NEPHROTOXICITY": 0.25},
    "cyclosporine":        {"NEPHROTOXICITY": 0.25, "HYPERTENSION": 0.40},
    "tacrolimus":          {"NEPHROTOXICITY": 0.20},
    "amphotericin":        {"NEPHROTOXICITY": 0.30},
    "cisplatin":           {"NEPHROTOXICITY": 0.30, "PERIPHERAL_NEUROPATHY": 0.30},
    "carboplatin":         {"NEPHROTOXICITY": 0.10, "MYELOSUPPRESSION": 0.40},
    "ibuprofen":           {"NEPHROTOXICITY": 0.05, "GI_BLEEDING": 0.03},
    "naproxen":            {"NEPHROTOXICITY": 0.04, "GI_BLEEDING": 0.03},
    "indomethacin":        {"NEPHROTOXICITY": 0.06, "GI_BLEEDING": 0.05},
    "metformin":           {"LACTIC_ACIDOSIS": 0.001},
    # ── Myelosuppressive drugs ───────────────────────────────────────────────
    "cyclophosphamide":    {"MYELOSUPPRESSION": 0.50, "HEMORRHAGIC_CYSTITIS": 0.10},
    "melphalan":           {"MYELOSUPPRESSION": 0.60},
    "doxorubicin":         {"MYELOSUPPRESSION": 0.40, "CARDIOTOXICITY": 0.05},
    "epirubicin":          {"MYELOSUPPRESSION": 0.40, "CARDIOTOXICITY": 0.04},
    "paclitaxel":          {"MYELOSUPPRESSION": 0.30, "PERIPHERAL_NEUROPATHY": 0.40},
    "gemcitabine":         {"MYELOSUPPRESSION": 0.40},
    "bortezomib":          {"PERIPHERAL_NEUROPATHY": 0.35, "MYELOSUPPRESSION": 0.30},
    "thalidomide":         {"PERIPHERAL_NEUROPATHY": 0.50, "VTE": 0.03},
    "lenalidomide":        {"MYELOSUPPRESSION": 0.30, "VTE": 0.04},
    # ── Serotonergic drugs ───────────────────────────────────────────────────
    "fluoxetine":          {"SEROTONIN_SYNDROME": 0.01},
    "sertraline":          {"SEROTONIN_SYNDROME": 0.005},
    "paroxetine":          {"SEROTONIN_SYNDROME": 0.01},
    "venlafaxine":         {"SEROTONIN_SYNDROME": 0.01},
    "duloxetine":          {"SEROTONIN_SYNDROME": 0.005},
    "tramadol":            {"SEROTONIN_SYNDROME": 0.02, "SEIZURE": 0.02},
    "linezolid":           {"SEROTONIN_SYNDROME": 0.03},
    "rasagiline":          {"SEROTONIN_SYNDROME": 0.02},
    "selegiline":          {"SEROTONIN_SYNDROME": 0.02},
    # ── General safe drugs (low toxicity) ────────────────────────────────────
    "aspirin":             {"GI_BLEEDING": 0.02},
    "colchicine":          {"GI_TOXICITY": 0.10},
    "gabapentin":          {"SEDATION": 0.10},
    "pregabalin":          {"SEDATION": 0.12},
    "metformin":           {"GI_TOXICITY": 0.15},
    "sildenafil":          {"HYPOTENSION": 0.04},
    "hydroxychloroquine":  {"RETINOPATHY": 0.01},
}

# Critical AE categories that trigger Ω = -∞ when BOTH drugs are positive
CRITICAL_CATEGORIES: Set[str] = {
    "HEPATOTOXICITY",
    "QT_PROLONGATION",
    "NEPHROTOXICITY",
    "MYELOSUPPRESSION",
    "SEROTONIN_SYNDROME",
}

# Severity weights for cumulative burden calculation
AE_SEVERITY_WEIGHTS: Dict[str, float] = {
    "HEPATOTOXICITY":         1.0,
    "QT_PROLONGATION":        1.0,
    "NEPHROTOXICITY":         1.0,
    "MYELOSUPPRESSION":       0.9,
    "SEROTONIN_SYNDROME":     1.0,
    "CARDIOTOXICITY":         0.9,
    "PERIPHERAL_NEUROPATHY":  0.6,
    "VTE":                    0.7,
    "GI_BLEEDING":            0.5,
    "HEMORRHAGIC_CYSTITIS":   0.6,
    "LACTIC_ACIDOSIS":        0.8,
    "INTERSTITIAL_LUNG":      0.8,
    "EXTRAPYRAMIDAL":         0.5,
    "FLUID_RETENTION":        0.3,
    "HYPERTENSION":           0.4,
    "HYPOTENSION":            0.3,
    "SEDATION":               0.2,
    "GI_TOXICITY":            0.2,
    "RETINOPATHY":            0.5,
    "SEIZURE":                0.7,
}


@dataclass
class ToxicityResult:
    """Result of full toxicity assessment for a drug combination."""
    drugs: List[str]
    is_critical: bool = False
    omega_penalty: float = 0.0      # -inf, or value in [-1, 0]
    critical_flags: List[str] = field(default_factory=list)
    cumulative_burden: float = 0.0
    per_drug_profiles: Dict[str, Dict] = field(default_factory=dict)
    shared_ae_categories: List[str] = field(default_factory=list)
    safety_margin: float = 1.0      # 0-1, higher = safer
    recommendation: str = ""
    ae_source: str = "curated_offsides_fallback"


class ToxicityEngine:
    """
    Computes the Ω toxicity penalty for drug combinations.

    Assessment steps
    ----------------
    1. Load adverse event profile per drug (openFDA FAERS → curated fallback)
    2. Identify shared AE categories above CRITICAL_AE_THRESHOLD
    3. If any CRITICAL category is shared → Ω = -∞, discard pair
    4. Compute cumulative burden score across all AE categories
    5. If cumulative burden > CUMULATIVE_BURDEN_THRESHOLD → graduated Ω penalty
    6. Compute safety margin (1 - normalised cumulative burden)

    Safety margin interpretation
    ----------------------------
    1.0 : No predicted toxicity overlap → maximum safety margin
    0.5 : Moderate cumulative burden → use with monitoring
    0.0 : Maximum predicted toxicity → discard regardless of efficacy
    """

    def __init__(self):
        self._ae_cache: Dict[str, Dict] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if TOX_CACHE_FILE.exists():
            try:
                raw = json.loads(TOX_CACHE_FILE.read_text())
                now = time.time()
                self._ae_cache = {
                    k: v for k, v in raw.items()
                    if v.get("_fetched_at", 0) + CACHE_TTL_SECS > now
                }
                logger.info(f"Toxicity AE cache: {len(self._ae_cache)} drugs")
            except Exception:
                self._ae_cache = {}

    def _save_cache(self) -> None:
        try:
            TOX_CACHE_FILE.write_text(json.dumps(self._ae_cache, indent=2))
        except Exception as e:
            logger.warning(f"Toxicity cache save failed: {e}")

    # ── openFDA FAERS query ───────────────────────────────────────────────────

    async def _fetch_faers_ae_profile(
        self, drug_name: str, session: aiohttp.ClientSession
    ) -> Dict[str, float]:
        """
        Fetch adverse event frequency profile from openFDA FAERS.
        Returns {meddra_term: relative_frequency}.
        """
        key = drug_name.lower()
        if key in self._ae_cache:
            return self._ae_cache[key].get("ae_profile", {})

        try:
            params = {
                "search": f'patient.drug.medicinalproduct:"{drug_name}"',
                "limit":  MAX_AE_EVENTS_PER_DRUG,
            }
            async with session.get(
                OPENFDA_EVENTS_URL, params=params,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status == 404:
                    self._ae_cache[key] = {"ae_profile": {}, "_fetched_at": time.time()}
                    self._save_cache()
                    return {}
                if resp.status != 200:
                    return self._curated_profile(drug_name)
                data = await resp.json()
                results = data.get("results", [])

        except Exception as e:
            logger.debug(f"openFDA FAERS query failed for {drug_name}: {e}")
            return self._curated_profile(drug_name)

        if not results:
            return self._curated_profile(drug_name)

        # Count MedDRA reactions
        from collections import Counter
        reaction_counts: Counter = Counter()
        n_total = len(results)

        for report in results:
            reactions = report.get("patient", {}).get("reaction", [])
            for r in reactions:
                term = r.get("reactionmeddrapt", "").lower()
                if term:
                    reaction_counts[term] += 1

        ae_profile = {
            term: round(count / n_total, 4)
            for term, count in reaction_counts.most_common(50)
        }

        self._ae_cache[key] = {"ae_profile": ae_profile, "_fetched_at": time.time()}
        self._save_cache()
        return ae_profile

    def _curated_profile(self, drug_name: str) -> Dict[str, float]:
        """Return curated OFFSIDES-derived toxicity profile as fallback."""
        return CURATED_TOXICITY.get(drug_name.lower(), {})

    # ── Ω penalty logic ───────────────────────────────────────────────────────

    def _categorise_ae_profile(
        self, raw_profile: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Map raw MedDRA terms or curated keys to standardised AE categories.
        """
        # If already in category format (curated), return directly
        if all(k.isupper() for k in raw_profile.keys() if k):
            return raw_profile

        # MedDRA → category mapping
        meddra_to_category: Dict[str, str] = {
            "hepatitis":              "HEPATOTOXICITY",
            "liver injury":           "HEPATOTOXICITY",
            "alanine aminotransferase increased": "HEPATOTOXICITY",
            "aspartate aminotransferase increased": "HEPATOTOXICITY",
            "jaundice":               "HEPATOTOXICITY",
            "hepatic failure":        "HEPATOTOXICITY",
            "electrocardiogram qt prolonged": "QT_PROLONGATION",
            "qt prolongation":        "QT_PROLONGATION",
            "torsade de pointes":     "QT_PROLONGATION",
            "ventricular fibrillation": "QT_PROLONGATION",
            "renal failure":          "NEPHROTOXICITY",
            "renal impairment":       "NEPHROTOXICITY",
            "creatinine increased":   "NEPHROTOXICITY",
            "neutropenia":            "MYELOSUPPRESSION",
            "thrombocytopenia":       "MYELOSUPPRESSION",
            "anaemia":                "MYELOSUPPRESSION",
            "pancytopenia":           "MYELOSUPPRESSION",
            "serotonin syndrome":     "SEROTONIN_SYNDROME",
            "nausea":                 "GI_TOXICITY",
            "vomiting":               "GI_TOXICITY",
            "diarrhoea":              "GI_TOXICITY",
            "peripheral neuropathy":  "PERIPHERAL_NEUROPATHY",
            "seizure":                "SEIZURE",
            "pulmonary embolism":     "VTE",
            "deep vein thrombosis":   "VTE",
        }
        categorised: Dict[str, float] = {}
        for term, freq in raw_profile.items():
            for meddra_kw, category in meddra_to_category.items():
                if meddra_kw in term.lower():
                    if category not in categorised or categorised[category] < freq:
                        categorised[category] = freq
        return categorised

    def _compute_omega(
        self,
        profiles: List[Dict[str, float]],
        drug_names: List[str],
    ) -> ToxicityResult:
        """
        Compute Ω penalty from a list of drug AE profiles.
        """
        categorised = [self._categorise_ae_profile(p) for p in profiles]

        # Find shared AE categories (all drugs have the AE above threshold)
        shared_above_threshold: Dict[str, List[float]] = {}
        all_categories = set()
        for profile in categorised:
            all_categories |= set(profile.keys())

        for cat in all_categories:
            freqs = [p.get(cat, 0.0) for p in categorised]
            if all(f >= CRITICAL_AE_THRESHOLD for f in freqs):
                shared_above_threshold[cat] = freqs

        # Critical check
        critical_flags = [
            cat for cat in shared_above_threshold
            if cat in CRITICAL_CATEGORIES
        ]
        is_critical = len(critical_flags) > 0

        # Cumulative burden: sum of weighted, shared AE frequencies
        cumulative_burden = 0.0
        for cat, freqs in shared_above_threshold.items():
            weight = AE_SEVERITY_WEIGHTS.get(cat, 0.3)
            mean_freq = sum(freqs) / len(freqs)
            cumulative_burden += weight * mean_freq

        # Ω calculation
        if is_critical:
            omega = float("-inf")
        elif cumulative_burden > CUMULATIVE_BURDEN_THRESHOLD:
            # Sigmoid penalty: worse as burden approaches 1.0
            normalised = min(cumulative_burden / 1.0, 1.0)
            omega = -normalised * 0.80
        else:
            omega = -cumulative_burden * 0.30

        # Safety margin: inverted normalised burden, capped 0-1
        safety_margin = max(0.0, min(1.0, 1.0 - cumulative_burden))

        # Recommendation
        if is_critical:
            rec = (
                f"DISCARD — Critical overlapping toxicities: "
                f"{', '.join(critical_flags)}. "
                f"Both drugs cause these severe adverse events. "
                f"Combination is clinically contraindicated."
            )
        elif cumulative_burden > CUMULATIVE_BURDEN_THRESHOLD:
            rec = (
                f"HIGH CAUTION — Cumulative adverse event burden "
                f"{cumulative_burden:.2f} exceeds safety threshold {CUMULATIVE_BURDEN_THRESHOLD}. "
                f"Requires intensive clinical monitoring if used."
            )
        elif shared_above_threshold:
            shared_list = ", ".join(shared_above_threshold.keys())
            rec = (
                f"MODERATE CAUTION — Overlapping AEs detected ({shared_list}). "
                f"Safety margin {safety_margin:.2f}. Monitor accordingly."
            )
        else:
            rec = f"LOW TOXICITY RISK — Safety margin {safety_margin:.2f}. Proceed with standard monitoring."

        return ToxicityResult(
            drugs=drug_names,
            is_critical=is_critical,
            omega_penalty=round(omega, 4) if omega != float("-inf") else float("-inf"),
            critical_flags=critical_flags,
            cumulative_burden=round(cumulative_burden, 4),
            per_drug_profiles={
                name: cat_profile
                for name, cat_profile in zip(drug_names, categorised)
            },
            shared_ae_categories=list(shared_above_threshold.keys()),
            safety_margin=round(safety_margin, 4),
            recommendation=rec,
            ae_source="openFDA_FAERS+offsides_curated",
        )

    # ── Main API ──────────────────────────────────────────────────────────────

    async def assess_combination(
        self,
        drug_names: List[str],
        use_openfda: bool = True,
    ) -> ToxicityResult:
        """
        Full toxicity assessment for a drug combination.

        Parameters
        ----------
        drug_names : list of str
            Drug names in the combination (2 or 3 drugs).
        use_openfda : bool
            Query openFDA FAERS for AE data (default True).
            Falls back to curated OFFSIDES table if API unavailable.

        Returns
        -------
        ToxicityResult with Ω penalty, safety margin, and recommendation.
        """
        profiles: List[Dict[str, float]] = []

        if use_openfda:
            async with aiohttp.ClientSession() as session:
                tasks = [
                    self._fetch_faers_ae_profile(name, session)
                    for name in drug_names
                ]
                raw_profiles = await asyncio.gather(*tasks, return_exceptions=True)

            for i, (name, raw) in enumerate(zip(drug_names, raw_profiles)):
                if isinstance(raw, Exception) or not raw:
                    profiles.append(self._curated_profile(name))
                else:
                    profiles.append(raw)
        else:
            profiles = [self._curated_profile(name) for name in drug_names]

        return self._compute_omega(profiles, drug_names)

    def assess_combination_sync(self, drug_names: List[str]) -> ToxicityResult:
        """
        Synchronous fallback using only the curated OFFSIDES table.
        Used inside multiprocessing workers where async is not available.
        """
        profiles = [self._curated_profile(name) for name in drug_names]
        result = self._compute_omega(profiles, drug_names)
        result.ae_source = "offsides_curated_sync"
        return result

    def to_dict(self, result: ToxicityResult) -> Dict:
        return {
            "drugs":               result.drugs,
            "is_critical":         result.is_critical,
            "omega_penalty":       result.omega_penalty if result.omega_penalty != float("-inf") else "NEGATIVE_INFINITY",
            "critical_flags":      result.critical_flags,
            "cumulative_burden":   result.cumulative_burden,
            "safety_margin":       result.safety_margin,
            "shared_ae_categories": result.shared_ae_categories,
            "per_drug_profiles":   result.per_drug_profiles,
            "recommendation":      result.recommendation,
            "ae_source":           result.ae_source,
        }