"""
main.py — TwinTrial Analytics FastAPI Server
============================================
Two primary endpoints:
  POST /treatment_plan  — main deliverable, generates ranked combo regimens
  POST /disease_rank    — returns disease opportunity ranking for pipeline planning
  GET  /               — health check

Legacy endpoint kept for validation scripts:
  POST /analyze         — wraps generate_treatment_plan(), old response shape
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from pipeline.production_pipeline import ProductionPipeline
from pipeline.disease_ranker import rank_diseases_by_opportunity, get_disease_brief
from pipeline.drug_filter import DrugSafetyFilter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="TwinTrial Analytics API",
    description=(
        "In-silico drug repurposing for generic combination therapies. "
        "Generates simulation-backed treatment plans for rare and common diseases "
        "using only off-patent drugs."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline: ProductionPipeline = None


@app.on_event("startup")
async def startup_event():
    global pipeline
    logger.info("TwinTrial Analytics API starting...")
    try:
        pipeline = ProductionPipeline()
        logger.info("Pipeline ready.")
    except Exception as e:
        logger.error(f"Pipeline init failed: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    global pipeline
    if pipeline:
        await pipeline.close()
    logger.info("API shutdown complete.")


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    return {
        "status":  "online",
        "service": "TwinTrial Analytics API",
        "version": "1.0.0",
    }


# ── Primary endpoint: treatment plan ─────────────────────────────────────────

@app.post("/treatment_plan", tags=["Treatment Plans"])
async def treatment_plan(request: dict):
    """
    Generate a ranked generic combination therapy treatment plan for a disease.

    Request body
    ------------
    {
        "disease_name":     "pulmonary arterial hypertension",   // required
        "max_regimens":     10,                                   // optional, default 10
        "include_triples":  true,                                 // optional, default true
        "fetch_ppi":        true,                                 // optional, default true
        "fetch_similarity": true,                                 // optional, default true
        "use_tissue":       true                                  // optional, default true
    }

    Returns
    -------
    Full treatment plan with:
      - ranked_regimens: list of combo drug plans with ORR/P2 estimates
      - wet_lab_brief:   top gene targets to validate in cell assay
      - pag_brief:       plain-language summary for patient advocacy groups
      - biotech_brief:   technical summary for pharma BD teams
      - header:          data provenance, drug pool stats, limitations
    """
    global pipeline

    if not pipeline:
        return {"success": False, "error": "Pipeline not initialised"}

    disease_name = request.get("disease_name", "").strip()
    if not disease_name:
        return {"success": False, "error": "Missing disease_name"}

    max_regimens     = int(request.get("max_regimens", 10))
    include_triples  = bool(request.get("include_triples", True))
    fetch_ppi        = bool(request.get("fetch_ppi", True))
    fetch_similarity = bool(request.get("fetch_similarity", True))
    use_tissue       = bool(request.get("use_tissue", True))

    logger.info(f"/treatment_plan request: {disease_name}")

    try:
        plan = await pipeline.generate_treatment_plan(
            disease_name=disease_name,
            max_regimens=max_regimens,
            include_triples=include_triples,
            fetch_ppi=fetch_ppi,
            fetch_similarity=fetch_similarity,
            use_tissue=use_tissue,
        )
        return plan
    except Exception as e:
        logger.error(f"Treatment plan error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


# ── Disease opportunity ranking ───────────────────────────────────────────────

@app.get("/disease_rank", tags=["Strategy"])
async def disease_rank():
    """
    Return the full disease opportunity ranking.

    Shows which diseases to target next based on:
      market size, pipeline feasibility, IP safety, PAG strength, combo potential.

    No request body needed — returns pre-computed ranking.
    """
    try:
        ranking = rank_diseases_by_opportunity()
        tiers = {"A": [], "B": [], "C": []}
        for r in ranking:
            tiers[r["tier"]].append(r)
        return {
            "success": True,
            "ranking": ranking,
            "tiers":   tiers,
            "summary": {
                "tier_a_count": len(tiers["A"]),
                "tier_b_count": len(tiers["B"]),
                "tier_c_count": len(tiers["C"]),
                "top_disease":  ranking[0]["disease"] if ranking else None,
            },
        }
    except Exception as e:
        logger.error(f"Disease rank error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


@app.post("/disease_brief", tags=["Strategy"])
async def disease_brief(request: dict):
    """
    Return opportunity data for a specific disease.

    Request body: {"disease_name": "pulmonary arterial hypertension"}
    """
    disease_name = request.get("disease_name", "").strip()
    if not disease_name:
        return {"success": False, "error": "Missing disease_name"}
    brief = get_disease_brief(disease_name)
    if not brief:
        return {"success": False, "error": f"Disease not in opportunity database: {disease_name}"}
    return {"success": True, "brief": brief}


# ── Clinical validation (unchanged from original) ────────────────────────────

@app.post("/validate_clinical", tags=["Validation"])
async def validate_clinical(request: dict):
    """
    Validate a drug candidate clinically using ClinicalTrials.gov, PubMed, OpenFDA.
    Same as original — useful for enriching treatment plan reports.
    """
    global pipeline
    if not pipeline:
        return {"success": False, "error": "Pipeline not initialised"}

    drug_name    = request.get("drug_name")
    disease_name = request.get("disease_name")
    drug_data    = request.get("drug_data", {})
    disease_data = request.get("disease_data", {})

    if not drug_name or not disease_name:
        return {"success": False, "error": "Missing drug_name or disease_name"}

    from pipeline.clinical_validator import ClinicalValidator
    validator = ClinicalValidator()
    try:
        result = await validator.validate_candidate(
            drug_name=drug_name,
            disease_name=disease_name,
            drug_data=drug_data,
            disease_data=disease_data,
        )
        return {"success": True, "validation": result}
    except Exception as e:
        logger.error(f"Clinical validation error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}
    finally:
        await validator.close()


# ── Legacy endpoint (backward compat for run_validation.py) ──────────────────

@app.post("/analyze", tags=["Legacy"])
async def analyze_disease(request: dict):
    """
    Legacy endpoint — wraps generate_treatment_plan() with the old response shape.
    Kept so run_validation.py and existing integrations don't break.
    """
    global pipeline
    if not pipeline:
        return {"success": False, "error": "Pipeline not initialised"}

    disease_name = request.get("disease_name", "").strip()
    min_score    = float(request.get("min_score", 0.2))
    max_results  = int(request.get("max_results", 10))

    if not disease_name:
        return {"success": False, "error": "Missing disease_name"}

    logger.info(f"/analyze (legacy) request: {disease_name}")

    try:
        result = await pipeline.analyze_disease(
            disease_name=disease_name,
            min_score=min_score,
            max_results=max_results,
        )

        # Apply safety filter on candidates for legacy callers
        candidates = result.get("candidates", [])
        if candidates:
            safety_filter = DrugSafetyFilter()
            safe_candidates, filtered_out = await safety_filter.filter_candidates(
                candidates=candidates,
                disease_name=disease_name,
                remove_absolute=True,
                remove_relative=True,
            )
            result["candidates"]      = safe_candidates[:max_results]
            result["filtered_count"]  = len(filtered_out)
            result["filtered_drugs"]  = [
                {
                    "drug_name": c["drug_name"],
                    "reason":    c.get("contraindication", {}).get("reason", "Unknown"),
                    "severity":  c.get("contraindication", {}).get("severity", "unknown"),
                }
                for c in filtered_out
            ]

        return result
    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}