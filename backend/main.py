"""
main.py — TwinTrial Analytics FastAPI Server
============================================
Primary endpoints:
  POST /treatment_plan    — generates ranked combo regimens
  POST /generate_rationale — generates AI pharmacological rationale
  POST /disease_rank      — returns disease opportunity ranking
  GET  /                  — health check
"""

import os
import logging
from dotenv import load_dotenv
from google import genai 
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 1. Load the .env file immediately to access GOOGLE_API_KEY
load_dotenv()

# 2. Initialize the Gemini client using the environment variable label
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError("GOOGLE_API_KEY not set in environment")
client = genai.Client(api_key=api_key)

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
    description="In-silico drug repurposing for generic combination therapies.",
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
    global pipeline
    if not pipeline:
        return {"success": False, "error": "Pipeline not initialised"}

    disease_name = request.get("disease_name", "").strip()
    if not disease_name:
        return {"success": False, "error": "Missing disease_name"}

    try:
        plan = await pipeline.generate_treatment_plan(
            disease_name=disease_name,
            max_regimens=int(request.get("max_regimens", 10)),
            include_triples=bool(request.get("include_triples", True)),
            fetch_ppi=bool(request.get("fetch_ppi", True)),
            fetch_similarity=bool(request.get("fetch_similarity", True)),
            use_tissue=bool(request.get("use_tissue", True)),
        )
        return plan
    except Exception as e:
        logger.error(f"Treatment plan error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

# ── AI Rationale endpoint ─────────────────────────────────────────────────────
@app.post("/generate_rationale", tags=["AI"])
async def generate_rationale(data: dict):
    regimen = data.get("regimen")
    disease = data.get("disease")
    mechanisms = data.get("mechanisms")
    
    prompt = f"""
    You are a clinical pharmacologist. Explain in 2 sentences why the drug combination 
    {regimen} is promising for {disease}. 
    Mechanisms: {mechanisms}.
    Keep it under 50 words and be highly technical.
    """
    
    try:
        response = client.models.generate_content(
        model="gemini-2.0-flash",   # ← Use a current model name
        contents=prompt
    )
        text = response.candidates[0].content.parts[0].text if response.candidates else None
        return {"rationale": text or "No rationale generated."}
    except Exception as e:
        logger.error(f"AI Rationale error: {e}")
        return {"rationale": "AI rationale generation failed. Consult technical documentation."}

# ── Strategy and Validation Endpoints ────────────────────────────────────────

@app.get("/disease_rank", tags=["Strategy"])
async def disease_rank():
    try:
        ranking = rank_diseases_by_opportunity()
        return {"success": True, "ranking": ranking}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/validate_clinical", tags=["Validation"])
async def validate_clinical(request: dict):
    global pipeline
    drug_name = request.get("drug_name")
    disease_name = request.get("disease_name")
    if not drug_name or not disease_name:
        return {"success": False, "error": "Missing parameters"}

    from pipeline.clinical_validator import ClinicalValidator
    validator = ClinicalValidator()
    try:
        result = await validator.validate_candidate(drug_name=drug_name, disease_name=disease_name)
        return {"success": True, "validation": result}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        await validator.close()