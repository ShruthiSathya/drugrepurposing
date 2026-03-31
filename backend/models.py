"""
models.py — TwinTrial Analytics Pydantic Models
================================================
Request/response models for the FastAPI endpoints.

New models added for TwinTrial:
  TreatmentPlanRequest   — input to /treatment_plan
  ComboRegimen           — one ranked combination in a treatment plan
  TreatmentPlan          — full treatment plan response
  DiseaseOpportunity     — one row in the disease opportunity ranking
  DiseaseRankResponse    — /disease_rank response

Existing models kept for backward compatibility:
  DrugCandidate          — single drug scoring result
  QueryRequest           — legacy /analyze request
  RepurposingResult      — legacy /analyze response
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any


# ─────────────────────────────────────────────────────────────────────────────
# TwinTrial models (new)
# ─────────────────────────────────────────────────────────────────────────────

class TreatmentPlanRequest(BaseModel):
    disease_name:     str
    max_regimens:     int   = Field(default=10,   ge=1, le=30)
    include_triples:  bool  = True
    fetch_ppi:        bool  = True
    fetch_similarity: bool  = True
    use_tissue:       bool  = True
    min_single_score: float = Field(default=0.10, ge=0.0, le=1.0)


class ScoreBreakdown(BaseModel):
    base_score:         float = 0.0
    synergy_bonus:      float = 0.0
    antagonism_penalty: float = 0.0
    coverage_bonus:     float = 0.0
    redundancy_penalty: float = 0.0


class TrialDetail(BaseModel):
    dcr:              float = 0.0
    median_pfs_weeks: float = 0.0
    n_patients:       int   = 200
    orr_ci_90:        List[float] = [0.0, 0.0]
    network_effect:   float = 0.0
    recommendation:   str   = ""


class ComboRegimen(BaseModel):
    rank:               int
    regimen:            str
    n_drugs:            int
    combo_score:        float
    orr_estimate:       float
    pfs6_estimate:      float
    p2_probability:     float
    priority:           str   # HIGH / MEDIUM / LOW
    is_synergistic:     bool
    is_antagonistic:    bool
    mechanism_a:        Optional[str] = None
    mechanism_b:        Optional[str] = None
    mechanism_c:        Optional[str] = None
    combined_gene_coverage: int   = 0
    shared_genes:       List[str] = []
    wet_lab_targets:    List[str] = []
    score_breakdown:    ScoreBreakdown = ScoreBreakdown()
    pag_one_liner:      str  = ""
    trial_detail:       Optional[TrialDetail] = None


class WetLabBrief(BaseModel):
    priority_targets:     List[str]
    rationale:            str
    suggested_assays:     List[str]
    university_partner_note: str


class TreatmentPlanHeader(BaseModel):
    title:             str
    disease:           str
    disease_id:        str
    built_at:          str
    pipeline_version:  str
    drug_pool:         Dict[str, Any]
    disease_genes_count:    int
    disease_pathways_count: int
    data_sources:      List[str]


class TreatmentPlan(BaseModel):
    success:           bool
    disease:           str
    header:            TreatmentPlanHeader
    ranked_regimens:   List[ComboRegimen]
    top_single_drugs:  List[Dict]
    wet_lab_brief:     WetLabBrief
    pag_brief:         str
    biotech_brief:     str
    limitations:       List[str]
    summary:           Dict[str, Any]
    pipeline_stats:    Optional[Dict[str, Any]] = None


class DiseaseOpportunity(BaseModel):
    disease:           str
    opportunity_score: float
    tier:              str   # A / B / C
    market_size_B:     float
    feasibility:       float
    ip_safety:         float
    pag_strength:      float
    combo_potential:   float
    orphan_flag:       bool
    pag_examples:      List[str] = []
    known_generics:    List[str] = []
    notes:             str  = ""


class DiseaseRankResponse(BaseModel):
    success:   bool
    ranking:   List[DiseaseOpportunity]
    tiers:     Dict[str, List[DiseaseOpportunity]]
    summary:   Dict[str, Any]


class DiseaseBriefRequest(BaseModel):
    disease_name: str


# ─────────────────────────────────────────────────────────────────────────────
# Legacy models (kept for backward compat)
# ─────────────────────────────────────────────────────────────────────────────

class DrugCandidate(BaseModel):
    drug_name:              str
    drug_id:                str
    original_indication:    str
    composite_score:        float
    pathway_overlap_score:  float
    gene_target_score:      float
    literature_score:       float
    shared_genes:           List[str]
    shared_pathways:        List[str]
    mechanism:              str
    explanation:            str
    confidence:             str
    patent_status:          Optional[str] = None


class QueryRequest(BaseModel):
    disease_name:       str
    top_k:              int   = 10
    min_score:          float = 0.3
    anthropic_api_key:  Optional[str] = None


class RepurposingResult(BaseModel):
    disease_name:    str
    disease_genes:   List[str]
    disease_pathways: List[str]
    candidates:      List[DrugCandidate]
    graph_stats:     dict
    data_sources:    List[str]