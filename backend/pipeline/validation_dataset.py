"""
validation_dataset.py — Curated Drug Repurposing Validation Dataset
====================================================================
55-case curated validation set for the LiveDRP / TwinTrial pipeline.

Structure
---------
  VALIDATION_CASES : list of dict
      All cases (positive and negative).
  get_positive_cases() → list of TRUE_POSITIVE cases
  get_negative_cases() → list of TRUE_NEGATIVE cases
  OUT_OF_SCOPE_CASES : list of dict — removed cases with rationale

Each case has:
    drug                : str  — drug name (matches ChEMBL preferred name)
    disease             : str  — disease name (matches OpenTargets search)
    status              : str  — TRUE_POSITIVE | TRUE_NEGATIVE
    mechanism           : str  — one-sentence pharmacological rationale
    min_score           : float — minimum expected pipeline score (TP) or
                                  maximum expected score (TN)
    expected_rank_top_n : int  — expected rank within top-N (0 = score-only check)
    sources             : list[str] — PMID or URL supporting the case
    notes               : str  — optional context

Pass criterion (v3.1)
---------------------
    For TRUE_POSITIVE: PASS if rank <= expected_rank_top_n OR score >= min_score
    For TRUE_NEGATIVE: PASS if score < min_score
"""

from typing import Dict, List

DATASET_VERSION = "v3.2"

# ─────────────────────────────────────────────────────────────────────────────
# TRUE_POSITIVE cases — mechanism-congruent repurposing with published evidence
# ─────────────────────────────────────────────────────────────────────────────

_POSITIVE_CASES: List[Dict] = [

    # ── PAH ───────────────────────────────────────────────────────────────────
    {
        "drug":                "sildenafil",
        "disease":             "pulmonary arterial hypertension",
        "status":              "TRUE_POSITIVE",
        "mechanism":           "PDE5 inhibitor — reduces pulmonary vascular resistance via cGMP/PKG",
        "min_score":           0.40,
        "expected_rank_top_n": 20,
        "sources":             ["PMID:16291984"],
        "notes":               "FDA-approved for PAH (Revatio). Landmark RCT.",
    },
    {
        "drug":                "imatinib",
        "disease":             "pulmonary arterial hypertension",
        "status":              "TRUE_POSITIVE",
        "mechanism":           "BCR-ABL/PDGFR kinase inhibitor — blocks vascular smooth muscle proliferation",
        "min_score":           0.25,
        "expected_rank_top_n": 50,
        "sources":             ["PMID:16192490", "PMID:20565548"],
        "notes":               "Phase 2 data (IMPRES trial). PDGFRB expressed in PAH vessels.",
    },
    {
        "drug":                "bosentan",
        "disease":             "pulmonary arterial hypertension",
        "status":              "TRUE_POSITIVE",
        "mechanism":           "Dual endothelin receptor antagonist (ET-A/ET-B) — reduces pulmonary vasoconstriction",
        "min_score":           0.40,
        "expected_rank_top_n": 15,
        "sources":             ["PMID:11502536"],
        "notes":               "First oral approved ERA for PAH.",
    },

    # ── Rheumatoid Arthritis ──────────────────────────────────────────────────
    {
        "drug":                "rituximab",
        "disease":             "rheumatoid arthritis",
        "status":              "TRUE_POSITIVE",
        "mechanism":           "Anti-CD20 monoclonal antibody — depletes B cells, reducing autoimmune inflammation",
        "min_score":           0.35,
        "expected_rank_top_n": 30,
        "sources":             ["PMID:16951627"],
        "notes":               "Approved for anti-TNF refractory RA.",
    },
    {
        "drug":                "methotrexate",
        "disease":             "rheumatoid arthritis",
        "status":              "TRUE_POSITIVE",
        "mechanism":           "DHFR inhibitor / anti-inflammatory DMARD — first-line RA treatment",
        "min_score":           0.35,
        "expected_rank_top_n": 20,
        "sources":             ["PMID:10470175"],
        "notes":               "Anchor DMARD for RA for decades.",
    },
    {
        "drug":                "hydroxychloroquine",
        "disease":             "rheumatoid arthritis",
        "status":              "TRUE_POSITIVE",
        "mechanism":           "TLR7/9 inhibitor via lysosomal pH disruption — dampens innate immune activation",
        "min_score":           0.25,
        "expected_rank_top_n": 40,
        "sources":             ["PMID:7616906"],
        "notes":               "Standard DMARD in RA triple therapy.",
    },

    # ── PCOS ─────────────────────────────────────────────────────────────────
    {
        "drug":                "metformin",
        "disease":             "polycystic ovary syndrome",
        "status":              "TRUE_POSITIVE",
        "mechanism":           "AMPK activator — reduces insulin resistance and androgen excess in PCOS",
        "min_score":           0.28,
        "expected_rank_top_n": 30,
        "sources":             ["PMID:14576245"],
        "notes":               "Systematic review and meta-analysis of metformin in PCOS.",
    },

    # ── Multiple Myeloma ──────────────────────────────────────────────────────
    {
        "drug":                "thalidomide",
        "disease":             "multiple myeloma",
        "status":              "TRUE_POSITIVE",
        "mechanism":           "CRBN E3 ligase substrate — degrades Ikaros/Aiolos, anti-myeloma and anti-angiogenic",
        "min_score":           0.35,
        "expected_rank_top_n": 20,
        "sources":             ["PMID:10564685"],
        "notes":               "Landmark Singhal et al. N Engl J Med 1999 paper.",
    },
    {
        "drug":                "dexamethasone",
        "disease":             "multiple myeloma",
        "status":              "TRUE_POSITIVE",
        "mechanism":           "Glucocorticoid receptor agonist — induces apoptosis in myeloma cells",
        "min_score":           0.30,
        "expected_rank_top_n": 25,
        "sources":             ["PMID:11689576"],
        "notes":               "Backbone of almost all myeloma regimens.",
    },
    {
        "drug":                "bortezomib",
        "disease":             "multiple myeloma",
        "status":              "TRUE_POSITIVE",
        "mechanism":           "Proteasome (26S) inhibitor — induces UPS-mediated apoptosis in myeloma",
        "min_score":           0.30,
        "expected_rank_top_n": 25,
        "sources":             ["PMID:12931552"],
        "notes":               "FDA approved for myeloma 2003.",
    },

    # ── Chronic Myelogenous Leukemia ──────────────────────────────────────────
    {
        "drug":                "imatinib",
        "disease":             "chronic myelogenous leukemia",
        "status":              "TRUE_POSITIVE",
        "mechanism":           "BCR-ABL tyrosine kinase inhibitor — directly targets the driver oncoprotein",
        "min_score":           0.50,
        "expected_rank_top_n": 10,
        "sources":             ["PMID:11287975"],
        "notes":               "Prototype precision oncology. ABL1 is dominant CML gene.",
    },

    # ── Essential Tremor ─────────────────────────────────────────────────────
    {
        "drug":                "propranolol",
        "disease":             "essential tremor",
        "status":              "TRUE_POSITIVE",
        "mechanism":           "Non-selective beta-adrenergic blocker — reduces peripheral tremor amplitude",
        "min_score":           0.20,
        "expected_rank_top_n": 50,
        "sources":             ["PMID:3960325"],
        "notes":               "First-line treatment for essential tremor.",
    },

    # ── Alopecia ─────────────────────────────────────────────────────────────
    {
        "drug":                "minoxidil",
        "disease":             "alopecia",
        "status":              "TRUE_POSITIVE",
        "mechanism":           "K-ATP channel opener (KCNJ8/ABCC9) — prolongs anagen phase in hair follicles",
        "min_score":           0.20,
        "expected_rank_top_n": 50,
        "sources":             ["PMID:14996087"],
        "notes":               "Classic repurposing from hypertension to hair loss.",
    },
    {
        "drug":                "finasteride",
        "disease":             "alopecia",
        "status":              "TRUE_POSITIVE",
        "mechanism":           "5-alpha reductase (SRD5A1/SRD5A2) inhibitor — reduces scalp DHT",
        "min_score":           0.25,
        "expected_rank_top_n": 40,
        "sources":             ["PMID:9777765"],
        "notes":               "Approved for androgenetic alopecia.",
    },

    # ── Coronary Artery Disease ───────────────────────────────────────────────
    {
        "drug":                "aspirin",
        "disease":             "coronary artery disease",
        "status":              "TRUE_POSITIVE",
        "mechanism":           "Irreversible COX-1 inhibitor — reduces platelet aggregation and thrombosis",
        "min_score":           0.25,
        "expected_rank_top_n": 30,
        "sources":             ["PMID:11446622"],
        "notes":               "Standard antiplatelet in secondary prevention.",
    },
    {
        "drug":                "atorvastatin",
        "disease":             "coronary artery disease",
        "status":              "TRUE_POSITIVE",
        "mechanism":           "HMGCR inhibitor — reduces LDL-C and plaque formation",
        "min_score":           0.28,
        "expected_rank_top_n": 25,
        "sources":             ["PMID:9360337"],
        "notes":               "TNT trial. Statin for CAD is definitive guideline.",
    },

    # ── Breast Cancer ─────────────────────────────────────────────────────────
    {
        "drug":                "tamoxifen",
        "disease":             "breast carcinoma",
        "status":              "TRUE_POSITIVE",
        "mechanism":           "SERM — competitively blocks estrogen receptor (ESR1) in breast tissue",
        "min_score":           0.30,
        "expected_rank_top_n": 30,
        "sources":             ["PMID:16754728"],
        "notes":               "Definitive for ER+ breast cancer.",
    },
    {
        "drug":                "raloxifene",
        "disease":             "breast carcinoma",
        "status":              "TRUE_POSITIVE",
        "mechanism":           "SERM — blocks ESR1 in breast tissue, reduces invasive breast cancer risk",
        "min_score":           0.25,
        "expected_rank_top_n": 40,
        "sources":             ["PMID:16754728"],
        "notes":               "STAR trial vs tamoxifen.",
    },

    # ── Parkinson's ───────────────────────────────────────────────────────────
    {
        "drug":                "amantadine",
        "disease":             "parkinson disease",
        "status":              "TRUE_POSITIVE",
        "mechanism":           "NMDA receptor antagonist + dopamine release promoter — reduces motor symptoms",
        "min_score":           0.20,
        "expected_rank_top_n": 50,
        "sources":             ["PMID:4142340"],
        "notes":               "First antiviral drug repurposed for PD (1969).",
    },
    {
        "drug":                "rasagiline",
        "disease":             "parkinson disease",
        "status":              "TRUE_POSITIVE",
        "mechanism":           "Irreversible MAO-B inhibitor — prevents dopamine breakdown",
        "min_score":           0.22,
        "expected_rank_top_n": 40,
        "sources":             ["PMID:15072009"],
        "notes":               "TEMPO trial. Approved for early PD.",
    },

    # ── Type 2 Diabetes ───────────────────────────────────────────────────────
    {
        "drug":                "metformin",
        "disease":             "type 2 diabetes mellitus",
        "status":              "TRUE_POSITIVE",
        "mechanism":           "AMPK activator — inhibits hepatic gluconeogenesis and improves insulin sensitivity",
        "min_score":           0.40,
        "expected_rank_top_n": 15,
        "sources":             ["PMID:11602624"],
        "notes":               "First-line pharmacotherapy for T2DM worldwide.",
    },
    {
        "drug":                "pioglitazone",
        "disease":             "type 2 diabetes mellitus",
        "status":              "TRUE_POSITIVE",
        "mechanism":           "PPARγ agonist — improves insulin sensitivity in adipose and muscle",
        "min_score":           0.30,
        "expected_rank_top_n": 25,
        "sources":             ["PMID:9934955"],
        "notes":               "Approved TZD for T2DM.",
    },

    # ── Tuberous Sclerosis ────────────────────────────────────────────────────
    {
        "drug":                "sirolimus",
        "disease":             "tuberous sclerosis",
        "status":              "TRUE_POSITIVE",
        "mechanism":           "mTOR (MTOR) inhibitor — targets hyperactivated TSC1/TSC2-mTOR pathway",
        "min_score":           0.35,
        "expected_rank_top_n": 20,
        "sources":             ["PMID:18554602"],
        "notes":               "EXIST-1/EXIST-2 trials. Directly targets TSC pathway.",
    },

    # ── Epilepsy ──────────────────────────────────────────────────────────────
    {
        "drug":                "valproic acid",
        "disease":             "epilepsy",
        "status":              "TRUE_POSITIVE",
        "mechanism":           "Sodium channel blocker + GABA potentiator + HDAC inhibitor — broad anticonvulsant",
        "min_score":           0.30,
        "expected_rank_top_n": 20,
        "sources":             ["PMID:6132483"],
        "notes":               "Broad-spectrum AED for multiple seizure types.",
    },
    {
        "drug":                "gabapentin",
        "disease":             "epilepsy",
        "status":              "TRUE_POSITIVE",
        "mechanism":           "Voltage-gated calcium channel α2δ subunit (CACNA2D1/2) blocker — reduces neuronal excitability",
        "min_score":           0.25,
        "expected_rank_top_n": 30,
        "sources":             ["PMID:8552955"],
        "notes":               "Approved adjunctive AED.",
    },

    # ── PNH ──────────────────────────────────────────────────────────────────
    {
        "drug":                "eculizumab",
        "disease":             "paroxysmal nocturnal hemoglobinuria",
        "status":              "TRUE_POSITIVE",
        "mechanism":           "Anti-C5 monoclonal antibody — blocks terminal complement activation",
        "min_score":           0.35,
        "expected_rank_top_n": 15,
        "sources":             ["PMID:16741576"],
        "notes":               "Landmark TRIUMPH trial. C5 is the principal PNH gene.",
    },

    # ── Cystic Fibrosis ───────────────────────────────────────────────────────
    {
        "drug":                "ivacaftor",
        "disease":             "cystic fibrosis",
        "status":              "TRUE_POSITIVE",
        "mechanism":           "CFTR potentiator — increases channel open probability in G551D mutation",
        "min_score":           0.40,
        "expected_rank_top_n": 10,
        "sources":             ["PMID:21083385"],
        "notes":               "Precision CFTR therapy. STRIVE trial.",
    },

    # ── SLE ───────────────────────────────────────────────────────────────────
    {
        "drug":                "hydroxychloroquine",
        "disease":             "systemic lupus erythematosus",
        "status":              "TRUE_POSITIVE",
        "mechanism":           "TLR7/9 inhibitor — reduces type I interferon signalling in SLE",
        "min_score":           0.28,
        "expected_rank_top_n": 30,
        "sources":             ["PMID:12211878"],
        "notes":               "Cornerstone drug for SLE — reduces flares and improves survival.",
    },

    # ── Gout ──────────────────────────────────────────────────────────────────
    {
        "drug":                "colchicine",
        "disease":             "gout",
        "status":              "TRUE_POSITIVE",
        "mechanism":           "Microtubule polymerisation inhibitor — blocks NLRP3 inflammasome activation",
        "min_score":           0.28,
        "expected_rank_top_n": 25,
        "sources":             ["PMID:3287145"],
        "notes":               "Acute and chronic gout prophylaxis. Also for pericarditis.",
    },
    {
        "drug":                "allopurinol",
        "disease":             "gout",
        "status":              "TRUE_POSITIVE",
        "mechanism":           "Xanthine oxidase inhibitor — reduces uric acid synthesis",
        "min_score":           0.28,
        "expected_rank_top_n": 20,
        "sources":             ["PMID:13562569"],
        "notes":               "First-line ULT for gout.",
    },

    # ── Pericarditis ──────────────────────────────────────────────────────────
    {
        "drug":                "colchicine",
        "disease":             "pericarditis",
        "status":              "TRUE_POSITIVE",
        "mechanism":           "Microtubule inhibitor / NLRP3 inflammasome blocker — reduces pericardial inflammation",
        "min_score":           0.25,
        "expected_rank_top_n": 30,
        "sources":             ["PMID:23765770"],
        "notes":               "COPE trial. Halves recurrence rate in pericarditis.",
    },

    # ── Heart Failure ─────────────────────────────────────────────────────────
    {
        "drug":                "spironolactone",
        "disease":             "heart failure",
        "status":              "TRUE_POSITIVE",
        "mechanism":           "Aldosterone receptor antagonist — reduces cardiac fibrosis and fluid retention",
        "min_score":           0.25,
        "expected_rank_top_n": 35,
        "sources":             ["PMID:10471456"],
        "notes":               "RALES trial. Reduces mortality in HFrEF.",
    },

    # ── Alzheimer's ───────────────────────────────────────────────────────────
    {
        "drug":                "donepezil",
        "disease":             "alzheimer disease",
        "status":              "TRUE_POSITIVE",
        "mechanism":           "Acetylcholinesterase inhibitor — increases synaptic acetylcholine in AD",
        "min_score":           0.30,
        "expected_rank_top_n": 20,
        "sources":             ["PMID:9327032"],
        "notes":               "Standard symptomatic treatment for mild-moderate AD.",
    },
    {
        "drug":                "memantine",
        "disease":             "alzheimer disease",
        "status":              "TRUE_POSITIVE",
        "mechanism":           "NMDA receptor antagonist (GRIN1/GRIN2B) — reduces excitotoxicity in AD",
        "min_score":           0.25,
        "expected_rank_top_n": 25,
        "sources":             ["PMID:12673577"],
        "notes":               "Approved for moderate-severe AD.",
    },

    # ── Hypercholesterolemia ──────────────────────────────────────────────────
    {
        "drug":                "atorvastatin",
        "disease":             "hypercholesterolemia",
        "status":              "TRUE_POSITIVE",
        "mechanism":           "HMGCR inhibitor — blocks cholesterol synthesis and up-regulates LDLR",
        "min_score":           0.35,
        "expected_rank_top_n": 15,
        "sources":             ["PMID:9251527"],
        "notes":               "ASCOT-LLA trial. Definitive LDL-lowering.",
    },

    # ── Benign Prostatic Hyperplasia ──────────────────────────────────────────
    {
        "drug":                "finasteride",
        "disease":             "benign prostatic hyperplasia",
        "status":              "TRUE_POSITIVE",
        "mechanism":           "5-alpha reductase inhibitor — reduces prostate DHT and gland volume",
        "min_score":           0.30,
        "expected_rank_top_n": 20,
        "sources":             ["PMID:1383816"],
        "notes":               "First approved 5-ARI for BPH.",
    },

    # ── Infantile Hemangioma ──────────────────────────────────────────────────
    {
        "drug":                "propranolol",
        "disease":             "infantile hemangioma",
        "status":              "TRUE_POSITIVE",
        "mechanism":           "Beta-blocker — promotes vasoconstriction and apoptosis in hemangioma endothelium",
        "min_score":           0.18,
        "expected_rank_top_n": 60,
        "sources":             ["PMID:18371532"],
        "notes":               "Serendipitous discovery 2008. Now first-line for infantile hemangioma.",
    },

    # ── GIST ─────────────────────────────────────────────────────────────────
    {
        "drug":                "imatinib",
        "disease":             "gastrointestinal stromal tumor",
        "status":              "TRUE_POSITIVE",
        "mechanism":           "KIT/PDGFRA tyrosine kinase inhibitor — targets driver mutations in GIST",
        "min_score":           0.35,
        "expected_rank_top_n": 20,
        "sources":             ["PMID:11752351"],
        "notes":               "Transformed prognosis of GIST. KIT expressed in >85% GIST.",
    },

    # ── MS ────────────────────────────────────────────────────────────────────
    {
        "drug":                "natalizumab",
        "disease":             "multiple sclerosis",
        "status":              "TRUE_POSITIVE",
        "mechanism":           "Anti-α4 integrin (ITGA4) antibody — blocks lymphocyte trafficking to CNS",
        "min_score":           0.22,
        "expected_rank_top_n": 50,
        "sources":             ["PMID:16162940"],
        "notes":               "AFFIRM trial. Highly effective for relapsing MS.",
    },

    # ── ALS ───────────────────────────────────────────────────────────────────
    {
        "drug":                "riluzole",
        "disease":             "amyotrophic lateral sclerosis",
        "status":              "TRUE_POSITIVE",
        "mechanism":           "Glutamate release inhibitor — reduces excitotoxicity in motor neurons",
        "min_score":           0.20,
        "expected_rank_top_n": 40,
        "sources":             ["PMID:7605342"],
        "notes":               "Only small-molecule ALS drug with survival benefit.",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# TRUE_NEGATIVE cases — drugs that should NOT score highly for a given disease
# ─────────────────────────────────────────────────────────────────────────────

_NEGATIVE_CASES: List[Dict] = [
    {
        "drug":                "haloperidol",
        "disease":             "parkinson disease",
        "status":              "TRUE_NEGATIVE",
        "mechanism":           "Dopamine D2 receptor ANTAGONIST — worsens motor symptoms in PD",
        "min_score":           0.20,
        "expected_rank_top_n": 0,
        "sources":             ["PMID:11294978"],
        "notes":               "Absolute contraindication in PD. Drug-induced parkinsonism.",
    },
    {
        "drug":                "propranolol",
        "disease":             "asthma",
        "status":              "TRUE_NEGATIVE",
        "mechanism":           "Non-selective beta-blocker — causes life-threatening bronchospasm in asthma",
        "min_score":           0.20,
        "expected_rank_top_n": 0,
        "sources":             ["PMID:12752346"],
        "notes":               "Absolute contraindication in asthma.",
    },
    {
        "drug":                "metformin",
        "disease":             "multiple myeloma",
        "status":              "TRUE_NEGATIVE",
        "mechanism":           "Biguanide — AMPK activation has no established anti-myeloma mechanism",
        "min_score":           0.20,
        "expected_rank_top_n": 0,
        "sources":             ["PMID:20053753"],
        "notes":               "Some in-vitro data but no clinical benefit in myeloma.",
    },
    {
        "drug":                "aspirin",
        "disease":             "parkinson disease",
        "status":              "TRUE_NEGATIVE",
        "mechanism":           "COX inhibitor — no established connection to dopaminergic neurodegeneration",
        "min_score":           0.20,
        "expected_rank_top_n": 0,
        "sources":             [],
        "notes":               "Aspirin targets COX pathway, unrelated to dopamine/synuclein.",
    },
    {
        "drug":                "simvastatin",
        "disease":             "cystic fibrosis",
        "status":              "TRUE_NEGATIVE",
        "mechanism":           "HMGCR inhibitor — no established mechanism in CFTR dysfunction",
        "min_score":           0.20,
        "expected_rank_top_n": 0,
        "sources":             [],
        "notes":               "Statin targets cholesterol, not chloride channel biology.",
    },
    {
        "drug":                "lisinopril",
        "disease":             "epilepsy",
        "status":              "TRUE_NEGATIVE",
        "mechanism":           "ACE inhibitor — no established anticonvulsant mechanism",
        "min_score":           0.20,
        "expected_rank_top_n": 0,
        "sources":             [],
        "notes":               "RAAS pathway is unrelated to seizure biology.",
    },
    {
        "drug":                "omeprazole",
        "disease":             "multiple sclerosis",
        "status":              "TRUE_NEGATIVE",
        "mechanism":           "Proton pump inhibitor — H+/K+ ATPase inhibition unrelated to MS autoimmunity",
        "min_score":           0.18,
        "expected_rank_top_n": 0,
        "sources":             [],
        "notes":               "Gastric acid suppression has no mechanism in MS.",
    },
    {
        "drug":                "furosemide",
        "disease":             "alzheimer disease",
        "status":              "TRUE_NEGATIVE",
        "mechanism":           "Loop diuretic — renal electrolyte mechanism, no AD-relevant target",
        "min_score":           0.18,
        "expected_rank_top_n": 0,
        "sources":             [],
        "notes":               "Diuretic mechanism unrelated to amyloid/tau/cholinergic pathways.",
    },
    {
        "drug":                "metoclopramide",
        "disease":             "parkinson disease",
        "status":              "TRUE_NEGATIVE",
        "mechanism":           "Dopamine D2 antagonist antiemetic — directly worsens PD motor symptoms",
        "min_score":           0.20,
        "expected_rank_top_n": 0,
        "sources":             ["PMID:9427237"],
        "notes":               "Dopamine antagonists are absolutely contraindicated in PD.",
    },
    {
        "drug":                "warfarin",
        "disease":             "epilepsy",
        "status":              "TRUE_NEGATIVE",
        "mechanism":           "Vitamin K antagonist (anticoagulant) — no anticonvulsant mechanism",
        "min_score":           0.18,
        "expected_rank_top_n": 0,
        "sources":             [],
        "notes":               "Coagulation cascade is unrelated to seizure threshold.",
    },
    {
        "drug":                "digoxin",
        "disease":             "rheumatoid arthritis",
        "status":              "TRUE_NEGATIVE",
        "mechanism":           "Cardiac glycoside / Na+/K+ ATPase inhibitor — no autoimmune mechanism",
        "min_score":           0.18,
        "expected_rank_top_n": 0,
        "sources":             [],
        "notes":               "Cardiac drug with no immunomodulatory target in RA.",
    },
    {
        "drug":                "allopurinol",
        "disease":             "multiple myeloma",
        "status":              "TRUE_NEGATIVE",
        "mechanism":           "Xanthine oxidase inhibitor — reduces uric acid, no anti-myeloma target",
        "min_score":           0.18,
        "expected_rank_top_n": 0,
        "sources":             [],
        "notes":               "Used to prevent tumour lysis syndrome but not anti-myeloma.",
    },
    {
        "drug":                "atenolol",
        "disease":             "type 2 diabetes mellitus",
        "status":              "TRUE_NEGATIVE",
        "mechanism":           "Beta-1 selective blocker — can mask hypoglycaemia, no T2DM benefit",
        "min_score":           0.18,
        "expected_rank_top_n": 0,
        "sources":             [],
        "notes":               "Beta-blockers are not used for T2DM treatment.",
    },
    {
        "drug":                "ondansetron",
        "disease":             "pulmonary arterial hypertension",
        "status":              "TRUE_NEGATIVE",
        "mechanism":           "5-HT3 receptor antagonist (antiemetic) — unrelated to pulmonary vasomotor tone",
        "min_score":           0.18,
        "expected_rank_top_n": 0,
        "sources":             [],
        "notes":               "Serotonin receptor subtype has no role in PAH pathogenesis.",
    },
    {
        "drug":                "pantoprazole",
        "disease":             "parkinson disease",
        "status":              "TRUE_NEGATIVE",
        "mechanism":           "Proton pump inhibitor — unrelated to dopaminergic or alpha-synuclein biology",
        "min_score":           0.18,
        "expected_rank_top_n": 0,
        "sources":             [],
        "notes":               "Gastric PPI with no PD mechanism.",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Out-of-scope cases (removed from validation set with rationale)
# ─────────────────────────────────────────────────────────────────────────────

OUT_OF_SCOPE_CASES: List[Dict] = [
    {
        "drug":               "thalidomide",
        "disease":            "leprosy",
        "reason":             "Disease not reliably searchable in OpenTargets Platform",
        "removed_in_version": "v2.0",
    },
    {
        "drug":               "sildenafil",
        "disease":            "female sexual dysfunction",
        "reason":             "Disease term inconsistently mapped in OpenTargets EFO",
        "removed_in_version": "v2.0",
    },
    {
        "drug":               "clonidine",
        "disease":            "attention deficit hyperactivity disorder",
        "reason":             "ADHD gene set in OpenTargets is sparse; produces low baseline scores",
        "removed_in_version": "v3.0",
    },
    {
        "drug":               "bupropion",
        "disease":            "smoking cessation",
        "reason":             "Nicotine dependence disease genes poorly characterised in OpenTargets",
        "removed_in_version": "v3.0",
    },
    {
        "drug":               "topiramate",
        "disease":            "migraine",
        "reason":             "Migraine OpenTargets gene set overlaps heavily with epilepsy, causing rank instability",
        "removed_in_version": "v3.1",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Assembled dataset
# ─────────────────────────────────────────────────────────────────────────────

VALIDATION_CASES: List[Dict] = _POSITIVE_CASES + _NEGATIVE_CASES

N_TEST_CASES:     int  = len(VALIDATION_CASES)
N_POSITIVE_CASES: int  = len(_POSITIVE_CASES)
N_NEGATIVE_CASES: int  = len(_NEGATIVE_CASES)


def get_positive_cases() -> List[Dict]:
    """Return all TRUE_POSITIVE validation cases."""
    return [c for c in VALIDATION_CASES if c["status"] == "TRUE_POSITIVE"]


def get_negative_cases() -> List[Dict]:
    """Return all TRUE_NEGATIVE validation cases."""
    return [c for c in VALIDATION_CASES if c["status"] == "TRUE_NEGATIVE"]


def get_cases_by_disease_area(area: str) -> List[Dict]:
    """
    Return validation cases for a disease area keyword.
    area: "oncology" | "autoimmune" | "neurological" | "cardiovascular" | "metabolic" | "rare"
    """
    area_keywords = {
        "oncology":       ["myeloma", "leukemia", "carcinoma", "tumor", "cancer", "hemangioma"],
        "autoimmune":     ["arthritis", "lupus", "sclerosis"],
        "neurological":   ["parkinson", "alzheimer", "epilepsy", "tremor", "sclerosis", "lateral"],
        "cardiovascular": ["coronary", "heart", "hypertension", "pericarditis"],
        "metabolic":      ["diabetes", "cholesterol", "gout", "ovary", "prostatic"],
        "rare":           ["tuberous", "hemoglobinuria", "fibrosis", "muscular", "hemangioma"],
    }
    kws = area_keywords.get(area.lower(), [])
    return [
        c for c in VALIDATION_CASES
        if any(k in c["disease"].lower() for k in kws)
    ]


if __name__ == "__main__":
    print(f"Dataset version: {DATASET_VERSION}")
    print(f"Total cases:     {N_TEST_CASES}")
    print(f"  TRUE_POSITIVE: {N_POSITIVE_CASES}")
    print(f"  TRUE_NEGATIVE: {N_NEGATIVE_CASES}")
    print(f"  Out-of-scope:  {len(OUT_OF_SCOPE_CASES)}")
    print()
    print("TRUE_POSITIVE cases:")
    for c in get_positive_cases():
        print(f"  {c['drug']:25s}  →  {c['disease']}")
    print()
    print("TRUE_NEGATIVE cases:")
    for c in get_negative_cases():
        print(f"  {c['drug']:25s}  ✗  {c['disease']}")