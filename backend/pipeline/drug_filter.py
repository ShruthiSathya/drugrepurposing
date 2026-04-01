

import logging
import re
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Drug synonym table
# Maps alternate names → canonical lookup key used in contraindication DB.
# Covers ChEMBL preferred names, INN names, and brand names.
# ─────────────────────────────────────────────────────────────────────────────
DRUG_SYNONYMS: Dict[str, str] = {
    # Aspirin variants
    "acetylsalicylic acid": "aspirin",
    "asa":                  "aspirin",
    # Acetaminophen variants
    "paracetamol":          "acetaminophen",
    # Beta-blocker variants
    "propranolol hydrochloride": "propranolol",
    # Antipsychotics
    "haloperidol decanoate":   "haloperidol",
    "olanzapine pamoate":      "olanzapine",
    "risperidone microspheres":"risperidone",
    # Tricyclics
    "amitriptyline hydrochloride": "amitriptyline",
    # Steroids
    "prednisolone":            "prednisone",   # treat as equivalent for safety
    "cortisone":               "hydrocortisone",
    # NSAIDs
    "ibuprofen lysine":        "ibuprofen",
    "naproxen sodium":         "naproxen",
    "indomethacin":            "indomethacin",
    # Antihistamines
    "diphenhydramine hydrochloride": "diphenhydramine",
    "cetirizine":              "cetirizine",   # non-sedating — kept for context
    # Metoclopramide
    "metoclopramide hydrochloride": "metoclopramide",
    # Thiazolidinediones
    "rosiglitazone maleate":   "rosiglitazone",
    "pioglitazone hydrochloride": "pioglitazone",
    # Immunosuppressants
    "tacrolimus monohydrate":  "tacrolimus",
    "cyclosporin":             "cyclosporine",
    "cyclosporin a":           "cyclosporine",
    # Bupropion
    "bupropion hydrochloride": "bupropion",
    # Anticoagulants
    "warfarin sodium":         "warfarin",
    # Digoxin
    "digitoxin":               "digoxin",      # treat similarly
    # Tramadol
    "tramadol hydrochloride":  "tramadol",
    # Clozapine
    "clozaril":                "clozapine",
    # Metformin
    "metformin hydrochloride": "metformin",
}


# ─────────────────────────────────────────────────────────────────────────────
# Mechanism-based class contraindications
# Applied when a drug's mechanism field contains a class keyword,
# even if the drug name is not in the per-drug database.
# Structure: {disease_keyword: [(mechanism_keyword, severity, reason), ...]}
# ─────────────────────────────────────────────────────────────────────────────
MECHANISM_CLASS_CONTRAINDICATIONS: Dict[str, List[Tuple[str, str, str]]] = {
    "parkinson": [
        ("dopamine antagonist",    "absolute",
         "Dopamine antagonists worsen motor symptoms in Parkinson's disease."),
        ("d2 antagonist",          "absolute",
         "D2 receptor blockade exacerbates parkinsonism."),
        ("antipsychotic",          "relative",
         "Most antipsychotics block dopamine — use only quetiapine or clozapine with caution."),
        ("antiemetic",             "absolute",
         "Most antiemetics are dopamine antagonists — metoclopramide, prochlorperazine contraindicated."),
    ],
    "alzheimer": [
        ("anticholinergic",        "absolute",
         "Anticholinergic drugs worsen cognitive function in Alzheimer's disease."),
        ("muscarinic antagonist",  "absolute",
         "Muscarinic blockade impairs memory and cognition."),
        ("sedative",               "relative",
         "Sedatives increase fall risk and worsen confusion in dementia."),
        ("hypnotic",               "relative",
         "Z-drugs and benzodiazepines cause acute confusion in dementia."),
    ],
    "dementia": [
        ("anticholinergic",        "absolute",
         "Anticholinergic drugs worsen cognitive function in dementia."),
        ("muscarinic antagonist",  "absolute",
         "Muscarinic blockade impairs memory and cognition."),
    ],
    "asthma": [
        ("non-selective beta-blocker", "absolute",
         "Non-selective beta-blockers cause life-threatening bronchospasm in asthma."),
        ("beta adrenergic blocker",    "relative",
         "Beta-blockers may cause bronchospasm — use cardioselective only if unavoidable."),
        ("nsaid",                      "relative",
         "NSAIDs trigger aspirin-exacerbated respiratory disease in ~10% of asthmatics."),
    ],
    "epilepsy": [
        ("seizure threshold lowering", "absolute",
         "This drug class lowers seizure threshold."),
    ],
    "heart failure": [
        ("negative inotrope",      "relative",
         "Negative inotropes may worsen cardiac output in heart failure."),
        ("thiazolidinedione",      "absolute",
         "Thiazolidinediones cause fluid retention and are contraindicated in heart failure."),
        ("nsaid",                  "relative",
         "NSAIDs cause sodium retention and worsen heart failure."),
    ],
    "pulmonary arterial hypertension": [
        (
            "vasoconstrictor",
            "absolute",
            "Vasoconstrictors increase pulmonary vascular resistance and worsen PAH.",
        ),
        (
            "alpha-adrenergic agonist",
            "absolute",
            "Alpha-adrenergic agonists cause pulmonary vasoconstriction.",
        ),
        (
            "alpha adrenergic agonist",
            "absolute",
            "Alpha-adrenergic agonists cause pulmonary vasoconstriction.",
        ),
        (
            "sympathomimetic",
            "relative",
            "Sympathomimetics may raise pulmonary arterial pressure.",
        ),
        (
            "ergot alkaloid",
            "absolute",
            "Ergot alkaloids are vasoconstrictors contraindicated in PAH.",
        ),
    ],
    "glaucoma": [
        ("anticholinergic",        "absolute",
         "Anticholinergic drugs precipitate acute angle-closure glaucoma."),
        ("sympathomimetic",        "relative",
         "Sympathomimetics may raise intraocular pressure."),
    ],
    "chronic kidney disease": [
        ("nephrotoxic",            "absolute",
         "Nephrotoxic drugs accelerate CKD progression."),
        ("nsaid",                  "relative",
         "NSAIDs reduce renal blood flow and worsen CKD."),
    ],
    "renal": [
        ("nephrotoxic",            "absolute",
         "Nephrotoxic drugs are contraindicated with impaired renal function."),
        ("nsaid",                  "relative",
         "NSAIDs reduce renal blood flow."),
    ],
}


class DrugSafetyFilter:
    """
    Filters drug candidates based on contraindications for specific diseases.

    Two complementary mechanisms:
      1. Per-drug lookup: exact drug name (with synonym expansion) matched
         against per-disease contraindication database.
      2. Mechanism-class filter: drug's mechanism text matched against
         pharmacological class keywords — catches drugs not in the database
         but belonging to a contraindicated class.
    """

    def __init__(self):
        self.CRITICAL_CONTRAINDICATIONS = self._build_contraindication_database()
        logger.info(
            f"✅ Loaded contraindications for "
            f"{len(self.CRITICAL_CONTRAINDICATIONS)} disease categories"
        )

    # ── Contraindication database ─────────────────────────────────────────────

    def _build_contraindication_database(self) -> Dict[str, Dict[str, Dict]]:
        return {
            # ── Metabolic ─────────────────────────────────────────────────────
            "diabetes": {
                "olanzapine":        {"severity": "absolute",
                    "reason": "Causes significant weight gain and worsens glycemic control.",
                    "mechanism": "Atypical antipsychotic that severely impairs glucose metabolism."},
                "clozapine":         {"severity": "absolute",
                    "reason": "High risk of hyperglycemia and diabetic ketoacidosis.",
                    "mechanism": "Atypical antipsychotic with severe metabolic effects."},
                "quetiapine":        {"severity": "relative",
                    "reason": "Can worsen glycemic control.",
                    "mechanism": "Atypical antipsychotic with metabolic side effects."},
                "risperidone":       {"severity": "relative",
                    "reason": "May impair glucose regulation.",
                    "mechanism": "Atypical antipsychotic."},
                "prednisone":        {"severity": "relative",
                    "reason": "Increases blood glucose levels significantly.",
                    "mechanism": "Corticosteroid promoting gluconeogenesis."},
                "dexamethasone":     {"severity": "relative",
                    "reason": "Severe hyperglycemia risk.",
                    "mechanism": "Potent corticosteroid."},
                "methylprednisolone":{"severity": "relative",
                    "reason": "Elevates blood sugar.",
                    "mechanism": "Corticosteroid."},
                "hydrocortisone":    {"severity": "relative",
                    "reason": "Can destabilize glucose control.",
                    "mechanism": "Corticosteroid."},
                "rosiglitazone":     {"severity": "relative",
                    "reason": "Cardiovascular risk — use pioglitazone cautiously.",
                    "mechanism": "Thiazolidinedione with cardiac risk signals."},
            },

            # ── Neurology — Parkinson's ───────────────────────────────────────
            "parkinson": {
                "haloperidol":       {"severity": "absolute",
                    "reason": "Blocks dopamine receptors, severely worsens motor symptoms.",
                    "mechanism": "Typical antipsychotic — dopamine D2 antagonist."},
                "perphenazine":      {"severity": "absolute",
                    "reason": "Dopamine antagonist exacerbating Parkinson's symptoms.",
                    "mechanism": "Typical antipsychotic."},
                "chlorpromazine":    {"severity": "absolute",
                    "reason": "Severe dopamine blockade.",
                    "mechanism": "Typical antipsychotic."},
                "fluphenazine":      {"severity": "absolute",
                    "reason": "Worsens motor symptoms.",
                    "mechanism": "Typical antipsychotic."},
                "metoclopramide":    {"severity": "absolute",
                    "reason": "Dopamine antagonist causing drug-induced parkinsonism.",
                    "mechanism": "Antiemetic with potent dopamine-blocking effects."},
                "prochlorperazine":  {"severity": "absolute",
                    "reason": "Can precipitate severe motor dysfunction.",
                    "mechanism": "Antiemetic dopamine antagonist."},
                "olanzapine":        {"severity": "relative",
                    "reason": "Some dopamine blockade — less severe than typicals.",
                    "mechanism": "Atypical antipsychotic."},
                "risperidone":       {"severity": "relative",
                    "reason": "May worsen motor symptoms at higher doses.",
                    "mechanism": "Atypical antipsychotic."},
                "droperidol":        {"severity": "absolute",
                    "reason": "Potent dopamine antagonist.",
                    "mechanism": "Butyrophenone antipsychotic."},
            },

            # ── Neurology — Alzheimer's / Dementia ────────────────────────────
            "alzheimer": {
                "diphenhydramine":   {"severity": "absolute",
                    "reason": "Anticholinergic — worsens cognitive function in dementia.",
                    "mechanism": "Blocks acetylcholine, critical for memory."},
                "benztropine":       {"severity": "absolute",
                    "reason": "Strong anticholinergic effects worsen dementia.",
                    "mechanism": "Anticholinergic agent."},
                "oxybutynin":        {"severity": "absolute",
                    "reason": "Anticholinergic — severe cognitive impairment risk.",
                    "mechanism": "Bladder medication with strong anticholinergic effects."},
                "tolterodine":       {"severity": "absolute",
                    "reason": "Anticholinergic for overactive bladder.",
                    "mechanism": "Muscarinic antagonist."},
                "hydroxyzine":       {"severity": "relative",
                    "reason": "Anticholinergic antihistamine.",
                    "mechanism": "Can impair cognition in elderly."},
                "scopolamine":       {"severity": "absolute",
                    "reason": "Potent anticholinergic.",
                    "mechanism": "Causes confusion and acute memory impairment."},
                "cyclobenzaprine":   {"severity": "relative",
                    "reason": "Muscle relaxant with anticholinergic properties.",
                    "mechanism": "Can worsen cognitive function."},
                "amitriptyline":     {"severity": "relative",
                    "reason": "Tricyclic antidepressant with strong anticholinergic effects.",
                    "mechanism": "May impair memory."},
                "trihexyphenidyl":   {"severity": "absolute",
                    "reason": "Highly anticholinergic — causes acute confusional state.",
                    "mechanism": "Anticholinergic antiparkinsonian."},
            },

            # ── Pulmonary — Asthma ────────────────────────────────────────────
            "asthma": {
                "propranolol":       {"severity": "absolute",
                    "reason": "Non-selective beta-blocker — causes life-threatening bronchospasm.",
                    "mechanism": "Blocks beta-2 receptors in airways."},
                "nadolol":           {"severity": "absolute",
                    "reason": "Non-selective beta-blocker.",
                    "mechanism": "Life-threatening bronchospasm risk."},
                "timolol":           {"severity": "absolute",
                    "reason": "Non-selective beta-blocker.",
                    "mechanism": "Even as eye drops can trigger asthma attack."},
                "sotalol":           {"severity": "absolute",
                    "reason": "Non-selective beta-blocker.",
                    "mechanism": "Bronchospasm risk."},
                "atenolol":          {"severity": "relative",
                    "reason": "Beta-1 selective but still risky in severe asthma.",
                    "mechanism": "Can cause bronchospasm at higher doses."},
                "metoprolol":        {"severity": "relative",
                    "reason": "Beta-1 selective blocker.",
                    "mechanism": "Some beta-2 blockade possible."},
                "bisoprolol":        {"severity": "relative",
                    "reason": "Beta-1 selective — caution in severe asthma.",
                    "mechanism": "Risk of bronchospasm."},
                "aspirin":           {"severity": "relative",
                    "reason": "Can trigger aspirin-exacerbated respiratory disease (AERD).",
                    "mechanism": "NSAID-induced bronchospasm in ~10% of asthmatics."},
                "ibuprofen":         {"severity": "relative",
                    "reason": "NSAIDs can worsen asthma.",
                    "mechanism": "Alternative arachidonic acid pathway activation."},
                "naproxen":          {"severity": "relative",
                    "reason": "NSAID with bronchospasm risk.",
                    "mechanism": "Can trigger asthma attacks."},
                "indomethacin":      {"severity": "relative",
                    "reason": "Potent NSAID — high risk of AERD.",
                    "mechanism": "Strong COX inhibitor."},
            },

            # ── Cardiology — Heart Failure ────────────────────────────────────
            "heart failure": {
                "ibuprofen":         {"severity": "relative",
                    "reason": "NSAIDs cause sodium retention and fluid overload.",
                    "mechanism": "Worsens heart failure via renal prostaglandin inhibition."},
                "naproxen":          {"severity": "relative",
                    "reason": "NSAID causing sodium retention.",
                    "mechanism": "Exacerbates heart failure."},
                "indomethacin":      {"severity": "relative",
                    "reason": "Potent NSAID — significant sodium retention.",
                    "mechanism": "Strong COX inhibition."},
                "rosiglitazone":     {"severity": "absolute",
                    "reason": "Causes severe fluid retention and worsens HF.",
                    "mechanism": "Thiazolidinedione — contraindicated in all HF classes."},
                "pioglitazone":      {"severity": "absolute",
                    "reason": "Fluid retention and edema worsen heart failure.",
                    "mechanism": "Thiazolidinedione."},
                "diltiazem":         {"severity": "relative",
                    "reason": "Negative inotrope — may worsen systolic HF.",
                    "mechanism": "Non-dihydropyridine calcium channel blocker."},
                "verapamil":         {"severity": "relative",
                    "reason": "Negative inotrope — contraindicated in systolic HF.",
                    "mechanism": "Non-dihydropyridine calcium channel blocker."},
                "dronedarone":       {"severity": "absolute",
                    "reason": "Doubles mortality in patients with recent decompensated HF.",
                    "mechanism": "Antiarrhythmic with negative inotropic effects."},
            },

            # ── Cardiology — Arrhythmia / QT prolongation ────────────────────
            "arrhythmia": {
                "hydroxychloroquine":{"severity": "relative",
                    "reason": "QT prolongation risk, especially in cardiac patients.",
                    "mechanism": "Blocks hERG potassium channel."},
                "azithromycin":      {"severity": "relative",
                    "reason": "QT prolongation — increased risk with pre-existing arrhythmia.",
                    "mechanism": "Macrolide with hERG channel blockade."},
                "clarithromycin":    {"severity": "relative",
                    "reason": "QT prolongation.",
                    "mechanism": "Macrolide with hERG channel blockade."},
                "haloperidol":       {"severity": "relative",
                    "reason": "QT prolongation — torsades de pointes risk.",
                    "mechanism": "hERG channel blockade."},
            },

            # ── Nephrology — Chronic Kidney Disease ───────────────────────────
            "kidney disease": {
                "metformin":         {"severity": "relative",
                    "reason": "Lactic acidosis risk when eGFR < 30.",
                    "mechanism": "Contraindicated in severe CKD (eGFR < 30 mL/min/1.73m²)."},
                "ibuprofen":         {"severity": "relative",
                    "reason": "NSAIDs worsen renal function by reducing blood flow.",
                    "mechanism": "Prostaglandin inhibition reduces afferent arteriolar dilation."},
                "naproxen":          {"severity": "relative",
                    "reason": "NSAID nephrotoxicity.",
                    "mechanism": "Can precipitate acute kidney injury."},
                "indomethacin":      {"severity": "relative",
                    "reason": "Potent NSAID — high nephrotoxicity risk.",
                    "mechanism": "Strong renal prostaglandin inhibition."},
                "lithium":           {"severity": "relative",
                    "reason": "Nephrotoxic with long-term use.",
                    "mechanism": "Causes nephrogenic diabetes insipidus and tubulointerstitial nephritis."},
                "vancomycin":        {"severity": "relative",
                    "reason": "Nephrotoxic — requires dose adjustment.",
                    "mechanism": "Direct tubular toxicity."},
                "aminoglycoside":    {"severity": "absolute",
                    "reason": "Highly nephrotoxic — contraindicated in significant CKD.",
                    "mechanism": "Proximal tubular cell toxicity."},
                "contrast dye":      {"severity": "relative",
                    "reason": "Contrast-induced nephropathy risk.",
                    "mechanism": "Direct tubular toxicity and vasoconstriction."},
            },

            # ── Ophthalmology — Glaucoma ──────────────────────────────────────
            "glaucoma": {
                "diphenhydramine":   {"severity": "absolute",
                    "reason": "Anticholinergic — precipitates acute angle-closure glaucoma.",
                    "mechanism": "Pupillary dilation blocks aqueous humor drainage."},
                "benztropine":       {"severity": "absolute",
                    "reason": "Strong anticholinergic — contraindicated in angle-closure.",
                    "mechanism": "Raises intraocular pressure."},
                "oxybutynin":        {"severity": "absolute",
                    "reason": "Anticholinergic effects raise intraocular pressure.",
                    "mechanism": "Muscarinic blockade in ciliary muscle."},
                "scopolamine":       {"severity": "absolute",
                    "reason": "Mydriasis precipitates angle closure.",
                    "mechanism": "Potent anticholinergic."},
                "atropine":          {"severity": "absolute",
                    "reason": "Mydriasis — acute angle-closure glaucoma risk.",
                    "mechanism": "Muscarinic antagonist causing pupil dilation."},
                "pseudoephedrine":   {"severity": "relative",
                    "reason": "Sympathomimetic can raise intraocular pressure.",
                    "mechanism": "Alpha-adrenergic agonist."},
            },

            # ── Neurology — Epilepsy ──────────────────────────────────────────
            "epilepsy": {
                "bupropion":         {"severity": "absolute",
                    "reason": "Dose-dependently lowers seizure threshold.",
                    "mechanism": "Dopamine/norepinephrine reuptake inhibitor."},
                "tramadol":          {"severity": "relative",
                    "reason": "Lowers seizure threshold, especially at high doses.",
                    "mechanism": "Weak opioid + serotonin/NE reuptake inhibitor."},
                "clozapine":         {"severity": "relative",
                    "reason": "Dose-dependent seizure risk.",
                    "mechanism": "Lowers seizure threshold via multiple mechanisms."},
                "maprotiline":       {"severity": "absolute",
                    "reason": "High seizure risk.",
                    "mechanism": "Tetracyclic antidepressant."},
                "meperidine":        {"severity": "relative",
                    "reason": "Normeperidine metabolite causes CNS excitability.",
                    "mechanism": "Proconvulsant metabolite accumulation."},
                "theophylline":      {"severity": "relative",
                    "reason": "High concentrations are proconvulsant.",
                    "mechanism": "Adenosine antagonism."},
                "fluoroquinolone":   {"severity": "relative",
                    "reason": "GABA-A receptor antagonism can lower seizure threshold.",
                    "mechanism": "CNS excitation via GABA blockade."},
            },

            # ── Cardiovascular — Hypertension ─────────────────────────────────
            "hypertension": {
                "pseudoephedrine":   {"severity": "relative",
                    "reason": "Sympathomimetic decongestant raises blood pressure.",
                    "mechanism": "Alpha-adrenergic agonist."},
                "phenylephrine":     {"severity": "relative",
                    "reason": "Vasoconstrictor raises blood pressure.",
                    "mechanism": "Alpha-1 adrenergic agonist."},
                "amphetamine":       {"severity": "absolute",
                    "reason": "Significant hypertensive effect.",
                    "mechanism": "Sympathomimetic."},
                "cocaine":           {"severity": "absolute",
                    "reason": "Severe hypertensive crisis risk.",
                    "mechanism": "Sympathomimetic + norepinephrine reuptake inhibitor."},
                "venlafaxine":       {"severity": "relative",
                    "reason": "Dose-dependent blood pressure increase.",
                    "mechanism": "Norepinephrine reuptake inhibition."},
                "epinephrine": {
                    "severity": "absolute",
                    "reason": "Potent vasoconstrictor — contraindicated in pulmonary hypertension.",
                    "mechanism": "Non-selective adrenergic agonist.",
                },
                "norepinephrine": {
                    "severity": "absolute",
                    "reason": "Strong vasopressor — worsens pulmonary hypertension.",
                    "mechanism": "Potent alpha-adrenergic agonist.",
                },
                "epinephrine bitartrate": {
                    "severity": "absolute",
                    "reason": "Vasoconstrictor — contraindicated in hypertension.",
                    "mechanism": "Epinephrine salt.",
                },
            },

            # ── Rheumatology — Gout ───────────────────────────────────────────
            "gout": {
                "aspirin":           {"severity": "relative",
                    "reason": "Low-dose aspirin raises uric acid by blocking renal excretion.",
                    "mechanism": "Inhibits tubular urate secretion at low doses."},
                "hydrochlorothiazide":{"severity": "relative",
                    "reason": "Thiazide diuretics raise serum uric acid.",
                    "mechanism": "Promotes urate reabsorption in proximal tubule."},
                "furosemide":        {"severity": "relative",
                    "reason": "Loop diuretics raise serum uric acid.",
                    "mechanism": "Reduces renal urate clearance."},
                "cyclosporine":      {"severity": "relative",
                    "reason": "Causes hyperuricemia — can precipitate gout.",
                    "mechanism": "Reduces renal uric acid excretion."},
                "pyrazinamide":      {"severity": "relative",
                    "reason": "Strongly raises uric acid.",
                    "mechanism": "Inhibits tubular urate secretion."},
                "niacin":            {"severity": "relative",
                    "reason": "High-dose niacin raises uric acid.",
                    "mechanism": "Competes with urate for tubular excretion."},
            },

            # ── Cardiology — Pericarditis ─────────────────────────────────────
            "pericarditis": {
                "anticoagulant":     {"severity": "relative",
                    "reason": "Risk of hemopericardium in acute pericarditis.",
                    "mechanism": "Anticoagulation can convert effusion to tamponade."},
                "warfarin":          {"severity": "relative",
                    "reason": "Hemopericardium risk in acute pericarditis.",
                    "mechanism": "Anticoagulation in inflamed pericardium."},
                "corticosteroid":    {"severity": "relative",
                    "reason": "Increases recurrence risk when used as initial therapy.",
                    "mechanism": "Paradoxically promotes relapsing pericarditis."},
                "prednisone":        {"severity": "relative",
                    "reason": "Increases relapse rate as first-line therapy.",
                    "mechanism": "Promotes chronicity — use colchicine first."},
            },

            # ── Immunology — Autoimmune ───────────────────────────────────────
            "autoimmune": {
                "live vaccine":      {"severity": "absolute",
                    "reason": "Live vaccines contraindicated during immunosuppressive therapy.",
                    "mechanism": "Risk of vaccine-strain infection."},
                "echinacea":         {"severity": "relative",
                    "reason": "Immune stimulant may worsen autoimmune disease.",
                    "mechanism": "Non-specific immune activation."},
            },

            # ── Pulmonary — Pulmonary Arterial Hypertension ───────────────────
           "pulmonary arterial hypertension": {
                # Sildenafil is APPROVED for PAH — do NOT list it as contraindicated.
                # The nitrate interaction is handled by cyp450_checker.py at combo level.
                "ergotamine": {
                    "severity": "absolute",
                    "reason": "Vasoconstrictor — worsens pulmonary hypertension.",
                    "mechanism": "Serotonin and alpha-adrenergic agonist causes pulmonary vasoconstriction.",
                },
                "ergotamine tartrate": {
                    "severity": "absolute",
                    "reason": "Vasoconstrictor — worsens pulmonary hypertension.",
                    "mechanism": "Ergot alkaloid.",
                },
                "epinephrine": {
                    "severity": "absolute",
                    "reason": "Potent vasoconstrictor — significantly worsens pulmonary vascular resistance.",
                    "mechanism": "Non-selective alpha and beta adrenergic agonist.",
                },
                "norepinephrine": {
                    "severity": "absolute",
                    "reason": "Strong alpha-1 vasoconstriction worsens pulmonary hypertension.",
                    "mechanism": "Potent alpha-adrenergic agonist.",
                },
                "epinephrine bitartrate": {
                    "severity": "absolute",
                    "reason": "Potent vasoconstrictor — worsens PAH.",
                    "mechanism": "Epinephrine salt — same mechanism.",
                },
                "pseudoephedrine": {
                    "severity": "relative",
                    "reason": "Sympathomimetic decongestant raises blood pressure.",
                    "mechanism": "Alpha-adrenergic agonist.",
                },
                "phenylephrine": {
                    "severity": "absolute",
                    "reason": "Vasoconstrictor raises pulmonary vascular resistance.",
                    "mechanism": "Selective alpha-1 agonist.",
                },
                "phenylephrine hydrochloride": {
                    "severity": "absolute",
                    "reason": "Vasoconstrictor raises pulmonary vascular resistance.",
                    "mechanism": "Selective alpha-1 agonist.",
                },
                "amphetamine": {
                    "severity": "absolute",
                    "reason": "Significant hypertensive and vasoconstrictive effect.",
                    "mechanism": "Sympathomimetic — releases norepinephrine.",
                },
                "amphetamine sulfate": {
                    "severity": "absolute",
                    "reason": "Significant hypertensive and vasoconstrictive effect.",
                    "mechanism": "Sympathomimetic.",
                },
                "venlafaxine": {
                    "severity": "relative",
                    "reason": "Dose-dependent blood pressure increase via NE reuptake inhibition.",
                    "mechanism": "SNRI — norepinephrine reuptake inhibition.",
                },
                "venlafaxine hydrochloride": {
                    "severity": "relative",
                    "reason": "Dose-dependent blood pressure increase.",
                    "mechanism": "SNRI.",
                },
            },
            # ── Oncology — General ────────────────────────────────────────────
            "cancer": {
                "live vaccine":      {"severity": "absolute",
                    "reason": "Contraindicated during active chemotherapy.",
                    "mechanism": "Immunosuppression from chemotherapy."},
                "st. john's wort":   {"severity": "absolute",
                    "reason": "CYP3A4 inducer reduces plasma levels of many chemotherapy drugs.",
                    "mechanism": "Herb-drug interaction via CYP3A4/P-gp induction."},
                "hypericum":         {"severity": "absolute",
                    "reason": "St. John's Wort — CYP3A4 inducer.",
                    "mechanism": "Reduces chemotherapy efficacy."},
            },

            # ── Rare disease — Cystic Fibrosis ────────────────────────────────
            "cystic fibrosis": {
                "aminoglycoside":    {"severity": "relative",
                    "reason": "High cumulative nephrotoxicity and ototoxicity risk in CF patients.",
                    "mechanism": "Required for Pseudomonas coverage but requires careful monitoring."},
                "nsaid":             {"severity": "relative",
                    "reason": "NSAIDs have variable benefit/risk in CF lung inflammation.",
                    "mechanism": "High-dose ibuprofen studied in CF — requires specialist guidance."},
            },

            # ── Rare disease — Tuberous Sclerosis ────────────────────────────
            "tuberous sclerosis": {
                "strong cyp3a4 inducer": {"severity": "relative",
                    "reason": "CYP3A4 inducers reduce sirolimus/everolimus plasma levels.",
                    "mechanism": "Reduces efficacy of mTOR inhibitor therapy."},
                "rifampin":          {"severity": "relative",
                    "reason": "Strong CYP3A4 inducer reduces mTOR inhibitor levels.",
                    "mechanism": "Pharmacokinetic interaction."},
                "phenytoin":         {"severity": "relative",
                    "reason": "CYP3A4 inducer reduces mTOR inhibitor levels.",
                    "mechanism": "Pharmacokinetic interaction."},
                "carbamazepine":     {"severity": "relative",
                    "reason": "Strong CYP3A4 inducer — reduces sirolimus efficacy.",
                    "mechanism": "Pharmacokinetic interaction."},
            },

            # ── Haematology — Haemophilia ─────────────────────────────────────
            "hemophilia": {
                "aspirin":           {"severity": "absolute",
                    "reason": "Antiplatelet effect severely worsens bleeding in haemophilia.",
                    "mechanism": "Irreversible COX-1 inhibition impairs platelet function."},
                "ibuprofen":         {"severity": "relative",
                    "reason": "NSAID antiplatelet effect worsens haemophilia bleeding.",
                    "mechanism": "Reversible platelet COX-1 inhibition."},
                "naproxen":          {"severity": "relative",
                    "reason": "NSAID antiplatelet effect.",
                    "mechanism": "COX-1 inhibition."},
                "warfarin":          {"severity": "absolute",
                    "reason": "Anticoagulation in haemophilia: extreme bleeding risk.",
                    "mechanism": "Combined coagulation factor deficiency + anticoagulant."},
                "heparin":           {"severity": "absolute",
                    "reason": "Anticoagulation in haemophilia: extreme bleeding risk.",
                    "mechanism": "Combined coagulation factor deficiency + anticoagulant."},
            },
        }

    # ── Normalization helpers ─────────────────────────────────────────────────

    def _normalize_name(self, name: str) -> str:
        """Lowercase, strip, remove salt/form suffixes."""
        if not name:
            return ""
        n = name.lower().strip()
        # Remove common salt and dosage form suffixes
        n = re.sub(
            r"\s+(sodium|potassium|hydrochloride|hcl|sulfate|tartrate|"
            r"maleate|mesylate|acetate|phosphate|fumarate|succinate|"
            r"monohydrate|dihydrate|anhydrous|extended.release|er|xr|sr|cr)$",
            "", n, flags=re.IGNORECASE,
        )
        return n.strip()

    def _resolve_synonym(self, drug_name: str) -> str:
        """Return canonical lookup key for a drug name."""
        normalized = self._normalize_name(drug_name)
        return DRUG_SYNONYMS.get(normalized, normalized)

    def _find_disease_keys(self, disease_name: str) -> List[str]:
        """Find all matching disease categories for a disease name."""
        normalized = disease_name.lower().strip()
        matching: List[str] = []
        for key in self.CRITICAL_CONTRAINDICATIONS:
            if key in normalized or normalized in key:
                matching.append(key)
            # Specific pattern expansions
            elif key == "diabetes" and ("diabet" in normalized or "hyperglycemi" in normalized):
                matching.append(key)
            elif key == "parkinson" and "parkinson" in normalized:
                matching.append(key)
            elif key == "alzheimer" and ("alzheimer" in normalized or "dementia" in normalized):
                matching.append(key)
            elif key == "dementia" and "dementia" in normalized:
                if key not in matching:
                    matching.append(key)
            elif key == "asthma" and ("asthma" in normalized or "bronchosp" in normalized):
                matching.append(key)
            elif key == "heart failure" and (
                "heart failure" in normalized or "cardiac failure" in normalized
                or "hfref" in normalized or "hfpef" in normalized
            ):
                matching.append(key)
            elif key == "kidney disease" and (
                "kidney" in normalized or "renal" in normalized or "ckd" in normalized
                or "nephrop" in normalized
            ):
                matching.append(key)
            elif key == "glaucoma" and "glaucoma" in normalized:
                matching.append(key)
            elif key == "epilepsy" and ("epilep" in normalized or "seizure" in normalized):
                matching.append(key)
            elif key == "hypertension" and ("hypertension" in normalized or "high blood pressure" in normalized):
                matching.append(key)
            elif key == "gout" and "gout" in normalized:
                matching.append(key)
            elif key == "pericarditis" and "pericarditis" in normalized:
                matching.append(key)
            elif key == "autoimmune" and ("autoimmune" in normalized or "lupus" in normalized
                or "rheumatoid" in normalized or "inflammatory bowel" in normalized):
                matching.append(key)
            elif key == "pulmonary arterial hypertension" and (
                "pulmonary" in normalized and "hypertension" in normalized
            ):
                matching.append(key)
            elif key == "cancer" and (
                "cancer" in normalized or "carcinoma" in normalized
                or "leukemia" in normalized or "lymphoma" in normalized
                or "tumor" in normalized or "tumour" in normalized
                or "myeloma" in normalized or "sarcoma" in normalized
            ):
                matching.append(key)
            elif key == "cystic fibrosis" and "cystic fibrosis" in normalized:
                matching.append(key)
            elif key == "tuberous sclerosis" and "tuberous sclerosis" in normalized:
                matching.append(key)
            elif key == "hemophilia" and ("hemophilia" in normalized or "haemophilia" in normalized):
                matching.append(key)

        return list(dict.fromkeys(matching))  # deduplicate preserving order

    def _check_mechanism_class(
        self, drug: Dict, disease_name: str
    ) -> Optional[Dict]:
        """
        Check if a drug's mechanism field matches a contraindicated class
        for this disease. Returns a contraindication dict or None.
        """
        mechanism = (drug.get("mechanism", "") or "").lower()
        if not mechanism:
            return None

        disease_lower = disease_name.lower()
        for disease_key, class_rules in MECHANISM_CLASS_CONTRAINDICATIONS.items():
            if disease_key not in disease_lower and disease_lower not in disease_key:
                continue
            for mech_keyword, severity, reason in class_rules:
                if mech_keyword in mechanism:
                    return {
                        "severity": severity,
                        "reason":   reason,
                        "mechanism": f"Matched mechanism class: '{mech_keyword}'",
                        "detection": "mechanism_class_filter",
                    }
        return None

    # ── Main public API ───────────────────────────────────────────────────────

    async def filter_candidates(
        self,
        candidates:       List[Dict],
        disease_name:     str,
        remove_absolute:  bool = True,
        remove_relative:  bool = False,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Filter drug candidates based on contraindications for a disease.

        Parameters
        ----------
        candidates : list of dict
            Drug candidates. Each must have 'drug_name' key.
            Optionally 'mechanism' key for class-based filtering.
        disease_name : str
            Name of the target disease.
        remove_absolute : bool
            Remove absolutely contraindicated drugs (default True).
        remove_relative : bool
            Remove relatively contraindicated drugs (default False).

        Returns
        -------
        (safe_candidates, filtered_out) : tuple of lists
        """
        logger.info(
            f"🔍 Safety filter: '{disease_name}' — "
            f"{len(candidates)} candidates, "
            f"remove_absolute={remove_absolute}, remove_relative={remove_relative}"
        )

        disease_keys = self._find_disease_keys(disease_name)
        if disease_keys:
            logger.info(f"   Matched disease categories: {disease_keys}")
        else:
            logger.info(
                f"   No exact disease match for '{disease_name}' — "
                f"applying mechanism-class filter only"
            )

        # Collect per-drug contraindications from the database
        contraindications: Dict[str, Dict] = {}
        for key in disease_keys:
            contraindications.update(self.CRITICAL_CONTRAINDICATIONS[key])

        safe_candidates = []
        filtered_out    = []

        for candidate in candidates:
            drug_name  = candidate.get("drug_name", "")
            canonical  = self._resolve_synonym(drug_name)
            normalized = self._normalize_name(drug_name)

            # Try both the canonical synonym and the raw normalized name
            ci = contraindications.get(canonical) or contraindications.get(normalized)

            # Fallback: mechanism-class filter
            if ci is None:
                ci = self._check_mechanism_class(candidate, disease_name)

            if ci is None:
                safe_candidates.append(candidate)
                continue

            severity      = ci["severity"]
            should_filter = (
                (severity == "absolute" and remove_absolute) or
                (severity == "relative" and remove_relative)
            )

            if should_filter:
                candidate["contraindication"] = ci
                filtered_out.append(candidate)
                logger.warning(
                    f"   ⛔ FILTERED: {drug_name} "
                    f"(severity={severity}, reason={ci['reason'][:80]})"
                )
            else:
                candidate["contraindication_warning"] = ci
                safe_candidates.append(candidate)
                if severity == "absolute":
                    logger.warning(
                        f"   ⚠️  KEPT WITH ABSOLUTE WARNING: {drug_name} — {ci['reason'][:60]}"
                    )
                else:
                    logger.info(
                        f"   ⚠️  KEPT WITH RELATIVE WARNING: {drug_name} — {ci['reason'][:60]}"
                    )

        logger.info(
            f"✅ Filter complete: {len(safe_candidates)} safe, "
            f"{len(filtered_out)} filtered out"
        )
        return safe_candidates, filtered_out

    def get_contraindications_for_disease(self, disease_name: str) -> Dict[str, Dict]:
        """Return all per-drug contraindications for a disease."""
        disease_keys = self._find_disease_keys(disease_name)
        result: Dict[str, Dict] = {}
        for key in disease_keys:
            result.update(self.CRITICAL_CONTRAINDICATIONS[key])
        return result

    def get_mechanism_warnings_for_disease(self, disease_name: str) -> List[Dict]:
        """Return all mechanism-class contraindications for a disease."""
        disease_lower = disease_name.lower()
        warnings = []
        for disease_key, class_rules in MECHANISM_CLASS_CONTRAINDICATIONS.items():
            if disease_key in disease_lower or disease_lower in disease_key:
                for mech_keyword, severity, reason in class_rules:
                    warnings.append({
                        "mechanism_class": mech_keyword,
                        "severity":        severity,
                        "reason":          reason,
                    })
        return warnings