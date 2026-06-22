"""
Clinical variable definitions for MIMIC-IV sepsis prediction pipeline.

This module defines all itemid mappings, ICD codes, and clinical constants
used across the pipeline. It is the single source of truth for variable
selection decisions — any change to which variables are included should be
made here and documented in docs/variable_logic.md.

Each constant follows the pattern:
    <NAME>_ITEMIDS      — dict mapping clinical label to list of itemids
    <NAME>_ITEMIDS_FLAT — flat list derived from dict, used in SQL IN clauses
    <NAME>_ITEMID_TO_LABEL — reverse map from itemid to label, used for
                             annotating extracted dataframes

Sources:
    - chartevents/d_items: mimiciv_icu
    - labevents/d_labitems: mimiciv_hosp
    - inputevents, outputevents, procedureevents: mimiciv_icu
    - diagnoses_icd: mimiciv_hosp
"""

# ---------------------------------------------------------------------------
# VITALS — from mimiciv_icu.chartevents
# Used for: SOFA scoring components, early sepsis signal features
# ---------------------------------------------------------------------------

VITAL_ITEMIDS = {
    # Heart Rate — SOFA cardiovascular proxy, tachycardia is early sepsis signal
    "heart_rate": [220045],
    # Arterial Blood Pressure (invasive) — preferred source for MAP calculation
    "abp_systolic": [220050],
    "abp_diastolic": [220051],
    "abp_mean": [220052],
    # Non-invasive Blood Pressure — fallback where arterial line not present
    "nibp_systolic": [220179],
    "nibp_diastolic": [220180],
    "nibp_mean": [220181],
    # Temperature — fever/hypothermia are classic sepsis indicators
    # Both units extracted; Fahrenheit converted to Celsius in clean.py
    "temperature_celsius": [223762],
    "temperature_fahrenheit": [223761],
    # Respiratory Rate — SOFA respiratory component, tachypnoea is early signal
    "respiratory_rate": [220210],
    # SpO2 and FiO2 — combined to derive PF ratio proxy for SOFA respiratory score
    "spo2": [220277],
    "fio2": [223835],
    # GCS components — SOFA neurological score derived by summing all three
    # Raw components used rather than pre-calculated APACHE scores to avoid leakage
    "gcs_eye": [220739],
    "gcs_motor": [223901],
    "gcs_verbal": [223900],
}

VITAL_ITEMID_TO_LABEL = {
    itemid: label for label, itemids in VITAL_ITEMIDS.items() for itemid in itemids
}

VITAL_ITEMIDS_FLAT = [
    itemid for itemids in VITAL_ITEMIDS.values() for itemid in itemids
]

# ---------------------------------------------------------------------------
# LABS — from mimiciv_hosp.labevents
# Used for: SOFA scoring components, sepsis severity markers
# ---------------------------------------------------------------------------

LAB_ITEMIDS = {
    # Creatinine (Blood) — SOFA renal component
    # itemid 50912 selected: Blood/Chemistry fluid. Excludes urine creatinine
    # (51082), serum/urine (51081), and whole blood gas assay (52024)
    "creatinine": [50912],
    # Bilirubin Total (Blood) — SOFA hepatic component
    # itemid 50885 selected: Blood fluid only. Excludes urine, CSF, pleural variants
    "bilirubin_total": [50885],
    # Platelet Count (Blood/Haematology) — SOFA coagulation component
    # itemid 51265 selected. Excludes platelet smear (qualitative) and clumps (artifact)
    "platelet_count": [51265],
    # Lactate (Blood Gas) — not a SOFA component but strong early sepsis marker
    # itemid 50813 selected as primary; highest coverage in MIMIC-IV
    "lactate": [50813],
    # WBC — elevated or severely depressed WBC is a classic infection signal
    "wbc": [51301],
    # Haemoglobin — anaemia context and general severity indicator
    "haemoglobin": [51222],
}

LAB_ITEMIDS_FLAT = [itemid for itemids in LAB_ITEMIDS.values() for itemid in itemids]

LAB_ITEMID_TO_LABEL = {
    itemid: label for label, itemids in LAB_ITEMIDS.items() for itemid in itemids
}

# ---------------------------------------------------------------------------
# BLOOD CULTURES — from mimiciv_hosp.microbiologyevents
# Used for: Sepsis-3 suspected infection criterion (culture + antibiotic)
# Neonatal culture type excluded — cohort is adults ≥18 only
# Post-mortem cultures excluded — outside prediction window
# ---------------------------------------------------------------------------

BLOOD_CULTURE_SPEC_TYPES = (
    "BLOOD CULTURE",
    "BLOOD CULTURE ( MYCO/F LYTIC BOTTLE)",
)

# ---------------------------------------------------------------------------
# ANTIBIOTIC ADMINISTRATION — from mimiciv_hosp.emar
# Used for: Sepsis-3 suspected infection criterion (culture + antibiotic)
# ---------------------------------------------------------------------------

# Event types confirming drug actually reached the patient
# Excludes: Not Given, Flushed, Hold Dose, Confirmed (order confirmation only)
# Started/Restarted included to capture IV infusion initiation events
ADMINISTERED_EVENT_TYPES = (
    "Administered",
    "Administered in Other Location",
    "Administered Bolus from IV Drip",
    "Delayed Administered",
    "Partial Administered",
    "Started",  # IV infusion initiation
    "Started in Other Location",
    "Delayed Started",
    "Restarted",  # Restarted after interruption
)

# Antibiotic name patterns for ILIKE matching against emar.medication
# Pattern-based approach handles MIMIC-IV's inconsistent capitalisation
# (e.g. Vancomycin, VANCOMYCIN, vancomycin all present in data)
ANTIBIOTIC_PATTERNS = (
    "vancomycin",
    "piperacillin",
    "meropenem",
    "ceftriaxone",
    "ciprofloxacin",
    "metronidazole",
    "ampicillin",
    "levofloxacin",
    "azithromycin",
    "clindamycin",
    "daptomycin",
    "linezolid",
    "nafcillin",
    "oxacillin",
    "doxycycline",
    "tobramycin",
    "penicillin",
    "amoxicillin",
    "tigecycline",
    "ceftolozane",
    "moxifloxacin",
)

# Pre-built SQL OR conditions for antibiotic matching, aliased to emar table alias 'e'
# Injected as f-string into extract_medications query — safe as patterns are
# hardcoded constants, not user input
antibiotic_conditions = " OR ".join(
    [f"e.medication ILIKE '%{pattern}%'" for pattern in ANTIBIOTIC_PATTERNS]
)

# ---------------------------------------------------------------------------
# VASOPRESSORS — from mimiciv_icu.inputevents
# Used for: SOFA cardiovascular component (vasopressor requirement = score 3-4)
# ---------------------------------------------------------------------------

VASOPRESSOR_ITEMIDS = {
    "norepinephrine": [221906],  # First-line vasopressor in septic shock
    # 229617 is a duplicate entry with trailing period — same drug
    "epinephrine": [221289, 229617],
    "dopamine": [221662],
    "dobutamine": [221653],  # Inotrope — cardiogenic shock
    "vasopressin": [222315],  # Second-line adjunct; units not mg — handle in features.py
    # Multiple itemids reflect different pre-mixed concentrations of same drug
    "phenylephrine": [221749, 229632, 229630, 229631],
    "milrinone": [221986],  # Phosphodiesterase inhibitor inotrope
}

VASOPRESSOR_ITEMIDS_FLAT = [
    itemid for itemids in VASOPRESSOR_ITEMIDS.values() for itemid in itemids
]

VASOPRESSOR_ITEMID_TO_LABEL = {
    itemid: label
    for label, itemids in VASOPRESSOR_ITEMIDS.items()
    for itemid in itemids
}

# ---------------------------------------------------------------------------
# URINE OUTPUT — from mimiciv_icu.outputevents
# Used for: SOFA renal component (alongside creatinine)
# Excluded: OR Urine (226627), PACU Urine (226631) — outside ICU stay
# Excluded: GU Irrigant/Urine Out (227489), Urine+GU Irrigant Out (226566)
#           — contaminated with irrigation fluid, not true urine output
# ---------------------------------------------------------------------------

URINE_OUTPUT_ITEMIDS = {
    "foley": [226559],  # Primary — indwelling urinary catheter
    "void": [226560],  # Spontaneous void — non-catheterised patients
    "condom_cath": [226561],  # Condom catheter
    "straight_cath": [226567],  # Intermittent catheterisation
    "suprapubic": [226563],  # Suprapubic catheter
    "ileoconduit": [226584],  # Urinary diversion
    "l_ureteral_stent": [226558],
    "r_ureteral_stent": [226557],
    "l_nephrostomy": [226565],
    "r_nephrostomy": [226564],
}

URINE_OUTPUT_ITEMIDS_FLAT = [
    itemid for itemids in URINE_OUTPUT_ITEMIDS.values() for itemid in itemids
]

URINE_OUTPUT_ITEMID_TO_LABEL = {
    itemid: label
    for label, itemids in URINE_OUTPUT_ITEMIDS.items()
    for itemid in itemids
}

# ---------------------------------------------------------------------------
# VENTILATION — from mimiciv_icu.procedureevents
# Used for: Confirming mechanical ventilation status for PF ratio interpretation
# in SOFA respiratory scoring
# Note: Ventilator settings (mode, type, tank levels) are in chartevents,
# not procedureevents — only onset/offset events are captured here
# ---------------------------------------------------------------------------

VENTILATION_ITEMIDS = {
    "invasive_ventilation": [225792],
    "non_invasive_ventilation": [225794],
}

VENTILATION_ITEMIDS_FLAT = [
    itemid for itemids in VENTILATION_ITEMIDS.values() for itemid in itemids
]

VENTILATION_ITEMID_TO_LABEL = {
    itemid: label
    for label, itemids in VENTILATION_ITEMIDS.items()
    for itemid in itemids
}

# ---------------------------------------------------------------------------
# SEPSIS ICD CODES — from mimiciv_hosp.diagnoses_icd
# Used for: Validation cross-check and comorbidity feature derivation
# NOT used as primary cohort exclusion — ICD codes are assigned at discharge
# and carry no onset timestamp. Sepsis-3 derived onset time is used instead.
# See docs/variable_logic.md for full discussion of this decision.
# Excluded: neonatal (P36x), obstetric (O0x, O85), puerperal (6702x) —
# outside adult cohort scope
# ---------------------------------------------------------------------------

SEPSIS_ICD_CODES = {
    # ICD-9
    "sepsis_icd9": "99591",
    "severe_sepsis_icd9": "99592",
    # ICD-10 — Streptococcal
    "streptococcal_sepsis": "A40",
    "streptococcal_sepsis_group_a": "A400",
    "streptococcal_sepsis_group_b": "A401",
    "streptococcal_sepsis_pneumoniae": "A403",
    "other_streptococcal_sepsis": "A408",
    "streptococcal_sepsis_unspecified": "A409",
    # ICD-10 — Other sepsis
    "other_sepsis": "A41",
    "sepsis_staph_aureus": "A410",
    "sepsis_mssa": "A4101",
    "sepsis_mrsa": "A4102",
    "sepsis_other_staph": "A411",
    "sepsis_unspecified_staph": "A412",
    "sepsis_haemophilus": "A413",
    "sepsis_anaerobes": "A414",
    "sepsis_gram_negative": "A415",
    "sepsis_gram_negative_unspecified": "A4150",
    "sepsis_ecoli": "A4151",
    "sepsis_pseudomonas": "A4152",
    "sepsis_serratia": "A4153",
    "other_gram_negative_sepsis": "A4159",
    "other_specified_sepsis": "A418",
    "sepsis_enterococcus": "A4181",
    "other_specified_sepsis_2": "A4189",
    "sepsis_unspecified_organism": "A419",
    # ICD-10 — Severe sepsis/septic shock
    "severe_sepsis_icd10": "R652",
    "severe_sepsis_without_shock": "R6520",
    "severe_sepsis_with_shock": "R6521",
    # ICD-10 — Organism specific
    "salmonella_sepsis": "A021",
    "anthrax_sepsis": "A227",
    "erysipelothrix_sepsis": "A267",
    "listerial_sepsis": "A327",
    "actinomycotic_sepsis": "A427",
    "gonococcal_sepsis": "A5486",
    "candidal_sepsis": "B377",
    # ICD-10 — Post-procedural
    "post_procedural_sepsis": "T8144",
    "post_procedural_sepsis_initial": "T8144XA",
    "post_procedural_sepsis_subsequent": "T8144XD",
    "post_procedural_sepsis_sequela": "T8144XS",
}

SEPSIS_ICD_CODES_FLAT = list(SEPSIS_ICD_CODES.values())

SEPSIS_ICD_TO_LABEL = {v: k for k, v in SEPSIS_ICD_CODES.items()}
