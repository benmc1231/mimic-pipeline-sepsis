"""
Clinical variable definitions for MIMIC-IV sepsis prediction pipeline.

This module defines all itemid mappings, ICD codes, and clinical constants
used across the pipeline. It is the single source of truth for variable
selection decisions. Any change to which variables are included should be
made here and documented in docs/variable_logic.md.

Each constant follows the pattern:
    <NAME>_ITEMIDS          dict mapping clinical label to list of itemids
    <NAME>_ITEMIDS_FLAT     flat list derived from dict, used in SQL IN clauses
    <NAME>_ITEMID_TO_LABEL  reverse map from itemid to label, used for
                            annotating extracted dataframes

Sources:
    chartevents/d_items:                mimiciv_icu
    labevents/d_labitems:               mimiciv_hosp
    inputevents, outputevents,
    procedureevents:                    mimiciv_icu
    diagnoses_icd:                      mimiciv_hosp
"""

# ---------------------------------------------------------------------------
# VITALS
# Source: mimiciv_icu.chartevents
# Used for: SOFA scoring components, early sepsis signal features
# ---------------------------------------------------------------------------

VITAL_ITEMIDS = {
    # Heart Rate - SOFA cardiovascular proxy, tachycardia is early sepsis signal
    "heart_rate": [220045],
    # Arterial Blood Pressure (invasive) - preferred source for MAP calculation
    "abp_systolic": [220050],
    "abp_diastolic": [220051],
    "abp_mean": [220052],
    # Non-invasive Blood Pressure - fallback where arterial line not present
    "nibp_systolic": [220179],
    "nibp_diastolic": [220180],
    "nibp_mean": [220181],
    # Temperature - fever/hypothermia are classic sepsis indicators
    # Both units extracted; Fahrenheit converted to Celsius in clean.py
    "temperature_celsius": [223762],
    "temperature_fahrenheit": [223761],
    # Respiratory Rate - SOFA respiratory component, tachypnoea is early signal
    "respiratory_rate": [220210],
    # SpO2 and FiO2 - combined to derive PF ratio proxy for SOFA respiratory score
    "spo2": [220277],
    "fio2": [223835],
    # GCS components - SOFA neurological score derived by summing all three
    # Raw components used rather than pre-calculated APACHE scores to avoid leakage
    "gcs_eye": [220739],
    "gcs_motor": [223901],
    "gcs_verbal": [223900],
}

VITAL_RANGES = {
    220045: (0, 300),  # Heart rate (bpm)
    220050: (0, 300),  # ABP systolic (mmHg)
    220051: (0, 200),  # ABP diastolic (mmHg)
    220052: (0, 200),  # ABP mean (mmHg)
    220179: (0, 300),  # NIBP systolic (mmHg)
    220180: (0, 200),  # NIBP diastolic (mmHg)
    220181: (0, 200),  # NIBP mean (mmHg)
    223762: (25, 45),  # Temperature Celsius
    223761: (77, 113),  # Temperature Fahrenheit, converted to Celsius in clean.py
    220210: (0, 80),  # Respiratory rate (breaths/min)
    220277: (50, 100),  # SpO2 (%)
    223835: (0.21, 1.0),  # FiO2 (fraction), percentage values standardised in clean.py
    220739: (1, 4),  # GCS Eye
    223901: (1, 6),  # GCS Motor
    223900: (1, 5),  # GCS Verbal
}

VITAL_ITEMID_TO_LABEL = {
    itemid: label for label, itemids in VITAL_ITEMIDS.items() for itemid in itemids
}

VITAL_ITEMIDS_FLAT = [
    itemid for itemids in VITAL_ITEMIDS.values() for itemid in itemids
]

# ---------------------------------------------------------------------------
# LABS
# Source: mimiciv_hosp.labevents
# Used for: SOFA scoring components, sepsis severity markers
# ---------------------------------------------------------------------------

LAB_ITEMIDS = {
    # Creatinine (Blood) - SOFA renal component
    # itemid 50912 selected: Blood/Chemistry fluid. Excludes urine creatinine
    # (51082), serum/urine (51081), and whole blood gas assay (52024)
    "creatinine": [50912],
    # Bilirubin Total (Blood) - SOFA hepatic component
    # itemid 50885 selected: Blood fluid only. Excludes urine, CSF, pleural variants
    "bilirubin_total": [50885],
    # Platelet Count (Blood/Haematology) - SOFA coagulation component
    # itemid 51265 selected. Excludes platelet smear (qualitative) and clumps (artifact)
    "platelet_count": [51265],
    # Lactate (Blood Gas) - not a SOFA component but strong early sepsis marker
    # itemid 50813 selected as primary; highest coverage in MIMIC-IV
    "lactate": [50813],
    # WBC - elevated or severely depressed WBC is a classic infection signal
    "wbc": [51301],
    # Haemoglobin - anaemia context and general severity indicator
    "haemoglobin": [51222],
}

LAB_RANGES = {
    50912: (0.1, 50),  # Creatinine (mg/dL)
    50885: (0.1, 150),  # Bilirubin Total (mg/dL)
    51265: (1, 3000),  # Platelet Count (K/uL)
    50813: (0.1, 30),  # Lactate (mmol/L)
    51301: (0.1, 500),  # WBC (K/uL)
    51222: (0, 25),  # Haemoglobin (g/dL)
}

LAB_ITEMIDS_FLAT = [itemid for itemids in LAB_ITEMIDS.values() for itemid in itemids]

LAB_ITEMID_TO_LABEL = {
    itemid: label for label, itemids in LAB_ITEMIDS.items() for itemid in itemids
}

# ---------------------------------------------------------------------------
# BLOOD CULTURES
# Source: mimiciv_hosp.microbiologyevents
# Used for: Sepsis-3 suspected infection criterion (culture + antibiotic)
# Excluded: neonatal culture type (cohort is adults 18+ only)
# Excluded: post-mortem cultures (outside prediction window)
# ---------------------------------------------------------------------------

BLOOD_CULTURE_SPEC_TYPES = (
    "BLOOD CULTURE",
    "BLOOD CULTURE ( MYCO/F LYTIC BOTTLE)",
)

# ---------------------------------------------------------------------------
# ANTIBIOTIC ADMINISTRATION
# Source: mimiciv_hosp.emar
# Used for: Sepsis-3 suspected infection criterion (culture + antibiotic)
# ---------------------------------------------------------------------------

# Event types confirming drug actually reached the patient.
# Excludes: Not Given, Flushed, Hold Dose, Confirmed (order confirmation only).
# Started/Restarted included to capture IV infusion initiation events.
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

# Antibiotic name patterns for ILIKE matching against emar.medication.
# Pattern-based approach handles MIMIC-IV's inconsistent capitalisation
# (e.g. Vancomycin, VANCOMYCIN, vancomycin all present in data).
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

# Pre-built SQL OR conditions for antibiotic matching, aliased to emar table alias 'e'.
# Injected as f-string into extract_medications query. Safe because ANTIBIOTIC_PATTERNS
# is a hardcoded constant, not user input.
antibiotic_conditions = " OR ".join(
    [f"e.medication ILIKE '%{pattern}%'" for pattern in ANTIBIOTIC_PATTERNS]
)

# ---------------------------------------------------------------------------
# VASOPRESSORS
# Source: mimiciv_icu.inputevents
# Used for: SOFA cardiovascular component (vasopressor requirement = score 3-4)
# ---------------------------------------------------------------------------

VASOPRESSOR_ITEMIDS = {
    "norepinephrine": [221906],  # First-line vasopressor in septic shock
    "epinephrine": [221289, 229617],  # 229617 is duplicate entry with trailing period
    "dopamine": [221662],
    "dobutamine": [221653],  # Inotrope, cardiogenic shock
    "vasopressin": [
        222315
    ],  # Second-line adjunct; units not mg, handled in features.py
    "phenylephrine": [
        221749,
        229632,
        229630,
        229631,
    ],  # Multiple concentrations, same drug
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

# Expected vasopressor rate units for SOFA cardiovascular scoring.
# Vasopressin standardised to units/hour (predominant in MIMIC-IV).
# Unit inconsistencies identified in data profiling are corrected in clean.py.
VASOPRESSOR_EXPECTED_UNITS = {
    221906: "mcg/kg/min",  # Norepinephrine
    221289: "mcg/kg/min",  # Epinephrine
    229617: "mcg/kg/min",  # Epinephrine duplicate
    221662: "mcg/kg/min",  # Dopamine
    221653: "mcg/kg/min",  # Dobutamine
    221749: "mcg/kg/min",  # Phenylephrine
    229632: "mcg/kg/min",  # Phenylephrine (200/250)
    229630: "mcg/kg/min",  # Phenylephrine (50/250)
    229631: "mcg/kg/min",  # Phenylephrine (200/250) OLD
    221986: "mcg/kg/min",  # Milrinone
    222315: "units/hour",  # Vasopressin
}

# ---------------------------------------------------------------------------
# URINE OUTPUT
# Source: mimiciv_icu.outputevents
# Used for: SOFA renal component (alongside creatinine)
# Excluded: OR Urine (226627), PACU Urine (226631) - outside ICU stay
# Excluded: GU Irrigant/Urine Out (227489), Urine+GU Irrigant Out (226566)
#           contaminated with irrigation fluid, not true urine output
# ---------------------------------------------------------------------------

URINE_OUTPUT_ITEMIDS = {
    "foley": [226559],  # Primary - indwelling urinary catheter
    "void": [226560],  # Spontaneous void - non-catheterised patients
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
# VENTILATION
# Source: mimiciv_icu.procedureevents
# Used for: Confirming mechanical ventilation status for PF ratio interpretation
#           in SOFA respiratory scoring
# Note: Ventilator settings (mode, rate, PEEP) are in chartevents, not
#       procedureevents. Only onset/offset events are captured here.
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
# SEPSIS ICD CODES
# Source: mimiciv_hosp.diagnoses_icd
# Used for: Validation cross-check and comorbidity feature derivation
# NOT used as primary cohort exclusion - ICD codes are assigned at discharge
# and carry no onset timestamp. Sepsis-3 derived onset time is used instead.
# See docs/variable_logic.md for full discussion of this decision.
# Excluded: neonatal (P36x), obstetric (O0x, O85), puerperal (6702x)
#           outside adult cohort scope
# ---------------------------------------------------------------------------

SEPSIS_ICD_CODES = {
    # ICD-9
    "sepsis_icd9": "99591",
    "severe_sepsis_icd9": "99592",
    # ICD-10 - Streptococcal
    "streptococcal_sepsis": "A40",
    "streptococcal_sepsis_group_a": "A400",
    "streptococcal_sepsis_group_b": "A401",
    "streptococcal_sepsis_pneumoniae": "A403",
    "other_streptococcal_sepsis": "A408",
    "streptococcal_sepsis_unspecified": "A409",
    # ICD-10 - Other sepsis
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
    # ICD-10 - Severe sepsis/septic shock
    "severe_sepsis_icd10": "R652",
    "severe_sepsis_without_shock": "R6520",
    "severe_sepsis_with_shock": "R6521",
    # ICD-10 - Organism specific
    "salmonella_sepsis": "A021",
    "anthrax_sepsis": "A227",
    "erysipelothrix_sepsis": "A267",
    "listerial_sepsis": "A327",
    "actinomycotic_sepsis": "A427",
    "gonococcal_sepsis": "A5486",
    "candidal_sepsis": "B377",
    # ICD-10 - Post-procedural
    "post_procedural_sepsis": "T8144",
    "post_procedural_sepsis_initial": "T8144XA",
    "post_procedural_sepsis_subsequent": "T8144XD",
    "post_procedural_sepsis_sequela": "T8144XS",
}

SEPSIS_ICD_CODES_FLAT = list(SEPSIS_ICD_CODES.values())

SEPSIS_ICD_TO_LABEL = {v: k for k, v in SEPSIS_ICD_CODES.items()}

# ---------------------------------------------------------------------------
# COMORBIDITY - CHRONIC KIDNEY DISEASE (CKD)
# Source: mimiciv_hosp.diagnoses_icd
# Used for: Individual comorbidity flag and severity staging in comorbidity
#           feature derivation. CKD severity affects baseline creatinine
#           interpretation for SOFA renal scoring.
# Staging: 1 (mild) through 5/ESRD (severe). Unspecified codes set has_ckd
#          True but contribute null to ckd_stage - see docs/variable_logic.md.
# Excluded: obstetric CKD codes (O10x) - pregnancy-specific, out of scope
# Excluded: diabetes-with-CKD combination codes - captured via diabetes flag
# ---------------------------------------------------------------------------

CKD_ICD_CODES = {
    "ckd_stage1": "N181",
    "ckd_stage2": "N182",
    "ckd_stage3a": "N1831",
    "ckd_stage3b": "N1832",
    "ckd_stage3_unspecified": "N1830",
    "ckd_stage3": "N183",
    "ckd_stage4": "N184",
    "ckd_stage5": "N185",
    "ckd_unspecified": "N189",
    "ckd_unspecified_icd9": "5859",
    "ckd_stage1_icd9": "5851",
    "ckd_stage2_icd9": "5852",
    "ckd_stage3_icd9": "5853",
    "ckd_stage4_icd9": "5854",
    "ckd_stage5_icd9": "5855",
    "esrd": "N186",
    "esrd_icd9": "5856",
    "htn_ckd_mild_to_moderate": "I129",
    "htn_ckd_severe_or_esrd": "I120",
    "htn_ckd_unspecified": "I12",
    "htn_ckd_mild_icd9": "40310",
    "htn_ckd_severe_icd9": "40311",
    "htn_ckd_malignant_mild_icd9": "40300",
    "htn_ckd_malignant_severe_icd9": "40301",
    "htn_ckd_unspecified_mild_icd9": "40390",
    "htn_ckd_unspecified_severe_icd9": "40391",
}

CKD_ICD_CODES_FLAT = list(CKD_ICD_CODES.values())

CKD_STAGE_MAP = {
    "N181": 1,
    "5851": 1,
    "N182": 2,
    "5852": 2,
    "N1831": 3,
    "N1832": 3,
    "N1830": 3,
    "N183": 3,
    "5853": 3,
    "N184": 4,
    "5854": 4,
    "N185": 5,
    "5855": 5,
    "N186": 5,
    "5856": 5,  # ESRD treated as equivalent severity to stage 5
    # Unspecified and hypertensive-unspecified codes intentionally excluded
    # from staging. See docs/variable_logic.md.
}

# ---------------------------------------------------------------------------
# COMORBIDITY - CHRONIC LIVER DISEASE
# Source: mimiciv_hosp.diagnoses_icd
# Used for: Individual comorbidity flag and severity staging. Liver disease
#           severity affects baseline bilirubin interpretation for SOFA
#           hepatic scoring.
# Staging: 1 (mild - chronic hepatitis/fibrosis without decompensation)
#          2 (severe - cirrhosis, hepatic failure, portal hypertension,
#             variceal bleeding)
# ---------------------------------------------------------------------------

LIVER_MILD_CODES_ICD9 = ("5715", "5716", "57140", "57149", "07044", "07054")
LIVER_MILD_CODES_ICD10 = (
    "K73",
    "K739",
    "K738",
    "K745",
    "K740",
    "K7400",
    "K7401",
    "K7402",
)
LIVER_SEVERE_CODES_ICD9 = (
    "5712",
    "4560",
    "4561",
    "45620",
    "45621",
    "5723",
    "5719",
    "5728",
)
LIVER_SEVERE_CODES_ICD10 = (
    "K70",
    "K703",
    "K7030",
    "K7031",
    "K704",
    "K7040",
    "K7041",
    "K72",
    "K720",
    "K7200",
    "K7201",
    "K721",
    "K7210",
    "K7211",
    "K729",
    "K7290",
    "K7291",
    "K74",
    "K742",
    "K743",
    "K744",
    "K746",
    "K7460",
    "K7469",
    "K717",
    "K766",
    "K9182",
    "I85",
    "I850",
    "I8500",
    "I8501",
    "I851",
    "I8510",
    "I8511",
)

LIVER_ICD_CODES_FLAT = list(
    LIVER_MILD_CODES_ICD9
    + LIVER_MILD_CODES_ICD10
    + LIVER_SEVERE_CODES_ICD9
    + LIVER_SEVERE_CODES_ICD10
)

LIVER_STAGE_MAP = {
    **{code: 1 for code in LIVER_MILD_CODES_ICD9 + LIVER_MILD_CODES_ICD10},
    **{code: 2 for code in LIVER_SEVERE_CODES_ICD9 + LIVER_SEVERE_CODES_ICD10},
}

# ---------------------------------------------------------------------------
# COMORBIDITY - MALIGNANCY
# Source: mimiciv_hosp.diagnoses_icd
# Used for: Individual comorbidity flag and severity staging. Malignancy
#           affects overall prognosis and baseline inflammatory markers.
# Approach: Prefix matching (not exact codes) due to 1000+ specific codes.
#           Primary malignancy = stage 1, metastatic = stage 2.
#           Matches Charlson/Elixhauser convention.
# Excluded: Screening/follow-up encounters (Z08, Z09, Z12x)
# Excluded: Family history codes (Z80x, V16x)
# Excluded: Genetic susceptibility codes (Z15x, V84x)
# ---------------------------------------------------------------------------

MALIGNANCY_ICD9_PREFIXES = (
    "140",
    "141",
    "142",
    "143",
    "144",
    "145",
    "146",
    "147",
    "148",
    "149",  # head/neck
    "150",
    "151",
    "152",
    "153",
    "154",
    "155",
    "156",
    "157",
    "158",
    "159",  # GI
    "160",
    "161",
    "162",
    "163",
    "164",
    "165",  # respiratory
    "170",
    "171",
    "172",
    "174",
    "175",
    "176",  # bone/skin/breast
    "179",
    "180",
    "181",
    "182",
    "183",
    "184",
    "185",
    "186",
    "187",
    "188",
    "189",  # GU/gynae
    "190",
    "191",
    "192",
    "193",
    "194",
    "195",  # eye/brain/endocrine
    "200",
    "201",
    "202",
    "203",
    "204",
    "205",
    "206",
    "207",
    "208",  # lymphoma/leukaemia
)
METASTATIC_ICD9_PREFIXES = ("196", "197", "198", "199")

MALIGNANCY_ICD10_PREFIXES = (
    "C00",
    "C01",
    "C02",
    "C03",
    "C04",
    "C05",
    "C06",
    "C07",
    "C08",
    "C09",
    "C10",
    "C11",
    "C12",
    "C13",
    "C14",
    "C15",
    "C16",
    "C17",
    "C18",
    "C19",
    "C20",
    "C21",
    "C22",
    "C23",
    "C24",
    "C25",
    "C26",
    "C30",
    "C31",
    "C32",
    "C33",
    "C34",
    "C37",
    "C38",
    "C39",
    "C40",
    "C41",
    "C43",
    "C45",
    "C46",
    "C47",
    "C48",
    "C49",
    "C50",
    "C51",
    "C52",
    "C53",
    "C54",
    "C55",
    "C56",
    "C57",
    "C58",
    "C60",
    "C61",
    "C62",
    "C63",
    "C64",
    "C65",
    "C66",
    "C67",
    "C68",
    "C69",
    "C70",
    "C71",
    "C72",
    "C73",
    "C74",
    "C75",
    "C81",
    "C82",
    "C83",
    "C84",
    "C85",
    "C86",
    "C88",
    "C90",
    "C91",
    "C92",
    "C93",
    "C94",
    "C95",
    "C96",
)
METASTATIC_ICD10_PREFIXES = ("C77", "C78", "C79", "C7B", "C80")

# Combined prefix tuples passed directly to _derive_staged_comorbidity_prefix
# in clean.py. Stage labels passed inline at call site.
MALIGNANCY_ICD_PREFIXES = MALIGNANCY_ICD9_PREFIXES + MALIGNANCY_ICD10_PREFIXES
METASTATIC_ICD_PREFIXES = METASTATIC_ICD9_PREFIXES + METASTATIC_ICD10_PREFIXES

# ---------------------------------------------------------------------------
# COMORBIDITY - DIABETES
# Source: mimiciv_hosp.diagnoses_icd
# Used for: Individual comorbidity flag (boolean only, no staging).
#           Diabetes affects lactate and glucose interpretation.
# Approach: Prefix matching due to 600+ specific ICD codes.
# Excluded: Gestational diabetes (O244x) - pregnancy-specific
# Excluded: Neonatal diabetes (P702, 7751)
# Excluded: Family history and screening codes
# ---------------------------------------------------------------------------

DIABETES_ICD9_PREFIXES = ("250",)
DIABETES_ICD10_PREFIXES = ("E08", "E09", "E10", "E11", "E13")

# Combined prefix tuple passed directly to _derive_binary_comorbidity_flag_prefix
DIABETES_ICD_PREFIXES = DIABETES_ICD9_PREFIXES + DIABETES_ICD10_PREFIXES

# ---------------------------------------------------------------------------
# COMORBIDITY - IMMUNOSUPPRESSION
# Source: mimiciv_hosp.diagnoses_icd
# Used for: Individual comorbidity flag (boolean only, no staging).
#           Immunosuppression affects infection risk and clinical presentation.
# Approach: Exact code matching. No clean severity hierarchy exists across
#           drug-induced, HIV, congenital, and post-transplant causes.
# Excluded: HIV counselling/screening/contact-exposure codes (not confirmed)
# Excluded: Inconclusive serology (R75, 79571)
# Excluded: Pregnancy-complicated HIV (O987x)
# ---------------------------------------------------------------------------

IMMUNOSUPPRESSION_ICD9_CODES = (
    "042",
    "07953",
    "27901",
    "27902",
    "27905",
    "27906",
    "27910",
    "28409",
    "28489",
    "28800",
    "28801",
    "28802",
    "28803",
    "28804",
    "28809",
    "V4983",
    "V5844",
    "V8746",
)
IMMUNOSUPPRESSION_ICD10_CODES = (
    "B20",
    "D610",
    "D611",
    "D612",
    "D613",
    "D61",
    "D70",
    "D80",
    "D81",
    "D82",
    "D83",
    "D84",
    "Z21",
    "Z9481",
    "T86",
    "Z796",
    "Z7960",
    "Z7962",
    "Z79620",
    "Z7969",
    "Z7682",
    "Z9225",
)

# Combined flat list used by _derive_binary_comorbidity_flag (exact match)
IMMUNOSUPPRESSION_ICD_CODES_FLAT = list(
    IMMUNOSUPPRESSION_ICD9_CODES + IMMUNOSUPPRESSION_ICD10_CODES
)
