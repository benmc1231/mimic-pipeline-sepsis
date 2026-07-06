# MIMIC-IV Early-Onset Sepsis Prediction Pipeline

## Project Summary

This project builds a clinical prediction model for early-onset sepsis in ICU patients using the MIMIC-IV electronic health record database. Using only data available in the first 6 hours of an ICU admission, the pipeline identifies patients at elevated risk of developing sepsis before clinical deterioration becomes apparent, giving clinical teams actionable lead time for earlier intervention.

The project is structured as a miniature research pipeline: clinically justified cohort definition, leakage-safe feature engineering, multi-model comparison with calibration and threshold analysis, and a clinical dashboard suitable for presentation to non-technical stakeholders. The engineering layer mirrors production clinical data infrastructure, reproducible, documented, and containerised.

---

## Prediction Question

### Early-Onset Sepsis Prediction in ICU Admissions

**Clinical Relevance**

Sepsis is defined under the Sepsis-3 consensus as life-threatening organ dysfunction caused by a dysregulated host response to infection. It remains one of the leading causes of ICU mortality, contributing to an estimated 20% of global deaths annually. Sepsis is not a single discrete event but a trajectory: organ dysfunction accumulates over hours, and intervention timing is the primary modifiable determinant of outcome. The Surviving Sepsis Campaign's Hour-1 Bundle reflects this directly. Earlier antibiotics, earlier fluid resuscitation, and earlier vasopressor initiation measurably reduce mortality.

Sepsis recognition at the point of clinical deterioration is already well supported by bedside tools. qSOFA, SIRS criteria, and clinical education campaigns are widespread. This project addresses a distinct and harder problem: predicting sepsis onset in ICU patients who are not yet septic at admission, using only data available in the first 6 hours of their ICU stay, before physiological deterioration becomes clinically apparent. A model that flags elevated risk 6 to 18 hours before onset gives clinical teams lead time that bedside observation alone cannot reliably provide.

**Feasibility Given MIMIC-IV Structure**

MIMIC-IV is well suited to this question. The ICU module provides high-frequency chartevents covering vitals and nursing observations. The hospital module provides laboratory measurements, microbiology cultures, and medication administration, all indexed to ICU stay and hospital admission respectively. Together these tables contain the components required to derive Sepsis-3 criteria:

- Suspected infection: blood culture order (microbiologyevents) combined with antibiotic administration (emar)
- Organ dysfunction: SOFA score increase of 2 or more, derivable from chartevents (GCS, MAP, SpO2/FiO2 ratio) and labevents (creatinine, bilirubin, platelet count)

MIMIC-IV v3.1 contains 94,458 ICU stays across 65,366 unique individuals, sufficient to define a meaningful prediction cohort after exclusions and support robust model training, validation and testing.

**Temporal Framing**

Temporal integrity is the central engineering challenge of this prediction task.

| Component | Definition |
|-----------|------------|
| Unit of analysis | ICU stay (stay_id), each treated independently. Train/test split performed at patient level (subject_id) to prevent leakage across stays for the same patient |
| Cohort | Adult ICU admissions (age 18 or older), not septic at ICU admission |
| Index time | ICU admission (hour 0) |
| Observation window | Hours 0 to 6: all features derived exclusively from data within this window |
| Prediction target | Sepsis onset between hours 6 and 24 post-admission |
| Exclusions | Sepsis present on admission, ICU stay under 6 hours, age under 18 |

All features are computed strictly from the observation window. No data from after hour 6 enters the feature set under any condition. Patients meeting Sepsis-3 criteria at or before hour 6 are excluded entirely as they represent a recognition problem rather than a prediction problem. This is the primary leakage risk in sepsis prediction models and is enforced explicitly in the feature engineering pipeline.

Each ICU stay is treated as an independent analytical unit. Restricting analysis to a patient's first recorded stay would not reliably identify their true first-ever ICU admission. MIMIC-IV captures admissions within a defined period, not a complete longitudinal record. Sepsis onset risk is a property of the current admission rather than ICU history, and excluding subsequent stays would introduce selection bias without defensible clinical justification. Non-independence across stays for the same patient is addressed by splitting at the patient level rather than the stay level.

**Why This Question Matters to Target Organisations**

The organisations this project targets, clinical research institutes, genomics and precision medicine groups, and health AI teams, share a common need: deriving clinically meaningful signal from complex longitudinal health data under rigorous methodological constraints. Sepsis-3 onset prediction requires cohort definition from messy real-world EHR data, temporal reasoning, leakage-safe feature engineering, and model evaluation framed in clinical rather than purely statistical terms. The question is well established in the literature and benchmarkable, but complex enough in its engineering requirements to demonstrate genuine data science maturity beyond standard classification tasks.

---

## Architecture

*Diagram to be added. See /docs/architecture.md*

---

## Pipeline Overview

### Phase 1 — Data Engineering (pipeline/)

#### extract.py

The first stage of the pipeline. Its sole responsibility is pulling raw data from PostgreSQL and writing it to disk as versioned parquet files. No transformation, imputation, or feature engineering is performed here. Keeping extraction pure means cleaning and feature decisions can be iterated on without re-querying the database, which matters on a 432 million row chartevents table.

Eight extraction functions run in sequence, each filtering to cohort stays only via PostgreSQL temp table joins. Functions check for an existing parquet file before querying and load from cache if found.

**Cohort Backbone**
Joins icustays to admissions and patients using INNER JOINs. Produces one row per ICU stay with demographics and admission metadata. This is the spine every subsequent extract filters against. No inclusion or exclusion criteria are applied here.

**Vitals**
Pulls chartevents filtered to: heart rate, arterial and non-invasive blood pressure, temperature (both Celsius and Fahrenheit), respiratory rate, SpO2, FiO2, and GCS components (eye, motor, verbal). Warning-flagged values excluded at source. Raw observations only, no aggregation or window filtering.

**Labs**
Pulls labevents filtered to: creatinine, bilirubin total, platelet count, lactate, WBC, and haemoglobin. Joined via hadm_id as labevents is a hospital-level table with no stay_id. All values extracted regardless of abnormal flag since normal lab values are informative for SOFA scoring and trend analysis.

**Infection Components**
Pulls microbiologyevents filtered to blood culture specimen types. Blood cultures form one half of the Sepsis-3 suspected infection criterion. The temporal relationship between cultures and antibiotics is evaluated in features.py.

**Medications**
Pulls emar filtered to antibiotic administrations. Pattern-based ILIKE matching handles MIMIC-IV's inconsistent medication name capitalisation. Ophthalmic, topical, and vaginal preparations are excluded along with heparin locks as these are non-systemic routes not relevant to sepsis treatment. Filtered to administration-confirming event types only.

**Vasopressors**
Pulls inputevents filtered to: norepinephrine, epinephrine, dopamine, dobutamine, vasopressin, phenylephrine, milrinone. Vasopressor requirement is the SOFA cardiovascular component. Rate and amount fields are retained for dose-based SOFA scoring. Vasopressin is recorded in units rather than mg and this is handled in features.py.

**Urine Output**
Pulls outputevents filtered to urine output itemids covering all catheter and voiding types. OR and PACU urine are excluded as these fall outside the ICU stay. GU irrigant volumes are excluded as they are contaminated with irrigation fluid rather than true urine output.

**Ventilation Events**
Pulls procedureevents filtered to invasive and non-invasive mechanical ventilation. Ventilation status is required for correct PF ratio interpretation in SOFA respiratory scoring. Ventilator settings are captured via chartevents in the vitals extract.

**Diagnosis Codes**
Pulls all ICD-9 and ICD-10 diagnosis codes for cohort admissions. All codes are extracted rather than filtered to support comorbidity feature derivation (Charlson, Elixhauser indices) and cross-validation of Sepsis-3 derived onset labels. ICD codes are not used as the primary cohort exclusion mechanism since they are assigned at discharge with no onset timestamp. See docs/variable_logic.md for full discussion.

> **Data quality note:** MIMIC-IV data is provided as-collected without cleaning, as noted in the official documentation. Implausible physiological values are present in the raw data. Value range filters are applied in clean.py using clinically defined thresholds documented in docs/variable_logic.md.

---

*Phases 2 to 5 documentation to be added on completion.*

---

## Modelling Decisions

*To be completed. See /docs/modelling_decisions.md*

---

## Key Findings

*To be completed on model evaluation.*

---

## Limitations

*To be completed.*

---

## How to Run

*To be completed. Docker instructions.*

---

## Stack

Python, PostgreSQL, SQLAlchemy, pandas, scikit-learn, XGBoost, LightGBM, SHAP, matplotlib, Plotly, Jupyter, Docker, Power BI
