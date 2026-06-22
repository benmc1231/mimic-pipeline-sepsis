"""
Raw MIMIC-IV data extraction pipeline.

This module is the first stage of the pipeline. Its sole responsibility is to
pull raw data from the PostgreSQL database and write it to disk as versioned
parquet files. No transformation, imputation, or feature engineering is
performed here — that is handled by clean.py and features.py respectively.

Keeping extraction pure means cleaning and feature decisions can be iterated
on without re-querying the database. On a 432M row chartevents table this
saves significant time and keeps the pipeline auditable.

Outputs (written to data/versioned/):
    cohort_base.parquet             — ICU stays joined to admissions and patients
    vitals_raw.parquet              — Chartevents filtered to target vital itemids
    labs_raw.parquet                — Labevents filtered to target lab itemids
    infection_components_raw.parquet — Blood cultures from microbiologyevents
    medications_raw.parquet         — Antibiotic administrations from emar
    vasopressors_raw.parquet        — Vasopressor infusions from inputevents
    urine_output_raw.parquet        — Urine output from outputevents
    ventilation_events_raw.parquet  — Ventilation events from procedureevents
    diagnosis_raw.parquet           — All ICD codes for cohort admissions

All extract functions check for an existing parquet file before querying.
Re-running the pipeline will load from cache unless the parquet is deleted.
"""

import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from pipeline.constants import (
    ADMINISTERED_EVENT_TYPES,
    BLOOD_CULTURE_SPEC_TYPES,
    LAB_ITEMIDS_FLAT,
    URINE_OUTPUT_ITEMIDS_FLAT,
    VASOPRESSOR_ITEMIDS_FLAT,
    VENTILATION_ITEMIDS_FLAT,
    VITAL_ITEMIDS_FLAT,
    antibiotic_conditions,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

load_dotenv()

engine = create_engine(
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

OUTPUT_DIR = Path("data/versioned")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# TEMP TABLE HELPERS
# Each extract function opens its own connection and creates a temp table
# scoped to that session. Temp tables are used in place of large IN clauses
# (94,458 stay_ids or hadm_ids) to allow PostgreSQL to use indexes on joins.
# ---------------------------------------------------------------------------


def _create_cohort_stays_table(conn, cohort_backbone):
    """Create a session-scoped temp table of cohort stay_ids for joining."""
    conn.execute(text("CREATE TEMP TABLE cohort_stays (stay_id BIGINT)"))
    conn.execute(
        text("INSERT INTO cohort_stays (stay_id) VALUES (:stay_id)"),
        [{"stay_id": sid} for sid in cohort_backbone["stay_id"].tolist()],
    )


def _create_cohort_admissions_table(conn, cohort_backbone):
    """Create a session-scoped temp table of cohort hadm_ids for joining."""
    conn.execute(text("CREATE TEMP TABLE cohort_admissions (hadm_id BIGINT)"))
    conn.execute(
        text("INSERT INTO cohort_admissions (hadm_id) VALUES (:hadm_id)"),
        [{"hadm_id": hid} for hid in cohort_backbone["hadm_id"].tolist()],
    )


# ---------------------------------------------------------------------------
# EXTRACTION FUNCTIONS
# ---------------------------------------------------------------------------


def define_cohort_backbone():
    """Build the cohort spine — one row per ICU stay with demographics.

    Joins icustays to admissions (hadm_id) and patients (subject_id) using
    INNER JOINs. Both joins are mandatory — an ICU stay without a matching
    admission or patient record is a data integrity issue, not a valid cohort
    member.

    No cohort filtering is applied here — inclusion/exclusion criteria
    (age ≥18, LOS ≥6h, sepsis not present on admission) are enforced in
    clean.py after all extracts are complete.
    """
    parquet_path = OUTPUT_DIR / "cohort_base.parquet"
    if parquet_path.exists():
        logging.info("Loading cohort backbone from parquet cache")
        return pd.read_parquet(parquet_path)

    df = pd.read_sql(
        """
        SELECT
            icu.stay_id,
            icu.subject_id,
            icu.hadm_id,
            icu.intime,
            icu.outtime,
            icu.los,
            icu.first_careunit,
            adm.admittime,
            adm.dischtime,
            adm.deathtime,
            adm.admission_type,
            adm.insurance,
            adm.marital_status,
            adm.race,
            adm.hospital_expire_flag,
            pat.gender,
            pat.anchor_age,
            pat.anchor_year,
            pat.anchor_year_group,
            pat.dod
        FROM mimiciv_icu.icustays icu
        INNER JOIN mimiciv_hosp.admissions adm ON icu.hadm_id = adm.hadm_id
        INNER JOIN mimiciv_hosp.patients pat ON icu.subject_id = pat.subject_id
        """,
        engine,
    )
    logging.info(f"Cohort backbone extracted: {len(df)} ICU stays")
    df.to_parquet(parquet_path, index=False)
    return df


def extract_vitals(cohort_backbone):
    """Extract raw vital sign observations from chartevents.

    Pulls heart rate, blood pressure (arterial and non-invasive), temperature,
    respiratory rate, SpO2, FiO2, and GCS components. These are the chartevents
    inputs for SOFA score derivation and early sepsis signal features.

    Warning-flagged values are excluded at source. No aggregation, imputation,
    or window filtering is applied — that occurs in clean.py and features.py.
    """
    parquet_path = OUTPUT_DIR / "vitals_raw.parquet"
    if parquet_path.exists():
        logging.info("Loading vitals from parquet cache")
        return pd.read_parquet(parquet_path)

    with engine.connect() as conn:
        _create_cohort_stays_table(conn, cohort_backbone)
        df = pd.read_sql(
            text("""
                SELECT c.stay_id, c.charttime, c.itemid, c.valuenum, c.valueuom
                FROM mimiciv_icu.chartevents c
                INNER JOIN cohort_stays cs ON c.stay_id = cs.stay_id
                WHERE c.itemid IN :itemids
                AND (c.warning = 0 OR c.warning IS NULL)
            """),
            conn,
            params={"itemids": tuple(VITAL_ITEMIDS_FLAT)},
        )

    logging.info(f"Vitals extracted: {len(df)} rows")
    df.to_parquet(parquet_path, index=False)
    return df


def extract_labs(cohort_backbone):
    """Extract raw laboratory results from labevents.

    Pulls creatinine, bilirubin, platelets, lactate, WBC, and haemoglobin —
    the lab inputs for SOFA score derivation and sepsis severity features.
    Joined via hadm_id as labevents is a hospital-level table with no stay_id.

    All values extracted regardless of abnormal flag — normal lab values are
    informative for SOFA scoring and trend analysis.
    """
    parquet_path = OUTPUT_DIR / "labs_raw.parquet"
    if parquet_path.exists():
        logging.info("Loading labs from parquet cache")
        return pd.read_parquet(parquet_path)

    with engine.connect() as conn:
        _create_cohort_admissions_table(conn, cohort_backbone)
        df = pd.read_sql(
            text("""
                SELECT l.subject_id, l.hadm_id, l.charttime, l.itemid, l.valuenum, l.valueuom
                FROM mimiciv_hosp.labevents l
                INNER JOIN cohort_admissions ca ON l.hadm_id = ca.hadm_id
                WHERE l.itemid IN :itemids
            """),
            conn,
            params={"itemids": tuple(LAB_ITEMIDS_FLAT)},
        )

    logging.info(f"Labs extracted: {len(df)} rows")
    df.to_parquet(parquet_path, index=False)
    return df


def extract_infection_components(cohort_backbone):
    """Extract blood culture orders from microbiologyevents.

    Blood cultures are one half of the Sepsis-3 suspected infection criterion.
    The other half (antibiotic administration) is extracted in extract_medications.
    The temporal relationship between cultures and antibiotics is evaluated in
    features.py — this extract pulls raw events only.

    spec_type_desc is filtered to blood culture types only. Urine, respiratory,
    and wound cultures are excluded as they are not used in Sepsis-3 derivation.
    """
    parquet_path = OUTPUT_DIR / "infection_components_raw.parquet"
    if parquet_path.exists():
        logging.info("Loading infection components from parquet cache")
        return pd.read_parquet(parquet_path)

    with engine.connect() as conn:
        _create_cohort_admissions_table(conn, cohort_backbone)
        df = pd.read_sql(
            text("""
                SELECT
                    m.subject_id,
                    m.hadm_id,
                    m.chartdate,
                    m.charttime,
                    m.spec_type_desc,
                    m.org_name,
                    m.ab_name,
                    m.micro_specimen_id
                FROM mimiciv_hosp.microbiologyevents m
                INNER JOIN cohort_admissions ca ON m.hadm_id = ca.hadm_id
                WHERE m.spec_type_desc IN :spec_types
            """),
            conn,
            params={"spec_types": tuple(BLOOD_CULTURE_SPEC_TYPES)},
        )

    logging.info(f"Infection components extracted: {len(df)} rows")
    df.to_parquet(parquet_path, index=False)
    return df


def extract_medications(cohort_backbone):
    """Extract antibiotic administrations from emar.

    Antibiotic administration is one half of the Sepsis-3 suspected infection
    criterion. Filtered using ILIKE pattern matching against ANTIBIOTIC_PATTERNS
    to handle MIMIC-IV's inconsistent medication name capitalisation.

    Exclusion filters remove ophthalmic, topical, vaginal preparations and
    heparin locks — non-systemic routes not relevant to sepsis treatment.
    Desensitization and graded challenge entries are protocol events, not
    treatment administration.

    event_txt is filtered to administration-confirming event types only.
    See ADMINISTERED_EVENT_TYPES in constants.py for full list and rationale.
    """
    parquet_path = OUTPUT_DIR / "medications_raw.parquet"
    if parquet_path.exists():
        logging.info("Loading medications from parquet cache")
        return pd.read_parquet(parquet_path)

    with engine.connect() as conn:
        _create_cohort_admissions_table(conn, cohort_backbone)
        df = pd.read_sql(
            # antibiotic_conditions is an f-string injected directly — safe because
            # ANTIBIOTIC_PATTERNS is a hardcoded constant, not user input
            text(f"""
                SELECT e.subject_id, e.hadm_id, e.charttime, e.medication, e.event_txt
                FROM mimiciv_hosp.emar e
                INNER JOIN cohort_admissions ca ON e.hadm_id = ca.hadm_id
                WHERE ({antibiotic_conditions})
                AND e.medication NOT ILIKE '%ophth%'
                AND e.medication NOT ILIKE '%heparin lock%'
                AND e.medication NOT ILIKE '%topical%'
                AND e.medication NOT ILIKE '%vaginal%'
                AND e.medication NOT ILIKE '%gel%'
                AND e.medication NOT ILIKE '%desensitization%'
                AND e.medication NOT ILIKE '%graded challenge%'
                AND e.medication NOT ILIKE '%placebo%'
                AND e.event_txt IN :event_types
            """),
            conn,
            params={"event_types": tuple(ADMINISTERED_EVENT_TYPES)},
        )

    logging.info(f"Medications extracted: {len(df)} rows")
    df.to_parquet(parquet_path, index=False)
    return df


def extract_vasopressors(cohort_backbone):
    """Extract vasopressor infusion records from inputevents.

    Vasopressor requirement is the SOFA cardiovascular component (score 3-4).
    Pulls starttime, endtime, rate, and amount for dose-based SOFA scoring
    in features.py. Note vasopressin is recorded in units rather than mg —
    unit handling is addressed in features.py.
    """
    parquet_path = OUTPUT_DIR / "vasopressors_raw.parquet"
    if parquet_path.exists():
        logging.info("Loading vasopressors from parquet cache")
        return pd.read_parquet(parquet_path)

    with engine.connect() as conn:
        _create_cohort_stays_table(conn, cohort_backbone)
        df = pd.read_sql(
            text("""
                SELECT
                    i.subject_id,
                    i.stay_id,
                    i.itemid,
                    i.starttime,
                    i.endtime,
                    i.amount,
                    i.amountuom,
                    i.rate,
                    i.rateuom
                FROM mimiciv_icu.inputevents i
                INNER JOIN cohort_stays cs ON i.stay_id = cs.stay_id
                WHERE i.itemid IN :itemids
            """),
            conn,
            params={"itemids": tuple(VASOPRESSOR_ITEMIDS_FLAT)},
        )

    logging.info(f"Vasopressors extracted: {len(df)} rows")
    df.to_parquet(parquet_path, index=False)
    return df


def extract_urine_output(cohort_backbone):
    """Extract urine output volumes from outputevents.

    Urine output is the SOFA renal component alongside creatinine. All catheter
    and voiding itemids are included — see URINE_OUTPUT_ITEMIDS in constants.py
    for inclusion/exclusion decisions and rationale. Aggregation to hourly
    totals for SOFA scoring is performed in features.py.
    """
    parquet_path = OUTPUT_DIR / "urine_output_raw.parquet"
    if parquet_path.exists():
        logging.info("Loading urine output from parquet cache")
        return pd.read_parquet(parquet_path)

    with engine.connect() as conn:
        _create_cohort_stays_table(conn, cohort_backbone)
        df = pd.read_sql(
            text("""
                SELECT o.subject_id, o.stay_id, o.itemid, o.charttime, o.value, o.valueuom
                FROM mimiciv_icu.outputevents o
                INNER JOIN cohort_stays cs ON o.stay_id = cs.stay_id
                WHERE o.itemid IN :itemids
            """),
            conn,
            params={"itemids": tuple(URINE_OUTPUT_ITEMIDS_FLAT)},
        )

    logging.info(f"Urine output extracted: {len(df)} rows")
    df.to_parquet(parquet_path, index=False)
    return df


def extract_ventilation_events(cohort_backbone):
    """Extract mechanical ventilation procedure events from procedureevents.

    Ventilation status is required for correct interpretation of the PF ratio
    in SOFA respiratory scoring — the threshold differs between ventilated and
    non-ventilated patients. Only invasive and non-invasive ventilation onset/
    offset events are captured here. Ventilator settings (mode, rate, PEEP)
    are in chartevents and captured via extract_vitals.
    """
    parquet_path = OUTPUT_DIR / "ventilation_events_raw.parquet"
    if parquet_path.exists():
        logging.info("Loading ventilation events from parquet cache")
        return pd.read_parquet(parquet_path)

    with engine.connect() as conn:
        _create_cohort_stays_table(conn, cohort_backbone)
        df = pd.read_sql(
            text("""
                SELECT
                    p.subject_id,
                    p.stay_id,
                    p.itemid,
                    p.starttime,
                    p.endtime,
                    p.value,
                    p.statusdescription
                FROM mimiciv_icu.procedureevents p
                INNER JOIN cohort_stays cs ON p.stay_id = cs.stay_id
                WHERE p.itemid IN :itemids
            """),
            conn,
            params={"itemids": tuple(VENTILATION_ITEMIDS_FLAT)},
        )

    logging.info(f"Ventilation events extracted: {len(df)} rows")
    df.to_parquet(parquet_path, index=False)
    return df


def extract_diagnosis(cohort_backbone):
    """Extract all ICD-9 and ICD-10 diagnosis codes for cohort admissions.

    All codes are extracted (not filtered to sepsis codes) to support two uses:
    1. Comorbidity feature derivation (Charlson, Elixhauser indices) in features.py
    2. Cross-validation of Sepsis-3 derived onset labels against sepsis ICD codes

    ICD codes are NOT used as the primary cohort exclusion mechanism — codes are
    assigned at discharge with no onset timestamp, making them unsuitable for
    determining whether sepsis was present at ICU admission. Sepsis-3 derived
    onset time is used for cohort exclusion instead. See docs/variable_logic.md.

    TRIM applied to icd_code to handle trailing whitespace present in some
    MIMIC-IV ICD-9 entries.
    """
    parquet_path = OUTPUT_DIR / "diagnosis_raw.parquet"
    if parquet_path.exists():
        logging.info("Loading diagnosis codes from parquet cache")
        return pd.read_parquet(parquet_path)

    with engine.connect() as conn:
        _create_cohort_admissions_table(conn, cohort_backbone)
        df = pd.read_sql(
            text("""
                SELECT
                    d.subject_id,
                    d.hadm_id,
                    TRIM(d.icd_code) AS icd_code,
                    d.icd_version,
                    d.seq_num
                FROM mimiciv_hosp.diagnoses_icd d
                INNER JOIN cohort_admissions ca ON d.hadm_id = ca.hadm_id
            """),
            conn,
        )

    logging.info(f"Diagnosis codes extracted: {len(df)} rows")
    df.to_parquet(parquet_path, index=False)
    return df


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------


def main():
    """Run all extraction functions in dependency order.

    cohort_backbone is extracted first and passed to all subsequent functions
    as the filtering spine. Functions that accept cohort_backbone use it to
    restrict extraction to cohort stays only via temp table joins.
    """
    cohort_backbone = define_cohort_backbone()
    extract_vitals(cohort_backbone)
    extract_labs(cohort_backbone)
    extract_infection_components(cohort_backbone)
    extract_medications(cohort_backbone)
    extract_vasopressors(cohort_backbone)
    extract_urine_output(cohort_backbone)
    extract_ventilation_events(cohort_backbone)
    extract_diagnosis(cohort_backbone)


if __name__ == "__main__":
    main()
