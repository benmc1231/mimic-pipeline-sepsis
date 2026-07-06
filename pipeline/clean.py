"""
Cleaning, validation, and missingness audit for MIMIC-IV pipeline extracts.

Reads raw parquet extracts produced by extract.py, applies clinically defined
quality filters, enforces cohort inclusion/exclusion criteria, standardises
units, and produces clean versioned parquets ready for features.py.

This module does not engineer features or apply the observation window.
Imputation of missing values occurs in features.py, not here. The role of
clean.py is to measure and report missingness, not resolve it.

Outputs (written to data/versioned/):
    cohort_clean.parquet
    vitals_clean.parquet
    labs_clean.parquet
    infection_components_clean.parquet
    medications_clean.parquet
    vasopressors_clean.parquet
    urine_output_clean.parquet
    ventilation_events_clean.parquet
    diagnosis_clean.parquet
    missingness_report.csv

Comorbidity feature derivation (Charlson, Elixhauser, CKD, liver, malignancy,
diabetes, immunosuppression) is handled separately in comorbidities.py, since
it produces analytical features rather than cleaned source data.

Sepsis-on-admission exclusion is deferred to features.py. Deriving Sepsis-3
criteria requires SOFA scores from vitals and labs, which are not available
at clean time. This creates a known dependency: the cohort returned here is
not final until features.py completes its exclusion pass. See
docs/variable_logic.md for full discussion.
"""

import logging
from pathlib import Path

import pandas as pd

from pipeline.constants import (
    LAB_ITEMID_TO_LABEL,
    LAB_RANGES,
    VITAL_ITEMID_TO_LABEL,
    VITAL_RANGES,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

EXTRACT_DIR = Path("data/versioned")


# ---------------------------------------------------------------------------
# I/O HELPERS
# ---------------------------------------------------------------------------


def load_raw_extracts() -> dict[str, pd.DataFrame]:
    """Load all raw parquet extracts from data/versioned/.

    Raises FileNotFoundError if any expected extract is missing rather than
    failing silently downstream.
    """
    expected_extracts = {
        "cohort": "cohort_base.parquet",
        "vitals": "vitals_raw.parquet",
        "labs": "labs_raw.parquet",
        "infection_components": "infection_components_raw.parquet",
        "medications": "medications_raw.parquet",
        "vasopressors": "vasopressors_raw.parquet",
        "urine_output": "urine_output_raw.parquet",
        "ventilation": "ventilation_events_raw.parquet",
        "diagnosis": "diagnosis_raw.parquet",
    }

    extracts = {}
    for name, filename in expected_extracts.items():
        path = EXTRACT_DIR / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Expected extract not found: {path}. "
                f"Run extract.py before clean.py."
            )
        logging.info(f"Loading {name} from {filename}")
        extracts[name] = pd.read_parquet(path)

    return extracts


def save_cleaned_extract(df: pd.DataFrame, filename: str) -> None:
    """Write a cleaned extract to data/versioned/."""
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    path = EXTRACT_DIR / filename
    df.to_parquet(path, index=False)
    logging.info(f"Wrote cleaned extract: {path} ({len(df)} rows)")


# ---------------------------------------------------------------------------
# COHORT EXCLUSION
# ---------------------------------------------------------------------------


def cohort_exclusion(cohort: pd.DataFrame) -> pd.DataFrame:
    """Apply cohort inclusion/exclusion criteria, logging size after each step.

    Filters applied in order:
        1. Age: exclude stays where age at admission < 18. Calculated from
           anchor_age offset by the difference between admittime year and
           anchor_year, since MIMIC-IV does not provide age directly at
           admission.
        2. LOS: exclude stays where los < 0.25 days (approx 6 hours).
        3. Sepsis on admission: deferred to features.py. Requires Sepsis-3
           derivation (suspected infection + SOFA increase >= 2) which
           depends on cleaned vitals and labs not available at this stage.
           See features.py: exclude_sepsis_on_admission().
    """
    initial = len(cohort)
    logging.info(f"Cohort exclusion: initial size {initial} ICU stays")

    # Age filter
    # anchor_age is recorded at anchor_year, not at admission time.
    # Approximate age at admission by adjusting for the year difference.
    cohort["age_at_admission"] = cohort["anchor_age"] + (
        cohort["admittime"].dt.year - cohort["anchor_year"]
    )
    cohort = cohort[cohort["age_at_admission"] >= 18]
    logging.info(
        f"Age filter: removed {initial - len(cohort)} stays "
        f"(MIMIC-IV excludes under-18s at source, expect ~0 removed), "
        f"{len(cohort)} remaining"
    )

    # LOS filter
    before_los = len(cohort)
    cohort = cohort[cohort["los"] >= 0.25]
    logging.info(
        f"LOS filter: removed {before_los - len(cohort)} stays under 6 hours, "
        f"{len(cohort)} remaining"
    )

    # Sepsis on admission exclusion deferred to features.py.
    # See module docstring for rationale.

    return cohort


# ---------------------------------------------------------------------------
# VITALS CLEANING
# ---------------------------------------------------------------------------


def clean_vitals(vitals: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply range filters and unit standardisation to raw vital observations.

    Values outside clinically defined ranges are set to null rather than
    dropped. Missingness is informative and handled separately in features.py.

    Unit standardisation:
        FiO2: percentage values (21-100) converted to fraction (0.21-1.0).
              MIMIC-IV records FiO2 inconsistently across both formats.
        Temperature: Fahrenheit (itemid 223761) converted to Celsius and
                     relabelled as itemid 223762 to consolidate both into
                     a single concept.
        Blood pressure: arterial line values (ABP) preferred over non-invasive
                        (NIBP) where both exist at the same stay/charttime.
                        Arterial readings are more accurate and continuously
                        measured.

    Returns:
        vitals: cleaned DataFrame
        stats: per-itemid dict with total_obs, null_obs, stays_with_data,
               captured at clean time for use in generate_missingness_report().
    """
    # Range filters - values outside clinical bounds set to null
    for itemid, (min_val, max_val) in VITAL_RANGES.items():
        mask = vitals["itemid"] == itemid
        out_of_range = mask & ~vitals["valuenum"].between(min_val, max_val)
        vitals.loc[out_of_range, "valuenum"] = None

    logging.info(
        f"Vitals range filters: {vitals['valuenum'].isna().sum()} null values total"
    )

    # Per-itemid stats captured here before unit transformations change the
    # itemid landscape (Fahrenheit rows are relabelled to Celsius below)
    stats = {}
    for itemid in VITAL_RANGES:
        item_df = vitals[vitals["itemid"] == itemid]
        stats[itemid] = {
            "total_obs": len(item_df),
            "null_obs": int(item_df["valuenum"].isna().sum()),
            "stays_with_data": item_df["stay_id"].nunique(),
        }

    # FiO2 unit standardisation
    fio2_mask = vitals["itemid"] == 223835
    percentage_mask = fio2_mask & (vitals["valuenum"] > 1.0)
    vitals.loc[percentage_mask, "valuenum"] = (
        vitals.loc[percentage_mask, "valuenum"] / 100
    )
    logging.info(
        f"FiO2: {percentage_mask.sum()} percentage values converted to fraction"
    )

    # Re-apply FiO2 range filter after conversion
    fio2_out_of_range = fio2_mask & ~vitals["valuenum"].between(0.21, 1.0)
    vitals.loc[fio2_out_of_range, "valuenum"] = None

    # Temperature consolidation
    fahrenheit_mask = vitals["itemid"] == 223761
    vitals.loc[fahrenheit_mask, "valuenum"] = (
        (vitals.loc[fahrenheit_mask, "valuenum"] - 32) * 5 / 9
    )
    vitals.loc[fahrenheit_mask, "itemid"] = 223762
    logging.info(
        f"Temperature: {fahrenheit_mask.sum()} Fahrenheit values converted to Celsius"
    )

    # Blood pressure consolidation
    # Where arterial (ABP) and non-invasive (NIBP) readings exist for the
    # same stay and charttime, null the NIBP value. ABP itemids map to NIBP
    # fallbacks: systolic 220050->220179, diastolic 220051->220180, mean 220052->220181
    abp_nibp_pairs = {220050: 220179, 220051: 220180, 220052: 220181}

    for abp_id, nibp_id in abp_nibp_pairs.items():
        abp_times = vitals.loc[vitals["itemid"] == abp_id, ["stay_id", "charttime"]]
        nibp_mask = vitals["itemid"] == nibp_id
        duplicate_mask = nibp_mask & vitals.set_index(
            ["stay_id", "charttime"]
        ).index.isin(abp_times.set_index(["stay_id", "charttime"]).index)
        vitals.loc[duplicate_mask, "valuenum"] = None

    logging.info(
        "Blood pressure: NIBP values nulled where ABP exists at same stay/charttime"
    )

    return vitals, stats


# ---------------------------------------------------------------------------
# LABS CLEANING
# ---------------------------------------------------------------------------


def clean_labs(labs: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply clinically defined range filters to raw laboratory results.

    Units confirmed via data profiling:
        creatinine and bilirubin in mg/dL
        platelets and WBC in K/uL (equivalent to x10^9/L)
        lactate in mmol/L
        haemoglobin in g/dL

    All values extracted regardless of abnormal flag. Normal lab values are
    informative for SOFA scoring and trend analysis.

    Returns:
        labs: cleaned DataFrame
        stats: per-itemid dict with total_obs, null_obs, hadms_with_data,
               captured at clean time for use in generate_missingness_report().
    """
    before_nulls = int(labs["valuenum"].isna().sum())

    for itemid, (min_val, max_val) in LAB_RANGES.items():
        mask = labs["itemid"] == itemid
        out_of_range = mask & ~labs["valuenum"].between(min_val, max_val)
        labs.loc[out_of_range, "valuenum"] = None

    after_nulls = int(labs["valuenum"].isna().sum())
    logging.info(
        f"Lab range filters: {after_nulls - before_nulls} values set to null, "
        f"{after_nulls} total null"
    )

    stats = {}
    for itemid in LAB_RANGES:
        item_df = labs[labs["itemid"] == itemid]
        stats[itemid] = {
            "total_obs": len(item_df),
            "null_obs": int(item_df["valuenum"].isna().sum()),
            "hadms_with_data": item_df["hadm_id"].nunique(),
        }

    return labs, stats


# ---------------------------------------------------------------------------
# EVENT TABLE CLEANING
# ---------------------------------------------------------------------------


def clean_medications(medications: pd.DataFrame) -> pd.DataFrame:
    """Clean antibiotic administration records from emar.

    Removes records without a charttime. Records without a timestamp cannot
    be used in Sepsis-3 suspected infection timing logic in features.py.
    No value range filters needed since this is categorical event data.
    """
    before = len(medications)
    medications = medications[medications["charttime"].notna()]
    logging.info(
        f"Medications: {before - len(medications)} rows removed due to null charttime, "
        f"{len(medications)} remaining"
    )
    return medications


def clean_infection_components(infection_components: pd.DataFrame) -> pd.DataFrame:
    """Clean blood culture records from microbiologyevents.

    Adds charttime_imputed flag unconditionally so downstream code in
    features.py can rely on its presence without checking column existence.

    Where charttime is null but chartdate is populated, imputes charttime as
    midnight of chartdate and sets charttime_imputed to True. Midnight
    timestamps are less precise for 6-hour window logic and should be treated
    cautiously in features.py.

    Data profiling on raw microbiologyevents found 14,785 null charttimes
    (1.8% of blood culture records), all with valid chartdates. The cleaned
    file confirmed 0 null charttimes remaining and all charttime_imputed
    values False, meaning imputation was not required for this dataset version.
    The logic is retained as a defensive check since this may differ across
    MIMIC-IV versions.
    """
    # Add flag unconditionally so features.py can always reference it
    infection_components = infection_components.copy()
    infection_components["charttime_imputed"] = False

    null_mask = infection_components["charttime"].isna()
    null_count = null_mask.sum()

    if null_count > 0:
        infection_components.loc[null_mask, "charttime"] = pd.to_datetime(
            infection_components.loc[null_mask, "chartdate"]
        )
        infection_components.loc[null_mask, "charttime_imputed"] = True
        logging.info(
            f"Microbiology: {null_count} charttime values imputed from chartdate "
            f"({null_count / len(infection_components) * 100:.1f}% of records)"
        )
    else:
        logging.info("Microbiology: no null charttimes found, no imputation required")

    remaining_nulls = infection_components["charttime"].isna().sum()
    if remaining_nulls > 0:
        logging.warning(
            f"Microbiology: {remaining_nulls} records have null charttime and null "
            f"chartdate and cannot be used for Sepsis-3 timing. Dropping."
        )
        infection_components = infection_components[
            infection_components["charttime"].notna()
        ]

    return infection_components


def clean_vasopressors(vasopressors: pd.DataFrame) -> pd.DataFrame:
    """Clean vasopressor infusion records from inputevents.

    Unit inconsistencies identified in data profiling and corrected here:
        Norepinephrine: 2 rows in mg/kg/min converted to mcg/kg/min (x1000).
                        mg/kg/min would be a lethal dose, almost certainly a
                        data entry error.
        Phenylephrine: 2 rows in mcg/min nulled. Cannot convert to mcg/kg/min
                       without patient weight.
        Vasopressin: 3 rows in units/min converted to units/hour (x60).
                     units/hour is the predominant unit in MIMIC-IV.
        Epinephrine duplicate (itemid 229617): 234 rows with no rateuom nulled.

    Rows with null starttime or endtime are removed since they cannot be used
    for SOFA cardiovascular timing logic in features.py.
    """
    before = len(vasopressors)
    vasopressors = vasopressors[
        vasopressors["starttime"].notna() & vasopressors["endtime"].notna()
    ]
    logging.info(
        f"Vasopressors: {before - len(vasopressors)} rows removed due to "
        f"null starttime or endtime"
    )

    norep_mg_mask = (vasopressors["itemid"] == 221906) & (
        vasopressors["rateuom"] == "mg/kg/min"
    )
    vasopressors.loc[norep_mg_mask, "rate"] *= 1000
    vasopressors.loc[norep_mg_mask, "rateuom"] = "mcg/kg/min"
    logging.info(
        f"Vasopressors: {norep_mg_mask.sum()} norepinephrine rows converted "
        f"from mg/kg/min to mcg/kg/min"
    )

    phenyl_mcgmin_mask = (vasopressors["itemid"] == 221749) & (
        vasopressors["rateuom"] == "mcg/min"
    )
    vasopressors.loc[phenyl_mcgmin_mask, "rate"] = None
    logging.info(
        f"Vasopressors: {phenyl_mcgmin_mask.sum()} phenylephrine rows nulled "
        f"(mcg/min cannot be converted to mcg/kg/min without patient weight)"
    )

    vasopressin_min_mask = (vasopressors["itemid"] == 222315) & (
        vasopressors["rateuom"] == "units/min"
    )
    vasopressors.loc[vasopressin_min_mask, "rate"] *= 60
    vasopressors.loc[vasopressin_min_mask, "rateuom"] = "units/hour"
    logging.info(
        f"Vasopressors: {vasopressin_min_mask.sum()} vasopressin rows converted "
        f"from units/min to units/hour"
    )

    epineph_no_unit_mask = (vasopressors["itemid"] == 229617) & (
        vasopressors["rateuom"].isna()
    )
    vasopressors.loc[epineph_no_unit_mask, "rate"] = None
    logging.info(
        f"Vasopressors: {epineph_no_unit_mask.sum()} epinephrine duplicate rows "
        f"nulled due to missing rateuom"
    )

    # Remove negative rates - physiologically impossible
    neg_mask = vasopressors["rate"] < 0
    if neg_mask.sum() > 0:
        logging.info(f"Vasopressors: {neg_mask.sum()} negative rate rows removed")
        vasopressors = vasopressors[~neg_mask]

    null_rates = vasopressors["rate"].isna().sum()
    logging.info(f"Vasopressors: {null_rates} rows with null rate after cleaning")

    return vasopressors


def clean_urine_output(urine_output: pd.DataFrame) -> pd.DataFrame:
    """Clean urine output records from outputevents.

    Negative values are removed as physiologically impossible data entry errors.
    Values over 2000 mL per entry are flagged rather than dropped since they
    may represent accumulated totals charted as a single entry rather than
    a genuine single void.
    """
    before = len(urine_output)
    urine_output = urine_output[urine_output["value"] >= 0]
    logging.info(
        f"Urine output: {before - len(urine_output)} negative value rows removed"
    )

    large_void_mask = urine_output["value"] > 2000
    urine_output["large_void_flag"] = large_void_mask
    logging.info(f"Urine output: {large_void_mask.sum()} rows flagged as over 2000 mL")

    return urine_output


def clean_ventilation(ventilation: pd.DataFrame) -> pd.DataFrame:
    """Clean mechanical ventilation procedure events from procedureevents.

    Removes records with null starttime or endtime. These cannot be used to
    determine ventilation status during the observation window in features.py.
    """
    before = len(ventilation)
    ventilation = ventilation[
        ventilation["starttime"].notna() & ventilation["endtime"].notna()
    ]
    logging.info(
        f"Ventilation: {before - len(ventilation)} rows removed due to "
        f"null starttime or endtime, {len(ventilation)} remaining"
    )
    return ventilation


# ---------------------------------------------------------------------------
# MISSINGNESS AUDIT
# ---------------------------------------------------------------------------


def _capture_coverage_stats(
    df: pd.DataFrame, cohort: pd.DataFrame, id_column: str
) -> dict:
    """Compute table-level coverage: % of cohort stays/admissions with at
    least one row in df.

    Used for tables where a single coverage figure per source is more
    meaningful than a per-itemid breakdown (vasopressors, urine output,
    ventilation, medications, infection components). For these tables, absence
    of a record is itself a clinical signal, not missing data in the
    traditional sense.
    """
    n_cohort = cohort[id_column].nunique()
    covered = df[id_column].nunique()
    return {
        "rows": len(df),
        "covered": covered,
        "total_cohort": n_cohort,
        "coverage_pct": round((covered / n_cohort) * 100, 2) if n_cohort else None,
    }


def generate_missingness_report(
    cohort: pd.DataFrame,
    vitals: pd.DataFrame,
    labs: pd.DataFrame,
    vital_stats: dict,
    lab_stats: dict,
    coverage_stats: dict,
) -> pd.DataFrame:
    """Assemble per-feature missingness and coverage report from cleaning outputs.

    Vitals and labs get per-itemid breakdown. Event tables (vasopressors, urine
    output, ventilation, medications, infection components) get a single
    table-level coverage figure.

    All rows include both missingness_pct and coverage_pct as complementary
    values summing to 100. raw_missingness_pct is clamped to 0 as a minimum
    to handle the pipeline ordering artefact where event table extracts include
    rows from LOS-excluded stays. Those stays inflate stays_with_data above
    the cleaned cohort size, producing a small apparent negative missingness.
    This is documented in docs/variable_logic.md.

    Required Phase 1 deliverable per the project brief.
    Output written to data/versioned/missingness_report.csv.
    """
    n_stays = cohort["stay_id"].nunique()
    n_hadms = cohort["hadm_id"].nunique()

    # Restrict to cohort IDs for accurate coverage calculation
    cohort_stay_ids = set(cohort["stay_id"])
    cohort_hadm_ids = set(cohort["hadm_id"])

    rows = []

    # Vitals: recompute stays_with_data restricted to cohort to avoid
    # negative missingness from LOS-excluded stays present in extract
    for itemid, s in vital_stats.items():
        item_in_cohort = vitals[
            (vitals["itemid"] == itemid) & (vitals["stay_id"].isin(cohort_stay_ids))
        ]
        stays_with_data = item_in_cohort["stay_id"].nunique()
        raw = max(0.0, (1 - stays_with_data / n_stays) * 100) if n_stays else None
        post = (s["null_obs"] / s["total_obs"]) * 100 if s["total_obs"] else None
        coverage = round(100 - raw, 2) if raw is not None else None
        rows.append(
            {
                "feature": VITAL_ITEMID_TO_LABEL.get(itemid, itemid),
                "itemid": itemid,
                "source": "vitals",
                "missingness_pct": round(raw, 2) if raw is not None else None,
                "post_cleaning_null_pct": round(post, 2) if post is not None else None,
                "coverage_pct": coverage,
            }
        )

    # Labs: same recomputation restricted to cohort hadm_ids
    for itemid, s in lab_stats.items():
        item_in_cohort = labs[
            (labs["itemid"] == itemid) & (labs["hadm_id"].isin(cohort_hadm_ids))
        ]
        hadms_with_data = item_in_cohort["hadm_id"].nunique()
        raw = max(0.0, (1 - hadms_with_data / n_hadms) * 100) if n_hadms else None
        post = (s["null_obs"] / s["total_obs"]) * 100 if s["total_obs"] else None
        coverage = round(100 - raw, 2) if raw is not None else None
        rows.append(
            {
                "feature": LAB_ITEMID_TO_LABEL.get(itemid, itemid),
                "itemid": itemid,
                "source": "labs",
                "missingness_pct": round(raw, 2) if raw is not None else None,
                "post_cleaning_null_pct": round(post, 2) if post is not None else None,
                "coverage_pct": coverage,
            }
        )

    # Event tables: single table-level coverage figure
    for source_name, s in coverage_stats.items():
        coverage = s["coverage_pct"]
        missingness = round(100 - coverage, 2) if coverage is not None else None
        rows.append(
            {
                "feature": source_name,
                "itemid": None,
                "source": source_name,
                "missingness_pct": missingness,
                "post_cleaning_null_pct": None,
                "coverage_pct": coverage,
            }
        )
        logging.info(
            f"{source_name}: {s['covered']}/{s['total_cohort']} cohort units "
            f"covered ({coverage}%)"
        )

    report = pd.DataFrame(rows)
    output_path = EXTRACT_DIR / "missingness_report.csv"
    report.to_csv(output_path, index=False)
    logging.info(f"Missingness report written: {output_path} ({len(report)} features)")
    return report


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------


def main():
    """Run all cleaning functions in dependency order.

    Each function reads from raw parquets produced by extract.py and writes
    a clean versioned parquet. The missingness report is generated after all
    cleaning completes. Comorbidity feature derivation is handled separately
    in comorbidities.py, called from features.py.
    """
    extracts = load_raw_extracts()

    cohort = cohort_exclusion(extracts["cohort"])
    vitals, vital_stats = clean_vitals(extracts["vitals"])
    labs, lab_stats = clean_labs(extracts["labs"])
    medications = clean_medications(extracts["medications"])
    infection_components = clean_infection_components(extracts["infection_components"])
    vasopressors = clean_vasopressors(extracts["vasopressors"])
    urine_output = clean_urine_output(extracts["urine_output"])
    ventilation = clean_ventilation(extracts["ventilation"])

    coverage_stats = {
        "vasopressors": _capture_coverage_stats(vasopressors, cohort, "stay_id"),
        "urine_output": _capture_coverage_stats(urine_output, cohort, "stay_id"),
        "ventilation": _capture_coverage_stats(ventilation, cohort, "stay_id"),
        "medications": _capture_coverage_stats(medications, cohort, "hadm_id"),
        "infection_components": _capture_coverage_stats(
            infection_components, cohort, "hadm_id"
        ),
    }

    save_cleaned_extract(cohort, "cohort_clean.parquet")
    save_cleaned_extract(vitals, "vitals_clean.parquet")
    save_cleaned_extract(labs, "labs_clean.parquet")
    save_cleaned_extract(infection_components, "infection_components_clean.parquet")
    save_cleaned_extract(medications, "medications_clean.parquet")
    save_cleaned_extract(vasopressors, "vasopressors_clean.parquet")
    save_cleaned_extract(urine_output, "urine_output_clean.parquet")
    save_cleaned_extract(ventilation, "ventilation_events_clean.parquet")
    save_cleaned_extract(extracts["diagnosis"], "diagnosis_clean.parquet")

    generate_missingness_report(
        cohort,
        vitals,
        labs,
        vital_stats,
        lab_stats,
        coverage_stats,
    )


if __name__ == "__main__":
    main()
