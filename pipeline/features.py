"""
Feature engineering pipeline with leakage prevention.

This is the central orchestration script for Phase 1 feature generation.
It loads cleaned parquet extracts, enforces the 6-hour observation window,
derives SOFA scores, labels sepsis onset, engineers predictive features,
applies imputation, splits by patient, and writes the modelling-ready dataset.

LEAKAGE PREVENTION GUARANTEE:
    All predictive features are derived exclusively from data within the
    0-6h observation window (hours 0 to 6 from ICU admission). The
    enforce_observation_window() function filters each event table to this
    window before any feature computation. A leakage assertion confirms
    no observation exceeds the boundary after filtering.

    The sepsis onset label is derived from full-stay data in
    sepsis_labelling.py, but no full-stay event data enters the feature
    matrix. The label is joined at the end as a separate column.

    Train/test split is performed at the patient level (subject_id) to
    prevent the same patient's multiple ICU stays from appearing in both
    train and test sets. Splitting on stay_id would allow this and inflate
    test performance through data leakage.

Pipeline execution order:
    1. Load cleaned extracts
    2. Enforce 6-hour observation window on all event tables
    3. Derive SOFA scores (sofa.py)
    4. Derive baseline SOFA from pre-admission labs (sepsis_labelling.py)
    5. Derive suspected infection (sepsis_labelling.py)
    6. Derive sepsis onset label (sepsis_labelling.py)
    7. Finalise cohort - exclude sepsis-on-admission stays
    8. Build feature matrix (feature_engineering.py)
    9. Impute missing values
    10. Train/test split at subject_id level
    11. Save outputs

Outputs (written to data/versioned/):
    features_train.parquet   training set, 80% of patients
    features_test.parquet    held-out test set, 20% of patients
    feature_names.json       ordered list of feature column names
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from pipeline.feature_engineering import build_feature_matrix
from pipeline.sepsis_labelling import (
    compute_baseline_sofa,
    derive_sepsis_onset,
    derive_suspected_infection,
    exclude_sepsis_on_admission,
)
from pipeline.sofa import compute_sofa_total

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

EXTRACT_DIR = Path("data/versioned")


# ---------------------------------------------------------------------------
# STEP 1 - LOAD CLEANED EXTRACTS
# ---------------------------------------------------------------------------


def load_cleaned_extracts() -> dict[str, pd.DataFrame]:
    """Load all cleaned parquet extracts from data/versioned/.

    Raises FileNotFoundError if any expected extract is missing rather than
    failing silently downstream. Run extract.py then clean.py then
    comorbidities.py before running features.py.
    """
    expected_extracts = {
        "cohort": "cohort_clean.parquet",
        "vitals": "vitals_clean.parquet",
        "labs": "labs_clean.parquet",
        "infection_components": "infection_components_clean.parquet",
        "medications": "medications_clean.parquet",
        "vasopressors": "vasopressors_clean.parquet",
        "urine_output": "urine_output_clean.parquet",
        "ventilation": "ventilation_events_clean.parquet",
        "diagnosis": "diagnosis_clean.parquet",
        "comorbidities": "comorbidities_clean.parquet",
    }

    extracts = {}
    for name, filename in expected_extracts.items():
        path = EXTRACT_DIR / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Expected extract not found: {path}. "
                f"Run extract.py, clean.py, and comorbidities.py first."
            )
        logging.info(f"Loading {name} from {filename}")
        extracts[name] = pd.read_parquet(path)

    return extracts


# ---------------------------------------------------------------------------
# STEP 2 - ENFORCE OBSERVATION WINDOW
# ---------------------------------------------------------------------------


def enforce_observation_window(
    cohort_df: pd.DataFrame,
    df: pd.DataFrame,
    timestamp_col: str,
    join_key: str = "stay_id",
    label: str = "",
) -> pd.DataFrame:
    """Filter an event table to the 6-hour observation window per ICU stay.

    This is the primary leakage prevention step. Joins intime from the
    cohort onto the event table via join_key, then excludes any observation
    where timestamp_col falls outside [intime, intime + 6h].

    The inner join implicitly drops orphaned rows from LOS-excluded stays.
    A leakage assertion after filtering confirms no observation exceeds the
    boundary. If this assertion fails, something upstream is wrong and
    pipeline execution is halted.

    For hadm_id-joined tables (labs, medications, infection components),
    intime is taken as the minimum intime per hadm_id where a patient has
    multiple ICU stays within one hospital admission. This anchors lab
    and medication timing to the first ICU stay of that admission.

    Args:
        cohort_df: cleaned cohort with stay_id, hadm_id, and intime
        df: event table to filter
        timestamp_col: timestamp column to filter on
        join_key: stay_id or hadm_id depending on source table
        label: name used in logging output
    """
    window = pd.Timedelta(hours=6)

    if join_key == "hadm_id":
        # For hospital-level tables, anchor to the first ICU stay of the
        # admission (earliest intime) and carry that stay_id onto the
        # windowed rows - downstream per-stay aggregation (SOFA components,
        # feature engineering) needs a stay_id to group on.
        cohort_keys = cohort_df.loc[
            cohort_df.groupby("hadm_id")["intime"].idxmin(),
            ["hadm_id", "stay_id", "intime"],
        ]
        merge_cols = ["hadm_id", "stay_id", "intime"]
    else:
        cohort_keys = cohort_df[["stay_id", "intime"]].copy()
        merge_cols = ["stay_id", "intime"]

    df = df.merge(cohort_keys[merge_cols], on=join_key, how="inner")

    before = len(df)
    df = df[
        (df[timestamp_col] >= df["intime"])
        & (df[timestamp_col] <= df["intime"] + window)
    ]

    logging.info(
        f"{label} observation window: {before:,} -> {len(df):,} rows "
        f"({before - len(df):,} outside 0-6h window)"
    )

    # Leakage assertion - halts pipeline if boundary is violated
    if not df.empty:
        max_offset = (df[timestamp_col] - df["intime"]).max()
        assert max_offset <= window, (
            f"LEAKAGE DETECTED in {label}: "
            f"max offset {max_offset} exceeds 6h boundary"
        )
        logging.info(f"{label} leakage check PASSED: max offset = {max_offset}")

    df = df.drop(columns=["intime"])
    return df


# ---------------------------------------------------------------------------
# STEP 9 - IMPUTATION
# ---------------------------------------------------------------------------


def impute_features(features: pd.DataFrame) -> pd.DataFrame:
    """Apply imputation strategies to the flat feature matrix.

    Imputation is applied after feature aggregation. The feature matrix
    at this point has one row per stay with null values where no data
    was recorded in the observation window.

    Strategies by feature group (see docs/variable_logic.md for rationale):

    Vital signs (all aggregation stats):
        Median imputation across the cohort. Forward fill within stay was
        the natural result of the aggregation window. Null vital features
        indicate no observations of that vital for that stay.

    Labs (first, last, min, max, delta):
        Median imputation, except lactate which is left null intentionally.
        Lactate missingness is clinically informative (ordered only on
        clinical suspicion of haemodynamic instability). The lactate_present
        binary flag captures this signal separately.

    Vasopressors, urine output, ventilation:
        Binary presence flags: fill with 0 (absence is meaningful).
        Duration and rate features: fill with 0 where binary flag is 0.

    Comorbidity scores:
        Charlson and Elixhauser: fill with 0 (no comorbidities assumed).
        Stage columns: leave null (unknown severity is distinct from absent).

    SOFA scores:
        Fill with 0 (consistent with component-level assumption in sofa.py).

    Demographics:
        No imputation needed - all patients have age, gender, admission type.
        Care unit dummies filled with 0 if unit not present.

    Logs missingness rate per feature before and after imputation.
    """
    n_stays = len(features)
    feature_cols = [
        c
        for c in features.columns
        if c not in ["stay_id", "subject_id", "label", "t_sepsis_hour", "sofa_increase"]
    ]

    # Log pre-imputation missingness
    pre_missing = features[feature_cols].isna().sum()
    pre_missing_pct = (pre_missing / n_stays * 100).round(2)
    missing_features = pre_missing[pre_missing > 0]
    if len(missing_features) > 0:
        logging.info(f"Pre-imputation: {len(missing_features)} features have nulls")

    # Vital sign features: median imputation
    vital_labels = list(
        set(
            col.rsplit("_", 1)[0]
            for col in feature_cols
            if any(
                col.startswith(v)
                for v in [
                    "heart_rate",
                    "abp_",
                    "nibp_",
                    "temperature_",
                    "respiratory_rate",
                    "spo2",
                    "fio2",
                    "gcs_",
                ]
            )
        )
    )
    vital_feature_cols = [
        c
        for c in feature_cols
        if any(
            c.startswith(v)
            for v in [
                "heart_rate",
                "abp_",
                "nibp_",
                "temperature_",
                "respiratory_rate",
                "spo2",
                "fio2",
                "gcs_",
            ]
        )
    ]
    for col in vital_feature_cols:
        if features[col].isna().any():
            features[col] = features[col].fillna(features[col].median())

    # Lab features: median imputation except lactate
    lab_feature_cols = [
        c
        for c in feature_cols
        if any(
            c.startswith(lab)
            for lab in [
                "creatinine",
                "bilirubin_total",
                "platelet_count",
                "wbc",
                "haemoglobin",
            ]
        )
    ]
    for col in lab_feature_cols:
        if features[col].isna().any():
            features[col] = features[col].fillna(features[col].median())

    # Lactate: leave null intentionally, only fill present flag
    lactate_present = "lactate_present"
    if lactate_present in features.columns:
        features[lactate_present] = features[lactate_present].fillna(0).astype(int)

    # Vasopressor features: 0 where not used
    vaso_cols = [
        c
        for c in feature_cols
        if c.startswith("vasopressor") or c.endswith("_peak_rate")
    ]
    for col in vaso_cols:
        features[col] = features[col].fillna(0)

    # Urine output features: 0 where not recorded
    urine_cols = [c for c in feature_cols if c.startswith("urine_")]
    for col in urine_cols:
        features[col] = features[col].fillna(0)

    # Ventilation features: 0 where not ventilated
    vent_cols = [c for c in feature_cols if c.startswith("ventilat")]
    for col in vent_cols:
        features[col] = features[col].fillna(0)

    # SOFA features: 0 where not derivable
    sofa_cols = [c for c in feature_cols if c.startswith("sofa_")]
    for col in sofa_cols:
        features[col] = features[col].fillna(0)

    # Comorbidity scores: 0 where no comorbidities
    for col in ["charlson_score", "elixhauser_score"]:
        if col in features.columns:
            features[col] = features[col].fillna(0)

    # Comorbidity boolean flags: 0 where absent
    comorbidity_flags = [
        "has_ckd",
        "has_liver_disease",
        "has_malignancy",
        "has_diabetes",
        "has_immunosuppression",
    ]
    for col in comorbidity_flags:
        if col in features.columns:
            features[col] = features[col].fillna(0).astype(int)

    # Care unit dummies: 0 where unit not present
    careunit_cols = [c for c in feature_cols if c.startswith("careunit_")]
    for col in careunit_cols:
        features[col] = features[col].fillna(0).astype(int)

    # Log post-imputation missingness
    post_missing = features[feature_cols].isna().sum()
    still_missing = post_missing[post_missing > 0]
    if len(still_missing) > 0:
        logging.info(
            f"Post-imputation: {len(still_missing)} features still have nulls "
            f"(intentional: {list(still_missing.index)})"
        )
    else:
        logging.info("Post-imputation: no remaining nulls in feature matrix")

    return features


# ---------------------------------------------------------------------------
# STEP 10 - TRAIN/TEST SPLIT
# ---------------------------------------------------------------------------


def split_train_test(
    features: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split feature matrix into train and test sets at the patient level.

    LEAKAGE PREVENTION: split is performed on subject_id (patient), not
    stay_id (ICU stay). A patient with multiple ICU stays must appear
    entirely in either train or test, never both. Splitting on stay_id
    would allow the same patient's stays to appear in both sets, inflating
    test performance by allowing the model to have implicitly seen the
    patient during training.

    GroupShuffleSplit from scikit-learn enforces this constraint explicitly.
    The random_state ensures reproducibility across runs.

    Args:
        features: complete feature matrix with stay_id, subject_id, label
        test_size: proportion of patients for test set (default 0.2)
        random_state: random seed for reproducibility

    Returns:
        train: training set DataFrame
        test: held-out test set DataFrame (not to be touched until final eval)
    """
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )

    train_idx, test_idx = next(splitter.split(features, groups=features["subject_id"]))

    train = features.iloc[train_idx].copy()
    test = features.iloc[test_idx].copy()

    # Verify no patient overlap between train and test
    train_patients = set(train["subject_id"])
    test_patients = set(test["subject_id"])
    overlap = train_patients & test_patients
    assert len(overlap) == 0, (
        f"LEAKAGE DETECTED: {len(overlap)} patients appear in both "
        f"train and test sets"
    )

    logging.info(
        f"Train/test split: {len(train):,} train stays "
        f"({train['label'].mean() * 100:.1f}% positive), "
        f"{len(test):,} test stays "
        f"({test['label'].mean() * 100:.1f}% positive)"
    )
    logging.info(
        f"Patients: {len(train_patients):,} train, "
        f"{len(test_patients):,} test, "
        f"0 overlap confirmed"
    )

    return train, test


# ---------------------------------------------------------------------------
# MAIN ORCHESTRATOR
# ---------------------------------------------------------------------------


def main():
    """Run the full feature engineering pipeline in dependency order.

    Steps 1-2 are data loading and leakage enforcement.
    Steps 3-7 are clinical derivation (SOFA, sepsis labelling, cohort finalisation).
    Steps 8-11 are feature engineering, imputation, splitting, and output.
    """
    logging.info("=== features.py: starting feature engineering pipeline ===")

    # Step 1 - load
    extracts = load_cleaned_extracts()
    cohort = extracts["cohort"]

    # Step 2 - enforce 6-hour observation window on all event tables
    # Note: infection_components and medications use FULL stay data for
    # sepsis labelling (steps 4-5). Windowed versions are passed to
    # feature engineering (step 8) only.
    logging.info("Enforcing 6-hour observation window...")
    vitals = enforce_observation_window(
        cohort, extracts["vitals"], "charttime", "stay_id", "vitals"
    )
    labs = enforce_observation_window(
        cohort, extracts["labs"], "charttime", "hadm_id", "labs"
    )
    vasopressors = enforce_observation_window(
        cohort, extracts["vasopressors"], "starttime", "stay_id", "vasopressors"
    )
    urine_output = enforce_observation_window(
        cohort, extracts["urine_output"], "charttime", "stay_id", "urine_output"
    )
    ventilation = enforce_observation_window(
        cohort, extracts["ventilation"], "starttime", "stay_id", "ventilation"
    )

    # Step 3 - derive SOFA scores from windowed data
    logging.info("Deriving SOFA scores...")
    sofa_scores = compute_sofa_total(
        cohort,
        vitals,
        labs,
        vasopressors,
        urine_output,
        ventilation,
    )

    # Step 4 - derive baseline SOFA from pre-admission labs (full labs, not windowed)
    logging.info("Computing baseline SOFA from pre-admission labs...")
    baseline_sofa = compute_baseline_sofa(cohort, extracts["labs"])

    # Step 5 - derive suspected infection from full stay data
    # Full stay infection/medication data used here intentionally.
    # These are NOT passed to feature engineering - only used for labelling.
    logging.info("Deriving suspected infection times...")
    t_suspicion = derive_suspected_infection(
        cohort,
        extracts["infection_components"],
        extracts["medications"],
    )

    # Step 6 - derive sepsis onset label
    logging.info("Deriving sepsis onset labels...")
    sepsis_labels = derive_sepsis_onset(
        cohort,
        sofa_scores,
        t_suspicion,
        baseline_sofa,
    )

    # Step 7 - finalise cohort: exclude stays with sepsis present on admission
    # This is the deferred exclusion from clean.py, now possible since SOFA
    # scores are available. Documented in docs/variable_logic.md.
    logging.info("Finalising cohort: excluding sepsis-on-admission stays...")
    cohort = exclude_sepsis_on_admission(cohort, sepsis_labels)

    # Step 8 - build feature matrix from windowed data
    logging.info("Building feature matrix...")
    features = build_feature_matrix(
        cohort_df=cohort,
        vitals_window=vitals,
        labs_window=labs,
        vasopressors_window=vasopressors,
        urine_output_window=urine_output,
        ventilation_window=ventilation,
        comorbidities_df=extracts["comorbidities"],
        sofa_scores=sofa_scores,
        sepsis_labels=sepsis_labels,
    )

    # Step 9 - imputation
    logging.info("Applying imputation...")
    features = impute_features(features)

    # Label-derived columns (t_sepsis_hour, sofa_increase) are useful for
    # evaluation (e.g. checking risk score trajectory vs. time-to-onset) but
    # must never sit in the training feature matrix - they are downstream of
    # the label itself and would leak sepsis-onset timing into the model.
    metadata_cols = ["stay_id", "subject_id", "label", "t_sepsis_hour", "sofa_increase"]
    labels_metadata = features[metadata_cols].copy()

    feature_cols = [
        c for c in features.columns if c not in metadata_cols
    ]
    features = features[["stay_id", "subject_id", "label"] + feature_cols]

    # Step 10 - train/test split at patient level
    # LEAKAGE PREVENTION: split on subject_id not stay_id.
    # See docstring for split_train_test() for full rationale.
    logging.info("Splitting train/test at patient level...")
    train, test = split_train_test(features)

    # Step 11 - save outputs
    logging.info("Saving feature matrix outputs...")
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    train.to_parquet(EXTRACT_DIR / "features_train.parquet", index=False)
    test.to_parquet(EXTRACT_DIR / "features_test.parquet", index=False)
    labels_metadata.to_parquet(EXTRACT_DIR / "labels_metadata.parquet", index=False)

    # Save feature names for modelling scripts
    with open(EXTRACT_DIR / "feature_names.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    logging.info(
        f"Outputs written: "
        f"features_train.parquet ({len(train):,} rows), "
        f"features_test.parquet ({len(test):,} rows), "
        f"labels_metadata.parquet ({len(labels_metadata):,} rows), "
        f"feature_names.json ({len(feature_cols)} features)"
    )
    logging.info("=== features.py complete ===")


if __name__ == "__main__":
    main()
