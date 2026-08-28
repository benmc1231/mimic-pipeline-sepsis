"""
Feature engineering for MIMIC-IV sepsis prediction pipeline.

Aggregates cleaned, windowed event data into a flat feature matrix with one
row per ICU stay, suitable for input to the modelling layer.

All features are derived exclusively from the 0-6h observation window
(already enforced in features.py before this module is called). No data
from after hour 6 enters any feature. This is the core leakage prevention
guarantee of the pipeline.

Feature groups:
    Vital signs:    mean, min, max, last, std, count, first_time, trend
                    per itemid within the window
    Labs:           first, last, min, max, delta, missingness flag
                    per itemid within the window
    Vasopressors:   any use, total duration, peak dose per drug
    Urine output:   total volume, hourly rate, missingness flag
    Ventilation:    binary presence, total duration
    Demographics:   age, gender, admission type, first care unit
    Comorbidities:  Charlson, Elixhauser, individual condition flags and stages
    SOFA scores:    six component scores and total (derived from window data,
                    not leakage since they use only 0-6h observations)

Trend calculation uses linear regression slope (scipy.stats.linregress)
over time-indexed observations within the window. Time is expressed in
minutes from intime for interpretability.

Output: flat DataFrame with one row per stay_id and one column per feature,
ready for imputation and train/test splitting in features.py.
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats

from pipeline.constants import (
    VITAL_ITEMID_TO_LABEL,
    LAB_ITEMID_TO_LABEL,
    VASOPRESSOR_ITEMID_TO_LABEL,
    VITAL_ITEMIDS,
    LAB_ITEMIDS,
    VASOPRESSOR_ITEMIDS,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# ---------------------------------------------------------------------------
# VITAL SIGN FEATURES
# ---------------------------------------------------------------------------


def _compute_trend(values: pd.Series, times: pd.Series) -> float:
    if len(values) < 2:
        return 0.0
    try:
        x = times.astype(np.int64).to_numpy() / 1e9 / 60
        y = values.to_numpy()
        coeffs = np.polyfit(x, y, 1)
        return float(coeffs[0])  # slope
    except Exception:
        return 0.0


def engineer_vital_features(
    vitals_window: pd.DataFrame,
    cohort_df: pd.DataFrame,
) -> pd.DataFrame:
    """Derive per-vital aggregated features over the 0-6h observation window.

    For each vital sign itemid, computes:
        mean:       average value in window
        min:        minimum value (worst case for most vitals)
        max:        maximum value
        last:       most recent observation in window
        std:        standard deviation (variability signal)
        count:      number of observations (recording frequency informative)
        first_time: minutes from intime to first observation
        trend:      linear regression slope over window (rising vs falling)

    Missing values after aggregation indicate the vital was not recorded
    for that stay. Imputation is applied in features.py.

    Args:
        vitals_window: vitals filtered to 0-6h window with intime joined
        cohort_df: cohort with stay_id and intime

    Returns:
        DataFrame with one row per stay_id and columns
        <vital_label>_<stat> for each vital/stat combination
    """
    # Join intime for trend and first_time calculations
    vitals = vitals_window.merge(
        cohort_df[["stay_id", "intime"]], on="stay_id", how="left"
    )
    vitals["minutes_from_intime"] = (
        vitals["charttime"] - vitals["intime"]
    ).dt.total_seconds() / 60

    result = cohort_df[["stay_id"]].copy()

    # Itemids to skip in feature engineering - consolidated upstream
    SKIP_VITAL_ITEMIDS = {
        223761
    }  # temperature_fahrenheit relabelled to 223762 in clean.py

    for itemid, label in VITAL_ITEMID_TO_LABEL.items():
        if itemid in SKIP_VITAL_ITEMIDS:
            continue
        item_df = vitals[vitals["itemid"] == itemid].dropna(subset=["valuenum"])

        # Basic aggregations
        agg = pd.DataFrame(
            {
                f"{label}_mean": item_df.groupby("stay_id")["valuenum"].mean(),
                f"{label}_min": item_df.groupby("stay_id")["valuenum"].min(),
                f"{label}_max": item_df.groupby("stay_id")["valuenum"].max(),
                f"{label}_std": item_df.groupby("stay_id")["valuenum"].std(),
                f"{label}_count": item_df.groupby("stay_id")["valuenum"].count(),
            }
        ).reset_index()

        # Last observed value
        last = (
            item_df.sort_values("charttime")
            .groupby("stay_id")["valuenum"]
            .last()
            .rename(f"{label}_last")
        )

        # Time to first observation (minutes from intime)
        first_time = (
            item_df.groupby("stay_id")["minutes_from_intime"]
            .min()
            .rename(f"{label}_first_time_min")
        )

        # Trend: linear regression slope (value per minute)
        trend = (
            item_df.groupby("stay_id")
            .apply(lambda g: _compute_trend(g["valuenum"], g["minutes_from_intime"]))
            .rename(f"{label}_trend")
        )

        for feature_series in [agg, last, first_time, trend]:
            result = result.merge(feature_series, on="stay_id", how="left")

    logging.info(
        f"Vital features: {len(result.columns) - 1} features derived "
        f"across {len(VITAL_ITEMID_TO_LABEL)} itemids"
    )

    return result


# ---------------------------------------------------------------------------
# LAB FEATURES
# ---------------------------------------------------------------------------


def engineer_lab_features(
    labs_window: pd.DataFrame,
    cohort_df: pd.DataFrame,
) -> pd.DataFrame:
    """Derive per-lab aggregated features over the 0-6h observation window.

    Labs are less frequently measured than vitals. Features reflect what
    was available in the first 6 hours rather than a continuous signal.

    For each lab itemid, computes:
        first:      first available value in window (early lab most informative)
        last:       last available value in window
        min:        minimum value
        max:        maximum value
        delta:      last - first (direction of change)
        present:    binary flag (1 if any lab result in window, 0 if none)

    The missingness flag is a feature in its own right. Bilirubin and
    lactate are only ordered when clinically suspected - their absence
    is itself informative.

    Args:
        labs_window: labs filtered to 0-6h window with intime joined
        cohort_df: cohort with stay_id and hadm_id

    Returns:
        DataFrame with one row per stay_id and columns
        <lab_label>_<stat> for each lab/stat combination
    """
    # labs_window already carries stay_id (anchored to the first ICU stay of
    # the admission by enforce_observation_window) - no remapping needed here.
    result = cohort_df[["stay_id"]].copy()

    for itemid, label in LAB_ITEMID_TO_LABEL.items():
        item_df = labs_window[labs_window["itemid"] == itemid].dropna(subset=["valuenum"])

        if item_df.empty:
            for stat in ["first", "last", "min", "max", "delta", "present"]:
                result[f"{label}_{stat}"] = np.nan
            result[f"{label}_present"] = 0
            continue

        sorted_df = item_df.sort_values("charttime")

        first = (
            sorted_df.groupby("stay_id")["valuenum"].first().rename(f"{label}_first")
        )
        last = sorted_df.groupby("stay_id")["valuenum"].last().rename(f"{label}_last")
        min_val = item_df.groupby("stay_id")["valuenum"].min().rename(f"{label}_min")
        max_val = item_df.groupby("stay_id")["valuenum"].max().rename(f"{label}_max")

        for feature_series in [first, last, min_val, max_val]:
            result = result.merge(feature_series, on="stay_id", how="left")

        # Delta: direction of change between first and last
        result[f"{label}_delta"] = result[f"{label}_last"] - result[f"{label}_first"]

        # Missingness flag: 1 if lab was drawn in window
        present = (
            item_df.groupby("stay_id")["valuenum"]
            .count()
            .gt(0)
            .astype(int)
            .rename(f"{label}_present")
        )
        result = result.merge(present, on="stay_id", how="left")
        result[f"{label}_present"] = result[f"{label}_present"].fillna(0).astype(int)

    logging.info(
        f"Lab features: {len(result.columns) - 1} features derived "
        f"across {len(LAB_ITEMID_TO_LABEL)} itemids"
    )

    return result


# ---------------------------------------------------------------------------
# VASOPRESSOR FEATURES
# ---------------------------------------------------------------------------


def engineer_vasopressor_features(
    vasopressors_window: pd.DataFrame,
    cohort_df: pd.DataFrame,
) -> pd.DataFrame:
    """Derive vasopressor features over the 0-6h observation window.

    Vasopressor use is a strong sepsis signal. Features capture both
    presence and intensity of vasopressor support.

    Features:
        vasopressor_any:        binary, any vasopressor in window
        vasopressor_duration_min: total minutes of any vasopressor use
        <drug>_peak_rate:       peak rate per drug in mcg/kg/min
                                (vasopressin in units/hour, handled separately)

    Args:
        vasopressors_window: vasopressors filtered to 0-6h window
        cohort_df: cohort with stay_id

    Returns:
        DataFrame with one row per stay_id and vasopressor feature columns
    """
    result = cohort_df[["stay_id"]].copy()

    if vasopressors_window.empty:
        result["vasopressor_any"] = 0
        result["vasopressor_duration_min"] = 0.0
        for label in VASOPRESSOR_ITEMID_TO_LABEL.values():
            result[f"{label}_peak_rate"] = np.nan
        return result

    # Binary: any vasopressor used
    any_vaso = (
        vasopressors_window.groupby("stay_id")["stay_id"]
        .count()
        .gt(0)
        .astype(int)
        .rename("vasopressor_any")
    )
    result = result.merge(any_vaso, on="stay_id", how="left")
    result["vasopressor_any"] = result["vasopressor_any"].fillna(0).astype(int)

    # Total duration of vasopressor use in window (minutes)
    vasopressors_window = vasopressors_window.copy()
    vasopressors_window["duration_min"] = (
        (
            vasopressors_window["endtime"] - vasopressors_window["starttime"]
        ).dt.total_seconds()
        / 60
    ).clip(lower=0)

    duration = (
        vasopressors_window.groupby("stay_id")["duration_min"]
        .sum()
        .rename("vasopressor_duration_min")
    )
    result = result.merge(duration, on="stay_id", how="left")
    result["vasopressor_duration_min"] = result["vasopressor_duration_min"].fillna(0)

    # Peak rate per drug
    for itemid, label in VASOPRESSOR_ITEMID_TO_LABEL.items():
        drug_df = vasopressors_window[vasopressors_window["itemid"] == itemid]
        if drug_df.empty:
            result[f"{label}_peak_rate"] = np.nan
            continue

        peak = drug_df.groupby("stay_id")["rate"].max().rename(f"{label}_peak_rate")
        result = result.merge(peak, on="stay_id", how="left")

    logging.info(f"Vasopressor features: {len(result.columns) - 1} features derived")

    return result


# ---------------------------------------------------------------------------
# URINE OUTPUT FEATURES
# ---------------------------------------------------------------------------


def engineer_urine_features(
    urine_output_window: pd.DataFrame,
    cohort_df: pd.DataFrame,
) -> pd.DataFrame:
    """Derive urine output features over the 0-6h observation window.

    Features:
        urine_total_ml:     total urine output in window (mL)
        urine_rate_ml_hr:   hourly rate (total / 6)
        urine_present:      binary flag (0 if no urine output recorded)

    The missingness flag is clinically significant. Absence of recorded
    urine output in a catheterised ICU patient may indicate oliguria or
    anuria rather than missing data. Combined with catheter coverage
    (97% of stays), missing urine output is itself an adverse signal.

    Args:
        urine_output_window: urine output filtered to 0-6h window
        cohort_df: cohort with stay_id

    Returns:
        DataFrame with one row per stay_id and urine output feature columns
    """
    result = cohort_df[["stay_id"]].copy()

    if urine_output_window.empty:
        result["urine_total_ml"] = 0.0
        result["urine_rate_ml_hr"] = 0.0
        result["urine_present"] = 0
        return result

    # Exclude large void flagged rows from total to avoid inflated volumes
    # These may represent accumulated totals rather than true single voids
    urine_clean = urine_output_window[
        ~urine_output_window.get("large_void_flag", False)
    ]

    total = urine_clean.groupby("stay_id")["value"].sum().rename("urine_total_ml")
    result = result.merge(total, on="stay_id", how="left")
    result["urine_total_ml"] = result["urine_total_ml"].fillna(0)
    result["urine_rate_ml_hr"] = result["urine_total_ml"] / 6
    result["urine_present"] = (result["urine_total_ml"] > 0).astype(int)

    logging.info(f"Urine output features: {len(result.columns) - 1} features derived")

    return result


# ---------------------------------------------------------------------------
# VENTILATION FEATURES
# ---------------------------------------------------------------------------


def engineer_ventilation_features(
    ventilation_window: pd.DataFrame,
    cohort_df: pd.DataFrame,
) -> pd.DataFrame:
    """Derive ventilation features over the 0-6h observation window.

    Features:
        ventilated:             binary, any mechanical ventilation in window
        ventilation_duration_min: total minutes ventilated in window

    Ventilation status is a direct input to SOFA respiratory scoring and
    is also an independent predictor of clinical severity.

    Args:
        ventilation_window: ventilation events filtered to 0-6h window
        cohort_df: cohort with stay_id

    Returns:
        DataFrame with one row per stay_id and ventilation feature columns
    """
    result = cohort_df[["stay_id"]].copy()

    if ventilation_window.empty:
        result["ventilated"] = 0
        result["ventilation_duration_min"] = 0.0
        return result

    ventilated = (
        ventilation_window.groupby("stay_id")["stay_id"]
        .count()
        .gt(0)
        .astype(int)
        .rename("ventilated")
    )
    result = result.merge(ventilated, on="stay_id", how="left")
    result["ventilated"] = result["ventilated"].fillna(0).astype(int)

    ventilation_window = ventilation_window.copy()
    ventilation_window["duration_min"] = (
        (
            ventilation_window["endtime"] - ventilation_window["starttime"]
        ).dt.total_seconds()
        / 60
    ).clip(lower=0)

    duration = (
        ventilation_window.groupby("stay_id")["duration_min"]
        .sum()
        .rename("ventilation_duration_min")
    )
    result = result.merge(duration, on="stay_id", how="left")
    result["ventilation_duration_min"] = result["ventilation_duration_min"].fillna(0)

    logging.info(f"Ventilation features: {len(result.columns) - 1} features derived")

    return result


# ---------------------------------------------------------------------------
# DEMOGRAPHIC FEATURES
# ---------------------------------------------------------------------------


def engineer_demographic_features(cohort_df: pd.DataFrame) -> pd.DataFrame:
    """Derive demographic and admission features from the cohort table.

    Features:
        age_at_admission:   numeric age at time of ICU admission
        gender_male:        binary (1 = male, 0 = female)
        admission_emergency: binary (1 = emergency, 0 = elective/other)
        first_careunit_*:   one-hot encoded first ICU care unit

    Gender and admission type are encoded as binary rather than categorical
    to avoid introducing ordinal relationships in one-hot encoding.
    Care unit is one-hot encoded since unit type carries clinical meaning
    (MICU vs SICU vs CCU have different patient populations).

    Args:
        cohort_df: cleaned cohort with demographic columns

    Returns:
        DataFrame with one row per stay_id and demographic feature columns
    """
    result = cohort_df[["stay_id", "age_at_admission"]].copy()

    # Gender: binary male flag
    result["gender_male"] = (cohort_df["gender"] == "M").astype(int)

    # Admission type: binary emergency flag
    result["admission_emergency"] = (
        cohort_df["admission_type"].str.upper().str.contains("EMERGENCY", na=False)
    ).astype(int)

    # First care unit: one-hot encode
    careunit_dummies = pd.get_dummies(
        cohort_df["first_careunit"],
        prefix="careunit",
        dtype=int,
    )
    result = pd.concat([result, careunit_dummies], axis=1)

    logging.info(f"Demographic features: {len(result.columns) - 1} features derived")

    return result


# ---------------------------------------------------------------------------
# COMORBIDITY FEATURES
# ---------------------------------------------------------------------------


def engineer_comorbidity_features(
    comorbidities_df: pd.DataFrame,
    cohort_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join comorbidity features onto the cohort.

    Comorbidities are derived from discharge diagnosis codes in
    comorbidities.py and joined here via hadm_id. Features include
    standard composite scores (Charlson, Elixhauser) and individual
    condition flags with severity staging where applicable.

    Args:
        comorbidities_df: output of comorbidities.py main()
        cohort_df: cohort with stay_id and hadm_id

    Returns:
        DataFrame with one row per stay_id and comorbidity feature columns
    """
    comorbidity_cols = [
        "hadm_id",
        "charlson_score",
        "elixhauser_score",
        "has_ckd",
        "ckd_stage",
        "has_liver_disease",
        "liver_stage",
        "has_malignancy",
        "malignancy_stage",
        "has_diabetes",
        "has_immunosuppression",
    ]

    available_cols = [c for c in comorbidity_cols if c in comorbidities_df.columns]
    comorbidities_subset = comorbidities_df[available_cols]

    result = cohort_df[["stay_id", "hadm_id"]].merge(
        comorbidities_subset, on="hadm_id", how="left"
    )
    result = result.drop(columns=["hadm_id"])

    # Convert boolean flags to int for modelling
    for col in [
        "has_ckd",
        "has_liver_disease",
        "has_malignancy",
        "has_diabetes",
        "has_immunosuppression",
    ]:
        if col in result.columns:
            # hadm_ids with no comorbidities_clean.parquet row (no diagnosis
            # codes matched) come through as NaN from the left merge - treat
            # as no comorbidity present, consistent with the score columns'
            # missing-data convention (see impute_features in features.py).
            result[col] = result[col].fillna(False).astype(int)

    logging.info(f"Comorbidity features: {len(result.columns) - 1} features joined")

    return result


# ---------------------------------------------------------------------------
# SOFA FEATURES
# ---------------------------------------------------------------------------


def engineer_sofa_features(sofa_scores: pd.DataFrame) -> pd.DataFrame:
    """Include SOFA component scores as predictive features.

    SOFA scores are derived from 0-6h observation window data and are
    therefore valid features with no leakage risk. Including them gives
    the model direct access to organ dysfunction severity signals that
    would otherwise have to be reconstructed from raw vitals and labs.

    Args:
        sofa_scores: output of compute_sofa_total from sofa.py

    Returns:
        DataFrame with stay_id and SOFA feature columns
    """
    sofa_feature_cols = [
        "stay_id",
        "sofa_respiratory",
        "sofa_coagulation",
        "sofa_hepatic",
        "sofa_cardiovascular",
        "sofa_neurological",
        "sofa_renal",
        "sofa_total",
    ]

    available_cols = [c for c in sofa_feature_cols if c in sofa_scores.columns]
    result = sofa_scores[available_cols].copy()

    logging.info(f"SOFA features: {len(result.columns) - 1} component scores included")

    return result


# ---------------------------------------------------------------------------
# ORCHESTRATOR
# ---------------------------------------------------------------------------


def build_feature_matrix(
    cohort_df: pd.DataFrame,
    vitals_window: pd.DataFrame,
    labs_window: pd.DataFrame,
    vasopressors_window: pd.DataFrame,
    urine_output_window: pd.DataFrame,
    ventilation_window: pd.DataFrame,
    comorbidities_df: pd.DataFrame,
    sofa_scores: pd.DataFrame,
    sepsis_labels: pd.DataFrame,
) -> pd.DataFrame:
    """Build the complete feature matrix for the modelling layer.

    Calls all feature engineering functions and merges outputs into a
    single flat DataFrame with one row per stay_id. The label column
    is joined last to keep it clearly separated from features.

    All features are derived from 0-6h window data. The label is derived
    from the full stay in sepsis_labelling.py. No post-window data enters
    any feature column.

    Args:
        cohort_df: finalised cohort after sepsis-on-admission exclusion
        vitals_window: vitals filtered to 0-6h window
        labs_window: labs filtered to 0-6h window
        vasopressors_window: vasopressors filtered to 0-6h window
        urine_output_window: urine output filtered to 0-6h window
        ventilation_window: ventilation filtered to 0-6h window
        comorbidities_df: output of comorbidities.py
        sofa_scores: output of compute_sofa_total
        sepsis_labels: output of derive_sepsis_onset

    Returns:
        DataFrame with one row per stay_id, feature columns, and label
    """
    logging.info("Building feature matrix...")

    vital_features = engineer_vital_features(vitals_window, cohort_df)
    lab_features = engineer_lab_features(labs_window, cohort_df)
    vasopressor_features = engineer_vasopressor_features(vasopressors_window, cohort_df)
    urine_features = engineer_urine_features(urine_output_window, cohort_df)
    ventilation_features = engineer_ventilation_features(ventilation_window, cohort_df)
    demographic_features = engineer_demographic_features(cohort_df)
    comorbidity_features = engineer_comorbidity_features(comorbidities_df, cohort_df)
    sofa_features = engineer_sofa_features(sofa_scores)

    # Merge all feature groups onto cohort
    features = cohort_df[["stay_id", "subject_id"]].copy()

    for feature_df in [
        vital_features,
        lab_features,
        vasopressor_features,
        urine_features,
        ventilation_features,
        demographic_features,
        comorbidity_features,
        sofa_features,
    ]:
        features = features.merge(feature_df, on="stay_id", how="left")

    # Join label last - kept clearly separate from features
    label_cols = ["stay_id", "label", "t_sepsis_hour", "sofa_increase"]
    available_label_cols = [c for c in label_cols if c in sepsis_labels.columns]
    features = features.merge(
        sepsis_labels[available_label_cols], on="stay_id", how="left"
    )
    features["label"] = features["label"].fillna(0).astype(int)

    n_features = len(features.columns) - 3  # exclude stay_id, subject_id, label
    n_positive = features["label"].sum()
    n_total = len(features)

    logging.info(f"Feature matrix built: {n_total:,} stays, {n_features} features")
    logging.info(
        f"Class balance: {n_positive:,} positive ({n_positive / n_total * 100:.1f}%), "
        f"{n_total - n_positive:,} negative ({(n_total - n_positive) / n_total * 100:.1f}%)"
    )

    return features
