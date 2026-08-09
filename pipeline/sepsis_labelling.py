"""
Sepsis-3 onset labelling for MIMIC-IV sepsis prediction pipeline.

This module implements the Sepsis-3 suspected infection criterion and derives
the binary sepsis onset label used for model training.

Sepsis-3 definition (Singer et al., JAMA 2016):
    Sepsis = life-threatening organ dysfunction caused by a dysregulated host
    response to infection, operationalised as:
        1. Suspected infection (t_suspicion)
        2. Acute SOFA increase of 2 or more points from baseline

Suspected infection is defined as either:
    - Antibiotic administration followed by blood culture within 24 hours, OR
    - Blood culture followed by antibiotic administration within 72 hours
    t_suspicion is the earlier of the two events when the pairing is met.

This module operates on FULL STAY data for infection components and medications,
not the 0-6h observation window. Suspected infection may occur at any point
during the stay and is needed to:
    a) Exclude patients septic on admission (t_sepsis <= 6h)
    b) Label patients with onset in the prediction window (6h < t_sepsis <= 24h)

SOFA scores are derived from the 0-6h observation window in sofa.py.

Baseline SOFA is estimated from pre-admission labs (up to 24h before intime)
where available. Only lab-based components (renal, hepatic, coagulation) can
be estimated this way. Respiratory, cardiovascular, and neurological baselines
default to 0 due to absence of pre-admission chartevents in MIMIC-IV.
Where no pre-admission labs exist, baseline defaults to 0. This is documented
in docs/modelling_decisions.md.

Prediction label:
    label = 1  where t_sepsis falls between hours 6 and 24 post-admission
    label = 0  where no sepsis onset occurs within hours 6-24
    Excluded:  stays where t_sepsis <= 6h (sepsis present on admission)
"""

import logging

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# ---------------------------------------------------------------------------
# BASELINE SOFA
# ---------------------------------------------------------------------------


def compute_baseline_sofa(
    cohort_df: pd.DataFrame,
    labs_df: pd.DataFrame,
) -> pd.DataFrame:
    """Estimate baseline SOFA from pre-admission labs where available.

    Uses lab values from the 24 hours before ICU admission to estimate
    baseline organ function. Only lab-based SOFA components are derivable
    here (coagulation, hepatic, renal via creatinine). Respiratory,
    cardiovascular, and neurological baselines default to 0 since
    pre-admission chartevents are not available in MIMIC-IV for most patients.

    Where no pre-admission labs exist for a component, that component
    baseline defaults to 0. This means the SOFA increase criterion is
    conservative for patients with pre-existing organ dysfunction who have
    no pre-admission labs.

    Data profiling found 161,404 pre-admission lab rows across the four
    target SOFA lab itemids, sufficient to meaningfully improve on a
    universal baseline of 0.

    Args:
        cohort_df: cleaned cohort with stay_id, hadm_id, intime
        labs_df: full (non-windowed) labs_clean.parquet

    Returns:
        DataFrame with stay_id and baseline_sofa (0 where no data)
    """
    cohort_keys = cohort_df[["stay_id", "hadm_id", "intime"]]

    # Filter labs to 24h pre-admission window
    labs = labs_df.merge(cohort_keys, on="hadm_id", how="inner")
    pre_admission = labs[
        (labs["charttime"] < labs["intime"])
        & (labs["charttime"] >= labs["intime"] - pd.Timedelta(hours=24))
    ]

    if pre_admission.empty:
        logging.info("Baseline SOFA: no pre-admission labs found, defaulting all to 0")
        result = cohort_df[["stay_id"]].copy()
        result["baseline_sofa"] = 0
        return result

    # Creatinine baseline (renal component)
    creatinine = (
        pre_admission[pre_admission["itemid"] == 50912]
        .groupby("stay_id")["valuenum"]
        .max()
    )
    creatinine_score = pd.cut(
        creatinine,
        bins=[-np.inf, 1.2, 2.0, 3.5, 5.0, np.inf],
        labels=[0, 1, 2, 3, 4],
        right=False,
    ).astype(float)

    # Bilirubin baseline (hepatic component)
    bilirubin = (
        pre_admission[pre_admission["itemid"] == 50885]
        .groupby("stay_id")["valuenum"]
        .max()
    )
    bilirubin_score = pd.cut(
        bilirubin,
        bins=[-np.inf, 1.2, 2.0, 6.0, 12.0, np.inf],
        labels=[0, 1, 2, 3, 4],
        right=False,
    ).astype(float)

    # Platelet baseline (coagulation component)
    platelets = (
        pre_admission[pre_admission["itemid"] == 51265]
        .groupby("stay_id")["valuenum"]
        .min()
    )
    platelet_score = pd.cut(
        platelets,
        bins=[-np.inf, 20, 50, 100, 150, np.inf],
        labels=[4, 3, 2, 1, 0],
        right=False,
    ).astype(float)

    # Merge component scores onto full cohort
    baseline = cohort_df[["stay_id"]].copy()
    baseline = baseline.merge(
        creatinine_score.rename("creatinine_baseline"), on="stay_id", how="left"
    )
    baseline = baseline.merge(
        bilirubin_score.rename("bilirubin_baseline"), on="stay_id", how="left"
    )
    baseline = baseline.merge(
        platelet_score.rename("platelet_baseline"), on="stay_id", how="left"
    )

    # Fill missing components with 0 and sum to total baseline
    component_cols = ["creatinine_baseline", "bilirubin_baseline", "platelet_baseline"]
    baseline[component_cols] = baseline[component_cols].fillna(0)
    baseline["baseline_sofa"] = baseline[component_cols].sum(axis=1).astype(int)

    n_with_baseline = (baseline["baseline_sofa"] > 0).sum()
    logging.info(
        f"Baseline SOFA: {n_with_baseline:,} stays have non-zero baseline "
        f"from pre-admission labs "
        f"({n_with_baseline / len(baseline) * 100:.1f}%)"
    )

    return baseline[["stay_id", "baseline_sofa"]]


# ---------------------------------------------------------------------------
# SUSPECTED INFECTION
# ---------------------------------------------------------------------------


def derive_suspected_infection(
    cohort_df: pd.DataFrame,
    infection_components_df: pd.DataFrame,
    medications_df: pd.DataFrame,
) -> pd.DataFrame:
    """Derive suspected infection time (t_suspicion) per ICU stay.

    Implements the Sepsis-3 operational definition of suspected infection:
        - Antibiotic given first: blood culture must follow within 24 hours
        - Culture ordered first: antibiotic must follow within 72 hours

    t_suspicion is the earlier of the two events (antibiotic or culture)
    when the pairing condition is satisfied. If multiple valid pairings
    exist for a stay, the earliest t_suspicion is used.

    Full stay data is used (not the 0-6h window) since suspected infection
    may occur at any point during the admission.

    The cross-join of cultures and antibiotics within each stay can be
    memory-intensive for stays with many events. If memory pressure is
    observed, consider processing stays in batches.

    Args:
        cohort_df: cleaned cohort with stay_id, hadm_id, intime
        infection_components_df: blood culture records (full stay)
        medications_df: antibiotic administration records (full stay)

    Returns:
        DataFrame with stay_id and t_suspicion (null if no suspected infection)
    """
    cohort_keys = cohort_df[["stay_id", "hadm_id", "intime"]].drop_duplicates()

    # Cultures: one row per culture order
    cultures = infection_components_df[["hadm_id", "charttime"]].rename(
        columns={"charttime": "culture_time"}
    )
    cultures = cultures.merge(
        cohort_keys[["stay_id", "hadm_id"]], on="hadm_id", how="inner"
    )

    # Antibiotics: one row per administration event
    antibiotics = medications_df[["hadm_id", "charttime"]].rename(
        columns={"charttime": "antibiotic_time"}
    )
    antibiotics = antibiotics.merge(
        cohort_keys[["stay_id", "hadm_id"]], on="hadm_id", how="inner"
    )

    # Cross join within stay to get all culture-antibiotic pairs
    pairs = cultures.merge(antibiotics, on="stay_id", how="inner")

    # Time difference: positive means culture before antibiotic
    pairs["time_diff"] = pairs["antibiotic_time"] - pairs["culture_time"]

    # Sepsis-3 pairing rules
    # Rule 1: antibiotic before culture, culture must follow within 24h
    rule1 = (pairs["time_diff"] < pd.Timedelta(0)) & (
        pairs["time_diff"] >= -pd.Timedelta(hours=24)
    )
    # Rule 2: culture before antibiotic, antibiotic must follow within 72h
    rule2 = (pairs["time_diff"] >= pd.Timedelta(0)) & (
        pairs["time_diff"] <= pd.Timedelta(hours=72)
    )

    valid_pairs = pairs[rule1 | rule2].copy()

    if valid_pairs.empty:
        logging.warning("No valid suspected infection pairs found")
        result = cohort_df[["stay_id"]].copy()
        result["t_suspicion"] = pd.NaT
        return result

    # t_suspicion is the earlier of the two events in each valid pair
    valid_pairs["t_suspicion"] = valid_pairs[["culture_time", "antibiotic_time"]].min(
        axis=1
    )

    # Earliest t_suspicion per stay where multiple valid pairs exist
    t_suspicion = valid_pairs.groupby("stay_id")["t_suspicion"].min().reset_index()

    result = cohort_df[["stay_id"]].merge(t_suspicion, on="stay_id", how="left")

    n_with_suspicion = result["t_suspicion"].notna().sum()
    logging.info(
        f"Suspected infection: {n_with_suspicion:,} of {len(result):,} stays "
        f"({n_with_suspicion / len(result) * 100:.1f}%)"
    )

    return result


# ---------------------------------------------------------------------------
# SEPSIS ONSET LABELLING
# ---------------------------------------------------------------------------


def derive_sepsis_onset(
    cohort_df: pd.DataFrame,
    sofa_scores: pd.DataFrame,
    t_suspicion_df: pd.DataFrame,
    baseline_sofa_df: pd.DataFrame,
) -> pd.DataFrame:
    """Derive sepsis onset time and binary prediction label.

    Sepsis-3 onset requires both:
        1. Suspected infection (t_suspicion from derive_suspected_infection)
        2. SOFA increase of 2 or more from baseline

    Baseline SOFA is estimated from pre-admission labs in compute_baseline_sofa.
    Where no pre-admission data exists, baseline defaults to 0. The SOFA
    increase criterion is therefore sofa_total - baseline_sofa >= 2.

    Sepsis onset time (t_sepsis) is defined as t_suspicion where both
    criteria are met. Only one time point per stay is derived since SOFA
    scores are computed over the 0-6h observation window as a single value
    rather than as a time series.

    Prediction label:
        label = 1  where 6h < t_sepsis - intime <= 24h
        label = 0  where no sepsis onset in 6-24h window
        Excluded:  where t_sepsis - intime <= 6h (septic on admission)

    Args:
        cohort_df: cleaned cohort with stay_id and intime
        sofa_scores: output of compute_sofa_total with stay_id and sofa_total
        t_suspicion_df: output of derive_suspected_infection
        baseline_sofa_df: output of compute_baseline_sofa

    Returns:
        DataFrame with stay_id, sofa_total, baseline_sofa, sofa_increase,
        t_suspicion, t_sepsis, t_sepsis_hour, sepsis_on_admission, label
    """
    result = cohort_df[["stay_id", "intime"]].merge(
        sofa_scores[["stay_id", "sofa_total"]], on="stay_id", how="left"
    )
    result = result.merge(t_suspicion_df, on="stay_id", how="left")
    result = result.merge(baseline_sofa_df, on="stay_id", how="left")

    result["sofa_total"] = result["sofa_total"].fillna(0).astype(int)
    result["baseline_sofa"] = result["baseline_sofa"].fillna(0).astype(int)
    result["sofa_increase"] = result["sofa_total"] - result["baseline_sofa"]

    # Sepsis-3 criteria
    sofa_criterion = result["sofa_increase"] >= 2
    infection_criterion = result["t_suspicion"].notna()

    # t_sepsis is t_suspicion where both criteria are met
    result["t_sepsis"] = pd.NaT
    sepsis_mask = sofa_criterion & infection_criterion
    result.loc[sepsis_mask, "t_sepsis"] = result.loc[sepsis_mask, "t_suspicion"]

    # Hours from ICU admission to sepsis onset
    result["t_sepsis_hour"] = (
        result["t_sepsis"] - result["intime"]
    ).dt.total_seconds() / 3600

    # Sepsis on admission: onset at or before hour 6
    # These stays are excluded from the cohort in exclude_sepsis_on_admission()
    result["sepsis_on_admission"] = result["t_sepsis_hour"].notna() & (
        result["t_sepsis_hour"] <= 6
    )

    # Prediction label: sepsis onset strictly between hours 6 and 24
    result["label"] = (
        (result["t_sepsis_hour"] > 6) & (result["t_sepsis_hour"] <= 24)
    ).astype(int)

    # Logging summary
    n_sepsis = result["t_sepsis"].notna().sum()
    n_on_admission = result["sepsis_on_admission"].sum()
    n_label_1 = result["label"].sum()
    n_label_0 = (result["label"] == 0).sum()
    n_total_labelled = n_label_1 + n_label_0

    logging.info(f"Sepsis onset derived: {n_sepsis:,} stays meet Sepsis-3 criteria")
    logging.info(f"Sepsis on admission (to exclude): {n_on_admission:,} stays")
    logging.info(
        f"Prediction labels: {n_label_1:,} positive (label=1), "
        f"{n_label_0:,} negative (label=0)"
    )
    if n_total_labelled > 0:
        logging.info(
            f"Class balance: {n_label_1 / n_total_labelled * 100:.1f}% positive"
        )

    return result[
        [
            "stay_id",
            "sofa_total",
            "baseline_sofa",
            "sofa_increase",
            "t_suspicion",
            "t_sepsis",
            "t_sepsis_hour",
            "sepsis_on_admission",
            "label",
        ]
    ]


# ---------------------------------------------------------------------------
# COHORT FINALISATION
# ---------------------------------------------------------------------------


def exclude_sepsis_on_admission(
    cohort_df: pd.DataFrame,
    sepsis_labels: pd.DataFrame,
) -> pd.DataFrame:
    """Remove stays where Sepsis-3 criteria are met within the first 6 hours.

    This is the final cohort exclusion deferred from clean.py. It cannot be
    applied at clean time because it requires SOFA scores derived from vitals
    and labs, which are not available until features.py runs.

    Patients septic on admission represent a recognition problem rather than
    a prediction problem and must be excluded from both training and test sets.

    Args:
        cohort_df: preliminary cohort from clean.py
        sepsis_labels: output of derive_sepsis_onset

    Returns:
        Final cohort with sepsis-on-admission stays removed
    """
    before = len(cohort_df)

    exclusion_ids = sepsis_labels.loc[sepsis_labels["sepsis_on_admission"], "stay_id"]

    cohort_final = cohort_df[~cohort_df["stay_id"].isin(exclusion_ids)].copy()

    logging.info(
        f"Sepsis-on-admission exclusion: {before - len(cohort_final):,} stays "
        f"removed, {len(cohort_final):,} remaining in final cohort"
    )

    return cohort_final
