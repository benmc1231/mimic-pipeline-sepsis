import logging

import pandas as pd
import numpy as np


def compute_sofa_respiratory(vitals_window, ventilation_window):
    """Respiratory (PF ratio)
    Compute SpO2/FiO2 ratio as a proxy for PaO2/FiO2 (arterial blood gas PaO2 is not reliably available in all patients).
    Use the worst (lowest) ratio observed in the window.
    PF ratio	SOFA score
    >= 400	0
    300-399	1
    200-299	2
    100-199 with ventilation	3
    < 100 with ventilation	4
    Ventilation status at each timepoint must be confirmed from ventilation_events_clean.parquet before assigning scores of 3 or 4.
    FiO2 imputation: assume 0.21 (room air) where FiO2 is missing and patient is not ventilated.
    """

    # Vitals: stay_id	charttime	itemid	valuenum	valueuom
    # 30000153	29/09/2174 6:00:00 PM	220045	111	bpm

    # ventilation: subject_id	stay_id	itemid	starttime	endtime	value	statusdescription
    #              12466550	30000153	225792	29/09/2174 12:00:00 PM	29/09/2174 5:43:00 PM	343	FinishedRunning

    """1. Pull SpO2 rows from vitals_window (itemid 220277)
    2. Pull FiO2 rows from vitals_window (itemid 223835)
    3. For each stay, forward fill FiO2 onto SpO2 timestamps
    4. Where FiO2 still null after forward fill, check ventilation
    5. If ventilated, leave FiO2 null (cannot assume room air for ventilated patient)
    6. If not ventilated, impute FiO2 = 0.21
    7. Compute ratio = SpO2 / FiO2 * 100 (to get ratio in same scale as PaO2/FiO2)
    8. Take minimum ratio per stay
    9. Check ventilation status at the time of the minimum ratio
    10. Apply scoring table using ratio and ventilation flag"""

    # 1
    spo2 = vitals_window[vitals_window["itemid"] == 220277][
        ["stay_id", "charttime", "valuenum"]
    ].rename(columns={"valuenum": "spo2"})
    # 2
    fio2 = vitals_window[vitals_window["itemid"] == 223835][
        ["stay_id", "charttime", "valuenum"]
    ].rename(columns={"valuenum": "fio2"})

    # Step 3 - forward fill FiO2 onto SpO2 timestamps
    combined = (
        pd.concat(
            [
                spo2[["stay_id", "charttime"]],
                fio2[["stay_id", "charttime", "fio2"]],
            ]
        )
        .sort_values(["stay_id", "charttime"])
        .reset_index(drop=True)
    )
    combined["fio2"] = combined.groupby("stay_id")["fio2"].ffill()

    # Filter back to SpO2 rows only
    spo2_times = spo2.set_index(["stay_id", "charttime"]).index
    combined_idx = combined.set_index(["stay_id", "charttime"]).index
    spo2_mask = combined_idx.isin(spo2_times)
    result = combined[spo2_mask].copy()
    result = result.merge(spo2, on=["stay_id", "charttime"], how="left")

    # 4
    # Step 4 - ventilation flag per SpO2 observation
    result_vent = result.merge(
        ventilation_window[["stay_id", "starttime", "endtime"]],
        on="stay_id",
        how="left",
    )

    # True where charttime falls within a ventilation interval
    result_vent["ventilated"] = (
        result_vent["charttime"] >= result_vent["starttime"]
    ) & (result_vent["charttime"] <= result_vent["endtime"])

    # Collapse back to one row per SpO2 observation
    # .any() gives True if ventilated in ANY interval
    ventilated_flag = (
        result_vent.groupby(["stay_id", "charttime"])["ventilated"].any().reset_index()
    )

    result = result.merge(ventilated_flag, on=["stay_id", "charttime"], how="left")
    result["ventilated"] = result["ventilated"].fillna(False)

    # Step 5 and 6 - impute FiO2 where still null
    # Not ventilated and no FiO2 recorded: assume room air
    result.loc[result["fio2"].isna() & ~result["ventilated"], "fio2"] = 0.21
    # Ventilated but no FiO2 recorded: leave null, cannot assume

    # Step 7 - compute ratio
    result["pf_ratio"] = result["spo2"] / result["fio2"] * 100

    # Step 8 - find the index of the minimum ratio per stay
    min_idx = result.groupby("stay_id")["pf_ratio"].idxmin()

    # Pull the full row at that index - gives you the charttime and ventilated status at the worst point
    worst = result.loc[min_idx, ["stay_id", "pf_ratio", "ventilated"]].reset_index(
        drop=True
    )
    # Step 10 - apply scoring table

    conditions = [
        worst["pf_ratio"] >= 400,
        worst["pf_ratio"] >= 300,
        worst["pf_ratio"] >= 200,
        (worst["pf_ratio"] >= 100) & worst["ventilated"],
        (worst["pf_ratio"] < 100) & worst["ventilated"],
    ]

    scores = [0, 1, 2, 3, 4]

    worst["sofa_respiratory"] = np.select(conditions, scores, default=2)

    return worst[["stay_id", "pf_ratio", "sofa_respiratory"]].rename(
        columns={"pf_ratio": "worst_pf_ratio"}
    )


def compute_sofa_coagulation(labs_window):
    """Derive SOFA coagulation score from platelet count.

    Uses the worst (lowest) platelet count observed in the 0-6h window.
    Platelets itemid: 51265 (K/uL).

    Missing platelet data: score imputed as 0 (best case assumption),
    consistent with published MIMIC-IV Sepsis-3 derivation convention.
    """
    platelets = labs_window[labs_window["itemid"] == 51265][
        ["stay_id", "valuenum"]
    ].rename(columns={"valuenum": "platelets"})

    # Worst (lowest) platelet count per stay within the window
    worst = (
        platelets.groupby("stay_id")["platelets"]
        .min()
        .reset_index()
        .rename(columns={"platelets": "worst_platelets"})
    )

    conditions = [
        worst["worst_platelets"] >= 150,
        worst["worst_platelets"] >= 100,
        worst["worst_platelets"] >= 50,
        worst["worst_platelets"] >= 20,
        worst["worst_platelets"] < 20,
    ]

    scores = [0, 1, 2, 3, 4]

    worst["sofa_coagulation"] = np.select(conditions, scores, default=0)

    return worst[["stay_id", "worst_platelets", "sofa_coagulation"]]


def compute_sofa_hepatic(labs_window):
    """Derive SOFA hepatic score from bilirubin count.

    Uses the worst (highest) bilirubin count observed in the 0-6h window.
    Bilirubin itemid: 50885 (mg/dL).

    Missing bilirubin data: score imputed as 0 (best case assumption),
    consistent with published MIMIC-IV Sepsis-3 derivation convention.
    """
    bilirubin = labs_window[labs_window["itemid"] == 50885][
        ["stay_id", "valuenum"]
    ].rename(columns={"valuenum": "bilirubin"})

    # Worst (highest) bilirubin count per stay within the window
    worst = (
        bilirubin.groupby("stay_id")["bilirubin"]
        .max()
        .reset_index()
        .rename(columns={"bilirubin": "worst_bilirubin"})
    )

    conditions = [
        worst["worst_bilirubin"] >= 12,
        worst["worst_bilirubin"] >= 6,
        worst["worst_bilirubin"] >= 2,
        worst["worst_bilirubin"] >= 1.2,
        worst["worst_bilirubin"] < 1.2,
    ]
    # Higher bilirubin = worse hepatic function, so the scores are reversed from the conditions
    scores = [4, 3, 2, 1, 0]

    worst["sofa_hepatic"] = np.select(conditions, scores, default=0)

    return worst[["stay_id", "worst_bilirubin", "sofa_hepatic"]]


def compute_sofa_cardiovascular(vitals_window, vasopressors_window):
    """Derive SOFA cardiovascular score from MAP and vasopressor use.

    Vasopressor scoring takes precedence over MAP scoring where both apply.
    Multiple simultaneous vasopressors: take the highest resulting score.

    Vasopressor itemids and scoring:
        Dopamine (221662):          <= 5 mcg/kg/min = 2, > 5 = 3, > 15 = 4
        Dobutamine (221653):        any dose = 2
        Norepinephrine (221906):    <= 0.1 mcg/kg/min = 3, > 0.1 = 4
        Epinephrine (221289,229617):  <= 0.1 mcg/kg/min = 3, > 0.1 = 4
        Vasopressin (222315):       any dose = 3 (Sepsis-3 convention)
        Phenylephrine (221749 etc): <= 0.1 mcg/kg/min = 3, > 0.1 = 4
        Milrinone (221986):         any dose = 2 (inotrope, same as dobutamine)

    Missing MAP and no vasopressors: score imputed as 0.
    """
    # --- MAP component ---
    # ABP mean (220052) preferred, NIBP mean (220181) as fallback
    # BP consolidation in clean.py already nulled NIBP where ABP exists
    # so taking min across both itemids gives correct worst MAP
    map_vitals = vitals_window[vitals_window["itemid"].isin([220052, 220181])][
        ["stay_id", "valuenum"]
    ].rename(columns={"valuenum": "map_value"})

    worst_map = (
        map_vitals.groupby("stay_id")["map_value"]
        .min()
        .reset_index()
        .rename(columns={"map_value": "worst_map"})
    )

    # MAP-based score (overridden if vasopressors present)
    worst_map["map_score"] = np.where(worst_map["worst_map"] < 70, 1, 0)

    # --- Vasopressor component ---
    # Score each vasopressor administration record individually
    # then take the maximum score per stay

    vaso = vasopressors_window[vasopressors_window["rate"].notna()].copy()

    def vasopressor_score(row):
        itemid = row["itemid"]
        rate = row["rate"]

        if itemid == 221662:  # Dopamine
            if rate > 15:
                return 4
            elif rate > 5:
                return 3
            else:
                return 2

        elif itemid == 221653:  # Dobutamine
            return 2

        elif itemid == 221986:  # Milrinone
            return 2

        elif itemid in [221906, 221289, 229617, 221749, 229632, 229630, 229631]:
            # Norepinephrine, epinephrine, phenylephrine
            if rate > 0.1:
                return 4
            else:
                return 3

        elif itemid == 222315:  # Vasopressin
            return 3

        return 0

    vaso["vaso_score"] = vaso.apply(vasopressor_score, axis=1)

    worst_vaso = (
        vaso.groupby("stay_id")["vaso_score"]
        .max()
        .reset_index()
        .rename(columns={"vaso_score": "worst_vaso_score"})
    )

    # --- Combine MAP and vasopressor scores ---
    result = worst_map.merge(worst_vaso, on="stay_id", how="left")
    result["worst_vaso_score"] = result["worst_vaso_score"].fillna(0)

    # Vasopressor score takes precedence where it exceeds MAP score
    result["sofa_cardiovascular"] = (
        result[["map_score", "worst_vaso_score"]].max(axis=1).astype(int)
    )

    return result[["stay_id", "worst_map", "worst_vaso_score", "sofa_cardiovascular"]]


def compute_sofa_neurological(vitals_window):
    """Derive SOFA neurological score from GCS components.

    Sums GCS Eye (220739), Motor (223901), and Verbal (223900) per
    observation to derive total GCS, then takes the worst (lowest)
    total within the 0-6h window.

    Raw GCS components are used rather than pre-calculated scores to
    avoid leakage from APACHE-derived fields. See variable_logic.md.

    Missing GCS data: score imputed as 0 (best case assumption),
    consistent with published MIMIC-IV Sepsis-3 derivation convention.
    """
    gcs_items = {
        220739: "gcs_eye",
        223901: "gcs_motor",
        223900: "gcs_verbal",
    }

    # Pull each GCS component and pivot to one row per stay/charttime
    components = []
    for itemid, label in gcs_items.items():
        component = vitals_window[vitals_window["itemid"] == itemid][
            ["stay_id", "charttime", "valuenum"]
        ].rename(columns={"valuenum": label})
        components.append(component)

    # Merge all three components on stay_id and charttime
    gcs = components[0]
    for component in components[1:]:
        gcs = gcs.merge(component, on=["stay_id", "charttime"], how="outer")

    # Sum components - only where all three are present
    # Partial GCS (e.g. intubated patients missing verbal) left as null
    gcs["gcs_total"] = gcs["gcs_eye"] + gcs["gcs_motor"] + gcs["gcs_verbal"]

    # Worst (lowest) GCS per stay
    worst = (
        gcs.groupby("stay_id")["gcs_total"]
        .min()
        .reset_index()
        .rename(columns={"gcs_total": "worst_gcs"})
    )

    conditions = [
        worst["worst_gcs"] == 15,
        worst["worst_gcs"] >= 13,
        worst["worst_gcs"] >= 10,
        worst["worst_gcs"] >= 6,
        worst["worst_gcs"] < 6,
    ]

    scores = [0, 1, 2, 3, 4]
    # Intubated patients cannot produce a verbal response and are typically scored GCS Verbal = 1 (no response) or recorded as null
    # Where any component is null the total GCS will be null and the score will default to 0
    # This is the conservative assumption - a patient with missing GCS is assumed neurologically intact

    worst["sofa_neurological"] = np.select(conditions, scores, default=0)

    return worst[["stay_id", "worst_gcs", "sofa_neurological"]]


def compute_sofa_renal(labs_window, urine_output_window, comorbidities_df=None):
    """Derive SOFA renal score from creatinine and urine output.

    Takes the maximum of creatinine-based and urine-output-based scores
    since either alone can indicate renal organ dysfunction.

    CKD confound: absolute creatinine is used without CKD adjustment in
    this implementation. A patient with CKD stage 4 may have a baseline
    creatinine of 3.0 mg/dL unrelated to acute sepsis-related dysfunction.
    This is a known limitation documented in docs/modelling_decisions.md.
    ckd_stage from comorbidities_clean.parquet is available for a future
    sensitivity analysis using creatinine change from estimated baseline.

    Missing creatinine and missing urine output: component score imputed
    as 0 (best case assumption), consistent with published MIMIC-IV
    Sepsis-3 derivation convention.
    """
    # comorbidities_df accepted for future CKD-adjusted creatinine baseline
    # sensitivity analysis. Currently unused - absolute creatinine applied.
    # See docs/modelling_decisions.md.

    # --- Creatinine component ---
    creatinine = labs_window[labs_window["itemid"] == 50912][
        ["stay_id", "valuenum"]
    ].rename(columns={"valuenum": "creatinine"})

    worst_creatinine = (
        creatinine.groupby("stay_id")["creatinine"]
        .max()
        .reset_index()
        .rename(columns={"creatinine": "worst_creatinine"})
    )

    creatinine_conditions = [
        worst_creatinine["worst_creatinine"] >= 5.0,
        worst_creatinine["worst_creatinine"] >= 3.5,
        worst_creatinine["worst_creatinine"] >= 2.0,
        worst_creatinine["worst_creatinine"] >= 1.2,
        worst_creatinine["worst_creatinine"] < 1.2,
    ]

    worst_creatinine["creatinine_score"] = np.select(
        creatinine_conditions, [4, 3, 2, 1, 0], default=0
    )

    # --- Urine output component ---
    # Sum all urine output within the 6-hour window per stay
    # Scale to mL/day equivalent for SOFA thresholds (x4 since window is 6h)
    urine = (
        urine_output_window[["stay_id", "value"]]
        .groupby("stay_id")["value"]
        .sum()
        .reset_index()
        .rename(columns={"value": "urine_output_6h"})
    )

    # Scale 6-hour total to 24-hour equivalent
    urine["urine_output_24h_equiv"] = urine["urine_output_6h"] * 4

    urine_conditions = [
        urine["urine_output_24h_equiv"] < 200,
        urine["urine_output_24h_equiv"] < 500,
        urine["urine_output_24h_equiv"] >= 500,
    ]

    urine["urine_score"] = np.select(urine_conditions, [4, 3, 0], default=0)

    # --- Combine ---
    result = worst_creatinine.merge(urine, on="stay_id", how="left")
    result["urine_score"] = result["urine_score"].fillna(0)

    # Take maximum of both component scores
    result["sofa_renal"] = (
        result[["creatinine_score", "urine_score"]].max(axis=1).astype(int)
    )

    return result[
        [
            "stay_id",
            "worst_creatinine",
            "urine_output_6h",
            "creatinine_score",
            "urine_score",
            "sofa_renal",
        ]
    ]


def compute_sofa_total(
    cohort_df,
    vitals_window,
    labs_window,
    vasopressors_window,
    urine_output_window,
    ventilation_window,
    comorbidities_df=None,
):
    """Derive total SOFA score from all six organ system components.

    Calls each component function and merges results onto the full cohort.
    Stays with no data for a given component receive a score of 0 for that
    component (best case / least organ dysfunction assumption). This is the
    convention used in published MIMIC-IV Sepsis-3 derivation research and
    is conservative in the correct direction: it underestimates SOFA in
    patients with missing data rather than overestimating.

    Missing component handling per component:
        Respiratory:    0 if no SpO2 or FiO2 data
        Coagulation:    0 if no platelet count
        Hepatic:        0 if no bilirubin
        Cardiovascular: 0 if no MAP and no vasopressors
        Neurological:   0 if no GCS components
        Renal:          0 if no creatinine and no urine output

    Returns one row per stay_id with columns for each component score,
    the worst values used, and the total SOFA score.
    """
    respiratory = compute_sofa_respiratory(vitals_window, ventilation_window)
    coagulation = compute_sofa_coagulation(labs_window)
    hepatic = compute_sofa_hepatic(labs_window)
    cardiovascular = compute_sofa_cardiovascular(vitals_window, vasopressors_window)
    neurological = compute_sofa_neurological(vitals_window)
    renal = compute_sofa_renal(labs_window, urine_output_window, comorbidities_df)

    # Start from full cohort so every stay is represented even with no data
    result = cohort_df[["stay_id"]].copy()

    # Merge each component - left join so stays with no component data
    # remain in the result with null scores
    for component_df, score_col in [
        (respiratory, "sofa_respiratory"),
        (coagulation, "sofa_coagulation"),
        (hepatic, "sofa_hepatic"),
        (cardiovascular, "sofa_cardiovascular"),
        (neurological, "sofa_neurological"),
        (renal, "sofa_renal"),
    ]:
        result = result.merge(component_df, on="stay_id", how="left")

    # Fill null component scores with 0 (best case assumption)
    score_cols = [
        "sofa_respiratory",
        "sofa_coagulation",
        "sofa_hepatic",
        "sofa_cardiovascular",
        "sofa_neurological",
        "sofa_renal",
    ]

    null_counts = result[score_cols].isna().sum()
    for col, count in null_counts.items():
        if count > 0:
            logging.info(f"SOFA {col}: {count} stays with no data, imputed as 0")

    result[score_cols] = result[score_cols].fillna(0).astype(int)

    # Total SOFA is the sum of all six components (range 0-24)
    result["sofa_total"] = result[score_cols].sum(axis=1)

    logging.info(
        f"SOFA scores derived: {len(result)} stays, "
        f"mean total SOFA {result['sofa_total'].mean():.2f}, "
        f"max {result['sofa_total'].max()}"
    )

    return result
