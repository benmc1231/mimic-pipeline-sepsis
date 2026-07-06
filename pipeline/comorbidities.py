"""
Comorbidity feature derivation for MIMIC-IV sepsis prediction pipeline.

Derives standard comorbidity indices and individual condition flags from ICD
diagnosis codes extracted in extract.py and cleaned in clean.py. Called from
features.py as part of the feature engineering stage.

Comorbidity features serve two purposes in this project:
    1. Direct predictive features: patients with CKD, liver disease, or active
       malignancy have materially different baseline risk and physiological
       profiles.
    2. Confound adjustment: comorbidities affect the interpretation of SOFA
       score components. CKD raises baseline creatinine, liver disease raises
       baseline bilirubin, diabetes affects lactate interpretation.

Outputs (written to data/versioned/):
    comorbidities_clean.parquet  -- one row per hadm_id with all comorbidity
                                    columns attached

Standard scores (Charlson, Elixhauser) are derived using the comorbidipy
library to avoid reimplementing validated ICD-to-comorbidity mappings.

Individual flags use exact code matching (CKD, liver disease,
immunosuppression) or prefix matching (malignancy, diabetes) depending on
code volume. See docs/variable_logic.md for selection rationale per condition.
"""

import logging
from pathlib import Path

import pandas as pd
import polars as pl
import comorbidipy

from pipeline.constants import (
    CKD_ICD_CODES_FLAT,
    CKD_STAGE_MAP,
    DIABETES_ICD_PREFIXES,
    IMMUNOSUPPRESSION_ICD_CODES_FLAT,
    LIVER_ICD_CODES_FLAT,
    LIVER_STAGE_MAP,
    MALIGNANCY_ICD_PREFIXES,
    METASTATIC_ICD_PREFIXES,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

EXTRACT_DIR = Path("data/versioned")


# ---------------------------------------------------------------------------
# STANDARD COMORBIDITY SCORES
# ---------------------------------------------------------------------------


def _derive_comorbidity_scores(diagnosis: pd.DataFrame) -> pd.DataFrame:
    """Derive Charlson and Elixhauser comorbidity scores using comorbidipy.

    Processes ICD-9 and ICD-10 codes separately since comorbidipy requires
    version-specific input. Scores are computed per version then combined by
    taking the maximum per hadm_id, which correctly handles admissions that
    have diagnosis codes under both coding systems.
    """
    if diagnosis.empty:
        return pd.DataFrame(columns=["hadm_id", "charlson_score", "elixhauser_score"])

    diagnosis = diagnosis.copy()
    diagnosis["icd_code"] = diagnosis["icd_code"].astype(str).str.strip()
    diagnosis["icd_version"] = diagnosis["icd_version"].astype(str).str.strip()

    version_map = {
        "9": "icd9",
        "10": "icd10",
        "icd9": "icd9",
        "icd10": "icd10",
        "ICD9": "icd9",
        "ICD10": "icd10",
    }
    diagnosis["icd_version_key"] = (
        diagnosis["icd_version"].map(version_map).fillna("icd10")
    )

    score_frames = {}
    for score_name in ("charlson", "elixhauser"):
        score_chunks = []
        for version in ("icd9", "icd10"):
            subset = diagnosis.loc[
                diagnosis["icd_version_key"] == version,
                ["hadm_id", "icd_code"],
            ]
            if subset.empty:
                continue

            polars_df = pl.DataFrame(
                {
                    "hadm_id": subset["hadm_id"].astype("Int64"),
                    "code": subset["icd_code"],
                }
            )
            score_df = comorbidipy.comorbidity(
                polars_df,
                id_col="hadm_id",
                code_col="code",
                score=comorbidipy.ScoreType(score_name),
                icd=comorbidipy.ICDVersion(version),
            ).to_pandas()
            score_df = score_df.rename(
                columns={"comorbidity_score": f"{score_name}_score"}
            )
            score_chunks.append(score_df[["hadm_id", f"{score_name}_score"]])

        if score_chunks:
            combined = pd.concat(score_chunks, ignore_index=True)
            combined = combined.groupby("hadm_id", as_index=False).max()
        else:
            combined = pd.DataFrame(
                {
                    "hadm_id": diagnosis["hadm_id"].drop_duplicates(),
                    f"{score_name}_score": 0,
                }
            )

        score_frames[score_name] = combined

    output = diagnosis[["hadm_id"]].drop_duplicates().reset_index(drop=True)
    output = output.merge(score_frames["charlson"], on="hadm_id", how="left")
    output = output.merge(score_frames["elixhauser"], on="hadm_id", how="left")
    output["charlson_score"] = output["charlson_score"].fillna(0).astype(int)
    output["elixhauser_score"] = output["elixhauser_score"].fillna(0).astype(int)
    return output


# ---------------------------------------------------------------------------
# INDIVIDUAL COMORBIDITY FLAGS - SHARED HELPERS
# ---------------------------------------------------------------------------


def _derive_staged_comorbidity(
    diagnosis: pd.DataFrame,
    cohort: pd.DataFrame,
    icd_codes_flat: list,
    stage_map: dict,
    flag_name: str,
    stage_col_name: str,
) -> pd.DataFrame:
    """Generic staged comorbidity flag using exact ICD code matching.

    Used for conditions with a manageable exact code list and a clear severity
    hierarchy (CKD, liver disease). Codes present in icd_codes_flat but absent
    from stage_map still set the presence flag True but contribute null to the
    severity column. This avoids fabricating precision for unspecified codes.
    See docs/variable_logic.md for per-condition staging decisions.
    """
    matched_rows = diagnosis[diagnosis["icd_code"].isin(icd_codes_flat)].copy()
    if matched_rows.empty:
        return (
            cohort[["hadm_id"]]
            .drop_duplicates()
            .assign(**{flag_name: False, stage_col_name: pd.NA})
        )

    matched_rows["_stage_numeric"] = matched_rows["icd_code"].map(stage_map)

    stage_by_admission = (
        matched_rows.groupby("hadm_id")["_stage_numeric"]
        .max()
        .reset_index()
        .rename(columns={"_stage_numeric": stage_col_name})
    )

    has_condition = matched_rows[["hadm_id"]].drop_duplicates().copy()
    has_condition[flag_name] = True

    condition_features = has_condition.merge(
        stage_by_admission, on="hadm_id", how="left"
    )

    cohort_flagged = (
        cohort[["hadm_id"]]
        .drop_duplicates()
        .merge(condition_features, on="hadm_id", how="left")
    )
    cohort_flagged[flag_name] = cohort_flagged[flag_name].fillna(False)
    return cohort_flagged


def _derive_staged_comorbidity_prefix(
    diagnosis: pd.DataFrame,
    cohort: pd.DataFrame,
    prefix_groups: dict,
    flag_name: str,
    stage_col_name: str,
) -> pd.DataFrame:
    """Generic staged comorbidity flag using ICD code prefix matching.

    Used where exhaustive exact code listing is impractical due to code volume
    (malignancy: 1000+ codes). prefix_groups is a dict of {stage_value: prefix_tuple}.
    The highest matched stage value per hadm_id is retained.
    """
    diagnosis = diagnosis.copy()
    diagnosis["_stage_numeric"] = pd.NA

    for stage_value, prefixes in prefix_groups.items():
        prefix_mask = diagnosis["icd_code"].str.startswith(tuple(prefixes))
        diagnosis.loc[prefix_mask, "_stage_numeric"] = stage_value

    matched_rows = diagnosis[diagnosis["_stage_numeric"].notna()].copy()
    if matched_rows.empty:
        return (
            cohort[["hadm_id"]]
            .drop_duplicates()
            .assign(**{flag_name: False, stage_col_name: pd.NA})
        )

    stage_by_admission = (
        matched_rows.groupby("hadm_id")["_stage_numeric"]
        .max()
        .reset_index()
        .rename(columns={"_stage_numeric": stage_col_name})
    )

    has_condition = matched_rows[["hadm_id"]].drop_duplicates().copy()
    has_condition[flag_name] = True

    condition_features = has_condition.merge(
        stage_by_admission, on="hadm_id", how="left"
    )

    cohort_flagged = (
        cohort[["hadm_id"]]
        .drop_duplicates()
        .merge(condition_features, on="hadm_id", how="left")
    )
    cohort_flagged[flag_name] = cohort_flagged[flag_name].fillna(False)
    return cohort_flagged


def _derive_binary_comorbidity_flag(
    diagnosis: pd.DataFrame,
    cohort: pd.DataFrame,
    icd_codes_flat: list,
    flag_name: str,
) -> pd.DataFrame:
    """Generic boolean comorbidity flag using exact ICD code matching.

    Used for conditions with no meaningful severity hierarchy (immunosuppression).
    Returns one row per hadm_id with a single boolean column.
    """
    matched_rows = diagnosis[diagnosis["icd_code"].isin(icd_codes_flat)]
    has_condition = matched_rows[["hadm_id"]].drop_duplicates().copy()
    has_condition[flag_name] = True

    cohort_flagged = (
        cohort[["hadm_id"]]
        .drop_duplicates()
        .merge(has_condition, on="hadm_id", how="left")
    )
    cohort_flagged[flag_name] = cohort_flagged[flag_name].fillna(False)
    return cohort_flagged


def _derive_binary_comorbidity_flag_prefix(
    diagnosis: pd.DataFrame,
    cohort: pd.DataFrame,
    prefixes: tuple,
    flag_name: str,
) -> pd.DataFrame:
    """Generic boolean comorbidity flag using ICD code prefix matching.

    Used where exhaustive exact code listing is impractical (diabetes: 600+
    codes) and no severity staging is required.
    """
    matched_rows = diagnosis[diagnosis["icd_code"].str.startswith(tuple(prefixes))]
    has_condition = matched_rows[["hadm_id"]].drop_duplicates().copy()
    has_condition[flag_name] = True

    cohort_flagged = (
        cohort[["hadm_id"]]
        .drop_duplicates()
        .merge(has_condition, on="hadm_id", how="left")
    )
    cohort_flagged[flag_name] = cohort_flagged[flag_name].fillna(False)
    return cohort_flagged


# ---------------------------------------------------------------------------
# ORCHESTRATOR
# ---------------------------------------------------------------------------


def derive_comorbidity_features(
    diagnosis: pd.DataFrame,
    cohort: pd.DataFrame,
) -> pd.DataFrame:
    """Derive all comorbidity features from ICD diagnosis codes.

    Combines standard scores (Charlson, Elixhauser) with individual condition
    flags into a single table with one row per hadm_id.

    Conditions and matching approach:
        CKD:               exact match, staged 1-5/ESRD
        Liver disease:     exact match, staged mild/severe
        Malignancy:        prefix match, staged primary/metastatic
        Diabetes:          prefix match, boolean only
        Immunosuppression: exact match, boolean only

    See docs/variable_logic.md for clinical rationale and code selection
    decisions per condition.
    """
    # Strip whitespace from ICD codes once before all downstream operations
    diagnosis = diagnosis.copy()
    diagnosis["icd_code"] = diagnosis["icd_code"].astype(str).str.strip()

    scores = _derive_comorbidity_scores(diagnosis)
    logging.info("Charlson and Elixhauser scores derived")

    ckd_features = _derive_staged_comorbidity(
        diagnosis, cohort, CKD_ICD_CODES_FLAT, CKD_STAGE_MAP, "has_ckd", "ckd_stage"
    )
    liver_features = _derive_staged_comorbidity(
        diagnosis,
        cohort,
        LIVER_ICD_CODES_FLAT,
        LIVER_STAGE_MAP,
        "has_liver_disease",
        "liver_stage",
    )
    malignancy_features = _derive_staged_comorbidity_prefix(
        diagnosis,
        cohort,
        {1: MALIGNANCY_ICD_PREFIXES, 2: METASTATIC_ICD_PREFIXES},
        "has_malignancy",
        "malignancy_stage",
    )
    diabetes_features = _derive_binary_comorbidity_flag_prefix(
        diagnosis, cohort, DIABETES_ICD_PREFIXES, "has_diabetes"
    )
    immunosuppression_features = _derive_binary_comorbidity_flag(
        diagnosis, cohort, IMMUNOSUPPRESSION_ICD_CODES_FLAT, "has_immunosuppression"
    )

    output = scores.merge(ckd_features, on="hadm_id", how="left")
    output = output.merge(liver_features, on="hadm_id", how="left")
    output = output.merge(malignancy_features, on="hadm_id", how="left")
    output = output.merge(diabetes_features, on="hadm_id", how="left")
    output = output.merge(immunosuppression_features, on="hadm_id", how="left")

    for col in [
        "has_ckd",
        "has_liver_disease",
        "has_malignancy",
        "has_diabetes",
        "has_immunosuppression",
    ]:
        output[col] = output[col].fillna(False)

    logging.info(
        f"Comorbidity features derived: {len(output)} admissions, "
        f"{output['has_ckd'].sum()} CKD, "
        f"{output['has_liver_disease'].sum()} liver disease, "
        f"{output['has_malignancy'].sum()} malignancy, "
        f"{output['has_diabetes'].sum()} diabetes, "
        f"{output['has_immunosuppression'].sum()} immunosuppression"
    )

    return output


def main():
    """Derive comorbidity features from cleaned diagnosis data.

    Called independently or from features.py. Reads diagnosis_clean.parquet
    and cohort_clean.parquet and writes comorbidities_clean.parquet.
    """
    diagnosis = pd.read_parquet(EXTRACT_DIR / "diagnosis_clean.parquet")
    cohort = pd.read_parquet(EXTRACT_DIR / "cohort_clean.parquet")

    comorbidities = derive_comorbidity_features(diagnosis, cohort)

    output_path = EXTRACT_DIR / "comorbidities_clean.parquet"
    comorbidities.to_parquet(output_path, index=False)
    logging.info(f"Comorbidity features written: {output_path}")


if __name__ == "__main__":
    main()
