"""Feature engineering with leakage prevention."""

from pathlib import Path
import pandas as pd
import logging

from pipeline.sofa import compute_sofa

EXTRACT_DIR = Path("data/versioned")


# Step 1 - Load cleaned extracts from data/versioned/ and return as a dictionary of DataFrames.
def load_cleaned_extracts() -> dict[str, pd.DataFrame]:
    """Load all cleaned parquet extracts from data/versioned/.

    Raises FileNotFoundError if any expected extract is missing rather than
    failing silently downstream.
    """
    expected_extracts = {
        "comorbidity": "comorbidity_clean.parquet",
        "cohort": "cohort_clean.parquet",
        "vitals": "vitals_clean.parquet",
        "labs": "labs_clean.parquet",
        "infection_components": "infection_components_clean.parquet",
        "medications": "medications_clean.parquet",
        "vasopressors": "vasopressors_clean.parquet",
        "urine_output": "urine_output_clean.parquet",
        "ventilation": "ventilation_events_clean.parquet",
        "diagnosis": "diagnosis_clean.parquet",
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


# Step 2 - enforce observation window
"""This is the primary leakage prevention step. All subsequent feature computation must draw exclusively from this filtered data.
For each time-series table (vitals, labs, vasopressors, urine output, ventilation, medications, infection components),
 filter observations to the window [intime, intime + 6 hours] using charttime, starttime, or the relevant timestamp column.
The index time is intime from cohort_clean.parquet. This must be joined to each table before filtering. 
No observation after intime + 6 hours may enter any feature.
Document this with explicit assertions or logging that confirms the maximum observation time in each filtered table 
does not exceed the window boundary. This is the engineering detail the brief calls differentiating.

Every filtering step that touches time must be verifiable. 
Add an assertion or logging check after the observation window filter confirming that max(charttime - intime) <= 6 hours for every stay 
in the filtered vitals and labs. If this assertion fails, something upstream is wrong. 
The brief is explicit: leakage prevention is the engineering detail that separates credible clinical ML from toy projects.\Make it visible in the code.

"""


def enforce_observation_window(
    cohort_df: pd.DataFrame,
    df: pd.DataFrame,
    timestamp_col: str,
    join_key: str = "stay_id",
    label: str = "",
) -> pd.DataFrame:
    """Filter an event table to the 6-hour observation window per ICU stay.

    Joins intime from cohort onto the event table via join_key, then excludes
    any observation where timestamp_col falls outside [intime, intime + 6h].

    The inner join implicitly drops orphaned rows from LOS-excluded stays.
    A leakage assertion confirms no observation exceeds the boundary after
    filtering. If this assertion fails something upstream is wrong.

    Args:
        cohort_df: cleaned cohort with stay_id, hadm_id, and intime columns
        df: event table to filter
        timestamp_col: name of the timestamp column to filter on
        join_key: column to join cohort on, stay_id or hadm_id
        label: name used in logging output
    """
    window = pd.Timedelta(hours=6)

    # Join key for hadm_id tables needs intime via stay_id first
    # cohort has both stay_id and hadm_id so either join works
    cohort_keys = cohort_df[["stay_id", "hadm_id", "intime"]].drop_duplicates(
        subset=[join_key]
    )

    df = df.merge(cohort_keys[[join_key, "intime"]], on=join_key, how="inner")

    before = len(df)
    df = df[
        (df[timestamp_col] >= df["intime"])
        & (df[timestamp_col] <= df["intime"] + window)
    ]

    logging.info(
        f"{label} observation window: {before:,} -> {len(df):,} rows "
        f"({before - len(df):,} outside 0-6h window)"
    )

    # Leakage assertion
    max_offset = (df[timestamp_col] - df["intime"]).max()
    assert (
        max_offset <= window
    ), f"LEAKAGE DETECTED in {label}: max offset {max_offset} exceeds 6h boundary"
    logging.info(f"{label} leakage check PASSED: max offset = {max_offset}")

    df = df.drop(columns=["intime"])
    return df


def main():
    cleaned_extracts = load_cleaned_extracts()

    vitals = enforce_observation_window(
        cleaned_extracts["cohort"],
        cleaned_extracts["vitals"],
        "charttime",
        "stay_id",
        "vitals",
    )
    labs = enforce_observation_window(
        cleaned_extracts["cohort"],
        cleaned_extracts["labs"],
        "charttime",
        "hadm_id",
        "labs",
    )
    vasopressors = enforce_observation_window(
        cleaned_extracts["cohort"],
        cleaned_extracts["vasopressors"],
        "starttime",
        "stay_id",
        "vasopressors",
    )
    urine_output = enforce_observation_window(
        cleaned_extracts["cohort"],
        cleaned_extracts["urine_output"],
        "charttime",
        "stay_id",
        "urine_output",
    )
    ventilation = enforce_observation_window(
        cleaned_extracts["cohort"],
        cleaned_extracts["ventilation"],
        "starttime",
        "stay_id",
        "ventilation",
    )
    medications = enforce_observation_window(
        cleaned_extracts["cohort"],
        cleaned_extracts["medications"],
        "charttime",
        "hadm_id",
        "medications",
    )
    infection_components = enforce_observation_window(
        cleaned_extracts["cohort"],
        cleaned_extracts["infection_components"],
        "charttime",
        "hadm_id",
        "infection_components",
    )

    # Step 3 - sofa (Sequential Organ Failure Assessment)
    sofa_scores = compute_sofa(
        cleaned_extracts["comorbidity"],
        cleaned_extracts["cohort"],
        vitals,
        labs,
        vasopressors,
        urine_output,
        ventilation,
    )


if __name__ == "__main__":
    main()
