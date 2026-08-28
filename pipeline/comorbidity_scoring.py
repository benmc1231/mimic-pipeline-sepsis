"""
Manual Charlson and Elixhauser comorbidity scoring from ICD-10 codes.

Implements published validated mappings rather than relying on external
libraries. Mappings are based on:
    Charlson: Quan et al. (2005) - ICD-10 adaptation of Charlson index
    Elixhauser: van Walraven et al. (2009) - ICD-10 Elixhauser adaptation

ICD-9 codes are present in MIMIC-IV but excluded here as the icd package
only provides ICD-10 mappings and manual ICD-9 mapping is out of scope.
ICD-10 codes cover the majority of admissions in MIMIC-IV v3.1. This
limitation is documented in docs/variable_logic.md.

Both indices use prefix matching rather than exact codes since ICD-10
subcodes share the clinical meaning of their parent code.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CHARLSON COMORBIDITY INDEX
# Quan et al. (2005) ICD-10 adaptation
# Each condition maps to (weight, [icd10_prefixes])
# ---------------------------------------------------------------------------

CHARLSON_ICD10 = {
    "myocardial_infarction": (1, ["I21", "I22", "I252"]),
    "congestive_heart_failure": (
        1,
        [
            "I099",
            "I110",
            "I130",
            "I132",
            "I255",
            "I420",
            "I425",
            "I426",
            "I427",
            "I428",
            "I429",
            "I43",
            "I50",
            "P290",
        ],
    ),
    "peripheral_vascular": (
        1,
        [
            "I70",
            "I71",
            "I731",
            "I738",
            "I739",
            "I771",
            "I790",
            "I792",
            "K551",
            "K558",
            "K559",
            "Z958",
            "Z959",
        ],
    ),
    "cerebrovascular": (
        1,
        [
            "G45",
            "G46",
            "I60",
            "I61",
            "I62",
            "I63",
            "I64",
            "I65",
            "I66",
            "I67",
            "I68",
            "I69",
            "H340",
        ],
    ),
    "dementia": (1, ["F00", "F01", "F02", "F03", "F051", "G30", "G311"]),
    "chronic_pulmonary": (
        1,
        [
            "J40",
            "J41",
            "J42",
            "J43",
            "J44",
            "J45",
            "J46",
            "J47",
            "J60",
            "J61",
            "J62",
            "J63",
            "J64",
            "J65",
            "J66",
            "J67",
            "J684",
            "J701",
            "J703",
        ],
    ),
    "rheumatic_disease": (
        1,
        ["M05", "M06", "M315", "M32", "M33", "M34", "M351", "M353", "M360"],
    ),
    "peptic_ulcer": (1, ["K25", "K26", "K27", "K28"]),
    "mild_liver": (
        1,
        [
            "B18",
            "K700",
            "K701",
            "K702",
            "K703",
            "K709",
            "K713",
            "K714",
            "K715",
            "K717",
            "K73",
            "K74",
            "K760",
            "K762",
            "K763",
            "K764",
            "K768",
            "K769",
            "Z944",
        ],
    ),
    "diabetes_uncomplicated": (
        1,
        [
            "E100",
            "E101",
            "E106",
            "E108",
            "E109",
            "E110",
            "E111",
            "E116",
            "E118",
            "E119",
            "E120",
            "E121",
            "E126",
            "E128",
            "E129",
            "E130",
            "E131",
            "E136",
            "E138",
            "E139",
            "E140",
            "E141",
            "E146",
            "E148",
            "E149",
        ],
    ),
    "diabetes_complicated": (
        2,
        [
            "E102",
            "E103",
            "E104",
            "E105",
            "E107",
            "E112",
            "E113",
            "E114",
            "E115",
            "E117",
            "E122",
            "E123",
            "E124",
            "E125",
            "E127",
            "E132",
            "E133",
            "E134",
            "E135",
            "E137",
            "E142",
            "E143",
            "E144",
            "E145",
            "E147",
        ],
    ),
    "hemiplegia_paraplegia": (
        2,
        [
            "G041",
            "G114",
            "G801",
            "G802",
            "G81",
            "G82",
            "G830",
            "G831",
            "G832",
            "G833",
            "G834",
            "G839",
        ],
    ),
    "renal_disease": (
        2,
        [
            "I120",
            "I131",
            "N032",
            "N033",
            "N034",
            "N035",
            "N036",
            "N037",
            "N052",
            "N053",
            "N054",
            "N055",
            "N056",
            "N057",
            "N18",
            "N19",
            "N250",
            "Z490",
            "Z491",
            "Z492",
            "Z940",
            "Z992",
        ],
    ),
    "malignancy": (
        2,
        [
            "C0",
            "C1",
            "C2",
            "C3",
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
            "C6",
            "C70",
            "C71",
            "C72",
            "C73",
            "C74",
            "C75",
            "C76",
            "C81",
            "C82",
            "C83",
            "C84",
            "C85",
            "C86",
            "C87",
            "C88",
            "C89",
            "C90",
            "C91",
            "C92",
            "C93",
            "C94",
            "C95",
            "C96",
            "C97",
        ],
    ),
    "severe_liver": (
        3,
        [
            "I850",
            "I859",
            "I864",
            "I982",
            "K704",
            "K711",
            "K721",
            "K729",
            "K765",
            "K766",
            "K767",
        ],
    ),
    "metastatic": (6, ["C77", "C78", "C79", "C80"]),
    "aids": (6, ["B20", "B21", "B22", "B24"]),
}


# ---------------------------------------------------------------------------
# ELIXHAUSER COMORBIDITY INDEX
# van Walraven et al. (2009) ICD-10 adaptation with published weights
# ---------------------------------------------------------------------------

ELIXHAUSER_ICD10 = {
    "congestive_heart_failure": (
        7,
        [
            "I099",
            "I110",
            "I130",
            "I132",
            "I255",
            "I420",
            "I425",
            "I426",
            "I427",
            "I428",
            "I429",
            "I43",
            "I50",
            "P290",
        ],
    ),
    "cardiac_arrhythmia": (
        5,
        [
            "I441",
            "I442",
            "I443",
            "I456",
            "I459",
            "I47",
            "I48",
            "I49",
            "R000",
            "R001",
            "R008",
            "T821",
            "Z450",
            "Z950",
        ],
    ),
    "valvular_disease": (
        -1,
        [
            "A520",
            "I05",
            "I06",
            "I07",
            "I08",
            "I091",
            "I098",
            "I34",
            "I35",
            "I36",
            "I37",
            "I38",
            "I39",
            "Q230",
            "Q231",
            "Q232",
            "Q233",
            "Z952",
            "Z953",
            "Z954",
        ],
    ),
    "pulmonary_circulation": (4, ["I26", "I27", "I280", "I288", "I289"]),
    "peripheral_vascular": (
        2,
        [
            "I70",
            "I71",
            "I731",
            "I738",
            "I739",
            "I771",
            "I790",
            "I792",
            "K551",
            "K558",
            "K559",
            "Z958",
            "Z959",
        ],
    ),
    "hypertension_uncomplicated": (0, ["I10"]),
    "hypertension_complicated": (0, ["I11", "I12", "I13", "I15"]),
    "paralysis": (
        7,
        [
            "G041",
            "G114",
            "G801",
            "G802",
            "G81",
            "G82",
            "G830",
            "G831",
            "G832",
            "G833",
            "G834",
            "G839",
        ],
    ),
    "other_neurological": (
        6,
        [
            "G10",
            "G11",
            "G12",
            "G13",
            "G20",
            "G21",
            "G22",
            "G254",
            "G255",
            "G312",
            "G318",
            "G319",
            "G32",
            "G35",
            "G36",
            "G37",
            "G40",
            "G41",
            "G931",
            "G934",
            "R470",
            "R56",
        ],
    ),
    "chronic_pulmonary": (
        3,
        [
            "J40",
            "J41",
            "J42",
            "J43",
            "J44",
            "J45",
            "J46",
            "J47",
            "J60",
            "J61",
            "J62",
            "J63",
            "J64",
            "J65",
            "J66",
            "J67",
            "J684",
            "J701",
            "J703",
        ],
    ),
    "diabetes_uncomplicated": (
        0,
        [
            "E100",
            "E101",
            "E106",
            "E108",
            "E109",
            "E110",
            "E111",
            "E116",
            "E118",
            "E119",
            "E120",
            "E121",
            "E126",
            "E128",
            "E129",
            "E130",
            "E131",
            "E136",
            "E138",
            "E139",
            "E140",
            "E141",
            "E146",
            "E148",
            "E149",
        ],
    ),
    "diabetes_complicated": (
        0,
        [
            "E102",
            "E103",
            "E104",
            "E105",
            "E107",
            "E112",
            "E113",
            "E114",
            "E115",
            "E117",
            "E122",
            "E123",
            "E124",
            "E125",
            "E127",
            "E132",
            "E133",
            "E134",
            "E135",
            "E137",
            "E142",
            "E143",
            "E144",
            "E145",
            "E147",
        ],
    ),
    "hypothyroidism": (0, ["E00", "E01", "E02", "E03", "E890"]),
    "renal_failure": (
        5,
        ["I120", "I131", "N18", "N19", "N250", "Z490", "Z491", "Z492", "Z940", "Z992"],
    ),
    "liver_disease": (
        11,
        [
            "B18",
            "I85",
            "I864",
            "I982",
            "K70",
            "K711",
            "K713",
            "K714",
            "K715",
            "K717",
            "K72",
            "K73",
            "K74",
            "K760",
            "K762",
            "K763",
            "K764",
            "K765",
            "K766",
            "K767",
            "K768",
            "K769",
            "Z944",
        ],
    ),
    "peptic_ulcer": (0, ["K25", "K26", "K27", "K28"]),
    "aids": (0, ["B20", "B21", "B22", "B24"]),
    "lymphoma": (
        9,
        [
            "C81",
            "C82",
            "C83",
            "C84",
            "C85",
            "C86",
            "C87",
            "C88",
            "C89",
            "C90",
            "C91",
            "C92",
            "C93",
            "C94",
            "C95",
            "C96",
            "C97",
        ],
    ),
    "metastatic_cancer": (12, ["C77", "C78", "C79", "C80"]),
    "solid_tumor": (
        4,
        [
            "C0",
            "C1",
            "C2",
            "C3",
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
            "C6",
            "C70",
            "C71",
            "C72",
            "C73",
            "C74",
            "C75",
            "C76",
        ],
    ),
    "rheumatoid_arthritis": (
        0,
        [
            "L940",
            "L941",
            "L943",
            "M05",
            "M06",
            "M08",
            "M120",
            "M123",
            "M30",
            "M310",
            "M311",
            "M312",
            "M313",
            "M32",
            "M33",
            "M34",
            "M35",
            "M45",
            "M461",
            "M468",
            "M469",
        ],
    ),
    "coagulopathy": (
        3,
        ["D65", "D66", "D67", "D68", "D691", "D693", "D694", "D695", "D696"],
    ),
    "obesity": (-4, ["E66"]),
    "weight_loss": (
        6,
        ["E40", "E41", "E42", "E43", "E44", "E45", "E46", "R634", "R64"],
    ),
    "fluid_electrolyte": (5, ["E222", "E86", "E87"]),
    "blood_loss_anaemia": (-2, ["D500"]),
    "deficiency_anaemia": (-2, ["D508", "D509", "D51", "D52", "D53"]),
    "alcohol_abuse": (
        0,
        [
            "F10",
            "E52",
            "G621",
            "I426",
            "K292",
            "K700",
            "K703",
            "K709",
            "T51",
            "Z502",
            "Z714",
            "Z721",
        ],
    ),
    "drug_abuse": (
        -7,
        ["F11", "F12", "F13", "F14", "F15", "F16", "F18", "F19", "Z715", "Z722"],
    ),
    "psychoses": (
        0,
        ["F20", "F22", "F23", "F24", "F25", "F28", "F29", "F302", "F312", "F315"],
    ),
    "depression": (
        -3,
        ["F204", "F313", "F314", "F315", "F32", "F33", "F341", "F412", "F432"],
    ),
}


def _score_from_mapping(
    diagnosis_icd10: pd.DataFrame,
    mapping: dict,
    id_col: str = "hadm_id",
    code_col: str = "icd_code",
) -> pd.DataFrame:
    """Compute a comorbidity score from a published ICD-10 prefix mapping.

    For each condition in the mapping, checks whether any diagnosis code
    for each admission starts with one of the condition's prefixes. The
    condition weight is then applied if the condition is present.

    Args:
        diagnosis_icd10: DataFrame of ICD-10 diagnosis codes (long format)
        mapping: dict of {condition: (weight, [prefixes])}
        id_col: admission identifier column
        code_col: ICD code column

    Returns:
        DataFrame with id_col and score column
    """
    ids = diagnosis_icd10[id_col].unique()
    scores = pd.Series(0.0, index=ids, name="score")

    for condition, (weight, prefixes) in mapping.items():
        # Find admissions with any code matching any prefix
        has_condition = diagnosis_icd10[
            diagnosis_icd10[code_col].str.startswith(tuple(prefixes), na=False)
        ][id_col].unique()
        scores.loc[scores.index.isin(has_condition)] += weight

    return scores.reset_index().rename(columns={"index": id_col})


def compute_charlson_score(
    diagnosis_df: pd.DataFrame,
    id_col: str = "hadm_id",
    code_col: str = "icd_code",
    version_col: str = "icd_version",
) -> pd.DataFrame:
    """Compute Charlson Comorbidity Index from ICD-10 diagnosis codes.

    Uses Quan et al. (2005) ICD-10 adaptation. ICD-9 codes are excluded
    as the mapping covers ICD-10 only. This affects admissions coded
    entirely under ICD-9 (earlier admissions in MIMIC-IV). Documented
    as a known limitation in docs/variable_logic.md.

    Returns DataFrame with hadm_id and charlson_score.
    """
    icd10 = diagnosis_df[
        diagnosis_df[version_col].astype(str).str.strip() == "10"
    ].copy()
    icd10[code_col] = icd10[code_col].astype(str).str.strip()

    if icd10.empty:
        return pd.DataFrame({id_col: [], "charlson_score": []})

    result = _score_from_mapping(icd10, CHARLSON_ICD10, id_col, code_col)
    result = result.rename(columns={"score": "charlson_score"})
    result["charlson_score"] = result["charlson_score"].clip(lower=0).astype(int)
    return result


def compute_elixhauser_score(
    diagnosis_df: pd.DataFrame,
    id_col: str = "hadm_id",
    code_col: str = "icd_code",
    version_col: str = "icd_version",
) -> pd.DataFrame:
    """Compute Elixhauser comorbidity score from ICD-10 diagnosis codes.

    Uses van Walraven et al. (2009) ICD-10 adaptation with published
    weights. Negative weights are retained as published -- conditions
    such as obesity and depression are associated with lower in-hospital
    mortality in some populations, reflected by negative weights.

    Returns DataFrame with hadm_id and elixhauser_score.
    """
    icd10 = diagnosis_df[
        diagnosis_df[version_col].astype(str).str.strip() == "10"
    ].copy()
    icd10[code_col] = icd10[code_col].astype(str).str.strip()

    if icd10.empty:
        return pd.DataFrame({id_col: [], "elixhauser_score": []})

    result = _score_from_mapping(icd10, ELIXHAUSER_ICD10, id_col, code_col)
    result = result.rename(columns={"score": "elixhauser_score"})
    result["elixhauser_score"] = result["elixhauser_score"].astype(int)
    return result
