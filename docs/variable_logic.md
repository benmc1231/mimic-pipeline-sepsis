# Variable Logic and Cohort Definition

This document records the clinical and methodological reasoning behind every
variable selection, cohort exclusion, and data quality decision in the pipeline.
It is the authoritative reference for decisions made in `constants.py`,
`clean.py`, `comorbidities.py`, and `features.py`.

This mirrors the source-to-target mapping and data governance documentation
produced for the Victorian Department of Health Common Data Layer integration,
adapted to a clinical research context.

---

## Cohort Definition

### Prediction Task

Among adult ICU admissions not septic at the time of ICU admission, identify
those at elevated risk of developing sepsis between hours 6 and 24 of their
stay, using only data available in the first 6 hours.

### Unit of Analysis

Each ICU stay (`stay_id`) is treated as an independent analytical unit.
Restricting to a patient's first recorded stay would not reliably identify
their true first-ever ICU admission since MIMIC-IV captures admissions within
a defined period rather than a complete longitudinal record. Sepsis onset risk
is a property of the current admission, not ICU history. Excluding subsequent
stays would introduce selection bias without defensible clinical justification.

Non-independence across stays for the same patient is addressed by performing
train/test splits at the patient level (`subject_id`) rather than the stay
level. This ensures a patient's stays do not appear in both train and test sets.

### Temporal Framing

| Component | Definition |
|-----------|------------|
| Index time | ICU admission (`intime`, hour 0) |
| Observation window | Hours 0 to 6: all features derived exclusively from this window |
| Prediction target | Sepsis onset between hours 6 and 24 post-admission |
| Leakage boundary | No data from after hour 6 enters the feature set under any condition |

### Inclusion Criteria

- Adult ICU admission (age 18 or older at time of admission)
- ICU stay duration 6 hours or more (sufficient observation window)
- Not meeting Sepsis-3 criteria within the first 6 hours of admission

### Exclusion Criteria

**Age under 18**
Age at admission is calculated as `anchor_age + (admittime.year - anchor_year)`
since MIMIC-IV records `anchor_age` at `anchor_year`, not at each individual
admission. MIMIC-IV excludes patients under 18 at source. This filter is
retained for explicit documentation and as a guard against edge cases. Data
profiling confirmed 0 stays removed by this filter.

**ICU stay under 6 hours**
`los < 0.25` days. Stays shorter than the observation window cannot produce a
valid feature set. Data profiling confirmed 1,234 stays removed (1.3% of the
initial cohort of 94,458), leaving 93,224 stays.

Note: event table extracts (vitals, labs, vasopressors, urine output,
ventilation) are produced before this exclusion is applied. Rows from excluded
stays therefore exist in the raw and clean parquets but are filtered out when
`features.py` joins event tables to the cohort on `stay_id`. This is expected
pipeline behaviour and not a data quality issue.

**Sepsis present on admission (deferred)**
Patients meeting Sepsis-3 criteria at or before hour 6 represent a recognition
problem rather than a prediction problem and must be excluded. However, deriving
Sepsis-3 criteria requires SOFA scores from vitals and labs, which are not
available at clean time. This exclusion is therefore applied in `features.py`
via `exclude_sepsis_on_admission()` after SOFA scores are derived.

This creates a known pipeline dependency: `clean.py` produces a preliminary
cohort and `features.py` finalises it. The exclusion is not applied twice.
This design decision is documented inline in both files.

---

## Vital Signs

### Source

`mimiciv_icu.chartevents`, filtered to target itemids from `mimiciv_icu.d_items`.

All vitals are linked to ICU stays via `stay_id`. Observations are extracted
across the full ICU stay at this stage; observation window filtering (hours 0
to 6) is applied in `features.py`, not `clean.py`.

### Itemid Selection

| Variable | Itemid(s) | Clinical role |
|----------|-----------|---------------|
| Heart rate | 220045 | SOFA cardiovascular proxy; tachycardia is early sepsis signal |
| ABP systolic | 220050 | MAP calculation (invasive, preferred) |
| ABP diastolic | 220051 | MAP calculation (invasive, preferred) |
| ABP mean | 220052 | SOFA cardiovascular component directly |
| NIBP systolic | 220179 | MAP fallback where no arterial line |
| NIBP diastolic | 220180 | MAP fallback |
| NIBP mean | 220181 | MAP fallback |
| Temperature Celsius | 223762 | Fever/hypothermia, classic sepsis indicators |
| Temperature Fahrenheit | 223761 | Same concept, converted to Celsius in clean.py |
| Respiratory rate | 220210 | SOFA respiratory component; tachypnoea is early sepsis signal |
| SpO2 | 220277 | PF ratio proxy for SOFA respiratory score |
| FiO2 | 223835 | PF ratio proxy for SOFA respiratory score |
| GCS Eye | 220739 | SOFA neurological component (raw, summed in features.py) |
| GCS Motor | 223901 | SOFA neurological component |
| GCS Verbal | 223900 | SOFA neurological component |

Raw GCS components are used rather than pre-calculated APACHE scores to avoid
leakage risk. APACHE scores are computed at a defined time point and may
incorporate post-observation data.

### Range Filters

Values outside clinically defined bounds are set to null rather than dropped.
Missingness is informative downstream and handled by imputation in `features.py`.
Data profiling found 999,157 null values after range filtering across all vital
itemids combined.

| Variable | Min | Max | Unit | Notes |
|----------|-----|-----|------|-------|
| Heart rate | 0 | 300 | bpm | |
| ABP systolic | 0 | 300 | mmHg | |
| ABP diastolic | 0 | 200 | mmHg | |
| ABP/NIBP mean | 0 | 200 | mmHg | |
| NIBP systolic | 0 | 300 | mmHg | |
| NIBP diastolic | 0 | 200 | mmHg | |
| Temperature Celsius | 25 | 45 | C | |
| Temperature Fahrenheit | 77 | 113 | F | Converted to Celsius before downstream use |
| Respiratory rate | 0 | 80 | breaths/min | |
| SpO2 | 50 | 100 | % | |
| FiO2 | 0.21 | 1.0 | fraction | See unit standardisation note below |
| GCS Eye | 1 | 4 | | |
| GCS Motor | 1 | 6 | | |
| GCS Verbal | 1 | 5 | | |

### Unit Standardisation

**FiO2**
FiO2 is recorded inconsistently in some MIMIC-IV installations as either a
fraction (0.21 to 1.0) or a percentage (21 to 100). The cleaning step checks
for percentage values (above 1.0) and divides by 100. Data profiling on this
dataset confirmed all FiO2 values were already stored as fractions (0 percentage
values required conversion). The conversion logic is retained in `clean.py` as
a defensive check since this behaviour is version and configuration dependent.

The missingness report shows FiO2 with 49.84% raw missingness and 99.98%
post-cleaning null rate. This is expected: FiO2 is only charted when a patient
is on supplemental oxygen with a documented FiO2 setting, which in MIMIC-IV
typically means they are on a ventilator or high-flow oxygen. Only 152 non-null
FiO2 values exist across 61 million vital observations. The room air default
(0.21) applied in `features.py` for non-ventilated patients without a recorded
FiO2 addresses this missingness in a clinically justified way.

**Temperature**
Temperature Fahrenheit (itemid 223761) records were present for 2,025,716
observations. These were converted to Celsius using `(F - 32) * 5 / 9` and
relabelled as the Celsius itemid (223762) so both map to a single concept
downstream. The missingness report retains a `temperature_fahrenheit` row
(itemid 223761) because stats are captured before the relabelling step. After
cleaning, no rows with itemid 223761 remain in `vitals_clean.parquet`.

**Blood pressure**
Arterial line (ABP) values are preferred over non-invasive (NIBP) values where
both exist for the same stay and charttime. Arterial readings are continuously
measured and more accurate than oscillometric non-invasive readings, which can
be inaccurate in haemodynamically unstable patients. Where an ABP reading exists
at a given stay/charttime, the corresponding NIBP value is set to null. NIBP
values at charttimes with no ABP reading are retained as the best available
measurement.

The missingness report shows ABP systolic/diastolic/mean at 61% raw missingness,
consistent with the proportion of ICU patients who do not have an arterial line
placed. NIBP missingness is under 0.1%, confirming near-universal non-invasive
BP monitoring.

### Missingness Report Interpretation Notes

**Negative raw missingness values** (heart rate -1.30%, respiratory rate -1.19%,
SpO2 -1.16%, GCS components -0.67% to -0.70%): these result from the missingness
denominator being the cleaned cohort (93,224 stays) while the vitals extract
includes the 1,234 LOS-excluded stays. The extra stay_ids inflate the numerator
(stays with data) above the denominator (cohort stays), producing a small
negative percentage. This is a known artefact of the pipeline ordering where
extraction precedes cohort exclusion. It does not indicate a data quality issue.

**Temperature Celsius raw missingness 89.43%**: temperature is not continuously
monitored in most ICU settings. It is charted at nursing assessment intervals,
typically every 4 hours. The low recording frequency is expected and not a
concern for feature engineering since forward fill imputation within the
observation window handles gaps.

---

## Laboratory Values

### Source

`mimiciv_hosp.labevents`, filtered to target itemids from `mimiciv_hosp.d_labitems`.

Labs are linked to hospital admissions via `hadm_id` rather than `stay_id` since
`labevents` is a hospital-level table with no ICU stay identifier. Observation
window filtering is applied in `features.py`.

The coverage denominator for medications and infection components in the
missingness report is 84,703 unique `hadm_id` values rather than 93,224 stays.
This is because some patients have multiple ICU stays within a single hospital
admission sharing the same `hadm_id`. This is expected behaviour and does not
indicate a data quality issue.

### Itemid Selection

| Variable | Itemid | Fluid | Unit | Clinical role |
|----------|--------|-------|------|---------------|
| Creatinine | 50912 | Blood | mg/dL | SOFA renal component |
| Bilirubin total | 50885 | Blood | mg/dL | SOFA hepatic component |
| Platelet count | 51265 | Blood | K/uL | SOFA coagulation component |
| Lactate | 50813 | Blood | mmol/L | Strong early sepsis marker (not a SOFA component) |
| WBC | 51301 | Blood | K/uL | Infection marker; elevated or severely depressed WBC signals infection |
| Haemoglobin | 51222 | Blood | g/dL | Anaemia context and general severity indicator |

**Creatinine**: itemid 50912 selected (Blood/Chemistry). Excludes urine
creatinine (51082), serum/urine ambiguous (51081), and whole blood gas assay
(52024) which uses a different analytical method.

**Bilirubin total**: itemid 50885 selected (Blood only). Excludes urine, CSF,
pleural, ascites, and neonatal variants.

**Platelet count**: itemid 51265 selected (Blood/Haematology). Excludes platelet
smear (51266, qualitative not quantitative) and platelet clumps (51264, artifact
flag not a measurement).

**Lactate**: itemid 50813 selected (Blood Gas) as primary source with highest
coverage in MIMIC-IV. Lactate is not a SOFA component but is a strong early
sepsis marker included for predictive value.

**Units confirmed via data profiling**: creatinine and bilirubin in mg/dL,
platelets and WBC in K/uL (numerically equivalent to x10^9/L), lactate in
mmol/L, haemoglobin in g/dL. One row each for creatinine, haemoglobin, platelets,
and WBC had no recorded unit; these are negligible in volume and caught by range
filters regardless.

### Range Filters

All values extracted regardless of abnormal flag. Normal lab values are
informative for SOFA scoring and trend analysis. A normal creatinine rules
out renal dysfunction; a creatinine trending upward from normal to mildly
elevated is clinically significant even if neither value triggers an abnormal
flag. Data profiling found 212 additional values set to null by range filters
from 6,450,276 total lab observations (0.003%).

| Variable | Min | Max | Unit |
|----------|-----|-----|------|
| Creatinine | 0.1 | 50 | mg/dL |
| Bilirubin total | 0.1 | 150 | mg/dL |
| Platelet count | 1 | 3000 | K/uL |
| Lactate | 0.1 | 30 | mmol/L |
| WBC | 0.1 | 500 | K/uL |
| Haemoglobin | 0 | 25 | g/dL |

Lactate and WBC minimums raised from 0 to 0.1 since zero is physiologically
impossible for either measure.

### Missingness Report Interpretation

Bilirubin (33.82% raw missingness) and lactate (32.21%) are not routinely
ordered for all patients. Bilirubin is ordered when liver dysfunction is
suspected; lactate when haemodynamic instability or sepsis is suspected.
The missingness is therefore informative rather than random: patients with
bilirubin or lactate measured are likely sicker than those without. This
should be accounted for in feature engineering via explicit missingness flags
alongside imputed values.

---

## Blood Cultures

### Source

`mimiciv_hosp.microbiologyevents`, filtered to blood culture specimen types.

Blood cultures form one half of the Sepsis-3 suspected infection criterion.
The temporal relationship between culture order and antibiotic administration
is evaluated in `features.py`. Data profiling found 299,992 blood culture
records after cleaning.

### Specimen Type Selection

| spec_type_desc | Included | Rationale |
|----------------|----------|-----------|
| BLOOD CULTURE | Yes | Standard blood culture |
| BLOOD CULTURE ( MYCO/F LYTIC BOTTLE) | Yes | Mycobacterial/fungal blood culture, valid infection signal |
| BLOOD CULTURE (POST-MORTEM) | No | Outside prediction window |
| BLOOD CULTURE - NEONATE | No | Cohort is adults 18+, neonatal cultures excluded |
| SEROLOGY/BLOOD | No | Serology testing, not a culture order |
| Blood (Toxo), Blood (EBV), Blood (CMV) | No | Viral/parasitic serology, not bacterial culture |

### Timestamp Handling

Data profiling on the raw microbiologyevents table found 14,785 of 823,483
blood culture records (1.8%) had null `charttime` with a valid `chartdate`.
These were imputed as midnight of `chartdate` and flagged via
`charttime_imputed`. Midnight timestamps are less precise for 6-hour window
logic and should be treated cautiously in `features.py`. Data profiling on
the cleaned file confirmed 0 null charttimes remaining and all
`charttime_imputed` values False, meaning imputation was not required for
this dataset version. The logic is retained as a defensive check.

Infection components coverage is 41.96% of cohort admissions, consistent with
the expected proportion of ICU patients who have blood cultures drawn.

---

## Antibiotic Administration

### Source

`mimiciv_hosp.emar`, filtered to antibiotic medications and confirmed
administration events. Data profiling found 885,555 records after cleaning,
with 0 records removed due to null charttime.

### Medication Pattern Matching

MIMIC-IV medication names are inconsistently capitalised (e.g. Vancomycin,
VANCOMYCIN, vancomycin all appear). Pattern-based ILIKE matching is used in
preference to exact string matching. Patterns used: vancomycin, piperacillin,
meropenem, ceftriaxone, ciprofloxacin, metronidazole, ampicillin, levofloxacin,
azithromycin, clindamycin, daptomycin, linezolid, nafcillin, oxacillin,
doxycycline, tobramycin, penicillin, amoxicillin, tigecycline, ceftolozane,
moxifloxacin.

**Excluded preparations**: ophthalmic, topical, vaginal, heparin locks,
desensitization protocols, graded challenge protocols, placebo entries.
These are non-systemic routes or research/allergy protocols not relevant to
sepsis treatment.

**Coverage**: 35.81% of cohort admissions had at least one antibiotic
administration record. This is the expected order of magnitude for a general
ICU population where not all patients receive antibiotics.

### Event Type Filtering

Only event types confirming the drug reached the patient are included.
Excluded: Not Given, Hold Dose, Flushed (line maintenance), Confirmed (order
confirmation only). Included: Administered, Started (IV infusion initiation),
Restarted, Delayed Administered, Partial Administered, and location variants
of each. Started and Restarted are included because IV antibiotic infusions
in ICU are typically charted as Started at infusion initiation.

---

## Vasopressors

### Source

`mimiciv_icu.inputevents`, filtered to vasopressor itemids from `mimiciv_icu.d_items`.
Data profiling found 850,936 records after cleaning. Coverage was 30.98% of
cohort stays, consistent with the known prevalence of haemodynamic instability
in the general ICU population.

Vasopressor requirement is the SOFA cardiovascular component. Vasopressor
administration (score 3-4) indicates haemodynamic instability consistent with
septic shock.

### Itemid Selection

| Drug | Itemid(s) | Notes |
|------|-----------|-------|
| Norepinephrine | 221906 | First-line vasopressor in septic shock |
| Epinephrine | 221289, 229617 | 229617 is duplicate entry with trailing period in label |
| Dopamine | 221662 | |
| Dobutamine | 221653 | Inotrope, used in cardiogenic shock |
| Vasopressin | 222315 | Second-line adjunct; recorded in units/hour not mg |
| Phenylephrine | 221749, 229632, 229630, 229631 | Multiple itemids reflect different pre-mixed concentrations |
| Milrinone | 221986 | Phosphodiesterase inhibitor inotrope |

Phenylephrine (Intubation) itemid 229789 was excluded. This is a one-off bolus
used during intubation to maintain blood pressure and does not reflect
sustained vasopressor support indicating cardiovascular organ dysfunction.

### Unit Corrections and Data Quality

All corrections confirmed by data profiling of the raw `inputevents` table.
0 records had null starttime or endtime after cleaning.

| Drug | Issue | Rows | Correction |
|------|-------|------|------------|
| Norepinephrine | Recorded as mg/kg/min | 2 | Converted to mcg/kg/min (x1000). mg/kg/min would be a lethal dose, almost certainly a data entry error |
| Phenylephrine | Recorded as mcg/min | 2 | Nulled. Cannot convert to mcg/kg/min without patient weight |
| Vasopressin | Recorded as units/min | 3 | Converted to units/hour (x60). units/hour is the predominant unit |
| Epinephrine (229617) | No rateuom | 234 | Nulled |

236 rows have null rate after cleaning. 2 rows were found with negative rate
values and are removed in `clean.py` as physiologically impossible. These are
separate from the unit correction rows and likely represent data entry errors.

---

## Urine Output

### Source

`mimiciv_icu.outputevents`, filtered to urine output itemids. Data profiling
found 4,130,915 records after cleaning. Coverage was 97.0% of cohort stays,
as expected for an ICU population where urinary catheterisation is near-universal.

Urine output is the SOFA renal component alongside creatinine. Low urine output
(oliguria or anuria) indicates acute kidney injury consistent with
sepsis-related organ dysfunction.

### Itemid Selection

Included: Foley catheter (226559), spontaneous void (226560), condom catheter
(226561), straight catheter (226567), suprapubic catheter (226563), ileoconduit
(226584), left/right ureteral stents (226558, 226557), left/right nephrostomies
(226565, 226564).

Excluded: OR Urine (226627) and PACU Urine (226631) since these fall outside
the ICU stay. GU Irrigant/Urine Out (227489) and Urine and GU Irrigant Out
(226566) excluded since these volumes are contaminated with bladder irrigation
fluid and do not reflect true urine production.

### Data Quality

24 negative value rows removed as physiologically impossible data entry errors.
479 rows flagged via `large_void_flag` where value exceeds 2000 mL per entry.
These are retained but flagged since they may represent accumulated totals
charted as a single entry rather than a genuine single void.

---

## Ventilation

### Source

`mimiciv_icu.procedureevents`, filtered to mechanical ventilation itemids.
Data profiling found 38,717 records after cleaning, covering 35.85% of cohort
stays. 0 records removed due to null timestamps.

Ventilation status is required for correct interpretation of the PF ratio
(SpO2/FiO2) in SOFA respiratory scoring. The threshold for respiratory organ
dysfunction differs between ventilated and non-ventilated patients.

### Itemid Selection

| itemid | label | Notes |
|--------|-------|-------|
| 225792 | Invasive ventilation | Endotracheal intubation with mechanical ventilation |
| 225794 | Non-invasive ventilation | BiPAP/CPAP without intubation |

Ventilator settings (mode, rate, PEEP, tidal volume) are charted in
`chartevents`, not `procedureevents`. These are captured via the vitals
extract. Only the procedural onset/offset events are captured here to
determine whether mechanical ventilation was active during the observation
window.

---

# Variable Logic and Cohort Definition

This document records the clinical and methodological reasoning behind every
variable selection, cohort exclusion, and data quality decision in the pipeline.
It is the authoritative reference for decisions made in `constants.py`,
`clean.py`, `comorbidities.py`, and `features.py`.

This mirrors the source-to-target mapping and data governance documentation
produced for the Victorian Department of Health Common Data Layer integration,
adapted to a clinical research context.

---

## Cohort Definition

### Prediction Task

Among adult ICU admissions not septic at the time of ICU admission, identify
those at elevated risk of developing sepsis between hours 6 and 24 of their
stay, using only data available in the first 6 hours.

### Unit of Analysis

Each ICU stay (`stay_id`) is treated as an independent analytical unit.
Restricting to a patient's first recorded stay would not reliably identify
their true first-ever ICU admission since MIMIC-IV captures admissions within
a defined period rather than a complete longitudinal record. Sepsis onset risk
is a property of the current admission, not ICU history. Excluding subsequent
stays would introduce selection bias without defensible clinical justification.

Non-independence across stays for the same patient is addressed by performing
train/test splits at the patient level (`subject_id`) rather than the stay
level. This ensures a patient's stays do not appear in both train and test sets.

### Temporal Framing

| Component | Definition |
|-----------|------------|
| Index time | ICU admission (`intime`, hour 0) |
| Observation window | Hours 0 to 6: all features derived exclusively from this window |
| Prediction target | Sepsis onset between hours 6 and 24 post-admission |
| Leakage boundary | No data from after hour 6 enters the feature set under any condition |

### Inclusion Criteria

- Adult ICU admission (age 18 or older at time of admission)
- ICU stay duration 6 hours or more (sufficient observation window)
- Not meeting Sepsis-3 criteria within the first 6 hours of admission

### Exclusion Criteria

**Age under 18**
Age at admission is calculated as `anchor_age + (admittime.year - anchor_year)`
since MIMIC-IV records `anchor_age` at `anchor_year`, not at each individual
admission. MIMIC-IV excludes patients under 18 at source. This filter is
retained for explicit documentation and as a guard against edge cases. Data
profiling confirmed 0 stays removed by this filter.

**ICU stay under 6 hours**
`los < 0.25` days. Stays shorter than the observation window cannot produce a
valid feature set. Data profiling confirmed 1,234 stays removed (1.3% of the
initial cohort of 94,458), leaving 93,224 stays.

Note: event table extracts (vitals, labs, vasopressors, urine output,
ventilation) are produced before this exclusion is applied. Rows from excluded
stays therefore exist in the raw and clean parquets but are filtered out when
`features.py` joins event tables to the cohort on `stay_id`. This is expected
pipeline behaviour and not a data quality issue.

**Sepsis present on admission (deferred)**
Patients meeting Sepsis-3 criteria at or before hour 6 represent a recognition
problem rather than a prediction problem and must be excluded. However, deriving
Sepsis-3 criteria requires SOFA scores from vitals and labs, which are not
available at clean time. This exclusion is therefore applied in `features.py`
via `exclude_sepsis_on_admission()` after SOFA scores are derived.

This creates a known pipeline dependency: `clean.py` produces a preliminary
cohort and `features.py` finalises it. The exclusion is not applied twice.
This design decision is documented inline in both files.

---

## Vital Signs

### Source

`mimiciv_icu.chartevents`, filtered to target itemids from `mimiciv_icu.d_items`.

All vitals are linked to ICU stays via `stay_id`. Observations are extracted
across the full ICU stay at this stage; observation window filtering (hours 0
to 6) is applied in `features.py`, not `clean.py`.

### Itemid Selection

| Variable | Itemid(s) | Clinical role |
|----------|-----------|---------------|
| Heart rate | 220045 | SOFA cardiovascular proxy; tachycardia is early sepsis signal |
| ABP systolic | 220050 | MAP calculation (invasive, preferred) |
| ABP diastolic | 220051 | MAP calculation (invasive, preferred) |
| ABP mean | 220052 | SOFA cardiovascular component directly |
| NIBP systolic | 220179 | MAP fallback where no arterial line |
| NIBP diastolic | 220180 | MAP fallback |
| NIBP mean | 220181 | MAP fallback |
| Temperature Celsius | 223762 | Fever/hypothermia, classic sepsis indicators |
| Temperature Fahrenheit | 223761 | Same concept, converted to Celsius in clean.py |
| Respiratory rate | 220210 | SOFA respiratory component; tachypnoea is early sepsis signal |
| SpO2 | 220277 | PF ratio proxy for SOFA respiratory score |
| FiO2 | 223835 | PF ratio proxy for SOFA respiratory score |
| GCS Eye | 220739 | SOFA neurological component (raw, summed in features.py) |
| GCS Motor | 223901 | SOFA neurological component |
| GCS Verbal | 223900 | SOFA neurological component |

Raw GCS components are used rather than pre-calculated APACHE scores to avoid
leakage risk. APACHE scores are computed at a defined time point and may
incorporate post-observation data.

### Range Filters

Values outside clinically defined bounds are set to null rather than dropped.
Missingness is informative downstream and handled by imputation in `features.py`.
Data profiling found 999,157 null values after range filtering across all vital
itemids combined.

| Variable | Min | Max | Unit | Notes |
|----------|-----|-----|------|-------|
| Heart rate | 0 | 300 | bpm | |
| ABP systolic | 0 | 300 | mmHg | |
| ABP diastolic | 0 | 200 | mmHg | |
| ABP/NIBP mean | 0 | 200 | mmHg | |
| NIBP systolic | 0 | 300 | mmHg | |
| NIBP diastolic | 0 | 200 | mmHg | |
| Temperature Celsius | 25 | 45 | C | |
| Temperature Fahrenheit | 77 | 113 | F | Converted to Celsius before downstream use |
| Respiratory rate | 0 | 80 | breaths/min | |
| SpO2 | 50 | 100 | % | |
| FiO2 | 0.21 | 1.0 | fraction | See unit standardisation note below |
| GCS Eye | 1 | 4 | | |
| GCS Motor | 1 | 6 | | |
| GCS Verbal | 1 | 5 | | |

### Unit Standardisation

**FiO2**
FiO2 is recorded inconsistently in some MIMIC-IV installations as either a
fraction (0.21 to 1.0) or a percentage (21 to 100). The cleaning step checks
for percentage values (above 1.0) and divides by 100. Data profiling on this
dataset confirmed all FiO2 values were already stored as fractions (0 percentage
values required conversion). The conversion logic is retained in `clean.py` as
a defensive check since this behaviour is version and configuration dependent.

The missingness report shows FiO2 with 49.84% raw missingness and 99.98%
post-cleaning null rate. This is expected: FiO2 is only charted when a patient
is on supplemental oxygen with a documented FiO2 setting, which in MIMIC-IV
typically means they are on a ventilator or high-flow oxygen. Only 152 non-null
FiO2 values exist across 61 million vital observations. The room air default
(0.21) applied in `features.py` for non-ventilated patients without a recorded
FiO2 addresses this missingness in a clinically justified way.

**Temperature**
Temperature Fahrenheit (itemid 223761) records were present for 2,025,716
observations. These were converted to Celsius using `(F - 32) * 5 / 9` and
relabelled as the Celsius itemid (223762) so both map to a single concept
downstream. The missingness report retains a `temperature_fahrenheit` row
(itemid 223761) because stats are captured before the relabelling step. After
cleaning, no rows with itemid 223761 remain in `vitals_clean.parquet`.

**Blood pressure**
Arterial line (ABP) values are preferred over non-invasive (NIBP) values where
both exist for the same stay and charttime. Arterial readings are continuously
measured and more accurate than oscillometric non-invasive readings, which can
be inaccurate in haemodynamically unstable patients. Where an ABP reading exists
at a given stay/charttime, the corresponding NIBP value is set to null. NIBP
values at charttimes with no ABP reading are retained as the best available
measurement.

The missingness report shows ABP systolic/diastolic/mean at 61% raw missingness,
consistent with the proportion of ICU patients who do not have an arterial line
placed. NIBP missingness is under 0.1%, confirming near-universal non-invasive
BP monitoring.

### Missingness Report Interpretation Notes

**Negative raw missingness values** (heart rate -1.30%, respiratory rate -1.19%,
SpO2 -1.16%, GCS components -0.67% to -0.70%): these result from the missingness
denominator being the cleaned cohort (93,224 stays) while the vitals extract
includes the 1,234 LOS-excluded stays. The extra stay_ids inflate the numerator
(stays with data) above the denominator (cohort stays), producing a small
negative percentage. This is a known artefact of the pipeline ordering where
extraction precedes cohort exclusion. It does not indicate a data quality issue.

**Temperature Celsius raw missingness 89.43%**: temperature is not continuously
monitored in most ICU settings. It is charted at nursing assessment intervals,
typically every 4 hours. The low recording frequency is expected and not a
concern for feature engineering since forward fill imputation within the
observation window handles gaps.

---

## Laboratory Values

### Source

`mimiciv_hosp.labevents`, filtered to target itemids from `mimiciv_hosp.d_labitems`.

Labs are linked to hospital admissions via `hadm_id` rather than `stay_id` since
`labevents` is a hospital-level table with no ICU stay identifier. Observation
window filtering is applied in `features.py`.

The coverage denominator for medications and infection components in the
missingness report is 84,703 unique `hadm_id` values rather than 93,224 stays.
This is because some patients have multiple ICU stays within a single hospital
admission sharing the same `hadm_id`. This is expected behaviour and does not
indicate a data quality issue.

### Itemid Selection

| Variable | Itemid | Fluid | Unit | Clinical role |
|----------|--------|-------|------|---------------|
| Creatinine | 50912 | Blood | mg/dL | SOFA renal component |
| Bilirubin total | 50885 | Blood | mg/dL | SOFA hepatic component |
| Platelet count | 51265 | Blood | K/uL | SOFA coagulation component |
| Lactate | 50813 | Blood | mmol/L | Strong early sepsis marker (not a SOFA component) |
| WBC | 51301 | Blood | K/uL | Infection marker; elevated or severely depressed WBC signals infection |
| Haemoglobin | 51222 | Blood | g/dL | Anaemia context and general severity indicator |

**Creatinine**: itemid 50912 selected (Blood/Chemistry). Excludes urine
creatinine (51082), serum/urine ambiguous (51081), and whole blood gas assay
(52024) which uses a different analytical method.

**Bilirubin total**: itemid 50885 selected (Blood only). Excludes urine, CSF,
pleural, ascites, and neonatal variants.

**Platelet count**: itemid 51265 selected (Blood/Haematology). Excludes platelet
smear (51266, qualitative not quantitative) and platelet clumps (51264, artifact
flag not a measurement).

**Lactate**: itemid 50813 selected (Blood Gas) as primary source with highest
coverage in MIMIC-IV. Lactate is not a SOFA component but is a strong early
sepsis marker included for predictive value.

**Units confirmed via data profiling**: creatinine and bilirubin in mg/dL,
platelets and WBC in K/uL (numerically equivalent to x10^9/L), lactate in
mmol/L, haemoglobin in g/dL. One row each for creatinine, haemoglobin, platelets,
and WBC had no recorded unit; these are negligible in volume and caught by range
filters regardless.

### Range Filters

All values extracted regardless of abnormal flag. Normal lab values are
informative for SOFA scoring and trend analysis. A normal creatinine rules
out renal dysfunction; a creatinine trending upward from normal to mildly
elevated is clinically significant even if neither value triggers an abnormal
flag. Data profiling found 212 additional values set to null by range filters
from 6,450,276 total lab observations (0.003%).

| Variable | Min | Max | Unit |
|----------|-----|-----|------|
| Creatinine | 0.1 | 50 | mg/dL |
| Bilirubin total | 0.1 | 150 | mg/dL |
| Platelet count | 1 | 3000 | K/uL |
| Lactate | 0.1 | 30 | mmol/L |
| WBC | 0.1 | 500 | K/uL |
| Haemoglobin | 0 | 25 | g/dL |

Lactate and WBC minimums raised from 0 to 0.1 since zero is physiologically
impossible for either measure.

### Missingness Report Interpretation

Bilirubin (33.82% raw missingness) and lactate (32.21%) are not routinely
ordered for all patients. Bilirubin is ordered when liver dysfunction is
suspected; lactate when haemodynamic instability or sepsis is suspected.
The missingness is therefore informative rather than random: patients with
bilirubin or lactate measured are likely sicker than those without. This
should be accounted for in feature engineering via explicit missingness flags
alongside imputed values.

---

## Blood Cultures

### Source

`mimiciv_hosp.microbiologyevents`, filtered to blood culture specimen types.

Blood cultures form one half of the Sepsis-3 suspected infection criterion.
The temporal relationship between culture order and antibiotic administration
is evaluated in `features.py`. Data profiling found 299,992 blood culture
records after cleaning.

### Specimen Type Selection

| spec_type_desc | Included | Rationale |
|----------------|----------|-----------|
| BLOOD CULTURE | Yes | Standard blood culture |
| BLOOD CULTURE ( MYCO/F LYTIC BOTTLE) | Yes | Mycobacterial/fungal blood culture, valid infection signal |
| BLOOD CULTURE (POST-MORTEM) | No | Outside prediction window |
| BLOOD CULTURE - NEONATE | No | Cohort is adults 18+, neonatal cultures excluded |
| SEROLOGY/BLOOD | No | Serology testing, not a culture order |
| Blood (Toxo), Blood (EBV), Blood (CMV) | No | Viral/parasitic serology, not bacterial culture |

### Timestamp Handling

Data profiling on the raw microbiologyevents table found 14,785 of 823,483
blood culture records (1.8%) had null `charttime` with a valid `chartdate`.
These were imputed as midnight of `chartdate` and flagged via
`charttime_imputed`. Midnight timestamps are less precise for 6-hour window
logic and should be treated cautiously in `features.py`. Data profiling on
the cleaned file confirmed 0 null charttimes remaining and all
`charttime_imputed` values False, meaning imputation was not required for
this dataset version. The logic is retained as a defensive check.

Infection components coverage is 41.96% of cohort admissions, consistent with
the expected proportion of ICU patients who have blood cultures drawn.

---

## Antibiotic Administration

### Source

`mimiciv_hosp.emar`, filtered to antibiotic medications and confirmed
administration events. Data profiling found 885,555 records after cleaning,
with 0 records removed due to null charttime.

### Medication Pattern Matching

MIMIC-IV medication names are inconsistently capitalised (e.g. Vancomycin,
VANCOMYCIN, vancomycin all appear). Pattern-based ILIKE matching is used in
preference to exact string matching. Patterns used: vancomycin, piperacillin,
meropenem, ceftriaxone, ciprofloxacin, metronidazole, ampicillin, levofloxacin,
azithromycin, clindamycin, daptomycin, linezolid, nafcillin, oxacillin,
doxycycline, tobramycin, penicillin, amoxicillin, tigecycline, ceftolozane,
moxifloxacin.

**Excluded preparations**: ophthalmic, topical, vaginal, heparin locks,
desensitization protocols, graded challenge protocols, placebo entries.
These are non-systemic routes or research/allergy protocols not relevant to
sepsis treatment.

**Coverage**: 35.81% of cohort admissions had at least one antibiotic
administration record. This is the expected order of magnitude for a general
ICU population where not all patients receive antibiotics.

### Event Type Filtering

Only event types confirming the drug reached the patient are included.
Excluded: Not Given, Hold Dose, Flushed (line maintenance), Confirmed (order
confirmation only). Included: Administered, Started (IV infusion initiation),
Restarted, Delayed Administered, Partial Administered, and location variants
of each. Started and Restarted are included because IV antibiotic infusions
in ICU are typically charted as Started at infusion initiation.

---

## Vasopressors

### Source

`mimiciv_icu.inputevents`, filtered to vasopressor itemids from `mimiciv_icu.d_items`.
Data profiling found 850,938 records after cleaning. Coverage was 30.98% of
cohort stays, consistent with the known prevalence of haemodynamic instability
in the general ICU population.

Vasopressor requirement is the SOFA cardiovascular component. Vasopressor
administration (score 3-4) indicates haemodynamic instability consistent with
septic shock.

### Itemid Selection

| Drug | Itemid(s) | Notes |
|------|-----------|-------|
| Norepinephrine | 221906 | First-line vasopressor in septic shock |
| Epinephrine | 221289, 229617 | 229617 is duplicate entry with trailing period in label |
| Dopamine | 221662 | |
| Dobutamine | 221653 | Inotrope, used in cardiogenic shock |
| Vasopressin | 222315 | Second-line adjunct; recorded in units/hour not mg |
| Phenylephrine | 221749, 229632, 229630, 229631 | Multiple itemids reflect different pre-mixed concentrations |
| Milrinone | 221986 | Phosphodiesterase inhibitor inotrope |

Phenylephrine (Intubation) itemid 229789 was excluded. This is a one-off bolus
used during intubation to maintain blood pressure and does not reflect
sustained vasopressor support indicating cardiovascular organ dysfunction.

### Unit Corrections and Data Quality

All corrections confirmed by data profiling of the raw `inputevents` table.
0 records had null starttime or endtime after cleaning.

| Drug | Issue | Rows | Correction |
|------|-------|------|------------|
| Norepinephrine | Recorded as mg/kg/min | 2 | Converted to mcg/kg/min (x1000). mg/kg/min would be a lethal dose, almost certainly a data entry error |
| Phenylephrine | Recorded as mcg/min | 2 | Nulled. Cannot convert to mcg/kg/min without patient weight |
| Vasopressin | Recorded as units/min | 3 | Converted to units/hour (x60). units/hour is the predominant unit |
| Epinephrine (229617) | No rateuom | 234 | Nulled |

236 rows have null rate after cleaning. 2 rows were found with negative rate
values and are removed in `clean.py` as physiologically impossible. These are
separate from the unit correction rows and likely represent data entry errors.

---

## Urine Output

### Source

`mimiciv_icu.outputevents`, filtered to urine output itemids. Data profiling
found 4,130,915 records after cleaning. Coverage was 97.0% of cohort stays,
as expected for an ICU population where urinary catheterisation is near-universal.

Urine output is the SOFA renal component alongside creatinine. Low urine output
(oliguria or anuria) indicates acute kidney injury consistent with
sepsis-related organ dysfunction.

### Itemid Selection

Included: Foley catheter (226559), spontaneous void (226560), condom catheter
(226561), straight catheter (226567), suprapubic catheter (226563), ileoconduit
(226584), left/right ureteral stents (226558, 226557), left/right nephrostomies
(226565, 226564).

Excluded: OR Urine (226627) and PACU Urine (226631) since these fall outside
the ICU stay. GU Irrigant/Urine Out (227489) and Urine and GU Irrigant Out
(226566) excluded since these volumes are contaminated with bladder irrigation
fluid and do not reflect true urine production.

### Data Quality

24 negative value rows removed as physiologically impossible data entry errors.
479 rows flagged via `large_void_flag` where value exceeds 2000 mL per entry.
These are retained but flagged since they may represent accumulated totals
charted as a single entry rather than a genuine single void.

---

## Ventilation

### Source

`mimiciv_icu.procedureevents`, filtered to mechanical ventilation itemids.
Data profiling found 38,717 records after cleaning, covering 35.85% of cohort
stays. 0 records removed due to null timestamps.

Ventilation status is required for correct interpretation of the PF ratio
(SpO2/FiO2) in SOFA respiratory scoring. The threshold for respiratory organ
dysfunction differs between ventilated and non-ventilated patients.

### Itemid Selection

| itemid | label | Notes |
|--------|-------|-------|
| 225792 | Invasive ventilation | Endotracheal intubation with mechanical ventilation |
| 225794 | Non-invasive ventilation | BiPAP/CPAP without intubation |

Ventilator settings (mode, rate, PEEP, tidal volume) are charted in
`chartevents`, not `procedureevents`. These are captured via the vitals
extract. Only the procedural onset/offset events are captured here to
determine whether mechanical ventilation was active during the observation
window.

---

## Diagnosis Codes

### Source

`mimiciv_hosp.diagnoses_icd`, all ICD-9 and ICD-10 codes for cohort admissions.
Data profiling found 1,759,242 diagnosis code records across 84,703 unique
admissions after cleaning.

ICD codes serve two purposes: comorbidity feature derivation (Charlson,
Elixhauser, individual flags) and cross-validation of Sepsis-3 derived onset
labels.

### Why ICD Codes Are Not Used for Cohort Exclusion

ICD diagnosis codes in MIMIC-IV are assigned at hospital discharge, not at
admission. They carry no onset timestamp. A patient coded with sepsis may have
developed it at any point during their admission. Using ICD codes as the
primary sepsis-on-admission exclusion criterion would therefore exclude patients
who developed sepsis mid-stay, which are precisely the positive cases the model
is trained to identify.

Sepsis-3 derived onset time (suspected infection + SOFA increase of 2 or more)
is used for cohort exclusion instead. ICD codes are retained for cross-validation:
admissions flagged by Sepsis-3 derivation should show higher rates of sepsis
ICD codes than those not flagged. If they do not, the derivation logic warrants
review.

### Trailing Whitespace

Some ICD-9 codes in MIMIC-IV have trailing whitespace (e.g. `"99591  "`).
`TRIM()` is applied in `extract.py` and `.str.strip()` is applied at the start
of `derive_comorbidity_features()` to ensure consistent matching.

---

## Comorbidity Features

### Standard Scores

**Charlson Comorbidity Index**: derived using a manual implementation of the
Quan et al. (2005) ICD-10 adaptation, in `comorbidity_scoring.py`. Each
condition maps to a published weight and a set of ICD-10 code prefixes;
an admission's score is the sum of weights for every matched condition,
clipped at a minimum of 0.

**Elixhauser Comorbidity Score**: derived using a manual implementation of
the van Walraven et al. (2009) ICD-10 adaptation, using the same
prefix-matching approach. Broader than Charlson, capturing more conditions
with a wider range of published weights, including several negative weights
(e.g. obesity, depression, drug abuse) reflecting their published association
with lower in-hospital mortality in some populations. Negative weights are
retained as published rather than floored at 0; a negative total score for
an individual admission is expected and not an error. See the Missingness
and Range Filter validation note under Known Limitations for how this
distinction was confirmed during Phase 1 QA.

**Why not `comorbidipy`**: an external library (`comorbidipy`) was initially
evaluated to avoid reimplementing validated ICD-to-comorbidity mappings, but
proved unreliable in practice and was dropped in favour of the manual
implementation described above, based directly on the published Quan et al.
and van Walraven et al. mappings.

**ICD-9 exclusion**: both scores are derived from ICD-10 codes only
(`icd_version == "10"`). ICD-9 codes are excluded entirely for these two
composite scores, since the Quan and van Walraven mappings used here are
ICD-10-specific and a parallel validated ICD-9 mapping was out of scope.
This affects admissions coded entirely under ICD-9, an earlier-era subset
of MIMIC-IV admissions, whose Charlson/Elixhauser scores will read as 0
regardless of true comorbidity burden. This is a known limitation, not an
oversight, and is distinct from the individual condition flags below
(CKD, liver disease, malignancy, diabetes, immunosuppression), which are
derived from `diagnosis_clean.parquet` without an ICD version filter and so
are not subject to this exclusion.

### Chronic Kidney Disease (CKD)

**Clinical relevance**: CKD affects baseline creatinine. A patient with CKD
stage 4 may have a baseline creatinine of 3.0 mg/dL on a good day. If the
model only sees raw creatinine without CKD context, it will systematically
over-score renal dysfunction in CKD patients regardless of sepsis. CKD is also
an independent risk factor for sepsis due to immunocompromise and frequent
vascular access.

**Staging**: 1 through 5/ESRD. Unspecified codes (N189, 5859) and hypertensive
CKD-unspecified codes (I12, 40390, 40391) confirm CKD presence but do not
contribute a numeric stage. `has_ckd` is set True; `ckd_stage` remains null.
This avoids fabricating severity precision from unspecified codes.

**Excluded codes**: obstetric CKD codes (O10x) since these describe
pregnancy-specific complications rather than standalone CKD. Diabetes-with-CKD
combination codes (E0822, E0922, E1322, E1022, E1122) excluded to avoid
double-counting; CKD is captured via the primary CKD codes and diabetes via
the separate diabetes flag.

### Liver Disease

**Clinical relevance**: liver disease affects baseline bilirubin. Cirrhosis and
portal hypertension raise bilirubin independently of sepsis, which confounds
the SOFA hepatic component. Severe liver disease also indicates impaired
synthetic function affecting coagulation and albumin.

**Staging**: 1 (mild, simple chronic hepatitis/fibrosis without decompensation)
and 2 (severe, cirrhosis/hepatic failure/portal hypertension/variceal bleeding).
This mirrors the Charlson convention of distinguishing mild from
moderate-severe liver disease.

### Malignancy

**Clinical relevance**: active malignancy affects prognosis and inflammatory
markers. Metastatic disease is substantially more severe than localised cancer
and is scored separately in both Charlson and Elixhauser.

**Matching approach**: prefix-based rather than exact codes due to 1000+ specific
malignancy codes. ICD-9 prefixes 140-208 (primary) and 196-199 (secondary/
metastatic). ICD-10 prefixes C00-C96 (primary) and C77-C80 (secondary/
metastatic). Staging: 1 (primary), 2 (metastatic). Excluded: screening and
follow-up encounter codes (Z08, Z09, Z12x), family history codes (Z80x, V16x),
genetic susceptibility codes (Z15x, V84x), obstetric malignancy codes (O9Ax).

### Diabetes

**Clinical relevance**: diabetes affects lactate interpretation (elevated lactate
in diabetic ketoacidosis) and glucose metabolism generally. A boolean flag is
sufficient since diabetes type does not materially change the interpretation
for sepsis prediction purposes.

**Matching approach**: prefix-based due to 600+ specific diabetes codes. ICD-9
prefix 250, ICD-10 prefixes E08, E09, E10, E11, E13. Excluded: gestational
diabetes (O244x), neonatal diabetes (P702, 7751), family history and screening
codes.

### Immunosuppression

**Clinical relevance**: immunosuppression affects infection risk and clinical
presentation. Immunocompromised patients may develop sepsis from organisms not
typically pathogenic in healthy adults and may mount an attenuated physiological
response, masking typical sepsis signs.

**Matching approach**: exact code matching. No clean severity ordering exists
across drug-induced, HIV, congenital, and post-transplant immunosuppression.
Excluded: HIV counselling/screening/exposure codes, inconclusive serology
(R75, 79571), pregnancy-complicated HIV (O987x).

---

## Imputation Strategy

Imputation is applied in `features.py` after the observation window (hours 0 to
6) has been applied. The strategies below reflect clinical reasoning about the
nature of each variable's missingness.

| Variable | Strategy | Rationale |
|----------|----------|-----------|
| Heart rate | Forward fill within stay, then median | High-frequency measurement; short gaps are sensor dropout or charting lag rather than true absence |
| Blood pressure | Forward fill within stay, then median | Same rationale as heart rate |
| Respiratory rate | Forward fill within stay, then median | Same rationale |
| SpO2 | Forward fill within stay, then median | Same rationale |
| Temperature | Forward fill within stay, then median | Temperature changes slowly; gaps are charting gaps not true absence |
| GCS | Forward fill within stay | Neurological status changes slowly; carry-forward is clinically standard practice |
| FiO2 | Assume 0.21 (room air) if not ventilated | A patient not on supplemental oxygen is breathing room air at 21% FiO2; this is a clinically justified default. The 99.98% post-cleaning null rate for FiO2 means this imputation applies to almost all patients |
| Creatinine | Last observation carried forward | Lab results are infrequent; LOCF is the clinical standard for stable biomarkers |
| Bilirubin | Last observation carried forward | Same rationale as creatinine |
| Platelet count | Last observation carried forward | Same rationale |
| Lactate | No carry-forward; missingness retained | Lactate reflects acute metabolic state; carrying forward a value from hours earlier would misrepresent the current state. Missingness flag created as a feature |
| WBC | Last observation carried forward | WBC changes slowly in most clinical contexts |
| Haemoglobin | Last observation carried forward | Haemoglobin is stable short-term absent acute haemorrhage |

Imputation strategy decisions are documented here rather than in code comments
because the justification is clinical, not technical. The code in `features.py`
implements these strategies; this document records why they were chosen.

---

## Known Limitations and Design Decisions

**Sepsis-on-admission exclusion circular dependency**: the cohort produced by
`clean.py` is preliminary. `features.py` finalises the cohort by excluding
stays where Sepsis-3 criteria are met within the first 6 hours. This creates
a dependency where `clean.py` outputs feed into `features.py` which then
modifies the effective cohort. This is a known and accepted design constraint
documented in both files.

**ICD codes assigned at discharge**: all diagnosis codes, including those used
for comorbidity scoring and sepsis cross-validation, reflect the discharge
summary rather than admission status. This is a fundamental limitation of
administrative coding data in retrospective EHR studies.

**Non-independence of ICU stays**: treating each stay independently means
patients with multiple ICU stays contribute multiple rows. This is handled
by patient-level train/test splitting but means the model may be influenced
by the same patient's different stays appearing in training data. This is a
known trade-off accepted in published MIMIC-IV research.

**hadm_id denominator for hospital-level tables**: medications and infection
components are linked via `hadm_id`. The denominator for coverage calculations
is 84,703 unique admissions rather than 93,224 stays, since some patients have
multiple ICU stays within a single hospital admission sharing the same `hadm_id`.
Coverage figures for these tables reflect admission-level rather than
stay-level coverage.

**Missingness report denominators**: raw missingness percentages for vitals are
calculated against the cleaned cohort of 93,224 stays. Event table extracts
were produced before the LOS exclusion filter was applied, so rows from excluded
stays appear in the clean parquets. This produces small negative raw missingness
values for frequently-measured vitals (heart rate, respiratory rate, SpO2, GCS).
These artefacts are cosmetic, do not affect feature engineering, and are
explained by pipeline ordering rather than data quality issues.

**MIMIC-IV temporal scope**: all dates are shifted into a deidentified future
period (2100-2200) with each patient assigned an independent offset. This does
not affect within-stay analyses since relative timestamps are preserved, but
precludes cross-patient temporal analysis without using `anchor_year_group`.

---

## Train/Test Split

### Split Level

The dataset is split at the **patient level** (`subject_id`), not the stay level (`stay_id`).

A patient with multiple ICU stays during the MIMIC-IV data collection period contributes multiple rows to the feature matrix. Splitting on `stay_id` would allow different stays from the same patient to appear in both train and test sets. Because patient-level characteristics (comorbidities, age, physiology) are correlated across stays, this would inflate test performance - the model would have implicitly seen the patient during training.

Splitting on `subject_id` using `GroupShuffleSplit` ensures complete patient separation: every stay for a given patient appears exclusively in either train or test. A post-split assertion confirms zero patient overlap.

### Split Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Test size | 20% of patients | Standard train/test proportion for a dataset of this size |
| Random state | 42 | Fixed seed ensures reproducibility across runs |
| Stratification | None | GroupShuffleSplit does not support simultaneous grouping and stratification. Class balance is monitored via logging after the split and handled in modelling via class weights or resampling |

### Output Files

| File | Contents |
|------|----------|
| `features_train.parquet` | Training set: 80% of patients, all stays |
| `features_test.parquet` | Held-out test set: 20% of patients, not touched until final evaluation |
| `feature_names.json` | Ordered list of feature column names for modelling scripts |

Both output parquets retain `stay_id` and `subject_id` alongside features so the split can be reproduced and individual stays can be traced back to patients.

### Class Imbalance

Sepsis onset between hours 6 and 24 is an uncommon event in the general ICU population. The class balance (proportion of `label = 1` in the training set) is logged at split time and must be accounted for in modelling via class weights, threshold selection, or resampling. Using accuracy as the primary metric on an imbalanced dataset would be misleading. AUC-ROC, Brier score, and calibration curves are the appropriate evaluation metrics for this task. See `docs/modelling_decisions.md`.

---

## Diagnosis Codes

### Source

`mimiciv_hosp.diagnoses_icd`, all ICD-9 and ICD-10 codes for cohort admissions.
Data profiling found 1,759,242 diagnosis code records across 84,703 unique
admissions after cleaning.

ICD codes serve two purposes: comorbidity feature derivation (Charlson,
Elixhauser, individual flags) and cross-validation of Sepsis-3 derived onset
labels.

### Why ICD Codes Are Not Used for Cohort Exclusion

ICD diagnosis codes in MIMIC-IV are assigned at hospital discharge, not at
admission. They carry no onset timestamp. A patient coded with sepsis may have
developed it at any point during their admission. Using ICD codes as the
primary sepsis-on-admission exclusion criterion would therefore exclude patients
who developed sepsis mid-stay, which are precisely the positive cases the model
is trained to identify.

Sepsis-3 derived onset time (suspected infection + SOFA increase of 2 or more)
is used for cohort exclusion instead. ICD codes are retained for cross-validation:
admissions flagged by Sepsis-3 derivation should show higher rates of sepsis
ICD codes than those not flagged. If they do not, the derivation logic warrants
review.

### Trailing Whitespace

Some ICD-9 codes in MIMIC-IV have trailing whitespace (e.g. `"99591  "`).
`TRIM()` is applied in `extract.py` and `.str.strip()` is applied at the start
of `derive_comorbidity_features()` to ensure consistent matching.

---

## Comorbidity Features

### Standard Scores

**Charlson Comorbidity Index**: derived using the `comorbidipy` library from
ICD-9 and ICD-10 codes. Higher scores indicate more severe comorbidity burden
and predict worse outcomes. ICD-9 and ICD-10 codes are processed separately
(comorbidipy requires version-specific input) and the maximum score per
admission is taken.

**Elixhauser Comorbidity Score**: broader than Charlson, captures more
conditions. Also derived via `comorbidipy`.

### Chronic Kidney Disease (CKD)

**Clinical relevance**: CKD affects baseline creatinine. A patient with CKD
stage 4 may have a baseline creatinine of 3.0 mg/dL on a good day. If the
model only sees raw creatinine without CKD context, it will systematically
over-score renal dysfunction in CKD patients regardless of sepsis. CKD is also
an independent risk factor for sepsis due to immunocompromise and frequent
vascular access.

**Staging**: 1 through 5/ESRD. Unspecified codes (N189, 5859) and hypertensive
CKD-unspecified codes (I12, 40390, 40391) confirm CKD presence but do not
contribute a numeric stage. `has_ckd` is set True; `ckd_stage` remains null.
This avoids fabricating severity precision from unspecified codes.

**Excluded codes**: obstetric CKD codes (O10x) since these describe
pregnancy-specific complications rather than standalone CKD. Diabetes-with-CKD
combination codes (E0822, E0922, E1322, E1022, E1122) excluded to avoid
double-counting; CKD is captured via the primary CKD codes and diabetes via
the separate diabetes flag.

### Liver Disease

**Clinical relevance**: liver disease affects baseline bilirubin. Cirrhosis and
portal hypertension raise bilirubin independently of sepsis, which confounds
the SOFA hepatic component. Severe liver disease also indicates impaired
synthetic function affecting coagulation and albumin.

**Staging**: 1 (mild, simple chronic hepatitis/fibrosis without decompensation)
and 2 (severe, cirrhosis/hepatic failure/portal hypertension/variceal bleeding).
This mirrors the Charlson convention of distinguishing mild from
moderate-severe liver disease.

### Malignancy

**Clinical relevance**: active malignancy affects prognosis and inflammatory
markers. Metastatic disease is substantially more severe than localised cancer
and is scored separately in both Charlson and Elixhauser.

**Matching approach**: prefix-based rather than exact codes due to 1000+ specific
malignancy codes. ICD-9 prefixes 140-208 (primary) and 196-199 (secondary/
metastatic). ICD-10 prefixes C00-C96 (primary) and C77-C80 (secondary/
metastatic). Staging: 1 (primary), 2 (metastatic). Excluded: screening and
follow-up encounter codes (Z08, Z09, Z12x), family history codes (Z80x, V16x),
genetic susceptibility codes (Z15x, V84x), obstetric malignancy codes (O9Ax).

### Diabetes

**Clinical relevance**: diabetes affects lactate interpretation (elevated lactate
in diabetic ketoacidosis) and glucose metabolism generally. A boolean flag is
sufficient since diabetes type does not materially change the interpretation
for sepsis prediction purposes.

**Matching approach**: prefix-based due to 600+ specific diabetes codes. ICD-9
prefix 250, ICD-10 prefixes E08, E09, E10, E11, E13. Excluded: gestational
diabetes (O244x), neonatal diabetes (P702, 7751), family history and screening
codes.

### Immunosuppression

**Clinical relevance**: immunosuppression affects infection risk and clinical
presentation. Immunocompromised patients may develop sepsis from organisms not
typically pathogenic in healthy adults and may mount an attenuated physiological
response, masking typical sepsis signs.

**Matching approach**: exact code matching. No clean severity ordering exists
across drug-induced, HIV, congenital, and post-transplant immunosuppression.
Excluded: HIV counselling/screening/exposure codes, inconclusive serology
(R75, 79571), pregnancy-complicated HIV (O987x).

---

## Imputation Strategy

Imputation is applied in `features.py` after the observation window (hours 0 to
6) has been applied. The strategies below reflect clinical reasoning about the
nature of each variable's missingness.

| Variable | Strategy | Rationale |
|----------|----------|-----------|
| Heart rate | Forward fill within stay, then median | High-frequency measurement; short gaps are sensor dropout or charting lag rather than true absence |
| Blood pressure | Forward fill within stay, then median | Same rationale as heart rate |
| Respiratory rate | Forward fill within stay, then median | Same rationale |
| SpO2 | Forward fill within stay, then median | Same rationale |
| Temperature | Forward fill within stay, then median | Temperature changes slowly; gaps are charting gaps not true absence |
| GCS | Forward fill within stay | Neurological status changes slowly; carry-forward is clinically standard practice |
| FiO2 | Assume 0.21 (room air) if not ventilated | A patient not on supplemental oxygen is breathing room air at 21% FiO2; this is a clinically justified default. The 99.98% post-cleaning null rate for FiO2 means this imputation applies to almost all patients |
| Creatinine | Last observation carried forward | Lab results are infrequent; LOCF is the clinical standard for stable biomarkers |
| Bilirubin | Last observation carried forward | Same rationale as creatinine |
| Platelet count | Last observation carried forward | Same rationale |
| Lactate | No carry-forward; missingness retained | Lactate reflects acute metabolic state; carrying forward a value from hours earlier would misrepresent the current state. Missingness flag created as a feature |
| WBC | Last observation carried forward | WBC changes slowly in most clinical contexts |
| Haemoglobin | Last observation carried forward | Haemoglobin is stable short-term absent acute haemorrhage |

Imputation strategy decisions are documented here rather than in code comments
because the justification is clinical, not technical. The code in `features.py`
implements these strategies; this document records why they were chosen.

---

## Known Limitations and Design Decisions

**Sepsis-on-admission exclusion circular dependency**: the cohort produced by
`clean.py` is preliminary. `features.py` finalises the cohort by excluding
stays where Sepsis-3 criteria are met within the first 6 hours. This creates
a dependency where `clean.py` outputs feed into `features.py` which then
modifies the effective cohort. This is a known and accepted design constraint
documented in both files.

**ICD codes assigned at discharge**: all diagnosis codes, including those used
for comorbidity scoring and sepsis cross-validation, reflect the discharge
summary rather than admission status. This is a fundamental limitation of
administrative coding data in retrospective EHR studies.

**Non-independence of ICU stays**: treating each stay independently means
patients with multiple ICU stays contribute multiple rows. This is handled
by patient-level train/test splitting but means the model may be influenced
by the same patient's different stays appearing in training data. This is a
known trade-off accepted in published MIMIC-IV research.

**hadm_id denominator for hospital-level tables**: medications and infection
components are linked via `hadm_id`. The denominator for coverage calculations
is 84,703 unique admissions rather than 93,224 stays, since some patients have
multiple ICU stays within a single hospital admission sharing the same `hadm_id`.
Coverage figures for these tables reflect admission-level rather than
stay-level coverage.

**Missingness report denominators**: raw missingness percentages for vitals are
calculated against the cleaned cohort of 93,224 stays. Event table extracts
were produced before the LOS exclusion filter was applied, so rows from excluded
stays appear in the clean parquets. This produces small negative raw missingness
values for frequently-measured vitals (heart rate, respiratory rate, SpO2, GCS).
These artefacts are cosmetic, do not affect feature engineering, and are
explained by pipeline ordering rather than data quality issues.

**MIMIC-IV temporal scope**: all dates are shifted into a deidentified future
period (2100-2200) with each patient assigned an independent offset. This does
not affect within-stay analyses since relative timestamps are preserved, but
precludes cross-patient temporal analysis without using `anchor_year_group`.

---

## Train/Test Split

### Split Level

The dataset is split at the **patient level** (`subject_id`), not the stay level (`stay_id`).

A patient with multiple ICU stays during the MIMIC-IV data collection period contributes multiple rows to the feature matrix. Splitting on `stay_id` would allow different stays from the same patient to appear in both train and test sets. Because patient-level characteristics (comorbidities, age, physiology) are correlated across stays, this would inflate test performance - the model would have implicitly seen the patient during training.

Splitting on `subject_id` using `GroupShuffleSplit` ensures complete patient separation: every stay for a given patient appears exclusively in either train or test. A post-split assertion confirms zero patient overlap.

### Split Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Test size | 20% of patients | Standard train/test proportion for a dataset of this size |
| Random state | 42 | Fixed seed ensures reproducibility across runs |
| Stratification | None | GroupShuffleSplit does not support simultaneous grouping and stratification. Class balance is monitored via logging after the split and handled in modelling via class weights or resampling |

### Output Files

| File | Contents |
|------|----------|
| `features_train.parquet` | Training set: 80% of patients, all stays |
| `features_test.parquet` | Held-out test set: 20% of patients, not touched until final evaluation |
| `feature_names.json` | Ordered list of feature column names for modelling scripts |

Both output parquets retain `stay_id` and `subject_id` alongside features so the split can be reproduced and individual stays can be traced back to patients.

### Class Imbalance

Sepsis onset between hours 6 and 24 is an uncommon event in the general ICU population. The class balance (proportion of `label = 1` in the training set) is logged at split time and must be accounted for in modelling via class weights, threshold selection, or resampling. Using accuracy as the primary metric on an imbalanced dataset would be misleading. AUC-ROC, Brier score, and calibration curves are the appropriate evaluation metrics for this task. See `docs/modelling_decisions.md`.