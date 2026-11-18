# What Can Be Known When a Mining Injury Is Reported?

Code and supporting data for

> Kahraman, A. What Can Be Known When a Mining Injury Is Reported? Temporal Availability and Severity Prediction in 24 Years of MSHA Records. *Applied Sciences* **2026**.

The study classifies every field of the MSHA accident record by the stage of the reporting lifecycle at which it becomes available. It then evaluates injury-severity models year by year (each year *y* is predicted by a model trained only on years before *y*) with four feature sets: notification, completed report, assessment and outcome-informed.

## Data

The analysis uses the public MSHA **Accidents** data set (`Accidents.txt`, pipe-delimited, Latin-1), available from the MSHA Open Government Data page: <https://www.msha.gov/data-and-reports>. The paper uses the records of 2000–2023 (259,443 records; 217,305 after the exclusions listed in Table S1). The data file is not redistributed here.

Put `Accidents.txt` in the folder from which the scripts are run, or set the environment variable `MSHA_ACCIDENTS` to its path.

## Requirements

The reported runs used Google Colab with Python 3.13.16 and the versions in `requirements.txt`:

```
pip install -r requirements.txt
```

`river` is pinned to 0.26.1 because the drift-detector defaults reported in Table S3 are those of that version.

## How to run

Each script was run as one Google Colab cell. Every step writes checkpoints to `MSHA_AEJ_v7/` (on Google Drive in Colab, in the current folder otherwise), and an interrupted step resumes from its checkpoints when rerun. Fixed random seed: `20260820`.

| Step | Script | Produces (paper numbering) | Needs |
|---|---|---|---|
| 1 | `code/01_prepare_data_and_feature_sets.py` | Severity mapping (Table S1), field availability classes (Table S2, Table 3), one data set per feature set (`arm_S`, `arm_B`, `arm_L`) | `Accidents.txt` |
| 2 | `code/02_sensitivity.py` | Balanced weights, 1000-term vocabulary, prior-year threshold, one-hot logistic regression (Table 6) | 1 |
| 3 | `code/03_vocabulary_check.py` | *Optional.* Check of the past-only vocabulary; used by step 4 as a cache if present | 1 |
| 4 | `code/04_main_analysis.py` | Completed-report, assessment and outcome-informed runs 2006–2023; four learners (Table 5); sensitivity analyses (Table 6); field-by-field attribution (Table 9); categorical encoding (Table S4); narrative contribution (Section 2.2); amputation analysis (Section 3.4); rolling windows and drift detectors (Section 3.6) | 1, 2 |
| 5 | `code/05_additional_analyses.py` | 1000-term model with prior-year threshold and notification-set threshold results (Table 6); P(severe \| value) scan (Section 3.4) | 2, 4 |
| 6 | `code/06_revision_analyses.py` | Notification set 2006–2023 (Tables 4 and 8); four learners on the notification set (Table 5); hyperparameter sensitivity (Table S5); label maturation (Table 10); vocabulary audit (Table S6, File S1); event-only models (Table 6); excluded records (Table S1); alert budgets (Table 7); alerts per mine (Table S8); PR curves, calibration and decision curves (Table S7). Prints the JSON block used for the figures | 2, 4, 5 |
| 7 | `code/07_versions_and_field_types.py` | Software versions, field encodings and detector defaults (Table S3) | 1 |
| 8 | `code/08_make_figures.py` | Figures 2–5 | `results/` |

Step 4 takes about 4–6 h and step 6 about 3.5 h on a standard Colab runtime.

Figure 1 is a schematic of the reporting lifecycle under 30 CFR Part 50 and is not drawn from data.

## Contents

```
code/       analysis scripts (steps 1-8)
data/       File_S1_vocabulary_1000_terms.csv  (File S1 of the paper)
results/    step06_summary.json                JSON block printed by step 6 (input of step 8)
            table9_field_attribution.csv       Table 9 values printed by step 4 (input of step 8)
            logs/                              full console output of steps 1-7 from the reported runs
figures/    Figures 2-5 as produced by step 8 (PDF and PNG)
```

`data/File_S1_vocabulary_1000_terms.csv` columns: `term`; `screen` (kept, or removed by the outcome-vocabulary screen); `lexicon_group`; `selected_2018` … `selected_2023` (1 if the term is in the vocabulary of that evaluation year or, for removed terms, was a candidate removed in that year); `df_2023` (document frequency in the training records for 2023; empty if the term is not selected for 2023).

## Notes

- Feature-set names in the code differ from those in the paper: `S` (strict) = notification fields, plus the narrative in the completed-report set; `N` / `nonarr` = notification set (structured fields only); `B` (borderline / triage) = assessment set; `L` (leaky) = outcome-informed set.
- Comments and console messages are partly in Turkish. The scripts are the code of the reported runs. The only change is the local data-path line marked `[repo]` in step 1.
- The scripts were developed with the help of Claude (Anthropic), as stated in the paper.

## License

Code: MIT License (see `LICENSE`). The MSHA data are public records of the U.S. Department of Labor.
