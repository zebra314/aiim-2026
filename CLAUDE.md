# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

Coursework for "2026 Fall Artificial Intelligence and Intelligent Medicine" (AIIM). Each homework lives in its own folder (`hw1/`, …) with its own `pyproject.toml`. There is no test suite or linter configured.

`hw1/AIIM_115_1_HW_1.tex` / `.pdf` is the **assignment handout** provided by the course (not the student's report). Read it for requirements, grading criteria, and the P3 appendix of forbidden example questions.

## hw1: PhysioNet 2012 in-hospital mortality prediction

### Setup and running

Python is pinned to 3.12 (`requires-python = "==3.12.*"`); `uv` is available.

```bash
cd hw1
tar xzf set-a.tar.gz && tar xzf set-b.tar.gz   # creates set-a/ and set-b/ (4000 patient .txt files each)
uv run python P12_preprocess.py                # runs one patient, a 5-patient range, then all of Set A

# Execute the submission notebook end-to-end (~45 s); local venv at hw1/.venv has jupyter/nbconvert
.venv/bin/jupyter nbconvert --to notebook --execute --inplace hw1_r15942158.ipynb
```

`P12_preprocess.py` resolves data paths relative to its own file (`DATA_DIR = Path(__file__).parent`), so `set-a/`, `set-b/`, `Outcomes-a.txt`, `Outcomes-b.txt` must sit next to it. The tarballs, `Outcomes-*.txt` and extracted `set-a/` / `set-b/` are gitignored.

### Submission notebook (`hw1_r15942158.ipynb`)

One notebook, one section per sub-question; later cells depend on variables from earlier ones (P2 reuses P1's `win_*`, `pad_*`, `X_*`, `y_*`, `clean_sentinels`, `STATIC`), so always run top to bottom. It imports `P12_preprocess` and extracts the tarballs itself if `set-a/`/`set-b/` are missing. P1(b) writes `132539_hours.txt`, which is part of the submission. P2 uses LightGBM on per-patient summary features of the un-imputed window (seed `SEED = 42`); hyperparameter/design choices are validated by 5-fold CV on Set A only.

**Any statistic learned across patients (e.g. imputation fallbacks via `fit_fallback`, scalers) must be fit on the training part only**: Set A for the final model, and the training folds inside CV — never on precomputed full-Set-A arrays like `X_train` when cross-validating. `cv_scores(build, y)` takes a `build(tr, va) -> (X_tr, X_va)` function for exactly this reason.

### Data format

- Each patient file (`set-*/<RecordID>.txt`) is long-format CSV: `Time,Parameter,Value`, where `Time` is `HH:MM` relative to ICU admission (0–48h). The first rows at `00:00` are general descriptors (`RecordID`, `Age`, `Gender`, `Height`, `ICUType`, `Weight`); everything else is a time-series observation.
- `Outcomes-*.txt` is keyed by `RecordID`; the label is `In-hospital_death`.
- Set A = training, Set B = test. **Set B must never be used for training** (zero credit for the baseline if it is).

### Preprocessing pipeline (`P12_preprocess.py`, course-provided code for P1(c))

`process_patient` chains: `load_patient_raw` (drops `RecordID` rows) → `clean_sentinels` → `convert_to_relative_hours` (adds `Hour`) → `assign_bins` → `pivot_patient`. `process_set(set_dir, outcomes_path, patient_range=(start, end))` concatenates patients; `patient_range` is a half-open slice of the sorted file list for fast iteration.

Non-obvious behavior to keep in mind:
- Bins are **right-closed** 2h bins labelled by their upper edge: `(0,2]→2, …, (46,48]→48`; `Hour == 0` is folded into bin 2. So `timestamp` in the output is the bin's end hour.
- Only bins with at least one observation get a row — patients can have fewer than 24 rows, and missing parameter values within a row are `NaN`.
- Multiple values for the same parameter in a bin are averaged.
- Static columns (`STATIC_COLS`) are broadcast to every row of the patient.
- `Height == -1` is converted to `NaN`; `Weight == -1` is deliberately **left as -1** (also a sentinel in the raw data).
- `seg_entry_cnt` = number of distinct raw timestamps in the bin (bookkeeping, not a feature); `target` = `In-hospital_death`.
- Output column order is `COL_ORDER`: `pid, timestamp, <41 FEATURE_COLS>, target, seg_entry_cnt`.

### Assignment constraints that affect code

- **P1 study design is mandatory**: anchor at hour 36, input window `12 ≤ timestamp ≤ 36` (inclusive), 2h bins → 13 bins per patient. Patients with 0 rows in the window are excluded; the rest are zero-padded to length 13 after imputation. P2/P3 may deviate from this.
- **P2** is scored on Set B AUROC **and** AUPRC (both must clear a tier): Simple ≥0.75/0.38, Medium ≥0.80/0.43, Strong ≥0.83/0.46.
- All seeds tried must be reported, and the final seed must be set in code; results must be reproducible.
- Final submission is graded on **Google Colab (GPU)**: `hw1_<student_ID>.zip` containing exactly one folder `hw1_<student_ID>/` with `hw1_<student_ID>.pdf` and `hw1_<student_ID>.ipynb` (plus any extra `.py` files). Report in English, ≤10 pages, ≥11pt.
- P3 must use only the P12 dataset and must not reuse P1, P2, or any appendix example as its question.
