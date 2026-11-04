"""Problem 1(c) — Per-Patient Time Series Structure.

USAGE
-----
This file provides two interfaces for turning raw PhysioNet 2012 patient
files into the pivoted (bin x parameter) table described in (c):

1. `process_patient(path, outcomes)` — a single, specified patient. Pass
   the path to one patient's `.txt` file (e.g. `set-a/132539.txt`) and the
   outcomes Series (from `load_outcomes`). Returns that one patient's wide
   table.

2. `process_set(set_dir, outcomes_path, patient_range=None)` — a whole
   set (Set A or Set B). By default processes every patient in `set_dir`.
   Pass `patient_range=(start, end)` (a half-open index range into the
   sorted file list, e.g. `(0, 50)` for the first 50 patients) to process
   only a specified subset instead — useful for quick iteration before
   running the full 4,000-patient set.

Place this file in the same folder as `set-a/`, `set-b/`, `Outcomes-a.txt`,
and `Outcomes-b.txt` (see the Data Setup instructions). Run with:

    python P12_preprocess.py

(tests both interfaces: one specified patient via `process_patient`, and a
small specified range plus the full Set A via `process_set`.)
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent
SET_A_DIR = DATA_DIR / "set-a"
SET_B_DIR = DATA_DIR / "set-b"
OUTCOMES_A = DATA_DIR / "Outcomes-a.txt"
OUTCOMES_B = DATA_DIR / "Outcomes-b.txt"

BIN_HOURS = 2

FEATURE_COLS = [
    "Weight", "ALP", "ALT", "AST", "Albumin", "BUN", "Bilirubin",
    "Cholesterol", "Creatinine", "DiasABP", "FiO2", "GCS", "Glucose", "HCO3",
    "HCT", "HR", "K", "Lactate", "MAP", "MechVent", "Mg", "NIDiasABP",
    "NIMAP", "NISysABP", "Na", "PaCO2", "PaO2", "Platelets", "RespRate",
    "SaO2", "SysABP", "Temp", "TroponinI", "TroponinT", "Urine", "WBC",
    "pH", "Age", "Height", "Gender", "ICUType",
]
COL_ORDER = ["pid", "timestamp"] + FEATURE_COLS + ["target", "seg_entry_cnt"]
STATIC_COLS = ["Age", "Gender", "Height", "ICUType", "Weight"]  # RecordID excluded — not a feature


def load_patient_raw(path: Path) -> pd.DataFrame:
    """Long-format table: one row per (Time, Parameter, Value)."""
    df = pd.read_csv(path)
    return df[df.Parameter != "RecordID"]  # RecordID is an identifier, not a feature


def load_outcomes(path: Path) -> pd.Series:
    return pd.read_csv(path).set_index("RecordID")["In-hospital_death"]


def convert_to_relative_hours(df: pd.DataFrame) -> pd.DataFrame:
    """(b)'s conversion — reused here since (c) needs it internally.
    Adds an `Hour` column; Time ("HH:MM") -> float hours.
    """
    df = df.copy()
    df["Hour"] = df["Time"].apply(
        lambda t: int(t.split(":")[0]) + int(t.split(":")[1]) / 60
    )
    return df


def assign_bins(df: pd.DataFrame, bin_hours: int = BIN_HOURS) -> pd.DataFrame:
    """Right-closed 2-hour bins: (0,2], (2,4], ..., (46,48]. A record at
    exactly Hour=0 (e.g. admission-time descriptors) falls into the first bin.
    """
    df = df.copy()
    df["Bin"] = (np.ceil(df["Hour"] / bin_hours) * bin_hours).astype(int)
    df.loc[df["Bin"] == 0, "Bin"] = bin_hours
    return df


def clean_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """Height uses -1 as an explicit "not recorded" sentinel; treat it as
    missing. (Weight also uses -1 as a sentinel in the raw files, but is
    left untouched here.)
    """
    df = df.copy()
    df.loc[(df.Parameter == "Height") & (df.Value == -1), "Value"] = np.nan
    return df


def pivot_patient(df: pd.DataFrame) -> pd.DataFrame:
    """df must already have Hour/Bin columns. Returns one row per bin, one
    column per parameter, plus `seg_entry_cnt` (number of distinct raw
    timestamps aggregated into that bin — a bookkeeping column, not a
    feature). Ties within a bin are resolved by taking the mean. Static
    descriptor columns (STATIC_COLS) are broadcast to every row, since
    they're only ever measured once (at timestamp=2) but are needed
    throughout, including after windowing in (d).
    """
    seg_entry_cnt = df.groupby("Bin")["Hour"].nunique()
    wide = df.pivot_table(index="Bin", columns="Parameter", values="Value", aggfunc="mean")
    wide = wide.reindex(columns=FEATURE_COLS)
    wide["seg_entry_cnt"] = seg_entry_cnt

    for col in STATIC_COLS:
        known = wide[col].dropna()
        if len(known):
            wide[col] = known.iloc[0]

    return wide


def process_patient(path: Path, outcomes: pd.Series) -> pd.DataFrame:
    """End-to-end (b)+(c) for one patient. Returns the full (0-48hr) wide
    table with pid/timestamp/target attached, in COL_ORDER.
    """
    pid = int(path.stem)
    df = load_patient_raw(path)
    df = clean_sentinels(df)
    df = convert_to_relative_hours(df)
    df = assign_bins(df)

    wide = pivot_patient(df)
    wide["pid"] = pid
    wide["timestamp"] = wide.index.astype(int)
    wide["target"] = outcomes.loc[pid]
    wide = wide.reset_index(drop=True)
    return wide[COL_ORDER]


def process_set(
    set_dir: Path, outcomes_path: Path, patient_range: tuple[int, int] | None = None
) -> pd.DataFrame:
    """Process a whole set. `patient_range=(start, end)` restricts to a
    specified half-open slice of the sorted file list; omit it (default)
    to process every patient in `set_dir`.
    """
    outcomes = load_outcomes(outcomes_path)
    files = sorted(set_dir.glob("*.txt"))
    if patient_range is not None:
        start, end = patient_range
        files = files[start:end]
    return pd.concat([process_patient(f, outcomes) for f in files], ignore_index=True)


def main() -> None:
    outcomes = load_outcomes(OUTCOMES_A)

    # Interface 1: a single, specified patient.
    print("=== process_patient: single specified patient (132539) ===")
    one = process_patient(SET_A_DIR / "132539.txt", outcomes)
    print(one.to_string())

    # Interface 2a: process_set with a specified patient range.
    print("\n=== process_set: specified range, first 5 patients of Set A ===")
    subset = process_set(SET_A_DIR, OUTCOMES_A, patient_range=(0, 5))
    print(f"shape: {subset.shape}, unique patients: {subset['pid'].nunique()}")

    # Interface 2b: process_set with no range — every patient in the set.
    print("\n=== process_set: all patients in Set A ===")
    full = process_set(SET_A_DIR, OUTCOMES_A)
    print(f"shape: {full.shape}, unique patients: {full['pid'].nunique()}")


if __name__ == "__main__":
    main()
