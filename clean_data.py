"""Reproduce the frozen NIH ChestX-ray14 cohort for the HS1502 project.

Study cohort:
- Atelectasis present, Pneumonia absent -> Atelectasis
- Pneumonia present, Atelectasis absent -> Pneumonia
- Both target diseases -> exclude (ambiguous binary target)
- Neither target disease -> exclude
- Other co-occurring findings are retained.
- Patient age must be between 1 and 120 years.

Age groups:
- < 65 years -> <65
- >= 65 years -> >=65

Data splitting:
- 60% train / 20% validation / 20% test at the patient level.
- RANDOM_STATE=42 is fixed so the cohort can be reproduced.
- All X-rays from a patient remain in exactly one split.
- Patients with X-rays in both age groups are forced into training so the
  validation and test age-group comparisons contain mutually exclusive patients.
- Remaining patients are stratified by age group and whether they contribute
  any Pneumonia image, helping preserve the scarce Pneumonia class.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_A = "Atelectasis"
TARGET_B = "Pneumonia"
AGE_CUTOFF = 65
RANDOM_STATE = 42

REQUIRED_COLUMNS = {
    "Image Index",
    "Finding Labels",
    "Patient ID",
    "Patient Age",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create and patient-split the Atelectasis-vs-Pneumonia cohort."
    )
    parser.add_argument(
        "metadata_csv",
        type=Path,
        help="Path to NIH Data_Entry_2017.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/cleaned_cohort.csv"),
        help="Output CSV path (default: data/cleaned_cohort.csv).",
    )
    return parser.parse_args()


def validate_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            "Metadata file is missing required column(s): "
            + ", ".join(sorted(missing))
        )


def clean_age(series: pd.Series) -> pd.Series:
    """Convert ages to numeric and mark missing/implausible ages invalid."""
    age = pd.to_numeric(series, errors="coerce")
    return age.where(age.between(1, 120))


def assign_target(labels: str) -> str | None:
    """Return the binary target when exactly one target disease is present."""
    findings = {item.strip() for item in str(labels).split("|")}
    has_atelectasis = TARGET_A in findings
    has_pneumonia = TARGET_B in findings

    if has_atelectasis and not has_pneumonia:
        return TARGET_A
    if has_pneumonia and not has_atelectasis:
        return TARGET_B
    return None


def build_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """Filter metadata and add project label and age-group columns."""
    validate_columns(df)

    cohort = df.copy()
    cohort["label"] = cohort["Finding Labels"].apply(assign_target)
    cohort = cohort[cohort["label"].notna()].copy()

    cohort["age"] = clean_age(cohort["Patient Age"])
    invalid_age_count = int(cohort["age"].isna().sum())
    cohort = cohort[cohort["age"].notna()].copy()
    cohort["age"] = cohort["age"].astype(int)

    cohort["age_group"] = cohort["age"].apply(
        lambda age: "<65" if age < AGE_CUTOFF else ">=65"
    )

    cohort.attrs["invalid_age_count"] = invalid_age_count
    return cohort


def split_by_patient(cohort: pd.DataFrame) -> tuple[pd.DataFrame, set[int]]:
    """Assign a reproducible 60/20/20 split without patient leakage."""
    cohort = cohort.copy()

    age_groups_per_patient = cohort.groupby("Patient ID")["age_group"].nunique()
    cross_age_patients = set(age_groups_per_patient[age_groups_per_patient > 1].index)

    normal = cohort[~cohort["Patient ID"].isin(cross_age_patients)].copy()

    patient_table = (
        normal.groupby("Patient ID")
        .agg(
            age_group=("age_group", "first"),
            has_pneumonia=("label", lambda x: int((x == TARGET_B).any())),
            n_images=("Image Index", "count"),
        )
        .reset_index()
    )

    patient_table["stratum"] = (
        patient_table["age_group"]
        + "_"
        + patient_table["has_pneumonia"].map({0: TARGET_A, 1: TARGET_B})
    )

    # 60% train, 40% temporary pool.
    train_patients, temp_patients = train_test_split(
        patient_table,
        test_size=0.40,
        random_state=RANDOM_STATE,
        stratify=patient_table["stratum"],
    )

    # Split the temporary pool equally -> 20% validation, 20% test.
    val_patients, test_patients = train_test_split(
        temp_patients,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_patients["stratum"],
    )

    train_ids = set(train_patients["Patient ID"])
    train_ids.update(cross_age_patients)
    val_ids = set(val_patients["Patient ID"])
    test_ids = set(test_patients["Patient ID"])

    if not train_ids.isdisjoint(val_ids):
        raise RuntimeError("Patient leakage detected between train and validation.")
    if not train_ids.isdisjoint(test_ids):
        raise RuntimeError("Patient leakage detected between train and test.")
    if not val_ids.isdisjoint(test_ids):
        raise RuntimeError("Patient leakage detected between validation and test.")

    def assign_split(patient_id: int) -> str:
        if patient_id in train_ids:
            return "train"
        if patient_id in val_ids:
            return "validation"
        if patient_id in test_ids:
            return "test"
        raise RuntimeError(f"Patient {patient_id} was not assigned a split.")

    cohort["split"] = cohort["Patient ID"].apply(assign_split)

    # Final safety check: every patient must occur in exactly one split.
    if cohort.groupby("Patient ID")["split"].nunique().max() != 1:
        raise RuntimeError("Final leakage check failed: a patient spans multiple splits.")

    return cohort, cross_age_patients


def organise_columns(cohort: pd.DataFrame) -> pd.DataFrame:
    # Match the frozen cleaned_cohort.csv schema used by all model notebooks.
    output_columns = [
        "Image Index",
        "Patient ID",
        "Patient Age",
        "Patient Gender",
        "View Position",
        "Finding Labels",
        "label",
        "age_group",
        "split",
    ]
    missing = [col for col in output_columns if col not in cohort.columns]
    if missing:
        raise ValueError("Cannot create frozen cohort; missing output column(s): " + ", ".join(missing))
    return cohort[output_columns].reset_index(drop=True)


def print_summary(
    original: pd.DataFrame,
    cohort: pd.DataFrame,
    cross_age_patients: set[int],
    invalid_age_count: int,
) -> None:
    print("\n=== NIH ChestX-ray14 study cohort ===")
    print(f"Original metadata rows: {len(original):,}")
    print(f"Final study X-rays:      {len(cohort):,}")
    print(f"Unique study patients:   {cohort['Patient ID'].nunique():,}")
    print(f"Invalid ages removed:    {invalid_age_count:,}")
    print(f"Cross-age patients forced into training: {len(cross_age_patients):,}")

    print("\nX-rays by split, age group and label:")
    image_summary = (
        cohort.groupby(["split", "age_group", "label"])
        .size()
        .unstack(fill_value=0)
    )
    print(image_summary.to_string())

    print("\nUnique patients by split, age group and label:")
    patient_summary = (
        cohort.groupby(["split", "age_group", "label"])["Patient ID"]
        .nunique()
        .unstack(fill_value=0)
    )
    print(patient_summary.to_string())

    older_pneumonia = cohort[
        (cohort["age_group"] == ">=65") & (cohort["label"] == TARGET_B)
    ]
    print("\n>=65 Pneumonia X-rays by split:")
    print(older_pneumonia["split"].value_counts().to_string())

    print("\n>=65 Pneumonia unique patients by split:")
    print(older_pneumonia.groupby("split")["Patient ID"].nunique().to_string())

    print("\nLeakage check: PASSED - every patient belongs to exactly one split.")


def main() -> None:
    args = parse_args()

    if not args.metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {args.metadata_csv}")

    original = pd.read_csv(args.metadata_csv)
    cohort = build_cohort(original)
    invalid_age_count = int(cohort.attrs.get("invalid_age_count", 0))

    cohort, cross_age_patients = split_by_patient(cohort)
    cohort = organise_columns(cohort)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(args.output, index=False)

    print_summary(
        original,
        cohort,
        cross_age_patients,
        invalid_age_count,
    )
    print(f"\nSaved cleaned and split cohort to: {args.output}")
    print("\nThis CSV is the frozen cohort definition. CNN, ViT and SVM experiments should")
    print("consume this fixed CSV rather than independently re-splitting the patients.")
    print("Class imbalance should be handled in the TRAINING split only.")


if __name__ == "__main__":
    main()
