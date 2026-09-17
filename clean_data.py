"""Prepare the NIH ChestX-ray14 metadata for the HS1502 project.

This first-stage script only creates the study cohort. It does not preprocess
images or create train/validation/test splits yet.

Cohort rule:
- Atelectasis present, Pneumonia absent -> Atelectasis
- Pneumonia present, Atelectasis absent -> Pneumonia
- Both present -> exclude (ambiguous binary target)
- Neither present -> exclude

Age groups:
- < 65 years -> under_65
- >= 65 years -> over_65
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TARGET_A = "Atelectasis"
TARGET_B = "Pneumonia"
AGE_CUTOFF = 65

REQUIRED_COLUMNS = {
    "Image Index",
    "Finding Labels",
    "Patient ID",
    "Patient Age",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the Atelectasis-vs-Pneumonia study cohort."
    )
    parser.add_argument(
        "metadata_csv",
        type=Path,
        help="Path to the NIH ChestX-ray14 metadata CSV (e.g. Data_Entry_2017.csv).",
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
    """Convert Patient Age to numeric values and mark impossible ages missing."""
    age = pd.to_numeric(series, errors="coerce")

    # ChestX-ray14 has historically contained some implausible age values.
    # Keep only a conservative human age range for this study.
    age = age.where(age.between(0, 120))
    return age


def assign_target(labels: str) -> str | None:
    """Return an unambiguous binary target from the pipe-separated labels."""
    findings = {item.strip() for item in str(labels).split("|")}

    has_atelectasis = TARGET_A in findings
    has_pneumonia = TARGET_B in findings

    if has_atelectasis and not has_pneumonia:
        return TARGET_A
    if has_pneumonia and not has_atelectasis:
        return TARGET_B

    # Both target diseases, or neither target disease.
    return None


def build_cohort(df: pd.DataFrame) -> pd.DataFrame:
    validate_columns(df)

    cohort = df.copy()
    cohort["label"] = cohort["Finding Labels"].apply(assign_target)

    # Keep only unambiguous Atelectasis-vs-Pneumonia examples.
    cohort = cohort[cohort["label"].notna()].copy()

    cohort["age"] = clean_age(cohort["Patient Age"])
    invalid_age_count = int(cohort["age"].isna().sum())
    cohort = cohort[cohort["age"].notna()].copy()
    cohort["age"] = cohort["age"].astype(int)

    cohort["age_group"] = cohort["age"].apply(
        lambda age: "under_65" if age < AGE_CUTOFF else "over_65"
    )

    # Preserve useful original metadata while adding explicit project columns.
    preferred_columns = [
        "Image Index",
        "Patient ID",
        "age",
        "age_group",
        "label",
        "Finding Labels",
    ]
    remaining_columns = [
        col for col in cohort.columns if col not in preferred_columns
    ]
    cohort = cohort[preferred_columns + remaining_columns]

    cohort = cohort.sort_values(
        ["age_group", "label", "Patient ID", "Image Index"]
    ).reset_index(drop=True)

    cohort.attrs["invalid_age_count"] = invalid_age_count
    return cohort


def print_summary(original: pd.DataFrame, cohort: pd.DataFrame) -> None:
    print("\n=== NIH ChestX-ray14 cohort summary ===")
    print(f"Original metadata rows: {len(original):,}")
    print(f"Final study rows:       {len(cohort):,}")
    print(
        "Rows removed for missing/implausible age after disease filtering: "
        f"{cohort.attrs.get('invalid_age_count', 0):,}"
    )

    print("\nCounts by age group and target label:")
    counts = pd.crosstab(cohort["age_group"], cohort["label"], margins=True)
    print(counts.to_string())

    print("\nUnique patients by age group and target label:")
    patients = (
        cohort.groupby(["age_group", "label"])["Patient ID"]
        .nunique()
        .unstack(fill_value=0)
    )
    print(patients.to_string())

    both_age_groups = cohort.groupby("Patient ID")["age_group"].nunique()
    inconsistent = int((both_age_groups > 1).sum())
    if inconsistent:
        print(
            f"\nWARNING: {inconsistent:,} patient(s) appear in both age groups. "
            "Inspect age metadata before creating patient-level splits."
        )


def main() -> None:
    args = parse_args()

    if not args.metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {args.metadata_csv}")

    df = pd.read_csv(args.metadata_csv)
    cohort = build_cohort(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(args.output, index=False)

    print_summary(df, cohort)
    print(f"\nSaved cleaned cohort to: {args.output}")
    print("\nNext step: inspect these counts before deciding train/validation/test splits.")


if __name__ == "__main__":
    main()
