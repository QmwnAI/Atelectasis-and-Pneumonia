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


def build_distribution_table(cohort: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Build a split/age/label summary with subtotals and grand totals.

    For mode='images', totals are image counts.
    For mode='patients', disease columns count unique patients per label,
    while each Total is calculated directly from distinct Patient IDs.
    """
    if mode not in {"images", "patients"}:
        raise ValueError("mode must be 'images' or 'patients'.")

    def counts_for(subset: pd.DataFrame) -> dict[str, int]:
        if mode == "images":
            label_counts = subset["label"].value_counts()
            total = len(subset)
        else:
            label_counts = (
                subset.groupby("label")["Patient ID"].nunique()
            )
            # Do not add disease-specific patient counts: one patient may
            # contribute images labelled with both target diseases.
            total = subset["Patient ID"].nunique()

        return {
            TARGET_A: int(label_counts.get(TARGET_A, 0)),
            TARGET_B: int(label_counts.get(TARGET_B, 0)),
            "Total": int(total),
        }

    rows: list[dict[str, object]] = []

    # Split-level rows and subtotals.
    for split in ("train", "validation", "test"):
        split_df = cohort[cohort["split"] == split]

        for age_group in ("<65", ">=65"):
            subset = split_df[split_df["age_group"] == age_group]
            rows.append({
                "Partition": split.title(),
                "Age Group": age_group,
                **counts_for(subset),
            })

        rows.append({
            "Partition": f"{split.title()} subtotal",
            "Age Group": "All ages",
            **counts_for(split_df),
        })

    # Age-group totals across all partitions.
    for age_group in ("<65", ">=65"):
        subset = cohort[cohort["age_group"] == age_group]
        rows.append({
            "Partition": "Age group total",
            "Age Group": age_group,
            **counts_for(subset),
        })

    # Grand totals are calculated directly from the full cohort.
    rows.append({
        "Partition": "GRAND TOTAL",
        "Age Group": "All ages",
        **counts_for(cohort),
    })

    return pd.DataFrame(rows)


def style_distribution_table(table: pd.DataFrame, title: str) -> pd.io.formats.style.Styler:
    """Apply consistent styling to cohort distribution tables."""
    def highlight_summary_rows(row: pd.Series) -> list[str]:
        partition = str(row["Partition"])
        if partition == "GRAND TOTAL":
            style = "background-color: #172554; color: white; font-weight: bold;"
        elif "subtotal" in partition:
            style = "background-color: #DBEAFE; color: #1E3A8A; font-weight: bold;"
        elif partition == "Age group total":
            style = "background-color: #EDE9FE; color: #5B21B6; font-weight: bold;"
        else:
            style = ""
        return [style] * len(row)

    return (
        table.style
        .format({
            TARGET_A: "{:,.0f}",
            TARGET_B: "{:,.0f}",
            "Total": "{:,.0f}",
        })
        .background_gradient(cmap="Blues", subset=[TARGET_A])
        .background_gradient(cmap="Oranges", subset=[TARGET_B])
        .apply(highlight_summary_rows, axis=1)
        .set_caption(title)
        .set_properties(**{
            "text-align": "center",
            "padding": "10px",
            "font-size": "13px",
            "border": "1px solid #E2E8F0",
        })
        .set_table_styles([
            {
                "selector": "caption",
                "props": [
                    ("font-size", "18px"),
                    ("font-weight", "bold"),
                    ("color", "#1E3A8A"),
                    ("padding", "14px"),
                    ("text-align", "left"),
                ],
            },
            {
                "selector": "th",
                "props": [
                    ("background-color", "#172554"),
                    ("color", "white"),
                    ("text-align", "center"),
                    ("padding", "11px"),
                    ("border", "1px solid #CBD5E1"),
                ],
            },
            {
                "selector": "table",
                "props": [
                    ("border-collapse", "collapse"),
                    ("width", "100%"),
                    ("font-family", "Arial, sans-serif"),
                ],
            },
        ])
        .hide(axis="index")
    )


def export_summary_html(
    cohort: pd.DataFrame,
    output_path: Path,
    cross_age_patients: set[int],
) -> None:
    """Export both styled distribution tables as a standalone HTML report."""
    image_table = build_distribution_table(cohort, mode="images")
    patient_table = build_distribution_table(cohort, mode="patients")

    image_html = style_distribution_table(
        image_table,
        "X-ray Distribution by Partition, Age Group and Diagnosis",
    ).to_html()

    patient_html = style_distribution_table(
        patient_table,
        "Unique Patient Distribution by Partition, Age Group and Diagnosis",
    ).to_html()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NIH ChestX-ray14 Cohort Summary</title>
<style>
    body {{
        margin: 0;
        padding: 32px;
        background: #F8FAFC;
        color: #0F172A;
        font-family: Arial, sans-serif;
    }}
    main {{
        max-width: 1100px;
        margin: 0 auto;
    }}
    h1 {{ color: #172554; margin-bottom: 6px; }}
    .subtitle {{ color: #64748B; margin-bottom: 24px; }}
    .cards {{
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 14px;
        margin: 22px 0 30px;
    }}
    .card {{
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px;
    }}
    .card .value {{
        display: block;
        color: #1D4ED8;
        font-size: 28px;
        font-weight: 700;
    }}
    .card .label {{ color: #64748B; font-size: 13px; }}
    section {{
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px;
        margin: 20px 0;
        overflow-x: auto;
    }}
    .note {{
        background: #EFF6FF;
        border-left: 4px solid #2563EB;
        padding: 14px 16px;
        border-radius: 6px;
        color: #1E3A8A;
        line-height: 1.5;
        font-size: 13px;
    }}
    @media (max-width: 700px) {{
        body {{ padding: 14px; }}
        .cards {{ grid-template-columns: 1fr; }}
    }}
</style>
</head>
<body>
<main>
    <h1>NIH ChestX-ray14 — Cohort Distribution</h1>
    <div class="subtitle">Frozen Atelectasis-versus-Pneumonia cohort summary</div>
    <div class="cards">
        <div class="card">
            <span class="value">{len(cohort):,}</span>
            <span class="label">Total X-ray images</span>
        </div>
        <div class="card">
            <span class="value">{cohort["Patient ID"].nunique():,}</span>
            <span class="label">Unique patients</span>
        </div>
        <div class="card">
            <span class="value">{len(cross_age_patients):,}</span>
            <span class="label">Cross-age patients assigned to training</span>
        </div>
    </div>
    <section>{image_html}</section>
    <section>{patient_html}</section>
    <div class="note">
        <strong>Counting note:</strong> X-ray totals are additive. In the unique-patient
        table, disease-specific counts may overlap because one patient can have
        images labelled with both findings. Each row total and the grand total
        therefore count distinct Patient IDs directly rather than summing the
        disease-specific columns. Cross-age patients may appear in both age-group
        totals but are assigned exclusively to the training partition.
    </div>
</main>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def print_summary(
    original: pd.DataFrame,
    cohort: pd.DataFrame,
    cross_age_patients: set[int],
    invalid_age_count: int,
    summary_html_path: Path,
) -> None:
    print("\n=== NIH ChestX-ray14 study cohort ===")
    print(f"Original metadata rows: {len(original):,}")
    print(f"Final study X-rays:      {len(cohort):,}")
    print(f"Unique study patients:   {cohort['Patient ID'].nunique():,}")
    print(f"Invalid ages removed:    {invalid_age_count:,}")
    print(f"Cross-age patients forced into training: {len(cross_age_patients):,}")

    image_summary = build_distribution_table(cohort, mode="images")
    patient_summary = build_distribution_table(cohort, mode="patients")

    print("\nX-ray distribution (including subtotals and grand total):")
    print(image_summary.to_string(index=False))

    print("\nUnique patient distribution (including subtotals and grand total):")
    print(patient_summary.to_string(index=False))

    older_pneumonia = cohort[
        (cohort["age_group"] == ">=65") & (cohort["label"] == TARGET_B)
    ]
    print("\n>=65 Pneumonia X-rays by split:")
    print(older_pneumonia["split"].value_counts().to_string())

    print("\n>=65 Pneumonia unique patients by split:")
    print(older_pneumonia.groupby("split")["Patient ID"].nunique().to_string())

    # Verify image totals and unique-patient totals independently.
    assert image_summary.iloc[-1]["Total"] == len(cohort)
    assert patient_summary.iloc[-1]["Total"] == cohort["Patient ID"].nunique()

    print("\nLeakage check: PASSED - every patient belongs to exactly one split.")
    print(f"Styled cohort summary saved to: {summary_html_path}")

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

    summary_html_path = args.output.with_name(
        f"{args.output.stem}_summary.html"
    )

    print_summary(
        original,
        cohort,
        cross_age_patients,
        invalid_age_count,
        summary_html_path,
    )
    export_summary_html(
        cohort,
        summary_html_path,
        cross_age_patients,
    )
    print(f"\nSaved cleaned and split cohort to: {args.output}")
    print("\nThis CSV is the frozen cohort definition. CNN, ViT and SVM experiments should")
    print("consume this fixed CSV rather than independently re-splitting the patients.")
    print("Class imbalance should be handled in the TRAINING split only.")


if __name__ == "__main__":
    main()
