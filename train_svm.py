"""Reproducible HOG + linear SVM experiment for HS1502.

Uses the frozen cleaned_cohort.csv patient-level splits; no re-splitting.
Produces svm_test_results.csv and svm_test_summary.csv (one model per age group).

Usage:
    python train_svm.py cleaned_cohort.csv /path/to/images --output-dir results/svm

Dependencies: numpy, pandas, Pillow, scikit-image, scikit-learn.
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import hog
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

SEED = 42
IMAGE_SIZE = 224
C_VALUES = (0.0001, 0.0003, 0.001, 0.003, 0.01)
LABEL_MAP = {"Atelectasis": 0, "Pneumonia": 1}
AGE_GROUPS = ("<65", ">=65")


def extract_hog_features(image_path: Path) -> np.ndarray:
    # Same preprocessing and HOG settings as the Kaggle SVM notebook.
    with Image.open(image_path) as image:
        image = image.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
        pixels = np.asarray(image, dtype=np.float32) / 255.0
    return hog(pixels, orientations=9, pixels_per_cell=(16, 16),
               cells_per_block=(2, 2), block_norm="L2-Hys",
               feature_vector=True).astype(np.float32)


def load_cohort(csv_path: Path, image_dir: Path) -> pd.DataFrame:
    cohort = pd.read_csv(csv_path)
    required = {"Image Index", "Patient ID", "label", "age_group", "split"}
    if not required.issubset(cohort.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(cohort.columns))}")
    if not cohort["label"].isin(LABEL_MAP).all():
        raise ValueError("Unexpected disease label")
    if not cohort["age_group"].isin(AGE_GROUPS).all():
        raise ValueError("Unexpected age group")
    if not cohort["split"].isin(("train", "validation", "test")).all():
        raise ValueError("Unexpected split")
    if cohort.groupby("Patient ID")["split"].nunique().max() != 1:
        raise ValueError("Patient leakage between splits")
    cohort["image_path"] = cohort["Image Index"].map(lambda name: image_dir / name)
    missing = [p for p in cohort["image_path"] if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} missing X-rays, e.g. {missing[0]}")
    print(f"Loaded {len(cohort):,} images from {cohort['Patient ID'].nunique():,} patients")
    print(cohort.groupby(["split", "age_group", "label"]).size())
    return cohort


def extract_all_features(cohort: pd.DataFrame) -> np.ndarray:
    features = []
    start = time.time()
    for i, path in enumerate(cohort["image_path"], 1):
        features.append(extract_hog_features(path))
        if i % 1000 == 0:
            print(f"HOG: {i:,}/{len(cohort):,} images ({time.time()-start:.1f}s)", flush=True)
    X = np.asarray(features, dtype=np.float32)
    print(f"Feature matrix: {X.shape}")
    return X


def select_model(X_train, y_train, X_val, y_val):
    best_model, best_c, best_auc = None, None, -np.inf
    history = []
    for c in C_VALUES:
        model = LinearSVC(C=c, class_weight="balanced", random_state=SEED,
                          max_iter=10000, dual="auto")
        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start
        auc = roc_auc_score(y_val, model.decision_function(X_val))
        history.append({"C": c, "Validation AUC": auc, "Training Time": elapsed})
        print(f"  C={c:g}: validation AUC={auc:.4f}, fit={elapsed:.1f}s", flush=True)
        if auc > best_auc:
            best_model, best_c, best_auc = model, c, auc
    return best_model, best_c, best_auc, pd.DataFrame(history)


def validation_threshold(model, X_val, y_val):
    scores = model.decision_function(X_val)
    fpr, tpr, thresholds = roc_curve(y_val, scores)
    index = int(np.argmax(tpr - fpr))  # Youden's J, validation only
    return float(thresholds[index])


def test_metrics(model, X_test, y_test, threshold, age_group):
    scores = model.decision_function(X_test)
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = map(int, confusion_matrix(y_test, predicted, labels=[0, 1]).ravel())
    return {
        "Age Group": age_group,
        "ROC-AUC": roc_auc_score(y_test, scores),
        "Sensitivity": recall_score(y_test, predicted, zero_division=0),
        "Specificity": tn / (tn + fp),
        "Precision": precision_score(y_test, predicted, zero_division=0),
        "F1-score": f1_score(y_test, predicted, zero_division=0),
        "Accuracy": accuracy_score(y_test, predicted),
        "False-Negative Rate": fn / (fn + tp),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }


def run(csv_path: Path, image_dir: Path, output_dir: Path):
    random.seed(SEED)
    np.random.seed(SEED)
    output_dir.mkdir(parents=True, exist_ok=True)
    cohort = load_cohort(csv_path, image_dir)
    X_all = extract_all_features(cohort)
    y_all = cohort["label"].map(LABEL_MAP).to_numpy(dtype=np.int8)
    result_rows, summary_rows = [], []

    for age in AGE_GROUPS:
        print(f"\n=== SVM {age} ===", flush=True)
        age_mask = (cohort["age_group"] == age).to_numpy()
        masks = {split: age_mask & (cohort["split"] == split).to_numpy()
                 for split in ("train", "validation", "test")}
        X_train, X_val, X_test = (X_all[masks[s]] for s in ("train", "validation", "test"))
        y_train, y_val, y_test = (y_all[masks[s]] for s in ("train", "validation", "test"))
        scaler = StandardScaler()  # fitted ONLY on this age group's training rows
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_val = scaler.transform(X_val).astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)

        model, best_c, val_auc, history = select_model(X_train, y_train, X_val, y_val)
        threshold = validation_threshold(model, X_val, y_val)
        metrics = test_metrics(model, X_test, y_test, threshold, age)
        print(f"Selected C={best_c:g}; validation AUC={val_auc:.4f}; "
              f"threshold={threshold:.4f}; test AUC={metrics['ROC-AUC']:.4f}")
        history.to_csv(output_dir / f"svm_validation_{'under65' if age == '<65' else 'over65'}.csv", index=False)

        result_rows.append({
            "Age Group": age, "Seed": SEED, "Best C": best_c,
            "Threshold": threshold, "ROC-AUC": metrics["ROC-AUC"],
            "TN": metrics["TN"], "FP": metrics["FP"],
            "FN": metrics["FN"], "TP": metrics["TP"],
            "Sensitivity": metrics["Sensitivity"],
            "Specificity": metrics["Specificity"],
            "Precision": metrics["Precision"],
            "False Negative Rate": metrics["False-Negative Rate"],
            "Accuracy": metrics["Accuracy"], "F1-score": metrics["F1-score"],
        })
        summary_rows.append({
            "Age Group": age, "ROC-AUC": metrics["ROC-AUC"],
            "Sensitivity": metrics["Sensitivity"],
            "Specificity": metrics["Specificity"],
            "Precision": metrics["Precision"],
            "False Negative Rate": metrics["False-Negative Rate"],
            "Accuracy": metrics["Accuracy"], "F1-score": metrics["F1-score"],
        })

    results = pd.DataFrame(result_rows)
    summary = pd.DataFrame(summary_rows)
    results.to_csv(output_dir / "svm_test_results.csv", index=False)
    summary.to_csv(output_dir / "svm_test_summary.csv", index=False)
    print("\n=== FINAL TEST RESULTS ===")
    print(results.to_string(index=False))
    print(f"\nCSV files saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="Frozen cleaned_cohort.csv")
    parser.add_argument("image_dir", type=Path, help="Directory containing X-ray PNGs")
    parser.add_argument("--output-dir", type=Path, default=Path("results/svm"))
    args = parser.parse_args()
    run(args.csv_path, args.image_dir, args.output_dir)


if __name__ == "__main__":
    main()
