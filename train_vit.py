"""Reproducible ViT experiment for the HS1502 Atelectasis-vs-Pneumonia study.

This script follows the completed Kaggle notebook implementation:
- Frozen cleaned_cohort.csv; no patient re-splitting
- ViT-B/16 pretrained on ImageNet-1K
- 224x224 grayscale X-rays replicated to 3 channels
- ImageNet normalization
- Batch size 32
- Seeds 42, 123, 2026
- Weighted BCE loss using training data only
- AdamW (lr=1e-4, weight_decay=1e-4)
- Early stopping on validation loss (20 epochs, patience 4)
- Threshold selected on validation data using Youden's J
- Untouched test set used only for final evaluation
- Outputs vit_test_results.csv and vit_test_summary.csv

Usage:
    python train_vit.py cleaned_cohort.csv /path/to/images --output-dir results/vit
"""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ViT_B_16_Weights, vit_b_16
from torchvision import transforms


SEEDS = [42, 123, 2026]
AGE_GROUPS = ["<65", ">=65"]

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
MAX_EPOCHS = 20
PATIENCE = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

LABEL_MAP = {
    "Atelectasis": 0,
    "Pneumonia": 1,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class ChestXrayDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        with Image.open(row["image_path"]) as image:
            image = image.convert("L")
            if self.transform:
                image = self.transform(image)

        label = torch.tensor(
            LABEL_MAP[row["label"]],
            dtype=torch.float32,
        )

        return image, label


def create_loaders(cohort: pd.DataFrame, age_group: str, seed: int):
    age_data = cohort[
        cohort["age_group"] == age_group
    ].copy()

    train_df = age_data[
        age_data["split"] == "train"
    ].copy()

    val_df = age_data[
        age_data["split"] == "validation"
    ].copy()

    test_df = age_data[
        age_data["split"] == "test"
    ].copy()

    train_dataset = ChestXrayDataset(train_df, transform=transform)
    val_dataset = ChestXrayDataset(val_df, transform=transform)
    test_dataset = ChestXrayDataset(test_df, transform=transform)

    generator = torch.Generator()
    generator.manual_seed(seed)

    common_args = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": True,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **common_args,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **common_args,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **common_args,
    )

    return train_loader, val_loader, test_loader


def calculate_pos_weight(cohort: pd.DataFrame, age_group: str) -> float:
    train_data = cohort[
        (cohort["age_group"] == age_group)
        & (cohort["split"] == "train")
    ]

    n_atelectasis = (
        train_data["label"] == "Atelectasis"
    ).sum()

    n_pneumonia = (
        train_data["label"] == "Pneumonia"
    ).sum()

    return float(n_atelectasis / n_pneumonia)


def train_vit(
    model,
    train_loader,
    val_loader,
    criterion,
    device,
    learning_rate=LEARNING_RATE,
    max_epochs=MAX_EPOCHS,
    patience=PATIENCE,
):
    print("=== TRAINING ViT ===")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        model.train()
        running_train_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            logits = model(images).squeeze(1)

            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * images.size(0)

        train_loss = (
            running_train_loss / len(train_loader.dataset)
        )

        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                logits = model(images).squeeze(1)

                loss = criterion(logits, labels)

                running_val_loss += loss.item() * images.size(0)

        val_loss = (
            running_val_loss / len(val_loader.dataset)
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"Epoch {epoch + 1:02d}/{max_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)

    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"Epochs trained: {len(history['train_loss'])}")

    return model, history


def get_predictions(model, loader, device):
    model.eval()

    all_probabilities = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)

            logits = model(images)
            probabilities = torch.sigmoid(logits)

            all_probabilities.extend(
                probabilities.squeeze(1).cpu().numpy()
            )
            all_labels.extend(labels.numpy())

    return (
        np.asarray(all_labels),
        np.asarray(all_probabilities),
    )


def find_optimal_threshold(labels, probabilities):
    fpr, tpr, thresholds = roc_curve(
        labels,
        probabilities,
    )

    j_scores = tpr - fpr
    best_index = np.argmax(j_scores)

    return float(thresholds[best_index])


def calculate_metrics(labels, probabilities, threshold):
    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    fnr = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else 0.0
    )

    return {
        "ROC-AUC": roc_auc_score(labels, probabilities),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision,
        "FNR": fnr,
        "Accuracy": accuracy_score(labels, predictions),
        "F1": f1_score(
            labels,
            predictions,
            zero_division=0,
        ),
    }


def run_vit_experiment(
    cohort,
    age_group,
    seed,
    device,
):
    print("\n" + "=" * 70)
    print(
        f"ViT experiment | Age group: {age_group} | Seed: {seed}"
    )
    print("=" * 70)

    set_seed(seed)

    model = vit_b_16(
        weights=ViT_B_16_Weights.IMAGENET1K_V1
    )

    model.heads.head = nn.Linear(
        model.heads.head.in_features,
        1,
    )

    model = model.to(device)

    train_loader, val_loader, test_loader = create_loaders(
        cohort,
        age_group,
        seed=seed,
    )

    pos_weight = calculate_pos_weight(
        cohort,
        age_group,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            pos_weight,
            dtype=torch.float32,
            device=device,
        )
    )

    print(f"\nTraining samples: {len(train_loader.dataset):,}")
    print(f"Validation samples: {len(val_loader.dataset):,}")
    print(f"Test samples: {len(test_loader.dataset):,}")
    print(f"Positive-class weight: {pos_weight:.4f}")

    model, history = train_vit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
    )

    val_labels, val_probabilities = get_predictions(
        model,
        val_loader,
        device,
    )

    threshold = find_optimal_threshold(
        val_labels,
        val_probabilities,
    )

    print(f"\nValidation threshold: {threshold:.4f}")

    test_labels, test_probabilities = get_predictions(
        model,
        test_loader,
        device,
    )

    test_metrics = calculate_metrics(
        labels=test_labels,
        probabilities=test_probabilities,
        threshold=threshold,
    )

    print("\n=== TEST RESULTS ===")
    print(f"ROC-AUC:      {test_metrics['ROC-AUC']:.4f}")
    print(f"Sensitivity:  {test_metrics['Sensitivity']:.4f}")
    print(f"Specificity:  {test_metrics['Specificity']:.4f}")
    print(f"Precision:    {test_metrics['Precision']:.4f}")
    print(f"FNR:          {test_metrics['FNR']:.4f}")
    print(f"Accuracy:     {test_metrics['Accuracy']:.4f}")
    print(f"F1-score:     {test_metrics['F1']:.4f}")

    print("\nConfusion matrix:")
    print(f"TN: {test_metrics['TN']}")
    print(f"FP: {test_metrics['FP']}")
    print(f"FN: {test_metrics['FN']}")
    print(f"TP: {test_metrics['TP']}")

    return {
        "age_group": age_group,
        "seed": seed,
        "threshold": threshold,
        "pos_weight": pos_weight,
        "history": history,
        "val_labels": val_labels,
        "val_probabilities": val_probabilities,
        "test_labels": test_labels,
        "test_probabilities": test_probabilities,
        **test_metrics,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "cleaned_csv",
        type=Path,
        help="Path to cleaned_cohort.csv",
    )

    parser.add_argument(
        "image_dir",
        type=Path,
        help="Directory containing prepared X-rays",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/vit"),
        help="Directory for CSV outputs",
    )

    args = parser.parse_args()

    cohort = pd.read_csv(args.cleaned_csv)

    cohort["image_path"] = cohort["Image Index"].apply(
        lambda filename: str(args.image_dir / filename)
    )

    missing = cohort[
        ~cohort["image_path"].apply(
            lambda path: Path(path).exists()
        )
    ]

    if len(missing):
        raise FileNotFoundError(
            f"{len(missing)} prepared X-rays are missing"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=== DEVICE ===")
    print("Device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA version:", torch.version.cuda)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for age_group in AGE_GROUPS:
        for seed in SEEDS:
            result = run_vit_experiment(
                cohort=cohort,
                age_group=age_group,
                seed=seed,
                device=device,
            )

            row = {
                "Age Group": result["age_group"],
                "Seed": result["seed"],
                "Threshold": result["threshold"],
                "ROC-AUC": result["ROC-AUC"],
                "TN": result["TN"],
                "FP": result["FP"],
                "FN": result["FN"],
                "TP": result["TP"],
                "Sensitivity": result["Sensitivity"],
                "Specificity": result["Specificity"],
                "Precision": result["Precision"],
                "False Negative Rate": result["FNR"],
                "Accuracy": result["Accuracy"],
            }

            rows.append(row)

    results = pd.DataFrame(rows)

    result_columns = [
        "Age Group",
        "Seed",
        "Threshold",
        "ROC-AUC",
        "TN",
        "FP",
        "FN",
        "TP",
        "Sensitivity",
        "Specificity",
        "Precision",
        "False Negative Rate",
        "Accuracy",
    ]

    results = results[result_columns]

    results.to_csv(
        args.output_dir / "vit_test_results.csv",
        index=False,
    )

    metrics = [
        "ROC-AUC",
        "Sensitivity",
        "Specificity",
        "Precision",
        "False Negative Rate",
        "Accuracy",
    ]

    summary_rows = []

    for age_group in AGE_GROUPS:
        subset = results[
            results["Age Group"] == age_group
        ]

        row = {
            "Age Group": age_group,
        }

        for metric in metrics:
            row[f"{metric} Mean"] = subset[metric].mean()
            row[f"{metric} SD"] = subset[metric].std(ddof=1)

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    summary_columns = [
        "Age Group",
        "ROC-AUC Mean",
        "ROC-AUC SD",
        "Sensitivity Mean",
        "Sensitivity SD",
        "Specificity Mean",
        "Specificity SD",
        "Precision Mean",
        "Precision SD",
        "False Negative Rate Mean",
        "False Negative Rate SD",
        "Accuracy Mean",
        "Accuracy SD",
    ]

    summary = summary[summary_columns]

    summary.to_csv(
        args.output_dir / "vit_test_summary.csv",
        index=False,
    )

    print("\n=== ViT TEST SUMMARY ===")
    print(summary.round(4).to_string(index=False))

    print("\nFiles saved:")
    print(args.output_dir / "vit_test_results.csv")
    print(args.output_dir / "vit_test_summary.csv")


if __name__ == "__main__":
    main()
