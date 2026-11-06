"""Reproducible CNN baseline for the HS1502 Atelectasis-vs-Pneumonia study.

Consumes the frozen cleaned_cohort.csv and prepared images. It never re-splits
patients. For each age group (<65, >=65), it trains the same CNN with seeds
42, 123 and 2026, uses weighted BCE loss, early stopping on validation loss,
selects a validation-only threshold with Youden's J, freezes that threshold,
and evaluates the untouched test set. Final metrics are mean +/- sample SD
across the three seeds.
"""

from __future__ import annotations
import argparse, copy, random
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

SEEDS=[42,123,2026]
IMAGE_SIZE=224
BATCH_SIZE=64
NUM_WORKERS=4
MAX_EPOCHS=20
PATIENCE=4
LEARNING_RATE=0.001
LABEL_MAP={"Atelectasis":0,"Pneumonia":1}

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class ChestXrayDataset(Dataset):
    def __init__(self, df, transform):
        self.df=df.reset_index(drop=True); self.transform=transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row=self.df.iloc[idx]
        with Image.open(row["image_path"]) as im:
            image=self.transform(im.convert("L"))
        label=torch.tensor(LABEL_MAP[row["label"]],dtype=torch.float32)
        return image,label

class ChestXrayCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(1,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1),nn.ReLU(),nn.MaxPool2d(2))
        self.pool=nn.AdaptiveAvgPool2d((1,1))
        self.classifier=nn.Sequential(nn.Flatten(),nn.Dropout(0.3),nn.Linear(128,1))
    def forward(self,x):
        return self.classifier(self.pool(self.features(x))).squeeze(1)

def make_loaders(cohort, age_group):
    tfm=transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
        transforms.ToTensor()])
    age=cohort[cohort["age_group"]==age_group]
    ds={s:ChestXrayDataset(age[age["split"]==s].copy(),tfm)
        for s in ("train","validation","test")}
    common=dict(batch_size=BATCH_SIZE,num_workers=NUM_WORKERS,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=NUM_WORKERS>0)
    return (DataLoader(ds["train"],shuffle=True,**common),
            DataLoader(ds["validation"],shuffle=False,**common),
            DataLoader(ds["test"],shuffle=False,**common))

def train_model(cohort, age_group, train_loader, val_loader, device, seed):
    set_seed(seed)
    model=ChestXrayCNN().to(device)
    train_df=cohort[(cohort["age_group"]==age_group)&(cohort["split"]=="train")]
    n0=(train_df["label"]=="Atelectasis").sum()
    n1=(train_df["label"]=="Pneumonia").sum()
    criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n0/n1],dtype=torch.float32,device=device))
    optimizer=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)
    best_loss=float("inf"); best_state=None; stale=0
    for epoch in range(MAX_EPOCHS):
        model.train(); train_sum=0.0
        for images,labels in train_loader:
            images=images.to(device,non_blocking=True); labels=labels.to(device,non_blocking=True)
            optimizer.zero_grad(); loss=criterion(model(images),labels)
            loss.backward(); optimizer.step(); train_sum+=loss.item()*images.size(0)
        model.eval(); val_sum=0.0
        with torch.no_grad():
            for images,labels in val_loader:
                images=images.to(device,non_blocking=True); labels=labels.to(device,non_blocking=True)
                val_sum+=criterion(model(images),labels).item()*images.size(0)
        train_loss=train_sum/len(train_loader.dataset)
        val_loss=val_sum/len(val_loader.dataset)
        print(f"{age_group} seed {seed} epoch {epoch+1:02d}: train={train_loss:.4f}, val={val_loss:.4f}")
        if val_loss<best_loss:
            best_loss=val_loss; best_state=copy.deepcopy(model.state_dict()); stale=0
        else:
            stale+=1
            if stale>=PATIENCE: break
    model.load_state_dict(best_state)
    return model

def predict(model, loader, device):
    model.eval(); ys=[]; ps=[]
    with torch.no_grad():
        for images,labels in loader:
            probs=torch.sigmoid(model(images.to(device,non_blocking=True)))
            ys.extend(labels.numpy()); ps.extend(probs.cpu().numpy())
    return np.asarray(ys),np.asarray(ps)

def choose_threshold(y,p):
    fpr,tpr,thresholds=roc_curve(y,p)
    return float(thresholds[int(np.argmax(tpr-fpr))])

def evaluate(model, loader, threshold, device):
    y,p=predict(model,loader,device); pred=(p>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {"ROC-AUC":roc_auc_score(y,p),"TN":int(tn),"FP":int(fp),"FN":int(fn),"TP":int(tp),
            "Sensitivity":tp/(tp+fn),"Specificity":tn/(tn+fp),
            "Precision":tp/(tp+fp) if tp+fp else 0.0,
            "False Negative Rate":fn/(fn+tp),
            "Accuracy":(tp+tn)/(tp+tn+fp+fn)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("cleaned_csv",type=Path)
    ap.add_argument("image_dir",type=Path)
    ap.add_argument("--output-dir",type=Path,default=Path("results/cnn"))
    args=ap.parse_args()
    cohort=pd.read_csv(args.cleaned_csv)
    cohort["image_path"]=cohort["Image Index"].apply(lambda x:str(args.image_dir/x))
    missing=cohort[~cohort["image_path"].apply(lambda x:Path(x).exists())]
    if len(missing): raise FileNotFoundError(f"{len(missing)} prepared X-rays are missing")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type=="cuda":
        torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    args.output_dir.mkdir(parents=True,exist_ok=True)
    rows=[]
    roc_runs = {"<65": [], ">=65": []}
    for age_group in ("<65",">=65"):
        train_loader,val_loader,test_loader=make_loaders(cohort,age_group)
        for seed in SEEDS:
            model=train_model(cohort,age_group,train_loader,val_loader,device,seed)
            vy,vp=predict(model,val_loader,device)
            threshold=choose_threshold(vy,vp)
            result=evaluate(model,test_loader,threshold,device)

            # Retain test-set ROC coordinates for visualisation.
            # This reuses the trained model and does not retrain it.
            y_test, p_test = predict(model, test_loader, device)
            fpr, tpr, _ = roc_curve(y_test, p_test)
            roc_runs[age_group].append({
                "fpr": fpr,
                "tpr": tpr,
                "auc": result["ROC-AUC"]
            })

            result.update({"Age Group":age_group,"Seed":seed,"Threshold":threshold})
            rows.append(result)
            age_name="under65" if age_group=="<65" else "over65"
            torch.save(model.state_dict(),args.output_dir/f"cnn_{age_name}_seed{seed}.pt")
    results=pd.DataFrame(rows)
    results.to_csv(args.output_dir/"cnn_test_results.csv",index=False)
    metrics=["ROC-AUC","Sensitivity","Specificity","Precision","False Negative Rate","Accuracy"]
    summary=[]
    for age_group in ("<65",">=65"):
        subset=results[results["Age Group"]==age_group]; row={"Age Group":age_group}
        for metric in metrics:
            row[f"{metric} Mean"]=subset[metric].mean()
            row[f"{metric} SD"]=subset[metric].std(ddof=1)
        summary.append(row)
    summary=pd.DataFrame(summary)
    summary.to_csv(args.output_dir/"cnn_test_summary.csv",index=False)

    # --------------------------------------------------------
    # ROC CURVES BY AGE GROUP
    # Mean ROC curve across the three seeds, with +/- 1 SD band.
    # --------------------------------------------------------
    fpr_grid = np.linspace(0.0, 1.0, 201)
    roc_colors = {"<65": "#2878B5", ">=65": "#F28E2B"}

    plt.figure(figsize=(9, 7))

    for age_group in ("<65", ">=65"):
        runs = roc_runs[age_group]
        interpolated_tprs = []

        for run in runs:
            interp_tpr = np.interp(fpr_grid, run["fpr"], run["tpr"])
            interp_tpr[0] = 0.0
            interp_tpr[-1] = 1.0
            interpolated_tprs.append(interp_tpr)

        mean_tpr = np.mean(interpolated_tprs, axis=0)
        sd_tpr = np.std(interpolated_tprs, axis=0, ddof=1)
        auc_values = results.loc[
            results["Age Group"] == age_group, "ROC-AUC"
        ]

        color = roc_colors[age_group]
        plt.plot(
            fpr_grid,
            mean_tpr,
            color=color,
            linewidth=2.5,
            label=(
                f"{age_group} "
                f"(AUC = {auc_values.mean():.3f} "
                f"+/- {auc_values.std(ddof=1):.3f})"
            )
        )
        plt.fill_between(
            fpr_grid,
            np.clip(mean_tpr - sd_tpr, 0, 1),
            np.clip(mean_tpr + sd_tpr, 0, 1),
            color=color,
            alpha=0.15
        )

    plt.plot(
        [0, 1], [0, 1],
        linestyle="--",
        color="green",
        linewidth=1.5,
        label="Random classifier (AUC = 0.50)"
    )
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("CNN ROC Curves by Age Group", fontsize=15, weight="bold")
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.25)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(
        args.output_dir / "cnn_roc_curves.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()
    plt.close()

    # --------------------------------------------------------
    # ACCURACY BY AGE GROUP
    # Mean +/- sample SD across the three seeds.
    # --------------------------------------------------------
    accuracy_means = [
        summary.loc[summary["Age Group"] == group, "Accuracy Mean"].iloc[0] * 100
        for group in ("<65", ">=65")
    ]
    accuracy_sds = [
        summary.loc[summary["Age Group"] == group, "Accuracy SD"].iloc[0] * 100
        for group in ("<65", ">=65")
    ]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(
        ["<65", ">=65"],
        accuracy_means,
        yerr=accuracy_sds,
        capsize=7,
        color=[roc_colors["<65"], roc_colors[">=65"]],
        width=0.55,
        edgecolor="black",
        linewidth=0.7
    )

    for bar, mean, sd in zip(bars, accuracy_means, accuracy_sds):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            mean + sd + 1,
            f"{mean:.2f}% +/- {sd:.2f}%",
            ha="center",
            fontsize=10,
            weight="bold"
        )

    plt.xlabel("Age Group", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title("CNN Accuracy by Age Group", fontsize=15, weight="bold")
    plt.ylim(0, 80)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        args.output_dir / "cnn_accuracy_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()
    plt.close()

    print("Saved visualisations:")
    print(args.output_dir / "cnn_roc_curves.png")
    print(args.output_dir / "cnn_accuracy_comparison.png")
    print(summary.round(4).to_string(index=False))

if __name__=="__main__":
    main()
