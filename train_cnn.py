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
    for age_group in ("<65",">=65"):
        train_loader,val_loader,test_loader=make_loaders(cohort,age_group)
        for seed in SEEDS:
            model=train_model(cohort,age_group,train_loader,val_loader,device,seed)
            vy,vp=predict(model,val_loader,device)
            threshold=choose_threshold(vy,vp)
            result=evaluate(model,test_loader,threshold,device)
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
    print(summary.round(4).to_string(index=False))

if __name__=="__main__":
    main()
