# Machine Learning for Radiographic Differential Diagnosis of Atelectasis and Pneumonia Across Age Groups

**HS1502 — Conceptual Introduction to Machine Learning**  
National University of Singapore

## Overview

Atelectasis and pneumonia can present with similar features on chest X-rays, potentially making radiographic differentiation challenging. Patient age may also influence these radiographic presentations and therefore affect the performance of machine learning models attempting to distinguish between the two conditions.

This project investigates whether different machine learning approaches perform differently when distinguishing **atelectasis from pneumonia across younger and older patient populations**.

Using the **NIH ChestX-ray14 dataset**, we will compare multiple machine learning models across two age groups and investigate whether model performance varies with age.

## Research Question

**Does the performance of machine learning models in differentiating atelectasis from pneumonia on chest X-rays vary across age groups?**

We aim to investigate:

- How well different machine learning models distinguish atelectasis from pneumonia.
- Whether their performance changes between younger and older patients.
- Whether different models are better suited to different age groups.
- What characteristics of the models, data, or radiographic presentation might explain any observed differences.

## Dataset

The original source is the **NIH ChestX-ray14 dataset**, containing 112,120 frontal chest X-ray images together with disease labels and patient metadata.

### Data Sources and Provenance

The full original dataset is **not stored in this GitHub repository**.

- **Original data provider:** NIH Clinical Center
- **Official NIH download:** https://nihcc.app.box.com/v/ChestXray-NIHCC
- **Official dataset documentation:** https://docs.cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest
- **Kaggle mirror used for computational access in this project:** https://www.kaggle.com/datasets/nih-chest-xrays/data

The NIH Clinical Center is the original data provider. Kaggle is used only as a convenient computational mirror of the NIH dataset; it is not the original source of the data.

The dataset should be cited as:

> Wang, X., Peng, Y., Lu, L., Lu, Z., Bagheri, M., & Summers, R. M. (2017). *ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases*. IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Paper: https://openaccess.thecvf.com/content_cvpr_2017/html/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.html

Our data flow is therefore:

`NIH ChestX-ray14 (112,120 X-rays) → clean_data.py / preparation pipeline → frozen cleaned_cohort.csv (12,464 X-rays) → CNN / ViT / SVM`

### Frozen Study Cohort

**All CNN, ViT, and SVM experiments in this project must use the same frozen `cleaned_cohort.csv`.** This CSV defines the exact **12,464 X-rays** included in the study and their fixed age-group and train/validation/test assignments. Models should not independently regenerate or randomly re-split the cohort, as doing so would make model comparisons inconsistent.

The cleaned CSV contains one row per selected X-ray, including its image filename, patient ID, recorded age, original finding labels, project target label, age group, and fixed dataset split.

The corresponding 12,464 original-resolution X-rays have also been prepared as a reduced **4.63 GB** image set from the Kaggle-hosted copy of ChestX-ray14. The full 112,120-image dataset is therefore not required for model development once the prepared cohort is available.

The repository's `clean_data.py` documents and reproduces the cohort-selection and patient-level splitting methodology. The frozen `cleaned_cohort.csv` is the authoritative split used for model comparison.

## Data Preparation

### Disease Selection

The study cohort includes X-rays containing exactly one of the two target conditions:

- **Atelectasis present, Pneumonia absent → Atelectasis**
- **Pneumonia present, Atelectasis absent → Pneumonia**
- Images containing both target diseases are excluded because the binary target would be ambiguous.
- Images containing neither target disease are excluded.
- Other co-occurring findings are retained.

After disease filtering and age validation, the final cohort contains:

- **12,464 X-rays**
- **5,286 unique patients**

### Age Groups

X-rays are divided according to the patient's recorded age at the time of the image:

- **< 65 years old**
- **≥ 65 years old**

Patient ages outside the range **1–120 years** are treated as invalid. Two records were removed for implausible ages.

| Age group | Atelectasis | Pneumonia | Total |
| --- | ---: | ---: | ---: |
| < 65 | 9,211 | 1,008 | 10,219 |
| ≥ 65 | 2,085 | 160 | 2,245 |
| **Total** | **11,296** | **1,168** | **12,464** |

### Patient-Level Splitting

The cohort is divided into:

- **60% training**
- **20% validation**
- **20% testing**

Splitting is performed at the **patient level**, meaning all X-rays belonging to the same patient remain in the same train, validation, or test partition. This prevents patient-level data leakage.

A small number of patients have X-rays recorded on both sides of the 65-year age threshold. These **49 cross-age patients are retained but assigned to the training set only**, preventing the same patient from contributing to both age groups in the validation or test comparison.

The remaining patients are stratified by age group and whether they contribute a Pneumonia image to help preserve representation of the minority class.

### Fixed Split Used by All Models

| Split | Age group | Atelectasis X-rays | Pneumonia X-rays |
| --- | --- | ---: | ---: |
| Train | < 65 | 5,570 | 613 |
| Train | ≥ 65 | 1,307 | 111 |
| Validation | < 65 | 1,867 | 188 |
| Validation | ≥ 65 | 436 | 22 |
| Test | < 65 | 1,774 | 207 |
| Test | ≥ 65 | 342 | 27 |

All 12,464 selected X-rays were successfully matched to their corresponding image files.

## Class Imbalance

Atelectasis substantially outnumbers Pneumonia in both age groups. Because a model could achieve high raw accuracy by disproportionately predicting Atelectasis, accuracy alone will not be treated as the primary measure of model performance.

Class imbalance is addressed **during training only** using class weighting. Validation and test distributions remain unaltered.

For the binary CNN, Pneumonia is encoded as the positive class and PyTorch `BCEWithLogitsLoss(pos_weight=...)` is used. The positive-class weight is calculated from the training data as `N_atelectasis / N_pneumonia`, giving approximately **9.086** for the <65 training set and **11.775** for the ≥65 training set. These values are calculated programmatically rather than hard-coded.

## Image Preprocessing

All selected NIH images are 1024 × 1024 pixels. Within the final cohort:

- 12,429 images are stored as grayscale (`L`).
- 35 images are stored as `RGBA`.

Images are converted to a consistent format when loaded rather than modifying the original files.

For the CNN baseline, the current preprocessing pipeline is:

1. Convert to grayscale.
2. Resize from **1024 × 1024** to **224 × 224**.
3. Convert pixel values to tensors scaled to **0–1**.

Model-specific preprocessing may differ where required, particularly for pretrained Vision Transformer architectures. Regardless of preprocessing, **all models must use the same rows and fixed split defined by `cleaned_cohort.csv`.**

## Reproducible Notebooks

The full Kaggle notebooks are available here:

- **Data preparation notebook:** https://www.kaggle.com/code/teddyqmwn/hs1502-prepared-atelectasis-pneumonia-dataset
- **CNN experiment notebook:** https://www.kaggle.com/code/teddyqmwn/cnn-notebook

The data-preparation notebook constructs the frozen 12,464-X-ray cohort from the NIH ChestX-ray14 source. The CNN notebook contains the full multi-seed training, validation-only threshold selection, confusion-matrix analysis, and final test evaluation. Clean script versions are also provided in this repository as `clean_data.py` and `train_cnn.py`.

## Final CNN Test Results

The CNN was trained separately for each age group using the same architecture and hyperparameters with three fixed seeds (42, 123, and 2026). For each run, early stopping used validation loss and the classification threshold was selected on validation data using Youden's J statistic. That threshold was then frozen before evaluation on the untouched test set.

Results below are **mean +/- sample SD across the three fixed-seed test runs**.

| Test metric | < 65 | >= 65 |
| --- | ---: | ---: |
| ROC-AUC | **0.6044 +/- 0.0031** | **0.4628 +/- 0.0483** |
| Pneumonia sensitivity | **0.6393 +/- 0.0590** | **0.3704 +/- 0.1697** |
| Specificity | **0.5115 +/- 0.0592** | **0.5936 +/- 0.0937** |
| Pneumonia precision | **0.1328 +/- 0.0062** | **0.0648 +/- 0.0141** |
| False-negative rate | **0.3607 +/- 0.0590** | **0.6296 +/- 0.1697** |
| Accuracy | **0.5248 +/- 0.0475** | **0.5772 +/- 0.0746** |

Across these three runs, the <65 CNN produced higher and more stable test ROC-AUC and higher Pneumonia sensitivity, whereas the >=65 CNN showed greater variability and a higher false-negative rate. These results should be interpreted cautiously: the >=65 test subgroup contains only **27 Pneumonia X-rays**, so its performance estimates are substantially less precise. The observed difference does not by itself establish that patient age caused the difference in model performance; sample size, class imbalance, and other cohort differences remain important alternative explanations.

## Planned Models

### 1. Convolutional Neural Network (CNN)

A CNN is being developed from scratch to learn image features directly from the chest X-rays and perform binary classification between atelectasis and pneumonia. The current baseline uses four convolutional blocks (16, 32, 64, and 128 channels), ReLU activations, max pooling, global average pooling, dropout, and a single binary output. It contains **97,281 trainable parameters**.

The CNN uses 224 × 224 single-channel inputs, Adam optimization with an initial learning rate of 0.001, weighted binary cross-entropy, and early stopping based on validation loss. The same architecture and training procedure are used for both age groups so differences in performance can be compared meaningfully.

Initial exploratory validation runs showed that results, particularly for the much smaller ≥65 Pneumonia subgroup, may be sensitive to random initialization. Reproducibility controls are therefore being added and the CNN comparison will use fixed random seeds before final test-set evaluation. The test sets remain untouched during model development.

### 2. Vision Transformer (ViT)

A Vision Transformer will be developed or adapted for the same classification task, allowing comparison between transformer-based and convolutional approaches.

### 3. Support Vector Machine (SVM)

An SVM using suitable engineered or extracted image features will provide a traditional machine learning approach to the classification problem.

## Experimental Design

Each model will be evaluated on both age groups:

| Model | < 65 | ≥ 65 |
| --- | :---: | :---: |
| CNN | ✓ | ✓ |
| Vision Transformer (ViT) | ✓ | ✓ |
| SVM | ✓ | ✓ |

This allows comparison both **across models within the same age group** and **across age groups for the same modelling approach**.

## Evaluation

Model performance will be assessed using multiple classification metrics.

### Confusion Matrix

For each model and age group, we will examine:

- True Positives (TP)
- True Negatives (TN)
- False Positives (FP)
- False Negatives (FN)

### False Negatives

Particular attention will be paid to false negatives given their potential importance in a medical classification setting. Rates rather than only raw counts will be used when comparing groups of different sizes.

### ROC and AUC

Receiver Operating Characteristic (ROC) curves and **Area Under the Curve (AUC)** will be used to evaluate how well each model separates the two classes across classification thresholds.

Additional metrics such as precision, recall/sensitivity, specificity, F1-score, and accuracy will also be considered where appropriate.

## Key Experimental Considerations

### Class Imbalance

Pneumonia is substantially less common than atelectasis in the selected cohort, particularly among patients aged 65 and above. Class weighting and appropriate evaluation metrics are therefore important.

### Sample Size

The ≥65 Pneumonia subgroup is small. The final test set contains 27 Pneumonia X-rays from 21 patients aged 65 and above. Small performance differences between models in this subgroup must therefore be interpreted cautiously.

### Patient-Level Data Leakage

Multiple X-rays may belong to the same patient. All images belonging to a patient are kept within a single train, validation, or test split.

### Multi-label Findings

ChestX-ray14 is a multi-label dataset. Other radiographic findings are permitted in the selected cohort as long as an image does not contain both Atelectasis and Pneumonia. These co-occurring findings may act as confounding features and will be acknowledged as a limitation.

### Clinical Interpretation

Differences in model performance across age groups do not by themselves establish that age causes differences in the radiographic presentation of either disease.

## Team

**HS1502 Group 3**

- Sizhe
- Brandon
- Yilun
- Violet
- Thaddeus

## Project Status

**Work in progress**

- Data cleaning and cohort selection: **Completed**
- Frozen `cleaned_cohort.csv` and fixed model splits: **Completed**
- Reduced 12,464-image cohort: **Prepared**
- Image loading and preprocessing pipeline: **Validated**
- CNN baseline architecture and training pipeline: **Completed**
- CNN three-seed validation and final test evaluation: **Completed**
- ViT development: **Planned**
- SVM development: **Planned**
- Model evaluation and comparison: **Planned**

## Disclaimer

This project is conducted for educational purposes as part of **HS1502: Conceptual Introduction to Machine Learning** at the National University of Singapore.

The models developed in this project are experimental and are **not intended for clinical diagnosis or medical decision-making**.
