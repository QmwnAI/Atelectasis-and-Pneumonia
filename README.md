# Machine Learning for Radiographic Differential Diagnosis of Atelectasis and Pneumonia Across Age Groups

**HS1502 — Conceptual Introduction to Machine Learning**  
National University of Singapore

## Overview

Atelectasis and pneumonia can present with similar features on chest X-rays, potentially making radiographic differentiation challenging. This project investigates whether machine-learning models distinguish the two conditions differently across patient age groups.

**Research question:** Does the performance of machine-learning models in differentiating atelectasis from pneumonia on chest X-rays vary between patients aged <65 and patients aged >=65?

This is an educational research project, not a clinical diagnostic tool.

## Dataset and provenance

The study uses the **NIH ChestX-ray14** dataset.

- Original data provider: NIH Clinical Center
- Official download: https://nihcc.app.box.com/v/ChestXray-NIHCC
- Official dataset documentation: https://docs.cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest
- Kaggle mirror used for computational access: https://www.kaggle.com/datasets/nih-chest-xrays/data
- Paper: Wang, X., Peng, Y., Lu, L., Lu, Z., Bagheri, M., & Summers, R. M. (2017). *ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases*. CVPR. https://openaccess.thecvf.com/content_cvpr_2017/html/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.html

The original images are not stored in this GitHub repository.

### Frozen cohort

The frozen cohort contains **12,464 X-rays from 5,286 unique patients**. All CNN, ViT, and SVM experiments use the same `cleaned_cohort.csv` cohort definition and fixed patient-level splits. Models should not independently regenerate or randomly re-split the cohort.

The selected image labels are:
- Atelectasis present and Pneumonia absent → **Atelectasis**
- Pneumonia present and Atelectasis absent → **Pneumonia**
- Images containing both target diseases, or neither target disease, are excluded.
- Other co-occurring findings are retained.

Patients' recorded ages are divided into **<65** and **>=65**. Ages outside 1–120 are invalid. Cross-age patients with images on both sides of the age threshold are retained in training only to avoid cross-age validation/test leakage.

| Split | Age group | Atelectasis | Pneumonia |
|---|---|---:|---:|
| Train | <65 | 5,570 | 613 |
| Train | >=65 | 1,307 | 111 |
| Validation | <65 | 1,867 | 188 |
| Validation | >=65 | 436 | 22 |
| Test | <65 | 1,774 | 207 |
| Test | >=65 | 342 | 27 |

All selected images were matched to files in the prepared 4.63 GB image cohort.

## Data preparation and preprocessing

The preparation pipeline is documented in `clean_data.py`. The prepared cohort is also available through the Kaggle data-preparation notebook below.

The CNN uses single-channel 224 x 224 images with pixel values scaled to 0–1. The ViT uses 224 x 224 three-channel inputs, with grayscale images replicated across channels and normalized for its ImageNet-pretrained backbone. The SVM extracts HOG features from grayscale 224 x 224 images (6,084 features per image) and standardizes features using training data only.

## Notebooks and code

- **Data preparation notebook:** https://www.kaggle.com/code/teddyqmwn/hs1502-prepared-atelectasis-pneumonia-dataset
- **CNN experiment notebook:** https://www.kaggle.com/code/teddyqmwn/cnn-notebook
- **ViT experiment notebook:** https://www.kaggle.com/code/yilunn/vit-notebook
- **SVM experiment notebook:** https://www.kaggle.com/code/tianningwangv/svm-notebook
- Cohort preparation script: `clean_data.py`
- CNN training script: `train_cnn.py`
- ViT training script: `train_vit.py`
- Detailed seed-level and summary CSVs are under `results/`.

## Model methodology

### CNN

The CNN is trained from scratch with four convolutional blocks (16, 32, 64, and 128 channels), ReLU, max pooling, global average pooling, dropout, and a binary output; it has 97,281 trainable parameters. Training uses Adam (learning rate 0.001), class-weighted binary cross-entropy, and early stopping on validation loss. Pneumonia is the positive class. The three fixed seeds are 42, 123, and 2026.

### Vision Transformer (ViT)

The ViT-B/16 uses an ImageNet-1K-pretrained backbone and a replacement binary classification head. Training uses AdamW (initial learning rate 0.0001), class-weighted binary cross-entropy, early stopping on validation loss, and the same three seeds for each age group.

### Support Vector Machine (SVM)

The SVM uses HOG features and a linear `LinearSVC` with balanced class weights. `StandardScaler` is fitted on training features only. The regularization parameter is selected using validation ROC-AUC from C values 0.0001, 0.0003, 0.001, 0.003, and 0.01. Selected values were 0.0001 for <65 and 0.003 for >=65.

### Threshold selection and evaluation

For CNN and ViT runs, early stopping uses validation loss and the binary decision threshold is selected on validation data using Youden's J statistic, then frozen before test evaluation. Test results are not used to tune models or thresholds. The SVM uses the same frozen cohort and validation/test separation.

Pneumonia is the positive class. Evaluation includes ROC-AUC, sensitivity, specificity, precision, false-negative rate, accuracy, and (where reported) F1-score. Because the classes are imbalanced, accuracy alone is not a suitable indicator of model quality.

## Final test results

CNN and ViT values are **mean +/- sample standard deviation across three fixed-seed runs**. SVM results are from a single final run, so across-seed standard deviations are not available.

### ROC-AUC comparison

| Age group | CNN | ViT | SVM |
|---|---:|---:|---:|
| <65 | 0.6044 +/- 0.0031 | 0.6512 +/- 0.0423 | 0.6615 |
| >=65 | 0.4628 +/- 0.0483 | 0.4755 +/- 0.0104 | 0.6004 |

### CNN test metrics

| Metric | <65 | >=65 |
|---|---:|---:|
| ROC-AUC | 0.6044 +/- 0.0031 | 0.4628 +/- 0.0483 |
| Pneumonia sensitivity | 0.6393 +/- 0.0590 | 0.3704 +/- 0.1697 |
| Specificity | 0.5115 +/- 0.0592 | 0.5936 +/- 0.0937 |
| Pneumonia precision | 0.1328 +/- 0.0062 | 0.0648 +/- 0.0141 |
| False-negative rate | 0.3607 +/- 0.0590 | 0.6296 +/- 0.1697 |
| Accuracy | 0.5248 +/- 0.0475 | 0.5772 +/- 0.0746 |

### ViT test metrics

| Metric | <65 | >=65 |
|---|---:|---:|
| ROC-AUC | 0.6512 +/- 0.0423 | 0.4755 +/- 0.0104 |
| Pneumonia sensitivity | 0.6506 +/- 0.0560 | 0.5062 +/- 0.0566 |
| Specificity | 0.5654 +/- 0.0282 | 0.4990 +/- 0.0440 |
| Pneumonia precision | 0.1490 +/- 0.0156 | 0.0738 +/- 0.0026 |
| False-negative rate | 0.3494 +/- 0.0560 | 0.4938 +/- 0.0566 |
| Accuracy | 0.5743 +/- 0.0276 | 0.4995 +/- 0.0368 |
| F1-score | 0.2424 +/- 0.0241 | 0.1287 +/- 0.0055 |

### SVM test metrics

| Metric | <65 | >=65 |
|---|---:|---:|
| ROC-AUC | 0.6615 | 0.6004 |
| Pneumonia sensitivity | 0.5797 | 0.5556 |
| Specificity | 0.6669 | 0.5936 |
| Pneumonia precision | 0.1688 | 0.0974 |
| False-negative rate | 0.4203 | 0.4444 |
| Accuracy | 0.6577 | 0.5908 |
| F1-score | 0.2614 | 0.1657 |

SVM confusion matrices:

| Age group | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| <65 | 1,183 | 591 | 87 | 120 |
| >=65 | 203 | 139 | 12 | 15 |

## Interpretation and limitations

In the <65 test group, the SVM had the highest single-run ROC-AUC (0.6615), followed closely by the ViT mean (0.6512). In the >=65 test group, the SVM had a higher ROC-AUC (0.6004) than the CNN and ViT means. This is a descriptive comparison, not a fully controlled estimate of model superiority, because the SVM results currently represent a single run while CNN and ViT each use three seeds.

Across the neural models, discrimination was higher in the <65 group than in the >=65 group. The same age-group pattern was smaller for the SVM. These results alone do not establish that age causes performance differences; sample size, class imbalance, cohort differences, and other co-occurring findings may contribute.

The >=65 test set contains only **27 Pneumonia X-rays**, compared with 207 in the <65 test set. Metrics for older patients—especially sensitivity, precision, and false-negative rate—are consequently estimated from few positive cases and should be interpreted cautiously. Performance may vary with cohort composition, and further external validation would be needed before drawing clinical conclusions.

## Team

**HS1502 Group 3**
- Sizhe
- Brandon
- Yilun
- Violet
- Thaddeus

## Project status

- Data cleaning and cohort selection: **Completed**
- Frozen cohort and patient-level splits: **Completed**
- CNN development and evaluation: **Completed**
- ViT development and evaluation: **Completed**
- SVM development and evaluation: **Completed**
- Final cross-model comparison and discussion: **In progress**

## Disclaimer

This is an educational project for **HS1502: Conceptual Introduction to Machine Learning** at the National University of Singapore. These experimental models are **not intended for clinical diagnosis or medical decision-making**.
