# Machine Learning for Radiographic Differential Diagnosis of Atelectasis and Pneumonia Across Age Groups

HS1502 — Conceptual Introduction to Machine Learning
National University of Singapore

## Overview

Atelectasis and pneumonia can present with similar radiographic features on chest X-rays, making differential diagnosis challenging. Patient age may further affect these radiographic presentations and, consequently, the performance of machine learning models used to distinguish between the two conditions.

This project investigates whether the ability of machine learning models to differentiate **atelectasis from pneumonia on chest X-rays varies across age groups**.

Using the **NIH Chest X-ray dataset**, we will train and evaluate multiple machine learning models for binary classification of atelectasis and pneumonia. Model performance will then be compared across different patient age groups.

## Research Question

**Does the performance of machine learning models in differentiating atelectasis from pneumonia on chest X-rays vary across age groups?**

More specifically, we aim to investigate:

* Whether classification performance differs between age groups.
* Whether different machine learning models exhibit different age-related performance patterns.
* Which models perform best within each age group.

## Dataset

This project uses the **NIH ChestX-ray14 dataset**, a publicly available dataset of frontal chest X-ray images with disease labels and associated patient metadata.

For this study, we will focus on X-rays labelled with:

* **Atelectasis**
* **Pneumonia**

Patient age information will be used to divide the dataset into age groups for comparative analysis.

## Planned Methodology

### 1. Data Preparation

* Extract chest X-rays labelled as atelectasis or pneumonia.
* Review and clean the associated patient metadata.
* Define appropriate patient age groups.
* Examine the class distribution within each age group.
* Address potential class imbalance where necessary.
* Split the data into appropriate training, validation, and test sets while avoiding patient-level data leakage.

### 2. Image Preprocessing

Chest X-rays will be standardised before model training. Depending on the requirements of each model, preprocessing may include:

* Image resizing
* Pixel-value normalisation
* Data augmentation
* Feature extraction or dimensionality reduction

### 3. Model Development

Multiple machine learning approaches will be evaluated to compare how different modelling strategies perform on the classification task.

The final set of models will be selected based on suitability for the dataset, computational feasibility, and relevance to concepts covered in HS1502.

### 4. Evaluation

Models will be evaluated using suitable classification metrics, potentially including:

* Accuracy
* Precision
* Recall / Sensitivity
* Specificity
* F1-score
* ROC-AUC
* Confusion matrices

Performance will then be compared across age groups to determine whether age is associated with systematic differences in classification performance.

## Experimental Considerations

Several factors will need to be considered when interpreting the results:

**Class imbalance**
The number of pneumonia and atelectasis cases may differ substantially. Raw accuracy may therefore be misleading, making additional classification metrics important.

**Sample size**
Dividing the dataset by age reduces the number of observations available within each subgroup. Age groups must therefore contain sufficient examples of both conditions for meaningful model training and evaluation.

**Patient-level leakage**
If multiple X-rays from the same patient are present, they should not be distributed across training and test sets, as this could artificially inflate model performance.

**Dataset labels**
ChestX-ray14 disease labels were derived from radiology reports and may contain label noise. Model performance therefore reflects classification relative to the available dataset labels rather than definitive clinical diagnoses.

**Clinical interpretation**
This project is an educational machine learning study. The resulting models are not intended for clinical diagnosis or medical decision-making.

## Project Structure

```text
Atelectasis-and-Pneumonia/
├── README.md
├── data/                 # Dataset metadata / local data preparation
├── notebooks/            # Exploratory analysis and experiments
├── src/                  # Preprocessing, training and evaluation code
├── results/              # Generated metrics, figures and tables
└── requirements.txt
```

The repository structure will evolve as the project progresses.

## Team

HS1502 Group 3

* Sizhe
* Brandon
* Yilun
* Violet
* Thaddeus

## Status

🚧 **Work in progress**

Current stage: project design, dataset preparation, and model selection.

## Disclaimer

This project is conducted for educational and research purposes as part of **HS1502: Conceptual Introduction to Machine Learning** at the National University of Singapore.

It is not a medical device and must not be used for clinical diagnosis or treatment decisions.
