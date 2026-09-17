# Machine Learning for Radiographic Differential Diagnosis of Atelectasis and Pneumonia Across Age Groups

**HS1502 — Conceptual Introduction to Machine Learning**
National University of Singapore

## Overview

Atelectasis and pneumonia can present with similar features on chest X-rays, potentially making radiographic differentiation challenging. Patient age may also influence these radiographic presentations and therefore affect the performance of machine learning models attempting to distinguish between the two conditions.

This project investigates whether different machine learning approaches perform differently when distinguishing **atelectasis from pneumonia across younger and older patient populations**.

Using the **NIH ChestX-ray14 dataset**, we will compare several machine learning models on two age groups and evaluate whether the most effective modelling approach differs between them.

## Research Question

**Does the performance of machine learning models in differentiating atelectasis from pneumonia on chest X-rays vary across age groups?**

We aim to investigate:

* How well different machine learning models distinguish atelectasis from pneumonia.
* Whether their performance changes between younger and older patients.
* Whether different models are better suited to different age groups.
* What characteristics of the models, data, or radiographic presentation might explain any observed differences.

## Dataset

The project uses the **NIH ChestX-ray14 dataset**, which contains frontal chest X-ray images together with disease labels and patient metadata.

For this project, we will focus on cases containing:

* **Atelectasis**
* **Pneumonia**

The dataset will then be divided according to patient age for separate model evaluation.

## Planned Methodology

> **Note:** The methodology below represents our current project plan and may change as we explore the dataset and evaluate the feasibility of each approach.

### 1. Data Cleaning and Preparation

The dataset will first be filtered and prepared for the classification task.

Current plan:

1. Retain relevant atelectasis and pneumonia cases.
2. Establish how multi-label cases will be handled.
3. Divide patients into two age groups:

   * **< 65 years old**
   * **≥ 65 years old**
4. Examine the distribution of atelectasis and pneumonia within each group.
5. Address class imbalance where necessary.
6. Create training, validation, and test splits while preventing patient-level data leakage.
7. Perform image preprocessing required by the different models.

### 2. Convolutional Neural Network (CNN)

Develop a CNN that learns image features directly from chest X-rays and performs binary classification between atelectasis and pneumonia.

### 3. Feature Extraction + Traditional Classification

Explore extracting informative representations from the X-ray images before applying a traditional machine learning classifier.

This provides a comparison between end-to-end image classification and approaches that separate feature extraction from classification.

### 4. Vision Transformer (ViT)

Develop or adapt a Vision Transformer approach for the same classification task, allowing comparison between transformer-based and convolutional approaches.

### 5. Support Vector Machine (SVM)

Train an SVM using suitable engineered or extracted image features as a traditional machine learning approach to the classification problem.

## Experimental Design

Each model will be evaluated on both age groups:

| Model                           | < 65 | ≥ 65 |
| ------------------------------- | ---- | ---- |
| CNN                             | ✓    | ✓    |
| Feature extraction + classifier | ✓    | ✓    |
| Vision Transformer              | ✓    | ✓    |
| SVM                             | ✓    | ✓    |

This allows comparison both **across models within an age group** and **across age groups for the same model**.

## Evaluation

Model performance will be assessed using multiple classification metrics.

### Confusion Matrix

For each model and age group, we will examine:

* True Positives (TP)
* True Negatives (TN)
* False Positives (FP)
* False Negatives (FN)

### False Negatives

Particular attention will be paid to false negatives given their potential importance in a medical classification setting. An appropriate rate, rather than simply the raw number of false negatives, will be used when comparing datasets of different sizes.

### ROC and AUC

Receiver Operating Characteristic (ROC) curves and **Area Under the Curve (AUC)** will be used to evaluate how well each model distinguishes between the two classes across classification thresholds.

Additional metrics such as accuracy, precision, recall/sensitivity, specificity, and F1-score may also be considered where appropriate.

## Analysis

After evaluating the models, we will investigate whether model performance differs systematically between the two age groups.

Where differences are observed, we will consider possible explanations including:

* Differences in radiographic presentation across age groups
* Disease and class distributions
* Image characteristics
* Features learned or extracted by different models
* Differences between convolutional, transformer-based, and feature-based approaches
* Dataset size and composition

These explanations will be treated as interpretations of the observed results rather than evidence of clinical causation.

## Key Experimental Considerations

### Class Imbalance

The number of atelectasis and pneumonia cases may differ substantially. Appropriate sampling strategies, class weighting, and evaluation metrics may therefore be necessary.

### Sample Size

Splitting the dataset by age reduces the amount of data available within each subgroup. Each group must contain sufficient examples of both conditions for meaningful comparison.

### Patient-Level Data Leakage

Multiple images may belong to the same patient. Images from the same patient should not appear across training and test sets, as this could artificially inflate measured performance.

### Label Definition

A clear inclusion/exclusion rule will be established for images containing both atelectasis and pneumonia or additional disease labels before model training.

### Clinical Interpretation

Differences in model performance across age groups do not by themselves establish that age causes differences in the radiographic presentation of either disease.

## Team

**HS1502 Group 3**

* Sizhe
* Brandon
* Yilun
* Violet
* Thaddeus

## Project Status

**Work in progress**

The methodology, age thresholds, model architectures, preprocessing pipeline, and evaluation procedure remain subject to refinement as the project progresses.

## Disclaimer

This project is conducted for educational purposes as part of **HS1502: Conceptual Introduction to Machine Learning** at the National University of Singapore.

The models developed in this project are experimental and are **not intended for clinical diagnosis or medical decision-making**.
