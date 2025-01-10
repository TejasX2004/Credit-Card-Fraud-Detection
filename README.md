# Credit Card Fraud Detection: Exploratory Data Analysis (EDA) and Machine Learning Models

This repository presents an analysis and predictive modeling workflow for detecting credit card fraud. The project includes exploratory data analysis (EDA), handling imbalanced datasets using **SMOTE (Synthetic Minority Oversampling Technique)**, and building predictive models using five machine learning algorithms.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [EDA](#eda)
- [Handling Imbalanced Data](#handling-imbalanced-data)
- [Machine Learning Models](#machine-learning-models)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview
Credit card fraud is a significant challenge in the financial sector. The primary goal of this project is to:
1. Perform **Exploratory Data Analysis (EDA)** to understand the dataset.
2. Address the class imbalance problem using **SMOTE**.
3. Evaluate and compare the performance of five machine learning models:
   - Logistic Regression
   - Support Vector Classifier (SVC)
   - Decision Tree Classifier
   - Random Forest Classifier
   - K-Nearest Neighbors (KNN)

---

## Dataset
The dataset used in this project is a publicly available credit card fraud detection dataset. It contains the following:
- **Features**: Numerical data (principal components obtained through PCA).
- **Target**: Binary variable (0 for legitimate transactions, 1 for fraudulent transactions).
- **Imbalance**: The dataset is highly imbalanced, with fraudulent transactions being significantly fewer than legitimate ones.

---

## EDA
Key steps performed in the EDA process:
- **Descriptive Statistics**: Summary statistics for numerical features.
- **Visualization**: Distribution of features, correlation heatmaps, and class distribution.
- **Insights**: Observations on trends, anomalies, and data quality.

---

## Handling Imbalanced Data
Given the severe class imbalance, **SMOTE** was applied to oversample the minority class and balance the dataset:
- **Before SMOTE**: Class 0 (~99%) vs Class 1 (~1%).
- **After SMOTE**: Classes were balanced to ensure equal representation.

---

## Machine Learning Models
The following models were trained and evaluated:
1. **Logistic Regression**: Baseline linear classifier.
2. **Support Vector Classifier (SVC)**: Effective for high-dimensional spaces.
3. **Decision Tree Classifier**: Interpretable tree-based model.
4. **Random Forest Classifier**: Ensemble model for improved performance.
5. **K-Nearest Neighbors (KNN)**: Distance-based classifier.

### Model Evaluation Metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **ROC-AUC Curve**

---

## Results
- **Best Model**: (Replace with your findings, e.g., Random Forest Classifier showed the highest F1-Score.)
- **Insights**: (Add any notable observations, e.g., Logistic Regression had faster training time but lower recall.)

---

## Installation
### Prerequisites
Ensure you have Python 3.8+ and the following libraries installed:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `imbalanced-learn`

Install the dependencies using:
```bash
pip install -r requirements.txt
