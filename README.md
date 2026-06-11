# ML Classification Project – Churn Prediction

## 📌 Overview
Supervised binary classification to predict customer churn, comparing **Logistic Regression** vs **Random Forest**.

## 📁 Repository Structure
```
ml_project/
├── ML_Classification_Notebook.ipynb   # Main notebook with code, plots & metrics
├── ml_classification.py               # Standalone Python script
├── confusion_matrices.png             # Confusion matrix plot
├── roc_curves.png                     # ROC curves comparison
├── metrics_comparison.png             # Bar chart of all metrics
├── feature_importances.png            # RF feature importance chart
├── cv_boxplots.png                    # Cross-validation score distributions
├── class_distribution.png            # Dataset class balance chart
├── test_metrics.csv                   # Test-set results table
├── cv_metrics.csv                     # Cross-validation results table
├── README.md                          # This file
└── ML_Classification_Report.docx      # Short written report
```

## ⚙️ Environment Setup

### Requirements
- Python 3.8+
- pip packages (see below)

### Install Dependencies
```bash
pip install scikit-learn pandas numpy matplotlib seaborn nbformat jupyter
```

### Run the Notebook
```bash
jupyter notebook ML_Classification_Notebook.ipynb
```

### Run as a Script
```bash
python ml_classification.py
```

## 📊 Key Results

| Metric | Logistic Regression | Random Forest |
|--------|-------------------|---------------|
| Accuracy | 0.860 | **0.955** |
| Precision | 0.754 | **0.982** |
| Recall | 0.803 | **0.869** |
| F1 Score | 0.778 | **0.922** |
| ROC-AUC | 0.904 | **0.967** |

**Selected model: Random Forest** — outperforms across all metrics with stable CV scores.

## 🔬 Methodology
1. Synthetic churn dataset (1000 samples, 10 features, 30% churn rate)
2. 80/20 stratified train-test split
3. StandardScaler for Logistic Regression; no scaling for RF
4. 5-fold Stratified Cross-Validation
5. Metrics: Accuracy, Precision, Recall, F1, ROC-AUC
# ml-classification-project
