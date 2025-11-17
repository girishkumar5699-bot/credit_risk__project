# 🏦 Credit Risk Prediction & Interpretability Project

## 📌 Overview
This project builds a robust machine learning pipeline to predict loan default risk using a credit dataset. It combines high-performance modeling with interpretable AI techniques to ensure transparency and fairness in financial decision-making.

## 🎯 Objectives
- Predict loan default risk using a tuned LightGBM classifier
- Analyze global and local feature importance using SHAP and LIME
- Audit fairness and detect potential bias in model decisions
- Deliver stakeholder-ready explanations for borderline cases

## 📁 Project Structure
├── bestmodel.pkl               # Trained LightGBM model ├── explanations/              # SHAP & LIME HTML explanations for 10 borderline cases │   ├── shap_instance_0.html │   ├── lime_instance_0.html │   └── ... ├── credit_risk_notebook.ipynb # Full pipeline: preprocessing, modeling, interpretability ├── README.md                  # Project summary and instructions

## 🧠 Model Performance

| Metric     | Value  |
|------------|--------|
| AUC        | 1.000  |
| Precision  | 1.000  |
| Recall     | 0.944  |

The model demonstrates excellent predictive power on the test set. Further validation is recommended to confirm generalizability.

## 🔍 Global Feature Importance (SHAP)

| Feature           | SHAP Importance |
|-------------------|-----------------|
| loan_amount       | 4.92            |
| income            | 4.70            |
| credit_score      | 0.28            |
| age               | 0.24            |
| employment_type   | 0.17            |
| loan_term         | 0.08            |
| gender            | 0.03            |

## 🧪 Local Explanations
- 10 borderline prediction instances (probability near 0.5) were selected
- SHAP force plots and LIME HTML explanations were generated for each
- Files are saved in the `explanations/` folder

## ⚖️ Fairness & Bias Audit
- Gender and employment_type were included as categorical features
- SHAP and LIME revealed that income and loan_amount dominate predictions
- Employment_type occasionally influenced borderline cases
- Recommendation: conduct fairness testing using disparate impact or equal opportunity metrics

## 🚀 How to Run
1. Open `credit_risk_notebook.ipynb` in Jupyter
2. Run all cells to reproduce results and regenerate explanations
3. View saved HTML files in the `explanations/` folder

## 📦 Requirements
- Python 3.8+
- lightgbm
- shap
- lime
- pandas, numpy, scikit-learn, matplotlib

Install dependencies:
```bash
pip install lightgbm shap lime pandas scikit-learn matplotlib
