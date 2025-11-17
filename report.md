# 📊 Credit Risk Prediction & Interpretability Report

## 1. Project Overview
This project aims to predict loan default risk using a credit dataset and explain model decisions using interpretable AI techniques. A LightGBM classifier was trained and tuned to achieve high predictive performance, and SHAP and LIME were used to analyze both global and local feature importance.

## 2. Data Preparation
- Missing values were handled appropriately
- Categorical features (e.g., gender, employment_type) were encoded
- Feature scaling and preprocessing were applied where necessary
- Train-test split ensured unbiased evaluation

## 3. Model Training
- **Model used**: LightGBM classifier
- **Validation strategy**: Cross-validation and test set evaluation
- **Performance metrics**:
  - AUC: 1.000
  - Precision: 1.000
  - Recall: 0.944

## 4. Global Feature Importance (SHAP)
SHAP summary plots revealed the following ranked importance:

| Feature           | SHAP Importance |
|-------------------|-----------------|
| loan_amount       | 4.92            |
| income            | 4.70            |
| credit_score      | 0.28            |
| age               | 0.24            |
| employment_type   | 0.17            |
| loan_term         | 0.08            |
| gender            | 0.03            |

Loan amount and income were the most influential features globally.

## 5. Local Explanations (SHAP & LIME)
- 10 borderline prediction instances (probability near 0.5) were selected
- SHAP force plots and LIME explanations were generated for each
- HTML files saved in the `explanations/` folder
- These explanations highlight how individual features influenced specific predictions

## 6. Fairness & Bias Analysis
- Gender and employment_type were included as categorical features
- SHAP and LIME revealed that income and loan_amount dominate predictions
- Employment_type occasionally influenced borderline cases
- Recommendation: Conduct fairness testing using disparate impact or equal opportunity metrics to ensure non-discriminatory decision-making

## 7. Deliverables
- `bestmodel.pkl`: Trained LightGBM model
- `explanations/`: SHAP and LIME HTML files for 10 instances
- `credit_risk_notebook.ipynb`: Full pipeline with code and outputs
- `README.md`: Project summary and instructions
- `requirements.txt`: Python dependencies

## 8. Conclusion
This project demonstrates a complete and interpretable credit risk prediction pipeline. The model performs exceptionally well and is supported by transparent explanations. Fairness considerations have been flagged for further testing to ensure ethical deployment.
