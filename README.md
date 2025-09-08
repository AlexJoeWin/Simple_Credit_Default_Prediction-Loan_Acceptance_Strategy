# 💼 Loan Default Prediction with XGBoost

This project trains a gradient boosting model to predict loan default risk using preprocessed credit data. It evaluates model performance through classification metrics, calibration analysis, and financial impact simulation — helping assess how loan acceptance strategies affect portfolio value.

---

## 📊 Problem Statement

Predict whether a loan will default (`loan_status = 1`) or not (`loan_status = 0`) based on borrower and loan features. The goal is to build a well-calibrated model that supports risk-aware lending decisions.

---

## 🚀 Workflow Overview

1. **Data Loading & Preparation**
   - Uses cleaned dataset: `cr_loan_preprocessed.csv`
   - Splits into training and test sets with stratification

2. **Model Training**
   - Trains an `XGBClassifier` with 60 boosting rounds
   - Predicts class probabilities and labels

3. **Evaluation**
   - Prints classification report with macro F1 score
   - Accepts model if F1 score > 0.85

4. **Calibration Curve**
   - Plots predicted vs. actual default probabilities
   - Accepts calibration if Brier Score < 0.1

5. **Expected Loss Simulation**
   - Calculates expected loss per loan using:
     - Predicted default probability
     - Loan amount
     - Fixed loss given default (LGD = 20%)

6. **Acceptance Strategy Analysis**
   - Simulates loan acceptance at various thresholds (50%–95%)
   - Computes bad rate and total portfolio value
   - Identifies optimal acceptance rate for maximum value

---

## 📈 Key Outputs

- **Classification Report**: Precision, recall, F1 score
- **Calibration Curve**: Visual check of probability reliability
- **Expected Loss**: Total loss in € millions
- **Acceptance Strategy Plot**: Total portfolio value vs. acceptance rate
- **Optimal Strategy**: Acceptance rate that maximizes value

---

## 🧠 Dependencies

```bash
pandas
numpy
matplotlib
seaborn
xgboost
scikit-learn
```

---

## 📂 File Structure

```
├── cr_loan_preprocessed.csv                     # Cleaned and preprocessed input data
├── simple_credit_default_prediction.ipynb       # Main script
├── README.md                                    # Project overview
```

---

## 🛠️ How to Run



Make sure `cr_loan_preprocessed.csv` is in the same directory.

---

## 📌 Notes

- The model assumes a fixed LGD of 20% for expected loss calculations.
