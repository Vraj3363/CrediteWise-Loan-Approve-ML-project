# CrediteWise-Loan-Approve-ML-project
# 🏦 CreditWise — Loan Approval Prediction System

A machine learning project to predict loan approval decisions for **SecureTrust Bank** using applicant financial, personal, and credit data.

---

## 📌 Problem Statement

SecureTrust Bank processes hundreds of loan applications daily using a slow, biased manual process where officers review income proofs, employment details, and credit history.

This project builds an intelligent ML system to **automatically predict whether a loan should be Approved or Rejected**, helping the bank make faster, consistent, and unbiased decisions.

**Two key challenges solved:**
- Good customers getting rejected → loss of business
- High-risk customers getting approved → financial losses

---

## 📂 Dataset

- **File:** `loan_approval_data.csv`
- **Size:** 1,000 applicants × 20 features
- **Target Column:** `Loan_Approved` (Yes = Approved, No = Rejected)
- **Class Distribution:** 298 Approved / 652 Rejected

| Column | Type | Description |
|---|---|---|
| Applicant_ID | Numeric | Unique identifier (dropped before training) |
| Applicant_Income | Numeric | Monthly income of applicant |
| Coapplicant_Income | Numeric | Monthly income of co-applicant |
| Employment_Status | Categorical | Salaried / Contract / Self-employed / Unemployed |
| Age | Numeric | Applicant age |
| Marital_Status | Categorical | Married / Single |
| Dependents | Numeric | Number of financial dependents |
| Credit_Score | Numeric | Bureau credit score |
| Existing_Loans | Numeric | Number of active loans |
| DTI_Ratio | Numeric | Debt-to-Income ratio |
| Savings | Numeric | Savings account balance |
| Collateral_Value | Numeric | Value of collateral pledged |
| Loan_Amount | Numeric | Requested loan amount |
| Loan_Term | Numeric | Loan duration in months |
| Loan_Purpose | Categorical | Business / Car / Home / Education / Personal |
| Property_Area | Categorical | Urban / Semi-Urban / Rural |
| Education_Level | Categorical | Graduate / Not Graduate |
| Gender | Categorical | Male / Female |
| Employer_Category | Categorical | Government / Private / Self |
| **Loan_Approved** | **Target** | **Yes = Approved, No = Rejected** |

---

## 🔧 Tech Stack

- **Language:** Python 3.x
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn
- **Notebook:** Jupyter Notebook

---

## 🔄 Project Pipeline

1. **Data Loading** — Load CSV with pandas, explore shape and types
2. **Missing Value Handling** — Mean imputation for numerical, Mode imputation for categorical columns
3. **Exploratory Data Analysis (EDA)** — Class balance, income distributions, boxplots, correlation heatmap
4. **Feature Encoding** — Label Encoding (Education_Level, target) + One-Hot Encoding (6 categorical columns)
5. **Feature Engineering** — Added `DTI_Ratio_sq` and `Credit_Score_sq` polynomial features
6. **Train/Test Split** — 80% train / 20% test, `random_state=42`
7. **Feature Scaling** — StandardScaler (fit on train, transform on both)
8. **Model Training & Evaluation** — 3 models × 2 rounds (before and after feature engineering)

---

## 📊 Model Results

### Round 1 — Baseline (Before Feature Engineering)

| Model | Precision | Recall | F1 Score | Accuracy |
|---|---|---|---|---|
| Logistic Regression | -- | -- | -- | -- |
| K-Nearest Neighbors (k=5) | -- | -- | -- | -- |
| **Naive Bayes (GaussianNB)** | **Best** | -- | -- | -- |

### Round 2 — After Feature Engineering

| Model | Precision | Recall | F1 Score | Accuracy |
|---|---|---|---|---|
| Logistic Regression | -- | -- | -- | -- |
| K-Nearest Neighbors (k=5) | -- | -- | -- | -- |
| Naive Bayes (GaussianNB) | -- | -- | -- | -- |

> ⚠️ Fill in your actual metric values from the notebook output before submitting.

**✅ Best Model: Naive Bayes** — Selected based on highest precision (minimizes false loan approvals / risky customers being approved).

---

## 📁 File Structure

```
creditwise-loan-approval/
├── credit_wise.ipynb         # Main Jupyter notebook
├── loan_approval_data.csv    # Dataset
├── requirements.txt          # Python dependencies
├── .gitignore                # Files to exclude from git
└── README.md                 # This file
```

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/creditwise-loan-approval.git
cd creditwise-loan-approval

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the notebook
jupyter notebook credit_wise.ipynb
```

---

## 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 🔑 Key Findings

- **Credit Score** is the strongest predictor of loan approval
- **DTI Ratio** (Debt-to-Income) is a strong negative predictor — higher DTI → more likely rejected
- **Savings balance** positively correlates with approval
- The dataset has class imbalance (~70% rejected) which affects model performance

---

## 👤 Author

**Your Name**  
📧 your.email@example.com  
🔗 [GitHub](https://github.com/YOUR_USERNAME) | [LinkedIn](https://linkedin.com/in/YOUR_PROFILE)

---

## 📄 License

This project is for educational purposes.
