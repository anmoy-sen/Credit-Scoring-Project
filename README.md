# 💳 Enterprise Credit Scoring & Risk Underwriting Platform

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-FF4B4B.svg)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5+-F7931E.svg)](https://scikit-learn.org/)
[![DVC Versioned](https://img.shields.io/badge/DVC-v3.50+-945DD6.svg)](https://dvc.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, production-ready **Credit Risk Scoring & Underwriting System** engineered for financial institutions, loan officers, and risk analysts. This system moves beyond toy classification models to address the genuine economic and regulatory realities of lending:

1. **Cost-Sensitive Optimization**: Minimizes real financial default losses using an asymmetric cost matrix rather than an arbitrary 0.50 threshold.
2. **Standardized Credit Scoring**: Translates raw default probabilities into an intuitive **FICO-style credit score (300 – 850)** with automated risk-tier segmentation.
3. **Regulatory Explainable AI (XAI)**: Deploys **TreeSHAP** to extract compliant **Adverse Action notices** required by the *Equal Credit Opportunity Act (ECOA)* and *Fair Credit Reporting Act (FCRA)*.
4. **Dual Production Serving**: Provides both a high-throughput **FastAPI microservice** and an interactive **Streamlit Underwriting Portal** ready for immediate zero-Docker deployment to **Streamlit Community Cloud**.

---

## 📌 Table of Contents
- [Business Problem & Motivation](#-business-problem--motivation)
- [Key Features](#-key-features)
- [Machine Learning Architecture & Benchmarks](#-machine-learning-architecture--benchmarks)
- [Cost-Matrix Financial Optimization](#-cost-matrix-financial-optimization)
- [Regulatory Explainability (TreeSHAP)](#-regulatory-explainability-treeshap)
- [Interactive Streamlit Underwriting Portal](#-interactive-streamlit-underwriting-portal)
- [Production REST API (FastAPI)](#-production-rest-api-fastapi)
- [Streamlit Community Cloud Deployment Guide](#-streamlit-community-cloud-deployment-guide)
- [Local Setup & Step-by-Step Execution](#-local-setup--step-by-step-execution)
- [Data Pipeline & Reproducibility (DVC)](#-data-pipeline--reproducibility-dvc)
- [Project Structure](#-project-structure)

---

## 💼 Business Problem & Motivation

In consumer and commercial credit underwriting, defaulting accounts generate substantial financial losses (often 5× to 10× greater than the opportunity margin lost by declining an eligible borrower). Traditional machine learning applications in credit risk often fail in production because:

* **Symmetric Loss Assumption**: Standard models optimize for raw Accuracy or F1 score at a static $0.50$ cutoff. In credit lending, a False Negative (approving a borrower who defaults) destroys capital, whereas a False Positive (declining a good borrower) only forfeits interest income.
* **The "Black Box" Barrier**: Strict consumer protection regulations (such as FCRA § 615(a) and ECOA Regulation B) mandate that rejected applicants receive key substantive reasons explaining why their credit was denied.
* **Non-Standardized Outputs**: Raw decimal probabilities (e.g. $p = 0.237$) do not integrate cleanly into underwriting guidelines or legacy core banking systems accustomed to three-digit credit scores.

This project delivers a complete enterprise solution resolving all three challenges.

---

## ✨ Key Features

* **End-to-End Scikit-Learn Pipeline**: Fully stateless custom feature engineering transformer (`CreditFeatureTransformer`) chained with numerical scaling, one-hot encoding, and gradient/ensemble estimators.
* **Cost-Sensitive Threshold Tuning**: Automated threshold scan across business cost matrices identifying the financial loss-minimizing cutoff ($\tau^* \approx 0.39 - 0.40$).
* **Calibrated Credit Score Engine**: Converts default probabilities into calibrated credit scores ranging from **300 (Deep Subprime)** to **850 (Super Prime)**.
* **TreeSHAP Adverse Action Reasoning**: Automates regulatory compliance by isolating the top positive and negative risk contributors for every individual applicant.
* **Human-Centered Underwriting UI**: Streamlit dashboard presenting a structured **6-Month Financial Ledger** (Months 1–6: September down to April) with human-friendly repayment statuses instead of cryptic column abbreviations.
* **Batch Portfolio Underwriting**: Process thousands of applicants via CSV with interactive risk tier distributions, credit score histograms, and scored result exports.
* **Built-in Demo Portfolio**: Instant one-click testing in Streamlit Community Cloud with 15 pre-configured applicant records.

---

## 🔬 Machine Learning Architecture & Benchmarks

The model is trained on the **UCI Default of Credit Card Clients Dataset** (30,000 borrower profiles tracking 6 months of billing and repayment histories with a 22.1% baseline default rate).

### Feature Engineering (`src/features.py`)
From 23 raw financial variables, the pipeline synthesizes behavioral risk indicators:
1. **Delinquency Trend Aggregations**:
   - `late_payment_count`: Total months with payment delay $\ge 1$ month.
   - `severe_late_count`: Frequency of severe delinquency ($\ge 3$ months past due).
   - `max_delay`: Peak historical delay severity.
   - `avg_pay_status`: Trajectory index of delinquency over time.
2. **Financial Ratios & Velocity**:
   - `PAY_RATIO_1` to `PAY_RATIO_6`: Monthly payment-to-statement ratio: $\frac{\text{Payment Amount}_t}{\text{Statement Balance}_t + 1}$.
   - `avg_payment_ratio`: Six-month average payment consistency.
   - `avg_utilization`: Revolving credit line utilization rate: $\frac{\text{Average Balance}}{\text{Credit Limit} + 1}$.
   - Aggregated expenditure velocity (`avg_bill`, `max_bill`, `avg_payment`, `max_payment`).

### Model Evaluation & Champion Selection
Four distinct model architectures were trained and systematically benchmarked using Stratified K-Fold validation and cost-sensitive evaluation:

| Model Architecture | ROC-AUC | PR-AUC | Accuracy | Default Recall | Optimal Threshold ($\tau^*$) | Total Financial Cost |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🏆 **Random Forest (Champion)** | **0.7910** | **0.5712** | **78.1%** | **73.8%** (@ $\tau^*=0.39$) | **0.39** | **2,550.0** |
| 🥈 **CatBoost** | 0.7894 | 0.5713 | 76.8% | 75.8% (@ $\tau^*=0.40$) | 0.40 | 2,497.0 |
| 🥉 **LightGBM** | 0.7843 | 0.5625 | 76.9% | 78.7% (@ $\tau^*=0.37$) | 0.37 | 2,576.0 |
| 🎖️ **XGBoost** | 0.7836 | 0.5588 | 77.3% | 77.3% (@ $\tau^*=0.38$) | 0.38 | 2,557.0 |

> **Champion Selection Rationale**: While CatBoost demonstrated slightly lower raw financial cost, **Random Forest** achieved the highest overall discriminatory power (**ROC-AUC 0.7910**) and superior probability calibration across validation folds, making it the chosen production champion pipeline artifact (`models/champion_pipeline.pkl`).

---

## 💰 Cost-Matrix Financial Optimization

Standard ML metrics (like Accuracy) treat all misclassification errors equally. In credit risk:
* **False Negative (FN)**: Lending to an applicant who defaults $\rightarrow$ Loss of loan principal ($C_{\text{FN}} = 5 \times \text{unit cost}$).
* **False Positive (FP)**: Declining an eligible applicant $\rightarrow$ Loss of potential interest revenue ($C_{\text{FP}} = 1 \times \text{unit cost}$).

The total financial cost objective function is formulated as:
$$\text{Total Cost}(\tau) = C_{\text{FN}} \cdot \text{FN}(\tau) + C_{\text{FP}} \cdot \text{FP}(\tau)$$

```
Financial Loss vs. Decision Cutoff
Cost
 ^
 |    Standard 0.50 cutoff (Sub-optimal: Cost ~ 2,669)
 |       \
 |        \         Optimal Cutoff: tau* = 0.39 - 0.40
 |         \       /   (Minimizes Total Loss: Cost ~ 2,550)
 |          \     v
 |           \___/\___
 |                    \
 +--------------------------------------------------> Decision Threshold (tau)
 0.10             0.40     0.50                 0.90
```

Lowering the operational decision threshold from the standard $0.50$ down to **$0.39 - 0.40$** captures **73.8% of defaulting borrowers** (compared to only 61.8% at 0.50), significantly reducing the net portfolio loss rate.

---

## ⚖️ Regulatory Explainability (TreeSHAP)

To satisfy **FCRA § 615(a)** and **ECOA Regulation B**, every prediction is audited using TreeSHAP (`src/explainability.py`). When an application is declined or flagged for review, the system automatically isolates:

1. **Adverse Action Factors**: The top individual variables that drove the applicant's default probability upward (e.g., *Recent Payment Delays in September*, *High Credit Line Utilization*, *Elevated Delinquent Month Count*).
2. **Protective Factors**: Mitigating variables that positively influenced the profile (e.g., *High Payment-to-Bill Ratio*, *Low Utilization*).
3. **Decision Attribution Chart**: A high-resolution horizontal attribution plot visualising the exact direction and magnitude of feature contributions.

---

## 🖥️ Interactive Streamlit Underwriting Portal

The Streamlit portal (`app/dashboard.py`) provides an intuitive cockpit for underwriters:

### 1. Single Applicant Underwriting
* **Demographics & Credit Line**: Clean input cards for Credit Limit, Age, Gender, Education, and Marital Status.
* **Six-Month Financial Ledger**: Structured 6-month ledger separated into **Recent Quarter (Months 1–3: Sep, Aug, Jul)** and **Prior Quarter (Months 4–6: Jun, May, Apr)**.
  - Repayment timeliness presented with human-friendly descriptions (*Paid in full*, *Revolving balance*, *1 month delay*...) instead of cryptic column names.
  - Side-by-side comparison of **Statement Balance** and **Payment Received** for each billing period.
* **Credit Score Gauge & Risk Tiering**: Visual progress bar indexing the applicant on the 300–850 FICO scale with qualitative tiers:
  - 🟢 **Excellent (750 - 850)**: Super-prime tier, auto-approve.
  - 🟢 **Good (700 - 749)**: Low default risk.
  - 🟡 **Fair (650 - 699)**: Moderate risk, standard terms.
  - 🟠 **Poor (600 - 649)**: Manual underwriting review.
  - 🔴 **Very Poor (300 - 599)**: Subprime risk, immediate decline.
* **Automated Adverse Action Notice**: One-click generation of regulatory rejection notices backed by TreeSHAP.

### 2. Batch Portfolio Underwriting
* Ingest an entire portfolio via CSV.
* Includes a **Built-in Demo Portfolio (15 Applicants)** for instant testing.
* Generates aggregate credit score distributions, risk tier pie charts, summary approval rates, and downloadable scored CSV files.

---

## ⚡ Production REST API (FastAPI)

A high-performance asynchronous microservice (`app/main.py`) exposes programmatic scoring endpoints:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | API version and metadata |
| `GET` | `/health` | Health check & model artifact verification |
| `POST` | `/predict` | Real-time single applicant credit scoring |
| `POST` | `/batch_predict` | Portfolio batch scoring (list of applicants) |
| `POST` | `/explain` | TreeSHAP feature attributions & adverse action reasons |

### Interactive API Documentation
When running the FastAPI server, access the live interactive Swagger documentation at:
```
http://localhost:8000/docs
```

---

## ☁️ Streamlit Community Cloud Deployment Guide

Deploying this application to **Streamlit Community Cloud** is fully automated and requires **zero Docker or container configuration**.

### Prerequisites Prepared in this Repository:
* Production champion pipeline artifact (`models/champion_pipeline.pkl`) is tracked in Git.
* `packages.txt` is present at the repository root containing `libgomp1` (required for OpenMP gradient boosting support on Linux runners).
* `.streamlit/config.toml` configures production server and theme settings.
* Built-in sample portfolio (`data/sample_applicants.csv`) is included for instant testing.

### Deployment Steps:
1. **Push your repository to GitHub**:
   ```bash
   git add .
   git commit -m "Prepare repository for Streamlit Community Cloud deployment"
   git push origin main
   ```
2. **Connect to Streamlit Cloud**:
   - Navigate to [share.streamlit.io](https://share.streamlit.io/) and sign in with your GitHub account.
   - Click the **"New app"** button.
3. **Configure Deployment Settings**:
   - **Repository**: `YourUsername/Credit-Scoring-Project`
   - **Branch**: `main` (or your active branch)
   - **Main file path**: `app/dashboard.py` *(Ensure the path points to `app/dashboard.py`)*
4. **Advanced Settings**:
   - Under **Python version**, select **3.11** or **3.12**.
5. **Click "Deploy!"**:
   - Streamlit will automatically install dependencies from `requirements.txt`, configure the environment, and provide a live public URL (e.g. `https://<your-app>.streamlit.app`).

---

## 🛠️ Local Setup & Step-by-Step Execution

### 1. Clone the Repository & Create Virtual Environment
```bash
# Clone the repository
git clone https://github.com/YourUsername/Credit-Scoring-Project.git
cd Credit-Scoring-Project

# Create and activate virtual environment
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# On macOS/Linux:
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Launch the Streamlit Dashboard
```bash
streamlit run app/dashboard.py
```
The dashboard will open automatically in your browser at `http://localhost:8501`.

### 4. Launch the FastAPI Backend
```bash
uvicorn app.main:app --reload --port 8000
```
Open `http://localhost:8000/docs` to test the interactive REST API.

### 5. Run Automated Test Suite
```bash
pytest -v
```
Executes all unit and integration tests across API endpoints and explainability modules.

---

## 🔄 Data Pipeline & Reproducibility (DVC)

All data ingestion, feature preprocessing, model training, and evaluation stages are version-controlled via **DVC (Data Version Control)**:

```bash
# Reproduce the entire end-to-end ML pipeline
dvc repro

# View experiment metrics
dvc metrics show
```

The pipeline stages configured in `dvc.yaml`:
1. `prepare`: Ingests and splits raw credit records into stratified train/test partitions.
2. `train`: Fits candidate models (Random Forest, XGBoost, LightGBM, CatBoost) and logs evaluation metrics.
3. `evaluate`: Executes cost-matrix threshold tuning, selects the champion model, and generates ROC/PR curves and confusion matrices.

---

## 📁 Project Structure

```
Credit-Scoring-Project/
├── .dvc/                             # DVC metadata and cache configuration
├── .streamlit/
│   └── config.toml                   # Streamlit server & theme configuration
├── app/
│   ├── dashboard.py                  # Interactive Streamlit Underwriting UI
│   └── main.py                       # High-performance FastAPI REST service
├── data/
│   ├── raw/
│   │   ├── credit.csv.dvc            # DVC tracked raw dataset pointer
│   │   └── credit.csv                # UCI Credit Card Default dataset
│   ├── processed/                    # Stratified train and test partitions
│   └── sample_applicants.csv         # 15-record sample portfolio for UI demo
├── models/
│   └── champion_pipeline.pkl         # Production champion pipeline artifact
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb  # Feature extraction experiments
│   └── 03_model_training.ipynb       # Model comparison and threshold analysis
├── reports/
│   ├── metrics.json                  # Performance metrics across all models
│   ├── model_comparison.csv          # Benchmark comparative summary
│   └── plots/                        # ROC, PR, and Confusion Matrix figures
├── src/
│   ├── __init__.py
│   ├── evaluation.py                 # Cost matrix, FICO calibration & metrics
│   ├── explainability.py             # TreeSHAP adverse action module
│   ├── features.py                   # Stateless feature engineering transformer
│   ├── models.py                     # Model builders & hyperparameter definitions
│   ├── pipeline.py                   # End-to-end pipeline assembly
│   ├── predict.py                    # Production inference & batch scoring
│   ├── schemas.py                    # Pydantic v2 validation models
│   └── train.py                      # Training and artifact serialization script
├── tests/
│   ├── test_api.py                   # FastAPI integration test suite
│   └── test_explainability.py        # TreeSHAP unit tests
├── .gitignore                        # Git ignore rules (champion model whitelisted)
├── dvc.yaml                          # DVC multi-stage pipeline definition
├── packages.txt                      # Debian dependencies for Streamlit Cloud (libgomp1)
├── params.yaml                       # Pipeline hyperparameters & cost configuration
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## 📜 License
This project is open-source under the [MIT License](LICENSE).
