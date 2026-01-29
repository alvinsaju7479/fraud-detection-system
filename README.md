# 🛡️ Fraud Detection System  
**Anomaly Detection + Supervised Machine Learning (Production-Style)**

An end-to-end **fraud detection platform** that combines **unsupervised anomaly detection** and **supervised machine learning**, with **probability scoring, threshold tuning, explainability readiness, and a FastAPI inference service**.

This project is designed to closely resemble **real-world fraud detection systems** used in **banks, payment providers, and fintech companies**.

---

## 🚀 Key Features

- 🔍 **Anomaly Detection** using Isolation Forest  
- 🤖 **Supervised Models** (Logistic Regression & XGBoost)  
- 📊 **Fraud Probability Scoring**
- ⚖️ **Cost-based Threshold Tuning**
- 🧠 **Explainability-ready (SHAP)**
- 🌐 **FastAPI Inference API**
- 📦 **Batch & Real-time Predictions**
- 🧱 **Production-grade project structure**

---

## 🧠 Why This Project Matters

Fraud detection is a **high-impact ML problem** where:

- Fraud cases are **extremely rare** (highly imbalanced data)
- Precision vs Recall trade-offs have **direct business cost**
- Decisions must be **explainable** for audits & compliance
- Thresholds must be **business-driven**, not arbitrary

This project demonstrates **how real fraud systems are designed end-to-end**, not just model training.

---

## 🏗️ System Architecture
```text
Raw Data
↓
Feature Engineering
↓
Train / Test Split
↓
Anomaly Detection (Isolation Forest)
↓
Supervised Models (LogReg / XGBoost)
↓
Probability Calibration
↓
Cost-based Threshold Selection
↓
Model Evaluation
↓
FastAPI Inference Service
```


---

## 📁 Project Structure

```text
fraud-detection-system/
├── data/
│   ├── raw/               # Raw data (ignored by Git)
│   └── processed/         # Processed parquet datasets
├── notebooks/             # EDA & experimental notebooks
├── src/
│   ├── data/              # Data ingestion & feature pipelines
│   ├── models/            # Model training, calibration & evaluation
│   ├── explain/           # SHAP explainability utilities
│   ├── monitoring/        # Data & model drift detection
│   ├── api/               # FastAPI inference service
│   └── utils/             # Logging, metrics & helpers
├── tests/                 # Unit & integration tests
├── models/                # Saved model artifacts
├── reports/               # Figures, SHAP plots & reports
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## 📊 Dataset

This project uses the **Credit Card Fraud Detection Dataset**.

- Source: Kaggle (ULB)
- Total transactions: **284,807**
- Fraud transactions: **492**
- Fraud rate: **~0.17%**

Due to **file size and licensing restrictions**, the dataset is **not included in this repository**.

---

### 📥 How to get the data

1. Download the dataset from Kaggle:  
   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

2. Place the CSV file in the following path:
    data/raw/creditcard.csv

3. Ensure the folder structure looks like:
data/
└── raw/
└── creditcard.csv

---

## ⚙️ How to Run the Project

### 1️⃣ Install dependencies
pip install -r requirements.txt
### 2️⃣ Build processed dataset
python -m src.data.make_dataset
python -m src.data.split
### 3️⃣ Train Models
```bash
python -m src.models.train_isolation_forest
python -m src.models.train_supervised
python -m src.models.calibrate
python -m src.models.evaluate
```
Trained models and thresholds will be saved in:
```text
models/
```
## 🌐 Run the API
```text
Start FastAPI server
uvicorn src.api.main:app --reload
Open Swagger UI
http://127.0.0.1:8000/docs
```
## 🛠️ Tech Stack
```text

Python

Pandas / NumPy

scikit-learn

XGBoost

SHAP

FastAPI

Uvicorn

Joblib

Git & GitHub
```
