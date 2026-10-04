# HEART DISEASES CLASSIFICATION
### Machine Learning Based Patient Data Analysis and Classification System

An academic healthcare analytics web application designed for analyzing patient medical data, performing automated data quality validation, classifying heart disease risk using benchmarked Machine Learning models, providing feature-level transparency with Explainable AI (SHAP), and generating professional clinical analytical reports.

---

> ### ⚠️ Mandatory Academic Research Notice
> **"Model-generated result for academic purposes only. This system is not a medical diagnostic tool."**  
> All probabilistic scores are presented as **Model Confidence / Predicted Probability** and **Model-Based Risk Category** (Low / Moderate / High). This software does not provide certified medical diagnoses or replace physician consultation.

---

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset & Clinical Features](#dataset--clinical-features)
3. [Key Application Modules](#key-application-modules)
4. [Machine Learning Pipeline](#machine-learning-pipeline)
5. [Explainable AI (SHAP)](#explainable-ai-shap)
6. [Design & User Experience](#design--user-experience)
7. [Project File Structure](#project-file-structure)
8. [Setup & Installation](#setup--installation)
9. [Running the Application](#running-the-application)

---

## 🩺 Project Overview
Coronary artery disease and cardiovascular ailments represent the leading cause of global morbidity. This project implements a modern clinical decision support prototype developed for academic review (B.Tech Information Technology).

Unlike rudimentary prediction forms, this platform provides:
* **Automated Data Quality Audit**: Ingestion of CSV and Excel datasets with null checks, duplicate detection, and physiological range verification.
* **Exploratory Clinical Analytics**: Interactive visualizations including distribution curves, feature correlation matrices, and biomarker comparisons.
* **Benchmarked Machine Learning**: Comparison of 5 classifiers with authentic holdout evaluation metrics (Accuracy, Precision, Recall, F1 Score, ROC-AUC).
* **Single & Batch Patient Classification**: Interactive medical input forms alongside cohort-wide automated batch inference.
* **Explainable AI (XAI)**: SHAP attribution waterfalls explaining the exact mathematical influence of each symptom or vital sign.
* **Exportable PDF Reports**: Formal patient analytical summaries formatted with ReportLab Platypus.

---

## 📊 Dataset & Clinical Features
The system utilizes the gold-standard **UCI Cleveland Heart Disease Dataset** (303 patient records):

| Feature Name | Clinical Description | Unit / Scale | Normal Reference |
| :--- | :--- | :--- | :--- |
| **`age`** | Patient age in years | Years (18–100) | 20–65 yrs |
| **`sex`** | Biological sex | 1 = Male; 0 = Female | Demographic |
| **`cp`** | Chest Pain Type | 0: Typical Angina, 1: Atypical Angina, 2: Non-anginal, 3: Asymptomatic | Clinical History |
| **`trestbps`** | Resting Blood Pressure | mm Hg | 90–120 mm Hg |
| **`chol`** | Serum Cholesterol | mg/dl | < 200 mg/dl |
| **`fbs`** | Fasting Blood Sugar > 120 mg/dl | 1 = True; 0 = False | Normal: <= 120 |
| **`restecg`** | Resting Electrocardiogram | 0: Normal, 1: ST-T Abnormality, 2: LV Hypertrophy | Cardiac Vitals |
| **`thalach`** | Maximum Heart Rate Achieved | bpm | 100–180 bpm |
| **`exang`** | Exercise-Induced Angina | 1 = Yes; 0 = No | Stress Test |
| **`oldpeak`** | ST Depression (exercise relative to rest) | mm (0.0 – 6.2) | 0.0 – 1.5 mm |
| **`slope`** | Slope of Peak Exercise ST | 0: Upsloping, 1: Flat, 2: Downsloping | Stress Test |
| **`ca`** | Number of Major Vessels Colored | 0 to 3 vessels | 0 (No stenosis) |
| **`thal`** | Thallium Scintigraphy Blood Flow | 1: Normal, 2: Fixed Defect, 3: Reversible Defect | Perfusion Scan |
| **`target`** | Heart Disease Ground Truth | 1 = Disease Present; 0 = No Disease | Academic Label |

---

## 🖥️ Website Structure & Pages
1. **🏠 Home**: Modern landing interface with hero graphics, key cohort metrics, and 7-step workflow timeline.
2. **📂 Dataset Upload**: Drag-and-drop CSV/XLSX uploader, validation audit, and instant model retraining.
3. **📊 Data Analysis**: Exploratory dashboard featuring correlation heatmaps, age histograms, and clinical biomarker comparisons.
4. **🧠 ML Model**: Pipeline diagram, comparative holdout metrics table, confusion matrices, and ROC curves for 5 algorithms.
5. **🩺 Patient Classification**:
   - *Option A: Manual Entry*: Medical input form with clinical presets, model confidence gauge, and risk category badges.
   - *Option B: Batch Analysis*: Cohort classification table with search, category filtering, and drill-down inspection.
6. **🔍 Explainable AI ("Why This Prediction?")**: SHAP feature attribution waterfall, positive/negative influence indicators, and clinical narrative summaries.
7. **📄 Patient Report**: Formal patient summary with one-click **PDF Report Download** and batch CSV export.

---

## 🔬 Machine Learning Pipeline
1. **Data Cleaning & Imputation**: Handles missing values via median imputation (numerical) and mode imputation (categorical).
2. **Standardization**: Scales numerical features (`StandardScaler`) to prevent gradient divergence.
3. **Stratified Split**: 80% training / 20% holdout test split preserving class balance.
4. **Classifiers Evaluated**:
   - **Random Forest Classifier** (Selected ensemble performer)
   - **Logistic Regression** (L2 Regularized)
   - **Support Vector Machine** (RBF kernel with calibrated probability output)
   - **K-Nearest Neighbors** (Distance-weighted k=7)
   - **Decision Tree** (Max depth 5)
5. **No Hardcoded Values**: All metrics (Accuracy, Precision, Recall, F1, ROC-AUC) are computed dynamically from actual test set predictions.

---

## 💡 Explainable AI (SHAP)
To address the "black-box" dilemma in medical machine learning, this system integrates **SHAP (SHapley Additive exPlanations)** based on cooperative game theory:
$$\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f(S \cup \{i\}) - f(S) \right]$$
* **Positive SHAP Values (Red)**: Physiological parameters that pushed the classification toward Heart Disease (elevated risk).
* **Negative SHAP Values (Green)**: Protective vitals that pushed the classification toward No Heart Disease.

---

## 📁 Project File Structure
```
heart_disease_classification/
├── app.py                     # Main Streamlit application and page orchestrator
├── .streamlit/
│   └── config.toml            # Startup light theme and application colors
├── preprocessing.py           # Feature definitions, validation audits, and data cleaning
├── train_model.py             # Multi-model training, holdout evaluation, and inference
├── explainability.py          # SHAP local and global attribution engine
├── report_generator.py        # PDF & CSV medical analysis report generator (ReportLab)
├── utils/
│   ├── __init__.py
│   ├── styles.py              # Scoped light healthcare CSS theme & typography
│   └── ui_components.py       # Reusable cards, SVG graphics, and Plotly charts
├── dataset/
│   ├── dataset.csv            # UCI Cleveland Heart Disease benchmark (303 records)
│   ├── sample_batch_patients.csv # Ready-to-upload patient batch with IDs
│   └── test_corrupted_sample.csv # Sample with missing values for quality checks
├── models/
│   └── trained_model.pkl      # Pre-trained models and scaler bundle
├── requirements.txt           # Project dependencies
└── README.md                  # Comprehensive academic documentation
```

---

## 🚀 Setup & Installation

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.10, 3.11, 3.12, 3.13, 3.14)
- Pip package manager

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Launch the Streamlit web application:
```bash
streamlit run app.py
```

Once running, the application will automatically open in your default browser at:
`http://localhost:8501`

---

## 🎓 Academic Presentation Tips
- **For Reviewers & Examiners**: Highlight that models are trained on real data without hardcoded performance metrics.
- **Explainable AI**: Emphasize how SHAP Bridges the gap between ML accuracy and clinical trust.
- **Safety Standard**: Point out that the interface strictly distinguishes mathematical model confidence from medical diagnosis.
