"""
Data Preprocessing, Clinical Feature Metadata, and Data Quality Validation Module
for Heart Diseases Classification Academic System.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import os

# Clinical definitions, units, and display labels for the 13 input features
FEATURE_METADATA = {
    'age': {
        'label': 'Age',
        'unit': 'years',
        'type': 'numerical',
        'min': 18,
        'max': 100,
        'default': 55,
        'normal_range': '20 – 65 years',
        'description': 'Age of the patient in years'
    },
    'sex': {
        'label': 'Sex',
        'unit': '',
        'type': 'categorical',
        'options': {0: 'Female', 1: 'Male'},
        'default': 1,
        'description': 'Biological sex of the patient'
    },
    'cp': {
        'label': 'Chest Pain Type',
        'unit': '',
        'type': 'categorical',
        'options': {
            0: 'Typical Angina (0)',
            1: 'Atypical Angina (1)',
            2: 'Non-anginal Pain (2)',
            3: 'Asymptomatic (3)'
        },
        'default': 0,
        'description': 'Type of chest discomfort reported by patient'
    },
    'trestbps': {
        'label': 'Resting Blood Pressure',
        'unit': 'mm Hg',
        'type': 'numerical',
        'min': 80,
        'max': 220,
        'default': 130,
        'normal_range': '90 – 120 mm Hg',
        'description': 'Resting blood pressure upon admission to hospital'
    },
    'chol': {
        'label': 'Serum Cholesterol',
        'unit': 'mg/dl',
        'type': 'numerical',
        'min': 100,
        'max': 600,
        'default': 240,
        'normal_range': '< 200 mg/dl',
        'description': 'Serum cholesterol level'
    },
    'fbs': {
        'label': 'Fasting Blood Sugar > 120 mg/dl',
        'unit': '',
        'type': 'categorical',
        'options': {0: 'False (<= 120 mg/dl)', 1: 'True (> 120 mg/dl)'},
        'default': 0,
        'description': 'Fasting blood sugar exceeding 120 mg/dl threshold'
    },
    'restecg': {
        'label': 'Resting ECG Results',
        'unit': '',
        'type': 'categorical',
        'options': {
            0: 'Normal (0)',
            1: 'ST-T Wave Abnormality (1)',
            2: 'Left Ventricular Hypertrophy (2)'
        },
        'default': 1,
        'description': 'Resting electrocardiographic evaluation'
    },
    'thalach': {
        'label': 'Maximum Heart Rate Achieved',
        'unit': 'bpm',
        'type': 'numerical',
        'min': 60,
        'max': 220,
        'default': 150,
        'normal_range': '100 – 180 bpm',
        'description': 'Maximum heart rate recorded during cardiac stress test'
    },
    'exang': {
        'label': 'Exercise-Induced Angina',
        'unit': '',
        'type': 'categorical',
        'options': {0: 'No (0)', 1: 'Yes (1)'},
        'default': 0,
        'description': 'Presence of angina pectoris induced by physical exercise'
    },
    'oldpeak': {
        'label': 'ST Depression (Oldpeak)',
        'unit': 'mm',
        'type': 'numerical',
        'min': 0.0,
        'max': 6.5,
        'default': 1.0,
        'normal_range': '0.0 – 1.5 mm',
        'description': 'ST depression induced by exercise relative to resting state'
    },
    'slope': {
        'label': 'Slope of Peak Exercise ST',
        'unit': '',
        'type': 'categorical',
        'options': {
            0: 'Upsloping (0)',
            1: 'Flat (1)',
            2: 'Downsloping (2)'
        },
        'default': 1,
        'description': 'Slope shape of peak exercise ST segment'
    },
    'ca': {
        'label': 'Major Vessels Colored by Fluoroscopy',
        'unit': 'vessels',
        'type': 'categorical',
        'options': {0: '0 Vessels', 1: '1 Vessel', 2: '2 Vessels', 3: '3 Vessels'},
        'default': 0,
        'description': 'Number of major coronary vessels (0-3) visible under fluoroscopy'
    },
    'thal': {
        'label': 'Thalassemia Indicator',
        'unit': '',
        'type': 'categorical',
        'options': {
            1: 'Normal (1)',
            2: 'Fixed Defect (2)',
            3: 'Reversible Defect (3)'
        },
        'default': 2,
        'description': 'Thallium cardiac scintigraphy blood flow defect status'
    }
}

REQUIRED_FEATURES = list(FEATURE_METADATA.keys())
NUMERICAL_COLS = [k for k, v in FEATURE_METADATA.items() if v['type'] == 'numerical']
CATEGORICAL_COLS = [k for k, v in FEATURE_METADATA.items() if v['type'] == 'categorical']


def validate_dataset(df: pd.DataFrame, require_target: bool = False) -> dict:
    """
    Comprehensive Data Quality Audit for uploaded healthcare datasets.
    Audits schema compliance, missingness, duplicates, outliers, and invalid ranges.
    """
    issues = []
    warnings = []
    
    # Check shape
    total_records = len(df)
    total_features = df.shape[1]
    
    if total_records == 0:
        return {
            'status': 'Needs Attention',
            'is_valid': False,
            'total_records': 0,
            'total_features': total_features,
            'issues': ['Dataset is completely empty.'],
            'warnings': [],
            'missing_count': 0,
            'duplicate_count': 0,
            'column_stats': {}
        }

    # Normalize column names: strip whitespace and lowercase
    col_mapping = {col: str(col).strip().lower() for col in df.columns}
    normalized_cols = set(col_mapping.values())
    
    # Missing required columns
    missing_required = [col for col in REQUIRED_FEATURES if col not in normalized_cols]
    if missing_required:
        issues.append(f"Missing required clinical columns: {', '.join(missing_required)}")

    if require_target and 'target' not in normalized_cols:
        issues.append("Missing target classification column ('target') for model training.")

    # Duplicate records
    duplicate_count = int(df.duplicated().sum())
    if duplicate_count > 0:
        warnings.append(f"Detected {duplicate_count} duplicate row(s). These should be deduplicated during cleaning.")

    # Missing values
    missing_counts = df.isnull().sum()
    total_missing = int(missing_counts.sum())
    if total_missing > 0:
        cols_with_missing = missing_counts[missing_counts > 0].to_dict()
        warnings.append(f"Missing values detected across {len(cols_with_missing)} columns ({total_missing} total empty cells).")

    # Column-level validation & statistical inspection
    column_stats = {}
    for col in df.columns:
        norm_col = col_mapping[col]
        series = pd.to_numeric(df[col], errors='coerce')
        
        stat = {
            'original_name': col,
            'normalized_name': norm_col,
            'data_type': str(df[col].dtype),
            'missing_count': int(df[col].isnull().sum()),
            'unique_count': int(df[col].nunique()),
            'sample_values': df[col].dropna().head(3).tolist()
        }

        if norm_col in FEATURE_METADATA:
            meta = FEATURE_METADATA[norm_col]
            stat['label'] = meta['label']
            stat['unit'] = meta['unit']
            stat['type'] = meta['type']
            
            # Range check for numerical features
            if meta['type'] == 'numerical':
                min_val = series.min()
                max_val = series.max()
                stat['min'] = float(min_val) if pd.notnull(min_val) else None
                stat['max'] = float(max_val) if pd.notnull(max_val) else None
                stat['mean'] = float(series.mean()) if pd.notnull(series.mean()) else None
                
                if min_val is not None and min_val < 0:
                    issues.append(f"Invalid negative values detected in '{col}' (Minimum: {min_val}).")
                if norm_col == 'trestbps' and (min_val < 50 or max_val > 300):
                    warnings.append(f"Atypical blood pressure values detected in '{col}' (Range: {min_val} - {max_val}).")
                if norm_col == 'chol' and (min_val < 50 or max_val > 800):
                    warnings.append(f"Atypical cholesterol values detected in '{col}' (Range: {min_val} - {max_val}).")
        
        column_stats[norm_col] = stat

    is_valid = len(issues) == 0
    status = 'Valid' if is_valid else 'Needs Attention'

    return {
        'status': status,
        'is_valid': is_valid,
        'total_records': total_records,
        'total_features': total_features,
        'issues': issues,
        'warnings': warnings,
        'missing_count': total_missing,
        'duplicate_count': duplicate_count,
        'column_stats': column_stats
    }


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw dataset by mapping column names, handling missing values,
    and deduplicating rows while preserving data integrity.
    """
    cleaned = df.copy()
    cleaned.columns = [str(c).strip().lower() for c in cleaned.columns]
    
    # Retain patient_id if present
    patient_ids = None
    if 'patient_id' in cleaned.columns:
        patient_ids = cleaned['patient_id']
        cleaned = cleaned.drop(columns=['patient_id'])

    # Deduplicate
    cleaned = cleaned.drop_duplicates()
    
    # Cast all clinical features to numeric
    for col in cleaned.columns:
        if col in REQUIRED_FEATURES or col == 'target':
            cleaned[col] = pd.to_numeric(cleaned[col], errors='coerce')
    
    # Median imputation for numerical features
    num_imputer = SimpleImputer(strategy='median')
    avail_num = [c for c in NUMERICAL_COLS if c in cleaned.columns]
    if avail_num:
        cleaned[avail_num] = num_imputer.fit_transform(cleaned[avail_num])
        
    # Mode imputation for categorical features
    cat_imputer = SimpleImputer(strategy='most_frequent')
    avail_cat = [c for c in CATEGORICAL_COLS if c in cleaned.columns]
    if avail_cat:
        cleaned[avail_cat] = cat_imputer.fit_transform(cleaned[avail_cat])
        
    # Ensure binary target if present
    if 'target' in cleaned.columns:
        cleaned['target'] = (cleaned['target'] > 0).astype(int)
        
    if patient_ids is not None:
        cleaned.insert(0, 'patient_id', patient_ids.iloc[cleaned.index])

    return cleaned


def calculate_risk_category(probability: float) -> dict:
    """
    Calculates academic Model-Based Risk Category from predicted probability.
    NOTE: academic category only, not clinically validated.
    """
    prob_percent = round(probability * 100, 1)
    
    if probability < 0.35:
        return {
            'category': 'Low',
            'color': '#059669',     # Green
            'badge_bg': '#ecfdf5',
            'badge_border': '#a7f3d0',
            'label': 'Low Model-Based Risk',
            'description': 'Model probability is low. Routine lifestyle adherence recommended.'
        }
    elif probability < 0.65:
        return {
            'category': 'Moderate',
            'color': '#d97706',     # Amber
            'badge_bg': '#fffbeb',
            'badge_border': '#fde68a',
            'label': 'Moderate Model-Based Risk',
            'description': 'Model probability is moderate. Closer clinical monitoring suggested.'
        }
    else:
        return {
            'category': 'High',
            'color': '#dc2626',     # Red
            'badge_bg': '#fef2f2',
            'badge_border': '#fecaca',
            'label': 'High Model-Based Risk',
            'description': 'Model probability is elevated. Comprehensive cardiac evaluation indicated.'
        }
