"""
Machine Learning Training, Multi-Model Evaluation, and Inference Pipeline
for Heart Diseases Classification Academic System.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

from preprocessing import REQUIRED_FEATURES, clean_dataset, calculate_risk_category

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
MODEL_BUNDLE_PATH = os.path.join(MODEL_DIR, 'trained_model.pkl')
DEFAULT_DATASET_PATH = os.path.join(os.path.dirname(__file__), 'dataset', 'dataset.csv')


def train_all_models(data_path: str = DEFAULT_DATASET_PATH) -> dict:
    """
    Trains 5 standard ML models on the patient dataset, computes genuine evaluation
    metrics on a stratified holdout test split, and persists the bundle.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 1. Load and clean
    df = pd.read_csv(data_path)
    df_clean = clean_dataset(df)
    
    if 'target' not in df_clean.columns:
        raise ValueError("Dataset does not contain the required 'target' label column.")
        
    X = df_clean[REQUIRED_FEATURES]
    y = df_clean['target'].astype(int)
    
    # 2. Stratified Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # 3. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert scaled arrays back to DataFrames to preserve feature names for SHAP/LIME
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=REQUIRED_FEATURES, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=REQUIRED_FEATURES, index=X_test.index)
    
    # 4. Instantiate 5 Diverse Machine Learning Classifiers
    candidate_models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=120, max_depth=6, min_samples_split=4, random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000, C=1.0, solver='lbfgs', random_state=42
        ),
        'Support Vector Machine': CalibratedClassifierCV(
            SVC(kernel='rbf', C=1.0, random_state=42), ensemble=False
        ),
        'K-Nearest Neighbors': KNeighborsClassifier(
            n_neighbors=7, weights='distance'
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=5, min_samples_split=6, random_state=42
        )
    }
    
    evaluation_metrics = {}
    roc_data = {}
    confusion_matrices = {}
    
    for name, model in candidate_models.items():
        # Train
        model.fit(X_train_scaled_df, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled_df)
        
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test_scaled_df)[:, 1]
        else:
            y_proba = model.decision_function(X_test_scaled_df)
            y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())
            
        # Actual Calculated Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)
        
        evaluation_metrics[name] = {
            'Accuracy': float(acc),
            'Precision': float(prec),
            'Recall': float(rec),
            'F1 Score': float(f1),
            'ROC-AUC': float(auc)
        }
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        confusion_matrices[name] = cm.tolist()
        
        # ROC Curve points
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data[name] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'auc': float(auc)
        }
    
    # 5. Determine Best Model (by F1-Score & ROC-AUC)
    best_model_name = max(
        evaluation_metrics.keys(),
        key=lambda m: (evaluation_metrics[m]['F1 Score'] + evaluation_metrics[m]['ROC-AUC']) / 2
    )
    
    bundle = {
        'models': candidate_models,
        'metrics': evaluation_metrics,
        'roc_data': roc_data,
        'confusion_matrices': confusion_matrices,
        'best_model_name': best_model_name,
        'scaler': scaler,
        'feature_names': REQUIRED_FEATURES,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_scaled': X_train_scaled_df,
        'X_test_scaled': X_test_scaled_df,
        'training_records': len(df_clean),
        'split_info': {'train_size': len(X_train), 'test_size': len(X_test)}
    }
    
    # Save bundle
    joblib.dump(bundle, MODEL_BUNDLE_PATH)
    print(f"Model bundle successfully trained and saved to {MODEL_BUNDLE_PATH}")
    print(f"Top Model Selected: {best_model_name} (Acc: {evaluation_metrics[best_model_name]['Accuracy']:.3f}, AUC: {evaluation_metrics[best_model_name]['ROC-AUC']:.3f})")
    
    return bundle


def load_model_bundle(force_retrain: bool = False, dataset_path: str = DEFAULT_DATASET_PATH) -> dict:
    """
    Loads saved model bundle, or trains on default dataset if not yet built.
    """
    if not force_retrain and os.path.exists(MODEL_BUNDLE_PATH):
        try:
            return joblib.load(MODEL_BUNDLE_PATH)
        except Exception as e:
            print(f"Error loading bundle ({e}), retraining fresh models...")
            return train_all_models(dataset_path)
    else:
        return train_all_models(dataset_path)


def predict_single_patient(bundle: dict, input_dict: dict, selected_model_name: str = None) -> dict:
    """
    Executes model inference for a single patient record.
    Returns predicted class, model confidence probability, and model-based risk category.
    """
    if selected_model_name is None or selected_model_name not in bundle['models']:
        selected_model_name = bundle['best_model_name']
        
    model = bundle['models'][selected_model_name]
    scaler = bundle['scaler']
    
    # Build single-row DataFrame aligned with training schema
    row_df = pd.DataFrame([input_dict])[REQUIRED_FEATURES]
    
    # Scale features
    row_scaled = scaler.transform(row_df)
    row_scaled_df = pd.DataFrame(row_scaled, columns=REQUIRED_FEATURES)
    
    # Prediction
    pred = int(model.predict(row_scaled_df)[0])
    
    # Probability estimation
    if hasattr(model, 'predict_proba'):
        proba = float(model.predict_proba(row_scaled_df)[0][1])
    else:
        dfunc = float(model.decision_function(row_scaled_df)[0])
        proba = 1.0 / (1.0 + np.exp(-dfunc))
        
    # Model confidence (probability of the predicted class or positive class)
    disease_label = "Heart Disease Detected (Model Classification)" if pred == 1 else "No Heart Disease Detected"
    risk_info = calculate_risk_category(proba)
    
    return {
        'prediction': pred,
        'prediction_label': disease_label,
        'probability': proba,
        'confidence_percentage': round(proba * 100, 1),
        'risk_category': risk_info['category'],
        'risk_color': risk_info['color'],
        'risk_badge_bg': risk_info['badge_bg'],
        'risk_badge_border': risk_info['badge_border'],
        'risk_label': risk_info['label'],
        'risk_description': risk_info['description'],
        'model_used': selected_model_name,
        'input_features': input_dict,
        'scaled_features_df': row_scaled_df
    }


def predict_batch_patients(bundle: dict, df_patients: pd.DataFrame, selected_model_name: str = None) -> pd.DataFrame:
    """
    Runs batch classification across an uploaded dataset containing multiple patient records.
    """
    if selected_model_name is None or selected_model_name not in bundle['models']:
        selected_model_name = bundle['best_model_name']
        
    model = bundle['models'][selected_model_name]
    scaler = bundle['scaler']
    
    cleaned = clean_dataset(df_patients)
    X = cleaned[REQUIRED_FEATURES]
    
    X_scaled = scaler.transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=REQUIRED_FEATURES)
    
    preds = model.predict(X_scaled_df)
    
    if hasattr(model, 'predict_proba'):
        probas = model.predict_proba(X_scaled_df)[:, 1]
    else:
        dfunc = model.decision_function(X_scaled_df)
        probas = 1.0 / (1.0 + np.exp(-dfunc))
        
    result_df = cleaned.copy()
    
    # Format results
    result_df['Prediction'] = preds
    result_df['Classification'] = [
        "Heart Disease" if p == 1 else "No Heart Disease" for p in preds
    ]
    result_df['Model Confidence (%)'] = np.round(probas * 100, 1)
    
    risk_cats = []
    for pb in probas:
        rc = calculate_risk_category(pb)
        risk_cats.append(rc['category'])
    result_df['Model-Based Risk Category'] = risk_cats
    
    return result_df
