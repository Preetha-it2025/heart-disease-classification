"""
Explainable AI (XAI) Module powered by SHAP (SHapley Additive exPlanations)
for Heart Diseases Classification Academic System.
"""

import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance
from preprocessing import FEATURE_METADATA, REQUIRED_FEATURES


class ModelExplainer:
    """
    Computes global and local Explainable AI metrics using SHAP and model introspection.
    Provides feature attribution values, importance rankings, and clinical narrative summaries.
    """
    
    def __init__(self, bundle: dict):
        self.bundle = bundle
        self.feature_names = REQUIRED_FEATURES
        self.scaler = bundle['scaler']
        self.X_train_scaled = bundle['X_train_scaled']
        self.X_test_scaled = bundle['X_test_scaled']
        self.y_test = bundle['y_test']
        self.best_model_name = bundle['best_model_name']
        self.model = bundle['models'][self.best_model_name]
        
        # Initialize background sample for SHAP explainers (50 representative samples for speed)
        self.background_sample = shap.sample(self.X_train_scaled, 50, random_state=42)
        self._init_explainer()

    def _init_explainer(self):
        """Initializes appropriate SHAP explainer based on model family."""
        try:
            if hasattr(self.model, "estimators_") or "Tree" in type(self.model).__name__:
                # Tree models (Random Forest, Decision Tree)
                self.explainer = shap.TreeExplainer(self.model)
                self.explainer_type = 'tree'
            elif hasattr(self.model, "coef_"):
                # Linear models (Logistic Regression)
                self.explainer = shap.LinearExplainer(self.model, self.background_sample)
                self.explainer_type = 'linear'
            else:
                # Kernel / Sampling Explainer for KNN, Calibrated SVC, etc.
                def predict_fn(X):
                    if hasattr(self.model, 'predict_proba'):
                        return self.model.predict_proba(X)[:, 1]
                    return self.model.predict(X)
                self.explainer = shap.KernelExplainer(predict_fn, self.background_sample)
                self.explainer_type = 'kernel'
        except Exception as e:
            print(f"Fallback to Sampling Explainer due to: {e}")
            def predict_fn(X):
                if hasattr(self.model, 'predict_proba'):
                    return self.model.predict_proba(X)[:, 1]
                return self.model.predict(X)
            self.explainer = shap.KernelExplainer(predict_fn, self.background_sample)
            self.explainer_type = 'kernel'

    def get_global_feature_importance(self, top_n: int = 13) -> pd.DataFrame:
        """
        Calculates global feature importance ranking across the dataset.
        Uses Tree feature importances, permutation importance, or mean absolute SHAP values.
        """
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        else:
            # Fallback to permutation importance
            perm = permutation_importance(
                self.model, self.X_test_scaled, self.y_test, n_repeats=5, random_state=42
            )
            importances = perm.importances_mean
            # normalize
            importances = np.maximum(importances, 0)
            if np.sum(importances) > 0:
                importances = importances / np.sum(importances)

        df_importance = pd.DataFrame({
            'feature': self.feature_names,
            'label': [FEATURE_METADATA[f]['label'] for f in self.feature_names],
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)

        df_importance['percentage'] = (
            df_importance['importance'] / df_importance['importance'].sum() * 100
        ).round(1)

        return df_importance

    def explain_patient_prediction(self, patient_dict: dict) -> dict:
        """
        Generates patient-specific local attribution values using SHAP.
        Determines which features pushed towards or away from Heart Disease classification.
        """
        # Convert single patient input to scaled DataFrame
        row_raw = pd.DataFrame([patient_dict])[self.feature_names]
        row_scaled = self.scaler.transform(row_raw)
        row_scaled_df = pd.DataFrame(row_scaled, columns=self.feature_names)
        
        # Calculate SHAP values
        shap_values_raw = None
        base_value = 0.5
        
        try:
            if self.explainer_type == 'tree':
                sv = self.explainer.shap_values(row_scaled_df)
                if isinstance(sv, list):
                    # Binary classification returns [class_0, class_1]
                    shap_values_raw = sv[1][0] if len(sv) > 1 else sv[0][0]
                elif len(sv.shape) == 3:
                    shap_values_raw = sv[0, :, 1]
                else:
                    shap_values_raw = sv[0]
                
                # Base value
                ev = self.explainer.expected_value
                base_value = float(ev[1] if isinstance(ev, (list, np.ndarray)) and len(ev) > 1 else ev)
                
            elif self.explainer_type == 'linear':
                sv = self.explainer.shap_values(row_scaled_df)
                shap_values_raw = sv[0] if len(sv.shape) == 2 else sv
                ev = self.explainer.expected_value
                base_value = float(ev[1] if isinstance(ev, (list, np.ndarray)) and len(ev) > 1 else ev)
                
            else:
                # Kernel Explainer
                sv = self.explainer.shap_values(row_scaled_df, nsamples=40)
                shap_values_raw = sv[0]
                base_value = float(self.explainer.expected_value)
        except Exception as e:
            print(f"SHAP computation warning ({e}), falling back to perturbation attributions...")
            # Deterministic perturbation attribution fallback
            base_pred = self.model.predict_proba(row_scaled_df)[0][1]
            shap_values_raw = []
            for col in self.feature_names:
                perturbed = row_scaled_df.copy()
                perturbed[col] = 0.0 # set to population mean
                new_pred = self.model.predict_proba(perturbed)[0][1]
                shap_values_raw.append(base_pred - new_pred)
            shap_values_raw = np.array(shap_values_raw)

        # Assemble attributions table
        attributions = []
        for i, feat in enumerate(self.feature_names):
            val_raw = patient_dict.get(feat, 0)
            meta = FEATURE_METADATA.get(feat, {})
            shap_val = float(shap_values_raw[i])
            
            # Format raw value with unit or option text
            if meta.get('type') == 'categorical' and 'options' in meta:
                display_val = meta['options'].get(int(val_raw), str(val_raw))
            else:
                unit = meta.get('unit', '')
                display_val = f"{val_raw} {unit}".strip()

            impact = "Increases Risk" if shap_val > 0.005 else ("Decreases Risk" if shap_val < -0.005 else "Neutral")
            color = "#dc2626" if shap_val > 0 else "#059669" # Red for risk increase, green for protective

            attributions.append({
                'feature': feat,
                'label': meta.get('label', feat),
                'patient_value': display_val,
                'raw_value': val_raw,
                'shap_value': round(shap_val, 4),
                'abs_shap': abs(shap_val),
                'impact': impact,
                'color': color,
                'description': meta.get('description', '')
            })

        # Sort by absolute SHAP value
        attributions_df = pd.DataFrame(attributions).sort_values('abs_shap', ascending=False)
        top_positive = attributions_df[attributions_df['shap_value'] > 0].head(4)
        top_negative = attributions_df[attributions_df['shap_value'] < 0].head(4)

        # Clinical narrative explanation
        narrative = self._generate_narrative(attributions_df)

        return {
            'attributions_df': attributions_df,
            'top_features': attributions_df.head(6),
            'top_positive': top_positive,
            'top_negative': top_negative,
            'base_value': base_value,
            'narrative': narrative
        }

    def _generate_narrative(self, attributions_df: pd.DataFrame) -> list:
        """
        Creates natural language explanations of feature attributions.
        Adheres to academic non-diagnostic guidelines.
        """
        top_factors = []
        for _, row in attributions_df.head(4).iterrows():
            if row['shap_value'] > 0.01:
                top_factors.append(
                    f"**{row['label']}** ({row['patient_value']}): Contributed positively towards higher predicted heart disease probability."
                )
            elif row['shap_value'] < -0.01:
                top_factors.append(
                    f"**{row['label']}** ({row['patient_value']}): Contributed negatively, shifting prediction towards normal classification."
                )
            else:
                top_factors.append(
                    f"**{row['label']}** ({row['patient_value']}): Had a neutral or marginal influence on this patient's prediction."
                )
        return top_factors
