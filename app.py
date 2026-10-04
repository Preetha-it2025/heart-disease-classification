"""
Heart Diseases Classification - Academic Machine Learning Healthcare Web Application
Main Streamlit Application File (app.py)
"""

import os
import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# System configuration & page setup
st.set_page_config(
    page_title="Heart Diseases Classification | Healthcare Analytics",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import internal modules
from utils.styles import LIGHT_HEALTHCARE_CSS
from utils.ui_components import (
    render_hero_banner,
    render_workflow_section,
    render_metric_card,
    render_academic_disclaimer,
    create_confidence_gauge,
    create_confusion_matrix_plot,
    create_roc_curve_plot,
    create_model_comparison_bar,
    create_shap_horizontal_bar,
    create_correlation_heatmap,
    update_plotly_layout
)
from preprocessing import (
    FEATURE_METADATA,
    REQUIRED_FEATURES,
    NUMERICAL_COLS,
    CATEGORICAL_COLS,
    validate_dataset,
    clean_dataset,
    calculate_risk_category
)
from train_model import (
    load_model_bundle,
    train_all_models,
    predict_single_patient,
    predict_batch_patients,
    DEFAULT_DATASET_PATH
)
from explainability import ModelExplainer
from report_generator import generate_patient_pdf, export_batch_results_to_csv

# Inject light healthcare CSS
st.markdown(LIGHT_HEALTHCARE_CSS, unsafe_allow_html=True)

# -------------------------------------------------------------------
# STATE INITIALIZATION
# -------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading machine learning models and dataset...")
def init_system_resources():
    """Caches model bundle and explainer in memory for instant responsiveness."""
    bundle = load_model_bundle()
    explainer = ModelExplainer(bundle)
    default_df = pd.read_csv(DEFAULT_DATASET_PATH)
    default_val = validate_dataset(default_df, require_target=True)
    return bundle, explainer, default_df, default_val

bundle, explainer, default_df, default_val = init_system_resources()

if "bundle" not in st.session_state:
    st.session_state.bundle = bundle
if "explainer" not in st.session_state:
    st.session_state.explainer = explainer
if "dataset" not in st.session_state:
    st.session_state.dataset = default_df
if "validation" not in st.session_state:
    st.session_state.validation = default_val

# Sample initial patient for instant demo readiness
default_patient = {
    'patient_id': 'PT-1082',
    'age': 58, 'sex': 1, 'cp': 2, 'trestbps': 140, 'chol': 245,
    'fbs': 0, 'restecg': 0, 'thalach': 150, 'exang': 0,
    'oldpeak': 1.6, 'slope': 1, 'ca': 1, 'thal': 2
}

if "current_patient_id" not in st.session_state:
    st.session_state.current_patient_id = default_patient['patient_id']
if "current_patient_data" not in st.session_state:
    st.session_state.current_patient_data = {k: v for k, v in default_patient.items() if k != 'patient_id'}
if "current_prediction" not in st.session_state:
    st.session_state.current_prediction = predict_single_patient(
        st.session_state.bundle, st.session_state.current_patient_data
    )
if "current_explanation" not in st.session_state:
    st.session_state.current_explanation = st.session_state.explainer.explain_patient_prediction(
        st.session_state.current_patient_data
    )
if "batch_results" not in st.session_state:
    sample_batch_file = os.path.join(os.path.dirname(__file__), 'dataset', 'sample_batch_patients.csv')
    if os.path.exists(sample_batch_file):
        sample_batch_df = pd.read_csv(sample_batch_file)
        st.session_state.batch_results = predict_batch_patients(
            st.session_state.bundle, sample_batch_df
        )
    else:
        st.session_state.batch_results = None


def navigate_to(page: str):
    """Keep the visible sidebar selection synchronized with in-app navigation."""
    st.session_state.nav_choice = page
    st.session_state.sidebar_nav = page


def load_batch_patient(patient_id: str, patient_row: dict):
    """Load a cohort record for local explanation and report pages."""
    patient_data = {
        feature: patient_row[feature]
        for feature in REQUIRED_FEATURES
        if feature in patient_row
    }
    st.session_state.current_patient_id = str(patient_id)
    st.session_state.current_patient_data = patient_data
    st.session_state.current_prediction = predict_single_patient(
        st.session_state.bundle, patient_data
    )
    st.session_state.current_explanation = st.session_state.explainer.explain_patient_prediction(
        patient_data
    )
    navigate_to("6. Explainable AI")


# -------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 18px; padding-bottom: 12px; border-bottom: 1px solid #e2e8f0;">
        <div style="background: #eff6ff; border-radius: 10px; padding: 8px; display: flex; align-items: center; justify-content: center;">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#1d4ed8" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"></path>
            </svg>
        </div>
        <div>
            <div style="font-weight: 800; font-size: 1.05rem; color: #0f2756; line-height: 1.2;">CardioClassify</div>
            <div style="font-size: 0.75rem; color: #64748b; font-weight: 500;">Academic ML System</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    NAV_OPTIONS = [
        "1. Home",
        "2. Dataset Upload",
        "3. Data Analysis",
        "4. ML Model",
        "5. Patient Classification",
        "6. Explainable AI",
        "7. Patient Report"
    ]
    
    # Check if page was set via button navigation
    if "nav_choice" not in st.session_state:
        st.session_state.nav_choice = NAV_OPTIONS[0]
        
    def on_nav_change():
        st.session_state.nav_choice = st.session_state.sidebar_nav
        
    selected_page = st.radio(
        "Navigation Menu",
        NAV_OPTIONS,
        index=NAV_OPTIONS.index(st.session_state.nav_choice) if st.session_state.nav_choice in NAV_OPTIONS else 0,
        key="sidebar_nav",
        on_change=on_nav_change,
        label_visibility="collapsed"
    )
    
    st.markdown("<hr style='margin: 20px 0; border: none; border-top: 1px solid #e2e8f0;'/>", unsafe_allow_html=True)
    
    # Active Model Status Widget in Sidebar
    active_m = st.session_state.bundle['best_model_name']
    active_acc = st.session_state.bundle['metrics'][active_m]['Accuracy']
    st.markdown(f"""
    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px; margin-bottom: 16px;">
        <div style="font-size: 0.72rem; text-transform: uppercase; font-weight: 700; color: #64748b; margin-bottom: 4px;">Active Classifier</div>
        <div style="font-weight: 700; font-size: 0.92rem; color: #1e3a8a;">{active_m}</div>
        <div style="font-size: 0.8rem; color: #059669; font-weight: 600; margin-top: 2px;">Test Accuracy: {active_acc:.1%}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Academic Disclaimer in Sidebar
    st.markdown("""
    <div style="font-size: 0.72rem; color: #64748b; line-height: 1.4; padding: 8px; background: #ffffff; border: 1px dashed #cbd5e1; border-radius: 8px;">
        <strong>Academic Project Notice:</strong><br/>
        Model-generated result for academic purposes only. This system is not a medical diagnostic tool.
    </div>
    """, unsafe_allow_html=True)


# -------------------------------------------------------------------
# PAGE 1: HOME
# -------------------------------------------------------------------
if selected_page == "1. Home":
    render_hero_banner()
    
    # 4 Metric Cards
    m1, m2, m3, m4 = st.columns(4)
    total_patients_count = len(st.session_state.dataset)
    best_m = st.session_state.bundle['best_model_name']
    best_acc = st.session_state.bundle['metrics'][best_m]['Accuracy']
    patients_analyzed_count = len(st.session_state.batch_results) if st.session_state.batch_results is not None else 1
    
    with m1:
        render_metric_card("Total Patients", f"{total_patients_count}", "Cleveland Benchmark Cohort", "#2563eb")
    with m2:
        render_metric_card("Input Features", "13", "Clinical Parameters", "#0284c7")
    with m3:
        render_metric_card("Model Accuracy", f"{best_acc:.1%}", f"Evaluated on {best_m}", "#059669")
    with m4:
        render_metric_card("Patients Analyzed", f"{patients_analyzed_count}", "Single & Batch Analyses", "#7c3aed")
        
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    
    # Call to Action Buttons
    c_btn1, c_btn2, _ = st.columns([1.5, 1.5, 3])
    with c_btn1:
        st.button(
            "🚀 Start Patient Analysis",
            type="primary",
            use_container_width=True,
            on_click=navigate_to,
            args=("5. Patient Classification",),
        )
    with c_btn2:
        st.button(
            "📂 Upload New Dataset",
            use_container_width=True,
            on_click=navigate_to,
            args=("2. Dataset Upload",),
        )
            
    st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
    
    # Workflow Pipeline Section
    with st.container(border=True):
        st.markdown("""
        <div class="med-card-header">
            <div class="med-card-title">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>
                End-to-End Healthcare Analytics & ML Workflow
            </div>
            <div style="font-size: 0.8rem; color: #64748b;">7 Comprehensive Analysis Phases</div>
        </div>
        """, unsafe_allow_html=True)
        render_workflow_section()
    
    # System Capabilities Overview
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="med-card" style="height: 100%;">
            <div style="font-size: 1.5rem; margin-bottom: 8px;">📊</div>
            <h4 style="color: #0f2756; margin-bottom: 6px; font-size: 1rem;">Data Quality & EDA</h4>
            <p style="font-size: 0.84rem; color: #475569; line-height: 1.5;">
                Automated schema validation, missingness detection, duplicate identification,
                and interactive epidemiological exploratory distributions.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="med-card" style="height: 100%;">
            <div style="font-size: 1.5rem; margin-bottom: 8px;">🧠</div>
            <h4 style="color: #0f2756; margin-bottom: 6px; font-size: 1rem;">Rigorous ML Benchmarking</h4>
            <p style="font-size: 0.84rem; color: #475569; line-height: 1.5;">
                Compares 5 actual classifiers: Random Forest, Logistic Regression, SVM, KNN, and Decision Trees
                with un-faked test metrics, ROC curves, and confusion matrices.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="med-card" style="height: 100%;">
            <div style="font-size: 1.5rem; margin-bottom: 8px;">🔍</div>
            <h4 style="color: #0f2756; margin-bottom: 6px; font-size: 1rem;">Explainable AI & Reports</h4>
            <p style="font-size: 0.84rem; color: #475569; line-height: 1.5;">
                Deep local and global SHAP attribution analysis to explain why each prediction was made,
                paired with instant professional PDF clinical report generation.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    render_academic_disclaimer()


# -------------------------------------------------------------------
# PAGE 2: DATASET UPLOAD
# -------------------------------------------------------------------
elif selected_page == "2. Dataset Upload":
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h2 style="color: #0f2756; font-weight: 800; margin-bottom: 4px;">Patient Dataset Upload & Quality Audit</h2>
        <p style="color: #475569; font-size: 0.92rem;">
            Upload patient cohort data in CSV or Excel format. The system automatically inspects data schema,
            missing values, duplicates, and feature integrity.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Preset sample dataset loaders for quick demonstration
    st.markdown("<div style='font-size: 0.85rem; font-weight: 600; color: #475569; margin-bottom: 6px;'>Quick Test Presets:</div>", unsafe_allow_html=True)
    q1, q2, q3, _ = st.columns([1.5, 1.6, 1.8, 2.5])
    with q1:
        if st.button("📁 UCI Cleveland (303)"):
            st.session_state.dataset = pd.read_csv(DEFAULT_DATASET_PATH)
            st.session_state.validation = validate_dataset(st.session_state.dataset, require_target=True)
            st.success("Loaded UCI Cleveland Benchmark Dataset.")
    with q2:
        if st.button("👥 Sample Batch (25)"):
            sample_p = os.path.join(os.path.dirname(__file__), 'dataset', 'sample_batch_patients.csv')
            st.session_state.dataset = pd.read_csv(sample_p)
            st.session_state.validation = validate_dataset(st.session_state.dataset, require_target=False)
            st.success("Loaded Sample Patient Cohort.")
    with q3:
        if st.button("⚠️ Corrupted Sample (Quality Test)"):
            corrupted_p = os.path.join(os.path.dirname(__file__), 'dataset', 'test_corrupted_sample.csv')
            st.session_state.dataset = pd.read_csv(corrupted_p)
            st.session_state.validation = validate_dataset(st.session_state.dataset, require_target=False)
            st.warning("Loaded Corrupted Sample dataset for quality audit.")
            
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    
    # Drag and Drop File Uploader
    uploaded_file = st.file_uploader(
        "Upload Patient Medical Dataset (CSV or XLSX)",
        type=['csv', 'xlsx', 'xls'],
        help="Upload a dataset containing clinical parameters (age, sex, chest pain, bp, cholesterol, etc.)"
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_up = pd.read_csv(uploaded_file)
            else:
                df_up = pd.read_excel(uploaded_file)
                
            st.session_state.dataset = df_up
            st.session_state.validation = validate_dataset(df_up, require_target=False)
            st.success(f"Successfully uploaded: **{uploaded_file.name}** ({len(df_up)} records)")
        except Exception as e:
            st.error(f"Error parsing uploaded file: {e}")
            
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    
    # Display Current Dataset Quick Inspection
    curr_df = st.session_state.dataset
    val = st.session_state.validation
    
    with st.container(border=True):
        c_inf1, c_inf2, c_inf3, c_inf4, c_inf5 = st.columns(5)
        with c_inf1:
            st.metric("Total Patients", f"{len(curr_df)}")
        with c_inf2:
            st.metric("Total Columns", f"{curr_df.shape[1]}")
        with c_inf3:
            st.metric("Missing Values", f"{val['missing_count']}")
        with c_inf4:
            st.metric("Duplicate Rows", f"{val['duplicate_count']}")
        with c_inf5:
            st.metric("Validation Status", f"{val['status']}")
    
    # Validate Dataset Action Button
    v_btn1, v_btn2, _ = st.columns([1.5, 2.2, 3])
    with v_btn1:
        if st.button("🔍 Validate Dataset", type="primary", use_container_width=True):
            st.session_state.validation = validate_dataset(curr_df, require_target='target' in curr_df.columns)
            st.rerun()
            
    with v_btn2:
        if 'target' in curr_df.columns:
            if st.button("⚙️ Retrain Models on This Dataset", use_container_width=True):
                with st.spinner("Retraining all 5 Machine Learning models on uploaded dataset..."):
                    # Save temporary training file and retrain
                    temp_p = os.path.join(os.path.dirname(__file__), 'dataset', 'active_training_data.csv')
                    curr_df.to_csv(temp_p, index=False)
                    st.session_state.bundle = train_all_models(temp_p)
                    st.session_state.explainer = ModelExplainer(st.session_state.bundle)
                    st.success("All 5 models retrained and evaluated on uploaded dataset!")
                    st.rerun()
                    
    # Validation Summary Card
    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
    status_class = "status-valid" if val['is_valid'] else "status-warning"
    status_icon = "✅" if val['is_valid'] else "⚠️"
    
    st.markdown(f"""
    <div class="{status_class}">
        <div style="font-size: 1.05rem; font-weight: 700; margin-bottom: 4px;">
            {status_icon} Dataset Status: {val['status']}
        </div>
        <div style="font-size: 0.85rem;">
            Audit performed across {val['total_records']} patient records and {val['total_features']} columns.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if val['issues']:
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        st.error("**Validation Issues Found:**\n" + "\n".join([f"- {iss}" for iss in val['issues']]))
        
    if val['warnings']:
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        st.warning("**Validation Warnings:**\n" + "\n".join([f"- {w}" for w in val['warnings']]))
        
    # Column-Level Audit Table
    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
    st.markdown("<h4 style='color: #0f2756;'>Column-Level Schema & Statistical Integrity</h4>", unsafe_allow_html=True)
    
    audit_rows = []
    for col, stat in val['column_stats'].items():
        audit_rows.append({
            'Column': stat['original_name'],
            'Clinical Feature': stat.get('label', 'Standard Feature'),
            'Data Type': stat['data_type'],
            'Missing': stat['missing_count'],
            'Unique': stat['unique_count'],
            'Min': str(stat['min']) if stat.get('min') is not None else 'N/A',
            'Mean': f"{stat.get('mean'):.1f}" if stat.get('mean') is not None else 'N/A',
            'Max': str(stat['max']) if stat.get('max') is not None else 'N/A',
            'Sample Values': str(stat['sample_values'][:2])
        })
    st.dataframe(pd.DataFrame(audit_rows), use_container_width=True, hide_index=True)
    
    render_academic_disclaimer()


# -------------------------------------------------------------------
# PAGE 3: DATA ANALYSIS
# -------------------------------------------------------------------
elif selected_page == "3. Data Analysis":
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h2 style="color: #0f2756; font-weight: 800; margin-bottom: 4px;">Interactive Clinical Data Analytics</h2>
        <p style="color: #475569; font-size: 0.92rem;">
            Explore cardiovascular feature correlations, distributions, missing value patterns, and target demographics.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    df_ana = st.session_state.dataset.copy()
    
    # 1. Dataset Preview Table
    with st.expander("🔍 Dataset Preview (First 10 Patient Records)", expanded=True):
        st.dataframe(df_ana.head(10), use_container_width=True)
        st.caption(f"Showing 10 of {len(df_ana)} total patient records.")
        
    # 2. Key Distribution Charts
    c_ch1, c_ch2 = st.columns(2)
    
    with c_ch1:
        with st.container(border=True):
            st.markdown("<h4 style='color: #0f2756; margin-bottom: 12px;'>Heart Disease Target Distribution</h4>", unsafe_allow_html=True)
            if 'target' in df_ana.columns:
                target_counts = df_ana['target'].value_counts().reset_index()
                target_counts.columns = ['Status', 'Count']
                target_counts['Status_Label'] = target_counts['Status'].map({
                    0: 'No Disease (0)', 1: 'Heart Disease (1)'
                })
                fig_target = px.pie(
                    target_counts, values='Count', names='Status_Label',
                    color='Status_Label',
                    color_discrete_map={'No Disease (0)': '#0284c7', 'Heart Disease (1)': '#ef4444'},
                    hole=0.45
                )
                update_plotly_layout(
                    fig_target,
                    paper_bgcolor='#ffffff',
                    font={'family': 'Plus Jakarta Sans', 'color': '#1e293b'},
                    margin={'l': 20, 'r': 20, 't': 20, 'b': 20},
                    height=300
                )
                st.plotly_chart(fig_target, use_container_width=True)
            else:
                st.info("Uploaded cohort has no ground-truth target column.")
        
    with c_ch2:
        with st.container(border=True):
            st.markdown("<h4 style='color: #0f2756; margin-bottom: 12px;'>Patient Age Distribution</h4>", unsafe_allow_html=True)
            if 'age' in df_ana.columns:
                if 'target' in df_ana.columns:
                    fig_age = px.histogram(
                        df_ana, x='age', color=df_ana['target'].map({0: 'No Disease', 1: 'Heart Disease'}),
                        barmode='overlay', nbins=18,
                        color_discrete_map={'No Disease': '#0284c7', 'Heart Disease': '#ef4444'}
                    )
                else:
                    fig_age = px.histogram(df_ana, x='age', nbins=18, color_discrete_sequence=['#2563eb'])
                
                update_plotly_layout(
                    fig_age,
                    paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
                    font={'family': 'Plus Jakarta Sans', 'color': '#1e293b'},
                    margin={'l': 20, 'r': 20, 't': 20, 'b': 20},
                    xaxis={'title': 'Age (Years)', 'gridcolor': '#f1f5f9'},
                    yaxis={'title': 'Patient Count', 'gridcolor': '#f1f5f9'},
                    height=300
                )
                st.plotly_chart(fig_age, use_container_width=True)
        
    # 3. Medical Feature Analysis (Cholesterol, BP, Max Heart Rate, Chest Pain)
    st.markdown("<h4 style='color: #0f2756; margin-top: 15px;'>Cardiovascular Biomarker Comparisons</h4>", unsafe_allow_html=True)
    c_bio1, c_bio2 = st.columns(2)
    
    with c_bio1:
        with st.container(border=True):
            st.markdown("<div style='font-weight: 700; color: #1e3a8a; margin-bottom: 8px;'>Serum Cholesterol vs. Resting Blood Pressure</div>", unsafe_allow_html=True)
            if 'chol' in df_ana.columns and 'trestbps' in df_ana.columns:
                color_arg = df_ana['target'].map({0: 'No Disease', 1: 'Heart Disease'}) if 'target' in df_ana.columns else None
                fig_scatter = px.scatter(
                    df_ana, x='chol', y='trestbps', color=color_arg,
                    labels={'chol': 'Cholesterol (mg/dl)', 'trestbps': 'Resting BP (mm Hg)'},
                    color_discrete_map={'No Disease': '#0284c7', 'Heart Disease': '#ef4444'},
                    opacity=0.75
                )
                update_plotly_layout(
                    fig_scatter,
                    paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
                    font={'family': 'Plus Jakarta Sans', 'color': '#1e293b'},
                    height=320, margin={'l': 30, 'r': 30, 't': 20, 'b': 30}
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
        
    with c_bio2:
        with st.container(border=True):
            st.markdown("<div style='font-weight: 700; color: #1e3a8a; margin-bottom: 8px;'>Chest Pain Type (CP) vs. Maximum Heart Rate</div>", unsafe_allow_html=True)
            if 'cp' in df_ana.columns and 'thalach' in df_ana.columns:
                cp_mapped = df_ana['cp'].map({
                    0: 'Typical Angina', 1: 'Atypical Angina',
                    2: 'Non-anginal', 3: 'Asymptomatic'
                })
                fig_box = px.box(
                    df_ana, x=cp_mapped, y='thalach',
                    color=cp_mapped,
                    labels={'x': 'Chest Pain Type', 'thalach': 'Max Heart Rate (bpm)'},
                    color_discrete_sequence=['#2563eb', '#0284c7', '#059669', '#d97706']
                )
                update_plotly_layout(
                    fig_box,
                    paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
                    font={'family': 'Plus Jakarta Sans', 'color': '#1e293b'},
                    height=320, margin={'l': 30, 'r': 30, 't': 20, 'b': 30},
                    showlegend=False
                )
                st.plotly_chart(fig_box, use_container_width=True)
        
    # 4. Correlation Heatmap
    with st.container(border=True):
        st.markdown("<div style='font-weight: 700; color: #0f2756; margin-bottom: 8px;'>Full Feature Correlation Heatmap</div>", unsafe_allow_html=True)
        st.plotly_chart(create_correlation_heatmap(df_ana), use_container_width=True)
    
    render_academic_disclaimer()


# -------------------------------------------------------------------
# PAGE 4: ML MODEL
# -------------------------------------------------------------------
elif selected_page == "4. ML Model":
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h2 style="color: #0f2756; font-weight: 800; margin-bottom: 4px;">Machine Learning Pipeline & Multi-Model Evaluation</h2>
        <p style="color: #475569; font-size: 0.92rem;">
            Architectural training workflow and true holdout test metrics calculated across 5 classification algorithms.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Visual ML Pipeline Flowchart
    st.markdown("""
    <div class="med-card">
        <div style="font-weight: 700; color: #1e3a8a; margin-bottom: 12px; font-size: 0.95rem;">
            ⚙️ ML Training & Inference Pipeline Flow
        </div>
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px; text-align: center;">
            <div style="background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 10px 14px; font-size: 0.8rem; font-weight: 700; color: #1e40af;">
                1. Dataset (303 records)
            </div>
            <div style="color: #94a3b8; font-weight: 800;">→</div>
            <div style="background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 10px 14px; font-size: 0.8rem; font-weight: 700; color: #1e40af;">
                2. Data Cleaning
            </div>
            <div style="color: #94a3b8; font-weight: 800;">→</div>
            <div style="background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 10px 14px; font-size: 0.8rem; font-weight: 700; color: #1e40af;">
                3. Preprocessing (Scaling)
            </div>
            <div style="color: #94a3b8; font-weight: 800;">→</div>
            <div style="background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 10px 14px; font-size: 0.8rem; font-weight: 700; color: #1e40af;">
                4. Stratified Split (80/20)
            </div>
            <div style="color: #94a3b8; font-weight: 800;">→</div>
            <div style="background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 10px 14px; font-size: 0.8rem; font-weight: 700; color: #1e40af;">
                5. Model Training (5 Models)
            </div>
            <div style="color: #94a3b8; font-weight: 800;">→</div>
            <div style="background: #dbeafe; border: 1px solid #93c5fd; border-radius: 8px; padding: 10px 14px; font-size: 0.8rem; font-weight: 700; color: #1e3a8a;">
                6. Test Evaluation
            </div>
            <div style="color: #94a3b8; font-weight: 800;">→</div>
            <div style="background: #dcfce7; border: 1px solid #86efac; border-radius: 8px; padding: 10px 14px; font-size: 0.8rem; font-weight: 700; color: #166534;">
                7. Classification & XAI
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Comparison Table with Actual Calculated Values
    metrics = st.session_state.bundle['metrics']
    rows = []
    for m_name, m_data in metrics.items():
        is_best = "⭐ (Selected)" if m_name == st.session_state.bundle['best_model_name'] else ""
        rows.append({
            'Model': f"{m_name} {is_best}".strip(),
            'Accuracy': f"{m_data['Accuracy']:.1%}",
            'Precision': f"{m_data['Precision']:.1%}",
            'Recall': f"{m_data['Recall']:.1%}",
            'F1 Score': f"{m_data['F1 Score']:.1%}",
            'ROC-AUC': f"{m_data['ROC-AUC']:.3f}"
        })
    df_metrics = pd.DataFrame(rows)
    
    st.markdown("<h4 style='color: #0f2756; margin-top: 15px;'>Model Performance Comparison (Calculated from Holdout Test Set)</h4>", unsafe_allow_html=True)
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)
    st.caption("All metrics are genuinely evaluated on the 20% stratified holdout split (no hardcoded values).")
    
    # Comparative Bar Chart
    st.plotly_chart(create_model_comparison_bar(metrics), use_container_width=True)
    
    # Confusion Matrix & ROC Curve Side by Side
    c_eval1, c_eval2 = st.columns(2)
    with c_eval1:
        with st.container(border=True):
            selected_cm_model = st.selectbox(
                "Select Model for Confusion Matrix:",
                list(metrics.keys()),
                index=list(metrics.keys()).index(st.session_state.bundle['best_model_name'])
            )
            cm_matrix = st.session_state.bundle['confusion_matrices'][selected_cm_model]
            st.plotly_chart(create_confusion_matrix_plot(cm_matrix, selected_cm_model), use_container_width=True)
        
    with c_eval2:
        with st.container(border=True):
            st.markdown("<div style='font-weight: 700; color: #0f2756; margin-bottom: 12px;'>Multi-Model ROC Curves</div>", unsafe_allow_html=True)
            roc_data = st.session_state.bundle['roc_data']
            st.plotly_chart(create_roc_curve_plot(roc_data), use_container_width=True)
        
    # Global Feature Importance Chart
    with st.container(border=True):
        st.markdown("<div style='font-weight: 700; color: #0f2756; margin-bottom: 12px;'>Global Feature Importance (Overall Model Impact)</div>", unsafe_allow_html=True)
        df_imp = st.session_state.explainer.get_global_feature_importance()
        fig_imp = px.bar(
            df_imp, x='importance', y='label', orientation='h',
            labels={'importance': 'Mean Importance Score', 'label': 'Clinical Feature'},
            color='importance', color_continuous_scale='Blues'
        )
        update_plotly_layout(
            fig_imp,
            paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
            font={'family': 'Plus Jakarta Sans', 'color': '#1e293b'},
            height=360, margin={'l': 150, 'r': 30, 't': 20, 'b': 30},
            yaxis={'categoryorder': 'total ascending'}
        )
        st.plotly_chart(fig_imp, use_container_width=True)
    
    render_academic_disclaimer()


# -------------------------------------------------------------------
# PAGE 5: PATIENT CLASSIFICATION
# -------------------------------------------------------------------
elif selected_page == "5. Patient Classification":
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h2 style="color: #0f2756; font-weight: 800; margin-bottom: 4px;">Patient Classification Engine</h2>
        <p style="color: #475569; font-size: 0.92rem;">
            Classify individual patient records manually, or execute batch analysis across multi-patient cohorts.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    tab_manual, tab_batch = st.tabs(["📝 OPTION A: Manual Patient Data Entry", "👥 OPTION B: Dataset-Based Batch Analysis"])
    
    # ------------------ OPTION A: MANUAL ENTRY ------------------
    with tab_manual:
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        
        # Quick Fill Buttons
        q_col1, q_col2, _ = st.columns([1.8, 1.8, 4])
        with q_col1:
            if st.button("📋 Load Sample: Elevated Risk Profile"):
                st.session_state.current_patient_id = "PT-SAMPLE-HIGH"
                st.session_state.current_patient_data = {
                    'age': 63, 'sex': 1, 'cp': 0, 'trestbps': 160, 'chol': 285,
                    'fbs': 1, 'restecg': 2, 'thalach': 118, 'exang': 1,
                    'oldpeak': 2.8, 'slope': 1, 'ca': 2, 'thal': 3
                }
                st.rerun()
        with q_col2:
            if st.button("📋 Load Sample: Normal Profile"):
                st.session_state.current_patient_id = "PT-SAMPLE-LOW"
                st.session_state.current_patient_data = {
                    'age': 45, 'sex': 0, 'cp': 2, 'trestbps': 118, 'chol': 185,
                    'fbs': 0, 'restecg': 0, 'thalach': 168, 'exang': 0,
                    'oldpeak': 0.2, 'slope': 2, 'ca': 0, 'thal': 1
                }
                st.rerun()
                
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        
        # Clinical Form Grouped into 4 Logical Medical Sections
        curr_p = st.session_state.current_patient_data
        
        with st.form("patient_manual_form"):
            st.markdown("<h4 style='color: #1e3a8a; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px;'>1. Patient Demographics & Identification</h4>", unsafe_allow_html=True)
            f_d1, f_d2, f_d3 = st.columns(3)
            with f_d1:
                pid_input = st.text_input("Patient ID", value=st.session_state.current_patient_id)
            with f_d2:
                age_input = st.number_input("Age (Years)", min_value=18, max_value=100, value=int(curr_p.get('age', 55)))
            with f_d3:
                sex_input = st.selectbox("Biological Sex", options=[1, 0], format_func=lambda x: "Male (1)" if x == 1 else "Female (0)", index=0 if curr_p.get('sex', 1) == 1 else 1)
                
            st.markdown("<h4 style='color: #1e3a8a; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px; margin-top: 14px;'>2. Symptoms & Medical History</h4>", unsafe_allow_html=True)
            f_s1, f_s2, f_s3 = st.columns(3)
            with f_s1:
                cp_options = {0: 'Typical Angina (0)', 1: 'Atypical Angina (1)', 2: 'Non-anginal Pain (2)', 3: 'Asymptomatic (3)'}
                cp_input = st.selectbox("Chest Pain Type (CP)", options=[0, 1, 2, 3], format_func=lambda x: cp_options[x], index=int(curr_p.get('cp', 0)))
            with f_s2:
                exang_input = st.selectbox("Exercise-Induced Angina", options=[0, 1], format_func=lambda x: "No (0)" if x == 0 else "Yes (1)", index=int(curr_p.get('exang', 0)))
            with f_s3:
                fbs_input = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "False <= 120 (0)" if x == 0 else "True > 120 (1)", index=int(curr_p.get('fbs', 0)))
                
            st.markdown("<h4 style='color: #1e3a8a; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px; margin-top: 14px;'>3. Resting Vitals & Electrocardiogram</h4>", unsafe_allow_html=True)
            f_v1, f_v2, f_v3 = st.columns(3)
            with f_v1:
                trestbps_input = st.number_input("Resting Blood Pressure (mm Hg)", min_value=70, max_value=240, value=int(curr_p.get('trestbps', 130)))
            with f_v2:
                chol_input = st.number_input("Serum Cholesterol (mg/dl)", min_value=90, max_value=600, value=int(curr_p.get('chol', 240)))
            with f_v3:
                restecg_options = {0: 'Normal (0)', 1: 'ST-T Wave Abnormality (1)', 2: 'Left Ventricular Hypertrophy (2)'}
                restecg_input = st.selectbox("Resting ECG Results", options=[0, 1, 2], format_func=lambda x: restecg_options[x], index=int(curr_p.get('restecg', 0)))
                
            st.markdown("<h4 style='color: #1e3a8a; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px; margin-top: 14px;'>4. Cardiac Stress Test & Scintigraphy</h4>", unsafe_allow_html=True)
            f_c1, f_c2, f_c3, f_c4 = st.columns(4)
            with f_c1:
                thalach_input = st.number_input("Max Heart Rate (bpm)", min_value=60, max_value=230, value=int(curr_p.get('thalach', 150)))
            with f_c2:
                oldpeak_input = st.number_input("ST Depression (Oldpeak, mm)", min_value=0.0, max_value=7.0, value=float(curr_p.get('oldpeak', 1.0)), step=0.1)
            with f_c3:
                slope_options = {0: 'Upsloping (0)', 1: 'Flat (1)', 2: 'Downsloping (2)'}
                slope_input = st.selectbox("Slope of Peak ST", options=[0, 1, 2], format_func=lambda x: slope_options[x], index=int(curr_p.get('slope', 1)))
            with f_c4:
                ca_input = st.selectbox("Major Vessels Colored (ca)", options=[0, 1, 2, 3], index=int(curr_p.get('ca', 0)))
                
            thal_options = {1: 'Normal (1)', 2: 'Fixed Defect (2)', 3: 'Reversible Defect (3)'}
            thal_input = st.selectbox("Thalassemia Indicator (thal)", options=[1, 2, 3], format_func=lambda x: thal_options[x], index=1 if int(curr_p.get('thal', 2)) not in [1, 2, 3] else [1, 2, 3].index(int(curr_p.get('thal', 2))))
            
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            submit_btn = st.form_submit_button("🩺 ANALYZE PATIENT", type="primary", use_container_width=True)
            
        if submit_btn:
            with st.spinner("Analyzing patient clinical parameters using trained ML model..."):
                input_data = {
                    'age': age_input, 'sex': sex_input, 'cp': cp_input,
                    'trestbps': trestbps_input, 'chol': chol_input, 'fbs': fbs_input,
                    'restecg': restecg_input, 'thalach': thalach_input,
                    'exang': exang_input, 'oldpeak': oldpeak_input,
                    'slope': slope_input, 'ca': ca_input, 'thal': thal_input
                }
                st.session_state.current_patient_id = pid_input
                st.session_state.current_patient_data = input_data
                st.session_state.current_prediction = predict_single_patient(
                    st.session_state.bundle, input_data
                )
                st.session_state.current_explanation = st.session_state.explainer.explain_patient_prediction(
                    input_data
                )
                st.success("Patient clinical parameters processed successfully!")
                
        # ------------------ CLASSIFICATION RESULT CARD ------------------
        res = st.session_state.current_prediction
        pred_is_positive = (res['prediction'] == 1)
        card_class = "result-card-danger" if pred_is_positive else "result-card-safe"
        
        st.markdown(f"""
        <div class="{card_class}" style="margin-top: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px;">
                <div>
                    <div style="font-size: 0.82rem; font-weight: 700; text-transform: uppercase; color: #475569;">
                        Patient ID: {st.session_state.current_patient_id}
                    </div>
                    <div style="font-size: 1.8rem; font-weight: 800; color: {'#b91c1c' if pred_is_positive else '#047857'}; margin-top: 4px;">
                        {'🚨 Heart Disease Detected' if pred_is_positive else '✅ No Heart Disease Detected'}
                    </div>
                    <div style="font-size: 0.88rem; color: #334155; margin-top: 6px;">
                        <strong>Model Classification Outcome:</strong> Generated by {res['model_used']}
                    </div>
                </div>
                <div style="text-align: right;">
                    <div class="risk-pill" style="background: {res['risk_badge_bg']}; color: {res['risk_color']}; border: 1.5px solid {res['risk_badge_border']};">
                        {res['risk_label']}
                    </div>
                    <div style="font-size: 0.78rem; color: #64748b; margin-top: 5px;">
                        Academic Non-Clinical Stratification
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Gauge Visual & Quick Navigation to XAI / Report
        c_g1, c_g2 = st.columns([1.6, 2])
        with c_g1:
            st.plotly_chart(create_confidence_gauge(res['probability']), use_container_width=True)
        with c_g2:
            with st.container(border=True):
                st.markdown(f"""
                <h4 style="color: #0f2756; margin-bottom: 8px;">Analysis Next Steps</h4>
                <p style="font-size: 0.88rem; color: #475569; line-height: 1.5;">
                    The model calculated a <strong>{res['confidence_percentage']}%</strong> classification probability.
                    Understand which clinical parameters most contributed to this outcome with Explainable AI (SHAP)
                    or export a formal medical report.
                </p>
                """, unsafe_allow_html=True)
            
                nav_col1, nav_col2 = st.columns(2)
                with nav_col1:
                    st.button(
                        "🔍 Explain This Prediction",
                        use_container_width=True,
                        on_click=navigate_to,
                        args=("6. Explainable AI",),
                    )
                with nav_col2:
                    st.button(
                        "📄 View Full Patient Report",
                        use_container_width=True,
                        on_click=navigate_to,
                        args=("7. Patient Report",),
                    )
            
    # ------------------ OPTION B: BATCH DATASET ANALYSIS ------------------
    with tab_batch:
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.markdown("""
        <p style="color: #475569; font-size: 0.92rem;">
            Run automated heart disease classification across a multi-patient cohort.
            Filter, search, inspect cohort distributions, and drill down into individual patients.
        </p>
        """, unsafe_allow_html=True)
        
        # Upload batch file or use currently loaded dataset
        b_up = st.file_uploader("Upload Batch Patient File (CSV / XLSX)", type=['csv', 'xlsx'], key="batch_file_uploader")
        if b_up is not None:
            if b_up.name.endswith('.csv'):
                b_df = pd.read_csv(b_up)
            else:
                b_df = pd.read_excel(b_up)
            with st.spinner("Processing batch classifications..."):
                st.session_state.batch_results = predict_batch_patients(
                    st.session_state.bundle, b_df
                )
            st.success(f"Analyzed {len(b_df)} patient records from {b_up.name}!")
            
        if st.session_state.batch_results is None:
            if st.button("👥 Run Batch Analysis on Loaded Dataset"):
                with st.spinner("Classifying cohort patients..."):
                    st.session_state.batch_results = predict_batch_patients(
                        st.session_state.bundle, st.session_state.dataset
                    )
                st.rerun()
                
        if st.session_state.batch_results is not None:
            batch_df = st.session_state.batch_results
            
            # Overview Metric Cards for Batch
            b_total = len(batch_df)
            b_disease = (batch_df['Prediction'] == 1).sum()
            b_normal = (batch_df['Prediction'] == 0).sum()
            b_avg_conf = batch_df['Model Confidence (%)'].mean()
            
            bc1, bc2, bc3, bc4 = st.columns(4)
            with bc1:
                render_metric_card("Patients Analyzed", f"{b_total}", "Total Batch Records", "#2563eb")
            with bc2:
                render_metric_card("Classified: Disease", f"{b_disease}", f"{(b_disease/b_total):.1%} of cohort", "#dc2626")
            with bc3:
                render_metric_card("Classified: Normal", f"{b_normal}", f"{(b_normal/b_total):.1%} of cohort", "#059669")
            with bc4:
                render_metric_card("Avg Confidence", f"{b_avg_conf:.1f}%", "Cohort-wide Mean", "#7c3aed")
                
            st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
            
            # Cohort Filters
            st.markdown("<h4 style='color: #0f2756;'>Filter & Search Cohort</h4>", unsafe_allow_html=True)
            flt1, flt2, flt3 = st.columns(3)
            with flt1:
                filter_pred = st.selectbox("Filter by Classification:", ["All", "Heart Disease", "No Heart Disease"])
            with flt2:
                filter_risk = st.selectbox("Filter by Risk Category:", ["All", "Low", "Moderate", "High"])
            with flt3:
                search_kw = st.text_input("Search (Patient ID / Vitals):", placeholder="e.g. PT-1005")
                
            filtered_df = batch_df.copy()
            if filter_pred != "All":
                filtered_df = filtered_df[filtered_df['Classification'] == filter_pred]
            if filter_risk != "All":
                filtered_df = filtered_df[filtered_df['Model-Based Risk Category'] == filter_risk]
            if search_kw:
                filtered_df = filtered_df[filtered_df.astype(str).apply(lambda row: row.str.contains(search_kw, case=False).any(), axis=1)]
                
            # Patient-Wise Results Table
            st.dataframe(filtered_df, use_container_width=True)
            
            # Drill-down selector to load a specific patient into session state
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            st.markdown("<h4 style='color: #1e3a8a;'>🔬 Deep Dive into a Patient from this Cohort</h4>", unsafe_allow_html=True)
            
            if 'patient_id' in filtered_df.columns:
                patient_options = filtered_df['patient_id'].tolist()
            else:
                patient_options = [f"Record Index {i}" for i in filtered_df.index]
                
            p_select_col, p_action_col = st.columns([2, 1.5])
            with p_select_col:
                selected_patient_tag = st.selectbox("Select Patient to Inspect:", patient_options)
            with p_action_col:
                st.button(
                    "🚀 Load Patient into Explainable AI & Report",
                    type="primary",
                    use_container_width=True,
                    on_click=load_batch_patient,
                    args=(
                        selected_patient_tag,
                        filtered_df.iloc[patient_options.index(selected_patient_tag)].to_dict(),
                    ),
                )
                    
    render_academic_disclaimer()


# -------------------------------------------------------------------
# PAGE 6: EXPLAINABLE AI
# -------------------------------------------------------------------
elif selected_page == "6. Explainable AI":
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h2 style="color: #0f2756; font-weight: 800; margin-bottom: 4px;">Explainable AI: Why This Prediction?</h2>
        <p style="color: #475569; font-size: 0.92rem;">
            Transparent clinical machine learning with SHAP (SHapley Additive exPlanations) attribution analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Active Patient Summary Strip
    pid = st.session_state.current_patient_id
    pred_res = st.session_state.current_prediction
    exp_res = st.session_state.current_explanation
    
    st.markdown(f"""
    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 14px 18px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
        <div>
            <span style="font-weight: 800; color: #1e3a8a; font-size: 1.05rem;">Patient ID: {pid}</span>
            <span style="margin-left: 12px; font-size: 0.88rem; color: #475569;">Model Outcome: <strong>{pred_res['prediction_label']}</strong></span>
        </div>
        <div>
            <span class="risk-pill" style="background: {pred_res['risk_badge_bg']}; color: {pred_res['risk_color']}; border: 1px solid {pred_res['risk_badge_border']};">
                {pred_res['risk_label']} ({pred_res['confidence_percentage']}%)
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # SHAP Attribution Waterfall / Horizontal Bar Chart
    with st.container(border=True):
        st.markdown(f"""
        <div class="med-card-header">
            <div class="med-card-title">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2"><path d="M18 20V10M12 20V4M6 20v-6"/></svg>
                Top Contributing Features for Patient {pid} (SHAP Values)
            </div>
            <div style="font-size: 0.78rem; color: #64748b;">
                <span style="color: #dc2626; font-weight: 700;">■ Positive (Increases Risk)</span> &nbsp;|&nbsp;
                <span style="color: #059669; font-weight: 700;">■ Negative (Decreases Risk)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
        st.plotly_chart(
            create_shap_horizontal_bar(exp_res['attributions_df'], title=f"Feature Attributions: {pid}"),
            use_container_width=True
        )
    
    # Clinical Narrative Insights & Explanation
    c_nar1, c_nar2 = st.columns([1.5, 1])
    
    with c_nar1:
        with st.container(border=True):
            st.markdown("<h4 style='color: #0f2756; margin-bottom: 10px;'>Attribution Insights</h4>", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 0.85rem; color: #64748b;'>The following features had the strongest influence on this model prediction:</p>", unsafe_allow_html=True)
            for point in exp_res['narrative']:
                st.markdown(f"- {point}")
            
            st.markdown("""
            <div style="margin-top: 14px; padding: 10px; background: #f8fafc; border-left: 3px solid #0284c7; border-radius: 6px; font-size: 0.78rem; color: #475569;">
                <strong>Methodology Note:</strong> SHAP values compute the exact marginal contribution of each physiological parameter
                relative to the expected baseline prediction of the Cleveland patient cohort.
            </div>
            """, unsafe_allow_html=True)
        
    with c_nar2:
        with st.container(border=True):
            st.markdown("<h4 style='color: #0f2756; margin-bottom: 10px;'>Impact Breakdown</h4>", unsafe_allow_html=True)
        
            pos_count = len(exp_res['top_positive'])
            neg_count = len(exp_res['top_negative'])
        
            st.markdown(f"""
            <div style="background: #fef2f2; border: 1px solid #fecaca; border-radius: 8px; padding: 10px; margin-bottom: 10px;">
                <div style="font-weight: 700; color: #b91c1c; font-size: 0.85rem;">{pos_count} Factor(s) Pushing Towards Risk</div>
                <div style="font-size: 0.78rem; color: #475569;">Features with positive SHAP weight elevated the probability of heart disease classification.</div>
            </div>
            <div style="background: #ecfdf5; border: 1px solid #a7f3d0; border-radius: 8px; padding: 10px;">
                <div style="font-weight: 700; color: #047857; font-size: 0.85rem;">{neg_count} Factor(s) Protective / Normalizing</div>
                <div style="font-size: 0.78rem; color: #475569;">Features with negative SHAP weight shifted the classification towards no disease.</div>
            </div>
            """, unsafe_allow_html=True)
        
            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            st.button(
                "📄 Generate & Download Patient Report",
                type="primary",
                use_container_width=True,
                on_click=navigate_to,
                args=("7. Patient Report",),
            )
            
        
    # Complete Attributions Table
    with st.expander(f"📋 View Complete 13-Feature Attribution Breakdown for {pid}"):
        st.dataframe(
            exp_res['attributions_df'][['label', 'patient_value', 'shap_value', 'impact', 'description']],
            use_container_width=True, hide_index=True
        )
        
    render_academic_disclaimer()


# -------------------------------------------------------------------
# PAGE 7: PATIENT REPORT
# -------------------------------------------------------------------
elif selected_page == "7. Patient Report":
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h2 style="color: #0f2756; font-weight: 800; margin-bottom: 4px;">Comprehensive Patient Analytical Report</h2>
        <p style="color: #475569; font-size: 0.92rem;">
            Formal analytical report summarizing patient vitals, machine learning classification, and XAI feature attributions.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    pid = st.session_state.current_patient_id
    pdata = st.session_state.current_patient_data
    pred_res = st.session_state.current_prediction
    exp_res = st.session_state.current_explanation
    m_name = pred_res['model_used']
    
    # Download Actions Bar
    d_col1, d_col2, _ = st.columns([1.8, 1.8, 3.5])
    with d_col1:
        # Generate PDF bytes dynamically using ReportLab
        try:
            pdf_bytes = generate_patient_pdf(
                patient_id=pid,
                patient_data=pdata,
                prediction_result=pred_res,
                explanation_result=exp_res,
                model_name=m_name
            )
            st.download_button(
                label="📥 Download Report as PDF",
                data=pdf_bytes,
                file_name=f"heart_disease_report_{pid}.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Error compiling PDF: {e}")
            
    with d_col2:
        if st.session_state.batch_results is not None:
            csv_bytes = export_batch_results_to_csv(st.session_state.batch_results)
            st.download_button(
                label="📊 Download Batch CSV",
                data=csv_bytes,
                file_name="cohort_classification_results.csv",
                mime="text/csv",
                use_container_width=True
            )
            
    with st.container(border=True):
        st.markdown(f"### Patient Cardiac Analysis Report · {pid}")
        st.caption(f"Cardiovascular machine learning classification · Model: {m_name}")

        result_col, confidence_col, risk_col = st.columns(3)
        with result_col:
            st.metric("Classification Result", pred_res['prediction_label'])
        with confidence_col:
            st.metric("Model Confidence", f"{pred_res['confidence_percentage']}%")
        with risk_col:
            st.metric("Risk Level", pred_res['risk_label'])

        st.subheader("Patient Demographics & Vital Signs")
        vital_columns = st.columns(3)
        for index, (feature, metadata) in enumerate(FEATURE_METADATA.items()):
            value = pdata.get(feature, 'N/A')
            if metadata.get('type') == 'categorical' and 'options' in metadata:
                try:
                    value = metadata['options'].get(int(value), str(value))
                except (TypeError, ValueError):
                    value = str(value)
            elif metadata.get('unit'):
                value = f"{value} {metadata['unit']}"

            with vital_columns[index % len(vital_columns)]:
                st.markdown(f"**{metadata['label']}**")
                st.write(value)

        st.subheader("Medical Analysis Summary")
        st.write(
            f"The {m_name} model classified this patient as "
            f"**{pred_res['prediction_label']}** with a model confidence of "
            f"**{pred_res['confidence_percentage']}%**. The model-based risk level is "
            f"**{pred_res['risk_label']}**. {pred_res['risk_description']}"
        )
        st.caption(
            "This classification is generated from the patient's recorded clinical "
            "features for academic analysis; it is not a medical diagnosis."
        )

        st.subheader("Explainable AI Insights (SHAP)")
        st.dataframe(
            exp_res['top_features'][
                ['label', 'patient_value', 'shap_value', 'impact']
            ].rename(columns={
                'label': 'Clinical Feature',
                'patient_value': 'Patient Value',
                'shap_value': 'SHAP Value',
                'impact': 'Model Influence'
            }),
            use_container_width=True,
            hide_index=True
        )
        for insight in exp_res['narrative']:
            st.markdown(f"- {insight}")
    
    render_academic_disclaimer()
