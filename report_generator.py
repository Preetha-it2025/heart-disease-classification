"""
Patient Analytical Report Generator (PDF and CSV formats)
for Heart Diseases Classification Academic System.
Powered by ReportLab Platypus.
"""

import os
import io
import datetime
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from preprocessing import FEATURE_METADATA


def generate_patient_pdf(
    patient_id: str,
    patient_data: dict,
    prediction_result: dict,
    explanation_result: dict,
    model_name: str = "Random Forest Classifier"
) -> bytes:
    """
    Generates a high-quality, professional medical analytics PDF report
    containing patient parameters, ML classification, SHAP attributions,
    and academic disclaimers.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Palette
    primary_color = colors.HexColor("#1e3a8a")     # Deep Clinical Navy
    secondary_color = colors.HexColor("#0284c7")   # Bright Clinical Cyan
    dark_text = colors.HexColor("#0f172a")         # Slate 900
    sub_text = colors.HexColor("#475569")          # Slate 600
    border_color = colors.HexColor("#cbd5e1")      # Slate 300
    bg_light = colors.HexColor("#f8fafc")          # Slate 50
    alert_red = colors.HexColor("#b91c1c")
    safe_green = colors.HexColor("#047857")
    caution_amber = colors.HexColor("#b45309")
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=primary_color
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=sub_text
    )
    
    h2_style = ParagraphStyle(
        'H2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=primary_color,
        spaceBefore=10,
        spaceAfter=6
    )
    
    cell_bold = ParagraphStyle(
        'CellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=dark_text
    )
    
    cell_regular = ParagraphStyle(
        'CellRegular',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=dark_text
    )
    
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#64748b")
    )
    
    elements = []
    
    # 1. Header Banner
    header_data = [
        [
            Paragraph("<b>HEART DISEASES CLASSIFICATION</b><br/><font size=8 color='#64748b'>Machine Learning Patient Analytics & Risk Categorization</font>", title_style),
            Paragraph(f"<b>Patient ID:</b> {patient_id}<br/><b>Date:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}<br/><b>Status:</b> Validated", cell_regular)
        ]
    ]
    t_header = Table(header_data, colWidths=[350, 180])
    t_header.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(t_header)
    elements.append(HRFlowable(width="100%", thickness=1.5, color=secondary_color, spaceBefore=4, spaceAfter=12))
    
    # 2. ML Classification Summary Card
    pred_val = prediction_result.get('prediction', 0)
    conf = prediction_result.get('confidence_percentage', 0.0)
    risk_cat = prediction_result.get('risk_category', 'Low')
    
    if pred_val == 1:
        badge_text = "<font color='#b91c1c'><b>HEART DISEASE INDICATED (MODEL CLASSIFICATION)</b></font>"
        card_border = alert_red
        card_bg = colors.HexColor("#fef2f2")
    else:
        badge_text = "<font color='#047857'><b>NO HEART DISEASE DETECTED</b></font>"
        card_border = safe_green
        card_bg = colors.HexColor("#ecfdf5")
        
    summary_data = [
        [
            Paragraph("<b>Model Classification Outcome:</b>", cell_bold),
            Paragraph(badge_text, cell_bold)
        ],
        [
            Paragraph("<b>Model Confidence / Predicted Probability:</b>", cell_regular),
            Paragraph(f"<b>{conf}%</b> (Model class probability output)", cell_regular)
        ],
        [
            Paragraph("<b>Model-Based Risk Category:</b>", cell_regular),
            Paragraph(f"<b>{risk_cat} Category</b> (Academic non-clinical risk stratification)", cell_regular)
        ],
        [
            Paragraph("<b>Algorithm Applied:</b>", cell_regular),
            Paragraph(f"{model_name} (Trained on Cleveland Heart Cohort)", cell_regular)
        ]
    ]
    t_summary = Table(summary_data, colWidths=[240, 290])
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), card_bg),
        ('BOX', (0, 0), (-1, -1), 1.5, card_border),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))
    elements.append(t_summary)
    elements.append(Spacer(1, 14))
    
    # 3. Patient Vitals & Clinical Data Table
    elements.append(Paragraph("Patient Clinical Parameters & Measured Vitals", h2_style))
    
    vitals_table_data = [
        [
            Paragraph("<b>Parameter</b>", cell_bold),
            Paragraph("<b>Patient Value</b>", cell_bold),
            Paragraph("<b>Reference Normal</b>", cell_bold),
            Paragraph("<b>Parameter</b>", cell_bold),
            Paragraph("<b>Patient Value</b>", cell_bold),
            Paragraph("<b>Reference Normal</b>", cell_bold)
        ]
    ]
    
    # Group into two columns of features
    keys = list(FEATURE_METADATA.keys())
    for i in range(0, len(keys), 2):
        k1 = keys[i]
        meta1 = FEATURE_METADATA[k1]
        val1 = patient_data.get(k1, 'N/A')
        if meta1.get('type') == 'categorical' and 'options' in meta1:
            disp1 = meta1['options'].get(int(val1) if val1 != 'N/A' else 0, str(val1))
        else:
            disp1 = f"{val1} {meta1.get('unit', '')}".strip()
        ref1 = meta1.get('normal_range', 'Categorical')
        
        if i + 1 < len(keys):
            k2 = keys[i + 1]
            meta2 = FEATURE_METADATA[k2]
            val2 = patient_data.get(k2, 'N/A')
            if meta2.get('type') == 'categorical' and 'options' in meta2:
                disp2 = meta2['options'].get(int(val2) if val2 != 'N/A' else 0, str(val2))
            else:
                disp2 = f"{val2} {meta2.get('unit', '')}".strip()
            ref2 = meta2.get('normal_range', 'Categorical')
        else:
            k2, disp2, ref2 = '', '', ''
            
        vitals_table_data.append([
            Paragraph(meta1['label'], cell_regular),
            Paragraph(disp1, cell_bold),
            Paragraph(ref1, disclaimer_style),
            Paragraph(meta2.get('label', '') if k2 else '', cell_regular),
            Paragraph(disp2, cell_bold),
            Paragraph(ref2, disclaimer_style)
        ])
        
    t_vitals = Table(vitals_table_data, colWidths=[110, 85, 70, 110, 85, 70])
    t_vitals.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ('GRID', (0, 0), (-1, -1), 0.5, border_color),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    elements.append(t_vitals)
    elements.append(Spacer(1, 14))
    
    # 4. Explainable AI Feature Attribution Table (SHAP)
    elements.append(Paragraph("Explainable AI: Key Contributing Features (SHAP Analysis)", h2_style))
    elements.append(Paragraph("The table below highlights which parameters exerted the strongest mathematical pull on the model classification outcome.", subtitle_style))
    elements.append(Spacer(1, 6))
    
    top_features = explanation_result.get('top_features')
    xai_data = [
        [
            Paragraph("<b>Clinical Feature</b>", cell_bold),
            Paragraph("<b>Patient Measured Value</b>", cell_bold),
            Paragraph("<b>SHAP Value</b>", cell_bold),
            Paragraph("<b>Influence on Prediction</b>", cell_bold)
        ]
    ]
    
    if top_features is not None and not top_features.empty:
        for _, r in top_features.iterrows():
            inf_color = "#b91c1c" if r['shap_value'] > 0 else "#047857"
            inf_text = f"<font color='{inf_color}'><b>{r['impact']}</b></font>"
            xai_data.append([
                Paragraph(str(r['label']), cell_regular),
                Paragraph(str(r['patient_value']), cell_bold),
                Paragraph(f"{r['shap_value']:+.4f}", cell_regular),
                Paragraph(inf_text, cell_regular)
            ])
            
    t_xai = Table(xai_data, colWidths=[180, 140, 90, 120])
    t_xai.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ('GRID', (0, 0), (-1, -1), 0.5, border_color),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(t_xai)
    elements.append(Spacer(1, 16))
    
    # 5. Academic Disclaimer Footer Box
    disclaimer_text = (
        "<b>ACADEMIC RESEARCH & CLINICAL DISCLAIMER:</b><br/>"
        "Model-generated result for academic purposes only. This system is not a medical diagnostic tool. "
        "The calculated probabilities and risk category represent mathematical classifications from machine learning "
        "algorithms trained on historical benchmark data, and must NEVER replace professional medical evaluation, clinical diagnosis, "
        "or physician consultation."
    )
    t_disc = Table([[Paragraph(disclaimer_text, disclaimer_style)]], colWidths=[530])
    t_disc.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#94a3b8")),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))
    elements.append(KeepTogether(t_disc))
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


def export_batch_results_to_csv(batch_results_df: pd.DataFrame) -> bytes:
    """Exports batch classified patient dataframe to formatted CSV bytes."""
    return batch_results_df.to_csv(index=False).encode('utf-8')
