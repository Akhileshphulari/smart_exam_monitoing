"""
Page 7: Sprint 4 – Data Quality & Statistical Validation
Smart Exam Monitoring & Integrity Analytics Platform - Team 4
14-point data quality audit, Chi-Square goodness-of-fit test,
out-of-order timestamp detection, correlation matrix, descriptive statistics.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.data_loader import load_all_datasets
from src.preprocessing import run_pipeline, audit_data_quality
from src.statistical_validation import (
    validate_event_distributions,
    check_missing_duplicate_events,
    detect_out_of_order_timestamps,
    validate_numerical_metrics,
    evaluate_test_datasets,
    generate_data_quality_score,
    compute_descriptive_statistics,
    compute_correlation_analysis
)
from src.outlier_detection import build_outlier_feature_matrix

st.set_page_config(page_title="Data Quality & Stats | ExamGuard", page_icon="🛡️", layout="wide")

# Dark Navy Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #0a0e27; color: #e2e8f0; }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0a0e27 0%, #131842 100%) !important; }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    h1, h2, h3, h4, h5, h6 { color: #ffffff !important; }
    .kpi-card {
        background: linear-gradient(135deg, #1a1f4e 0%, #252b6a 100%);
        border: 1px solid #3b3f7a; border-radius: 16px; padding: 20px 22px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .kpi-title { font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.6px; color: #8b92c4; margin-bottom: 6px; }
    .kpi-value { font-size: 1.85rem; font-weight: 700; color: #ffffff; margin-bottom: 4px; }
    .kpi-delta { font-size: 0.82rem; font-weight: 500; }
    .content-card {
        background: #111538; border: 1px solid #2a2f5e; border-radius: 14px;
        padding: 22px 24px; margin-bottom: 20px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.25);
    }
    .content-card-title { font-size: 1.1rem; font-weight: 700; color: #ffffff; margin-bottom: 4px; }
    .content-card-subtitle { font-size: 0.85rem; color: #7b83b5; margin-bottom: 16px; }
    [data-testid="stMetricValue"] { color: #ffffff !important; }
    [data-testid="stMetricLabel"] { color: #8b92c4 !important; }
</style>
""", unsafe_allow_html=True)

def apply_dark_theme(fig, height=340):
    fig.update_layout(
        font=dict(family="Inter, sans-serif", size=12, color="#c5cae8"),
        plot_bgcolor="#111538", paper_bgcolor="#111538",
        margin=dict(t=30, b=25, l=35, r=20), height=height,
        hoverlabel=dict(bgcolor="#1a1f4e", font_size=12, font_family="Inter", font_color="#ffffff")
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#1e2454", linecolor="#3b3f7a")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#1e2454", linecolor="#3b3f7a")
    return fig

# Header
st.markdown("""
<div style="background: linear-gradient(135deg, #1a1f4e 0%, #252b6a 100%); border: 1px solid #3b3f7a; border-radius: 14px; padding: 20px 24px; margin-bottom: 24px;">
    <h1 style="margin:0; font-size:1.55rem; font-weight:700; color:#ffffff !important;">Sprint 4: Data Quality & Statistical Validation</h1>
    <p style="margin:4px 0 0 0; font-size:0.88rem; color:#7b83b5;">
        14-Point Data Quality Audit, Goodness-of-Fit Testing, Chronological Integrity, Correlation Analysis & Benchmark Evaluation
    </p>
</div>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)
def get_validation_data():
    raw = load_all_datasets()
    pipeline = run_pipeline(raw)
    return raw, pipeline

raw_data, pipeline_data = get_validation_data()
candidates = pipeline_data["candidates"]
sessions = pipeline_data["sessions"]
events = pipeline_data["events"]
scores = pipeline_data["scores"]
risk = pipeline_data["risk"]

# 1. High-Level Data Quality Score
dq_score_res = generate_data_quality_score(events, sessions, scores, candidates)

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Overall Quality Score</div>
    <div class="kpi-value" style="color:#10b981;">{dq_score_res['overall_score']}%</div>
    <div class="kpi-delta" style="color:#10b981;">● Weighted Health</div></div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Completeness</div>
    <div class="kpi-value">{dq_score_res['completeness_score']}%</div>
    <div class="kpi-delta" style="color:#6366f1;">● Null Free Rate</div></div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Consistency</div>
    <div class="kpi-value">{dq_score_res['consistency_score']}%</div>
    <div class="kpi-delta" style="color:#8b5cf6;">● Foreign Keys Valid</div></div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Accuracy</div>
    <div class="kpi-value">{dq_score_res['accuracy_score']}%</div>
    <div class="kpi-delta" style="color:#10b981;">● Domain Constraints</div></div>""", unsafe_allow_html=True)
with k5:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Uniqueness</div>
    <div class="kpi-value">{dq_score_res['uniqueness_score']}%</div>
    <div class="kpi-delta" style="color:#10b981;">● No Duplicates</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 2. 14-Point Pre-Cleansing vs Post-Cleansing Quality Audit
st.markdown("""<div class="content-card"><div class="content-card-title">14-Point Data Quality Audit</div>
<div class="content-card-subtitle">Comprehensive validation against integrity rules and relational schemas</div>""", unsafe_allow_html=True)

dq_audit = pipeline_data.get("data_quality_report", {})
if dq_audit:
    checks_df = dq_audit.get("checks_table", pd.DataFrame())
    if not checks_df.empty:
        st.dataframe(checks_df, use_container_width=True, hide_index=True)
st.markdown("</div>", unsafe_allow_html=True)

# 3. Statistical Event Distribution Validation (Chi-Square) & Out-of-Order Events
c1, c2 = st.columns(2)

with c1:
    st.markdown("""<div class="content-card"><div class="content-card-title">Chi-Square Goodness-of-Fit Test</div>
    <div class="content-card-subtitle">Testing uniformity of event distribution</div>""", unsafe_allow_html=True)
    chi_res = validate_event_distributions(events)
    st.info(chi_res["interpretation"])
    st.metric("Chi-Square Statistic", f"{chi_res['chi2_stat']:.2f}")
    st.metric("p-value", f"{chi_res['p_value']:.4e}")
    if not chi_res["distribution_table"].empty:
        st.dataframe(chi_res["distribution_table"], use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown("""<div class="content-card"><div class="content-card-title">Out-of-Order Timestamp Detection</div>
    <div class="content-card-subtitle">Events occurring prior to exam start or post completion</div>""", unsafe_allow_html=True)
    ooo_df = detect_out_of_order_timestamps(events, sessions)
    if not ooo_df.empty:
        st.warning(f"Detected {len(ooo_df)} chronological timestamp violations.")
        st.dataframe(ooo_df, use_container_width=True, hide_index=True)
    else:
        st.success("Zero out-of-order timestamps detected. All events fall strictly within exam duration windows.")
    
    st.markdown("##### Numerical Metric Boundary Validation")
    num_valid_df = validate_numerical_metrics(sessions, scores)
    st.dataframe(num_valid_df, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

# 4. Correlation Matrix & Descriptive Statistics
col_cr, col_ds = st.columns(2)

with col_cr:
    st.markdown("""<div class="content-card"><div class="content-card-title">Feature Correlation Heatmap</div>
    <div class="content-card-subtitle">Pearson correlation coefficients among numerical metrics</div>""", unsafe_allow_html=True)
    feat_matrix = build_outlier_feature_matrix(sessions, events, scores)
    corr_matrix = compute_correlation_analysis(feat_matrix)
    if not corr_matrix.empty:
        fig_heat = px.imshow(
            corr_matrix, text_auto=True, aspect="auto",
            color_continuous_scale="Viridis",
            labels=dict(color="Correlation")
        )
        apply_dark_theme(fig_heat, height=300)
        st.plotly_chart(fig_heat, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_ds:
    st.markdown("""<div class="content-card"><div class="content-card-title">Descriptive Statistics & Higher Moments</div>
    <div class="content-card-subtitle">Mean, Std Dev, Min/Max, Skewness, Kurtosis</div>""", unsafe_allow_html=True)
    desc_stats = compute_descriptive_statistics(sessions, events, scores)
    if not desc_stats["summary_table"].empty:
        st.dataframe(desc_stats["summary_table"], use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

# 5. Independent Test Dataset Evaluation Benchmark
st.markdown("""<div class="content-card"><div class="content-card-title">Independent Synthetic Test Evaluation Benchmark</div>
<div class="content-card-subtitle">Precision, Recall, F1 against reproducible injected anomalies (seed=42)</div>""", unsafe_allow_html=True)
test_eval = evaluate_test_datasets(events, sessions)
tb1, tb2 = st.columns([1, 1.5])
with tb1:
    st.dataframe(test_eval["metrics_table"], use_container_width=True, hide_index=True)
with tb2:
    st.markdown(f"""
    - **True Positives (TP):** {test_eval['tp']} correctly identified anomalous exam sessions
    - **False Positives (FP):** {test_eval['fp']} nominal sessions flagged as anomalous
    - **False Negatives (FN):** {test_eval['fn']} anomalous sessions missed by detector
    - **True Negatives (TN):** {test_eval['tn']} nominal sessions correctly cleared
    """)
st.markdown("</div>", unsafe_allow_html=True)

