"""
Page 4: Sprint 3 – Face Monitoring Analytics
Smart Exam Monitoring & Integrity Analytics Platform - Team 4
Analyzes existing face monitoring telemetry: presence/absence frequencies,
absence durations, lighting condition impacts, accuracy metrics with labeled ground truth,
confusion matrix, and model limitations report.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.data_loader import load_all_datasets
from src.preprocessing import run_pipeline
from src.face_analysis import (
    analyze_face_frequencies,
    analyze_absence_durations,
    compare_lighting_conditions,
    evaluate_face_detection_accuracy,
    generate_model_limitations_report
)

st.set_page_config(page_title="Sprint 3 - Face Monitoring | ExamGuard", page_icon="👤", layout="wide")

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
    <h1 style="margin:0; font-size:1.55rem; font-weight:700; color:#ffffff !important;">Sprint 3: Face Monitoring Analytics</h1>
    <p style="margin:4px 0 0 0; font-size:0.88rem; color:#7b83b5;">
        Analyzes existing face monitoring telemetry & labeled ground-truth evaluation. (No live webcam built).
    </p>
</div>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)
def get_face_data():
    raw = load_all_datasets()
    pipeline = run_pipeline(raw)
    gt = pipeline.get("face_ground_truth", pd.DataFrame())
    events = pipeline.get("events", pd.DataFrame())
    return gt, events

ground_truth, events = get_face_data()

# 1. Frequency Analysis & KPIs
freq_res = analyze_face_frequencies(events, ground_truth)
dur_res = analyze_absence_durations(events, ground_truth)

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Total Face Checks</div>
    <div class="kpi-value">{freq_res['total_checks']}</div>
    <div class="kpi-delta" style="color:#6366f1;">● Telemetry Samples</div></div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Face Present</div>
    <div class="kpi-value">{freq_res['face_present_count']}</div>
    <div class="kpi-delta" style="color:#10b981;">● {freq_res['face_present_pct']}% of checks</div></div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Face Absent</div>
    <div class="kpi-value">{freq_res['face_absent_count']}</div>
    <div class="kpi-delta" style="color:#ef4444;">● {freq_res['face_absent_pct']}% of checks</div></div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Detection Unavailable</div>
    <div class="kpi-value">{freq_res['detection_unavailable_count']}</div>
    <div class="kpi-delta" style="color:#f59e0b;">● {freq_res['detection_unavailable_pct']}% obscure</div></div>""", unsafe_allow_html=True)
with k5:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Avg Absence Time</div>
    <div class="kpi-value">{dur_res['mean_duration_seconds']:.1f}s</div>
    <div class="kpi-delta" style="color:#8b5cf6;">● Max: {dur_res['max_duration_seconds']:.0f}s</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 2. Charts: Detection Frequency Pie + Absence Timeline
c1, c2 = st.columns([1, 1.3])

with c1:
    st.markdown("""<div class="content-card"><div class="content-card-title">Detection Frequency Breakdown</div>
    <div class="content-card-subtitle">Distribution of face detection states</div>""", unsafe_allow_html=True)
    if not freq_res["frequency_table"].empty:
        fig_pie = px.pie(
            freq_res["frequency_table"], values="Count", names="State",
            color="State",
            color_discrete_map={
                "Face Present": "#10b981",
                "Face Absent": "#ef4444",
                "Detection Unavailable": "#f59e0b"
            },
            hole=0.5
        )
        apply_dark_theme(fig_pie, height=280)
        st.plotly_chart(fig_pie, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown("""<div class="content-card"><div class="content-card-title">Face Presence Timeline</div>
    <div class="content-card-subtitle">Chronological sequence of face verification states</div>""", unsafe_allow_html=True)
    if not ground_truth.empty and "timestamp" in ground_truth.columns:
        sample_gt = ground_truth.head(60).copy()
        sample_gt["timestamp"] = pd.to_datetime(sample_gt["timestamp"])
        fig_tl = px.scatter(
            sample_gt, x="timestamp", y="detected_state",
            color="detected_state",
            color_discrete_map={
                "face_present": "#10b981",
                "face_absent": "#ef4444",
                "detection_unavailable": "#f59e0b"
            },
            size=[10]*len(sample_gt),
            hover_data=["session_id", "confidence"]
        )
        apply_dark_theme(fig_tl, height=280)
        st.plotly_chart(fig_tl, use_container_width=True)
    else:
        st.info("Ground-truth timeline logs not available.")
    st.markdown("</div>", unsafe_allow_html=True)

# 3. Formal Ground-Truth Model Evaluation Metrics
st.markdown("""<div class="content-card"><div class="content-card-title">Ground-Truth Detection Accuracy & Error Rates</div>
<div class="content-card-subtitle">Rigorous statistical evaluation against labeled ground truth</div>""", unsafe_allow_html=True)

eval_res = evaluate_face_detection_accuracy(ground_truth)

if eval_res["reliable"]:
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Accuracy", f"{eval_res['accuracy']*100:.2f}%")
    m2.metric("Precision", f"{eval_res['precision']*100:.2f}%")
    m3.metric("Recall (Sensitivity)", f"{eval_res['recall']*100:.2f}%")
    m4.metric("F1-Score", f"{eval_res['f1']*100:.2f}%")
    m5.metric("False Positive Rate (FPR)", f"{eval_res['fpr']*100:.2f}%")
    m6.metric("False Negative Rate (FNR)", f"{eval_res['fnr']*100:.2f}%")
    
    col_t, col_cm = st.columns([1.2, 1])
    with col_t:
        st.markdown("##### Detailed Metric Breakdown")
        st.dataframe(eval_res["metrics_table"], use_container_width=True, hide_index=True)
    
    with col_cm:
        st.markdown("##### Confusion Matrix")
        st.dataframe(eval_res["confusion_matrix_df"], use_container_width=True)
        st.caption(f"True Positives (TP): {eval_res['tp']} | True Negatives (TN): {eval_res['tn']} | False Positives (FP): {eval_res['fp']} | False Negatives (FN): {eval_res['fn']}")
else:
    st.warning(eval_res["message"])

st.markdown("</div>", unsafe_allow_html=True)

# 4. Lighting Condition Impact & Absence by Session
l1, l2 = st.columns(2)

with l1:
    st.markdown("""<div class="content-card"><div class="content-card-title">Detection Comparison Across Lighting Conditions</div>
    <div class="content-card-subtitle">Normal vs Low Light vs Overexposed evaluation</div>""", unsafe_allow_html=True)
    light_df = compare_lighting_conditions(ground_truth)
    if not light_df.empty:
        st.dataframe(light_df, use_container_width=True, hide_index=True)
    else:
        st.info("Lighting condition telemetry not available.")
    st.markdown("</div>", unsafe_allow_html=True)

with l2:
    st.markdown("""<div class="content-card"><div class="content-card-title">Face Absence Duration by Session</div>
    <div class="content-card-subtitle">Top sessions with accumulated camera absence</div>""", unsafe_allow_html=True)
    sess_abs = dur_res["session_absence_summary"]
    if not sess_abs.empty:
        st.dataframe(sess_abs.head(10), use_container_width=True, hide_index=True)
    else:
        st.info("No face absence sessions detected.")
    st.markdown("</div>", unsafe_allow_html=True)

# 5. Model Limitations Report
st.markdown("""<div class="content-card"><div class="content-card-title">Engineering Limitations & Technical Safeguards</div>
<div class="content-card-subtitle">Documented model boundaries, environmental constraints, and ethical safeguards</div>""", unsafe_allow_html=True)
lims = generate_model_limitations_report()
st.dataframe(lims["limitations_table"], use_container_width=True, hide_index=True)
st.markdown("</div>", unsafe_allow_html=True)

