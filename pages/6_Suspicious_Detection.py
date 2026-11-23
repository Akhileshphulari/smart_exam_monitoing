"""
Page 6: Sprint 4 – Suspicious Event Detection
Smart Exam Monitoring & Integrity Analytics Platform - Team 4
Identifies repeated tab switches, prolonged focus loss, repeated face absence,
and unusual event combinations. Presents findings as analytical risk indicators
requiring human review (never automated accusations).
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
from src.preprocessing import run_pipeline
from src.suspicious_detection import (
    detect_candidate_patterns,
    detect_suspicious_patterns
)
from src.outlier_detection import (
    build_outlier_feature_matrix,
    detect_iqr_outliers,
    detect_zscore_outliers,
    explain_outliers
)

st.set_page_config(page_title="Sprint 4 - Suspicious Event Detection | ExamGuard", page_icon="🚨", layout="wide")

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
    .notice-box {
        padding: 14px 18px; border-radius: 10px; background: rgba(99, 102, 241, 0.12);
        border: 1px solid #6366f1; color: #c5cae8; margin-bottom: 20px; font-size: 0.9rem;
    }
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
<div style="background: linear-gradient(135deg, #1a1f4e 0%, #252b6a 100%); border: 1px solid #3b3f7a; border-radius: 14px; padding: 20px 24px; margin-bottom: 20px;">
    <h1 style="margin:0; font-size:1.55rem; font-weight:700; color:#ffffff !important;">Sprint 4: Suspicious Event & Pattern Detection</h1>
    <p style="margin:4px 0 0 0; font-size:0.88rem; color:#7b83b5;">
        Multi-Vector Pattern Recognition, IQR/Z-Score Outlier Analysis & Transparent Review Priorities
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="notice-box">
    <b>Ethical Analytical Notice:</b> In strict compliance with proctoring standards, detected behavioral patterns are
    classified as <b>transparent analytical risk indicators requiring human supervisory review</b>. Candidates are
    never automatically accused or labeled as cheaters by this platform.
</div>
""", unsafe_allow_html=True)

@st.cache_data(ttl=600)
def get_pipeline_data():
    if "pipeline_data" in st.session_state:
        return st.session_state["pipeline_data"]
    raw = load_all_datasets()
    pipeline = run_pipeline(raw)
    return pipeline

data = get_pipeline_data()
sessions = data["sessions"]
events = data["events"]
scores = data["scores"]

# 1. Configurable Pattern Thresholds
with st.expander("⚙️ Configure Pattern Detection Sensitivity Thresholds", expanded=False):
    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        tab_thresh = st.slider("Repeated Tab Switches Threshold", 1, 10, 3)
    with tc2:
        face_thresh = st.slider("Repeated Face Absence Threshold", 1, 10, 2)
    with tc3:
        focus_thresh = st.slider("Prolonged Focus Loss Threshold", 1, 10, 2)

patterns_df = detect_suspicious_patterns(events, sessions, tab_thresh, face_thresh, focus_thresh)

# 2. Key Metrics
high_prio = int((patterns_df["analytical_alert_priority"] == "High Priority Review").sum()) if not patterns_df.empty else 0
mod_prio = int((patterns_df["analytical_alert_priority"] == "Moderate Monitoring").sum()) if not patterns_df.empty else 0
combo_count = int(patterns_df["unusual_event_combination"].sum()) if not patterns_df.empty else 0
rep_tabs = int(patterns_df["repeated_tab_switches"].sum()) if not patterns_df.empty else 0
rep_face = int(patterns_df["repeated_face_absences"].sum()) if not patterns_df.empty else 0

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-title">High Priority Review</div>
    <div class="kpi-value" style="color:#ef4444;">{high_prio}</div>
    <div class="kpi-delta" style="color:#ef4444;">● Multi-Vector Flags</div></div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Moderate Review</div>
    <div class="kpi-value" style="color:#f59e0b;">{mod_prio}</div>
    <div class="kpi-delta" style="color:#f59e0b;">● Single Vector</div></div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Unusual Combos</div>
    <div class="kpi-value">{combo_count}</div>
    <div class="kpi-delta" style="color:#8b5cf6;">● Concurrence</div></div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Repeated Tabs</div>
    <div class="kpi-value">{rep_tabs}</div>
    <div class="kpi-delta" style="color:#6366f1;">● Exceeded limit</div></div>""", unsafe_allow_html=True)
with k5:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Repeated Face Out</div>
    <div class="kpi-value">{rep_face}</div>
    <div class="kpi-delta" style="color:#10b981;">● Camera gaps</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 3. Session Patterns Table with Explanations
st.markdown("""<div class="content-card"><div class="content-card-title">Session-Level Suspicious Pattern Indicators</div>
<div class="content-card-subtitle">Transparent indicators with human review recommendations</div>""", unsafe_allow_html=True)

if not patterns_df.empty:
    display_cols = [
        "session_id", "candidate_id", "total_events", "tab_switches",
        "face_absences", "focus_losses", "analytical_alert_priority",
        "detected_indicators", "human_review_recommendation"
    ]
    st.dataframe(patterns_df[display_cols], use_container_width=True, hide_index=True, height=350)
st.markdown("</div>", unsafe_allow_html=True)

# 4. Outlier Detection Pipeline (Step 3: IQR & Z-Score)
st.markdown("""<div class="content-card"><div class="content-card-title">Outlier Detection Analysis (IQR & Z-Score)</div>
<div class="content-card-subtitle">Detect unusually high event counts and long durations using empirical thresholds</div>""", unsafe_allow_html=True)

feat_matrix = build_outlier_feature_matrix(sessions, events, scores)
features = ["duration_minutes", "total_event_count", "suspicious_activity_count"]
iqr_df, bounds = detect_iqr_outliers(feat_matrix, features)
z_df, z_stats = detect_zscore_outliers(iqr_df, features)
explanations_df = explain_outliers(z_df, features, bounds, z_stats)

o1, o2 = st.columns([1, 1.3])

with o1:
    st.markdown("##### Outlier Detection Summary")
    total_iqr = int(z_df["iqr_is_outlier"].sum())
    total_z = int(z_df["zscore_is_outlier"].sum())
    consensus = int((z_df["iqr_is_outlier"] & z_df["zscore_is_outlier"]).sum())
    
    st.metric("Tukey's IQR Outliers (1.5x IQR)", total_iqr)
    st.metric("Z-Score Outliers (|Z| > 2.5)", total_z)
    st.metric("Consensus Outliers (Both Methods)", consensus)

with o2:
    st.markdown("##### Plain-Language Explanations of Detected Outliers")
    if not explanations_df.empty:
        st.dataframe(explanations_df, use_container_width=True, hide_index=True, height=260)
    else:
        st.info("No consensus outliers detected in cohort.")

st.markdown("</div>", unsafe_allow_html=True)

# 5. Candidate Pattern Aggregation
st.markdown("""<div class="content-card"><div class="content-card-title">Candidate-Level Event Aggregation</div>
<div class="content-card-subtitle">Cumulative event frequencies across candidates</div>""", unsafe_allow_html=True)
cand_patterns = detect_candidate_patterns(events)
if not cand_patterns.empty:
    st.dataframe(cand_patterns, use_container_width=True, hide_index=True, height=280)
st.markdown("</div>", unsafe_allow_html=True)

