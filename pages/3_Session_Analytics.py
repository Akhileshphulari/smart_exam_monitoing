"""
Page 3: Sprint 2 – Session Analytics
Smart Exam Monitoring & Integrity Analytics Platform - Team 4
Session duration metrics, event counts per session, candidate summaries,
Pandas Series and DataFrame analysis, data type validation, missing-value handling.
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
from src.session_analysis import (
    compute_session_metrics,
    get_session_kpis,
    compute_duration_statistics,
    generate_candidate_summaries,
    calculate_missing_data_breakdown
)

st.set_page_config(page_title="Sprint 2 - Session Analytics | ExamGuard", page_icon="⏱️", layout="wide")

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
    <h1 style="margin:0; font-size:1.55rem; font-weight:700; color:#ffffff !important;">Sprint 2: Session Analytics Module</h1>
    <p style="margin:4px 0 0 0; font-size:0.88rem; color:#7b83b5;">Session Duration Metrics, Event Aggregations, Missing Data Analysis & Candidate Summaries</p>
</div>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)
def get_processed_data():
    raw = load_all_datasets()
    pipeline = run_pipeline(raw)
    return pipeline

data = get_processed_data()
candidates = data["candidates"]
sessions = data["sessions"]
events = data["events"]
scores = data["scores"]

# 1. High-Level KPIs
kpis = get_session_kpis(candidates, sessions, events)
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Total Candidates</div>
    <div class="kpi-value">{kpis['total_candidates']}</div>
    <div class="kpi-delta" style="color:#10b981;">● Active Registry</div></div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Total Sessions</div>
    <div class="kpi-value">{kpis['total_sessions']}</div>
    <div class="kpi-delta" style="color:#6366f1;">● Sat Exams</div></div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Avg Session Duration</div>
    <div class="kpi-value">{kpis['avg_duration_minutes']:.1f}m</div>
    <div class="kpi-delta" style="color:#8b5cf6;">● Minutes / Exam</div></div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Total Monitoring Events</div>
    <div class="kpi-value">{kpis['total_events']}</div>
    <div class="kpi-delta" style="color:#f59e0b;">● Telemetry Logs</div></div>""", unsafe_allow_html=True)
with k5:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Missing Data %</div>
    <div class="kpi-value">{kpis['missing_data_pct']:.2f}%</div>
    <div class="kpi-delta" style="color:#ef4444;">● Session Null Rate</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 2. Session Metrics Table & Duration Analysis
session_metrics = compute_session_metrics(sessions, events, scores)

col_s1, col_s2 = st.columns([1.5, 1])

with col_s1:
    st.markdown("""<div class="content-card"><div class="content-card-title">Enriched Session Metrics</div>
    <div class="content-card-subtitle">Session duration, total event counts, critical flags & integrity score</div>""", unsafe_allow_html=True)
    st.dataframe(session_metrics, use_container_width=True, hide_index=True, height=350)
    st.markdown("</div>", unsafe_allow_html=True)

with col_s2:
    st.markdown("""<div class="content-card"><div class="content-card-title">Duration Statistics (Pandas Series)</div>
    <div class="content-card-subtitle">Quartiles, IQR, Mean, Standard Deviation</div>""", unsafe_allow_html=True)
    dur_stats = compute_duration_statistics(sessions)
    st.dataframe(dur_stats, use_container_width=True, hide_index=True, height=350)
    st.markdown("</div>", unsafe_allow_html=True)

# 3. Visual Charts: Duration Distribution & Events Per Session
c1, c2 = st.columns(2)

with c1:
    st.markdown("""<div class="content-card"><div class="content-card-title">Session Duration Distribution</div>
    <div class="content-card-subtitle">Histogram of exam durations in minutes</div>""", unsafe_allow_html=True)
    if "duration_minutes" in session_metrics.columns:
        fig_dur = px.histogram(session_metrics, x="duration_minutes", nbins=15,
                               color_discrete_sequence=["#8b5cf6"])
        fig_dur.update_layout(xaxis_title="Duration (Minutes)", yaxis_title="Number of Sessions")
        apply_dark_theme(fig_dur, height=280)
        st.plotly_chart(fig_dur, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown("""<div class="content-card"><div class="content-card-title">Event Count vs Duration</div>
    <div class="content-card-subtitle">Relationship between exam duration and telemetry frequency</div>""", unsafe_allow_html=True)
    if not session_metrics.empty:
        fig_scat = px.scatter(session_metrics, x="duration_minutes", y="total_events",
                              color="critical_events" if "critical_events" in session_metrics.columns else None,
                              hover_data=["session_id", "candidate_id"],
                              color_continuous_scale=["#10b981", "#f59e0b", "#ef4444"])
        fig_scat.update_layout(xaxis_title="Duration (Minutes)", yaxis_title="Total Events Logged")
        apply_dark_theme(fig_scat, height=280)
        st.plotly_chart(fig_scat, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# 4. Candidate-Wise Summaries & Missing Data Breakdown
col_b1, col_b2 = st.columns([1.4, 1])

with col_b1:
    st.markdown("""<div class="content-card"><div class="content-card-title">Candidate-Wise Session Summaries</div>
    <div class="content-card-subtitle">Aggregated exam count, accumulated events, average score & duration</div>""", unsafe_allow_html=True)
    cand_summary = generate_candidate_summaries(candidates, session_metrics)
    st.dataframe(cand_summary, use_container_width=True, hide_index=True, height=320)
    st.markdown("</div>", unsafe_allow_html=True)

with col_b2:
    st.markdown("""<div class="content-card"><div class="content-card-title">Missing Data Analysis Breakdown</div>
    <div class="content-card-subtitle">Per-dataset and per-column missingness metrics</div>""", unsafe_allow_html=True)
    missing_df = calculate_missing_data_breakdown(data)
    if not missing_df.empty:
        st.dataframe(missing_df, use_container_width=True, hide_index=True, height=320)
    else:
        st.success("No missing values detected across cleaned tables.")
    st.markdown("</div>", unsafe_allow_html=True)

