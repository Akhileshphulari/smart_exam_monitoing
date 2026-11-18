"""
Page 1: Overview Dashboard
Smart Exam Monitoring & Integrity Analytics Platform - Team 4
Displays KPI cards, summary charts, and system health overview.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure src is importable
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.data_loader import load_all_datasets
from src.preprocessing import run_pipeline
from src.session_analysis import get_session_kpis, compute_session_metrics
from src.event_analysis import analyze_event_distributions, analyze_hourly_patterns

st.set_page_config(page_title="Overview | ExamGuard", page_icon="📊", layout="wide")

# ---------- Dark Navy Theme ----------
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
    .insight-item {
        padding: 12px 16px; border-radius: 8px; background: #151a42;
        border-left: 4px solid #6366f1; margin-bottom: 10px; font-size: 0.88rem; color: #c5cae8;
    }
    [data-testid="stMetricValue"] { color: #ffffff !important; }
    [data-testid="stMetricLabel"] { color: #8b92c4 !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=300)
def load_data():
    raw = load_all_datasets()
    pipeline = run_pipeline(raw)
    return raw, pipeline


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


RISK_PALETTE = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981", "Unknown": "#6b7280"}

# ---------- Load Data ----------
try:
    raw_data, pipeline_data = load_data()
except Exception as e:
    st.error(f"Error loading datasets: {e}")
    st.info("Ensure CSV files exist in the `data/` directory.")
    st.stop()

candidates = pipeline_data["candidates"]
sessions = pipeline_data["sessions"]
events = pipeline_data["events"]
scores = pipeline_data["scores"]
risk = pipeline_data["risk"]

# ---------- Header ----------
st.markdown("""
<div style="background: linear-gradient(135deg, #1a1f4e 0%, #252b6a 100%); border: 1px solid #3b3f7a; border-radius: 14px; padding: 20px 24px; margin-bottom: 24px; display: flex; justify-content: space-between; align-items: center;">
    <div>
        <h1 style="margin:0; font-size:1.55rem; font-weight:700; color:#ffffff !important;">Smart Exam Monitoring & Integrity Analytics Platform</h1>
        <p style="margin:4px 0 0 0; font-size:0.88rem; color:#7b83b5;">Team 4 Data Science Module | Sprint 2, 3 & 4</p>
    </div>
    <div style="display:inline-flex; align-items:center; gap:6px; padding:6px 14px; background:rgba(16,185,129,0.15); border:1px solid #10b981; color:#10b981; border-radius:9999px; font-size:0.8rem; font-weight:600;">
        <span style="width:8px;height:8px;border-radius:50%;background:#10b981;"></span> System Active
    </div>
</div>
""", unsafe_allow_html=True)

# ---------- KPIs ----------
kpis = get_session_kpis(candidates, sessions, events)

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Total Candidates</div>
    <div class="kpi-value">{kpis['total_candidates']:,}</div>
    <div class="kpi-delta" style="color:#10b981;">Active Registry</div></div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Exam Sessions</div>
    <div class="kpi-value">{kpis['total_sessions']:,}</div>
    <div class="kpi-delta" style="color:#6366f1;">Monitored</div></div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Avg Duration</div>
    <div class="kpi-value">{kpis['avg_duration_minutes']:.1f}m</div>
    <div class="kpi-delta" style="color:#8b5cf6;">Per Session</div></div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Total Events</div>
    <div class="kpi-value">{kpis['total_events']:,}</div>
    <div class="kpi-delta" style="color:#f59e0b;">Telemetry Flags</div></div>""", unsafe_allow_html=True)
with k5:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Missing Data</div>
    <div class="kpi-value">{kpis['missing_data_pct']:.1f}%</div>
    <div class="kpi-delta" style="color:#ef4444;">Quality Gap</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------- Charts Row ----------
c1, c2, c3 = st.columns([1, 1, 1.2])

with c1:
    st.markdown("""<div class="content-card"><div class="content-card-title">Risk Distribution</div>
    <div class="content-card-subtitle">Sessions by risk tier</div>""", unsafe_allow_html=True)
    if not risk.empty and "risk_level" in risk.columns:
        risk_counts = risk["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["risk_level", "count"]
        fig_donut = px.pie(risk_counts, values="count", names="risk_level",
                           color="risk_level", color_discrete_map=RISK_PALETTE, hole=0.55)
        fig_donut.update_traces(textposition='inside', textinfo='percent+label',
                                marker=dict(line=dict(color='#111538', width=2)))
        apply_dark_theme(fig_donut, height=280)
        st.plotly_chart(fig_donut, use_container_width=True)
    else:
        st.info("Risk data not available.")
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown("""<div class="content-card"><div class="content-card-title">Integrity Score Distribution</div>
    <div class="content-card-subtitle">Histogram across thresholds</div>""", unsafe_allow_html=True)
    if not scores.empty and "integrity_score" in scores.columns:
        fig_hist = px.histogram(scores, x="integrity_score", nbins=14,
                                color_discrete_sequence=["#6366f1"])
        fig_hist.add_vline(x=65, line_dash="dash", line_color="#ef4444", annotation_text="High (<65)",
                           annotation=dict(font_color="#ef4444"))
        fig_hist.add_vline(x=80, line_dash="dash", line_color="#10b981", annotation_text="Low (>=80)",
                           annotation=dict(font_color="#10b981"))
        fig_hist.update_layout(showlegend=False, xaxis_title="Integrity Score", yaxis_title="Sessions")
        apply_dark_theme(fig_hist, height=280)
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("Scores data not available.")
    st.markdown("</div>", unsafe_allow_html=True)

with c3:
    st.markdown("""<div class="content-card"><div class="content-card-title">Event Trend Across Exam Day</div>
    <div class="content-card-subtitle">Hourly event distribution</div>""", unsafe_allow_html=True)
    hourly = analyze_hourly_patterns(events)
    if not hourly.empty:
        fig_trend = px.line(hourly, x="hour", y="event_count", markers=True, line_shape="spline")
        fig_trend.update_traces(line=dict(color="#8b5cf6", width=3), marker=dict(size=7, color="#8b5cf6"))
        fig_trend.update_layout(xaxis=dict(tickmode="linear", tick0=0, dtick=2),
                                xaxis_title="Hour (24h)", yaxis_title="Events")
        apply_dark_theme(fig_trend, height=280)
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("No temporal event data available.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Event Distribution & Data Quality ----------
d1, d2 = st.columns([1.4, 1])

with d1:
    st.markdown("""<div class="content-card"><div class="content-card-title">Event Type Distribution</div>
    <div class="content-card-subtitle">Frequency of monitoring event categories</div>""", unsafe_allow_html=True)
    dist = analyze_event_distributions(events)
    if not dist["frequency_table"].empty:
        fig_bar = px.bar(dist["frequency_table"], x="event_type", y="count", color="event_type",
                         text="percentage", color_discrete_sequence=px.colors.qualitative.Prism)
        fig_bar.update_traces(texttemplate="%{text}%", textposition="outside")
        fig_bar.update_layout(showlegend=False, xaxis_title="Event Type", yaxis_title="Count")
        apply_dark_theme(fig_bar, height=300)
        st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with d2:
    st.markdown("""<div class="content-card"><div class="content-card-title">Data Quality Summary</div>
    <div class="content-card-subtitle">Pipeline health indicators</div>""", unsafe_allow_html=True)
    dq = pipeline_data.get("data_quality_report", {})
    if dq:
        passed = dq.get("passed_checks", 0)
        total = dq.get("total_checks", 14)
        score = dq.get("quality_score_pct", 0)
        st.metric("Quality Score", f"{score}%")
        st.progress(passed / max(total, 1), text=f"{passed}/{total} checks passed")
        checks_df = dq.get("checks_table", pd.DataFrame())
        if not checks_df.empty:
            st.dataframe(checks_df, use_container_width=True, hide_index=True, height=250)
    else:
        st.info("Run preprocessing pipeline to generate quality report.")
    st.markdown("</div>", unsafe_allow_html=True)

