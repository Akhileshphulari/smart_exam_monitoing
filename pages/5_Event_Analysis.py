"""
Page 5: Sprint 4 – Event Distribution Analysis
Smart Exam Monitoring & Integrity Analytics Platform - Team 4
Analyzes frequency of tab switch, focus lost, face absent, reconnection,
and normal activity. Severity classification and temporal patterns.
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
from src.event_analysis import (
    analyze_event_distributions,
    analyze_event_severities,
    analyze_hourly_patterns
)

st.set_page_config(page_title="Sprint 4 - Event Analysis | ExamGuard", page_icon="📡", layout="wide")

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
    <h1 style="margin:0; font-size:1.55rem; font-weight:700; color:#ffffff !important;">Sprint 4: Event Distribution Analysis</h1>
    <p style="margin:4px 0 0 0; font-size:0.88rem; color:#7b83b5;">
        Frequency of Tab Switches, Focus Losses, Face Absences, Reconnections, and Severity Breakdown
    </p>
</div>
""", unsafe_allow_html=True)

@st.cache_data(ttl=600)
def get_clean_events():
    if "pipeline_data" in st.session_state:
        return st.session_state["pipeline_data"]["events"]
    raw = load_all_datasets()
    pipeline = run_pipeline(raw)
    return pipeline["events"]

events = get_clean_events()

# 1. Event Distribution Analysis
dist_res = analyze_event_distributions(events)
sp_counts = dist_res["specific_counts"]

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Normal Activity</div>
    <div class="kpi-value">{sp_counts.get('normal_activity', 0)}</div>
    <div class="kpi-delta" style="color:#10b981;">● Baseline Presence</div></div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Tab Switches</div>
    <div class="kpi-value">{sp_counts.get('tab_switch', 0)}</div>
    <div class="kpi-delta" style="color:#f59e0b;">● App Switching</div></div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Focus Losses</div>
    <div class="kpi-value">{sp_counts.get('focus_lost', 0)}</div>
    <div class="kpi-delta" style="color:#f59e0b;">● Attention Drops</div></div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Face Absences</div>
    <div class="kpi-value">{sp_counts.get('face_absent', 0)}</div>
    <div class="kpi-delta" style="color:#ef4444;">● Out of Frame</div></div>""", unsafe_allow_html=True)
with k5:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Reconnections</div>
    <div class="kpi-value">{sp_counts.get('reconnect', 0)}</div>
    <div class="kpi-delta" style="color:#6366f1;">● Network Resumes</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 2. Charts: Raw Frequency & Specific Categories
c1, c2 = st.columns(2)

with c1:
    st.markdown("""<div class="content-card"><div class="content-card-title">Event Category Breakdown</div>
    <div class="content-card-subtitle">Volume and percentage across all monitoring vectors</div>""", unsafe_allow_html=True)
    if not dist_res["frequency_table"].empty:
        fig_bar = px.bar(
            dist_res["frequency_table"], x="event_type", y="count",
            color="event_type", text="percentage",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig_bar.update_traces(texttemplate="%{text}%", textposition="outside")
        fig_bar.update_layout(showlegend=False, xaxis_title="Event Type", yaxis_title="Occurrences")
        apply_dark_theme(fig_bar, height=290)
        st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown("""<div class="content-card"><div class="content-card-title">Severity Level Distribution</div>
    <div class="content-card-subtitle">Tier 0 (Normal) vs Tier 1 (Warning) vs Tier 2 (Critical)</div>""", unsafe_allow_html=True)
    sev_df = analyze_event_severities(events)
    if not sev_df.empty:
        fig_pie = px.pie(
            sev_df, values="count", names="severity_label",
            color="severity_label",
            color_discrete_map={
                "Normal (Tier 0)": "#10b981",
                "Warning (Tier 1)": "#f59e0b",
                "Critical (Tier 2)": "#ef4444"
            },
            hole=0.5
        )
        apply_dark_theme(fig_pie, height=290)
        st.plotly_chart(fig_pie, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# 3. Detailed Summary Table & Hourly Patterns
t1, t2 = st.columns([1.3, 1])

with t1:
    st.markdown("""<div class="content-card"><div class="content-card-title">Target Event Frequency Audit Table</div>
    <div class="content-card-subtitle">Mapped directly to requirement specifications</div>""", unsafe_allow_html=True)
    if "specific_summary_df" in dist_res:
        st.dataframe(dist_res["specific_summary_df"], use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

with t2:
    st.markdown("""<div class="content-card"><div class="content-card-title">Hourly Pattern Breakdown</div>
    <div class="content-card-subtitle">Distribution of telemetry by examination hour</div>""", unsafe_allow_html=True)
    hourly_df = analyze_hourly_patterns(events)
    if not hourly_df.empty:
        st.dataframe(hourly_df, use_container_width=True, hide_index=True)
    else:
        st.info("No temporal patterns found.")
    st.markdown("</div>", unsafe_allow_html=True)

