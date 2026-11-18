"""
Smart Exam Monitoring & Integrity Analytics Platform
Team 4 Data Science Module — Sprint 2, Sprint 3, and Sprint 4
Enterprise Multi-Page Streamlit Analytics Application

Theme: Dark Navy Background (#0a0e27), Blue & Violet Accents (#6366f1, #8b5cf6), White Typography.
Strictly analytical indicators requiring human review (never automated accusations).
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure root directory
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.data_loader import (
    load_all_datasets,
    get_dataset_summary,
    get_column_profiles,
    validate_uploaded_csv
)
from src.preprocessing import run_pipeline, audit_data_quality
from src.session_analysis import (
    compute_session_metrics,
    get_session_kpis,
    compute_duration_statistics,
    generate_candidate_summaries,
    calculate_missing_data_breakdown
)
from src.face_analysis import (
    analyze_face_frequencies,
    analyze_absence_durations,
    compare_lighting_conditions,
    evaluate_face_detection_accuracy,
    generate_model_limitations_report
)
from src.event_analysis import (
    analyze_event_distributions,
    analyze_event_severities,
    analyze_hourly_patterns
)
from src.outlier_detection import (
    build_outlier_feature_matrix,
    detect_iqr_outliers,
    detect_zscore_outliers,
    explain_outliers
)
from src.suspicious_detection import (
    detect_candidate_patterns,
    detect_suspicious_patterns
)
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
from src.risk_analysis import (
    compute_risk_scores,
    get_indicator_contributions,
    get_risk_distribution,
    get_risk_summary_by_candidate,
    get_configurable_thresholds
)
from src.report_generator import (
    generate_cleaning_report,
    generate_session_analytics_report,
    generate_face_detection_report,
    generate_event_frequency_report,
    generate_outlier_report,
    generate_suspicious_event_report,
    generate_statistical_validation_report,
    generate_data_quality_report,
    export_all_reports,
    REPORTS_DIR
)

# ─────────────────────────────────────────────────────────────────────────────
# Streamlit Page Configuration & Dark Navy CSS Theme
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ExamGuard AI | Team 4 Data Science",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #e2e8f0;
    }

    .stApp {
        background-color: #0a0e27 !important;
        color: #e2e8f0;
    }

    /* ─── Dark Navy Sidebar ─── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #070a1e 0%, #0f153a 100%) !important;
        border-right: 1px solid #1e2554 !important;
    }
    section[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label {
        background: rgba(99, 102, 241, 0.06);
        border: 1px solid rgba(99, 102, 241, 0.15);
        padding: 8px 12px;
        border-radius: 8px;
        margin-bottom: 5px;
        transition: all 0.2s ease;
    }
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover {
        background: rgba(99, 102, 241, 0.25) !important;
        border-color: #6366f1 !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 700;
        letter-spacing: -0.3px;
    }

    /* ─── Top Brand Header ─── */
    .brand-header {
        background: linear-gradient(135deg, #13194a 0%, #1e2768 100%);
        border: 1px solid #2e3880;
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.35);
    }
    .brand-header h1 {
        margin: 0;
        font-size: 1.55rem;
        color: #ffffff !important;
    }
    .brand-header p {
        margin: 4px 0 0 0;
        font-size: 0.88rem;
        color: #94a3b8;
    }

    /* ─── Rounded KPI Cards ─── */
    .kpi-card {
        background: linear-gradient(135deg, #13194a 0%, #1c2463 100%);
        border: 1px solid #2e3880;
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: #6366f1;
    }
    .kpi-title {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 4px;
    }
    .kpi-delta {
        font-size: 0.82rem;
        font-weight: 500;
    }

    /* ─── Content Cards ─── */
    .content-card {
        background: #0f153a;
        border: 1px solid #1e265c;
        border-radius: 14px;
        padding: 22px 24px;
        margin-bottom: 20px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.25);
    }
    .content-card-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 4px;
    }
    .content-card-subtitle {
        font-size: 0.85rem;
        color: #8b92c4;
        margin-bottom: 16px;
    }

    /* ─── Ethical Notice Box ─── */
    .notice-box {
        padding: 14px 18px;
        border-radius: 10px;
        background: rgba(99, 102, 241, 0.12);
        border: 1px solid #6366f1;
        color: #cbd5e1;
        margin-bottom: 20px;
        font-size: 0.88rem;
        line-height: 1.45;
    }

    /* Streamlit Widget Overrides */
    [data-testid="stMetricValue"] { color: #ffffff !important; }
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Plotly Dark Navy Helper
# ─────────────────────────────────────────────────────────────────────────────
def apply_dark_theme(fig, height=340):
    fig.update_layout(
        font=dict(family="Inter, sans-serif", size=12, color="#c5cae8"),
        plot_bgcolor="#0f153a",
        paper_bgcolor="#0f153a",
        margin=dict(t=30, b=25, l=35, r=20),
        height=height,
        hoverlabel=dict(bgcolor="#1c2463", font_size=12, font_family="Inter", font_color="#ffffff")
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#1a204d", linecolor="#2e3880")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#1a204d", linecolor="#2e3880")
    return fig

RISK_PALETTE = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981", "Unknown": "#64748b"}

# ─────────────────────────────────────────────────────────────────────────────
# Data Loading & Pipeline Cache
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def get_platform_data():
    raw_datasets = load_all_datasets()
    pipeline_results = run_pipeline(raw_datasets)
    return raw_datasets, pipeline_results

try:
    raw_dict, pipeline_dict = get_platform_data()
except Exception as err:
    st.error(f"Error loading datasets: {err}")
    st.stop()

candidates = pipeline_dict["candidates"]
sessions = pipeline_dict["sessions"]
events = pipeline_dict["events"]
scores = pipeline_dict["scores"]
risk = pipeline_dict["risk"]
gt = pipeline_dict.get("face_ground_truth", pd.DataFrame())

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar: Navigation across 11 Master Pages & Global Filters
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:12px; padding: 10px 0 16px 0; border-bottom: 1px solid #1e2554; margin-bottom: 16px;">
        <div style="background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); width: 40px; height: 40px; border-radius: 10px; display:flex; align-items:center; justify-content:center; font-size: 1.3rem;">
            🛡️
        </div>
        <div>
            <div style="font-weight: 700; font-size: 1.05rem; color: #ffffff !important;">ExamGuard AI</div>
            <div style="font-size: 0.72rem; color: #94a3b8 !important; text-transform: uppercase; letter-spacing: 0.5px;">Team 4 Data Science</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    page_selection = st.radio(
        "Platform Modules",
        options=[
            "1. Overview Dashboard",
            "2. Dataset Explorer",
            "3. Sprint 2 – Session Analytics",
            "4. Sprint 3 – Face Monitoring",
            "5. Sprint 4 – Suspicious Events",
            "6. Data Preprocessing",
            "7. Outlier Detection",
            "8. Statistical Analysis",
            "9. Risk Analysis",
            "10. Data Quality Report",
            "11. Export Reports"
        ],
        index=0
    )

    st.markdown("<hr style='border:0; border-top:1px solid #1e2554; margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.75rem; text-transform:uppercase; color:#94a3b8; font-weight:600; letter-spacing:0.6px; margin-bottom:12px;'>Global Filters</div>", unsafe_allow_html=True)

    cand_search = st.text_input("Candidate Search", placeholder="e.g. C001, C015")
    selected_risk = st.selectbox("Risk Filter", ["All Risks", "High", "Medium", "Low"])

    st.markdown("<hr style='border:0; border-top:1px solid #1e2554; margin:16px 0;'>", unsafe_allow_html=True)
    if st.button("🔄 Refresh Data Cache", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("""
    <div style="text-align:center; font-size:0.72rem; color:#64748b; margin-top:12px;">
        Sprint 2 • Sprint 3 • Sprint 4<br>Infosys Virtual Internship
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Top Brand Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="brand-header">
    <div>
        <h1>Smart Exam Monitoring &amp; Integrity Analytics Platform</h1>
        <p>Team 4 Data Science Module • {page_selection}</p>
    </div>
    <div style="display:inline-flex; align-items:center; gap:8px; padding:6px 14px; background:rgba(16,185,129,0.15); border:1px solid #10b981; color:#10b981; border-radius:9999px; font-size:0.8rem; font-weight:600;">
        <span style="width:8px; height:8px; border-radius:50%; background:#10b981;"></span> Engine Active
    </div>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# 1. OVERVIEW DASHBOARD
# =============================================================================
if page_selection == "1. Overview Dashboard":
    kpis = get_session_kpis(candidates, sessions, events)
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Total Candidates</div><div class="kpi-value">{kpis['total_candidates']}</div><div class="kpi-delta" style="color:#10b981;">Active Registry</div></div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Exam Sessions</div><div class="kpi-value">{kpis['total_sessions']}</div><div class="kpi-delta" style="color:#6366f1;">Monitored</div></div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Avg Duration</div><div class="kpi-value">{kpis['avg_duration_minutes']:.1f}m</div><div class="kpi-delta" style="color:#8b5cf6;">Per Session</div></div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Monitoring Events</div><div class="kpi-value">{kpis['total_events']}</div><div class="kpi-delta" style="color:#f59e0b;">Telemetry Flags</div></div>""", unsafe_allow_html=True)
    with k5:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Missing Data %</div><div class="kpi-value">{kpis['missing_data_pct']:.2f}%</div><div class="kpi-delta" style="color:#ef4444;">Quality Gap</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1, 1.2])
    with c1:
        st.markdown("""<div class="content-card"><div class="content-card-title">Risk Distribution</div><div class="content-card-subtitle">Sessions by evaluated tier</div>""", unsafe_allow_html=True)
        if not risk.empty and "risk_level" in risk.columns:
            rc = risk["risk_level"].value_counts().reset_index()
            rc.columns = ["risk_level", "count"]
            f_donut = px.pie(rc, values="count", names="risk_level", color="risk_level", color_discrete_map=RISK_PALETTE, hole=0.55)
            apply_dark_theme(f_donut, height=270)
            st.plotly_chart(f_donut, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("""<div class="content-card"><div class="content-card-title">Integrity Scores</div><div class="content-card-subtitle">Histogram across thresholds</div>""", unsafe_allow_html=True)
        if not scores.empty and "integrity_score" in scores.columns:
            f_hist = px.histogram(scores, x="integrity_score", nbins=14, color_discrete_sequence=["#6366f1"])
            f_hist.add_vline(x=65, line_dash="dash", line_color="#ef4444", annotation_text="High (<65)")
            f_hist.add_vline(x=80, line_dash="dash", line_color="#10b981", annotation_text="Low (>=80)")
            apply_dark_theme(f_hist, height=270)
            st.plotly_chart(f_hist, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c3:
        st.markdown("""<div class="content-card"><div class="content-card-title">Hourly Pattern</div><div class="content-card-subtitle">Temporal anomaly frequency</div>""", unsafe_allow_html=True)
        hourly = analyze_hourly_patterns(events)
        if not hourly.empty:
            f_line = px.line(hourly, x="hour", y="event_count", markers=True, line_shape="spline")
            f_line.update_traces(line=dict(color="#8b5cf6", width=3), marker=dict(size=7, color="#8b5cf6"))
            apply_dark_theme(f_line, height=270)
            st.plotly_chart(f_line, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# 2. DATASET EXPLORER
# =============================================================================
elif page_selection == "2. Dataset Explorer":
    st.markdown("""<div class="content-card"><div class="content-card-title">Dataset Management &amp; Profiler</div><div class="content-card-subtitle">Browse records, column data types, missing values, duplicates, and upload custom CSVs</div>""", unsafe_allow_html=True)

    dataset_map = {
        "candidates": "Candidates (candidates.csv)",
        "sessions": "Sessions (sessions.csv)",
        "events": "Events (events.csv)",
        "scores": "Scores (scores.csv)",
        "risk": "Risk (risk.csv)",
        "ai_report_metadata": "AI Metadata (ai_report_metadata.csv)",
        "face_ground_truth": "Face Ground Truth (face_monitoring_ground_truth.csv)"
    }
    sel_key = st.selectbox("Select Dataset to Explore:", list(dataset_map.keys()), format_func=lambda k: dataset_map[k])
    active_df = raw_dict.get(sel_key, pd.DataFrame())

    sum_dict = get_dataset_summary(active_df, dataset_map.get(sel_key, sel_key))
    sk1, sk2, sk3, sk4 = st.columns(4)
    sk1.metric("Rows x Columns", f"{sum_dict['rows']} x {sum_dict['columns']}")
    sk2.metric("Missing %", f"{sum_dict['missing_pct']}%")
    sk3.metric("Duplicate Rows", sum_dict['duplicate_rows'])
    sk4.metric("Memory", f"{sum_dict['memory_usage_kb']:.1f} KB")

    if not active_df.empty:
        st.dataframe(active_df, use_container_width=True, hide_index=True, height=350)
        st.markdown("##### Column Profiles")
        profiles = get_column_profiles(active_df)
        st.dataframe(profiles, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# 3. SPRINT 2 – SESSION ANALYTICS
# =============================================================================
elif page_selection == "3. Sprint 2 – Session Analytics":
    st.markdown("""<div class="content-card"><div class="content-card-title">Sprint 2: Session Analytics Module</div><div class="content-card-subtitle">Duration metrics, candidate summaries, and missing data breakdown</div>""", unsafe_allow_html=True)

    s_metrics = compute_session_metrics(sessions, events, scores)
    dur_stats = compute_duration_statistics(sessions)
    cand_sum = generate_candidate_summaries(candidates, s_metrics)
    miss_break = calculate_missing_data_breakdown(pipeline_dict)

    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.markdown("##### Enriched Session Records")
        st.dataframe(s_metrics, use_container_width=True, hide_index=True, height=300)
    with col2:
        st.markdown("##### Duration Statistics (Quartiles & IQR)")
        st.dataframe(dur_stats, use_container_width=True, hide_index=True, height=300)

    st.markdown("##### Candidate-Wise Summaries")
    st.dataframe(cand_sum, use_container_width=True, hide_index=True, height=250)
    st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# 4. SPRINT 3 – FACE MONITORING
# =============================================================================
elif page_selection == "4. Sprint 3 – Face Monitoring":
    st.markdown("""<div class="content-card"><div class="content-card-title">Sprint 3: Face Monitoring Analytics</div><div class="content-card-subtitle">Presence frequencies, absence durations, lighting impacts, and ground-truth accuracy</div>""", unsafe_allow_html=True)

    face_freq = analyze_face_frequencies(events, gt)
    abs_dur = analyze_absence_durations(events, gt)
    f_eval = evaluate_face_detection_accuracy(gt)
    f_light = compare_lighting_conditions(gt)

    fk1, fk2, fk3, fk4 = st.columns(4)
    fk1.metric("Face Present", f"{face_freq['face_present_pct']}%")
    fk2.metric("Face Absent", f"{face_freq['face_absent_pct']}%")
    fk3.metric("Unavailable", f"{face_freq['detection_unavailable_pct']}%")
    fk4.metric("Avg Absence Time", f"{abs_dur['mean_duration_seconds']:.1f}s")

    if f_eval["reliable"]:
        st.markdown("##### Ground-Truth Evaluation Metrics")
        st.dataframe(f_eval["metrics_table"], use_container_width=True, hide_index=True)
        st.markdown("##### Confusion Matrix")
        st.dataframe(f_eval["confusion_matrix_df"], use_container_width=True)
    else:
        st.warning(f_eval["message"])

    st.markdown("##### Performance Across Lighting Conditions")
    st.dataframe(f_light, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# 5. SPRINT 4 – SUSPICIOUS EVENTS
# =============================================================================
elif page_selection == "5. Sprint 4 – Suspicious Events":
    st.markdown("""<div class="notice-box"><b>Notice:</b> Behavioral flags are analytical risk indicators requiring human supervisory review, strictly avoiding automated accusations of misconduct.</div>""", unsafe_allow_html=True)
    st.markdown("""<div class="content-card"><div class="content-card-title">Sprint 4: Suspicious Event &amp; Pattern Detection</div><div class="content-card-subtitle">Repeated tab switching, prolonged focus loss, repeated face absence, and multi-vector combinations</div>""", unsafe_allow_html=True)

    susp_patterns = detect_suspicious_patterns(events, sessions)
    st.dataframe(susp_patterns[[
        "session_id", "candidate_id", "total_events", "tab_switches",
        "face_absences", "focus_losses", "analytical_alert_priority",
        "detected_indicators", "human_review_recommendation"
    ]], use_container_width=True, hide_index=True, height=350)
    st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# 6. DATA PREPROCESSING
# =============================================================================
elif page_selection == "6. Data Preprocessing":
    st.markdown("""<div class="content-card"><div class="content-card-title">Data Preprocessing Pipeline (Module 4 Step 1)</div><div class="content-card-subtitle">Deduplication, datetime normalization, orphan pruning, and duration recalculation</div>""", unsafe_allow_html=True)

    c_audit = pipeline_dict["candidate_audit"]
    s_audit = pipeline_dict["session_audit"]
    e_audit = pipeline_dict["event_audit"]

    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown("##### Candidate Audit")
        st.write(f"- Raw Candidates: {c_audit.get('initial_candidates', 0)}")
        st.write(f"- Final Clean: {c_audit.get('final_candidates', 0)}")
        st.write(f"- Duplicates Removed: {c_audit.get('duplicates_removed', 0)}")
    with p2:
        st.markdown("##### Session Audit")
        st.write(f"- Raw Sessions: {s_audit.get('initial_sessions', 0)}")
        st.write(f"- Durations Repaired: {s_audit.get('durations_repaired', 0)}")
        st.write(f"- Missing Timestamps: {s_audit.get('missing_start_times', 0)}")
    with p3:
        st.markdown("##### Event Audit")
        st.write(f"- Raw Events: {e_audit.get('raw_events', 0)}")
        st.write(f"- Orphans Pruned: {e_audit.get('orphan_events_removed', 0)}")
        st.write(f"- Out-of-Order Flags: {e_audit.get('out_of_order_events', 0)}")
    st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# 7. OUTLIER DETECTION
# =============================================================================
elif page_selection == "7. Outlier Detection":
    st.markdown("""<div class="content-card"><div class="content-card-title">Outlier Detection Analysis (Module 4 Step 3)</div><div class="content-card-subtitle">Tukey's IQR &amp; Z-score outlier detection methods with plain-language explanations</div>""", unsafe_allow_html=True)

    f_matrix = build_outlier_feature_matrix(sessions, events, scores)
    features = ["duration_minutes", "total_event_count", "suspicious_activity_count"]
    iqr_df, bounds = detect_iqr_outliers(f_matrix, features)
    z_df, z_stats = detect_zscore_outliers(iqr_df, features)
    explanations = explain_outliers(z_df, features, bounds, z_stats)

    ok1, ok2, ok3 = st.columns(3)
    ok1.metric("IQR Outliers", int(z_df["iqr_is_outlier"].sum()))
    ok2.metric("Z-Score Outliers", int(z_df["zscore_is_outlier"].sum()))
    ok3.metric("Consensus Outliers", int((z_df["iqr_is_outlier"] & z_df["zscore_is_outlier"]).sum()))

    st.markdown("##### Detected Outliers with Plain-Language Explanations")
    st.dataframe(explanations, use_container_width=True, hide_index=True, height=300)
    st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# 8. STATISTICAL ANALYSIS
# =============================================================================
elif page_selection == "8. Statistical Analysis":
    st.markdown("""<div class="content-card"><div class="content-card-title">Statistical Validation (Module 4 Step 5)</div><div class="content-card-subtitle">Goodness-of-Fit Chi-Square test, Pearson correlation matrix, and descriptive statistics</div>""", unsafe_allow_html=True)

    chi_res = validate_event_distributions(events)
    st.info(chi_res["interpretation"])

    col_corr, col_desc = st.columns(2)
    with col_corr:
        st.markdown("##### Correlation Matrix")
        f_matrix = build_outlier_feature_matrix(sessions, events, scores)
        corr_m = compute_correlation_analysis(f_matrix)
        if not corr_m.empty:
            f_heat = px.imshow(corr_m, text_auto=True, color_continuous_scale="Viridis")
            apply_dark_theme(f_heat, height=280)
            st.plotly_chart(f_heat, use_container_width=True)
    with col_desc:
        st.markdown("##### Descriptive Statistics")
        desc = compute_descriptive_statistics(sessions, events, scores)
        st.dataframe(desc["summary_table"], use_container_width=True, hide_index=True, height=280)
    st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# 9. RISK ANALYSIS
# =============================================================================
elif page_selection == "9. Risk Analysis":
    st.markdown("""<div class="content-card"><div class="content-card-title">Configurable Risk Analytics (Module 5)</div><div class="content-card-subtitle">Transparent multi-indicator scoring into Low, Medium, and High Risk categories</div>""", unsafe_allow_html=True)

    risk_df = compute_risk_scores(sessions, events, scores)
    risk_dist = get_risk_distribution(risk_df)
    cand_risk = get_risk_summary_by_candidate(risk_df)

    rk1, rk2, rk3 = st.columns(3)
    rk1.metric("Low Risk Sessions", risk_dist.get("low_count", 0))
    rk2.metric("Medium Risk Sessions", risk_dist.get("medium_count", 0))
    rk3.metric("High Risk Sessions", risk_dist.get("high_count", 0))

    st.markdown("##### Session Risk Scoring & Weight Contribution")
    st.dataframe(risk_df, use_container_width=True, hide_index=True, height=300)

    st.markdown("##### Candidate-Level Risk Summary")
    st.dataframe(cand_risk, use_container_width=True, hide_index=True, height=250)
    st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# 10. DATA QUALITY REPORT
# =============================================================================
elif page_selection == "10. Data Quality Report":
    st.markdown("""<div class="content-card"><div class="content-card-title">Data Quality Report (14-Point Audit)</div><div class="content-card-subtitle">Completeness, Consistency, Accuracy, and Uniqueness health breakdown</div>""", unsafe_allow_html=True)

    dq_overall = generate_data_quality_score(events, sessions, scores, candidates)
    st.metric("Overall Quality Score", f"{dq_overall['overall_score']}%")
    st.dataframe(dq_overall["dimensions_table"], use_container_width=True, hide_index=True)

    dq_audit = pipeline_dict.get("data_quality_report", {})
    if dq_audit:
        st.markdown("##### 14-Point Verification Checklist")
        st.dataframe(dq_audit.get("checks_table", pd.DataFrame()), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# 11. EXPORT REPORTS
# =============================================================================
elif page_selection == "11. Export Reports":
    st.markdown("""<div class="content-card"><div class="content-card-title">Report Generator (Module 7)</div><div class="content-card-subtitle">Generate and download CSV reports across all platform modules</div>""", unsafe_allow_html=True)

    all_reps = {
        "Data_Cleaning_Report": generate_cleaning_report({
            "candidate_audit": pipeline_dict["candidate_audit"],
            "session_audit": pipeline_dict["session_audit"],
            "event_audit": pipeline_dict["event_audit"]
        }),
        "Session_Analytics_Report": generate_session_analytics_report(
            compute_session_metrics(sessions, events, scores),
            get_session_kpis(candidates, sessions, events)
        ),
        "Face_Detection_Report": generate_face_detection_report(
            analyze_face_frequencies(events, gt),
            evaluate_face_detection_accuracy(gt),
            compare_lighting_conditions(gt)
        ),
        "Event_Frequency_Report": generate_event_frequency_report(
            analyze_event_distributions(events),
            analyze_event_severities(events)
        ),
        "Outlier_Report": generate_outlier_report(
            build_outlier_feature_matrix(sessions, events, scores),
            explain_outliers(
                detect_zscore_outliers(
                    detect_iqr_outliers(
                        build_outlier_feature_matrix(sessions, events, scores),
                        ["duration_minutes", "total_event_count", "suspicious_activity_count"]
                    )[0],
                    ["duration_minutes", "total_event_count", "suspicious_activity_count"]
                )[0],
                ["duration_minutes", "total_event_count", "suspicious_activity_count"],
                {}, {}
            )
        ),
        "Suspicious_Event_Report": generate_suspicious_event_report(
            detect_suspicious_patterns(events, sessions)
        ),
        "Data_Quality_Report": generate_data_quality_report(pipeline_dict.get("data_quality_report", {}))
    }

    if st.button("🚀 Export All Reports to Disk (CSV)", type="primary"):
        res = export_all_reports(all_reps, str(REPORTS_DIR))
        st.success(f"Exported {len(res)} CSV reports to `{REPORTS_DIR}`!")

    sel_rep = st.selectbox("Inspect Report:", list(all_reps.keys()))
    if sel_rep:
        st.dataframe(all_reps[sel_rep], use_container_width=True, hide_index=True, height=300)
        st.download_button(
            label=f"📥 Download {sel_rep}.csv",
            data=all_reps[sel_rep].to_csv(index=False).encode("utf-8"),
            file_name=f"{sel_rep}.csv",
            mime="text/csv"
        )
    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Platform Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.8rem; padding: 24px 0 12px 0;">
    ExamGuard AI • Smart Exam Monitoring &amp; Integrity Analytics Platform • Team 4 Data Science Module (Sprint 2, 3 &amp; 4)
</div>
""", unsafe_allow_html=True)
