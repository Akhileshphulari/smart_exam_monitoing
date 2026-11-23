"""
Page 8: Reports & Exports
Smart Exam Monitoring & Integrity Analytics Platform - Team 4
Generate and export CSV analytical reports for all modules:
Cleaning, Session Analytics, Face Monitoring, Event Frequencies,
Outlier Detection, Suspicious Events, Statistical Validation, and Data Quality.
"""

import sys
from pathlib import Path
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.data_loader import load_all_datasets
from src.preprocessing import run_pipeline
from src.session_analysis import compute_session_metrics, get_session_kpis
from src.face_analysis import analyze_face_frequencies, evaluate_face_detection_accuracy, compare_lighting_conditions
from src.event_analysis import analyze_event_distributions, analyze_event_severities
from src.outlier_detection import build_outlier_feature_matrix, detect_iqr_outliers, detect_zscore_outliers, explain_outliers
from src.suspicious_detection import detect_suspicious_patterns
from src.statistical_validation import generate_data_quality_score
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

st.set_page_config(page_title="Reports & Exports | ExamGuard", page_icon="📑", layout="wide")

# Dark Navy Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #0a0e27; color: #e2e8f0; }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0a0e27 0%, #131842 100%) !important; }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    h1, h2, h3, h4, h5, h6 { color: #ffffff !important; }
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

# Header
st.markdown("""
<div style="background: linear-gradient(135deg, #1a1f4e 0%, #252b6a 100%); border: 1px solid #3b3f7a; border-radius: 14px; padding: 20px 24px; margin-bottom: 24px;">
    <h1 style="margin:0; font-size:1.55rem; font-weight:700; color:#ffffff !important;">Module 7: Analytical Report Generator</h1>
    <p style="margin:4px 0 0 0; font-size:0.88rem; color:#7b83b5;">
        Export Complete Preprocessing, Session, Face, Event, Outlier, and Suspicious Pattern Reports in CSV Format
    </p>
</div>
""", unsafe_allow_html=True)

@st.cache_data(ttl=600)
def build_all_reports():
    if "pipeline_data" in st.session_state:
        pipeline = st.session_state["pipeline_data"]
    else:
        raw = load_all_datasets()
        pipeline = run_pipeline(raw)

    candidates = pipeline["candidates"]
    sessions = pipeline["sessions"]
    events = pipeline["events"]
    scores = pipeline["scores"]
    gt = pipeline.get("face_ground_truth", pd.DataFrame())

    # 1. Cleaning report
    clean_rep = generate_cleaning_report({
        "candidate_audit": pipeline["candidate_audit"],
        "session_audit": pipeline["session_audit"],
        "event_audit": pipeline["event_audit"]
    })

    # 2. Session analytics report
    s_metrics = compute_session_metrics(sessions, events, scores)
    s_kpis = get_session_kpis(candidates, sessions, events)
    sess_rep = generate_session_analytics_report(s_metrics, s_kpis)

    # 3. Face report
    f_freq = analyze_face_frequencies(events, gt)
    f_acc = evaluate_face_detection_accuracy(gt)
    f_light = compare_lighting_conditions(gt)
    face_rep = generate_face_detection_report(f_freq, f_acc, f_light)

    # 4. Event frequency report
    e_dist = analyze_event_distributions(events)
    e_sev = analyze_event_severities(events)
    event_rep = generate_event_frequency_report(e_dist, e_sev)

    # 5. Outlier report
    f_mat = build_outlier_feature_matrix(sessions, events, scores)
    feats = ["duration_minutes", "total_event_count", "suspicious_activity_count"]
    iqr_df, bounds = detect_iqr_outliers(f_mat, feats)
    z_df, z_stats = detect_zscore_outliers(iqr_df, feats)
    expl_df = explain_outliers(z_df, feats, bounds, z_stats)
    outlier_rep = generate_outlier_report(z_df, expl_df)

    # 6. Suspicious pattern report
    susp_df = detect_suspicious_patterns(events, sessions)
    susp_rep = generate_suspicious_event_report(susp_df)

    # 7. Statistical validation report
    stat_score = generate_data_quality_score(events, sessions, scores, candidates)
    stat_rep = generate_statistical_validation_report(stat_score)

    # 8. Data quality report
    dq_rep = generate_data_quality_report(pipeline.get("data_quality_report", {}))

    return {
        "Data_Cleaning_Report": clean_rep,
        "Session_Analytics_Report": sess_rep,
        "Face_Detection_Analysis_Report": face_rep,
        "Event_Frequency_Report": event_rep,
        "Outlier_Detection_Report": outlier_rep,
        "Suspicious_Event_Report": susp_rep,
        "Statistical_Validation_Report": stat_rep,
        "Data_Quality_Report": dq_rep
    }

reports_dict = build_all_reports()

# Master Export Button
col_btn, col_info = st.columns([1, 2])
with col_btn:
    if st.button("🚀 Export All 8 Reports to Disk (CSV)", type="primary", use_container_width=True):
        exported_files = export_all_reports(reports_dict, str(REPORTS_DIR))
        st.success(f"Successfully exported {len(exported_files)} reports to `{REPORTS_DIR}`!")

with col_info:
    st.caption(f"Reports directory target: `{REPORTS_DIR}`. All reports are immediately downloadable via individual buttons below.")

st.markdown("<br>", unsafe_allow_html=True)

# Individual Report Inspection & Download Tabs
report_names = list(reports_dict.keys())
selected_tab = st.selectbox("Select Report to Inspect & Download:", report_names)

if selected_tab:
    rep_df = reports_dict[selected_tab]
    st.markdown(f"""<div class="content-card"><div class="content-card-title">{selected_tab.replace('_', ' ')}</div>
    <div class="content-card-subtitle">{len(rep_df)} rows generated</div>""", unsafe_allow_html=True)
    
    st.dataframe(rep_df, use_container_width=True, hide_index=True, height=350)
    
    csv_bytes = rep_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"📥 Download {selected_tab}.csv",
        data=csv_bytes,
        file_name=f"{selected_tab}.csv",
        mime="text/csv",
        key=f"dl_{selected_tab}"
    )
    st.markdown("</div>", unsafe_allow_html=True)

