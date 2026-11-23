"""
Page 2: Dataset Explorer
Smart Exam Monitoring & Integrity Analytics Platform - Team 4
Load, display, profile datasets and support CSV upload.
"""

import sys
from pathlib import Path
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.data_loader import load_all_datasets, get_dataset_summary, get_column_profiles, validate_uploaded_csv

st.set_page_config(page_title="Dataset Explorer | ExamGuard", page_icon="📁", layout="wide")

# Dark Navy Theme
st.markdown("""
<style>
    .stApp { background-color: #0a0e27; color: #e2e8f0; }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0a0e27 0%, #131842 100%) !important; }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    h1, h2, h3, h4, h5, h6 { color: #ffffff !important; }
    .content-card {
        background: #111538; border: 1px solid #2a2f5e; border-radius: 14px;
        padding: 22px 24px; margin-bottom: 20px;
    }
    .content-card-title { font-size: 1.1rem; font-weight: 700; color: #ffffff; margin-bottom: 4px; }
    .content-card-subtitle { font-size: 0.85rem; color: #7b83b5; margin-bottom: 16px; }
    .kpi-card {
        background: linear-gradient(135deg, #1a1f4e 0%, #252b6a 100%);
        border: 1px solid #3b3f7a; border-radius: 16px; padding: 18px 20px;
    }
    .kpi-title { font-size: 0.75rem; font-weight: 600; text-transform: uppercase; color: #8b92c4; margin-bottom: 4px; }
    .kpi-value { font-size: 1.5rem; font-weight: 700; color: #ffffff; }
    [data-testid="stMetricValue"] { color: #ffffff !important; }
    [data-testid="stMetricLabel"] { color: #8b92c4 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("# Dataset Explorer")
st.markdown("*Load, profile, and explore all examination monitoring datasets*")

# ---------- Load datasets ----------
@st.cache_data(ttl=600)
def load_data():
    if "raw_data" in st.session_state:
        return st.session_state["raw_data"]
    return load_all_datasets()

datasets = load_data()

# ---------- Dataset Selector ----------
dataset_names = {
    "candidates": "Candidates (candidates.csv)",
    "sessions": "Sessions (sessions.csv)",
    "events": "Events (events.csv)",
    "scores": "Scores (scores.csv)",
    "risk": "Risk (risk.csv / risk_data.csv)",
    "ai_reports": "AI Reports (ai_reports.csv)",
    "ai_report_metadata": "AI Report Metadata (ai_report_metadata.csv)",
    "face_ground_truth": "Face Monitoring Ground Truth (face_monitoring_ground_truth.csv)"
}

col1, col2 = st.columns([1, 3])

with col1:
    st.markdown("""<div class="content-card"><div class="content-card-title">Select Dataset</div>""", unsafe_allow_html=True)
    selected_key = st.selectbox("Choose a dataset:", list(dataset_names.keys()),
                                 format_func=lambda k: dataset_names[k], label_visibility="collapsed")
    
    df = datasets.get(selected_key, pd.DataFrame())
    summary = get_dataset_summary(df, dataset_names.get(selected_key, selected_key))
    
    st.markdown(f"""
    <div class="kpi-card" style="margin-top:12px;">
        <div class="kpi-title">Rows x Columns</div>
        <div class="kpi-value">{summary['rows']:,} x {summary['columns']}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.metric("Missing Values", f"{summary['missing_pct']}%")
    st.metric("Duplicate Rows", summary['duplicate_rows'])
    st.metric("Memory Usage", f"{summary['memory_usage_kb']:.1f} KB")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("""<div class="content-card"><div class="content-card-title">Dataset Records</div>
    <div class="content-card-subtitle">Showing loaded data from CSV files</div>""", unsafe_allow_html=True)
    
    if not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True, height=400)
        
        st.markdown("##### Data Types")
        dtypes_df = pd.DataFrame({
            "Column": df.columns,
            "Data Type": [str(dt) for dt in df.dtypes],
            "Category": ["Numerical" if pd.api.types.is_numeric_dtype(df[c]) else "Categorical/Text" for c in df.columns]
        })
        st.dataframe(dtypes_df, use_container_width=True, hide_index=True)
    else:
        st.warning(f"Dataset '{selected_key}' is empty or not found.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Column Profiles ----------
st.markdown("""<div class="content-card"><div class="content-card-title">Column Profiling</div>
<div class="content-card-subtitle">Detailed column statistics and sample values</div>""", unsafe_allow_html=True)

if not df.empty:
    profiles = get_column_profiles(df)
    if not profiles.empty:
        st.dataframe(profiles, use_container_width=True, hide_index=True)
    
    # Numerical columns: show describe()
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if num_cols:
        st.markdown("##### Numerical Column Statistics")
        st.dataframe(df[num_cols].describe().round(2), use_container_width=True)
    
    # Categorical columns: show value counts
    cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    if cat_cols:
        st.markdown("##### Categorical Column Value Counts")
        selected_cat = st.selectbox("Select categorical column:", cat_cols)
        if selected_cat:
            vc = df[selected_cat].value_counts().head(20).reset_index()
            vc.columns = [selected_cat, "Count"]
            st.dataframe(vc, use_container_width=True, hide_index=True)
else:
    st.info("Select a dataset to view profiling.")
st.markdown("</div>", unsafe_allow_html=True)

# ---------- CSV Upload ----------
st.markdown("""<div class="content-card"><div class="content-card-title">Upload Custom CSV</div>
<div class="content-card-subtitle">Upload your own examination data for analysis</div>""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    success, result = validate_uploaded_csv(uploaded_file)
    if success:
        st.success(f"Successfully loaded {len(result)} rows, {len(result.columns)} columns")
        up_summary = get_dataset_summary(result, uploaded_file.name)
        
        uk1, uk2, uk3, uk4 = st.columns(4)
        uk1.metric("Rows", up_summary["rows"])
        uk2.metric("Columns", up_summary["columns"])
        uk3.metric("Missing %", f"{up_summary['missing_pct']}%")
        uk4.metric("Duplicates", up_summary["duplicate_rows"])
        
        st.dataframe(result, use_container_width=True, hide_index=True, height=300)
        
        up_profiles = get_column_profiles(result)
        if not up_profiles.empty:
            st.markdown("##### Uploaded File Column Profiles")
            st.dataframe(up_profiles, use_container_width=True, hide_index=True)
    else:
        st.error(result)

st.markdown("</div>", unsafe_allow_html=True)

