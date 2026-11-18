"""
Module 2: Sprint 2 Session Analytics
Team 4: Smart Exam Monitoring & Integrity Analytics Platform
Implements session duration metrics, event aggregations, candidate summaries,
data type validation, and missing-data statistics.
"""

from typing import Dict, Tuple
import pandas as pd
import numpy as np


def compute_session_metrics(sessions: pd.DataFrame, events: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """
    Computes session duration, total event count per session, and merges scores.
    Returns an enriched session-level metrics DataFrame.
    """
    df = sessions.copy()
    
    # 1. Ensure duration in minutes
    if "duration_seconds" in df.columns:
        df["duration_minutes"] = (pd.to_numeric(df["duration_seconds"], errors="coerce") / 60.0).round(2)
    else:
        df["duration_minutes"] = 0.0
        
    # 2. Total events per session (Pandas Series aggregation)
    if not events.empty and "session_id" in events.columns:
        ev_counts = events.groupby("session_id").size().rename("total_events")
        df = df.merge(ev_counts, on="session_id", how="left")
        df["total_events"] = df["total_events"].fillna(0).astype(int)
        
        # Events by severity if present
        if "severity" in events.columns:
            crit_counts = events[events["severity"] == 2].groupby("session_id").size().rename("critical_events")
            warn_counts = events[events["severity"] == 1].groupby("session_id").size().rename("warning_events")
            df = df.merge(crit_counts, on="session_id", how="left").merge(warn_counts, on="session_id", how="left")
            df["critical_events"] = df["critical_events"].fillna(0).astype(int)
            df["warning_events"] = df["warning_events"].fillna(0).astype(int)
    else:
        df["total_events"] = 0
        df["critical_events"] = 0
        df["warning_events"] = 0
        
    # 3. Merge scores
    if not scores.empty and "session_id" in scores.columns:
        df = df.merge(scores[["session_id", "integrity_score"]], on="session_id", how="left")
        df["integrity_score"] = pd.to_numeric(df["integrity_score"], errors="coerce")
        
    return df


def get_session_kpis(candidates: pd.DataFrame, sessions: pd.DataFrame, events: pd.DataFrame) -> Dict:
    """
    Computes high-level KPI summary:
    - Total candidates
    - Total sessions
    - Average session duration (minutes)
    - Total monitoring events
    - Missing data percentage across sessions
    """
    total_candidates = int(candidates["candidate_id"].nunique()) if not candidates.empty and "candidate_id" in candidates.columns else len(candidates)
    total_sessions = int(sessions["session_id"].nunique()) if not sessions.empty and "session_id" in sessions.columns else len(sessions)
    
    dur_series = pd.to_numeric(sessions["duration_seconds"], errors="coerce") if "duration_seconds" in sessions.columns else pd.Series(dtype=float)
    avg_duration_min = round(float(dur_series.mean() / 60.0), 2) if not dur_series.empty and dur_series.notna().any() else 0.0
    
    total_events = len(events) if not events.empty else 0
    
    # Missing data percentage in sessions
    total_cells = sessions.size if not sessions.empty else 1
    missing_cells = int(sessions.isna().sum().sum()) if not sessions.empty else 0
    missing_pct = round((missing_cells / total_cells) * 100, 2)
    
    return {
        "total_candidates": total_candidates,
        "total_sessions": total_sessions,
        "avg_duration_minutes": avg_duration_min,
        "total_events": total_events,
        "missing_data_pct": missing_pct
    }


def compute_duration_statistics(sessions: pd.DataFrame) -> pd.DataFrame:
    """
    Computes comprehensive synthetic & empirical duration statistics (Pandas Series analysis).
    """
    dur = pd.to_numeric(sessions["duration_seconds"], errors="coerce").dropna() / 60.0
    if dur.empty:
        return pd.DataFrame()
        
    Q1 = dur.quantile(0.25)
    Q3 = dur.quantile(0.75)
    IQR = Q3 - Q1
    
    stats_dict = {
        "Metric": [
            "Mean Duration (Minutes)",
            "Median Duration (Minutes)",
            "Std Deviation (Minutes)",
            "Minimum Duration (Minutes)",
            "25th Percentile / Q1",
            "75th Percentile / Q3",
            "Interquartile Range (IQR)",
            "Maximum Duration (Minutes)"
        ],
        "Value": [
            round(float(dur.mean()), 2),
            round(float(dur.median()), 2),
            round(float(dur.std()), 2),
            round(float(dur.min()), 2),
            round(float(Q1), 2),
            round(float(Q3), 2),
            round(float(IQR), 2),
            round(float(dur.max()), 2)
        ]
    }
    return pd.DataFrame(stats_dict)


def generate_candidate_summaries(candidates: pd.DataFrame, 
                                 session_metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Generates candidate-wise session aggregations:
    - Candidate ID & Group
    - Total exam sessions sat
    - Total events accumulated
    - Average integrity score
    - Average exam duration
    """
    if session_metrics.empty:
        return pd.DataFrame()
        
    grouped = session_metrics.groupby("candidate_id").agg(
        sessions_count=("session_id", "count"),
        total_events=("total_events", "sum"),
        avg_score=("integrity_score", "mean"),
        avg_duration_min=("duration_minutes", "mean")
    ).reset_index()
    
    if not candidates.empty and "candidate_group" in candidates.columns:
        cand_sub = candidates[["candidate_id", "candidate_group"]].drop_duplicates()
        grouped = grouped.merge(cand_sub, on="candidate_id", how="left")
        grouped["candidate_group"] = grouped["candidate_group"].fillna("Unassigned")
        
    grouped["avg_score"] = grouped["avg_score"].round(2)
    grouped["avg_duration_min"] = grouped["avg_duration_min"].round(2)
    
    return grouped.sort_values(by="total_events", ascending=False)


def calculate_missing_data_breakdown(data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Computes missing data metrics per dataset and per column.
    """
    rows = []
    for name, df in data_dict.items():
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            continue
        total_r = len(df)
        for col in df.columns:
            n_miss = int(df[col].isna().sum())
            if n_miss > 0:
                rows.append({
                    "Dataset": name,
                    "Column": col,
                    "Missing Count": n_miss,
                    "Total Rows": total_r,
                    "Missing %": round((n_miss / total_r) * 100, 2)
                })
                
    if not rows:
        return pd.DataFrame(columns=["Dataset", "Column", "Missing Count", "Total Rows", "Missing %"])
    return pd.DataFrame(rows).sort_values("Missing %", ascending=False)

