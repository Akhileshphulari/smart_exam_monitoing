"""
Module 4 (Step 2): Event Distribution Analysis
Team 4: Smart Exam Monitoring & Integrity Analytics Platform
Computes event frequencies, normalized categories, severity breakdowns,
and hourly distribution patterns across the examination session logs.
"""

from typing import Dict
import pandas as pd
import numpy as np


def analyze_event_distributions(events: pd.DataFrame) -> Dict:
    """
    Analyzes frequency of:
    - Tab switch frequency
    - Focus lost frequency
    - Face absent frequency
    - Face not visible frequency
    - Reconnection frequency
    - Normal activity frequency
    """
    if events.empty or "event_type" not in events.columns:
        return {
            "total_events": 0,
            "frequency_table": pd.DataFrame(),
            "specific_counts": {}
        }
        
    df = events.copy()
    total = len(df)
    
    # Raw frequency
    raw_counts = df["event_type"].value_counts().reset_index()
    raw_counts.columns = ["event_type", "count"]
    raw_counts["percentage"] = (raw_counts["count"] / total * 100).round(2)
    
    # Specific requested frequency targets
    tab_switch_cnt = int((df["event_type"] == "tab_switch").sum())
    focus_lost_cnt = int((df["event_type"] == "focus_lost").sum())
    face_absent_cnt = int((df["event_type"].isin(["face_absent", "face_not_visible"])).sum())
    reconnect_cnt = int((df["event_type"].isin(["reconnect", "reconnection"])).sum())
    normal_cnt = int((df["event_type"] == "normal_activity").sum())
    
    specific_metrics = [
        {"Event Category": "Normal Activity Frequency", "Count": normal_cnt, "Percentage": f"{round(normal_cnt/total*100, 2)}%", "Severity Tier": "Tier 0 (Normal)"},
        {"Event Category": "Tab Switch Frequency", "Count": tab_switch_cnt, "Percentage": f"{round(tab_switch_cnt/total*100, 2)}%", "Severity Tier": "Tier 1 (Warning)"},
        {"Event Category": "Focus Lost Frequency", "Count": focus_lost_cnt, "Percentage": f"{round(focus_lost_cnt/total*100, 2)}%", "Severity Tier": "Tier 1 (Warning)"},
        {"Event Category": "Face Absent / Invisibility Frequency", "Count": face_absent_cnt, "Percentage": f"{round(face_absent_cnt/total*100, 2)}%", "Severity Tier": "Tier 2 (Critical)"},
        {"Event Category": "Reconnection Frequency", "Count": reconnect_cnt, "Percentage": f"{round(reconnect_cnt/total*100, 2)}%", "Severity Tier": "Tier 1 (Technical)"}
    ]
    
    return {
        "total_events": total,
        "frequency_table": raw_counts,
        "specific_summary_df": pd.DataFrame(specific_metrics),
        "specific_counts": {
            "tab_switch": tab_switch_cnt,
            "focus_lost": focus_lost_cnt,
            "face_absent": face_absent_cnt,
            "reconnect": reconnect_cnt,
            "normal_activity": normal_cnt
        }
    }


def analyze_event_severities(events: pd.DataFrame) -> pd.DataFrame:
    """
    Computes distribution by severity tier:
    0 = Normal, 1 = Warning, 2 = Critical
    """
    if events.empty or "severity" not in events.columns:
        return pd.DataFrame()
        
    sev_map = {0: "Normal (Tier 0)", 1: "Warning (Tier 1)", 2: "Critical (Tier 2)"}
    sev_df = events["severity"].value_counts().reset_index()
    sev_df.columns = ["severity_code", "count"]
    sev_df["severity_label"] = sev_df["severity_code"].map(sev_map).fillna("Unknown")
    sev_df["percentage"] = (sev_df["count"] / sev_df["count"].sum() * 100).round(2)
    return sev_df.sort_values(by="severity_code")


def analyze_hourly_patterns(events: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes temporal occurrence of events by hour of the examination day.
    """
    if events.empty:
        return pd.DataFrame()
        
    df = events.copy()
    time_col = "event_time" if "event_time" in df.columns else ("timestamp" if "timestamp" in df.columns else None)
    if not time_col:
        return pd.DataFrame()
        
    df["event_dt"] = pd.to_datetime(df[time_col], errors="coerce")
    df["hour"] = df["event_dt"].dt.hour
    valid_hours = df.dropna(subset=["hour"])
    
    if valid_hours.empty:
        return pd.DataFrame()
        
    hourly = valid_hours.groupby("hour").size().reset_index(name="event_count")
    hourly["hour_formatted"] = hourly["hour"].apply(lambda h: f"{int(h):02d}:00")
    hourly["percentage"] = (hourly["event_count"] / hourly["event_count"].sum() * 100).round(2)
    return hourly.sort_values("hour")

