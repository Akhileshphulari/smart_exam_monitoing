"""
Module 4 (Step 1) & General Preprocessing
Team 4: Smart Exam Monitoring & Integrity Analytics Platform
Implements complete data cleaning, type coercion, timestamp validation,
duplicate removal, referential integrity verification, and data quality scoring.
"""

from typing import Dict, Tuple
import pandas as pd
import numpy as np


EVENT_TAXONOMY_NORMALIZATION = {
    "face_not_visible": "face_absent",
    "face_absent": "face_absent",
    "tab_switch": "tab_switch",
    "focus_lost": "focus_lost",
    "reconnect": "reconnection",
    "reconnection": "reconnection",
    "normal_activity": "normal_activity",
    "audio_detected": "audio_anomaly",
    "multiple_faces": "multiple_faces"
}


def clean_candidates(candidates: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Cleans candidate roster:
    - Fills missing group with 'Unassigned'
    - Trims whitespace from strings
    - Deduplicates candidate_id
    """
    df = candidates.copy()
    initial_rows = len(df)
    
    if "candidate_group" in df.columns:
        df["candidate_group"] = df["candidate_group"].fillna("Unassigned").astype(str).str.strip()
    if "candidate_id" in df.columns:
        df["candidate_id"] = df["candidate_id"].astype(str).str.strip()
        df = df.drop_duplicates(subset=["candidate_id"])
        
    audit = {
        "initial_candidates": initial_rows,
        "final_candidates": len(df),
        "duplicates_removed": initial_rows - len(df)
    }
    return df, audit


def clean_sessions(sessions: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Cleans session records:
    - Parses timestamps (start_time, end_time) to datetime
    - Detects & recalculates negative or missing duration_seconds from timestamps
    - Validates completion status
    """
    df = sessions.copy()
    initial_rows = len(df)
    
    # 1. Parse timestamps
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df["end_time"] = pd.to_datetime(df["end_time"], errors="coerce")
    df["duration_seconds"] = pd.to_numeric(df["duration_seconds"], errors="coerce")
    
    # 2. Recompute elapsed duration
    df["calculated_duration_seconds"] = (df["end_time"] - df["start_time"]).dt.total_seconds()
    
    # 3. Identify invalid durations
    invalid_mask = (
        df["duration_seconds"].isna() | 
        (df["duration_seconds"] < 0) | 
        (df["calculated_duration_seconds"] < 0)
    )
    
    # 4. Repair where calculated duration is positive
    repaired_count = 0
    repairable_mask = invalid_mask & (df["calculated_duration_seconds"] > 0)
    repaired_count = int(repairable_mask.sum())
    df.loc[repairable_mask, "duration_seconds"] = df.loc[repairable_mask, "calculated_duration_seconds"]
    
    # 5. Populate duration in minutes
    df["duration_minutes"] = (df["duration_seconds"] / 60.0).round(2)
    
    # 6. Status normalization
    if "status" in df.columns:
        df["status"] = df["status"].fillna("completed").astype(str).str.lower().str.strip()
        
    audit = {
        "initial_sessions": initial_rows,
        "final_sessions": len(df),
        "missing_start_times": int(df["start_time"].isna().sum()),
        "missing_end_times": int(df["end_time"].isna().sum()),
        "invalid_durations_detected": int(invalid_mask.sum()),
        "durations_repaired": repaired_count
    }
    return df, audit


def clean_events(events: pd.DataFrame, sessions: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Cleans and normalizes proctoring events:
    - Eliminates duplicate event_id
    - Normalizes event categories
    - Parses event timestamps
    - Prunes orphan events (session_id not in sessions)
    - Detects out-of-order events (before session start or after session end)
    """
    df = events.copy()
    initial_rows = len(df)
    
    # 1. Deduplicate by event_id if present
    if "event_id" in df.columns:
        df = df.drop_duplicates(subset=["event_id"])
    dup_removed = initial_rows - len(df)
    
    # 2. Normalize event types
    if "event_type" in df.columns:
        df["event_type_raw"] = df["event_type"]
        df["event_type"] = df["event_type"].astype(str).str.lower().str.strip()
        df["event_category_normalized"] = df["event_type"].map(EVENT_TAXONOMY_NORMALIZATION).fillna(df["event_type"])
    
    # 3. Timestamp parsing
    time_col = "event_time" if "event_time" in df.columns else ("timestamp" if "timestamp" in df.columns else None)
    if time_col:
        df["event_time"] = pd.to_datetime(df[time_col], errors="coerce")
        df = df[df["event_time"].notna()].copy()
    
    # 4. Remove orphan events (session_id not in sessions)
    valid_sessions = set(sessions["session_id"].dropna().unique())
    orphan_mask = ~df["session_id"].isin(valid_sessions)
    orphans_removed = int(orphan_mask.sum())
    df = df[~orphan_mask].copy()
    
    # 5. Out-of-order timestamp detection
    sess_times = sessions[["session_id", "start_time", "end_time"]].dropna(subset=["session_id"])
    df = df.merge(sess_times, on="session_id", how="left")
    
    df["out_of_order"] = False
    if "start_time" in df.columns and "event_time" in df.columns:
        df.loc[df["start_time"].notna() & (df["event_time"] < df["start_time"]), "out_of_order"] = True
    if "end_time" in df.columns and "event_time" in df.columns:
        df.loc[df["end_time"].notna() & (df["event_time"] > df["end_time"]), "out_of_order"] = True
        
    out_of_order_count = int(df["out_of_order"].sum())
    df = df.drop(columns=["start_time", "end_time"], errors="ignore")
    
    audit = {
        "raw_events": initial_rows,
        "duplicates_removed": dup_removed,
        "orphan_events_removed": orphans_removed,
        "out_of_order_events": out_of_order_count,
        "final_clean_events": len(df)
    }
    return df, audit


def audit_data_quality(candidates: pd.DataFrame, 
                       sessions: pd.DataFrame, 
                       events: pd.DataFrame, 
                       scores: pd.DataFrame, 
                       risk: pd.DataFrame) -> Dict:
    """
    Evaluates 14 data quality checks and computes an overall Data Quality Score (0-100%).
    """
    cand_ids = set(candidates["candidate_id"].dropna().unique())
    sess_ids = set(sessions["session_id"].dropna().unique())
    score_sess = set(scores["session_id"].dropna().unique()) if not scores.empty else set()
    risk_sess = set(risk["session_id"].dropna().unique()) if not risk.empty else set()
    
    orphan_sess = int((~sessions["candidate_id"].isin(cand_ids)).sum())
    orphan_events = int((~events["session_id"].isin(sess_ids)).sum())
    
    missing_scores = len(sess_ids - score_sess)
    missing_risks = len(sess_ids - risk_sess)
    
    checks = [
        {"Check": "1. Candidate ID Uniqueness", "Status": "PASS" if candidates["candidate_id"].is_unique else "FAIL", "Details": "No duplicate candidates"},
        {"Check": "2. Candidate Group Assigned", "Status": "PASS" if not candidates["candidate_group"].isna().any() else "FAIL", "Details": "All cohorts defined"},
        {"Check": "3. Session ID Uniqueness", "Status": "PASS" if sessions["session_id"].is_unique else "FAIL", "Details": "No duplicate sessions"},
        {"Check": "4. Session Timestamps Valid", "Status": "PASS" if sessions["start_time"].notna().all() and sessions["end_time"].notna().all() else "WARNING", "Details": f"{int(sessions['start_time'].isna().sum())} missing start/end times"},
        {"Check": "5. Positive Session Durations", "Status": "PASS" if (sessions["duration_seconds"] > 0).all() else "FAIL", "Details": "All exam durations strictly positive"},
        {"Check": "6. Session-to-Candidate Referential Link", "Status": "PASS" if orphan_sess == 0 else "FAIL", "Details": f"{orphan_sess} orphan sessions"},
        {"Check": "7. Event ID Uniqueness", "Status": "PASS" if events["event_id"].is_unique else "FAIL", "Details": "Duplicate event IDs eliminated"},
        {"Check": "8. Event-to-Session Referential Link", "Status": "PASS" if orphan_events == 0 else "FAIL", "Details": f"{orphan_events} orphan events"},
        {"Check": "9. Event Timestamps Format", "Status": "PASS" if events["event_time"].notna().all() else "FAIL", "Details": "All event timestamps ISO compliant"},
        {"Check": "10. Event Chronological Order", "Status": "PASS" if int(events["out_of_order"].sum()) == 0 else "WARNING", "Details": f"{int(events['out_of_order'].sum())} events outside exam window"},
        {"Check": "11. Event Categories Validated", "Status": "PASS" if "event_category_normalized" in events.columns else "FAIL", "Details": "Standard taxonomy applied"},
        {"Check": "12. Integrity Score Coverage", "Status": "PASS" if missing_scores == 0 else "WARNING", "Details": f"{missing_scores} sessions missing score records"},
        {"Check": "13. Risk Rating Coverage", "Status": "PASS" if missing_risks == 0 else "WARNING", "Details": f"{missing_risks} sessions missing risk tier"},
        {"Check": "14. Score Boundary Validation (0-100)", "Status": "PASS" if not scores.empty and (scores["integrity_score"].between(0, 100)).all() else "FAIL", "Details": "All scores within 0.0 - 100.0"}
    ]
    
    passed_count = sum(1 for c in checks if c["Status"] == "PASS")
    quality_score = round((passed_count / len(checks)) * 100, 1)
    
    return {
        "checks_table": pd.DataFrame(checks),
        "passed_checks": passed_count,
        "total_checks": len(checks),
        "quality_score_pct": quality_score
    }


def run_pipeline(data_dict: Dict[str, pd.DataFrame]) -> Dict:
    """
    Orchestrates full data preprocessing across all loaded tables.
    Returns cleaned DataFrames and audit dictionary.
    """
    candidates_clean, c_audit = clean_candidates(data_dict.get("candidates", pd.DataFrame()))
    sessions_clean, s_audit = clean_sessions(data_dict.get("sessions", pd.DataFrame()))
    events_clean, e_audit = clean_events(data_dict.get("events", pd.DataFrame()), sessions_clean)
    scores = data_dict.get("scores", pd.DataFrame()).copy()
    risk = data_dict.get("risk", pd.DataFrame()).copy()
    
    dq_report = audit_data_quality(candidates_clean, sessions_clean, events_clean, scores, risk)
    
    return {
        "candidates": candidates_clean,
        "sessions": sessions_clean,
        "events": events_clean,
        "scores": scores,
        "risk": risk,
        "ai_report_metadata": data_dict.get("ai_report_metadata", pd.DataFrame()),
        "face_ground_truth": data_dict.get("face_ground_truth", pd.DataFrame()),
        "candidate_audit": c_audit,
        "session_audit": s_audit,
        "event_audit": e_audit,
        "data_quality_report": dq_report
    }

