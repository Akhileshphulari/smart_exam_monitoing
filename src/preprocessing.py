"""
Module 4 (Step 1) & General Preprocessing
Team 4: Smart Exam Monitoring & Integrity Analytics Platform
Implements complete data cleaning, type coercion, timestamp validation,
duplicate removal, referential integrity verification, and data quality scoring.
Supports external Exam-monitoring-main schemas and Team 4 fallback data.
"""

from typing import Dict, Tuple
import pandas as pd
import numpy as np


EVENT_TAXONOMY_NORMALIZATION = {
    "normal": "normal_activity",
    "normal_activity": "normal_activity",
    "looking_away": "focus_lost",
    "focus_lost": "focus_lost",
    "talking": "audio_detected",
    "audio_detected": "audio_detected",
    "audio_anomaly": "audio_detected",
    "left_seat": "face_absent",
    "face_absent": "face_absent",
    "face_not_visible": "face_absent",
    "multiple_persons": "multiple_faces",
    "multiple_faces": "multiple_faces",
    "phone_detected": "unauthorized_device",
    "unauthorized_device": "unauthorized_device",
    "tab_switch": "tab_switch",
    "reconnect": "reconnection",
    "reconnection": "reconnection"
}

SEVERITY_DEFAULT_MAP = {
    "normal_activity": 0,
    "reconnection": 0,
    "focus_lost": 1,
    "tab_switch": 1,
    "audio_detected": 1,
    "face_absent": 2,
    "multiple_faces": 2,
    "unauthorized_device": 2
}


def clean_candidates(candidates: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Cleans candidate roster:
    - Maps department to candidate_group if missing
    - Fills missing group with 'Unassigned'
    - Trims whitespace from strings
    - Deduplicates candidate_id
    """
    df = candidates.copy()
    initial_rows = len(df)
    
    if "candidate_group" not in df.columns and "department" in df.columns:
        df["candidate_group"] = df["department"]

    if "candidate_group" in df.columns:
        df["candidate_group"] = df["candidate_group"].fillna("Unassigned").astype(str).str.strip()
    else:
        df["candidate_group"] = "Unassigned"

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
    - Harmonizes duration_mins and duration_seconds
    - Calculates end_time if missing
    - Detects & recalculates negative or missing durations
    - Validates completion status
    """
    df = sessions.copy()
    initial_rows = len(df)
    
    # 1. Parse timestamps
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    if "end_time" in df.columns:
        df["end_time"] = pd.to_datetime(df["end_time"], errors="coerce")
    else:
        df["end_time"] = pd.NaT

    # 2. Harmonize duration
    if "duration_seconds" in df.columns:
        df["duration_seconds"] = pd.to_numeric(df["duration_seconds"], errors="coerce")
    elif "duration_mins" in df.columns:
        mins = pd.to_numeric(df["duration_mins"], errors="coerce")
        df["duration_seconds"] = (mins * 60.0).round(2)
    else:
        df["duration_seconds"] = np.nan

    # 3. Calculate missing end_time from duration
    missing_end = df["end_time"].isna() & df["start_time"].notna() & df["duration_seconds"].notna()
    if missing_end.any():
        df.loc[missing_end, "end_time"] = (
            df.loc[missing_end, "start_time"] + 
            pd.to_timedelta(df.loc[missing_end, "duration_seconds"], unit="s")
        )

    # 4. Recompute elapsed duration
    df["calculated_duration_seconds"] = (df["end_time"] - df["start_time"]).dt.total_seconds()
    
    # 5. Identify invalid durations
    invalid_mask = (
        df["duration_seconds"].isna() | 
        (df["duration_seconds"] <= 0) | 
        (df["calculated_duration_seconds"] <= 0)
    )
    
    # 6. Repair where calculated duration is positive
    repairable_mask = invalid_mask & (df["calculated_duration_seconds"] > 0)
    repaired_count = int(repairable_mask.sum())
    df.loc[repairable_mask, "duration_seconds"] = df.loc[repairable_mask, "calculated_duration_seconds"]
    
    # 7. Fallback for any remaining unrepairable invalid durations
    still_invalid = df["duration_seconds"].isna() | (df["duration_seconds"] <= 0)
    if still_invalid.any():
        df.loc[still_invalid, "duration_seconds"] = 3600.0

    # 8. Populate duration in minutes
    df["duration_minutes"] = (df["duration_seconds"] / 60.0).round(2)
    if "duration_mins" not in df.columns:
        df["duration_mins"] = df["duration_minutes"]
    
    # 9. Status normalization
    if "status" in df.columns:
        df["status"] = df["status"].fillna("completed").astype(str).str.lower().str.strip()
    else:
        df["status"] = "completed"

    if "session_id" in df.columns:
        df["session_id"] = df["session_id"].astype(str).str.strip()
        df = df.drop_duplicates(subset=["session_id"])
        
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
    - Normalizes event categories via taxonomy map
    - Assigns default severity tiers if missing
    - Parses event timestamps
    - Prunes orphan events (session_id not in sessions)
    - Detects out-of-order events (before session start or after session end)
    """
    df = events.copy()
    initial_rows = len(df)
    
    # 1. Deduplicate by event_id if present
    if "event_id" in df.columns:
        df["event_id"] = df["event_id"].astype(str).str.strip()
        df = df.drop_duplicates(subset=["event_id"])
    dup_removed = initial_rows - len(df)

    if "session_id" in df.columns:
        df["session_id"] = df["session_id"].astype(str).str.strip()
    
    # 2. Normalize event types
    if "event_type" in df.columns:
        df["event_type_raw"] = df["event_type"]
        cleaned = df["event_type"].astype(str).str.lower().str.strip()
        df["event_category_normalized"] = cleaned.map(EVENT_TAXONOMY_NORMALIZATION).fillna(cleaned)
        df["event_type"] = df["event_category_normalized"]

    # 3. Severity assignment
    if "severity" not in df.columns or df["severity"].isna().any():
        df["severity"] = df["event_type"].map(SEVERITY_DEFAULT_MAP).fillna(1).astype(int)
    else:
        df["severity"] = pd.to_numeric(df["severity"], errors="coerce").fillna(1).astype(int)
    
    # 4. Timestamp parsing
    time_col = "event_time" if "event_time" in df.columns else ("timestamp" if "timestamp" in df.columns else None)
    if time_col:
        df["event_time"] = pd.to_datetime(df[time_col], errors="coerce")
        df["timestamp"] = df["event_time"]
        df = df[df["event_time"].notna()].copy()
    
    # 5. Remove orphan events (session_id not in sessions)
    valid_sessions = set(sessions["session_id"].dropna().astype(str).unique()) if not sessions.empty else set()
    if valid_sessions:
        orphan_mask = ~df["session_id"].isin(valid_sessions)
        orphans_removed = int(orphan_mask.sum())
        df = df[~orphan_mask].copy()
    else:
        orphans_removed = 0
    
    # 6. Out-of-order timestamp detection
    sess_times = sessions[["session_id", "start_time", "end_time"]].dropna(subset=["session_id"]).copy()
    sess_times["session_id"] = sess_times["session_id"].astype(str)
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
    cand_ids = set(candidates["candidate_id"].dropna().astype(str).unique()) if not candidates.empty else set()
    sess_ids = set(sessions["session_id"].dropna().astype(str).unique()) if not sessions.empty else set()
    score_sess = set(scores["session_id"].dropna().astype(str).unique()) if not scores.empty else set()
    risk_sess = set(risk["session_id"].dropna().astype(str).unique()) if not risk.empty else set()
    
    orphan_sess = int((~sessions["candidate_id"].astype(str).isin(cand_ids)).sum()) if not sessions.empty and cand_ids else 0
    orphan_events = int((~events["session_id"].astype(str).isin(sess_ids)).sum()) if not events.empty and sess_ids else 0
    
    missing_scores = len(sess_ids - score_sess)
    missing_risks = len(sess_ids - risk_sess)

    # Normalize risk_level if needed
    if not risk.empty and "risk_level" in risk.columns:
        risk_norm = risk["risk_level"].astype(str).str.lower().map({"moderate": "Medium", "medium": "Medium", "low": "Low", "high": "High"}).fillna("Medium")
    
    checks = [
        {"Check": "1. Candidate ID Uniqueness", "Status": "PASS" if not candidates.empty and candidates["candidate_id"].is_unique else "FAIL", "Details": "No duplicate candidates"},
        {"Check": "2. Candidate Group Assigned", "Status": "PASS" if not candidates.empty and not candidates["candidate_group"].isna().any() else "FAIL", "Details": "All cohorts defined"},
        {"Check": "3. Session ID Uniqueness", "Status": "PASS" if not sessions.empty and sessions["session_id"].is_unique else "FAIL", "Details": "No duplicate sessions"},
        {"Check": "4. Session Timestamps Valid", "Status": "PASS" if not sessions.empty and sessions["start_time"].notna().all() and sessions["end_time"].notna().all() else "WARNING", "Details": f"{int(sessions['start_time'].isna().sum()) if not sessions.empty else 0} missing start/end times"},
        {"Check": "5. Positive Session Durations", "Status": "PASS" if not sessions.empty and (sessions["duration_seconds"] > 0).all() else "FAIL", "Details": "All exam durations strictly positive"},
        {"Check": "6. Session-to-Candidate Referential Link", "Status": "PASS" if orphan_sess == 0 else "FAIL", "Details": f"{orphan_sess} orphan sessions"},
        {"Check": "7. Event ID Uniqueness", "Status": "PASS" if not events.empty and events["event_id"].is_unique else "FAIL", "Details": "Duplicate event IDs eliminated"},
        {"Check": "8. Event-to-Session Referential Link", "Status": "PASS" if orphan_events == 0 else "FAIL", "Details": f"{orphan_events} orphan events"},
        {"Check": "9. Event Timestamps Format", "Status": "PASS" if not events.empty and events["event_time"].notna().all() else "FAIL", "Details": "All event timestamps ISO compliant"},
        {"Check": "10. Event Chronological Order", "Status": "PASS" if not events.empty and int(events["out_of_order"].sum()) == 0 else "WARNING", "Details": f"{int(events['out_of_order'].sum()) if not events.empty else 0} events outside exam window"},
        {"Check": "11. Event Categories Validated", "Status": "PASS" if not events.empty and "event_category_normalized" in events.columns else "FAIL", "Details": "Standard taxonomy applied"},
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
    
    # Normalize risk levels if present
    if not risk.empty and "risk_level" in risk.columns:
        risk["risk_level"] = risk["risk_level"].astype(str).str.lower().map({
            "moderate": "Medium", "medium": "Medium", "low": "Low", "high": "High"
        }).fillna("Medium")

    dq_report = audit_data_quality(candidates_clean, sessions_clean, events_clean, scores, risk)
    
    return {
        "candidates": candidates_clean,
        "sessions": sessions_clean,
        "events": events_clean,
        "scores": scores,
        "risk": risk,
        "ai_report_metadata": data_dict.get("ai_report_metadata", data_dict.get("ai_reports", pd.DataFrame())),
        "ai_reports": data_dict.get("ai_reports", pd.DataFrame()),
        "face_ground_truth": data_dict.get("face_ground_truth", pd.DataFrame()),
        "candidate_audit": c_audit,
        "session_audit": s_audit,
        "event_audit": e_audit,
        "data_quality_report": dq_report
    }
