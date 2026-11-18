"""
Module 4 (Step 4): Suspicious Event Detection
Team 4: Smart Exam Monitoring & Integrity Analytics Platform
Detects patterns: repeated tab switches, prolonged focus loss, repeated face absence,
and composite unusual event combinations. Formulates transparent risk indicators.
"""

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np


def detect_candidate_patterns(events: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates event frequencies and counts per candidate.
    """
    if events.empty or "candidate_id" not in events.columns:
        return pd.DataFrame()
        
    cand_agg = events.groupby("candidate_id").agg(
        total_events=("event_id", "count"),
        tab_switches=("event_type", lambda s: (s == "tab_switch").sum()),
        focus_losses=("event_type", lambda s: (s == "focus_lost").sum()),
        face_absences=("event_type", lambda s: (s.isin(["face_absent", "face_not_visible"])).sum()),
        reconnections=("event_type", lambda s: (s.isin(["reconnect", "reconnection"])).sum())
    ).reset_index()
    
    cand_agg["suspicious_total"] = (
        cand_agg["tab_switches"] + cand_agg["focus_losses"] + cand_agg["face_absences"]
    )
    return cand_agg.sort_values(by="suspicious_total", ascending=False)


def detect_suspicious_patterns(events: pd.DataFrame, 
                               sessions: pd.DataFrame,
                               tab_switch_threshold: int = 3,
                               face_absence_threshold: int = 2,
                               focus_loss_threshold: int = 2) -> pd.DataFrame:
    """
    Identifies specific behavioral risk patterns at the session level:
    - Repeated tab switches (>= threshold)
    - Prolonged/repeated focus loss (>= threshold)
    - Repeated face absence (>= threshold)
    - Unusual event combinations (e.g., both tab switch and face absence occurred)
    
    Presents findings as analytical risk indicators requiring human review,
    strictly avoiding automated cheating accusations.
    """
    if sessions.empty:
        return pd.DataFrame()
        
    results = []
    
    for _, sess in sessions.iterrows():
        s_id = sess["session_id"]
        c_id = sess["candidate_id"]
        
        sess_ev = events[events["session_id"] == s_id] if not events.empty else pd.DataFrame()
        
        tab_cnt = int((sess_ev["event_type"] == "tab_switch").sum()) if not sess_ev.empty else 0
        focus_cnt = int((sess_ev["event_type"] == "focus_lost").sum()) if not sess_ev.empty else 0
        face_cnt = int((sess_ev["event_type"].isin(["face_absent", "face_not_visible"])).sum()) if not sess_ev.empty else 0
        recon_cnt = int((sess_ev["event_type"].isin(["reconnect", "reconnection"])).sum()) if not sess_ev.empty else 0
        total_ev = len(sess_ev)
        
        # Pattern flags
        has_repeated_tab = tab_cnt >= tab_switch_threshold
        has_repeated_face = face_cnt >= face_absence_threshold
        has_repeated_focus = focus_cnt >= focus_loss_threshold
        has_unusual_combo = (tab_cnt >= 2 and face_cnt >= 1) or (tab_cnt >= 2 and focus_cnt >= 2)
        
        # Risk indicators compile
        indicators = []
        if has_repeated_tab:
            indicators.append(f"Frequent Application Switching ({tab_cnt} tab switches)")
        if has_repeated_face:
            indicators.append(f"Repeated Camera Absence ({face_cnt} face absence events)")
        if has_repeated_focus:
            indicators.append(f"Prolonged Attention Disruption ({focus_cnt} focus losses)")
        if has_unusual_combo:
            indicators.append("Multi-Vector Telemetry Concurrence (simultaneous tab + camera flags)")
            
        indicator_count = len(indicators)
        
        # Analytical risk status (never cheating label)
        if indicator_count >= 2:
            alert_priority = "High Priority Review"
            review_recommendation = "Manual video log review strongly recommended to verify legitimate secondary monitor or app usage."
        elif indicator_count == 1:
            alert_priority = "Moderate Monitoring"
            review_recommendation = "Standard review recommended; verify student explanation or network stability."
        else:
            alert_priority = "Low / Standard"
            review_recommendation = "Nominal behavior; no immediate supervisory intervention required."
            
        results.append({
            "session_id": s_id,
            "candidate_id": c_id,
            "total_events": total_ev,
            "tab_switches": tab_cnt,
            "face_absences": face_cnt,
            "focus_losses": focus_cnt,
            "reconnections": recon_cnt,
            "repeated_tab_switches": has_repeated_tab,
            "repeated_face_absences": has_repeated_face,
            "repeated_focus_losses": has_repeated_focus,
            "unusual_event_combination": has_unusual_combo,
            "flagged_indicators_count": indicator_count,
            "detected_indicators": "; ".join(indicators) if indicators else "Nominal baseline proctoring behavior",
            "analytical_alert_priority": alert_priority,
            "human_review_recommendation": review_recommendation
        })
        
    df_res = pd.DataFrame(results)
    return df_res.sort_values(by="flagged_indicators_count", ascending=False)

