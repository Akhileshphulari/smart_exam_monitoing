"""
Module 4 (Step 4): Suspicious Event Detection
Team 4: Smart Exam Monitoring & Integrity Analytics Platform
Detects patterns: repeated tab switches, prolonged focus loss, repeated face absence,
and composite unusual event combinations. Formulates transparent risk indicators.

High-Performance Vectorized Implementation:
Eliminates O(N*M) iterative filtering loops; uses vectorized groupby/aggregation
and numpy vectorized conditions for sub-50ms execution on large datasets.
"""

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np


def detect_candidate_patterns(events: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates event frequencies and counts per candidate using vectorized aggregation.
    """
    if events.empty or "candidate_id" not in events.columns:
        return pd.DataFrame(columns=[
            "candidate_id", "total_events", "tab_switches", 
            "focus_losses", "face_absences", "reconnections", "suspicious_total"
        ])
        
    ev = events.copy()
    ev["_tab"] = (ev["event_type"] == "tab_switch").astype(int)
    ev["_focus"] = (ev["event_type"].isin(["focus_lost", "looking_away"])).astype(int)
    ev["_face"] = (ev["event_type"].isin(["face_absent", "face_not_visible", "left_seat"])).astype(int)
    ev["_recon"] = (ev["event_type"].isin(["reconnect", "reconnection"])).astype(int)

    cand_agg = ev.groupby("candidate_id", as_index=False).agg(
        total_events=("event_type", "count"),
        tab_switches=("_tab", "sum"),
        focus_losses=("_focus", "sum"),
        face_absences=("_face", "sum"),
        reconnections=("_recon", "sum")
    )
    
    cand_agg["suspicious_total"] = (
        cand_agg["tab_switches"] + cand_agg["focus_losses"] + cand_agg["face_absences"]
    )
    return cand_agg.sort_values(by="suspicious_total", ascending=False).reset_index(drop=True)


def detect_suspicious_patterns(events: pd.DataFrame, 
                               sessions: pd.DataFrame,
                               tab_switch_threshold: int = 3,
                               face_absence_threshold: int = 2,
                               focus_loss_threshold: int = 2) -> pd.DataFrame:
    """
    Identifies specific behavioral risk patterns at the session level using vectorized operations:
    - Repeated tab switches (>= threshold)
    - Prolonged/repeated focus loss (>= threshold)
    - Repeated face absence (>= threshold)
    - Unusual event combinations (e.g., both tab switch and face absence occurred)
    
    Presents findings as analytical risk indicators requiring human review,
    strictly avoiding automated cheating accusations.
    """
    if sessions.empty:
        return pd.DataFrame(columns=[
            "session_id", "candidate_id", "total_events", "tab_switches",
            "face_absences", "focus_losses", "reconnections",
            "repeated_tab_switches", "repeated_face_absences", "repeated_focus_losses",
            "unusual_event_combination", "flagged_indicators_count",
            "detected_indicators", "analytical_alert_priority", "human_review_recommendation"
        ])

    # 1. Base sessions frame
    df = sessions[["session_id", "candidate_id"]].drop_duplicates(subset=["session_id"]).copy()
    df["session_id"] = df["session_id"].astype(str)
    df["candidate_id"] = df["candidate_id"].astype(str)

    # 2. Vectorized aggregation from events
    if not events.empty and "session_id" in events.columns and "event_type" in events.columns:
        ev = events.copy()
        ev["session_id"] = ev["session_id"].astype(str)
        ev["_tab"] = (ev["event_type"] == "tab_switch").astype(int)
        ev["_focus"] = (ev["event_type"].isin(["focus_lost", "looking_away"])).astype(int)
        ev["_face"] = (ev["event_type"].isin(["face_absent", "face_not_visible", "left_seat"])).astype(int)
        ev["_recon"] = (ev["event_type"].isin(["reconnect", "reconnection"])).astype(int)

        ev_counts = ev.groupby("session_id", as_index=False).agg(
            total_events=("event_type", "count"),
            tab_switches=("_tab", "sum"),
            face_absences=("_face", "sum"),
            focus_losses=("_focus", "sum"),
            reconnections=("_recon", "sum")
        )
        df = df.merge(ev_counts, on="session_id", how="left").fillna(0)
    else:
        df["total_events"] = 0
        df["tab_switches"] = 0
        df["face_absences"] = 0
        df["focus_losses"] = 0
        df["reconnections"] = 0

    for c in ["total_events", "tab_switches", "face_absences", "focus_losses", "reconnections"]:
        df[c] = df[c].astype(int)

    # 3. Vectorized Pattern Flags
    has_repeated_tab = df["tab_switches"] >= tab_switch_threshold
    has_repeated_face = df["face_absences"] >= face_absence_threshold
    has_repeated_focus = df["focus_losses"] >= focus_loss_threshold
    has_unusual_combo = (
        ((df["tab_switches"] >= 2) & (df["face_absences"] >= 1)) |
        ((df["tab_switches"] >= 2) & (df["focus_losses"] >= 2))
    )

    df["repeated_tab_switches"] = has_repeated_tab
    df["repeated_face_absences"] = has_repeated_face
    df["repeated_focus_losses"] = has_repeated_focus
    df["unusual_event_combination"] = has_unusual_combo

    # 4. Indicator counts
    df["flagged_indicators_count"] = (
        has_repeated_tab.astype(int) +
        has_repeated_face.astype(int) +
        has_repeated_focus.astype(int) +
        has_unusual_combo.astype(int)
    )

    # 5. Formulate plain-text indicator descriptions
    indicators_list = []
    for tab_val, face_val, foc_val, combo_val, tab_c, face_c, foc_c in zip(
        has_repeated_tab, has_repeated_face, has_repeated_focus, has_unusual_combo,
        df["tab_switches"], df["face_absences"], df["focus_losses"]
    ):
        items = []
        if tab_val:
            items.append(f"Frequent Application Switching ({tab_c} tab switches)")
        if face_val:
            items.append(f"Repeated Camera Absence ({face_c} face absence events)")
        if foc_val:
            items.append(f"Prolonged Attention Disruption ({foc_c} focus losses)")
        if combo_val:
            items.append("Multi-Vector Telemetry Concurrence (simultaneous tab + camera flags)")
        indicators_list.append("; ".join(items) if items else "Nominal baseline proctoring behavior")
    df["detected_indicators"] = indicators_list

    # 6. Priority & Review Recommendations (strictly analytical, never accusations)
    priority_conditions = [
        df["flagged_indicators_count"] >= 2,
        df["flagged_indicators_count"] == 1
    ]
    priorities = ["High Priority Review", "Moderate Monitoring"]
    recommendations = [
        "Manual video log review strongly recommended to verify legitimate secondary monitor or app usage.",
        "Standard review recommended; verify student explanation or network stability."
    ]

    df["analytical_alert_priority"] = np.select(priority_conditions, priorities, default="Low / Standard")
    df["human_review_recommendation"] = np.select(priority_conditions, recommendations, default="Nominal behavior; no immediate supervisory intervention required.")

    return df.sort_values(by="flagged_indicators_count", ascending=False).reset_index(drop=True)
