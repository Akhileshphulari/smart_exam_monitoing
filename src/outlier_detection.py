"""
Module 4 (Step 3): Outlier Detection
Team 4: Smart Exam Monitoring & Integrity Analytics Platform
Implements IQR and Z-score outlier detection methods on session durations,
event counts, and suspicious activity. Generates plain-language explanations.
"""

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats


def build_outlier_feature_matrix(sessions: pd.DataFrame, events: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs a normalized feature matrix per session for statistical outlier analysis:
    - duration_seconds / duration_minutes
    - total_event_count
    - tab_switch_count
    - face_absent_count
    - focus_lost_count
    - integrity_score
    """
    df = sessions[["session_id", "candidate_id"]].copy()
    
    # Duration in minutes
    if "duration_seconds" in sessions.columns:
        df["duration_minutes"] = (pd.to_numeric(sessions["duration_seconds"], errors="coerce") / 60.0).round(2).fillna(0)
    else:
        df["duration_minutes"] = 0.0
        
    # Aggregate event metrics
    if not events.empty and "session_id" in events.columns:
        ev = events.copy()
        total_cnt = ev.groupby("session_id").size().rename("total_event_count")
        df = df.merge(total_cnt, on="session_id", how="left")
        df["total_event_count"] = df["total_event_count"].fillna(0).astype(int)
        
        for ev_type in ["tab_switch", "face_absent", "face_not_visible", "focus_lost"]:
            cnt = ev[ev["event_type"] == ev_type].groupby("session_id").size().rename(f"{ev_type}_count")
            df = df.merge(cnt, on="session_id", how="left")
            df[f"{ev_type}_count"] = df[f"{ev_type}_count"].fillna(0).astype(int)
            
        # Standardize face absent column
        if "face_not_visible_count" in df.columns:
            df["face_absence_count"] = df["face_absent_count"] + df["face_not_visible_count"]
        else:
            df["face_absence_count"] = df["face_absent_count"]
    else:
        df["total_event_count"] = 0
        df["tab_switch_count"] = 0
        df["face_absence_count"] = 0
        df["focus_lost_count"] = 0
        
    df["suspicious_activity_count"] = df["tab_switch_count"] + df["face_absence_count"] + df["focus_lost_count"]
    
    # Merge integrity score
    if not scores.empty and "session_id" in scores.columns:
        df = df.merge(scores[["session_id", "integrity_score"]], on="session_id", how="left")
        df["integrity_score"] = pd.to_numeric(df["integrity_score"], errors="coerce").fillna(70.0)
    else:
        df["integrity_score"] = 70.0
        
    return df


def detect_iqr_outliers(feature_df: pd.DataFrame, 
                        features: List[str], 
                        multiplier: float = 1.5) -> Tuple[pd.DataFrame, Dict]:
    """
    Applies Tukey's Interquartile Range (IQR) fence method:
    Outlier if Value < Q1 - multiplier*IQR or Value > Q3 + multiplier*IQR
    """
    df = feature_df.copy()
    bounds_info = {}
    
    for feat in features:
        series = df[feat].dropna()
        if series.empty:
            continue
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = round(Q1 - multiplier * IQR, 2)
        upper_bound = round(Q3 + multiplier * IQR, 2)
        
        bounds_info[feat] = {
            "Q1": round(Q1, 2),
            "Q3": round(Q3, 2),
            "IQR": round(IQR, 2),
            "lower_bound": lower_bound,
            "upper_bound": upper_bound
        }
        
        df[f"iqr_outlier_{feat}"] = (df[feat] < lower_bound) | (df[feat] > upper_bound)
        
    flag_cols = [f"iqr_outlier_{f}" for f in features if f"iqr_outlier_{f}" in df.columns]
    df["iqr_is_outlier"] = df[flag_cols].any(axis=1)
    
    return df, bounds_info


def detect_zscore_outliers(feature_df: pd.DataFrame, 
                           features: List[str], 
                           threshold: float = 2.5) -> Tuple[pd.DataFrame, Dict]:
    """
    Applies standard Z-score method:
    Outlier if |(X - Mean) / Std| > threshold
    """
    df = feature_df.copy()
    z_stats = {}
    
    for feat in features:
        series = df[feat].dropna()
        if series.empty:
            continue
        mean_val = series.mean()
        std_val = series.std(ddof=1) if len(series) > 1 else 1.0
        if std_val == 0:
            std_val = 1e-6
            
        z_scores = np.abs((df[feat] - mean_val) / std_val)
        df[f"zscore_{feat}"] = z_scores.round(3)
        df[f"zscore_outlier_{feat}"] = z_scores > threshold
        
        z_stats[feat] = {
            "mean": round(mean_val, 2),
            "std": round(std_val, 2),
            "threshold": threshold
        }
        
    flag_cols = [f"zscore_outlier_{f}" for f in features if f"zscore_outlier_{f}" in df.columns]
    df["zscore_is_outlier"] = df[flag_cols].any(axis=1)
    
    return df, z_stats


def explain_outliers(df_with_flags: pd.DataFrame, features: List[str], bounds_info: Dict, z_stats: Dict) -> pd.DataFrame:
    """
    Generates human-readable, plain-language explanations for all flagged session outliers.
    """
    explanations = []
    
    for _, row in df_with_flags.iterrows():
        s_id = row["session_id"]
        c_id = row["candidate_id"]
        is_iqr = row.get("iqr_is_outlier", False)
        is_z = row.get("zscore_is_outlier", False)
        
        if not (is_iqr or is_z):
            continue
            
        reasons = []
        for feat in features:
            val = row.get(feat, 0)
            feat_label = feat.replace("_", " ").title()
            
            # Check IQR
            if row.get(f"iqr_outlier_{feat}", False) and feat in bounds_info:
                b = bounds_info[feat]
                if val > b["upper_bound"]:
                    reasons.append(f"{feat_label} ({val}) exceeded upper IQR threshold ({b['upper_bound']})")
                elif val < b["lower_bound"]:
                    reasons.append(f"{feat_label} ({val}) fell below lower IQR threshold ({b['lower_bound']})")
                    
            # Check Z-score
            if row.get(f"zscore_outlier_{feat}", False) and feat in z_stats:
                z = row.get(f"zscore_{feat}", 0)
                reasons.append(f"{feat_label} Z-score of {z} deviates beyond 2.5 standard deviations from cohort mean ({z_stats[feat]['mean']})")
                
        reason_text = "; ".join(reasons) if reasons else "Multiple composite minor statistical deviations."
        consensus = "High Confidence (Both IQR & Z-score)" if (is_iqr and is_z) else ("Flagged by IQR" if is_iqr else "Flagged by Z-score")
        
        explanations.append({
            "Session ID": s_id,
            "Candidate ID": c_id,
            "Detection Consensus": consensus,
            "Total Events": row.get("total_event_count", 0),
            "Duration (Min)": row.get("duration_minutes", 0),
            "Integrity Score": row.get("integrity_score", 0),
            "Plain-Language Explanation": reason_text
        })
        
    return pd.DataFrame(explanations)

