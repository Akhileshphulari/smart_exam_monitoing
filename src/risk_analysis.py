"""
Module 5: Risk Analytics
Team 4: Smart Exam Monitoring & Integrity Analytics Platform

Transparent, configurable risk scoring system for exam monitoring sessions.
Computes weighted multi-indicator risk scores from behavioral event telemetry,
normalizes indicators within the examination cohort, and classifies sessions
into Low / Medium / High analytical risk categories.

IMPORTANT: Risk scores are analytical alerts intended to support human review.
They are NOT proof of misconduct and must never be used to label candidates
as cheaters. All flagged sessions require manual verification.
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHTS: Dict[str, float] = {
    "tab_switch_freq": 0.25,
    "focus_loss_freq": 0.20,
    "face_absence_duration": 0.20,
    "unusual_event_count": 0.20,
    "session_data_quality": 0.15,
}

_DEFAULT_THRESHOLDS: Dict[str, Dict] = {
    "tab_switch_freq": {
        "weight": 0.25,
        "description": "Frequency of application / browser tab switches during the session.",
        "event_types": ["tab_switch"],
        "rationale": "Frequent tab switches may indicate reference to external material.",
    },
    "focus_loss_freq": {
        "weight": 0.20,
        "description": "Frequency of window focus-loss events recorded by the proctor client.",
        "event_types": ["focus_lost"],
        "rationale": "Repeated focus loss may indicate multi-window activity.",
    },
    "face_absence_duration": {
        "weight": 0.20,
        "description": "Count of face-absent or face-not-visible camera events.",
        "event_types": ["face_absent", "face_not_visible"],
        "rationale": "Prolonged camera absence may warrant visual log review.",
    },
    "unusual_event_count": {
        "weight": 0.20,
        "description": "Aggregate count of non-normal, non-reconnection events (composite unusual behaviour indicator).",
        "event_types": [],  # derived
        "rationale": "High unusual-event volume correlates with complex behavioural anomalies.",
    },
    "session_data_quality": {
        "weight": 0.15,
        "description": "Inverse data-quality indicator — proportion of missing or null fields in the session record.",
        "event_types": [],  # derived from session row
        "rationale": "Poor data quality may indicate client tampering or connectivity issues.",
    },
}

_RISK_CATEGORIES = {
    "Low": (0, 33),
    "Medium": (34, 66),
    "High": (67, 100),
}


# ---------------------------------------------------------------------------
# Helper: min-max normalisation within cohort
# ---------------------------------------------------------------------------

def _min_max_normalize(series: pd.Series) -> pd.Series:
    """
    Normalizes a pandas Series to the 0–100 scale using min-max scaling.
    Returns 0 when all values are identical (zero range).

    Parameters
    ----------
    series : pd.Series
        Raw numeric values to normalize.

    Returns
    -------
    pd.Series
        Normalized values in [0, 100].
    """
    s = pd.to_numeric(series, errors="coerce").fillna(0)
    s_min = s.min()
    s_max = s.max()
    if s_max == s_min:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return ((s - s_min) / (s_max - s_min) * 100).round(4)


# ---------------------------------------------------------------------------
# 1. compute_risk_scores
# ---------------------------------------------------------------------------

def compute_risk_scores(
    sessions: pd.DataFrame,
    events: pd.DataFrame,
    scores: pd.DataFrame,
    config: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Computes a transparent, weighted risk score for every exam session.

    For each session the function:
    1. Extracts raw indicator counts from the events log.
    2. Normalizes each indicator to 0–100 using min-max scaling across the
       examination cohort.
    3. Applies configurable weights to produce a composite weighted risk score.
    4. Classifies the session as Low (0-33), Medium (34-66), or High (67-100).

    Parameters
    ----------
    sessions : pd.DataFrame
        Session-level records; must contain ``session_id`` and ``candidate_id``.
    events : pd.DataFrame
        Event log; expected columns include ``session_id`` and ``event_type``.
    scores : pd.DataFrame
        Score records; optionally contains ``session_id`` and ``integrity_score``.
    config : dict, optional
        Weight overrides keyed by indicator name.  Unspecified indicators
        retain their default weight.  Keys:
        ``tab_switch_freq``, ``focus_loss_freq``, ``face_absence_duration``,
        ``unusual_event_count``, ``session_data_quality``.

    Returns
    -------
    pd.DataFrame
        One row per session with columns:
        ``session_id``, ``candidate_id``,
        ``raw_tab_switches``, ``raw_focus_losses``, ``raw_face_absences``,
        ``raw_unusual_events``, ``raw_data_quality_gaps``,
        ``norm_tab_switches``, ``norm_focus_losses``, ``norm_face_absences``,
        ``norm_unusual_events``, ``norm_data_quality_gaps``,
        ``weighted_risk_score``, ``risk_category``.
    """
    # Merge config with defaults
    weights = dict(_DEFAULT_WEIGHTS)
    if config:
        for key in weights:
            if key in config:
                weights[key] = float(config[key])

    # --- Handle empty inputs ---
    if sessions.empty:
        return pd.DataFrame(columns=[
            "session_id", "candidate_id",
            "raw_tab_switches", "raw_focus_losses", "raw_face_absences",
            "raw_unusual_events", "raw_data_quality_gaps",
            "norm_tab_switches", "norm_focus_losses", "norm_face_absences",
            "norm_unusual_events", "norm_data_quality_gaps",
            "weighted_risk_score", "risk_category",
        ])

    df = sessions[["session_id", "candidate_id"]].copy()

    # ----- Raw indicator extraction per session -----

    if not events.empty and "session_id" in events.columns and "event_type" in events.columns:
        ev = events.copy()

        # Tab switches
        tab_counts = (
            ev[ev["event_type"] == "tab_switch"]
            .groupby("session_id")
            .size()
            .rename("raw_tab_switches")
        )

        # Focus losses
        focus_counts = (
            ev[ev["event_type"] == "focus_lost"]
            .groupby("session_id")
            .size()
            .rename("raw_focus_losses")
        )

        # Face absences (includes face_not_visible)
        face_counts = (
            ev[ev["event_type"].isin(["face_absent", "face_not_visible"])]
            .groupby("session_id")
            .size()
            .rename("raw_face_absences")
        )

        # Unusual events: everything that is NOT normal_activity / reconnect / reconnection
        normal_types = {"normal_activity", "reconnect", "reconnection"}
        unusual_counts = (
            ev[~ev["event_type"].isin(normal_types)]
            .groupby("session_id")
            .size()
            .rename("raw_unusual_events")
        )

        df = df.merge(tab_counts, on="session_id", how="left")
        df = df.merge(focus_counts, on="session_id", how="left")
        df = df.merge(face_counts, on="session_id", how="left")
        df = df.merge(unusual_counts, on="session_id", how="left")
    else:
        df["raw_tab_switches"] = 0
        df["raw_focus_losses"] = 0
        df["raw_face_absences"] = 0
        df["raw_unusual_events"] = 0

    # Fill NaN counts with 0
    for col in ["raw_tab_switches", "raw_focus_losses", "raw_face_absences", "raw_unusual_events"]:
        df[col] = df[col].fillna(0).astype(int)

    # ----- Session data quality indicator -----
    # Proportion of null cells in the original session row (0 = perfect, 1 = all missing)
    session_nulls = sessions.set_index("session_id").isnull().sum(axis=1)
    session_total_cols = sessions.shape[1]
    quality_series = (session_nulls / session_total_cols * 100).rename("raw_data_quality_gaps")
    quality_series = quality_series.round(2)
    df = df.merge(quality_series, on="session_id", how="left")
    df["raw_data_quality_gaps"] = df["raw_data_quality_gaps"].fillna(0)

    # ----- Cohort-level min-max normalization (0-100) -----
    df["norm_tab_switches"] = _min_max_normalize(df["raw_tab_switches"])
    df["norm_focus_losses"] = _min_max_normalize(df["raw_focus_losses"])
    df["norm_face_absences"] = _min_max_normalize(df["raw_face_absences"])
    df["norm_unusual_events"] = _min_max_normalize(df["raw_unusual_events"])
    df["norm_data_quality_gaps"] = _min_max_normalize(df["raw_data_quality_gaps"])

    # ----- Weighted composite risk score -----
    df["weighted_risk_score"] = (
        df["norm_tab_switches"] * weights["tab_switch_freq"]
        + df["norm_focus_losses"] * weights["focus_loss_freq"]
        + df["norm_face_absences"] * weights["face_absence_duration"]
        + df["norm_unusual_events"] * weights["unusual_event_count"]
        + df["norm_data_quality_gaps"] * weights["session_data_quality"]
    ).round(2)

    # Clamp to [0, 100]
    df["weighted_risk_score"] = df["weighted_risk_score"].clip(0, 100)

    # ----- Risk category classification -----
    def _classify(score: float) -> str:
        if score <= 33:
            return "Low"
        elif score <= 66:
            return "Medium"
        else:
            return "High"

    df["risk_category"] = df["weighted_risk_score"].apply(_classify)

    return df


# ---------------------------------------------------------------------------
# 2. get_indicator_contributions
# ---------------------------------------------------------------------------

def get_indicator_contributions(
    risk_df: pd.DataFrame,
    config: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Breaks down the contribution of each indicator to the final weighted risk
    score for every session.

    Parameters
    ----------
    risk_df : pd.DataFrame
        Output of :func:`compute_risk_scores`.
    config : dict, optional
        Weight overrides; same schema as ``compute_risk_scores`` config.

    Returns
    -------
    pd.DataFrame
        Columns: ``session_id``, ``indicator_name``, ``raw_value``,
        ``normalized_value``, ``weight``, ``contribution_pct``.
    """
    if risk_df.empty:
        return pd.DataFrame(columns=[
            "session_id", "indicator_name", "raw_value",
            "normalized_value", "weight", "contribution_pct",
        ])

    weights = dict(_DEFAULT_WEIGHTS)
    if config:
        for key in weights:
            if key in config:
                weights[key] = float(config[key])

    indicator_map = {
        "tab_switch_freq": ("raw_tab_switches", "norm_tab_switches"),
        "focus_loss_freq": ("raw_focus_losses", "norm_focus_losses"),
        "face_absence_duration": ("raw_face_absences", "norm_face_absences"),
        "unusual_event_count": ("raw_unusual_events", "norm_unusual_events"),
        "session_data_quality": ("raw_data_quality_gaps", "norm_data_quality_gaps"),
    }

    rows: List[Dict] = []

    for _, session in risk_df.iterrows():
        s_id = session["session_id"]
        total_score = session.get("weighted_risk_score", 0)

        for indicator_name, (raw_col, norm_col) in indicator_map.items():
            raw_val = float(session.get(raw_col, 0))
            norm_val = float(session.get(norm_col, 0))
            w = weights[indicator_name]
            contribution = norm_val * w

            # Contribution percentage relative to total weighted score
            if total_score > 0:
                contrib_pct = round((contribution / total_score) * 100, 2)
            else:
                contrib_pct = 0.0

            rows.append({
                "session_id": s_id,
                "indicator_name": indicator_name,
                "raw_value": raw_val,
                "normalized_value": round(norm_val, 2),
                "weight": w,
                "contribution_pct": contrib_pct,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. get_risk_distribution
# ---------------------------------------------------------------------------

def get_risk_distribution(risk_df: pd.DataFrame) -> Dict:
    """
    Summarises the distribution of sessions across Low / Medium / High risk
    categories.

    Parameters
    ----------
    risk_df : pd.DataFrame
        Output of :func:`compute_risk_scores`; must contain ``risk_category``.

    Returns
    -------
    dict
        Keys:
        - ``total_sessions`` (int)
        - ``counts`` — dict mapping category name → count
        - ``percentages`` — dict mapping category name → percentage (float)
        - ``summary_table`` — pd.DataFrame with Category, Count, Percentage
    """
    if risk_df.empty or "risk_category" not in risk_df.columns:
        empty_summary = pd.DataFrame({
            "Category": ["Low", "Medium", "High"],
            "Count": [0, 0, 0],
            "Percentage": [0.0, 0.0, 0.0],
        })
        return {
            "total_sessions": 0,
            "counts": {"Low": 0, "Medium": 0, "High": 0},
            "percentages": {"Low": 0.0, "Medium": 0.0, "High": 0.0},
            "summary_table": empty_summary,
        }

    total = len(risk_df)
    cat_counts = risk_df["risk_category"].value_counts()

    counts: Dict[str, int] = {}
    percentages: Dict[str, float] = {}
    rows: List[Dict] = []

    for cat in ["Low", "Medium", "High"]:
        cnt = int(cat_counts.get(cat, 0))
        pct = round((cnt / total) * 100, 2) if total > 0 else 0.0
        counts[cat] = cnt
        percentages[cat] = pct
        rows.append({"Category": cat, "Count": cnt, "Percentage": pct})

    return {
        "total_sessions": total,
        "counts": counts,
        "percentages": percentages,
        "summary_table": pd.DataFrame(rows),
    }


# ---------------------------------------------------------------------------
# 4. get_risk_summary_by_candidate
# ---------------------------------------------------------------------------

def get_risk_summary_by_candidate(risk_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates risk analytics at the candidate level.

    Parameters
    ----------
    risk_df : pd.DataFrame
        Output of :func:`compute_risk_scores`; must contain ``candidate_id``,
        ``weighted_risk_score``, and ``risk_category``.

    Returns
    -------
    pd.DataFrame
        Columns: ``candidate_id``, ``total_sessions``, ``avg_risk_score``,
        ``max_risk_score``, ``overall_risk_category``.
    """
    if risk_df.empty or "candidate_id" not in risk_df.columns:
        return pd.DataFrame(columns=[
            "candidate_id", "total_sessions", "avg_risk_score",
            "max_risk_score", "overall_risk_category",
        ])

    grouped = risk_df.groupby("candidate_id").agg(
        total_sessions=("session_id", "count"),
        avg_risk_score=("weighted_risk_score", "mean"),
        max_risk_score=("weighted_risk_score", "max"),
    ).reset_index()

    grouped["avg_risk_score"] = grouped["avg_risk_score"].round(2)
    grouped["max_risk_score"] = grouped["max_risk_score"].round(2)

    # Classify using the candidate's maximum risk score
    def _classify_candidate(score: float) -> str:
        if score <= 33:
            return "Low"
        elif score <= 66:
            return "Medium"
        else:
            return "High"

    grouped["overall_risk_category"] = grouped["max_risk_score"].apply(
        _classify_candidate
    )

    return grouped.sort_values(by="max_risk_score", ascending=False)


# ---------------------------------------------------------------------------
# 5. get_configurable_thresholds
# ---------------------------------------------------------------------------

def get_configurable_thresholds() -> Dict:
    """
    Returns the default threshold and weight configuration with human-readable
    descriptions for each risk indicator.

    Returns
    -------
    dict
        Keys:
        - ``weights`` — default weight mapping
        - ``risk_categories`` — boundary definitions for Low / Medium / High
        - ``indicators`` — per-indicator metadata (weight, description,
          event_types, rationale)
        - ``disclaimer`` — transparency notice
    """
    return {
        "weights": dict(_DEFAULT_WEIGHTS),
        "risk_categories": dict(_RISK_CATEGORIES),
        "indicators": {k: dict(v) for k, v in _DEFAULT_THRESHOLDS.items()},
        "disclaimer": (
            "Risk scores are transparent analytical alerts derived from "
            "behavioural telemetry. They are NOT proof of misconduct. "
            "All flagged sessions require qualified human review before "
            "any administrative action is taken."
        ),
    }
