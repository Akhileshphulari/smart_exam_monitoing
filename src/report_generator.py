"""
Module 7: Report Generation
Team 4: Smart Exam Monitoring & Integrity Analytics Platform
Generates structured report DataFrames from analytics outputs and exports
them as CSV files for downstream consumption by the Streamlit dashboard
and stakeholder review workflows.
"""

from typing import Dict, List
from pathlib import Path
import pandas as pd
import numpy as np

# Base directory for the project (two levels up from this file)
BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"


def generate_cleaning_report(audit_data: Dict) -> pd.DataFrame:
    """
    Flattens preprocessing audit dictionaries into a single human-readable
    report DataFrame with Section, Metric, and Value columns.

    Parameters
    ----------
    audit_data : Dict
        Must contain keys: 'candidate_audit', 'session_audit', 'event_audit',
        each mapping to a dict of metric_name -> numeric_value.

    Returns
    -------
    pd.DataFrame
        Columns: ['Section', 'Metric', 'Value']
    """
    if not audit_data:
        return pd.DataFrame(columns=["Section", "Metric", "Value"])

    section_map = {
        "candidate_audit": "Candidate Cleaning",
        "session_audit": "Session Cleaning",
        "event_audit": "Event Cleaning",
    }

    rows: List[Dict] = []
    for key, section_label in section_map.items():
        sub_dict = audit_data.get(key, {})
        if not isinstance(sub_dict, dict):
            continue
        for metric_name, metric_value in sub_dict.items():
            rows.append({
                "Section": section_label,
                "Metric": metric_name.replace("_", " ").title(),
                "Value": metric_value,
            })

    if not rows:
        return pd.DataFrame(columns=["Section", "Metric", "Value"])

    return pd.DataFrame(rows)


def generate_session_analytics_report(
    session_metrics: pd.DataFrame, kpis: Dict
) -> pd.DataFrame:
    """
    Combines session-level metrics with a KPI summary row appended at the
    bottom of the DataFrame.

    Parameters
    ----------
    session_metrics : pd.DataFrame
        Per-session metrics (e.g. from session_analysis.compute_session_metrics).
    kpis : Dict
        High-level KPIs (e.g. from session_analysis.get_session_kpis).

    Returns
    -------
    pd.DataFrame
        Session metrics with an appended KPI summary row (session_id = 'KPI_SUMMARY').
    """
    if session_metrics is None or session_metrics.empty:
        # Return KPIs as a single-row DataFrame if no session data
        if kpis:
            kpi_rows = [{"Metric": k.replace("_", " ").title(), "Value": v}
                        for k, v in kpis.items()]
            return pd.DataFrame(kpi_rows)
        return pd.DataFrame()

    report = session_metrics.copy()

    # Append a KPI summary row using the same columns where applicable
    if kpis:
        kpi_row: Dict = {}
        for col in report.columns:
            kpi_row[col] = np.nan
        kpi_row["session_id"] = "KPI_SUMMARY"
        if "candidate_id" in report.columns:
            kpi_row["candidate_id"] = "---"

        # Map known KPI values into matching columns
        kpi_col_map = {
            "total_candidates": "candidate_id",
            "total_sessions": "session_id",
            "avg_duration_minutes": "duration_minutes",
            "total_events": "total_events",
            "missing_data_pct": "missing_data_pct",
        }
        for kpi_key, col_name in kpi_col_map.items():
            if kpi_key in kpis and col_name in report.columns:
                kpi_row[col_name] = kpis[kpi_key]

        # For KPI fields that have no matching column, add them as new columns
        for kpi_key, kpi_val in kpis.items():
            if kpi_key not in kpi_col_map:
                col_label = kpi_key.replace("_", " ").title()
                report[col_label] = np.nan
                kpi_row[col_label] = kpi_val

        kpi_row["session_id"] = "KPI_SUMMARY"
        kpi_df = pd.DataFrame([kpi_row])
        report = pd.concat([report, kpi_df], ignore_index=True)

    return report


def generate_face_detection_report(
    freq_data: Dict, accuracy_data: Dict, lighting_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Consolidates face detection analytics into a single report DataFrame.

    Sections:
      1. Frequency Analysis – face-present / absent / unavailable counts.
      2. Accuracy Metrics   – precision, recall, F1 (if ground truth available).
      3. Lighting Comparison – per-condition accuracy and error rates.

    Parameters
    ----------
    freq_data : Dict
        Output from face_analysis.analyze_face_frequencies.
    accuracy_data : Dict
        Output from face_analysis.evaluate_face_detection_accuracy.
    lighting_df : pd.DataFrame
        Output from face_analysis.compare_lighting_conditions.

    Returns
    -------
    pd.DataFrame
        Columns: ['Section', 'Metric', 'Value']
    """
    rows: List[Dict] = []

    # --- Section 1: Frequency Analysis ---
    if freq_data:
        freq_metrics = {
            "Total Checks": freq_data.get("total_checks", 0),
            "Face Present Count": freq_data.get("face_present_count", 0),
            "Face Present %": freq_data.get("face_present_pct", 0.0),
            "Face Absent Count": freq_data.get("face_absent_count", 0),
            "Face Absent %": freq_data.get("face_absent_pct", 0.0),
            "Detection Unavailable Count": freq_data.get("detection_unavailable_count", 0),
            "Detection Unavailable %": freq_data.get("detection_unavailable_pct", 0.0),
        }
        for metric, value in freq_metrics.items():
            rows.append({"Section": "Frequency Analysis", "Metric": metric, "Value": value})

    # --- Section 2: Accuracy Metrics ---
    if accuracy_data:
        if accuracy_data.get("reliable", False):
            acc_metrics = {
                "Accuracy": accuracy_data.get("accuracy", 0),
                "Precision": accuracy_data.get("precision", 0),
                "Recall": accuracy_data.get("recall", 0),
                "F1-Score": accuracy_data.get("f1", 0),
                "False Positive Rate": accuracy_data.get("fpr", 0),
                "False Negative Rate": accuracy_data.get("fnr", 0),
                "True Positives": accuracy_data.get("tp", 0),
                "True Negatives": accuracy_data.get("tn", 0),
                "False Positives": accuracy_data.get("fp", 0),
                "False Negatives": accuracy_data.get("fn", 0),
            }
            for metric, value in acc_metrics.items():
                rows.append({"Section": "Accuracy Metrics", "Metric": metric, "Value": value})
        else:
            rows.append({
                "Section": "Accuracy Metrics",
                "Metric": "Status",
                "Value": accuracy_data.get("message", "Ground-truth unavailable"),
            })

    # --- Section 3: Lighting Comparison ---
    if lighting_df is not None and not lighting_df.empty:
        for _, row in lighting_df.iterrows():
            condition = row.get("Lighting Condition", "Unknown")
            for col in lighting_df.columns:
                if col == "Lighting Condition":
                    continue
                rows.append({
                    "Section": f"Lighting – {condition}",
                    "Metric": col,
                    "Value": row[col],
                })

    if not rows:
        return pd.DataFrame(columns=["Section", "Metric", "Value"])

    return pd.DataFrame(rows)


def generate_event_frequency_report(
    distributions: Dict, severity_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Combines event distribution analysis with severity breakdown into a
    unified report.

    Parameters
    ----------
    distributions : Dict
        Output from event_analysis.analyze_event_distributions.
    severity_df : pd.DataFrame
        Output from event_analysis.analyze_event_severities.

    Returns
    -------
    pd.DataFrame
        Columns: ['Section', 'Metric', 'Value']
    """
    rows: List[Dict] = []

    # --- Event Distributions ---
    if distributions:
        rows.append({
            "Section": "Event Overview",
            "Metric": "Total Events",
            "Value": distributions.get("total_events", 0),
        })

        # Specific counts
        specific = distributions.get("specific_counts", {})
        for event_name, count in specific.items():
            rows.append({
                "Section": "Event Frequency",
                "Metric": event_name.replace("_", " ").title(),
                "Value": count,
            })

        # Frequency table (raw counts with percentages)
        freq_table = distributions.get("frequency_table", pd.DataFrame())
        if freq_table is not None and not freq_table.empty:
            for _, frow in freq_table.iterrows():
                etype = frow.get("event_type", "unknown")
                count = frow.get("count", 0)
                pct = frow.get("percentage", 0.0)
                rows.append({
                    "Section": "Detailed Frequency Table",
                    "Metric": f"{etype} (count / %)",
                    "Value": f"{count} ({pct}%)",
                })

    # --- Severity Breakdown ---
    if severity_df is not None and not severity_df.empty:
        for _, srow in severity_df.iterrows():
            label = srow.get("severity_label", f"Code {srow.get('severity_code', '?')}")
            count = srow.get("count", 0)
            pct = srow.get("percentage", 0.0)
            rows.append({
                "Section": "Severity Breakdown",
                "Metric": label,
                "Value": f"{count} ({pct}%)",
            })

    if not rows:
        return pd.DataFrame(columns=["Section", "Metric", "Value"])

    return pd.DataFrame(rows)


def generate_outlier_report(
    outlier_df: pd.DataFrame, explanations_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merges statistical outlier detection flags with plain-language explanations
    for stakeholder review.

    Parameters
    ----------
    outlier_df : pd.DataFrame
        Feature matrix with IQR and Z-score outlier flags
        (from outlier_detection.detect_iqr_outliers / detect_zscore_outliers).
    explanations_df : pd.DataFrame
        Plain-language explanations (from outlier_detection.explain_outliers).

    Returns
    -------
    pd.DataFrame
        Combined report with outlier summary statistics followed by the
        per-session explanation rows.
    """
    rows: List[Dict] = []

    # --- Summary statistics ---
    if outlier_df is not None and not outlier_df.empty:
        total_sessions = len(outlier_df)
        iqr_outliers = int(outlier_df["iqr_is_outlier"].sum()) if "iqr_is_outlier" in outlier_df.columns else 0
        z_outliers = int(outlier_df["zscore_is_outlier"].sum()) if "zscore_is_outlier" in outlier_df.columns else 0
        both_mask = pd.Series([False] * total_sessions, index=outlier_df.index)
        if "iqr_is_outlier" in outlier_df.columns and "zscore_is_outlier" in outlier_df.columns:
            both_mask = outlier_df["iqr_is_outlier"] & outlier_df["zscore_is_outlier"]
        consensus_count = int(both_mask.sum())

        summary_rows = [
            {"Section": "Outlier Summary", "Metric": "Total Sessions Analyzed", "Value": total_sessions},
            {"Section": "Outlier Summary", "Metric": "IQR Outliers", "Value": iqr_outliers},
            {"Section": "Outlier Summary", "Metric": "Z-Score Outliers", "Value": z_outliers},
            {"Section": "Outlier Summary", "Metric": "Consensus Outliers (Both Methods)", "Value": consensus_count},
            {"Section": "Outlier Summary", "Metric": "IQR Outlier %", "Value": round(iqr_outliers / max(total_sessions, 1) * 100, 2)},
            {"Section": "Outlier Summary", "Metric": "Z-Score Outlier %", "Value": round(z_outliers / max(total_sessions, 1) * 100, 2)},
        ]
        rows.extend(summary_rows)

    # --- Detailed explanations ---
    if explanations_df is not None and not explanations_df.empty:
        for _, erow in explanations_df.iterrows():
            rows.append({
                "Section": "Outlier Detail",
                "Metric": f"Session {erow.get('Session ID', 'N/A')} "
                          f"(Candidate {erow.get('Candidate ID', 'N/A')})",
                "Value": erow.get("Plain-Language Explanation", ""),
            })

    if not rows:
        return pd.DataFrame(columns=["Section", "Metric", "Value"])

    return pd.DataFrame(rows)


def generate_suspicious_event_report(patterns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Formats suspicious pattern detection results into a report-ready DataFrame
    with analytical risk indicators and human review recommendations.

    Parameters
    ----------
    patterns_df : pd.DataFrame
        Output from suspicious_detection.detect_suspicious_patterns.

    Returns
    -------
    pd.DataFrame
        Report-ready version of the suspicious patterns analysis.
        Returns a copy with selected and renamed columns for readability,
        sorted by flagged indicator count descending.
    """
    if patterns_df is None or patterns_df.empty:
        return pd.DataFrame(columns=[
            "Session ID", "Candidate ID", "Total Events",
            "Tab Switches", "Face Absences", "Focus Losses",
            "Flagged Indicators", "Detected Indicators",
            "Alert Priority", "Review Recommendation",
        ])

    col_rename = {
        "session_id": "Session ID",
        "candidate_id": "Candidate ID",
        "total_events": "Total Events",
        "tab_switches": "Tab Switches",
        "face_absences": "Face Absences",
        "focus_losses": "Focus Losses",
        "reconnections": "Reconnections",
        "flagged_indicators_count": "Flagged Indicators",
        "detected_indicators": "Detected Indicators",
        "analytical_alert_priority": "Alert Priority",
        "human_review_recommendation": "Review Recommendation",
    }

    available_cols = [c for c in col_rename if c in patterns_df.columns]
    report = patterns_df[available_cols].copy()
    report = report.rename(columns=col_rename)

    sort_col = "Flagged Indicators" if "Flagged Indicators" in report.columns else None
    if sort_col:
        report = report.sort_values(by=sort_col, ascending=False).reset_index(drop=True)

    return report


def generate_statistical_validation_report(quality_data: Dict) -> pd.DataFrame:
    """
    Formats data quality audit results (from preprocessing.audit_data_quality)
    into a report DataFrame including the checks table and overall score.

    Parameters
    ----------
    quality_data : Dict
        Output from preprocessing.audit_data_quality containing:
        - 'checks_table': pd.DataFrame with Check, Status, Details columns
        - 'passed_checks': int
        - 'total_checks': int
        - 'quality_score_pct': float

    Returns
    -------
    pd.DataFrame
        Columns: ['Section', 'Metric', 'Value']
    """
    rows: List[Dict] = []

    if not quality_data:
        return pd.DataFrame(columns=["Section", "Metric", "Value"])

    # Overall score summary
    rows.append({
        "Section": "Quality Score",
        "Metric": "Data Quality Score",
        "Value": f"{quality_data.get('quality_score_pct', 0)}%",
    })
    rows.append({
        "Section": "Quality Score",
        "Metric": "Checks Passed",
        "Value": f"{quality_data.get('passed_checks', 0)} / {quality_data.get('total_checks', 0)}",
    })

    # Individual checks
    checks_table = quality_data.get("checks_table", pd.DataFrame())
    if checks_table is not None and not checks_table.empty:
        for _, crow in checks_table.iterrows():
            check_name = crow.get("Check", "Unknown")
            status = crow.get("Status", "N/A")
            details = crow.get("Details", "")
            rows.append({
                "Section": "Validation Checks",
                "Metric": check_name,
                "Value": f"{status} — {details}",
            })

    return pd.DataFrame(rows)


def generate_data_quality_report(dq_audit: Dict) -> pd.DataFrame:
    """
    Produces a concise data quality audit report with pass/fail checks
    from the preprocessing pipeline audit outputs.

    Parameters
    ----------
    dq_audit : Dict
        Combined audit dictionary that may contain:
        - 'candidate_audit': cleaning metrics for candidates
        - 'session_audit': cleaning metrics for sessions
        - 'event_audit': cleaning metrics for events
        - 'data_quality_report': output from audit_data_quality

    Returns
    -------
    pd.DataFrame
        Columns: ['Category', 'Check', 'Status', 'Details']
    """
    results: List[Dict] = []

    if not dq_audit:
        return pd.DataFrame(columns=["Category", "Check", "Status", "Details"])

    # --- Candidate quality checks ---
    c_audit = dq_audit.get("candidate_audit", {})
    if c_audit:
        dup_removed = c_audit.get("duplicates_removed", 0)
        results.append({
            "Category": "Candidates",
            "Check": "Duplicate Removal",
            "Status": "PASS" if dup_removed == 0 else "REMEDIATED",
            "Details": f"{dup_removed} duplicates removed",
        })
        results.append({
            "Category": "Candidates",
            "Check": "Final Record Count",
            "Status": "INFO",
            "Details": f"{c_audit.get('final_candidates', 'N/A')} candidates retained",
        })

    # --- Session quality checks ---
    s_audit = dq_audit.get("session_audit", {})
    if s_audit:
        invalid_dur = s_audit.get("invalid_durations_detected", 0)
        repaired = s_audit.get("durations_repaired", 0)
        results.append({
            "Category": "Sessions",
            "Check": "Invalid Durations",
            "Status": "PASS" if invalid_dur == 0 else "REMEDIATED",
            "Details": f"{invalid_dur} detected, {repaired} repaired",
        })
        missing_start = s_audit.get("missing_start_times", 0)
        missing_end = s_audit.get("missing_end_times", 0)
        results.append({
            "Category": "Sessions",
            "Check": "Timestamp Completeness",
            "Status": "PASS" if (missing_start == 0 and missing_end == 0) else "WARNING",
            "Details": f"{missing_start} missing start, {missing_end} missing end",
        })
        results.append({
            "Category": "Sessions",
            "Check": "Final Record Count",
            "Status": "INFO",
            "Details": f"{s_audit.get('final_sessions', 'N/A')} sessions retained",
        })

    # --- Event quality checks ---
    e_audit = dq_audit.get("event_audit", {})
    if e_audit:
        dup_removed = e_audit.get("duplicates_removed", 0)
        orphans = e_audit.get("orphan_events_removed", 0)
        ooo = e_audit.get("out_of_order_events", 0)
        results.append({
            "Category": "Events",
            "Check": "Duplicate Removal",
            "Status": "PASS" if dup_removed == 0 else "REMEDIATED",
            "Details": f"{dup_removed} duplicates removed",
        })
        results.append({
            "Category": "Events",
            "Check": "Orphan Event Removal",
            "Status": "PASS" if orphans == 0 else "REMEDIATED",
            "Details": f"{orphans} orphan events removed",
        })
        results.append({
            "Category": "Events",
            "Check": "Chronological Integrity",
            "Status": "PASS" if ooo == 0 else "WARNING",
            "Details": f"{ooo} out-of-order events detected",
        })
        results.append({
            "Category": "Events",
            "Check": "Final Record Count",
            "Status": "INFO",
            "Details": f"{e_audit.get('final_clean_events', 'N/A')} events retained",
        })

    # --- Statistical validation checks (if nested) ---
    dq_report = dq_audit.get("data_quality_report", {})
    if isinstance(dq_report, dict):
        checks_table = dq_report.get("checks_table", pd.DataFrame())
        if checks_table is not None and not checks_table.empty:
            for _, crow in checks_table.iterrows():
                results.append({
                    "Category": "Statistical Validation",
                    "Check": crow.get("Check", "Unknown"),
                    "Status": crow.get("Status", "N/A"),
                    "Details": crow.get("Details", ""),
                })

    if not results:
        return pd.DataFrame(columns=["Category", "Check", "Status", "Details"])

    return pd.DataFrame(results)


def export_report_csv(
    report_df: pd.DataFrame, filename: str, reports_dir: str = ""
) -> str:
    """
    Saves a report DataFrame to CSV in the specified reports directory.

    Parameters
    ----------
    report_df : pd.DataFrame
        The report DataFrame to export.
    filename : str
        Name for the CSV file (e.g. 'cleaning_report.csv').
    reports_dir : str, optional
        Target directory path. Defaults to BASE_DIR / 'reports'.

    Returns
    -------
    str
        Absolute filepath of the saved CSV file.
    """
    if not reports_dir:
        out_dir = REPORTS_DIR
    else:
        out_dir = Path(reports_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure the filename ends with .csv
    if not filename.lower().endswith(".csv"):
        filename = filename + ".csv"

    filepath = out_dir / filename

    if report_df is not None and not report_df.empty:
        report_df.to_csv(filepath, index=False, encoding="utf-8-sig")
    else:
        # Write an empty CSV with a header noting no data
        pd.DataFrame({"Info": ["No data available for this report"]}).to_csv(
            filepath, index=False, encoding="utf-8-sig"
        )

    return str(filepath.resolve())


def export_all_reports(
    all_reports: Dict[str, pd.DataFrame], reports_dir: str = ""
) -> List[str]:
    """
    Exports all report DataFrames as individual CSV files.

    Parameters
    ----------
    all_reports : Dict[str, pd.DataFrame]
        Mapping of report_name -> DataFrame. The report_name is used as
        the CSV filename (e.g. {'cleaning_report': df} → cleaning_report.csv).
    reports_dir : str, optional
        Target directory path. Defaults to BASE_DIR / 'reports'.

    Returns
    -------
    List[str]
        List of absolute filepaths for all saved CSV files.
    """
    if not all_reports:
        return []

    saved_paths: List[str] = []
    for report_name, report_df in all_reports.items():
        filepath = export_report_csv(report_df, report_name, reports_dir)
        saved_paths.append(filepath)

    return saved_paths
