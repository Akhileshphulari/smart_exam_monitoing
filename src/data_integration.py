"""
Data Integration Pipeline Module
Team 4: Smart Exam Monitoring & Integrity Analytics Platform

Connects external exam-monitoring CSV datasets (from Exam-monitoring-main)
with the Team 4 Data Science analytical pipeline.

Key Capabilities:
1. Automated schema detection, validation, and type harmonization.
2. Robust event category taxonomy mapping and severity assignment.
3. Duration harmonization (duration_mins <-> duration_seconds).
4. Referential integrity checking between candidates, sessions, events, scores, and risk.
5. Ingestion from configurable directories or live multi-file CSV uploads.
6. Safe fallback to demo data if external files are missing or incomplete.
7. Comprehensive Data Quality & Provenance Reporting.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
import io
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_EXTERNAL_DIR = BASE_DIR / "data" / "external"
DEFAULT_DATA_DIR = BASE_DIR / "data"
DEFAULT_FALLBACK_DIR = BASE_DIR / "data" / "fallback"

# Standard file resolution aliases
FILE_ALIASES = {
    "candidates": ["candidates_large.csv", "candidates.csv"],
    "sessions": ["sessions_large.csv", "sessions.csv"],
    "events": ["events_large.csv", "events.csv"],
    "scores": ["scores_large.csv", "scores.csv"],
    "risk": ["risk_data_large.csv", "risk_data.csv", "risk.csv"],
    "ai_reports": ["ai_reports_large.csv", "ai_reports.csv", "ai_report_metadata.csv"],
    "face_ground_truth": ["face_monitoring_ground_truth.csv"]
}

# Event taxonomy mapping from external generator to unified Team 4 taxonomy
EVENT_TAXONOMY_MAP = {
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

# Severity tier: 0 = Normal, 1 = Warning, 2 = Critical
SEVERITY_TIER_MAP = {
    "normal_activity": 0,
    "reconnection": 0,
    "focus_lost": 1,
    "tab_switch": 1,
    "audio_detected": 1,
    "face_absent": 2,
    "multiple_faces": 2,
    "unauthorized_device": 2
}

# Risk level normalization
RISK_LEVEL_MAP = {
    "moderate": "Medium",
    "medium": "Medium",
    "low": "Low",
    "high": "High",
    "critical": "High"
}


class DataIntegrationPipeline:
    """
    Ingests, validates, normalizes, and harmonizes datasets from external or fallback sources.
    """

    def __init__(self, source_dir: Optional[Union[str, Path]] = None):
        self.source_dir = Path(source_dir) if source_dir else self._resolve_primary_dir()
        self.last_ingested_at: Optional[datetime] = None
        self.source_provenance: str = "Uninitialized"
        self.is_external: bool = False
        self.raw_datasets: Dict[str, pd.DataFrame] = {}
        self.normalized_datasets: Dict[str, pd.DataFrame] = {}
        self.audit_report: Dict[str, Any] = {}

    def _resolve_primary_dir(self) -> Path:
        """Finds the best available dataset directory in order of priority."""
        # 1. Check data/external
        if DEFAULT_EXTERNAL_DIR.exists() and any(DEFAULT_EXTERNAL_DIR.glob("*.csv")):
            return DEFAULT_EXTERNAL_DIR
        # 2. Check data/
        if DEFAULT_DATA_DIR.exists() and any(DEFAULT_DATA_DIR.glob("*.csv")):
            return DEFAULT_DATA_DIR
        # 3. Fallback
        return DEFAULT_FALLBACK_DIR

    def detect_provenance(self, datasets: Dict[str, pd.DataFrame]) -> Tuple[str, bool]:
        """
        Determines whether loaded data originates from Exam-monitoring-main or fallback demo data.
        """
        sess = datasets.get("sessions", pd.DataFrame())
        cand = datasets.get("candidates", pd.DataFrame())
        ev = datasets.get("events", pd.DataFrame())

        is_external = False
        if not sess.empty and "duration_mins" in sess.columns:
            is_external = True
        elif not cand.empty and "department" in cand.columns:
            is_external = True
        elif not ev.empty and "timestamp" in ev.columns and "event_time" not in ev.columns:
            is_external = True

        if is_external:
            provenance = "Exam-monitoring-main (External Production Telemetry)"
        else:
            provenance = "Team 4 Fallback (Synthetic Baseline)"

        return provenance, is_external

    def locate_and_read_files(self, directory: Path) -> Dict[str, pd.DataFrame]:
        """
        Reads CSV files from a directory using standard file aliases.
        """
        datasets = {}
        for key, aliases in FILE_ALIASES.items():
            loaded = False
            for alias in aliases:
                fpath = directory / alias
                if fpath.exists():
                    try:
                        df = pd.read_csv(fpath)
                        datasets[key] = df
                        loaded = True
                        break
                    except Exception as err:
                        print(f"Warning: Failed reading {fpath}: {err}")
            
            # If face_ground_truth is missing from directory, fallback to default ground truth
            if not loaded and key == "face_ground_truth":
                gt_fallback = DEFAULT_FALLBACK_DIR / "face_monitoring_ground_truth.csv"
                if gt_fallback.exists():
                    try:
                        datasets[key] = pd.read_csv(gt_fallback)
                        loaded = True
                    except Exception:
                        pass
                if not loaded:
                    gt_orig = DEFAULT_DATA_DIR / "face_monitoring_ground_truth.csv"
                    if gt_orig.exists():
                        try:
                            datasets[key] = pd.read_csv(gt_orig)
                            loaded = True
                        except Exception:
                            pass

            if not loaded:
                datasets[key] = pd.DataFrame()

        return datasets

    def validate_and_normalize(self, raw_data: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
        """
        Normalizes schemas, aligns columns, handles missing/invalid values,
        and conducts relational integrity validation.
        """
        norm = {}
        issues = []
        orphan_audit = {}

        # -------------------------------------------------------------
        # 1. Candidates Normalization
        # -------------------------------------------------------------
        cand = raw_data.get("candidates", pd.DataFrame()).copy()
        cand_initial = len(cand)
        if not cand.empty:
            cand["candidate_id"] = cand["candidate_id"].astype(str).str.strip()
            # Map department -> candidate_group if not present
            if "candidate_group" not in cand.columns:
                if "department" in cand.columns:
                    cand["candidate_group"] = cand["department"].fillna("Unassigned").astype(str).str.strip()
                else:
                    cand["candidate_group"] = "Unassigned"
            else:
                cand["candidate_group"] = cand["candidate_group"].fillna("Unassigned").astype(str).str.strip()

            # Clean name & department if present
            if "name" in cand.columns:
                cand["name"] = cand["name"].astype(str).str.strip()
            if "department" in cand.columns:
                cand["department"] = cand["department"].astype(str).str.strip()
            if "roll_number" in cand.columns:
                cand["roll_number"] = cand["roll_number"].astype(str).str.strip()

            cand = cand.drop_duplicates(subset=["candidate_id"])
        cand_final = len(cand)
        cand_duplicates = cand_initial - cand_final
        norm["candidates"] = cand

        # -------------------------------------------------------------
        # 2. Sessions Normalization
        # -------------------------------------------------------------
        sess = raw_data.get("sessions", pd.DataFrame()).copy()
        sess_initial = len(sess)
        if not sess.empty:
            sess["session_id"] = sess["session_id"].astype(str).str.strip()
            if "candidate_id" in sess.columns:
                sess["candidate_id"] = sess["candidate_id"].astype(str).str.strip()

            # Parse start_time
            sess["start_time"] = pd.to_datetime(sess["start_time"], errors="coerce")

            # Duration handling (duration_mins vs duration_seconds)
            if "duration_mins" in sess.columns and "duration_seconds" not in sess.columns:
                sess["duration_mins"] = pd.to_numeric(sess["duration_mins"], errors="coerce")
                sess["duration_seconds"] = (sess["duration_mins"] * 60.0).round(2)
                sess["duration_minutes"] = sess["duration_mins"].round(2)
            elif "duration_seconds" in sess.columns:
                sess["duration_seconds"] = pd.to_numeric(sess["duration_seconds"], errors="coerce")
                sess["duration_minutes"] = (sess["duration_seconds"] / 60.0).round(2)
                if "duration_mins" not in sess.columns:
                    sess["duration_mins"] = sess["duration_minutes"]
            else:
                sess["duration_seconds"] = 3600.0
                sess["duration_minutes"] = 60.0
                sess["duration_mins"] = 60.0

            # Calculate end_time if missing or null
            if "end_time" in sess.columns:
                sess["end_time"] = pd.to_datetime(sess["end_time"], errors="coerce")
            else:
                sess["end_time"] = pd.NaT

            missing_end = sess["end_time"].isna() & sess["start_time"].notna()
            if missing_end.any():
                sess.loc[missing_end, "end_time"] = (
                    sess.loc[missing_end, "start_time"] + 
                    pd.to_timedelta(sess.loc[missing_end, "duration_minutes"].fillna(60), unit="m")
                )

            # Repair negative/invalid durations
            calc_dur = (sess["end_time"] - sess["start_time"]).dt.total_seconds()
            invalid_dur = (sess["duration_seconds"].isna()) | (sess["duration_seconds"] <= 0)
            repairable = invalid_dur & (calc_dur > 0)
            sess.loc[repairable, "duration_seconds"] = calc_dur[repairable]
            sess["duration_minutes"] = (sess["duration_seconds"] / 60.0).round(2)
            sess["duration_mins"] = sess["duration_minutes"]

            # Status normalization
            if "status" in sess.columns:
                sess["status"] = sess["status"].fillna("completed").astype(str).str.lower().str.strip()
            else:
                sess["status"] = "completed"

            sess = sess.drop_duplicates(subset=["session_id"])

        sess_final = len(sess)
        sess_duplicates = sess_initial - sess_final
        norm["sessions"] = sess

        # -------------------------------------------------------------
        # 3. Events Normalization
        # -------------------------------------------------------------
        ev = raw_data.get("events", pd.DataFrame()).copy()
        ev_initial = len(ev)
        out_of_order_count = 0
        if not ev.empty:
            ev["event_id"] = ev["event_id"].astype(str).str.strip()
            ev["session_id"] = ev["session_id"].astype(str).str.strip()
            if "candidate_id" in ev.columns:
                ev["candidate_id"] = ev["candidate_id"].astype(str).str.strip()

            # Align timestamp / event_time
            if "timestamp" in ev.columns and "event_time" not in ev.columns:
                ev["event_time"] = pd.to_datetime(ev["timestamp"], errors="coerce")
            elif "event_time" in ev.columns:
                ev["event_time"] = pd.to_datetime(ev["event_time"], errors="coerce")
            else:
                ev["event_time"] = pd.NaT
            ev["timestamp"] = ev["event_time"]

            # Event taxonomy mapping
            if "event_type" in ev.columns:
                ev["event_type_raw"] = ev["event_type"].astype(str)
                cleaned_type = ev["event_type"].astype(str).str.lower().str.strip()
                ev["event_type"] = cleaned_type.map(EVENT_TAXONOMY_MAP).fillna(cleaned_type)
                ev["event_category_normalized"] = ev["event_type"]

            # Severity tier mapping
            if "severity" not in ev.columns or ev["severity"].isna().any():
                ev["severity"] = ev["event_type"].map(SEVERITY_TIER_MAP).fillna(1).astype(int)
            else:
                ev["severity"] = pd.to_numeric(ev["severity"], errors="coerce").fillna(1).astype(int)

            # Metadata fields
            if "source" not in ev.columns:
                ev["source"] = "Exam-monitoring-main"
            if "confidence" in ev.columns:
                ev["confidence"] = pd.to_numeric(ev["confidence"], errors="coerce").fillna(1.0)
            if "duration_seconds" in ev.columns:
                ev["duration_seconds"] = pd.to_numeric(ev["duration_seconds"], errors="coerce").fillna(0.0)

            # Deduplicate by event_id
            ev = ev.drop_duplicates(subset=["event_id"])

            # Check chronology vs session window
            if not sess.empty and "start_time" in sess.columns and "end_time" in sess.columns:
                s_bounds = sess[["session_id", "start_time", "end_time"]].dropna(subset=["session_id"])
                ev_merged = ev.merge(s_bounds, on="session_id", how="left")
                ev["out_of_order"] = False
                if "event_time" in ev_merged.columns:
                    before_start = ev_merged["start_time"].notna() & (ev_merged["event_time"] < ev_merged["start_time"])
                    after_end = ev_merged["end_time"].notna() & (ev_merged["event_time"] > ev_merged["end_time"])
                    ev["out_of_order"] = (before_start | after_end).fillna(False)
                    out_of_order_count = int(ev["out_of_order"].sum())

        ev_final = len(ev)
        ev_duplicates = ev_initial - ev_final
        norm["events"] = ev

        # -------------------------------------------------------------
        # 4. Scores Normalization
        # -------------------------------------------------------------
        scores = raw_data.get("scores", pd.DataFrame()).copy()
        if not scores.empty:
            scores["session_id"] = scores["session_id"].astype(str).str.strip()
            if "candidate_id" in scores.columns:
                scores["candidate_id"] = scores["candidate_id"].astype(str).str.strip()
            if "score_id" in scores.columns:
                scores["score_id"] = scores["score_id"].astype(str).str.strip()

            for sc_col in ["integrity_score", "behavior_score", "attention_score"]:
                if sc_col in scores.columns:
                    scores[sc_col] = pd.to_numeric(scores[sc_col], errors="coerce")
            scores = scores.drop_duplicates(subset=["session_id"])
        norm["scores"] = scores

        # -------------------------------------------------------------
        # 5. Risk Normalization
        # -------------------------------------------------------------
        risk = raw_data.get("risk", pd.DataFrame()).copy()
        if not risk.empty:
            risk["session_id"] = risk["session_id"].astype(str).str.strip()
            if "candidate_id" in risk.columns:
                risk["candidate_id"] = risk["candidate_id"].astype(str).str.strip()
            if "risk_id" in risk.columns:
                risk["risk_id"] = risk["risk_id"].astype(str).str.strip()

            if "risk_level" in risk.columns:
                raw_risk = risk["risk_level"].astype(str).str.lower().str.strip()
                risk["risk_level"] = raw_risk.map(RISK_LEVEL_MAP).fillna("Medium")
            else:
                risk["risk_level"] = "Medium"

            if "integrity_score" in risk.columns:
                risk["integrity_score"] = pd.to_numeric(risk["integrity_score"], errors="coerce")

            risk = risk.drop_duplicates(subset=["session_id"])
        norm["risk"] = risk

        # -------------------------------------------------------------
        # 6. AI Reports Normalization
        # -------------------------------------------------------------
        ai_rep = raw_data.get("ai_reports", pd.DataFrame()).copy()
        if not ai_rep.empty:
            ai_rep["session_id"] = ai_rep["session_id"].astype(str).str.strip()
            if "candidate_id" in ai_rep.columns:
                ai_rep["candidate_id"] = ai_rep["candidate_id"].astype(str).str.strip()
            if "report_id" in ai_rep.columns:
                ai_rep["report_id"] = ai_rep["report_id"].astype(str).str.strip()

            if "report_status" not in ai_rep.columns:
                ai_rep["report_status"] = "generated"
            if "notes" not in ai_rep.columns:
                ai_rep["notes"] = ai_rep.get("summary", "Automated Proctor Report")
            if "model_version" not in ai_rep.columns:
                ai_rep["model_version"] = "v1.0-standard"

            ai_rep = ai_rep.drop_duplicates(subset=["session_id"])
        norm["ai_reports"] = ai_rep
        norm["ai_report_metadata"] = ai_rep  # alias for backward compatibility

        # -------------------------------------------------------------
        # 7. Face Ground Truth Benchmark
        # -------------------------------------------------------------
        gt = raw_data.get("face_ground_truth", pd.DataFrame()).copy()
        if not gt.empty:
            if "session_id" in gt.columns:
                gt["session_id"] = gt["session_id"].astype(str).str.strip()
            if "candidate_id" in gt.columns:
                gt["candidate_id"] = gt["candidate_id"].astype(str).str.strip()
        norm["face_ground_truth"] = gt

        # -------------------------------------------------------------
        # 8. Referential Integrity & Orphan Tracking
        # -------------------------------------------------------------
        cand_ids = set(cand["candidate_id"].dropna().unique()) if not cand.empty else set()
        sess_ids = set(sess["session_id"].dropna().unique()) if not sess.empty else set()

        orphan_sessions_cnt = 0
        orphan_events_cnt = 0
        orphan_scores_cnt = 0
        orphan_risks_cnt = 0

        if not sess.empty and cand_ids:
            orphan_sessions_cnt = int((~sess["candidate_id"].isin(cand_ids)).sum())
            if orphan_sessions_cnt > 0:
                issues.append(f"{orphan_sessions_cnt} session(s) reference non-existent candidate IDs.")

        if not ev.empty and sess_ids:
            orphan_events_cnt = int((~ev["session_id"].isin(sess_ids)).sum())
            if orphan_events_cnt > 0:
                issues.append(f"{orphan_events_cnt} event(s) reference non-existent session IDs.")

        if not scores.empty and sess_ids:
            orphan_scores_cnt = int((~scores["session_id"].isin(sess_ids)).sum())
            if orphan_scores_cnt > 0:
                issues.append(f"{orphan_scores_cnt} score record(s) reference non-existent session IDs.")

        if not risk.empty and sess_ids:
            orphan_risks_cnt = int((~risk["session_id"].isin(sess_ids)).sum())
            if orphan_risks_cnt > 0:
                issues.append(f"{orphan_risks_cnt} risk record(s) reference non-existent session IDs.")

        orphan_audit = {
            "orphan_sessions": orphan_sessions_cnt,
            "orphan_events": orphan_events_cnt,
            "orphan_scores": orphan_scores_cnt,
            "orphan_risks": orphan_risks_cnt
        }

        # -------------------------------------------------------------
        # 9. Ingestion Audit Summary
        # -------------------------------------------------------------
        provenance, is_external = self.detect_provenance(raw_data)
        self.source_provenance = provenance
        self.is_external = is_external
        self.last_ingested_at = datetime.now()

        audit = {
            "source_provenance": provenance,
            "is_external": is_external,
            "ingested_at": self.last_ingested_at.strftime("%Y-%m-%d %H:%M:%S"),
            "record_counts": {
                "candidates": len(cand),
                "sessions": len(sess),
                "events": len(ev),
                "scores": len(scores),
                "risk": len(risk),
                "ai_reports": len(ai_rep),
                "face_ground_truth": len(gt)
            },
            "duplicates_removed": {
                "candidates": cand_duplicates,
                "sessions": sess_duplicates,
                "events": ev_duplicates
            },
            "orphan_audit": orphan_audit,
            "out_of_order_events": out_of_order_count,
            "validation_issues": issues,
            "event_categories": ev["event_type"].value_counts().to_dict() if not ev.empty else {}
        }

        self.normalized_datasets = norm
        self.audit_report = audit

        return norm, audit

    def load_from_directory(self, directory: Optional[Union[str, Path]] = None) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
        """Loads and normalizes datasets from a local directory path."""
        target_dir = Path(directory) if directory else self.source_dir
        if not target_dir.exists():
            raise FileNotFoundError(f"Target directory does not exist: {target_dir}")

        raw = self.locate_and_read_files(target_dir)
        self.raw_datasets = raw
        return self.validate_and_normalize(raw)

    def load_from_uploaded_files(self, uploaded_dict: Dict[str, Any]) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
        """
        Loads and normalizes CSV files uploaded through Streamlit file_uploader.
        """
        raw = {}
        for key, aliases in FILE_ALIASES.items():
            loaded = False
            for alias in aliases:
                # Check direct alias match or basename
                for up_name, file_obj in uploaded_dict.items():
                    if up_name.lower() == alias.lower():
                        try:
                            df = pd.read_csv(file_obj)
                            raw[key] = df
                            loaded = True
                            break
                        except Exception as e:
                            print(f"Error reading uploaded file {up_name}: {e}")
                if loaded:
                    break

            if not loaded:
                # If optional or missing, use fallback if available
                fallback_path = DEFAULT_FALLBACK_DIR / aliases[0]
                if fallback_path.exists():
                    try:
                        raw[key] = pd.read_csv(fallback_path)
                        loaded = True
                    except Exception:
                        pass
            if not loaded:
                raw[key] = pd.DataFrame()

        self.raw_datasets = raw
        return self.validate_and_normalize(raw)


# Module-level singleton instance for convenience
_PIPELINE = DataIntegrationPipeline()


def get_integrated_datasets(source_dir: Optional[Union[str, Path]] = None, force_reload: bool = False) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """
    Standard entrypoint for retrieving normalized datasets and audit reports.
    Uses cached datasets unless force_reload is True or a new source_dir is specified.
    """
    global _PIPELINE
    if force_reload or source_dir or not _PIPELINE.normalized_datasets:
        if source_dir:
            _PIPELINE = DataIntegrationPipeline(source_dir)
        norm, audit = _PIPELINE.load_from_directory()
        return norm, audit
    return _PIPELINE.normalized_datasets, _PIPELINE.audit_report


def ingest_uploaded_csvs(uploaded_dict: Dict[str, Any]) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """Ingests uploaded files and returns normalized datasets."""
    global _PIPELINE
    return _PIPELINE.load_from_uploaded_files(uploaded_dict)


def get_current_pipeline_status() -> Dict[str, Any]:
    """Returns status dictionary of the current data integration state."""
    global _PIPELINE
    return {
        "source_provenance": _PIPELINE.source_provenance,
        "is_external": _PIPELINE.is_external,
        "last_ingested_at": _PIPELINE.last_ingested_at.strftime("%Y-%m-%d %H:%M:%S") if _PIPELINE.last_ingested_at else "Not Ingested",
        "source_directory": str(_PIPELINE.source_dir),
        "audit": _PIPELINE.audit_report
    }
