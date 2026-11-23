"""
Unit Tests for Data Integration Pipeline Module
Team 4: Smart Exam Monitoring & Integrity Analytics Platform

Tests automated ingestion, schema harmonization, event category normalization,
referential integrity verification, and fallback handling.
"""

import sys
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure root directory is importable
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.data_integration import (
    DataIntegrationPipeline,
    get_integrated_datasets,
    EVENT_TAXONOMY_MAP,
    SEVERITY_TIER_MAP
)
from src.session_analysis import compute_session_metrics
from src.suspicious_detection import detect_suspicious_patterns
from src.risk_analysis import compute_risk_scores
from src.outlier_detection import build_outlier_feature_matrix


class TestDataIntegration(unittest.TestCase):

    def setUp(self):
        self.pipeline = DataIntegrationPipeline()
        self.external_dir = BASE_DIR / "data" / "external"
        self.fallback_dir = BASE_DIR / "data" / "fallback"

    def test_external_dataset_ingestion(self):
        """Test ingestion of external Exam-monitoring-main datasets."""
        if not self.external_dir.exists():
            self.skipTest("External datasets not found.")

        datasets, audit = self.pipeline.load_from_directory(self.external_dir)
        self.assertTrue(audit["is_external"], "External dataset must be recognized as external")
        self.assertIn("Exam-monitoring-main", audit["source_provenance"])
        
        # Check required tables
        self.assertIn("candidates", datasets)
        self.assertIn("sessions", datasets)
        self.assertIn("events", datasets)
        self.assertIn("scores", datasets)
        self.assertIn("risk", datasets)
        self.assertIn("ai_reports", datasets)

        # Check row counts
        counts = audit["record_counts"]
        self.assertEqual(counts["candidates"], 100)
        self.assertEqual(counts["sessions"], 150)
        self.assertEqual(counts["events"], 3747)

    def test_fallback_dataset_ingestion(self):
        """Test fallback demo dataset loading."""
        if not self.fallback_dir.exists():
            self.skipTest("Fallback directory not found.")

        datasets, audit = self.pipeline.load_from_directory(self.fallback_dir)
        self.assertFalse(audit["is_external"], "Fallback data must be recognized as baseline demo data")
        self.assertIn("Team 4 Fallback", audit["source_provenance"])
        self.assertGreater(len(datasets["sessions"]), 0)

    def test_duration_harmonization(self):
        """Test duration_mins -> duration_seconds and duration_minutes conversion."""
        raw_sessions = pd.DataFrame({
            "session_id": ["S101", "S102"],
            "candidate_id": ["C101", "C102"],
            "start_time": ["2026-03-19 10:00:00", "2026-03-19 11:00:00"],
            "duration_mins": [90, 120]
        })
        norm, _ = self.pipeline.validate_and_normalize({"sessions": raw_sessions})
        sess = norm["sessions"]

        self.assertIn("duration_seconds", sess.columns)
        self.assertIn("duration_minutes", sess.columns)
        self.assertEqual(sess.loc[0, "duration_seconds"], 5400.0)
        self.assertEqual(sess.loc[0, "duration_minutes"], 90.0)
        self.assertEqual(sess.loc[1, "duration_seconds"], 7200.0)
        self.assertEqual(sess.loc[1, "duration_minutes"], 120.0)
        
        # Check calculated end_time
        self.assertFalse(sess["end_time"].isna().any())
        self.assertEqual(str(sess.loc[0, "end_time"]), "2026-03-19 11:30:00")

    def test_event_taxonomy_and_severity_mapping(self):
        """Test normalization of external event types and severity assignment."""
        raw_events = pd.DataFrame({
            "event_id": ["E1", "E2", "E3", "E4", "E5", "E6"],
            "session_id": ["S1", "S1", "S1", "S1", "S1", "S1"],
            "event_type": ["normal", "looking_away", "left_seat", "phone_detected", "talking", "multiple_persons"],
            "timestamp": ["2026-03-19 10:05:00"] * 6
        })
        norm, _ = self.pipeline.validate_and_normalize({"events": raw_events})
        ev = norm["events"]

        self.assertIn("event_time", ev.columns)
        types = ev["event_type"].tolist()
        self.assertEqual(types, [
            "normal_activity", "focus_lost", "face_absent", 
            "unauthorized_device", "audio_detected", "multiple_faces"
        ])

        severities = ev["severity"].tolist()
        self.assertEqual(severities, [0, 1, 2, 2, 1, 2])

    def test_risk_level_normalization(self):
        """Test normalization of risk levels (Moderate -> Medium)."""
        raw_risk = pd.DataFrame({
            "risk_id": [1, 2, 3],
            "session_id": ["S1", "S2", "S3"],
            "risk_level": ["Moderate", "Low", "High"]
        })
        norm, _ = self.pipeline.validate_and_normalize({"risk": raw_risk})
        risk = norm["risk"]

        self.assertEqual(risk.loc[0, "risk_level"], "Medium")
        self.assertEqual(risk.loc[1, "risk_level"], "Low")
        self.assertEqual(risk.loc[2, "risk_level"], "High")

    def test_orphan_detection(self):
        """Test that orphan sessions and events are tracked in the audit."""
        raw_cand = pd.DataFrame({"candidate_id": ["C1"]})
        raw_sess = pd.DataFrame({
            "session_id": ["S1", "S2"],
            "candidate_id": ["C1", "C999"],  # C999 is orphan
            "start_time": ["2026-03-19 10:00:00", "2026-03-19 11:00:00"],
            "duration_mins": [60, 60]
        })
        raw_ev = pd.DataFrame({
            "event_id": ["E1", "E2"],
            "session_id": ["S1", "S999"],  # S999 is orphan
            "event_type": ["normal", "tab_switch"],
            "timestamp": ["2026-03-19 10:10:00", "2026-03-19 10:20:00"]
        })

        norm, audit = self.pipeline.validate_and_normalize({
            "candidates": raw_cand,
            "sessions": raw_sess,
            "events": raw_ev
        })

        orphans = audit["orphan_audit"]
        self.assertEqual(orphans["orphan_sessions"], 1)
        self.assertEqual(orphans["orphan_events"], 1)

    def test_downstream_analytical_compatibility(self):
        """Test that downstream analytics execute smoothly with integrated external data."""
        datasets, _ = self.pipeline.load_from_directory(self.external_dir)
        cand = datasets["candidates"]
        sess = datasets["sessions"]
        ev = datasets["events"]
        scores = datasets["scores"]

        # 1. Session metrics
        sess_metrics = compute_session_metrics(sess, ev, scores)
        self.assertFalse(sess_metrics.empty)
        self.assertIn("total_events", sess_metrics.columns)
        self.assertIn("duration_minutes", sess_metrics.columns)

        # 2. Suspicious detection
        suspicious = detect_suspicious_patterns(ev, sess)
        self.assertFalse(suspicious.empty)
        self.assertIn("analytical_alert_priority", suspicious.columns)

        # 3. Risk scoring
        risk_df = compute_risk_scores(sess, ev, scores)
        self.assertFalse(risk_df.empty)
        self.assertIn("weighted_risk_score", risk_df.columns)

        # 4. Outlier detection
        outlier_matrix = build_outlier_feature_matrix(sess, ev, scores)
        self.assertFalse(outlier_matrix.empty)
        self.assertIn("suspicious_activity_count", outlier_matrix.columns)


if __name__ == "__main__":
    unittest.main()
