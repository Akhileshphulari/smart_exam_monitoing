"""
Unit Tests for Module 4 (Step 1) & General Preprocessing
Team 4: Smart Exam Monitoring & Integrity Analytics Platform
Tests data cleaning, type coercion, timestamp validation,
duplicate removal, referential integrity verification, and data quality scoring.
"""

import sys
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure src is importable
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.preprocessing import (
    clean_candidates,
    clean_sessions,
    clean_events,
    audit_data_quality,
    run_pipeline
)


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        """Prepare sample test datasets."""
        self.raw_candidates = pd.DataFrame({
            "candidate_id": ["C001", "C002", "C001", "C003 "],
            "candidate_group": ["Group A", None, "Group A", "Group B "]
        })

        self.raw_sessions = pd.DataFrame({
            "session_id": ["S001", "S002", "S003"],
            "candidate_id": ["C001", "C002", "C099"],  # C099 is orphan
            "start_time": ["2026-10-20 09:00:00", "2026-10-20 10:00:00", "invalid_time"],
            "end_time": ["2026-10-20 10:00:00", "2026-10-20 11:30:00", "2026-10-20 12:00:00"],
            "duration_seconds": [3600.0, -500.0, 1800.0],
            "status": ["completed", " interrupted ", None]
        })

        self.raw_events = pd.DataFrame({
            "event_id": ["E001", "E002", "E001", "E003", "E004"],
            "session_id": ["S001", "S001", "S001", "S999", "S002"],  # S999 is orphan
            "candidate_id": ["C001", "C001", "C001", "C099", "C002"],
            "event_type": ["tab_switch", "face_not_visible", "tab_switch", "focus_lost", "normal_activity"],
            "event_time": ["2026-10-20 09:15:00", "2026-10-20 08:30:00", "2026-10-20 09:15:00", "2026-10-20 10:00:00", "2026-10-20 10:30:00"],
            "severity": [1, 2, 1, 1, 0]
        })

        self.raw_scores = pd.DataFrame({
            "session_id": ["S001", "S002"],
            "integrity_score": [85.0, 72.0]
        })

        self.raw_risk = pd.DataFrame({
            "session_id": ["S001", "S002"],
            "risk_level": ["Low", "Medium"]
        })

    def test_clean_candidates_deduplication(self):
        """Test candidate deduplication and whitespace trimming."""
        clean_df, audit = clean_candidates(self.raw_candidates)
        self.assertEqual(len(clean_df), 3, "Expected 3 unique candidates")
        self.assertEqual(audit["duplicates_removed"], 1)
        self.assertIn("Unassigned", clean_df["candidate_group"].values)
        self.assertFalse(clean_df["candidate_id"].str.endswith(" ").any())

    def test_clean_sessions_duration_repair(self):
        """Test timestamp parsing and invalid duration repair."""
        clean_df, audit = clean_sessions(self.raw_sessions)
        self.assertEqual(len(clean_df), 3)
        # S002 had negative duration -500, should be recalculated from 10:00 to 11:30 = 5400s
        s002_dur = clean_df.loc[clean_df["session_id"] == "S002", "duration_seconds"].values[0]
        self.assertEqual(s002_dur, 5400.0)
        self.assertGreater(audit["durations_repaired"], 0)

    def test_clean_events_orphans_and_dedup(self):
        """Test event deduplication and orphan removal."""
        clean_sess, _ = clean_sessions(self.raw_sessions)
        clean_df, audit = clean_events(self.raw_events, clean_sess)

        # E001 duplicate should be dropped
        self.assertEqual(audit["duplicates_removed"], 1)
        # S999 is orphan, should be dropped
        self.assertEqual(audit["orphan_events_removed"], 1)
        # Event type normalized
        self.assertIn("face_absent", clean_df["event_category_normalized"].values)

    def test_audit_data_quality_score(self):
        """Test 14-point data quality assessment."""
        clean_c, _ = clean_candidates(self.raw_candidates)
        clean_s, _ = clean_sessions(self.raw_sessions)
        clean_e, _ = clean_events(self.raw_events, clean_s)

        dq = audit_data_quality(clean_c, clean_s, clean_e, self.raw_scores, self.raw_risk)
        self.assertEqual(dq["total_checks"], 14)
        self.assertGreater(dq["quality_score_pct"], 0.0)
        self.assertLessEqual(dq["quality_score_pct"], 100.0)
        self.assertIn("checks_table", dq)

    def test_run_pipeline_end_to_end(self):
        """Test full preprocessing pipeline orchestration."""
        data_dict = {
            "candidates": self.raw_candidates,
            "sessions": self.raw_sessions,
            "events": self.raw_events,
            "scores": self.raw_scores,
            "risk": self.raw_risk
        }
        res = run_pipeline(data_dict)
        self.assertIn("candidates", res)
        self.assertIn("sessions", res)
        self.assertIn("events", res)
        self.assertIn("data_quality_report", res)


if __name__ == "__main__":
    unittest.main()

