"""
Unit Tests for Module 4 (Step 2): Event Distribution Analysis
Team 4: Smart Exam Monitoring & Integrity Analytics Platform
Tests event frequencies, category normalization, severity breakdowns,
and hourly pattern extractions.
"""

import sys
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure src is importable
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.event_analysis import (
    analyze_event_distributions,
    analyze_event_severities,
    analyze_hourly_patterns
)


class TestEventAnalysis(unittest.TestCase):

    def setUp(self):
        """Prepare sample events data."""
        self.events = pd.DataFrame({
            "event_id": [f"E{i:03d}" for i in range(1, 11)],
            "session_id": ["S001"] * 5 + ["S002"] * 5,
            "candidate_id": ["C001"] * 5 + ["C002"] * 5,
            "event_type": [
                "normal_activity", "tab_switch", "focus_lost",
                "face_not_visible", "reconnect", "normal_activity",
                "tab_switch", "tab_switch", "focus_lost", "normal_activity"
            ],
            "event_time": [
                "2026-10-20 09:10:00", "2026-10-20 09:20:00", "2026-10-20 09:30:00",
                "2026-10-20 09:40:00", "2026-10-20 09:50:00", "2026-10-20 14:15:00",
                "2026-10-20 14:25:00", "2026-10-20 14:35:00", "2026-10-20 14:45:00",
                "2026-10-20 15:00:00"
            ],
            "severity": [0, 1, 1, 2, 1, 0, 1, 1, 1, 0]
        })

    def test_analyze_event_distributions_counts(self):
        """Test event type counts and target frequencies."""
        res = analyze_event_distributions(self.events)
        self.assertEqual(res["total_events"], 10)

        counts = res["specific_counts"]
        self.assertEqual(counts["normal_activity"], 3)
        self.assertEqual(counts["tab_switch"], 3)
        self.assertEqual(counts["focus_lost"], 2)
        self.assertEqual(counts["face_absent"], 1)
        self.assertEqual(counts["reconnect"], 1)

    def test_analyze_event_distributions_empty(self):
        """Test graceful handling of empty events DataFrame."""
        res = analyze_event_distributions(pd.DataFrame())
        self.assertEqual(res["total_events"], 0)
        self.assertTrue(res["frequency_table"].empty)

    def test_analyze_event_severities(self):
        """Test severity tier mapping (0: Normal, 1: Warning, 2: Critical)."""
        sev_df = analyze_event_severities(self.events)
        self.assertFalse(sev_df.empty)
        self.assertEqual(len(sev_df), 3)  # Severities 0, 1, 2
        # Verify percentages sum to approximately 100%
        self.assertAlmostEqual(sev_df["percentage"].sum(), 100.0, places=1)

    def test_analyze_hourly_patterns(self):
        """Test extraction of hourly occurrence distribution."""
        hourly = analyze_hourly_patterns(self.events)
        self.assertFalse(hourly.empty)
        # We have events in hour 9 and hour 14, 15
        hours_present = set(hourly["hour"].values)
        self.assertIn(9, hours_present)
        self.assertIn(14, hours_present)
        self.assertIn(15, hours_present)


if __name__ == "__main__":
    unittest.main()

