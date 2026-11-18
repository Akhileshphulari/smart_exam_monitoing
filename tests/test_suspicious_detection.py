"""
Unit Tests for Module 4 (Step 4): Suspicious Event Detection
Team 4: Smart Exam Monitoring & Integrity Analytics Platform
Tests pattern detection, thresholding, analytical risk indicator formulation,
and ethical human review boundaries.
"""

import sys
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure src is importable
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.suspicious_detection import (
    detect_candidate_patterns,
    detect_suspicious_patterns
)


class TestSuspiciousDetection(unittest.TestCase):

    def setUp(self):
        """Prepare sample events and sessions."""
        self.sessions = pd.DataFrame({
            "session_id": ["S001", "S002", "S003"],
            "candidate_id": ["C001", "C002", "C003"]
        })

        # S001 has 4 tab switches (>= 3) and 2 face absences (>= 2) -> High Priority
        # S002 has 2 focus losses (>= 2) -> Moderate Monitoring
        # S003 has normal activity only -> Low / Standard
        self.events = pd.DataFrame({
            "event_id": [f"E{i:03d}" for i in range(1, 10)],
            "session_id": [
                "S001", "S001", "S001", "S001", "S001", "S001",
                "S002", "S002",
                "S003"
            ],
            "candidate_id": [
                "C001", "C001", "C001", "C001", "C001", "C001",
                "C002", "C002",
                "C003"
            ],
            "event_type": [
                "tab_switch", "tab_switch", "tab_switch", "tab_switch",
                "face_not_visible", "face_absent",
                "focus_lost", "focus_lost",
                "normal_activity"
            ],
            "severity": [1, 1, 1, 1, 2, 2, 1, 1, 0]
        })

    def test_detect_candidate_patterns(self):
        """Test candidate-level behavioral aggregations."""
        cand_df = detect_candidate_patterns(self.events)
        self.assertEqual(len(cand_df), 3)

        c001 = cand_df[cand_df["candidate_id"] == "C001"].iloc[0]
        self.assertEqual(c001["tab_switches"], 4)
        self.assertEqual(c001["face_absences"], 2)
        self.assertEqual(c001["suspicious_total"], 6)

    def test_detect_suspicious_patterns_priorities(self):
        """Test session pattern flagging and analytical alert priorities."""
        patterns = detect_suspicious_patterns(
            self.events, self.sessions,
            tab_switch_threshold=3,
            face_absence_threshold=2,
            focus_loss_threshold=2
        )

        s001 = patterns[patterns["session_id"] == "S001"].iloc[0]
        self.assertTrue(s001["repeated_tab_switches"])
        self.assertTrue(s001["repeated_face_absences"])
        self.assertEqual(s001["analytical_alert_priority"], "High Priority Review")

        s002 = patterns[patterns["session_id"] == "S002"].iloc[0]
        self.assertTrue(s002["repeated_focus_losses"])
        self.assertEqual(s002["analytical_alert_priority"], "Moderate Monitoring")

        s003 = patterns[patterns["session_id"] == "S003"].iloc[0]
        self.assertEqual(s003["analytical_alert_priority"], "Low / Standard")

    def test_ethical_safeguard_no_cheating_accusations(self):
        """Ensure no output column labels candidate as cheater."""
        patterns = detect_suspicious_patterns(self.events, self.sessions)
        for col in patterns.columns:
            self.assertNotIn("cheat", col.lower(), "Column names must avoid accusatory terminology")

        for val in patterns["analytical_alert_priority"].astype(str):
            self.assertNotIn("cheat", val.lower(), "Alert priorities must remain purely analytical")


if __name__ == "__main__":
    unittest.main()

