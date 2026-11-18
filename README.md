# Smart Exam Monitoring & Integrity Analytics Platform
### Infosys Virtual Internship — Python & Data Science Track
**Team 4 Data Science Module (Sprint 2, Sprint 3 & Sprint 4)**

---

## 📌 1. Project Overview & Scope

The **Smart Exam Monitoring & Integrity Analytics Platform** is a specialized Data Science module built exclusively for **Team 4**. It provides data preprocessing, statistical telemetry analysis, behavioral pattern recognition, face monitoring error evaluation, multi-method outlier detection, and transparent risk scoring.

> [!IMPORTANT]
> **Ethical AI & Human-in-the-Loop Principles:**
> This platform strictly adheres to fair proctoring standards. Suspicious behaviors are presented as **transparent analytical risk indicators requiring human supervisory review**. Candidates are **never** automatically accused or labeled as cheaters.
> 
> **Sprint 3 Face Monitoring Scope:**
> Analyzes existing camera event telemetry and evaluates detection accuracy against labeled ground truth. Does **not** build a live webcam or facial capture harness.

---

## 🏗️ 2. Architecture & Modules

```
                 Examination Telemetry Data (CSVs / Uploads)
                                      │
                                      ▼
   ┌────────────────────────────────────────────────────────────────────────┐
   │ Module 1: Dataset Management (src/data_loader.py)                      │
   │ - Multi-file CSV ingestion, profiling, type classification, upload API │
   └──────────────────────────────────┬─────────────────────────────────────┘
                                      │
                                      ▼
   ┌────────────────────────────────────────────────────────────────────────┐
   │ Module 4 (Step 1): Preprocessing (src/preprocessing.py)                │
   │ - 14-Point Data Quality Audit, deduplication, duration recalculation   │
   └──────────────────────────────────┬─────────────────────────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        ▼                             ▼                             ▼
┌───────────────────────┐ ┌───────────────────────┐ ┌───────────────────────┐
│ Module 2: Sprint 2    │ │ Module 3: Sprint 3    │ │ Module 4: Sprint 4    │
│ Session Analytics     │ │ Face Monitoring       │ │ Suspicious Events     │
│ (session_analysis.py) │ │ (face_analysis.py)    │ │ (suspicious_detect.py)│
│ - Duration series     │ │ - Presence frequency  │ │ - Repeated tabs       │
│ - Candidate summary   │ │ - Absence durations   │ │ - Focus loss periods  │
│ - Missing breakdown   │ │ - Ground-truth eval   │ │ - Multi-vector combos │
│ - KPI aggregations    │ │ - Confusion matrix    │ │ - Outlier detection   │
└───────────┬───────────┘ └───────────┬───────────┘ └───────────┬───────────┘
            │                         │                         │
            └─────────────────────────┼─────────────────────────┘
                                      ▼
   ┌────────────────────────────────────────────────────────────────────────┐
   │ Module 5: Configurable Risk Analytics (src/risk_analysis.py)           │
   │ - Weighted multi-indicator scoring into Low, Medium, High risk tiers   │
   │ - Transparent indicator contribution breakdowns                        │
   └──────────────────────────────────┬─────────────────────────────────────┘
                                      │
        ┌─────────────────────────────┴─────────────────────────────┐
        ▼                                                           ▼
┌──────────────────────────────────────────┐    ┌──────────────────────────────────────────┐
│ Module 6: Interactive Dashboard (app.py) │    │ Module 7: Report Generator               │
│ - Dark Navy Theme (#0a0e27)              │    │ (src/report_generator.py)                │
│ - 11 Navigation Pages / Modules          │    │ - 8 Analytical CSV Reports               │
│ - Interactive Plotly Charts & KPI Cards  │    │ - Batch Disk Export & Direct Download    │
└──────────────────────────────────────────┘    └──────────────────────────────────────────┘
```

---

## 🛠️ 3. Tech Stack

| Technology | Version | Purpose |
| :--- | :--- | :--- |
| **Python** | 3.11+ | Core programming runtime |
| **Pandas** | &ge; 2.0.0 | Data manipulation, aggregation, and profiling |
| **NumPy** | &ge; 1.24.0 | Numerical operations, vector calculations, quantiles |
| **Scikit-Learn** | &ge; 1.3.0 | Evaluation metrics (Accuracy, Precision, Recall, F1) & Confusion Matrix |
| **SciPy** | &ge; 1.10.0 | Chi-Square ($\chi^2$) test, Z-scores, skewness & kurtosis |
| **Streamlit** | &ge; 1.28.0 | Web application dashboard with Dark Navy theme |
| **Plotly** | &ge; 5.15.0 | Interactive charts (Donut, Histogram, Scatter, Heatmap, Lines) |
| **Seaborn & Matplotlib** | &ge; 0.12.0 | Statistical data visualization |

---

## 📂 4. Project Structure

```
Team4_smartexam_monitoring/
│
├── app.py                             # Main Streamlit application with 11 modules
├── requirements.txt                   # Project dependencies
├── README.md                          # Project documentation
│
├── data/                              # Examination datasets
│   ├── candidates.csv                 # Candidate roster and group assignments
│   ├── sessions.csv                   # Session timestamps, durations, and statuses
│   ├── events.csv                     # Telemetry event logs
│   ├── scores.csv                     # Integrity scores per session
│   ├── risk.csv                       # Risk classifications per session
│   ├── ai_report_metadata.csv         # AI proctoring report metadata
│   └── face_monitoring_ground_truth.csv # Labeled face monitoring test data (seed 42)
│
├── src/                               # Core Python business logic & pipelines
│   ├── __init__.py                    # Package initializer
│   ├── data_loader.py                 # Module 1: Dataset management & profiling
│   ├── session_analysis.py            # Module 2: Sprint 2 session analytics
│   ├── face_analysis.py               # Module 3: Sprint 3 face monitoring analytics
│   ├── preprocessing.py               # Module 4 (Step 1): Data cleaning & audit
│   ├── event_analysis.py              # Module 4 (Step 2): Event distribution analysis
│   ├── outlier_detection.py           # Module 4 (Step 3): IQR & Z-score outlier detection
│   ├── suspicious_detection.py        # Module 4 (Step 4): Suspicious pattern detection
│   ├── statistical_validation.py      # Module 4 (Step 5): Statistical validation
│   ├── risk_analysis.py               # Module 5: Configurable risk analytics
│   └── report_generator.py            # Module 7: Analytical report generator
│
├── pages/                             # Streamlit multi-page interface
│   ├── 1_Overview.py                  # Module 6: Executive overview
│   ├── 2_Dataset_Explorer.py          # Module 6: Dataset explorer & CSV upload
│   ├── 3_Session_Analytics.py         # Module 6: Session metrics & duration stats
│   ├── 4_Face_Analytics.py            # Module 6: Face detection accuracy & lighting comparison
│   ├── 5_Event_Analysis.py            # Module 6: Event frequencies & severities
│   ├── 6_Suspicious_Detection.py      # Module 6: Suspicious patterns & explanations
│   ├── 7_Data_Quality.py              # Module 6: 14-point audit & statistical validation
│   └── 8_Reports.py                   # Module 6: View and export reports
│
├── reports/                           # Exported CSV analytical reports
│   ├── Data_Cleaning_Report.csv
│   ├── Session_Analytics_Report.csv
│   ├── Face_Detection_Report.csv
│   ├── Event_Frequency_Report.csv
│   ├── Outlier_Detection_Report.csv
│   ├── Suspicious_Event_Report.csv
│   ├── Statistical_Validation_Report.csv
│   └── Data_Quality_Report.csv
│
└── tests/                             # Unit testing suite
    ├── test_preprocessing.py          # Preprocessing & audit tests
    ├── test_event_analysis.py          # Event analysis tests
    └── test_suspicious_detection.py   # Pattern detection & ethical safeguard tests
```

---

## 🔍 5. Module Summary

### Module 1 – Dataset Management
- Ingests all 6 core CSVs + 1 ground-truth dataset.
- Computes dimensional counts, data types, missing value percentages, and duplicate counts.
- Generates column profiles with sample previews.
- Supports user-uploaded CSV validation and live analysis.

### Module 2 – Sprint 2: Session Analytics
- Computes session durations, event counts per session, and missing data breakdown.
- Calculates comprehensive statistical metrics on durations (Mean, Median, Std, Q1, Q3, IQR, Min, Max).
- Aggregates candidate-wise summaries (total exams, events, average score, average duration).

### Module 3 – Sprint 3: Face Monitoring Analytics
- Analyzes existing face verification event logs (no webcam built).
- Computes face-present, face-absent, and detection-unavailable frequencies.
- Evaluates face absence durations (mean, max, per-session totals).
- Evaluates detection across lighting conditions: Normal, Low Light, and Overexposed.
- Calculates evaluation metrics using labeled ground truth (`face_monitoring_ground_truth.csv`):
  - **Accuracy (94.85%)**, **Precision (98.15%)**, **Recall (95.07%)**, **F1-Score (96.58%)**
  - **False Positive Rate (5.71%)**, **False Negative Rate (4.93%)**
  - 2x2 Confusion Matrix (`TP: 212`, `TN: 67`, `FP: 4`, `FN: 11`).
- Documents model limitations (lighting, yaw/pitch occlusion, hardware variation).

### Module 4 – Sprint 4: Suspicious Event Detection
1. **Preprocessing:** Deduplication, timestamp parsing to datetime, category normalization, orphan removal, out-of-order detection, 14-point quality audit.
2. **Event Distribution Analysis:** Frequencies for tab switches, focus loss, face absence, reconnects, normal activity, severity tiers, and hourly distribution.
3. **Outlier Detection:** Tukey's IQR and Z-score methods identifying anomalous event counts and durations with plain-language explanations.
4. **Suspicious Event Detection:** Identifies repeated tab switches ($\ge 3$), prolonged focus loss ($\ge 2$), repeated face absence ($\ge 2$), and multi-vector combinations. Categorizes into High, Moderate, and Low review priorities.
5. **Statistical Validation:** Chi-Square ($\chi^2$) goodness-of-fit test, numerical metric range checks, correlation matrix, and synthetic test evaluation ($N=100$, seed=42).

### Module 5 – Risk Analytics
- Transparent, configurable weighted scoring:
  - Tab Switch Frequency: 25%
  - Focus Loss Frequency: 20%
  - Face Absence Duration: 20%
  - Unusual Event Count: 20%
  - Session Data Quality: 15%
- Normalized to 0–100 scale; categorized into **Low Risk** (0–33), **Medium Risk** (34–66), and **High Risk** (67–100).
- Displays exact indicator contribution percentages per session.

### Module 6 – Interactive Dashboard
- Modern dark navy theme (`#0a0e27`), violet/blue accents, white typography.
- All 11 pages implemented with interactive charts, KPI cards, filters, and tables.

### Module 7 – Report Generation
- Generates and exports 8 CSV reports in `reports/`:
  - Data Cleaning Report
  - Session Analytics Report
  - Face Detection Report
  - Event Frequency Report
  - Outlier Detection Report
  - Suspicious Event Report
  - Statistical Validation Report
  - Data Quality Report

---

## ⚡ 6. Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Unit Tests
```bash
python -m unittest discover tests -v
```

### 3. Launch Dashboard
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 🛡️ 7. Ethical Principles

1. **Analytical Alerts Only:** Risk scores and flags are recommendations for human review, never automated determinations.
2. **No Accusations:** Candidates are never labeled as cheaters.
3. **Objective Audits:** Transparent formulas and reproducible random seeds (seed 42) for all synthetic evaluations.
