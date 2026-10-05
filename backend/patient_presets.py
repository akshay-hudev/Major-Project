"""Pre-configured clinical patient trajectories for simulation and demonstration."""

from __future__ import annotations

from typing import Any, Dict, List
from backend.feature_extractor import FEATURE_NAMES, FEATURE_METADATA


def _generate_visit(base: Dict[str, float], step: int) -> Dict[str, float]:
    """Helper to populate complete 59-feature visit using baseline defaults."""
    v: Dict[str, float] = {}
    for feat in FEATURE_NAMES:
        # Use provided value if present, else fall back to metadata default
        if feat in base:
            v[feat] = base[feat]
        else:
            meta = FEATURE_METADATA.get(feat, {})
            v[feat] = float(meta.get("default", 0.0))

    v["visit_index"] = float(step + 1)
    v["time_delta_days"] = 30.0 + step * 5.0
    return v


# ── Scenario 1: Stable Post-Op Recovery ────────────────────────────────────────

def get_stable_recovery() -> Dict[str, Any]:
    # Normalizing vitals, reducing inflammation, stable biomarkers
    hr = [86.0, 82.0, 80.0, 77.0, 75.0, 72.0]
    sbp = [132.0, 128.0, 126.0, 124.0, 122.0, 120.0]
    dbp = [82.0, 80.0, 78.0, 76.0, 75.0, 74.0]
    mbp = [98.0, 96.0, 94.0, 92.0, 90.0, 89.0]
    resp = [19.0, 18.0, 17.0, 16.0, 16.0, 15.0]
    spo2 = [95.0, 96.0, 96.0, 97.0, 98.0, 98.0]
    trop_t = [0.06, 0.04, 0.03, 0.02, 0.015, 0.012]
    bnp = [220.0, 180.0, 150.0, 120.0, 95.0, 80.0]
    creat = [1.2, 1.1, 1.0, 1.0, 0.95, 0.90]
    bun = [22.0, 20.0, 18.0, 17.0, 16.0, 15.0]
    wbc = [11.8, 10.2, 8.8, 7.9, 7.2, 6.8]
    bicarb = [22.5, 23.0, 24.0, 24.5, 25.0, 25.5]

    visits = []
    for i in range(6):
        data = {
            "heart_rate_mean": hr[i], "sbp_mean": sbp[i], "dbp_mean": dbp[i], "mbp_mean": mbp[i],
            "resp_rate_mean": resp[i], "spo2_mean": spo2[i], "troponin_t_mean": trop_t[i],
            "troponin_t_max": trop_t[i] * 1.2, "bnp_mean": bnp[i], "bnp_max": bnp[i] * 1.15,
            "creatinine_mean": creat[i], "bun_mean": bun[i], "wbc_mean": wbc[i],
            "bicarbonate_mean": bicarb[i], "age": 62.0, "prior_mi": 0.0, "prior_hf": 1.0
        }
        visits.append(_generate_visit(data, i))

    return {
        "id": "DEMO-001",
        "name": "Stable Post-Op Recovery",
        "subtitle": "Resolving inflammatory markers and stabilizing hemodynamic profile",
        "expected_outcome": "Stable / Improving Condition",
        "clinical_narrative": (
            "62-year-old post-revascularization patient showing consistent clinical recovery. "
            "Heart rate and blood pressure have normalized into target ranges, SpO2 has improved to 98%, "
            "and cardiac biomarkers (Troponin T and BNP) show steady downward trends across all six visits."
        ),
        "sample_data": {
            "patient_id": "PT-STABLE-001",
            "patient_name": "Synthetic Case A — Stable Recovery",
            "visits": visits,
            "threshold": 0.335
        }
    }


# ── Scenario 2: Acute Decompensated Heart Failure ───────────────────────────

def get_decompensated_hf() -> Dict[str, Any]:
    # Progressive cardiac strain, volume overload, dropping BP, surging BNP
    hr = [76.0, 82.0, 88.0, 94.0, 102.0, 112.0]
    sbp = [135.0, 128.0, 118.0, 108.0, 98.0, 92.0]
    dbp = [80.0, 76.0, 72.0, 68.0, 62.0, 58.0]
    mbp = [98.0, 93.0, 87.0, 81.0, 74.0, 69.0]
    resp = [17.0, 19.0, 21.0, 23.0, 26.0, 29.0]
    spo2 = [97.0, 96.0, 95.0, 93.0, 91.0, 88.0]
    trop_t = [0.02, 0.03, 0.05, 0.07, 0.10, 0.14]
    bnp = [160.0, 280.0, 440.0, 620.0, 880.0, 1150.0]
    creat = [1.0, 1.2, 1.4, 1.7, 2.1, 2.6]
    bun = [18.0, 24.0, 31.0, 39.0, 48.0, 58.0]
    wbc = [7.5, 8.4, 9.8, 11.2, 13.0, 15.5]
    bicarb = [24.0, 23.0, 21.5, 20.0, 18.5, 17.0]

    visits = []
    for i in range(6):
        data = {
            "heart_rate_mean": hr[i], "sbp_mean": sbp[i], "dbp_mean": dbp[i], "mbp_mean": mbp[i],
            "resp_rate_mean": resp[i], "spo2_mean": spo2[i], "troponin_t_mean": trop_t[i],
            "troponin_t_max": trop_t[i] * 1.25, "bnp_mean": bnp[i], "bnp_max": bnp[i] * 1.2,
            "creatinine_mean": creat[i], "bun_mean": bun[i], "wbc_mean": wbc[i],
            "bicarbonate_mean": bicarb[i], "age": 73.0, "prior_mi": 1.0, "prior_hf": 1.0
        }
        visits.append(_generate_visit(data, i))

    return {
        "id": "DEMO-002",
        "name": "Acute Decompensated Heart Failure (ADHF)",
        "subtitle": "Surging BNP, progressive desaturation, and cardiorenal syndrome",
        "expected_outcome": "Worsening Progression (High Risk)",
        "clinical_narrative": (
            "73-year-old with ischemic cardiomyopathy presenting with progressive hemodynamic "
            "deterioration. Natriuretic peptides (BNP) have surged from 160 to 1150 pg/mL, "
            "accompanied by compensatory resting tachycardia (112 bpm), declining systolic BP (92 mmHg), "
            "and worsening cardiorenal function (creatinine doubled from 1.0 to 2.6 mg/dL)."
        ),
        "sample_data": {
            "patient_id": "PT-DECOMP-002",
            "patient_name": "Synthetic Case B — Decompensated HF",
            "visits": visits,
            "threshold": 0.335
        }
    }


# ── Scenario 3: Post-Myocardial Infarction Deterioration ─────────────────────

def get_post_mi_deterioration() -> Dict[str, Any]:
    # Infarct expansion, rising troponin, ischemic heart strain
    hr = [72.0, 78.0, 84.0, 92.0, 98.0, 104.0]
    sbp = [126.0, 122.0, 116.0, 112.0, 106.0, 101.0]
    dbp = [78.0, 75.0, 72.0, 70.0, 68.0, 65.0]
    mbp = [94.0, 90.0, 86.0, 84.0, 80.0, 77.0]
    resp = [16.0, 17.0, 19.0, 20.0, 22.0, 24.0]
    spo2 = [98.0, 97.0, 96.0, 94.0, 93.0, 92.0]
    trop_t = [0.03, 0.08, 0.18, 0.32, 0.48, 0.65]
    bnp = [110.0, 160.0, 240.0, 350.0, 480.0, 620.0]
    creat = [0.95, 1.05, 1.15, 1.30, 1.45, 1.65]
    bun = [16.0, 18.0, 21.0, 25.0, 29.0, 34.0]
    wbc = [7.2, 8.8, 10.5, 12.4, 14.1, 16.0]
    bicarb = [24.5, 24.0, 23.0, 22.0, 21.0, 20.0]

    visits = []
    for i in range(6):
        data = {
            "heart_rate_mean": hr[i], "sbp_mean": sbp[i], "dbp_mean": dbp[i], "mbp_mean": mbp[i],
            "resp_rate_mean": resp[i], "spo2_mean": spo2[i], "troponin_t_mean": trop_t[i],
            "troponin_t_max": trop_t[i] * 1.3, "bnp_mean": bnp[i], "bnp_max": bnp[i] * 1.25,
            "creatinine_mean": creat[i], "bun_mean": bun[i], "wbc_mean": wbc[i],
            "bicarbonate_mean": bicarb[i], "age": 69.0, "prior_mi": 1.0, "prior_hf": 0.0
        }
        visits.append(_generate_visit(data, i))

    return {
        "id": "DEMO-003",
        "name": "Post-Infarction Deterioration",
        "subtitle": "Accelerating myocardial necrosis markers and secondary inflammatory storm",
        "expected_outcome": "Worsening Progression (High Risk)",
        "clinical_narrative": (
            "69-year-old following anterior myocardial infarction demonstrating progressive "
            "myonecrosis and infarct expansion. Troponin T displays persistent upward velocity "
            "(0.03 -> 0.65 ng/mL) accompanied by systemic leukocytosis (WBC 16.0) and "
            "emerging secondary left ventricular pump dysfunction."
        ),
        "sample_data": {
            "patient_id": "PT-POSTMI-003",
            "patient_name": "Synthetic Case C — Post-MI Deterioration",
            "visits": visits,
            "threshold": 0.335
        }
    }


# ── Scenario 4: Borderline / Fragile Patient ─────────────────────────────────

def get_borderline_fragile() -> Dict[str, Any]:
    # Subtle drifting parameters hovering right around the 33.5% threshold
    hr = [78.0, 80.0, 83.0, 85.0, 88.0, 91.0]
    sbp = [125.0, 122.0, 120.0, 117.0, 114.0, 111.0]
    dbp = [76.0, 75.0, 74.0, 73.0, 71.0, 70.0]
    mbp = [92.0, 90.0, 89.0, 87.0, 85.0, 83.0]
    resp = [17.0, 17.0, 18.0, 19.0, 19.0, 20.0]
    spo2 = [97.0, 96.0, 96.0, 95.0, 95.0, 94.0]
    trop_t = [0.02, 0.025, 0.03, 0.035, 0.04, 0.045]
    bnp = [140.0, 165.0, 195.0, 230.0, 270.0, 310.0]
    creat = [1.1, 1.15, 1.2, 1.25, 1.35, 1.45]
    bun = [19.0, 21.0, 22.0, 24.0, 26.0, 28.0]
    wbc = [7.8, 8.2, 8.6, 9.1, 9.7, 10.4]
    bicarb = [24.0, 23.5, 23.0, 22.5, 22.0, 21.5]

    visits = []
    for i in range(6):
        data = {
            "heart_rate_mean": hr[i], "sbp_mean": sbp[i], "dbp_mean": dbp[i], "mbp_mean": mbp[i],
            "resp_rate_mean": resp[i], "spo2_mean": spo2[i], "troponin_t_mean": trop_t[i],
            "troponin_t_max": trop_t[i] * 1.15, "bnp_mean": bnp[i], "bnp_max": bnp[i] * 1.15,
            "creatinine_mean": creat[i], "bun_mean": bun[i], "wbc_mean": wbc[i],
            "bicarbonate_mean": bicarb[i], "age": 71.0, "prior_mi": 0.0, "prior_hf": 1.0
        }
        visits.append(_generate_visit(data, i))

    return {
        "id": "DEMO-004",
        "name": "Borderline / Fragile Trajectory",
        "subtitle": "Subtle multi-variable drift straddling the clinical decision threshold",
        "expected_outcome": "Borderline Flag (Moderate / Threshold Watch)",
        "clinical_narrative": (
            "71-year-old patient exhibiting insidious but low-amplitude decline across multiple organs. "
            "While no single biomarker is critically abnormal, the concurrent drift in renal clearance, "
            "natriuretic peptide levels, and oxygenation places this patient near the decision boundary."
        ),
        "sample_data": {
            "patient_id": "PT-BORDER-004",
            "patient_name": "Synthetic Case D — Borderline Fragile",
            "visits": visits,
            "threshold": 0.335
        }
    }


# ── Scenario 5: Cardiogenic Shock Pre-cursor ────────────────────────────────

def get_cardiogenic_shock() -> Dict[str, Any]:
    # Rapid hemodynamic collapse in final visits
    hr = [82.0, 86.0, 94.0, 105.0, 118.0, 128.0]
    sbp = [128.0, 120.0, 108.0, 94.0, 84.0, 76.0]
    dbp = [78.0, 72.0, 65.0, 56.0, 50.0, 44.0]
    mbp = [94.0, 88.0, 79.0, 68.0, 61.0, 54.0]
    resp = [18.0, 20.0, 23.0, 27.0, 31.0, 35.0]
    spo2 = [96.0, 95.0, 93.0, 90.0, 87.0, 83.0]
    trop_t = [0.03, 0.06, 0.15, 0.35, 0.65, 0.95]
    bnp = [210.0, 350.0, 580.0, 920.0, 1380.0, 1950.0]
    creat = [1.0, 1.25, 1.6, 2.1, 2.8, 3.6]
    bun = [18.0, 26.0, 38.0, 54.0, 72.0, 92.0]
    wbc = [8.5, 10.5, 13.0, 16.5, 20.0, 24.5]
    bicarb = [24.0, 22.0, 19.5, 17.0, 14.5, 12.0]

    visits = []
    for i in range(6):
        data = {
            "heart_rate_mean": hr[i], "sbp_mean": sbp[i], "dbp_mean": dbp[i], "mbp_mean": mbp[i],
            "resp_rate_mean": resp[i], "spo2_mean": spo2[i], "troponin_t_mean": trop_t[i],
            "troponin_t_max": trop_t[i] * 1.25, "bnp_mean": bnp[i], "bnp_max": bnp[i] * 1.2,
            "creatinine_mean": creat[i], "bun_mean": bun[i], "wbc_mean": wbc[i],
            "bicarbonate_mean": bicarb[i], "age": 77.0, "prior_mi": 1.0, "prior_hf": 1.0
        }
        visits.append(_generate_visit(data, i))

    return {
        "id": "DEMO-005",
        "name": "Pre-Shock Hemodynamic Collapse",
        "subtitle": "Severe refractory hypotension, metabolic acidosis, and end-organ hypoperfusion",
        "expected_outcome": "Critical Worsening (Immediate Escalation Required)",
        "clinical_narrative": (
            "77-year-old patient entering overt cardiogenic shock and multi-system hypoperfusion. "
            "Mean Arterial Pressure has collapsed to 54 mmHg despite severe tachycardia (128 bpm). "
            "Serum bicarbonate dropped to 12 mEq/L (lactic acidemia) and creatinine surged to 3.6 mg/dL, "
            "indicating acute fulminant deterioration."
        ),
        "sample_data": {
            "patient_id": "PT-SHOCK-005",
            "patient_name": "Synthetic Case E — Cardiogenic Shock Trajectory",
            "visits": visits,
            "threshold": 0.335
        }
    }


ALL_SCENARIOS = [
    get_stable_recovery(),
    get_decompensated_hf(),
    get_post_mi_deterioration(),
    get_borderline_fragile(),
    get_cardiogenic_shock(),
]
