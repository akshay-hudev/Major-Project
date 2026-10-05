"""Feature engineering and clinical feature dictionary for Longitudinal Cardiac Progression."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np

# ── Canonical 59 Features in exact MIMIC-IV order ────────────────────────────

FEATURE_NAMES: List[str] = [
    # Vital signs (16)
    "heart_rate_mean", "heart_rate_std",
    "sbp_mean", "sbp_std",
    "dbp_mean", "dbp_std",
    "mbp_mean", "mbp_std",
    "resp_rate_mean", "resp_rate_std",
    "temperature_mean", "temperature_std",
    "spo2_mean", "spo2_std",
    "glucose_fingerstick_mean", "glucose_fingerstick_std",
    # Cardiac markers (8)
    "troponin_t_mean", "troponin_t_max",
    "troponin_i_mean", "troponin_i_max",
    "bnp_mean", "bnp_max",
    "nt_probnp_mean", "nt_probnp_max",
    # Metabolic (6)
    "creatinine_mean", "creatinine_max",
    "bun_mean", "bun_max",
    "glucose_lab_mean", "glucose_lab_std",
    # Electrolytes (8)
    "potassium_mean", "potassium_std",
    "sodium_mean", "sodium_std",
    "chloride_mean", "chloride_std",
    "bicarbonate_mean", "bicarbonate_std",
    # CBC (8)
    "hematocrit_mean", "hematocrit_std",
    "hemoglobin_mean", "hemoglobin_std",
    "wbc_mean", "wbc_max",
    "platelets_mean", "platelets_std",
    # Clinical context (8)
    "age", "los_days", "num_procedures",
    "prior_mi", "prior_hf", "prior_arrhythmia",
    "num_medications", "icu_flag",
    # Admission-level & Temporal (5)
    "admission_type_emergency", "admission_type_elective",
    "days_since_last_admission", "time_delta_days", "visit_index",
]

NUM_FEATURES = len(FEATURE_NAMES)  # 59
SEQUENCE_LENGTH = 6

# ── Feature Metadata Dictionary ──────────────────────────────────────────────

FEATURE_METADATA: Dict[str, Dict[str, Any]] = {
    # Vitals
    "heart_rate_mean": {
        "name": "Heart Rate", "category": "Vital Signs", "unit": "bpm",
        "normal_min": 60, "normal_max": 100, "default": 78.0,
        "description": "Tachycardia (>100 bpm) or bradycardia indicates myocardial stress or failure."
    },
    "heart_rate_std": {
        "name": "Heart Rate Variability", "category": "Vital Signs", "unit": "bpm",
        "normal_min": 2, "normal_max": 8, "default": 4.0,
        "description": "Blunted or excessive variability correlates with autonomic dysfunction in shock."
    },
    "sbp_mean": {
        "name": "Systolic Blood Pressure", "category": "Vital Signs", "unit": "mmHg",
        "normal_min": 90, "normal_max": 130, "default": 120.0,
        "description": "Falling systolic BP across ICU visits indicates declining left ventricular output."
    },
    "sbp_std": {
        "name": "Systolic BP Volatility", "category": "Vital Signs", "unit": "mmHg",
        "normal_min": 4, "normal_max": 14, "default": 7.0,
        "description": "Hemodynamic instability with high pressure swings."
    },
    "dbp_mean": {
        "name": "Diastolic Blood Pressure", "category": "Vital Signs", "unit": "mmHg",
        "normal_min": 60, "normal_max": 85, "default": 75.0,
        "description": "Important for coronary perfusion pressure."
    },
    "dbp_std": {
        "name": "Diastolic BP Volatility", "category": "Vital Signs", "unit": "mmHg",
        "normal_min": 3, "normal_max": 10, "default": 5.0,
        "description": "Volatility in diastolic blood pressure."
    },
    "mbp_mean": {
        "name": "Mean Arterial Pressure", "category": "Vital Signs", "unit": "mmHg",
        "normal_min": 70, "normal_max": 100, "default": 88.0,
        "description": "Crucial organ perfusion benchmark; <65 mmHg signals end-organ hypoperfusion."
    },
    "mbp_std": {
        "name": "Mean Arterial Pressure Volatility", "category": "Vital Signs", "unit": "mmHg",
        "normal_min": 3, "normal_max": 10, "default": 5.5,
        "description": "Volatility in mean perfusion pressure."
    },
    "resp_rate_mean": {
        "name": "Respiratory Rate", "category": "Vital Signs", "unit": "breaths/min",
        "normal_min": 12, "normal_max": 20, "default": 17.0,
        "description": "Tachypnea (>22) signals pulmonary congestion or metabolic acidemia."
    },
    "resp_rate_std": {
        "name": "Respiratory Rate Variability", "category": "Vital Signs", "unit": "/min",
        "normal_min": 1, "normal_max": 4, "default": 2.0,
        "description": "Unstable breathing pattern associated with decompensation."
    },
    "temperature_mean": {
        "name": "Core Temperature", "category": "Vital Signs", "unit": "°C",
        "normal_min": 36.5, "normal_max": 37.5, "default": 37.0,
        "description": "Hypothermia or pyrexia adds hemodynamic burden."
    },
    "temperature_std": {
        "name": "Temperature Variability", "category": "Vital Signs", "unit": "°C",
        "normal_min": 0.1, "normal_max": 0.6, "default": 0.3,
        "description": "Thermal instability."
    },
    "spo2_mean": {
        "name": "Oxygen Saturation (SpO₂)", "category": "Vital Signs", "unit": "%",
        "normal_min": 95, "normal_max": 100, "default": 97.0,
        "description": "One of the most sensitive temporal flags; progressive desaturation denotes acute worsening."
    },
    "spo2_std": {
        "name": "SpO₂ Volatility", "category": "Vital Signs", "unit": "%",
        "normal_min": 0.5, "normal_max": 2.0, "default": 1.0,
        "description": "Desaturation dips are a strong hallmark of cardiorespiratory deterioration."
    },
    "glucose_fingerstick_mean": {
        "name": "Fingerstick Glucose", "category": "Vital Signs", "unit": "mg/dL",
        "normal_min": 70, "normal_max": 140, "default": 115.0,
        "description": "Stress hyperglycemia or hypoglycemic instability."
    },
    "glucose_fingerstick_std": {
        "name": "Glucose Volatility", "category": "Vital Signs", "unit": "mg/dL",
        "normal_min": 5, "normal_max": 25, "default": 12.0,
        "description": "Glycemic swings reflect severe systemic stress."
    },

    # Cardiac Markers
    "troponin_t_mean": {
        "name": "Troponin T Mean", "category": "Cardiac Biomarkers", "unit": "ng/mL",
        "normal_min": 0.0, "normal_max": 0.014, "default": 0.02,
        "description": "Myocardial cell damage biomarker; elevation denotes ongoing myocardial injury."
    },
    "troponin_t_max": {
        "name": "Troponin T Peak", "category": "Cardiac Biomarkers", "unit": "ng/mL",
        "normal_min": 0.0, "normal_max": 0.014, "default": 0.03,
        "description": "Peak myocardial damage in ICU admission."
    },
    "troponin_i_mean": {
        "name": "Troponin I Mean", "category": "Cardiac Biomarkers", "unit": "ng/mL",
        "normal_min": 0.0, "normal_max": 0.04, "default": 0.01,
        "description": "Specific cardiac troponin isoform for ischemia."
    },
    "troponin_i_max": {
        "name": "Troponin I Peak", "category": "Cardiac Biomarkers", "unit": "ng/mL",
        "normal_min": 0.0, "normal_max": 0.04, "default": 0.02,
        "description": "Peak Troponin I during ICU stay."
    },
    "bnp_mean": {
        "name": "Brain Natriuretic Peptide (BNP)", "category": "Cardiac Biomarkers", "unit": "pg/mL",
        "normal_min": 0, "normal_max": 100, "default": 160.0,
        "description": "Ventricular stretch marker; exponential rise indicates heart failure decompensation."
    },
    "bnp_max": {
        "name": "BNP Peak", "category": "Cardiac Biomarkers", "unit": "pg/mL",
        "normal_min": 0, "normal_max": 100, "default": 200.0,
        "description": "Highest BNP measurement recorded in the stay."
    },
    "nt_probnp_mean": {
        "name": "NT-proBNP Mean", "category": "Cardiac Biomarkers", "unit": "pg/mL",
        "normal_min": 0, "normal_max": 300, "default": 350.0,
        "description": "Longer half-life natriuretic peptide reflecting sustained ventricular strain."
    },
    "nt_probnp_max": {
        "name": "NT-proBNP Peak", "category": "Cardiac Biomarkers", "unit": "pg/mL",
        "normal_min": 0, "normal_max": 300, "default": 420.0,
        "description": "Peak NT-proBNP value."
    },

    # Renal & Metabolic
    "creatinine_mean": {
        "name": "Serum Creatinine", "category": "Renal & Metabolic", "unit": "mg/dL",
        "normal_min": 0.6, "normal_max": 1.2, "default": 1.0,
        "description": "Elevations signal cardiorenal syndrome and poor renal perfusion."
    },
    "creatinine_max": {
        "name": "Peak Creatinine", "category": "Renal & Metabolic", "unit": "mg/dL",
        "normal_min": 0.6, "normal_max": 1.2, "default": 1.1,
        "description": "Maximum creatinine during the stay."
    },
    "bun_mean": {
        "name": "Blood Urea Nitrogen (BUN)", "category": "Renal & Metabolic", "unit": "mg/dL",
        "normal_min": 7, "normal_max": 20, "default": 18.0,
        "description": "Elevated BUN/creatinine ratio reflects neurohormonal activation and hypovolemia."
    },
    "bun_max": {
        "name": "Peak BUN", "category": "Renal & Metabolic", "unit": "mg/dL",
        "normal_min": 7, "normal_max": 20, "default": 21.0,
        "description": "Maximum BUN."
    },
    "glucose_lab_mean": {
        "name": "Lab Blood Glucose", "category": "Renal & Metabolic", "unit": "mg/dL",
        "normal_min": 70, "normal_max": 110, "default": 112.0,
        "description": "Serum laboratory glucose measurement."
    },
    "glucose_lab_std": {
        "name": "Lab Glucose Volatility", "category": "Renal & Metabolic", "unit": "mg/dL",
        "normal_min": 4, "normal_max": 20, "default": 10.0,
        "description": "Volatility in serum glucose."
    },

    # Electrolytes
    "potassium_mean": {
        "name": "Serum Potassium", "category": "Electrolytes", "unit": "mEq/L",
        "normal_min": 3.5, "normal_max": 5.0, "default": 4.2,
        "description": "Dyskalemia triggers life-threatening cardiac arrhythmias."
    },
    "potassium_std": {
        "name": "Potassium Volatility", "category": "Electrolytes", "unit": "mEq/L",
        "normal_min": 0.1, "normal_max": 0.4, "default": 0.2,
        "description": "Fluctuations in potassium."
    },
    "sodium_mean": {
        "name": "Serum Sodium", "category": "Electrolytes", "unit": "mEq/L",
        "normal_min": 135, "normal_max": 145, "default": 139.0,
        "description": "Hyponatremia (<135) is an ominous sign of end-stage heart failure."
    },
    "sodium_std": {
        "name": "Sodium Volatility", "category": "Electrolytes", "unit": "mEq/L",
        "normal_min": 0.5, "normal_max": 2.5, "default": 1.4,
        "description": "Sodium fluctuations."
    },
    "chloride_mean": {
        "name": "Serum Chloride", "category": "Electrolytes", "unit": "mEq/L",
        "normal_min": 96, "normal_max": 106, "default": 102.0,
        "description": "Important for acid-base equilibrium and diuretic responsiveness."
    },
    "chloride_std": {
        "name": "Chloride Volatility", "category": "Electrolytes", "unit": "mEq/L",
        "normal_min": 0.5, "normal_max": 2.5, "default": 1.5,
        "description": "Chloride fluctuations."
    },
    "bicarbonate_mean": {
        "name": "Serum Bicarbonate", "category": "Electrolytes", "unit": "mEq/L",
        "normal_min": 22, "normal_max": 29, "default": 24.5,
        "description": "Low bicarbonate (<22) reflects metabolic acidosis and tissue hypoperfusion."
    },
    "bicarbonate_std": {
        "name": "Bicarbonate Volatility", "category": "Electrolytes", "unit": "mEq/L",
        "normal_min": 0.5, "normal_max": 2.0, "default": 1.2,
        "description": "Bicarbonate volatility."
    },

    # Hematology (CBC)
    "hematocrit_mean": {
        "name": "Hematocrit", "category": "Hematology (CBC)", "unit": "%",
        "normal_min": 36, "normal_max": 50, "default": 39.0,
        "description": "Anemia reduces oxygen delivery to failing cardiac tissue."
    },
    "hematocrit_std": {
        "name": "Hematocrit Volatility", "category": "Hematology (CBC)", "unit": "%",
        "normal_min": 0.5, "normal_max": 2.5, "default": 1.3,
        "description": "Rapid shifts in hematocrit indicate active bleeding or hemodilution."
    },
    "hemoglobin_mean": {
        "name": "Hemoglobin", "category": "Hematology (CBC)", "unit": "g/dL",
        "normal_min": 12.0, "normal_max": 17.5, "default": 13.0,
        "description": "Low hemoglobin exacerbates cardiac ischemia."
    },
    "hemoglobin_std": {
        "name": "Hemoglobin Volatility", "category": "Hematology (CBC)", "unit": "g/dL",
        "normal_min": 0.2, "normal_max": 1.0, "default": 0.5,
        "description": "Hemoglobin variability."
    },
    "wbc_mean": {
        "name": "White Blood Cell Count (WBC)", "category": "Hematology (CBC)", "unit": "K/µL",
        "normal_min": 4.5, "normal_max": 11.0, "default": 7.8,
        "description": "Leukocytosis (>11.0) indicates systemic inflammation, infection, or post-infarction stress."
    },
    "wbc_max": {
        "name": "Peak WBC Count", "category": "Hematology (CBC)", "unit": "K/µL",
        "normal_min": 4.5, "normal_max": 11.0, "default": 8.5,
        "description": "Maximum WBC count during the stay."
    },
    "platelets_mean": {
        "name": "Platelet Count", "category": "Hematology (CBC)", "unit": "K/µL",
        "normal_min": 150, "normal_max": 450, "default": 230.0,
        "description": "Thrombocytopenia correlates with sepsis or severe multisystem illness."
    },
    "platelets_std": {
        "name": "Platelet Volatility", "category": "Hematology (CBC)", "unit": "K/µL",
        "normal_min": 5, "normal_max": 30, "default": 15.0,
        "description": "Platelet swings."
    },

    # Clinical Context & History
    "age": {
        "name": "Patient Age", "category": "Clinical Context", "unit": "years",
        "normal_min": 18, "normal_max": 95, "default": 67.0,
        "description": "Advanced age increases vulnerability to cardiovascular decline."
    },
    "los_days": {
        "name": "Length of Stay", "category": "Clinical Context", "unit": "days",
        "normal_min": 1, "normal_max": 30, "default": 3.4,
        "description": "Extended ICU stay duration reflects clinical complexity."
    },
    "num_procedures": {
        "name": "Number of Procedures", "category": "Clinical Context", "unit": "count",
        "normal_min": 0, "normal_max": 10, "default": 1.0,
        "description": "Procedural invasiveness."
    },
    "prior_mi": {
        "name": "Prior Myocardial Infarction", "category": "Clinical Context", "unit": "flag",
        "normal_min": 0, "normal_max": 1, "default": 0.0,
        "description": "History of heart attack increases recurring event probability."
    },
    "prior_hf": {
        "name": "Prior Heart Failure", "category": "Clinical Context", "unit": "flag",
        "normal_min": 0, "normal_max": 1, "default": 1.0,
        "description": "Pre-existing congestive heart failure."
    },
    "prior_arrhythmia": {
        "name": "Prior Arrhythmia", "category": "Clinical Context", "unit": "flag",
        "normal_min": 0, "normal_max": 1, "default": 0.0,
        "description": "History of atrial fibrillation or ventricular arrhythmias."
    },
    "num_medications": {
        "name": "Active Medications Count", "category": "Clinical Context", "unit": "count",
        "normal_min": 1, "normal_max": 25, "default": 8.0,
        "description": "Polypharmacy marker."
    },
    "icu_flag": {
        "name": "ICU Stay Flag", "category": "Clinical Context", "unit": "flag",
        "normal_min": 0, "normal_max": 1, "default": 1.0,
        "description": "Admission to Intensive Care Unit."
    },

    # Temporal & Admission
    "admission_type_emergency": {
        "name": "Emergency Admission", "category": "Temporal & Context", "unit": "flag",
        "normal_min": 0, "normal_max": 1, "default": 1.0,
        "description": "Urgent/Emergency admission vs elective."
    },
    "admission_type_elective": {
        "name": "Elective Admission", "category": "Temporal & Context", "unit": "flag",
        "normal_min": 0, "normal_max": 1, "default": 0.0,
        "description": "Elective planned admission."
    },
    "days_since_last_admission": {
        "name": "Days Since Last Admission", "category": "Temporal & Context", "unit": "days",
        "normal_min": 1, "normal_max": 365, "default": 45.0,
        "description": "Rapid readmission (<30 days) is a known high-risk flag."
    },
    "time_delta_days": {
        "name": "Visit Time Delta", "category": "Temporal & Context", "unit": "days",
        "normal_min": 1, "normal_max": 180, "default": 28.0,
        "description": "Interval between sequential hospital encounters."
    },
    "visit_index": {
        "name": "Visit Index", "category": "Temporal & Context", "unit": "index",
        "normal_min": 1, "normal_max": 6, "default": 6.0,
        "description": "Sequence step in the 6-stay trajectory."
    },
}


# ── Feature Engineering Pipeline ─────────────────────────────────────────────

def dicts_to_array(visits: List[Dict[str, Any]]) -> np.ndarray:
    """Convert list of 6 visit dictionaries into a (6, 59) NumPy array."""
    if len(visits) != SEQUENCE_LENGTH:
        raise ValueError(f"Expected {SEQUENCE_LENGTH} visits, received {len(visits)}")

    arr = np.zeros((SEQUENCE_LENGTH, NUM_FEATURES), dtype=np.float32)
    for t, visit in enumerate(visits):
        for f_idx, feat in enumerate(FEATURE_NAMES):
            val = visit.get(feat)
            if val is None:
                val = FEATURE_METADATA.get(feat, {}).get("default", 0.0)
            arr[t, f_idx] = float(val)

    # Sanitize NaN/Inf
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def extract_temporal_features(X: np.ndarray) -> np.ndarray:
    """Transform (N, T, F) or (T, F) into the engineered temporal feature matrix.
    
    Produces:
      - Raw flattened sequence: F * T (59 * 6 = 354)
      - Last visit values: F (59)
      - First visit values: F (59)
      - Mean values: F (59)
      - Standard deviation (volatility): F (59)
      - Minimum values: F (59)
      - Maximum values: F (59)
      - Net Delta (last - first): F (59)
      - Linear slope trend: F (59)
    Total = 59 * (6 + 8) = 826 features.
    """
    if X.ndim == 2:
        X = X[np.newaxis, :, :]  # Add batch dim -> (1, T, F)

    N, T, F = X.shape
    last = X[:, -1, :]
    first = X[:, 0, :]
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    mn = X.min(axis=1)
    mx = X.max(axis=1)
    delta = last - first

    t = np.arange(T, dtype=np.float64)
    t -= t.mean()
    denom = (t ** 2).sum()
    slope = (X * t[None, :, None]).sum(axis=1) / (denom + 1e-8)

    flat = X.reshape(N, T * F)
    features = np.concatenate(
        [flat, last, first, mean, std, mn, mx, delta, slope], axis=1
    ).astype(np.float32)

    return features


def get_feature_names_engineered() -> List[str]:
    """Generate human-readable names for all 826 engineered features."""
    names = []
    # Flat features @t0 to @t5
    for t in range(SEQUENCE_LENGTH):
        for f in FEATURE_NAMES:
            names.append(f"{f}@t{t}")
    # Summary descriptors
    for tag in ["last", "first", "mean", "std", "min", "max", "delta", "slope"]:
        for f in FEATURE_NAMES:
            names.append(f"{f}_{tag}")
    return names
