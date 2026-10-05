"""Data models and schemas for the Cardiac Progression Temporal Prediction API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Feature Metadata ─────────────────────────────────────────────────────────

class FeatureDefinition(BaseModel):
    id: str
    name: str
    category: str
    unit: str
    normal_min: Optional[float] = None
    normal_max: Optional[float] = None
    default_val: float
    description: str


# ── Visit & Trajectory Data ──────────────────────────────────────────────────

class VisitData(BaseModel):
    """Represents clinical features for a single ICU visit (1 of 6)."""
    # Key vitals (frequently observed & edited)
    heart_rate_mean: float = Field(..., description="Mean Heart Rate (bpm)")
    heart_rate_std: float = Field(default=3.5, description="Heart Rate Std Dev")
    sbp_mean: float = Field(..., description="Systolic Blood Pressure (mmHg)")
    sbp_std: float = Field(default=8.0, description="Systolic BP Std Dev")
    dbp_mean: float = Field(default=75.0, description="Diastolic Blood Pressure (mmHg)")
    dbp_std: float = Field(default=6.0, description="Diastolic BP Std Dev")
    mbp_mean: float = Field(default=90.0, description="Mean Blood Pressure (mmHg)")
    mbp_std: float = Field(default=6.5, description="Mean BP Std Dev")
    resp_rate_mean: float = Field(..., description="Respiratory Rate (breaths/min)")
    resp_rate_std: float = Field(default=2.0, description="Respiratory Rate Std Dev")
    temperature_mean: float = Field(default=37.0, description="Core Temperature (°C)")
    temperature_std: float = Field(default=0.3, description="Temperature Std Dev")
    spo2_mean: float = Field(..., description="Pulse Oximetry Oxygen Saturation (%)")
    spo2_std: float = Field(default=1.0, description="SpO2 Std Dev")
    glucose_fingerstick_mean: float = Field(default=110.0, description="Fingerstick Glucose (mg/dL)")
    glucose_fingerstick_std: float = Field(default=12.0, description="Fingerstick Glucose Std Dev")

    # Cardiac markers
    troponin_t_mean: float = Field(default=0.02, description="Troponin T Mean (ng/mL)")
    troponin_t_max: float = Field(default=0.03, description="Troponin T Max (ng/mL)")
    troponin_i_mean: float = Field(default=0.01, description="Troponin I Mean (ng/mL)")
    troponin_i_max: float = Field(default=0.02, description="Troponin I Max (ng/mL)")
    bnp_mean: float = Field(default=150.0, description="Brain Natriuretic Peptide Mean (pg/mL)")
    bnp_max: float = Field(default=180.0, description="Brain Natriuretic Peptide Max (pg/mL)")
    nt_probnp_mean: float = Field(default=300.0, description="NT-proBNP Mean (pg/mL)")
    nt_probnp_max: float = Field(default=350.0, description="NT-proBNP Max (pg/mL)")

    # Renal & Metabolic
    creatinine_mean: float = Field(default=1.0, description="Serum Creatinine Mean (mg/dL)")
    creatinine_max: float = Field(default=1.1, description="Serum Creatinine Max (mg/dL)")
    bun_mean: float = Field(default=18.0, description="Blood Urea Nitrogen Mean (mg/dL)")
    bun_max: float = Field(default=20.0, description="Blood Urea Nitrogen Max (mg/dL)")
    glucose_lab_mean: float = Field(default=115.0, description="Lab Glucose Mean (mg/dL)")
    glucose_lab_std: float = Field(default=10.0, description="Lab Glucose Std Dev")

    # Electrolytes
    potassium_mean: float = Field(default=4.1, description="Serum Potassium Mean (mEq/L)")
    potassium_std: float = Field(default=0.2, description="Serum Potassium Std Dev")
    sodium_mean: float = Field(default=140.0, description="Serum Sodium Mean (mEq/L)")
    sodium_std: float = Field(default=1.5, description="Serum Sodium Std Dev")
    chloride_mean: float = Field(default=102.0, description="Serum Chloride Mean (mEq/L)")
    chloride_std: float = Field(default=1.8, description="Serum Chloride Std Dev")
    bicarbonate_mean: float = Field(default=24.0, description="Serum Bicarbonate Mean (mEq/L)")
    bicarbonate_std: float = Field(default=1.2, description="Serum Bicarbonate Std Dev")

    # Complete Blood Count (CBC)
    hematocrit_mean: float = Field(default=38.0, description="Hematocrit Mean (%)")
    hematocrit_std: float = Field(default=1.5, description="Hematocrit Std Dev")
    hemoglobin_mean: float = Field(default=12.5, description="Hemoglobin Mean (g/dL)")
    hemoglobin_std: float = Field(default=0.6, description="Hemoglobin Std Dev")
    wbc_mean: float = Field(default=7.5, description="White Blood Cell Count Mean (K/µL)")
    wbc_max: float = Field(default=8.2, description="White Blood Cell Count Max (K/µL)")
    platelets_mean: float = Field(default=220.0, description="Platelets Count Mean (K/µL)")
    platelets_std: float = Field(default=15.0, description="Platelets Count Std Dev")

    # Clinical Context & History
    age: float = Field(default=68.0, description="Patient Age (years)")
    los_days: float = Field(default=3.2, description="Length of Stay (days)")
    num_procedures: float = Field(default=1.0, description="Number of Procedures")
    prior_mi: float = Field(default=0.0, description="Prior Myocardial Infarction Flag (0 or 1)")
    prior_hf: float = Field(default=1.0, description="Prior Heart Failure Flag (0 or 1)")
    prior_arrhythmia: float = Field(default=0.0, description="Prior Arrhythmia Flag (0 or 1)")
    num_medications: float = Field(default=8.0, description="Number of Active Medications")
    icu_flag: float = Field(default=1.0, description="ICU Admission Flag")

    # Admission-level & Temporal
    admission_type_emergency: float = Field(default=1.0, description="Emergency Admission (0 or 1)")
    admission_type_elective: float = Field(default=0.0, description="Elective Admission (0 or 1)")
    days_since_last_admission: float = Field(default=45.0, description="Days Since Previous Admission")
    time_delta_days: float = Field(default=30.0, description="Inter-visit Interval (days)")
    visit_index: float = Field(default=1.0, description="Index of ICU visit (1 to 6)")

    model_config = {
        "extra": "allow"
    }


class PatientTrajectoryRequest(BaseModel):
    """Prediction request for a 6-visit patient trajectory."""
    patient_id: Optional[str] = Field(default="PT-1049", description="Optional patient identifier")
    patient_name: Optional[str] = Field(default="Anonymous Patient", description="Optional name/tag")
    visits: List[VisitData] = Field(..., description="Array of 6 longitudinal ICU visits")
    threshold: Optional[float] = Field(default=0.335, description="Custom decision threshold (default: 0.335)")


# ── Prediction Output ────────────────────────────────────────────────────────

class RiskContributor(BaseModel):
    feature_name: str
    feature_id: str
    temporal_descriptor: str
    impact_score: float
    direction: str  # 'risk_elevating' or 'protective'
    explanation: str
    current_value: Optional[float] = None
    baseline_value: Optional[float] = None


class ClinicalRecommendation(BaseModel):
    urgency: str  # 'Routine', 'Priority', 'Critical'
    title: str
    rationale: str
    suggested_actions: List[str]


class PredictionResponse(BaseModel):
    patient_id: str
    probability: float  # 0.0 to 1.0
    probability_percent: str  # "74.2%"
    prediction: int  # 0 or 1
    classification: str  # 'Worsening Cardiac Progression' or 'Stable / Improving'
    decision_threshold: float
    risk_level: str  # 'Low', 'Moderate', 'High', 'Critical'
    risk_color: str  # Hex code for UI
    confidence_interval: List[float]
    top_contributors: List[RiskContributor]
    recommendations: List[ClinicalRecommendation]
    model_used: str
    timestamp: str


# ── Batch Processing ─────────────────────────────────────────────────────────

class BatchPatientRecord(BaseModel):
    patient_id: str
    visits: List[VisitData]


class BatchPredictionRequest(BaseModel):
    patients: List[BatchPatientRecord]
    threshold: Optional[float] = Field(default=0.335)


class BatchPatientSummary(BaseModel):
    patient_id: str
    probability: float
    prediction: int
    classification: str
    risk_level: str
    primary_risk_driver: str


class BatchPredictionResponse(BaseModel):
    total_processed: int
    worsening_count: int
    stable_count: int
    worsening_rate: float
    results: List[BatchPatientSummary]


# ── Presets & Metadata ───────────────────────────────────────────────────────

class ScenarioSummary(BaseModel):
    id: str
    name: str
    subtitle: str
    expected_outcome: str
    clinical_narrative: str
    sample_data: PatientTrajectoryRequest


class SystemStatusResponse(BaseModel):
    status: str
    service: str
    version: str
    active_model: str
    feature_count: int
    sequence_length: int
    dataset: str
    best_benchmark_auc: float
    best_benchmark_f1: float
    inference_mode: str = "demo_synthetic"
    research_best_model: str = "GBM Soft-Vote Ensemble"
    demo_disclaimer: str = (
        "Live predictions use the local deployable HistGBM bundle. "
        "Published MIMIC-IV metrics (F1/AUC) come from results_improved.json "
        "and are shown separately under Model Benchmarks."
    )
