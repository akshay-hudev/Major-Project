"""FastAPI Application for Longitudinal Temporal Disease Progression Prediction."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.feature_extractor import FEATURE_METADATA, FEATURE_NAMES, NUM_FEATURES, SEQUENCE_LENGTH
from backend.inference_engine import get_inference_engine
from backend.patient_presets import ALL_SCENARIOS
from backend.schemas import (
    BatchPatientRecord,
    BatchPatientSummary,
    BatchPredictionRequest,
    BatchPredictionResponse,
    FeatureDefinition,
    PatientTrajectoryRequest,
    PredictionResponse,
    ScenarioSummary,
    SystemStatusResponse,
)

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT / "frontend"
RESULTS_ASSETS_DIR = ROOT / "docs" / "assets" / "results"
RESULTS_IMPROVED_PATH = ROOT / "results_improved.json"
RESULTS_BASELINE_PATH = ROOT / "results.json"

app = FastAPI(
    title="Longitudinal Temporal Disease Progression API",
    description="Deep learning and gradient boosting API for predicting acute cardiac progression in ICU patients across 6 visits.",
    version="2.0.0",
)

# Enable CORS for local testing and web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── System Health & Status ───────────────────────────────────────────────────

@app.get("/api/status", response_model=SystemStatusResponse)
def get_system_status() -> SystemStatusResponse:
    """Return system readiness, active model metadata, and benchmark performance."""
    engine = get_inference_engine()
    return SystemStatusResponse(
        status="operational",
        service="Cardiac Progression Temporal Intelligence",
        version="2.0.0",
        active_model=engine.model_name,
        feature_count=NUM_FEATURES,
        sequence_length=SEQUENCE_LENGTH,
        dataset="MIMIC-IV 2.1 Cardiac Progression Cohort (35,256 stays)",
        best_benchmark_auc=0.9462,
        best_benchmark_f1=0.7334,
    )


# ── Feature Definitions & Metadata ───────────────────────────────────────────

@app.get("/api/features", response_model=List[FeatureDefinition])
def get_features_dictionary() -> List[FeatureDefinition]:
    """Return dictionary of all 59 clinical features with units, ranges, and clinical context."""
    results = []
    for feat_id in FEATURE_NAMES:
        meta = FEATURE_METADATA.get(feat_id, {})
        results.append(
            FeatureDefinition(
                id=feat_id,
                name=meta.get("name", feat_id),
                category=meta.get("category", "General"),
                unit=meta.get("unit", ""),
                normal_min=meta.get("normal_min"),
                normal_max=meta.get("normal_max"),
                default_val=float(meta.get("default", 0.0)),
                description=meta.get("description", "MIMIC-IV clinical parameter."),
            )
        )
    return results


# ── Patient Scenarios ────────────────────────────────────────────────────────

@app.get("/api/scenarios", response_model=List[ScenarioSummary])
def get_scenarios() -> List[ScenarioSummary]:
    """Retrieve pre-built realistic longitudinal patient trajectories."""
    summaries = []
    for s in ALL_SCENARIOS:
        summaries.append(
            ScenarioSummary(
                id=s["id"],
                name=s["name"],
                subtitle=s["subtitle"],
                expected_outcome=s["expected_outcome"],
                clinical_narrative=s["clinical_narrative"],
                sample_data=PatientTrajectoryRequest(**s["sample_data"]),
            )
        )
    return summaries


@app.get("/api/scenarios/{scenario_id}", response_model=ScenarioSummary)
def get_scenario_by_id(scenario_id: str) -> ScenarioSummary:
    """Retrieve a specific patient scenario by ID."""
    for s in ALL_SCENARIOS:
        if s["id"].lower() == scenario_id.lower():
            return ScenarioSummary(
                id=s["id"],
                name=s["name"],
                subtitle=s["subtitle"],
                expected_outcome=s["expected_outcome"],
                clinical_narrative=s["clinical_narrative"],
                sample_data=PatientTrajectoryRequest(**s["sample_data"]),
            )
    raise HTTPException(status_code=404, detail=f"Scenario '{scenario_id}' not found.")


# ── Prediction Endpoints ─────────────────────────────────────────────────────

@app.post("/api/predict", response_model=PredictionResponse)
def predict_cardiac_progression(request: PatientTrajectoryRequest) -> PredictionResponse:
    """Run temporal feature engineering and inference on a 6-visit patient trajectory."""
    if len(request.visits) != SEQUENCE_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Exactly {SEQUENCE_LENGTH} visits are required. Received {len(request.visits)}.",
        )

    engine = get_inference_engine()
    result = engine.predict(
        visits=request.visits,
        patient_id=request.patient_id or "PT-ANONYMOUS",
        custom_threshold=request.threshold,
    )
    return result


@app.post("/api/predict/batch", response_model=BatchPredictionResponse)
def batch_predict(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Run inference across a batch of multiple patient trajectories."""
    if not request.patients:
        raise HTTPException(status_code=400, detail="Patients list is empty.")

    engine = get_inference_engine()
    summaries: List[BatchPatientSummary] = []
    worsening_count = 0

    for patient in request.patients:
        if len(patient.visits) != SEQUENCE_LENGTH:
            continue

        res = engine.predict(
            visits=patient.visits,
            patient_id=patient.patient_id,
            custom_threshold=request.threshold,
        )
        if res.prediction == 1:
            worsening_count += 1

        top_driver = res.top_contributors[0].feature_name if res.top_contributors else "Unspecified"
        summaries.append(
            BatchPatientSummary(
                patient_id=res.patient_id,
                probability=res.probability,
                prediction=res.prediction,
                classification=res.classification,
                risk_level=res.risk_level,
                primary_risk_driver=top_driver,
            )
        )

    total = len(summaries)
    rate = round(worsening_count / total, 4) if total > 0 else 0.0

    return BatchPredictionResponse(
        total_processed=total,
        worsening_count=worsening_count,
        stable_count=total - worsening_count,
        worsening_rate=rate,
        results=summaries,
    )


@app.post("/api/predict/csv")
async def upload_csv_predict(file: UploadFile = File(...)) -> BatchPredictionResponse:
    """Upload a CSV file containing multiple patient visits and score them in batches.
    
    Expected format: patient_id, visit_index (1 to 6), feature_1, feature_2, ...
    """
    content = await file.read()
    text = content.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))

    # Group by patient_id
    patient_visits: Dict[str, List[Dict[str, float]]] = {}
    for row in reader:
        pid = row.get("patient_id", f"PT-{len(patient_visits)+1}")
        clean_row: Dict[str, float] = {}
        for k, v in row.items():
            if k in FEATURE_NAMES:
                try:
                    clean_row[k] = float(v)
                except (ValueError, TypeError):
                    clean_row[k] = float(FEATURE_METADATA.get(k, {}).get("default", 0.0))

        if pid not in patient_visits:
            patient_visits[pid] = []
        patient_visits[pid].append(clean_row)

    batch_patients = []
    for pid, visits in patient_visits.items():
        # Ensure exactly 6 visits (pad or truncate)
        if len(visits) < SEQUENCE_LENGTH:
            last_v = visits[-1] if visits else {}
            while len(visits) < SEQUENCE_LENGTH:
                visits.append(last_v.copy())
        elif len(visits) > SEQUENCE_LENGTH:
            visits = visits[:SEQUENCE_LENGTH]

        batch_patients.append(BatchPatientRecord(patient_id=pid, visits=visits))  # type: ignore

    return batch_predict(BatchPredictionRequest(patients=batch_patients))


# ── Sample CSV Generator ─────────────────────────────────────────────────────

@app.get("/api/sample-csv")
def download_sample_csv():
    """Generate and return a ready-to-test CSV with sample patient trajectories."""
    output = io.StringIO()
    headers = ["patient_id"] + FEATURE_NAMES
    writer = csv.DictWriter(output, fieldnames=headers)
    writer.writeheader()

    for s in ALL_SCENARIOS:
        pid = s["sample_data"]["patient_id"]
        visits = s["sample_data"]["visits"]
        for idx, v in enumerate(visits):
            row = {"patient_id": pid, "visit_index": idx + 1}
            for feat in FEATURE_NAMES:
                row[feat] = v.get(feat, 0.0)
            writer.writerow(row)

    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=cardiac_patient_samples.csv"},
    )


# ── Model Benchmarks ─────────────────────────────────────────────────────────

@app.get("/api/benchmarks")
def get_benchmarks() -> Dict[str, Any]:
    """Retrieve full comparative metrics, confusion matrices, and training specs for all models."""
    models_comparison = [
        {
            "name": "GBM Soft-Vote Ensemble ⭐",
            "type": "Ensemble (HistGB + LightGBM + XGBoost)",
            "accuracy": 0.9083,
            "precision": 0.7226,
            "recall": 0.7444,
            "f1": 0.7334,
            "auc_roc": 0.9462,
            "pr_auc": 0.8302,
            "is_best": True,
            "notes": "State-of-the-art on temporal features; calibrated + F1-tuned threshold."
        },
        {
            "name": "GBM Stacked Ensemble",
            "type": "Meta-Learner Ensemble",
            "accuracy": 0.9125,
            "precision": 0.7612,
            "recall": 0.7042,
            "f1": 0.7316,
            "auc_roc": 0.9462,
            "pr_auc": 0.8307,
            "is_best": False,
            "notes": "Logistic regression stacker on out-of-fold GBM predictions."
        },
        {
            "name": "HistGradientBoosting (Rich)",
            "type": "Histogram Decision Trees",
            "accuracy": 0.9126,
            "precision": 0.7747,
            "recall": 0.6830,
            "f1": 0.7260,
            "auc_roc": 0.9434,
            "pr_auc": 0.8093,
            "is_best": False,
            "notes": "Highest single-model precision; lightweight 3.3MB CPU artifact."
        },
        {
            "name": "LightGBM",
            "type": "Gradient Boosted Trees",
            "accuracy": 0.9079,
            "precision": 0.7332,
            "recall": 0.7176,
            "f1": 0.7253,
            "auc_roc": 0.9439,
            "pr_auc": 0.8153,
            "is_best": False,
            "notes": "Fast leaf-wise tree splitting."
        },
        {
            "name": "XGBoost",
            "type": "Extreme Gradient Boosting",
            "accuracy": 0.9032,
            "precision": 0.6983,
            "recall": 0.7545,
            "f1": 0.7253,
            "auc_roc": 0.9433,
            "pr_auc": 0.8069,
            "is_best": False,
            "notes": "Highest recall among tree models."
        },
        {
            "name": "Transformer Encoder",
            "type": "Deep Sequence Attention",
            "accuracy": 0.8399,
            "precision": 0.5174,
            "recall": 0.8114,
            "f1": 0.6319,
            "auc_roc": 0.9168,
            "pr_auc": 0.6720,
            "is_best": False,
            "notes": "Best neural sequence baseline. High recall but lower precision due to imbalance."
        },
        {
            "name": "BiGRU",
            "type": "Recurrent Neural Network",
            "accuracy": 0.8211,
            "precision": 0.4837,
            "recall": 0.8259,
            "f1": 0.6101,
            "auc_roc": 0.9099,
            "pr_auc": 0.6527,
            "is_best": False,
            "notes": "Bidirectional GRU sequence model."
        },
        {
            "name": "BiLSTM Attention",
            "type": "Recurrent + Bahdanau Attention",
            "accuracy": 0.8308,
            "precision": 0.5003,
            "recall": 0.8103,
            "f1": 0.6187,
            "auc_roc": 0.9060,
            "pr_auc": 0.6376,
            "is_best": False,
            "notes": "BiLSTM with temporal attention over the 6 timesteps."
        },
    ]

    return {
        "dataset_summary": {
            "name": "MIMIC-IV 2.1 Cardiac Progression",
            "total_samples": 35256,
            "train_samples": 24678,
            "val_samples": 5289,
            "test_samples": 5289,
            "positive_prevalence": "16.9%",
            "sequence_length": 6,
            "feature_dim": 59,
        },
        "models": models_comparison,
        "figures": [
            {"id": "comparison", "title": "Model Comparison", "filename": "improved_model_comparison.png"},
            {"id": "roc", "title": "ROC Curves", "filename": "improved_roc_curves.png"},
            {"id": "pr", "title": "Precision-Recall Curves", "filename": "improved_pr_curves.png"},
            {"id": "confusion", "title": "Confusion Matrix (GBM Ensemble)", "filename": "improved_confusion_matrix.png"},
            {"id": "features", "title": "Temporal Feature Importance", "filename": "improved_feature_importance.png"},
        ],
    }


# ── Static Results Plots Serving ─────────────────────────────────────────────

if RESULTS_ASSETS_DIR.exists():
    app.mount("/api/results/plots", StaticFiles(directory=str(RESULTS_ASSETS_DIR)), name="result_plots")


# ── Frontend Static Files ────────────────────────────────────────────────────

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def serve_frontend_index():
        return FileResponse(FRONTEND_DIR / "index.html")
