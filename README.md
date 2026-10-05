# Longitudinal Temporal Disease Progression

Research prototype for **binary cardiac progression prediction** from longitudinal ICU visit sequences (MIMIC-IV 2.1).

- **Label:** worsening (`1`) vs stable/improving (`0`)
- **Input:** 6 ICU visits × 59 clinical features
- **Best published model (held-out test):** calibrated GBM soft-vote ensemble — **F1 0.7334**, **ROC-AUC 0.9462**
- **Demo app:** CardioPulse AI (FastAPI + frontend)

This is a **research / decision-support prototype**, not a clinically validated or deployment-ready medical device.

---

## Setup

### Requirements

- Python **3.11+** recommended (this machine has been validated with **Python 3.14**)
- Packages in `requirements.txt` (FastAPI, scikit-learn, joblib, torch optional for training only)

### Installation

```bash
py -3.14 -m pip install -r requirements.txt
```

Or with a virtual environment:

```bash
py -3.14 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## Running the project (recommended for review)

**One command:**

```bash
py -3.14 run_app.py
```

Then open:

| URL | Purpose |
|---|---|
| http://127.0.0.1:8000 | CardioPulse AI dashboard |
| http://127.0.0.1:8000/docs | Interactive API docs |

Options:

```bash
py -3.14 run_app.py --port 8080 --no-browser
```

### Running Demo Mode

1. Start with `py -3.14 run_app.py`
2. On **Patient Predictor**, choose a **Clinical Scenario** (synthetic cases, no PHI)
3. Inspect the 6-visit trajectory chart
4. Click **Run Temporal Analysis**
5. Read probability, risk tier, and temporal signals
6. Open **Model Benchmarks** for **published MIMIC-IV held-out metrics** (from `results_improved.json`)

**Integrity note:** Live scenario predictions use the local HistGBM inference bundle under `models/deployable/`. Published soft-vote ensemble metrics (F1 / AUC) are shown separately under Benchmarks and come from saved experiment result files — they are not invented at demo time.

Optional static page (no API):

```bash
py -3.14 demo_server.py
```

Prefer `run_app.py` for the faculty review.

### Automated smoke tests

```bash
py -3.14 -m unittest backend/test_api.py
```

---

## Model

| Track | Models |
|---|---|
| Sequence baselines | BiLSTM + attention, BiGRU, Transformer encoder |
| Improved | Temporal descriptors → HistGB / LightGBM / XGBoost + soft-vote / stacked ensembles |

**Selected research result:** soft-vote ensemble of HistGB + LightGBM + XGBoost, isotonic-calibrated, validation F1-tuned threshold **0.335**.

**Deployable demo artifact:** `models/deployable/gbm_temporal.joblib` (HistGBM + calibrator + threshold on 826 temporal features).

---

## Dataset

| Item | Value |
|---|---|
| Source | MIMIC-IV 2.1 cardiac cohort (PhysioNet; not redistributed) |
| Samples | 35,256 sequences |
| Split | 24,678 / 5,289 / 5,289 (70/15/15, stratified, `random_state=42`) |
| Positive rate | ~16.9% |
| Sequence | 6 visits × 59 features |
| Label | CSI increase >10% (mortality fallback on terminal stays) |

Raw MIMIC ZIP and preprocessed `.npy` arrays are gitignored and not required to **run the demo**.

Rebuild (credentialed MIMIC access only):

```bash
python src/preprocessing/build_cardiac_progression_dataset.py --zip data/mimic_iv_raw/mimic-iv-2-1.zip --chunk 100000 --seq_len 6
```

---

## Results

Traceable result files:

- `results.json` — original Colab sequence-model run
- `results_baseline_reproduced.json` — fair sequence baselines on shared arrays
- `results_improved.json` — GBM family + ensembles (source of best numbers)
- Plots: `docs/assets/results/`

**Verified best (soft-vote, calibrated/tuned):**

| Metric | Value |
|---|---:|
| Accuracy | 0.9083 |
| Precision | 0.7226 |
| Recall | 0.7444 |
| F1 | 0.7334 |
| ROC-AUC | 0.9462 |
| PR-AUC | 0.8302 |

Fair Transformer baseline: F1 **0.6319**, ROC-AUC **0.9168**.

---

## Paper

- LaTeX: `Paper/IEEE_cardiac_progression_paper.tex`
- BibTeX: `Paper/references.bib` (~26 real references)
- Figures: `Paper/figures/`

Compile (TeX Live / MiKTeX):

```bash
cd Paper
pdflatex IEEE_cardiac_progression_paper.tex
bibtex IEEE_cardiac_progression_paper
pdflatex IEEE_cardiac_progression_paper.tex
pdflatex IEEE_cardiac_progression_paper.tex
```

Presentation (literature survey source): use the root file  
`Updated_Longitudinal_Temporal_Disease_Progression.pptx`  
(do **not** use the broken copies under `Paper/`).

---

## Project Architecture

```text
MIMIC-IV 2.1
  → preprocessing (cohort, CSI labels, 6×59 sequences)
  → sequence models  OR  temporal feature engineering + GBM
  → evaluation (results_*.json, plots)
  → inference API (backend/)
  → frontend dashboard (frontend/)
```

| Path | Role |
|---|---|
| `backend/` | FastAPI, inference, presets, schemas |
| `frontend/` | CardioPulse UI (+ local Chart.js vendor) |
| `src/preprocessing/` | MIMIC cohort construction |
| `src/model_training/` | Training scripts |
| `src/reporting/` | Plots / evaluation |
| `models/deployable/` | Inference bundle |
| `Paper/` | IEEE paper + figures |
| `docs/` | Dataset / architecture / results notes |

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `ModuleNotFoundError: fastapi` | Use `py -3.14` (or install into the Python you launch with) |
| Port 8000 in use | `run_app.py` auto-falls back to the next free port |
| Charts blank | Chart.js is local at `frontend/vendor/chart.umd.min.js` (offline OK) |
| Prediction fails | Ensure server started; check `/api/status` |
| Want published metrics | Use **Model Benchmarks** tab / `results_improved.json` — not the live demo probability alone |
| Missing MIMIC data | Expected for demos; raw data cannot be redistributed |
| Paper placeholders | Removed; recompile from `Paper/` |

---

## Limitations (honest)

- Sample-level (not patient-disjoint) splits may inflate generalization estimates
- CSI is an operational proxy label, not clinician-adjudicated progression
- Single-center retrospective MIMIC-IV study
- Demo scenarios are synthetic; do not treat live demo probabilities as clinical advice
