# Review Demo Script (2–3 minutes)

## Before the panel enters (30 seconds)

```bash
py -3.14 run_app.py
```

Confirm browser opens to `http://127.0.0.1:8000` and the orange **Demo Mode** banner is visible.

---

## Spoken flow (~2.5 minutes)

**1. Problem (20s)**  
“Cardiovascular deterioration is longitudinal. We predict whether a cardiac ICU patient’s condition is *worsening* versus *stable/improving* using six successive ICU visits and 59 clinical features from MIMIC-IV 2.1.”

**2. Data & pipeline (25s)**  
“We build a cardiac cohort from ICD codes, compute a Cardiac Severity Index per stay, label worsening when CSI rises more than 10%, form sequences of length six, and evaluate on a held-out test set of 5,289 samples with about 16.9% positives.”

**3. Live demo — stable case (40s)**  
- Select **Stable Post-Op Recovery**  
- Point at the trajectory chart (HR, SBP, SpO₂, BNP)  
- Click **Run Temporal Analysis**  
- Say: “Probability is low; classification is stable. Features and explanations come from the local inference pipeline—not hardcoded outcomes.”

**4. Live demo — worsening case (40s)**  
- Select **Acute Decompensated Heart Failure** (or cardiogenic shock)  
- Run analysis again  
- Point to rising risk, contributors (SpO₂ / BNP / BP trends)

**5. Research results (40s)**  
- Open **Model Benchmarks**  
- “These numbers are from the saved MIMIC-IV experiments: soft-vote GBM F1 **0.7334**, ROC-AUC **0.9462**, beating the fair Transformer baseline F1 **0.6319** / AUC **0.9168**. The banner separates live demo inference from published held-out metrics.”

**6. Contribution + honesty (20s)**  
“Contribution: for short ICU sequences, explicit temporal descriptors plus calibrated gradient boosting outperformed heavier sequence models in this internal study. Limitations: sample-level splits, proxy CSI label, single-center retrospective data—not clinically deployable.”

---

## If something fails

| Failure | Recovery |
|---|---|
| Wrong Python / missing fastapi | `py -3.14 -m pip install -r requirements.txt` then relaunch |
| Port busy | App auto-switches port; use the printed URL |
| Charts blank | Should work offline via `frontend/vendor/chart.umd.min.js` |
| Prediction error | Hard refresh; retry scenario; check `/api/status` |
| Faculty asks “is this the 0.9462 model?” | “Published 0.9462 is the soft-vote ensemble on MIMIC test set (Benchmarks). Live demo uses the local HistGBM bundle on synthetic scenarios for a safe, repeatable demo.” |
