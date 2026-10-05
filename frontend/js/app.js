/**
 * CardioPulse AI — Longitudinal Cardiac Progression Intelligence
 * Frontend Application Controller
 */

// ── Application State ────────────────────────────────────────────────────────
const state = {
  status: null,
  features: [],
  scenarios: [],
  activeScenarioId: null,
  patientData: {
    patient_id: "PT-1049",
    patient_name: "Anonymous Patient",
    threshold: 0.335,
    visits: []
  },
  activeVisitIndex: 5, // 0 to 5 (6th visit by default)
  trajectoryChart: null,
  lastPrediction: null
};

// ── Core Initializer ─────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", async () => {
  setupNavigation();
  await loadSystemStatus();
  await loadFeatures();
  await loadScenarios();
  await loadBenchmarks();
  initBatchHandlers();
});

// ── Navigation Tabs ──────────────────────────────────────────────────────────
function setupNavigation() {
  const tabs = document.querySelectorAll(".nav-tab");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      const target = tab.dataset.tab;
      document.querySelectorAll(".nav-tab").forEach(t => t.classList.remove("active"));
      document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));
      tab.classList.add("active");
      const targetPane = document.getElementById(target);
      if (targetPane) targetPane.classList.add("active");
    });
  });
}

// ── API Loaders ──────────────────────────────────────────────────────────────
async function loadSystemStatus() {
  try {
    const res = await fetch("/api/status");
    if (!res.ok) throw new Error("Status API offline");
    state.status = await res.json();
    const statusText = document.getElementById("headerStatusText");
    if (statusText) {
      statusText.textContent = `Model: ${state.status.active_model} | ROC-AUC: ${state.status.best_benchmark_auc}`;
    }
  } catch (err) {
    console.warn("Could not fetch status:", err);
  }
}

async function loadFeatures() {
  try {
    const res = await fetch("/api/features");
    state.features = await res.json();
    renderFeatureDictionary(state.features);
  } catch (err) {
    console.error("Failed to load features:", err);
  }
}

async function loadScenarios() {
  try {
    const res = await fetch("/api/scenarios");
    state.scenarios = await res.json();
    const select = document.getElementById("scenarioSelect");
    if (select) {
      select.innerHTML = state.scenarios.map((s, idx) => 
        `<option value="${idx}">[${s.id}] ${s.name} (${s.expected_outcome})</option>`
      ).join("");

      select.addEventListener("change", (e) => {
        applyScenario(parseInt(e.target.value, 10));
      });
    }

    if (state.scenarios.length > 0) {
      applyScenario(0);
    }
  } catch (err) {
    console.error("Failed to load scenarios:", err);
  }
}

// ── Scenario & Patient Management ────────────────────────────────────────────
function applyScenario(index) {
  const scenario = state.scenarios[index];
  if (!scenario) return;

  state.activeScenarioId = scenario.id;
  state.patientData = JSON.parse(JSON.stringify(scenario.sample_data));
  state.activeVisitIndex = 5; // Default to latest visit

  // Update UI headers
  const narrative = document.getElementById("scenarioNarrative");
  if (narrative) narrative.textContent = scenario.clinical_narrative;

  const tag = document.getElementById("patientTag");
  if (tag) tag.textContent = `${state.patientData.patient_id} · ${state.patientData.patient_name}`;

  const threshInput = document.getElementById("thresholdSlider");
  const threshOut = document.getElementById("thresholdValue");
  if (threshInput && threshOut) {
    threshInput.value = state.patientData.threshold || 0.335;
    threshOut.textContent = `${(threshInput.value * 100).toFixed(1)}%`;
  }

  renderVisitTimeline();
  renderFormInputs();
  renderTrajectoryChart();
  runPrediction();
}

function renderVisitTimeline() {
  const container = document.getElementById("visitTimeline");
  if (!container) return;

  container.innerHTML = [0, 1, 2, 3, 4, 5].map(idx => `
    <button class="visit-pill ${idx === state.activeVisitIndex ? 'active' : ''}" data-idx="${idx}">
      Visit ${idx + 1} ${idx === 5 ? '(Latest)' : ''}
    </button>
  `).join("");

  container.querySelectorAll(".visit-pill").forEach(btn => {
    btn.addEventListener("click", () => {
      state.activeVisitIndex = parseInt(btn.dataset.idx, 10);
      renderVisitTimeline();
      renderFormInputs();
    });
  });
}

// ── Input Controls Rendering ─────────────────────────────────────────────────
function renderFormInputs() {
  const container = document.getElementById("inputsContainer");
  if (!container || !state.patientData.visits || !state.patientData.visits[state.activeVisitIndex]) return;

  const currentVisit = state.patientData.visits[state.activeVisitIndex];

  // Group features by category
  const categories = {
    "Vital Signs": ["heart_rate_mean", "sbp_mean", "dbp_mean", "mbp_mean", "resp_rate_mean", "temperature_mean", "spo2_mean", "glucose_fingerstick_mean"],
    "Cardiac Biomarkers": ["troponin_t_mean", "troponin_t_max", "troponin_i_mean", "bnp_mean", "bnp_max", "nt_probnp_mean"],
    "Renal & Metabolic": ["creatinine_mean", "creatinine_max", "bun_mean", "bun_max", "glucose_lab_mean"],
    "Electrolytes": ["potassium_mean", "sodium_mean", "chloride_mean", "bicarbonate_mean"],
    "Hematology (CBC)": ["hematocrit_mean", "hemoglobin_mean", "wbc_mean", "wbc_max", "platelets_mean"],
    "Clinical Context": ["age", "los_days", "prior_mi", "prior_hf", "num_medications"]
  };

  let html = "";
  for (const [catName, featIds] of Object.entries(categories)) {
    html += `
      <div class="form-category">
        <div class="form-category-header">
          <span>${catName} (Visit ${state.activeVisitIndex + 1})</span>
          <small style="color:#64748b;">${featIds.length} Key Parameters</small>
        </div>
        <div class="form-grid">
    `;

    for (const fid of featIds) {
      const meta = state.features.find(f => f.id === fid) || {
        name: fid, unit: "", normal_min: 0, normal_max: 100
      };
      const val = currentVisit[fid] !== undefined ? currentVisit[fid] : meta.default_val;

      html += `
        <div class="form-group">
          <label for="input_${fid}">
            <span>${meta.name}</span>
            <span class="unit-tag">${meta.unit || ''}</span>
          </label>
          <input 
            type="number" 
            id="input_${fid}" 
            data-feat="${fid}" 
            step="${fid.includes('troponin') || fid.includes('creat') ? '0.01' : fid.includes('wbc') ? '0.1' : '1'}"
            value="${val}"
          />
        </div>
      `;
    }

    html += `</div></div>`;
  }

  container.innerHTML = html;

  // Add listeners
  container.querySelectorAll("input").forEach(input => {
    input.addEventListener("input", (e) => {
      const fid = e.target.dataset.feat;
      const numVal = parseFloat(e.target.value);
      if (!isNaN(numVal)) {
        state.patientData.visits[state.activeVisitIndex][fid] = numVal;
        updateTrajectoryChart();
      }
    });
  });
}

// ── Chart.js Trajectory Rendering ────────────────────────────────────────────
function renderTrajectoryChart() {
  const canvas = document.getElementById("trajectoryCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  const visits = state.patientData.visits;
  const labels = ["Visit 1", "Visit 2", "Visit 3", "Visit 4", "Visit 5", "Visit 6"];

  const hrData = visits.map(v => v.heart_rate_mean);
  const sbpData = visits.map(v => v.sbp_mean);
  const spo2Data = visits.map(v => v.spo2_mean);
  const bnpData = visits.map(v => v.bnp_mean);
  const tropData = visits.map(v => (v.troponin_t_mean || 0.02) * 500); // Scaled for visibility

  if (state.trajectoryChart) {
    state.trajectoryChart.destroy();
  }

  state.trajectoryChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Heart Rate (bpm)",
          data: hrData,
          borderColor: "#e11d48",
          backgroundColor: "rgba(225, 29, 72, 0.05)",
          yAxisID: "yVitals",
          tension: 0.3,
          borderWidth: 2.5
        },
        {
          label: "Systolic BP (mmHg)",
          data: sbpData,
          borderColor: "#7c3aed",
          backgroundColor: "rgba(124, 58, 237, 0.05)",
          yAxisID: "yVitals",
          tension: 0.3,
          borderWidth: 2.5
        },
        {
          label: "SpO₂ Saturation (%)",
          data: spo2Data,
          borderColor: "#2563eb",
          backgroundColor: "rgba(37, 99, 235, 0.05)",
          yAxisID: "yVitals",
          tension: 0.3,
          borderWidth: 2.5
        },
        {
          label: "BNP (pg/mL)",
          data: bnpData,
          borderColor: "#d97706",
          yAxisID: "yBNP",
          borderDash: [5, 5],
          tension: 0.3,
          borderWidth: 2
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: "index",
        intersect: false
      },
      plugins: {
        legend: {
          position: "top",
          labels: { boxWidth: 12, font: { size: 12, weight: "bold" } }
        },
        tooltip: {
          padding: 10
        }
      },
      scales: {
        x: {
          grid: { color: "#f1f5f9" }
        },
        yVitals: {
          type: "linear",
          position: "left",
          min: 60,
          max: 160,
          title: { display: true, text: "Vitals (bpm / mmHg / %)" },
          grid: { color: "#f1f5f9" }
        },
        yBNP: {
          type: "linear",
          position: "right",
          title: { display: true, text: "BNP (pg/mL)" },
          grid: { display: false }
        }
      }
    }
  });
}

function updateTrajectoryChart() {
  if (!state.trajectoryChart) return;
  const visits = state.patientData.visits;
  state.trajectoryChart.data.datasets[0].data = visits.map(v => v.heart_rate_mean);
  state.trajectoryChart.data.datasets[1].data = visits.map(v => v.sbp_mean);
  state.trajectoryChart.data.datasets[2].data = visits.map(v => v.spo2_mean);
  state.trajectoryChart.data.datasets[3].data = visits.map(v => v.bnp_mean);
  state.trajectoryChart.update();
}

// ── Predict Action ───────────────────────────────────────────────────────────
async function runPrediction() {
  const predictBtn = document.getElementById("predictBtn");
  if (predictBtn) {
    predictBtn.disabled = true;
    predictBtn.innerHTML = `<span>Computing Temporal Signals...</span>`;
  }

  // Update threshold from slider if present
  const threshSlider = document.getElementById("thresholdSlider");
  if (threshSlider) {
    state.patientData.threshold = parseFloat(threshSlider.value);
  }

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(state.patientData)
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Prediction request failed");
    }

    const data = await res.json();
    state.lastPrediction = data;
    renderPredictionResults(data);
  } catch (err) {
    alert("Prediction error: " + err.message);
  } finally {
    if (predictBtn) {
      predictBtn.disabled = false;
      predictBtn.innerHTML = `<span>Run Temporal Analysis →</span>`;
    }
  }
}

function renderPredictionResults(data) {
  // 1. Gauge needle: map probability 0.0 -> -90deg, 1.0 -> +90deg
  const needle = document.getElementById("gaugeNeedle");
  if (needle) {
    const deg = -90 + data.probability * 180;
    needle.style.transform = `rotate(${deg}deg)`;
  }

  // 2. Probability Text
  const probElem = document.getElementById("probPercent");
  if (probElem) {
    probElem.textContent = data.probability_percent;
  }

  // 3. Classification Badge
  const badge = document.getElementById("classificationBadge");
  if (badge) {
    badge.textContent = data.classification;
    badge.className = `classification-badge badge-${data.risk_level.toLowerCase()}`;
  }

  // 4. Threshold & CI
  const threshElem = document.getElementById("thresholdDisplay");
  if (threshElem) {
    threshElem.innerHTML = `
      Operating threshold: <strong>${(data.decision_threshold * 100).toFixed(1)}%</strong> | 
      95% CI: <strong>${(data.confidence_interval[0] * 100).toFixed(1)}% – ${(data.confidence_interval[1] * 100).toFixed(1)}%</strong>
    `;
  }

  // 5. Contributors List
  const contContainer = document.getElementById("contributorsList");
  if (contContainer) {
    if (!data.top_contributors || data.top_contributors.length === 0) {
      contContainer.innerHTML = `<p style="font-size:13px; color:#64748b;">No prominent deteriorating signals detected.</p>`;
    } else {
      contContainer.innerHTML = data.top_contributors.map(c => `
        <div class="contributor-item ${c.direction}">
          <div class="contributor-header">
            <span>${c.feature_name}</span>
            <span style="color:${c.direction === 'risk_elevating' ? '#dc2626' : '#16a34a'}">
              ${c.direction === 'risk_elevating' ? '+' : ''}${c.impact_score}%
            </span>
          </div>
          <div class="contributor-desc">${c.temporal_descriptor}</div>
          <div class="contributor-bar-track">
            <div class="contributor-bar-fill ${c.direction}" style="width: ${c.impact_score}%"></div>
          </div>
          <div style="font-size:11px; color:#64748b; margin-top:4px;">${c.explanation}</div>
        </div>
      `).join("");
    }
  }

  // 6. Clinical Recommendations
  const recContainer = document.getElementById("recommendationsList");
  if (recContainer) {
    if (!data.recommendations || data.recommendations.length === 0) {
      recContainer.innerHTML = `<p style="font-size:13px; color:#64748b;">Standard observation recommended.</p>`;
    } else {
      recContainer.innerHTML = data.recommendations.map(r => `
        <div class="recommendation-card rec-${r.urgency}">
          <div class="rec-title">
            <span>● [${r.urgency}] ${r.title}</span>
          </div>
          <div>${r.rationale}</div>
          <ul class="rec-actions">
            ${r.suggested_actions.map(act => `<li>${act}</li>`).join("")}
          </ul>
        </div>
      `).join("");
    }
  }
}

// ── Batch Processing ─────────────────────────────────────────────────────────
function initBatchHandlers() {
  const dropzone = document.getElementById("batchDropzone");
  const fileInput = document.getElementById("batchFileInput");
  const runBtn = document.getElementById("runBatchBtn");
  let selectedFile = null;

  if (dropzone && fileInput) {
    dropzone.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", (e) => {
      if (e.target.files.length > 0) {
        selectedFile = e.target.files[0];
        dropzone.querySelector("p").textContent = `Selected: ${selectedFile.name} (${(selectedFile.size / 1024).toFixed(1)} KB)`;
        if (runBtn) runBtn.disabled = false;
      }
    });

    dropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropzone.style.borderColor = "#2563eb";
    });

    dropzone.addEventListener("dragleave", () => {
      dropzone.style.borderColor = "#cbd5e1";
    });

    dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropzone.style.borderColor = "#cbd5e1";
      if (e.dataTransfer.files.length > 0) {
        selectedFile = e.dataTransfer.files[0];
        dropzone.querySelector("p").textContent = `Selected: ${selectedFile.name} (${(selectedFile.size / 1024).toFixed(1)} KB)`;
        if (runBtn) runBtn.disabled = false;
      }
    });
  }

  if (runBtn) {
    runBtn.addEventListener("click", async () => {
      if (!selectedFile) return;
      runBtn.disabled = true;
      runBtn.textContent = "Processing Cohort...";

      const formData = new FormData();
      formData.append("file", selectedFile);

      try {
        const res = await fetch("/api/predict/csv", {
          method: "POST",
          body: formData
        });

        if (!res.ok) throw new Error("Batch processing failed");
        const batchData = await res.json();
        renderBatchResults(batchData);
      } catch (err) {
        alert("Batch error: " + err.message);
      } finally {
        runBtn.disabled = false;
        runBtn.textContent = "Run Batch Inference";
      }
    });
  }
}

function renderBatchResults(data) {
  const resultsCard = document.getElementById("batchResultsCard");
  if (!resultsCard) return;
  resultsCard.style.display = "block";

  document.getElementById("batchTotal").textContent = data.total_processed;
  document.getElementById("batchWorsening").textContent = `${data.worsening_count} (${(data.worsening_rate * 100).toFixed(1)}%)`;
  document.getElementById("batchStable").textContent = data.stable_count;

  const tbody = document.getElementById("batchTableBody");
  if (tbody) {
    tbody.innerHTML = data.results.map(r => `
      <tr>
        <td><strong>${r.patient_id}</strong></td>
        <td>${(r.probability * 100).toFixed(1)}%</td>
        <td><span class="classification-badge badge-${r.risk_level.toLowerCase()}">${r.risk_level}</span></td>
        <td>${r.classification}</td>
        <td>${r.primary_risk_driver}</td>
      </tr>
    `).join("");
  }
}

// ── Model Benchmarks ─────────────────────────────────────────────────────────
async function loadBenchmarks() {
  try {
    const res = await fetch("/api/benchmarks");
    const data = await res.json();

    const tbody = document.getElementById("benchmarkTableBody");
    if (tbody && data.models) {
      tbody.innerHTML = data.models.map(m => `
        <tr class="${m.is_best ? 'highlight' : ''}">
          <td>
            <strong>${m.name}</strong>
            ${m.is_best ? '<span class="benchmark-badge">Best Model</span>' : ''}
            <div style="font-size:11px; color:#64748b;">${m.type}</div>
          </td>
          <td>${(m.accuracy * 100).toFixed(2)}%</td>
          <td>${(m.precision * 100).toFixed(2)}%</td>
          <td>${(m.recall * 100).toFixed(2)}%</td>
          <td><strong>${m.f1.toFixed(4)}</strong></td>
          <td><strong style="color:#2563eb;">${m.auc_roc.toFixed(4)}</strong></td>
          <td>${m.pr_auc.toFixed(4)}</td>
        </tr>
      `).join("");
    }

    const gallery = document.getElementById("benchmarkGallery");
    if (gallery && data.figures) {
      gallery.innerHTML = data.figures.map(f => `
        <div class="figure-card">
          <img src="/api/results/plots/${f.filename}" alt="${f.title}" loading="lazy" />
          <h4>${f.title}</h4>
        </div>
      `).join("");
    }
  } catch (err) {
    console.error("Failed to load benchmarks:", err);
  }
}

// ── Feature Dictionary ───────────────────────────────────────────────────────
function renderFeatureDictionary(features) {
  const tbody = document.getElementById("dictTableBody");
  if (!tbody) return;

  const renderRows = (list) => {
    tbody.innerHTML = list.map(f => `
      <tr>
        <td>
          <strong>${f.name}</strong>
          <div style="font-size:11px; color:#64748b; font-family:monospace;">${f.id}</div>
        </td>
        <td><span style="font-size:12px; font-weight:600; color:#1e40af;">${f.category}</span></td>
        <td>${f.unit || '—'}</td>
        <td>${f.normal_min !== null ? `${f.normal_min} – ${f.normal_max}` : '—'}</td>
        <td style="font-size:12px; color:#475569;">${f.description}</td>
      </tr>
    `).join("");
  };

  renderRows(features);

  const searchInput = document.getElementById("dictSearch");
  const catFilter = document.getElementById("dictCategoryFilter");

  const filter = () => {
    const q = (searchInput?.value || "").toLowerCase();
    const cat = catFilter?.value || "";

    const filtered = features.filter(f => {
      const matchQ = f.name.toLowerCase().includes(q) || f.id.toLowerCase().includes(q) || f.description.toLowerCase().includes(q);
      const matchCat = !cat || f.category === cat;
      return matchQ && matchCat;
    });

    renderRows(filtered);
  };

  if (searchInput) searchInput.addEventListener("input", filter);
  if (catFilter) catFilter.addEventListener("change", filter);
}
