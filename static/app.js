const state = {
  categories: [],
  selected: new Set(),
  searchQuery: "",
  activeCategory: "ALL",
};

const el = {
  themeToggle: document.getElementById("theme-toggle"),
  themeIcon: document.getElementById("theme-icon"),
  headerStats: document.getElementById("header-stats"),
  presetsContainer: document.getElementById("presets-container"),
  symptomSearch: document.getElementById("symptom-search"),
  categoryTabs: document.getElementById("category-tabs"),
  selectedCount: document.getElementById("selected-count"),
  trayQuality: document.getElementById("tray-quality"),
  selectedChips: document.getElementById("selected-chips"),
  symptomCategories: document.getElementById("symptom-categories"),
  predictBtn: document.getElementById("predict-btn"),
  clearBtn: document.getElementById("clear-btn"),
  emptyState: document.getElementById("empty-state"),
  analyzingState: document.getElementById("analyzing-state"),
  predictionOutput: document.getElementById("prediction-output"),
  topDisease: document.getElementById("top-disease"),
  topConfidence: document.getElementById("top-confidence"),
  topBand: document.getElementById("top-band"),
  gaugeFill: document.getElementById("gauge-fill"),
  topUrgency: document.getElementById("top-urgency"),
  topUrgencyText: document.getElementById("top-urgency-text"),
  topSpecialist: document.getElementById("top-specialist"),
  topDescription: document.getElementById("top-description"),
  matchedCore: document.getElementById("matched-core"),
  matchedSecondary: document.getElementById("matched-secondary"),
  precautionsList: document.getElementById("precautions-list"),
  probabilitiesList: document.getElementById("probabilities-list"),
  contributionsList: document.getElementById("contributions-list"),
  mAcc: document.getElementById("m-acc"),
  mCv: document.getElementById("m-cv"),
  mOob: document.getElementById("m-oob"),
  mF1: document.getElementById("m-f1"),
  globalImportances: document.getElementById("global-importances"),
};

const GAUGE_CIRCUMFERENCE = 2 * Math.PI * 50;

function animateNumber(targetEl, endValue, suffix = "%", decimals = 1, duration = 650) {
  const startTime = performance.now();
  function tick(now) {
    const progress = Math.min(1, (now - startTime) / duration);
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = endValue * eased;
    targetEl.textContent = `${current.toFixed(decimals)}${suffix}`;
    if (progress < 1) {
      requestAnimationFrame(tick);
    }
  }
  requestAnimationFrame(tick);
}

function initTheme() {
  const saved = localStorage.getItem("mediforest-theme") || "dark";
  document.documentElement.setAttribute("data-theme", saved);
  el.themeIcon.textContent = saved === "dark" ? "☀" : "☾";

  el.themeToggle.addEventListener("click", () => {
    const current = document.documentElement.getAttribute("data-theme") || "dark";
    const next = current === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("mediforest-theme", next);
    el.themeIcon.textContent = next === "dark" ? "☀" : "☾";
  });
}

function renderCategoryTabs() {
  el.categoryTabs.innerHTML = "";
  const tabs = [{ key: "ALL", label: "All Systems" }].concat(
    state.categories.map((c) => ({ key: c.category, label: c.category }))
  );

  for (const tab of tabs) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `cat-tab ${state.activeCategory === tab.key ? "active" : ""}`;
    btn.textContent = tab.label;
    btn.addEventListener("click", () => {
      state.activeCategory = tab.key;
      renderCategoryTabs();
      renderCategories();
    });
    el.categoryTabs.appendChild(btn);
  }
}

function renderSelected() {
  const count = state.selected.size;
  el.selectedCount.textContent = String(count);
  el.predictBtn.disabled = count === 0;

  if (count === 0) {
    el.trayQuality.textContent = "Select at least 2 symptoms";
  } else if (count === 1) {
    el.trayQuality.textContent = "Add 1–2 more for sharper triage";
  } else {
    el.trayQuality.textContent = "Ready for RF ensemble";
  }

  if (count === 0) {
    el.selectedChips.innerHTML =
      '<span class="muted-hint">Choose symptoms below to assemble the patient feature vector.</span>';
    return;
  }

  const labelMap = new Map();
  state.categories.forEach((cat) =>
    cat.symptoms.forEach((s) => labelMap.set(s.key, s.label))
  );

  el.selectedChips.innerHTML = "";
  for (const key of state.selected) {
    const chip = document.createElement("span");
    chip.className = "selected-chip";
    chip.innerHTML = `<span>${labelMap.get(key) || key}</span>`;
    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.setAttribute("aria-label", `Remove ${key}`);
    removeBtn.textContent = "×";
    removeBtn.addEventListener("click", () => {
      state.selected.delete(key);
      renderSelected();
      renderCategories();
    });
    chip.appendChild(removeBtn);
    el.selectedChips.appendChild(chip);
  }
}

function renderCategories() {
  const query = state.searchQuery.trim().toLowerCase();
  el.symptomCategories.innerHTML = "";

  for (const group of state.categories) {
    if (state.activeCategory !== "ALL" && group.category !== state.activeCategory) {
      continue;
    }

    const filtered = group.symptoms.filter(
      (s) =>
        !query ||
        s.label.toLowerCase().includes(query) ||
        s.key.toLowerCase().includes(query)
    );
    if (filtered.length === 0) continue;

    const block = document.createElement("div");
    block.className = "category-block";

    const heading = document.createElement("h3");
    heading.textContent = group.category;
    block.appendChild(heading);

    const grid = document.createElement("div");
    grid.className = "symptom-grid";

    for (const sym of filtered) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = `symptom-pill ${state.selected.has(sym.key) ? "active" : ""}`;
      btn.textContent = sym.label;
      btn.addEventListener("click", () => {
        if (state.selected.has(sym.key)) {
          state.selected.delete(sym.key);
        } else {
          state.selected.add(sym.key);
        }
        renderSelected();
        renderCategories();
      });
      grid.appendChild(btn);
    }

    block.appendChild(grid);
    el.symptomCategories.appendChild(block);
  }
}

function renderMatchedTags(container, items, fallbackText) {
  container.innerHTML = "";
  if (!items || items.length === 0) {
    container.innerHTML = `<span class="muted-hint">${fallbackText}</span>`;
    return;
  }
  for (const label of items) {
    const tag = document.createElement("span");
    tag.className = "mini-tag";
    tag.textContent = label;
    container.appendChild(tag);
  }
}

function renderPrediction(data) {
  const top = data.top_prediction;
  el.emptyState.classList.add("hidden");
  el.analyzingState.classList.add("hidden");
  el.predictionOutput.classList.remove("hidden");
  el.predictionOutput.classList.remove("prediction-reveal");
  void el.predictionOutput.offsetWidth;
  el.predictionOutput.classList.add("prediction-reveal");

  el.topDisease.textContent = top.disease;
  el.topBand.textContent = top.confidence_band;

  // Reset and animate circular SVG gauge
  el.gaugeFill.style.strokeDashoffset = String(GAUGE_CIRCUMFERENCE);
  requestAnimationFrame(() => {
    const pct = Math.max(0, Math.min(100, top.confidence_pct));
    const targetOffset = GAUGE_CIRCUMFERENCE * (1 - pct / 100);
    el.gaugeFill.style.strokeDashoffset = String(targetOffset);
  });
  animateNumber(el.topConfidence, top.confidence_pct, "%", 1, 800);

  const urgencyLower = (top.urgency || "Moderate").toLowerCase();
  el.topUrgency.className = `meta-tag urgency-${urgencyLower}`;
  el.topUrgencyText.textContent = `Urgency: ${top.urgency}`;
  el.topSpecialist.textContent = `Specialist: ${top.specialist}`;
  el.topDescription.textContent = top.description;

  renderMatchedTags(el.matchedCore, top.matched_core_symptoms, "None directly matched");
  renderMatchedTags(el.matchedSecondary, top.matched_secondary_symptoms, "None");

  el.precautionsList.innerHTML = "";
  top.precautions.forEach((step, idx) => {
    const li = document.createElement("li");
    li.style.setProperty("--item-delay", `${idx * 0.06}s`);
    li.textContent = step;
    el.precautionsList.appendChild(li);
  });

  el.probabilitiesList.innerHTML = "";
  data.predictions.forEach((pred, idx) => {
    const row = document.createElement("div");
    row.className = "prob-row";
    row.style.setProperty("--row-delay", `${idx * 0.05}s`);
    row.innerHTML = `
      <span><strong>${pred.disease}</strong> &middot; <span class="muted-hint">${pred.urgency}</span></span>
      <span><strong>${pred.confidence_pct.toFixed(1)}%</strong></span>
      <div class="prob-bar-track">
        <div class="prob-bar-fill"></div>
      </div>
    `;
    el.probabilitiesList.appendChild(row);
    const fill = row.querySelector(".prob-bar-fill");
    requestAnimationFrame(() => {
      setTimeout(() => {
        fill.style.width = `${Math.max(2, pred.confidence_pct)}%`;
      }, 40 + idx * 65);
    });
  });

  el.contributionsList.innerHTML = "";
  for (const item of data.symptom_contributions) {
    const pill = document.createElement("span");
    pill.className = "contrib-pill";
    pill.textContent = `${item.label}: ${(item.global_importance * 100).toFixed(2)}%`;
    el.contributionsList.appendChild(pill);
  }
}

async function runPrediction() {
  if (state.selected.size === 0) return;
  el.predictBtn.disabled = true;
  el.emptyState.classList.add("hidden");
  el.predictionOutput.classList.add("hidden");
  el.analyzingState.classList.remove("hidden");

  try {
    const minScanDelay = new Promise((resolve) => setTimeout(resolve, 460));
    const fetchPromise = fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        symptoms: Array.from(state.selected),
        top_k: 5,
      }),
    });

    const [res] = await Promise.all([fetchPromise, minScanDelay]);
    const data = await res.json();
    if (!res.ok) {
      el.analyzingState.classList.add("hidden");
      el.emptyState.classList.remove("hidden");
      alert(data.error || "Prediction failed.");
      return;
    }
    renderPrediction(data);
  } catch (err) {
    el.analyzingState.classList.add("hidden");
    el.emptyState.classList.remove("hidden");
    alert(`Network error: ${err.message}`);
  } finally {
    el.predictBtn.disabled = state.selected.size === 0;
  }
}

async function init() {
  initTheme();

  const [symRes, metricsRes] = await Promise.all([
    fetch("/api/symptoms"),
    fetch("/api/metrics"),
  ]);

  const symData = await symRes.json();
  const metricsData = await metricsRes.json();

  state.categories = symData.categories || [];
  renderCategoryTabs();
  renderCategories();
  renderSelected();

  el.presetsContainer.innerHTML = "";
  for (const preset of symData.presets || []) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "preset-btn";
    btn.textContent = preset.name;
    btn.addEventListener("click", () => {
      state.selected = new Set(preset.symptoms);
      renderSelected();
      renderCategories();
      runPrediction();
    });
    el.presetsContainer.appendChild(btn);
  }

  const evalData = metricsData.evaluation || {};
  const dsData = metricsData.dataset || {};
  const hpData = metricsData.hyperparameters || {};

  el.headerStats.innerHTML = `
    <span class="stat-pill">${hpData.n_estimators || 300} Trees</span>
    <span class="stat-pill">${dsData.num_diseases || 0} Diseases</span>
    <span class="stat-pill">${dsData.num_features || 0} Symptoms</span>
    <span class="stat-pill">CV ${((evalData.cv_mean_accuracy || 0) * 100).toFixed(1)}%</span>
  `;

  animateNumber(el.mAcc, (evalData.test_accuracy || 0) * 100, "%", 1, 700);
  animateNumber(el.mCv, (evalData.cv_mean_accuracy || 0) * 100, "%", 1, 780);
  animateNumber(el.mOob, (evalData.production_oob_score || 0) * 100, "%", 1, 840);
  animateNumber(el.mF1, (evalData.test_f1_weighted || 0) * 100, "%", 1, 900);

  const topFeatures = (metricsData.top_feature_importances || []).slice(0, 8);
  const maxImp = Math.max(...topFeatures.map((f) => f.importance || 0), 0.01);

  el.globalImportances.innerHTML = "";
  topFeatures.forEach((feat, idx) => {
    const row = document.createElement("div");
    row.className = "imp-bar-row";
    const pctText = `${(feat.importance * 100).toFixed(2)}%`;
    const widthPct = Math.round((feat.importance / maxImp) * 100);
    row.innerHTML = `
      <span class="imp-name" title="${feat.label}">${feat.label}</span>
      <div class="imp-track"><div class="imp-fill"></div></div>
      <span class="imp-val">${pctText}</span>
    `;
    el.globalImportances.appendChild(row);
    const fill = row.querySelector(".imp-fill");
    requestAnimationFrame(() => {
      setTimeout(() => {
        fill.style.width = `${widthPct}%`;
      }, 100 + idx * 70);
    });
  });
}

el.symptomSearch.addEventListener("input", (e) => {
  state.searchQuery = e.target.value;
  renderCategories();
});

el.clearBtn.addEventListener("click", () => {
  state.selected.clear();
  renderSelected();
  renderCategories();
  el.predictionOutput.classList.add("hidden");
  el.analyzingState.classList.add("hidden");
  el.emptyState.classList.remove("hidden");
});

el.predictBtn.addEventListener("click", runPrediction);

init().catch((err) => {
  el.headerStats.innerHTML = `<span class="stat-pill">Error loading model: ${err.message}</span>`;
});
