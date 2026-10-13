const state = {
  modelReady: false,
  clinicalFields: [],
  dicomReady: false,
  dicomBusy: false,
  explanationController: null,
  maxUploadBytes: 0,
  maxDicomFiles: 0,
  uploads: {
    Muscle: null,
    Radiomics: null,
  },
};

// Reported by mm_worker.py as it reaches each step, so the label always
// reflects what the server is actually doing.
const MUSCLE_STAGE_LABELS = {
  reading: "Reading the NIfTI image…",
  segmenting: "Running MuscleMap segmentation…",
  segment_retry: "Retrying segmentation on a duplicated slice…",
  previewing: "Saving the segmentation preview…",
  squeezing: "Matching image and mask geometry…",
  extracting: "Extracting GMM tissue metrics…",
  fallback: "MuscleMap CLI failed — recomputing metrics in process…",
  no_muscle_labels: "No muscle segmented in this series — it adds no slices…",
  aggregating: "Aggregating to the 16 model features…",
};

// Reported by totalseg_worker.py for the two MR segmentations and intersection,
// then by fae_worker.py, one event per PyRadiomics image filter.
const RADIOMICS_STAGE_LABELS = {
  preparing: "Preparing TotalSegmentator\u2026",
  weights: "Downloading TotalSegmentator weights (first run only)\u2026",
  body_segmentation: "Segmenting vertebral bodies with total_mr\u2026",
  level_segmentation: "Identifying vertebral levels with vertebrae_mr\u2026",
  vertebrae_segmentation: "Segmenting vertebral bodies (total_mr) and levels (vertebrae_mr) in parallel\u2026",
  intersecting: "Intersecting the body and selected-level masks\u2026",
  saving: "Saving the segmentation\u2026",
  segmented: "Segmentation done\u2026",
  reading: "Reading the MRI and VOI mask\u2026",
  extracting: "Running PyRadiomics\u2026",
  mapping: "Collecting feature values\u2026",
  selecting: "Selecting the model features\u2026",
};

const MUSCLE_ROWS = [
  ["ES_L", "Erector spinae, left"],
  ["ES_R", "Erector spinae, right"],
  ["MF_L", "Multifidus, left"],
  ["MF_R", "Multifidus, right"],
  ["PM_L", "Psoas major, left"],
  ["PM_R", "Psoas major, right"],
  ["QL_L", "Quadratus lumborum, left"],
  ["QL_R", "Quadratus lumborum, right"],
];

const elements = {
  globalAlert: document.getElementById("global-alert"),
  modelStatus: document.getElementById("model-status"),
  modelRoutes: document.getElementById("model-routes"),
  clinicalLoading: document.getElementById("clinical-loading"),
  clinicalFields: document.getElementById("clinical-fields"),
  predictionForm: document.getElementById("prediction-form"),
  predictButton: document.getElementById("predict-button"),
  resultSection: document.getElementById("result-section"),
};

document.addEventListener("DOMContentLoaded", async () => {
  try {
    bindEvents();
    await loadApplicationState();
    started = true;
  } catch (error) {
    // Without this the page would sit on its loading placeholders forever.
    reportStartupFailure(error);
  }
});

let started = false;

window.addEventListener("error", (event) => reportFailure(event.error));
window.addEventListener("unhandledrejection", (event) =>
  reportFailure(event.reason),
);

function reportStartupFailure(error) {
  const loading = document.getElementById("clinical-loading");
  if (loading) {
    loading.textContent =
      "The page failed to start. Reload with Ctrl+F5; if it persists, check the "
      + "browser console.";
  }
  const dot = document.getElementById("model-status-dot");
  if (dot) {
    dot.className = "status-dot status-error";
  }
  reportFailure(error);
}

function reportFailure(error) {
  const message = error?.message || String(error);
  showAlert(
    started
      ? `Something went wrong: ${message}`
      : `The page could not start: ${message}`,
    "danger",
  );
  console.error(error);
}

function bindEvents() {
  document.getElementById("dicom-process-button").addEventListener("click", processDicomFolder);
  document.getElementById("dicom-files").addEventListener("change", () => {
    resetDicomResults();
    renderDicomSelection();
    updateFormState();
  });
  document.getElementById("dicom-level").addEventListener("change", () => {
    resetDicomResults();
    updateFormState();
  });
  elements.predictionForm.addEventListener("submit", submitPrediction);
  elements.modelStatus.addEventListener("click", () => toggleModelRoutes());
  document.addEventListener("click", (event) => {
    if (!event.target.closest(".model-status-wrapper")) toggleModelRoutes(false);
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") toggleModelRoutes(false);
  });
}

async function loadApplicationState() {
  const healthResponse = await fetch("/api/health");
  const health = await parseJson(healthResponse);
  if (!healthResponse.ok) {
    setModelStatus(false, "Model unavailable");
    showAlert(health.message || health.error || "Model bundle is not loaded.", "danger");
    elements.clinicalLoading.textContent =
      "Clinical fields will become available after a valid model bundle is loaded.";
    return;
  }

  state.modelReady = true;
  const routes = health.model?.inference_routes || {};
  const routeCount = Object.keys(routes).length;
  setModelStatus(true, `Model ready · ${routeCount || 1} input routes`);
  renderModelRoutes(routes);

  const schemaResponse = await fetch("/api/schema/clinical");
  const schema = await parseJson(schemaResponse);
  if (!schemaResponse.ok) {
    showAlert(schema.message || "Cannot load Clinical schema.", "danger");
    return;
  }
  state.clinicalFields = schema.fields || [];
  renderClinicalFields(state.clinicalFields);

  const dicomResponse = await fetch("/api/schema/dicom");
  const dicom = await parseJson(dicomResponse);
  if (dicomResponse.ok) {
    state.dicomReady = true;
    state.maxUploadBytes = dicom.max_upload_bytes;
    state.maxDicomFiles = dicom.max_files;
    const select = document.getElementById("dicom-level");
    select.replaceChildren();
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "Select level…";
    placeholder.disabled = true;
    placeholder.selected = true;
    select.appendChild(placeholder);
    (dicom.levels || []).forEach((level) => {
      const option = document.createElement("option");
      option.value = level.value;
      option.textContent = level.label;
      select.appendChild(option);
    });
  } else {
    showAlert(dicom.message || "Cannot load DICOM processing options. Restart the server after updating the app.", "warning");
  }
  updateFormState();
}

function renderClinicalFields(fields) {
  elements.clinicalLoading.classList.add("d-none");
  elements.clinicalFields.replaceChildren();

  fields.forEach((field) => {
    const wrapper = document.createElement("div");
    wrapper.className = "clinical-field";

    const label = document.createElement("label");
    label.className = "form-label";
    label.htmlFor = field.name;
    label.textContent = field.label;

    let input;
    if (field.input_type === "select") {
      input = document.createElement("select");
      input.className = "form-select";
      const placeholder = document.createElement("option");
      placeholder.value = "";
      placeholder.textContent = "Select…";
      placeholder.selected = true;
      placeholder.disabled = true;
      input.appendChild(placeholder);
      (field.choices || []).forEach((choice) => {
        const option = document.createElement("option");
        const namedChoice = typeof choice === "object" && choice !== null;
        option.value = String(namedChoice ? choice.value : choice);
        option.textContent = String(namedChoice ? choice.label : choice);
        input.appendChild(option);
      });
    } else {
      input = document.createElement("input");
      input.type = "number";
      input.className = "form-control";
      input.step = field.step === undefined ? "any" : String(field.step);
      input.inputMode = field.value_type === "integer" ? "numeric" : "decimal";
      if (Number.isFinite(field.min)) input.min = String(field.min);
      if (Number.isFinite(field.max)) input.max = String(field.max);
    }
    input.id = field.name;
    input.name = field.name;
    input.required = true;
    input.dataset.valueType = field.value_type || "number";
    const onClinicalChange = () => {
      clearLateFusionExplanation();
      elements.resultSection.classList.add("d-none");
      updateFormState();
    };
    input.addEventListener("input", onClinicalChange);
    input.addEventListener("change", onClinicalChange);
    wrapper.append(label, input);

    const range = document.createElement("small");
    range.className = "field-range";
    range.id = `${field.name}-range`;
    if (
      (field.source_name === "AI_CCI" || field.name === "Clinical___AI_CCI") &&
      field.input_type === "number" &&
      Number.isFinite(field.min) &&
      Number.isFinite(field.max)
    ) {
      range.textContent = `Range: ${field.min}–${field.max}`;
      input.setAttribute("aria-describedby", range.id);
    } else {
      // Share the optional note row so the CCI range does not offset controls.
      range.setAttribute("aria-hidden", "true");
    }
    wrapper.appendChild(range);
    elements.clinicalFields.appendChild(wrapper);
  });
}

function resetDicomResults() {
  state.uploads.Muscle = null;
  state.uploads.Radiomics = null;
  renderMuscleFeatures(null);
  renderRadiomicsFeatures(null);
  renderMuscleSegmentationNote(null);
  renderSegmentationNote(null);
  elements.resultSection.classList.add("d-none");
  clearLateFusionExplanation();
  document.getElementById("dicom-status").classList.add("d-none");
  document.getElementById("dicom-warnings").replaceChildren();
  document.getElementById("dicom-warnings").classList.add("d-none");
  document.getElementById("dicom-discovery").classList.add("d-none");
  ["dicom-branch-status", "dicom-saved-series"].forEach((id) => {
    const panel = document.getElementById(id);
    panel.replaceChildren();
    panel.classList.add("d-none");
  });
}

function renderDicomWarnings(warnings) {
  const panel = document.getElementById("dicom-warnings");
  panel.replaceChildren();
  panel.className = "alert alert-warning mt-3" + (warnings.length ? "" : " d-none");
  const list = document.createElement("ul");
  list.className = "mb-0";
  warnings.forEach((message) => {
    const item = document.createElement("li");
    item.textContent = message;
    list.appendChild(item);
  });
  panel.appendChild(list);
}

async function processDicomFolder() {
  if (state.dicomBusy) return;
  const files = selectedDicomFiles();
  const level = document.getElementById("dicom-level").value;
  const status = document.getElementById("dicom-status");
  if (!files.length || !level) {
    setUploadStatus(status, "Select DICOM files and a lumbar level.", "error");
    return;
  }
  const bytes = files.reduce((total, file) => total + file.size, 0);
  if (files.length > state.maxDicomFiles || bytes > state.maxUploadBytes) {
    setUploadStatus(status, "This selection exceeds the upload limit of " +
      state.maxDicomFiles + " files or " + Math.round(state.maxUploadBytes / 1048576) + " MB.", "error");
    return;
  }
  resetDicomResults();
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file, file.name));
  formData.append("level", level);
  state.dicomBusy = true;
  updateFormState();
  const startedAt = Date.now();
  let elapsedTimer = null;
  let stageLabel = "Reading DICOM metadata…";
  let completed = null;
  let streamError = null;
  let extractionModalities = [];
  const branchStatus = {};
  const setBranchStatus = (modality, text, type = "pending") => {
    if (!modality) return;
    branchStatus[modality] = { text, type };
    renderDicomBranchStatus(branchStatus);
  };
  const showStage = () => setUploadProgress(status, {
    label: stageLabel,
    elapsedSeconds: Math.round((Date.now() - startedAt) / 1000),
  });
  setUploadProgress(status, { label: "Sending DICOM files…", percent: 0 });
  try {
    const response = await postWithProgress("/api/upload/dicom", formData, {
      onUploadProgress: (ratio) => {
        if (ratio < 1) {
          setUploadProgress(status, { label: "Sending DICOM files…", percent: Math.round(ratio * 100) });
        } else if (elapsedTimer === null) {
          showStage();
          elapsedTimer = window.setInterval(showStage, 1000);
        }
      },
      onLine: (event) => {
        if (event.event === "discovery") {
          renderDicomDiscovery(event);
          renderDicomWarnings(event.warnings || []);
        } else if (event.event === "saved_series") {
          renderSavedDicomSeries(event);
        } else if (event.event === "stage") {
          if (event.segmentation) {
            if (event.modality === "Muscle") renderMuscleSegmentationNote(event.segmentation);
            if (event.modality === "Radiomics") renderSegmentationNote(event.segmentation);
          }
          const labels = event.modality === "Muscle" ? MUSCLE_STAGE_LABELS : RADIOMICS_STAGE_LABELS;
          if (event.stage === "dicom_scanning") {
            stageLabel = "Reading DICOM metadata…";
          } else if (event.stage === "dicom_converting") {
            stageLabel = "Converting " + event.modality + " series to NIfTI…";
          } else if (event.stage === "features_start") {
            extractionModalities = event.modalities || [];
            stageLabel = "Extracting " + extractionModalities.join(" and ") + " features" +
              (event.parallel ? " in parallel…" : "…");
            extractionModalities.forEach((name) => setBranchStatus(name, "Calculating features…"));
          } else if (event.stage === "device") {
            stageLabel = "Segmenting Radiomics on " + (event.device === "cuda" ? "GPU" : "CPU") + "…";
          } else if (event.image_type) {
            stageLabel = "Radiomics: " + event.image_type + " (filter " + event.filter_index + " of " + event.filter_total + ")…";
          } else {
            stageLabel = (event.modality ? event.modality + ": " : "") +
              (labels[event.stage] || "Processing MRI…");
            if (event.series && event.series_total > 1) {
              stageLabel += " Series " + event.series + " of " + event.series_total + ".";
            }
          }
          if (event.modality) {
            setBranchStatus(event.modality, stageLabel);
            if (event.phase === "feature_extraction") {
              stageLabel = "Extracting " + extractionModalities.join(" and ") + " features" +
                (extractionModalities.length === 2 ? " in parallel…" : "…");
            }
          }
          showStage();
        } else if (event.event === "result") {
          state.uploads[event.modality] = event;
          setBranchStatus(event.modality, "Features ready", "ready");
          if (event.modality === "Muscle") {
            renderMuscleFeatures(event.features, event.image_filenames, event.segmentation);
            renderMuscleSegmentationNote(event.segmentation);
          } else if (event.modality === "Radiomics") {
            renderRadiomicsFeatures(event.features, event.image_filename, event.mask_filename, event.segmentation);
            renderSegmentationNote(event.segmentation);
          }
        } else if (event.event === "modality_error") {
          setBranchStatus(event.modality, event.message, "error");
        } else if (event.event === "complete") {
          completed = event;
        } else if (event.event === "error") {
          streamError = event;
        }
      },
    });
    if (!response.ok) {
      const detail = await parseJson(response);
      throw new Error(detail.message || "DICOM upload failed.");
    }
    if (streamError) throw new Error(streamError.message);
    if (!completed) throw new Error("Processing stopped before the server returned a complete result.");
    renderDicomWarnings(completed.warnings || []);
    setUploadStatus(status,
      completed.modalities.join(" and ") + " features are ready." +
      (completed.warnings.length ? " Review the warnings below." : ""),
      completed.warnings.length ? "pending" : "ready",
    );
  } catch (error) {
    state.uploads.Muscle = null;
    state.uploads.Radiomics = null;
    renderMuscleFeatures(null);
    renderRadiomicsFeatures(null);
    setUploadStatus(status, error.message, "error");
  } finally {
    if (elapsedTimer !== null) window.clearInterval(elapsedTimer);
    state.dicomBusy = false;
    updateFormState();
  }
}

function postWithProgress(url, formData, { onUploadProgress, onLine } = {}) {
  // fetch() cannot report request-body progress, and XHR exposes the response
  // as it arrives, which is what lets NDJSON stage events show up live.
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", url);

    let consumed = 0;
    const drain = (flush) => {
      if (!onLine) {
        return;
      }
      const text = request.responseText;
      for (
        let breakAt = text.indexOf("\n", consumed);
        breakAt !== -1;
        breakAt = text.indexOf("\n", consumed)
      ) {
        const line = text.slice(consumed, breakAt).trim();
        consumed = breakAt + 1;
        if (line) {
          try {
            onLine(JSON.parse(line));
          } catch {
            // A truncated line is impossible here: it ended with a newline.
          }
        }
      }
      const tail = flush ? text.slice(consumed).trim() : "";
      if (tail) {
        consumed = text.length;
        try {
          onLine(JSON.parse(tail));
        } catch {
          // The server closed mid-line; the caller reports the missing result.
        }
      }
    };

    request.upload.addEventListener("progress", (event) => {
      if (event.lengthComputable) {
        onUploadProgress?.(event.loaded / event.total);
      }
    });
    request.upload.addEventListener("load", () => onUploadProgress?.(1));
    request.addEventListener("progress", () => drain(false));
    request.addEventListener("load", () => {
      drain(true);
      resolve({
        ok: request.status >= 200 && request.status < 300,
        status: request.status,
        // Same shape parseJson() expects from fetch().
        json: async () => JSON.parse(request.responseText),
      });
    });
    request.addEventListener("error", () =>
      reject(new Error("The connection to the server failed during upload.")),
    );
    request.addEventListener("abort", () =>
      reject(new Error("The upload was cancelled.")),
    );
    request.addEventListener("timeout", () =>
      reject(new Error("The upload timed out.")),
    );
    request.send(formData);
  });
}

async function submitPrediction(event) {
  event.preventDefault();
  hideAlert();
  elements.resultSection.classList.add("d-none");
  clearLateFusionExplanation();

  const clinicalStatus = getClinicalStatus();
  if (!clinicalStatus.complete) {
    clearClinicalValidation();
    clinicalStatus.missing.forEach((field) => {
      document.getElementById(field.name)?.classList.add("is-invalid");
    });
    const first = document.getElementById(clinicalStatus.missing[0]?.name);
    first?.scrollIntoView({ behavior: "smooth", block: "center" });
    first?.focus({ preventScroll: true });
    showAlert(
      `Clinical, Surgery and Xray information is required. Complete or correct the ${clinicalStatus.missing.length} `
        + `highlighted field${clinicalStatus.missing.length === 1 ? "" : "s"}: `
        + `${clinicalStatus.missing.map((field) => field.label).join(", ")}.`,
      "warning",
    );
    return;
  }

  const body = {
    clinical: collectClinicalValues(),
    muscle: state.uploads.Muscle?.features,
    radiomics: state.uploads.Radiomics?.features,
  };

  elements.predictButton.disabled = true;
  elements.predictButton.textContent = "Predicting…";
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const result = await parseJson(response);
    if (!response.ok) {
      throw new Error(formatApiError(result));
    }
    renderResult(result);
    if (document.getElementById("explain-section")) {
      loadLateFusionExplanation(body, result);
    }
  } catch (error) {
    showAlert(error.message, "warning");
  } finally {
    elements.predictButton.textContent = "Predict subsidence risk";
    updateFormState();
  }
}

function collectClinicalValues() {
  const values = {};
  state.clinicalFields.forEach((field) => {
    const input = document.getElementById(field.name);
    const raw = input.value;
    values[field.name] = input.dataset.valueType === "integer"
      ? Number.parseInt(raw, 10)
      : Number.parseFloat(raw);
  });
  return values;
}

function renderResult(result) {
  const base = result.base_probabilities || {};
  document.getElementById("risk-value").textContent = formatPercent(
    result.probability_subsidence,
  );
  document.getElementById("result-class").textContent = result.classification;
  document.getElementById("result-threshold").textContent = formatNumber(
    result.decision_threshold,
  );
  document.getElementById("result-inputs").textContent =
    (result.input_modalities || []).join(" + ") || "—";
  document.getElementById("result-model").textContent = [
    result.selected_route,
    result.model_name,
  ].filter(Boolean).join(" · ") || "—";
  document.getElementById("result-clinical").textContent = formatPercent(base.clinical);
  document.getElementById("result-muscle").textContent = formatPercent(base.muscle);
  document.getElementById("result-radiomics").textContent = formatPercent(base.radiomics);
  elements.resultSection.classList.remove("d-none");
  elements.resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

function updateFormState() {
  const enabled = state.modelReady && state.dicomReady && !state.dicomBusy;
  ["dicom-files", "dicom-level"].forEach((id) => {
    document.getElementById(id).disabled = !enabled;
  });
  document.getElementById("dicom-process-button").disabled =
    !enabled || !selectedDicomFiles().length ||
    !document.getElementById("dicom-level").value;
  if (getClinicalStatus().complete) clearClinicalValidation();
  elements.predictButton.disabled = !state.modelReady || state.dicomBusy;
}

function clearClinicalValidation() {
  state.clinicalFields.forEach((field) => {
    document.getElementById(field.name)?.classList.remove("is-invalid");
  });
}

function getClinicalStatus() {
  if (!state.clinicalFields.length) {
    return { complete: false, partiallyComplete: false, missing: [] };
  }
  const missing = state.clinicalFields.filter((field) => {
    const input = document.getElementById(field.name);
    return !input || input.value === "" || !input.validity.valid;
  });
  const completed = state.clinicalFields.length - missing.length;
  return {
    complete: missing.length === 0,
    partiallyComplete: completed > 0 && missing.length > 0,
    missing,
  };
}

function setModelStatus(ready, text) {
  document.getElementById("model-status-dot").className =
    `status-dot ${ready ? "status-ready" : "status-error"}`;
  document.getElementById("model-status-text").textContent = text;
}

function renderModelRoutes(routes) {
  const panel = elements.modelRoutes;
  panel.textContent = "";
  const entries = Object.entries(routes || {});
  if (!entries.length) {
    elements.modelStatus.disabled = true;
    return;
  }

  const caption = document.createElement("p");
  caption.className = "model-routes-caption";
  caption.textContent =
    "The route is chosen automatically from the inputs you provide.";
  panel.appendChild(caption);

  const table = document.createElement("table");
  table.className = "table table-sm model-routes-table mb-0";
  const head = table.createTHead().insertRow();
  ["Inputs", "Strategy", "Model", "CV AUC", "Threshold"].forEach((title, index) => {
    const cell = document.createElement("th");
    cell.scope = "col";
    cell.textContent = title;
    if (index >= 3) {
      cell.className = "text-end";
    }
    head.appendChild(cell);
  });

  const body = table.createTBody();
  entries.forEach(([routeKey, route]) => {
    const row = body.insertRow();
    const label = document.createElement("th");
    label.scope = "row";
    label.textContent = routeKey.split("+").join(" + ");
    row.appendChild(label);
    row.insertCell().textContent = (route.strategy || "").replace("_", " ");
    row.insertCell().textContent = route.model_name || "—";
    [route.selection_cv_auc, route.decision_threshold].forEach((value) => {
      const cell = row.insertCell();
      cell.className = "text-end";
      cell.textContent = typeof value === "number" ? value.toFixed(3) : "—";
    });
  });

  panel.appendChild(table);
  elements.modelStatus.disabled = false;
}

function toggleModelRoutes(force) {
  const expanded =
    typeof force === "boolean"
      ? force
      : elements.modelStatus.getAttribute("aria-expanded") !== "true";
  elements.modelStatus.setAttribute("aria-expanded", String(expanded));
  elements.modelRoutes.classList.toggle("d-none", !expanded);
}

function setUploadStatus(element, text, kind) {
  element.className = `upload-status mt-3 is-${kind}`;
  element.textContent = text;
  delete element.dataset.progressPhase;
}

function renderRadiomicsFeatures(features, imageName, maskName, segmentation) {
  renderSegmentationPreviews("Radiomics", features ? segmentation : null);
  const card = document.getElementById("radiomics-features-card");
  const container = document.getElementById("radiomics-features");
  const source = document.getElementById("radiomics-features-source");
  container.textContent = "";
  source.textContent = "";
  if (!features) {
    card.classList.add("d-none");
    return;
  }
  if (imageName) {
    const voi = maskName
      || (segmentation?.levels?.length
        ? `${segmentation.levels.join("+")} VOI`
        : "");
    source.textContent = voi ? `${imageName} + ${voi}` : imageName;
  }

  const table = document.createElement("table");
  table.className = "table muscle-features-table mb-0";
  const head = table.createTHead().insertRow();
  ["#", "Feature", "Value"].forEach((title, index) => {
    const cell = document.createElement("th");
    cell.scope = "col";
    cell.textContent = title;
    if (index === 2) {
      cell.className = "text-end";
    }
    head.appendChild(cell);
  });

  const body = table.createTBody();
  Object.entries(features).forEach(([name, value], index) => {
    const row = body.insertRow();
    const number = document.createElement("th");
    number.scope = "row";
    number.className = "muscle-features-index";
    number.textContent = String(index + 1);
    row.appendChild(number);
    const featureCell = row.insertCell();
    featureCell.className = "muscle-features-code";
    // Every selected feature carries the Radiomics___MRI_ prefix.
    featureCell.textContent = name.replace("Radiomics___MRI_", "");
    const valueCell = row.insertCell();
    valueCell.className = "text-end";
    valueCell.textContent =
      typeof value === "number" ? value.toPrecision(6) : "\u2014";
  });

  container.appendChild(table);

  const footnote = document.createElement("p");
  footnote.className = "muscle-features-caption mt-3 mb-0";
  footnote.textContent =
    `${Object.keys(features).length} features, extracted with the FAE settings in `
    + "configs/fae_radiomics_all_1688.yaml. These values are held in your "
    + "browser and sent with the prediction request.";
  container.appendChild(footnote);

  card.classList.remove("d-none");
}

function describeDevice(segmentation) {
  const trace = segmentation.device_trace || {};
  if (segmentation.device === "cuda") {
    const size = trace.gpu_total_mb
      ? ` · ${(trace.gpu_total_mb / 1024).toFixed(1)} GB`
      : "";
    return `GPU — ${trace.gpu_name || "CUDA device"}${size}`;
  }
  if (segmentation.device === "cpu") {
    return trace.cuda_available === false
      ? "CPU — torch reports no CUDA device"
      : "CPU";
  }
  return segmentation.device || "";
}

// The device label says what was asked for; these numbers say what the card
// actually did. A run that never moves off its idle memory did not use it.
function describeGpuTrace(trace) {
  if (!trace) {
    return "";
  }
  const peak = `${trace.peak_util_percent}% peak utilisation`;
  const memory = `${trace.peak_used_mb.toLocaleString()} MB peak`;
  const delta = `+${trace.peak_delta_mb.toLocaleString()} MB over idle`;
  return `${peak} · ${memory} (${delta}) · ${trace.samples} samples`;
}

function renderSegmentationNote(segmentation) {
  const note = document.getElementById("radiomics-segmentation");
  note.textContent = "";
  if (!segmentation) {
    note.classList.add("d-none");
    return;
  }

  const heading = document.createElement("div");
  heading.textContent =
    `TotalSegmentator ${segmentation.version || ""}`.trim()
    + ` built the VOI from ${(segmentation.levels || []).join(", ")}`
    + ", vertebral body only.";
  note.appendChild(heading);

  const trace = segmentation.device_trace || {};
  const list = document.createElement("dl");
  const rows = [
    [
      "Level masks",
      Object.entries(segmentation.level_voxel_counts || {})
        .map(([code, count]) => `${code}: ${count.toLocaleString()}`)
        .join(" · "),
    ],
    [
      "Body voxels by level",
      Object.entries(segmentation.voxel_counts || {})
        .map(([code, count]) => `${code}: ${count.toLocaleString()}`)
        .join(" · "),
    ],
    ["Used for extraction", `${(segmentation.mask_voxels || 0).toLocaleString()} voxels`],
    ["Intersection", segmentation.intersection_method || ""],
    ["Device", describeDevice(segmentation)],
    ["GPU activity", describeGpuTrace(segmentation.gpu_trace)],
    [
      "Torch",
      trace.torch_version
        ? `${trace.torch_version}`
          + (trace.torch_cuda ? ` · CUDA ${trace.torch_cuda}` : " · no CUDA build")
        : "",
    ],
    [
      "Runtime",
      segmentation.duration_seconds ? `${segmentation.duration_seconds} s` : "",
    ],
    ["Saved to", segmentation.directory || ""],
  ];
  rows.forEach(([term, value]) => {
    if (!value) {
      return;
    }
    const dt = document.createElement("dt");
    dt.textContent = term;
    const dd = document.createElement("dd");
    dd.textContent = value;
    if (term === "Saved to") {
      dd.className = "segmentation-note-path";
    }
    list.append(dt, dd);
  });
  note.appendChild(list);
  appendSegmentationFiles(note, segmentation);
  note.classList.remove("d-none");
}

function appendSegmentationFiles(container, segmentation) {
  const files = Object.entries(segmentation.files || {});
  if (!files.length) return;
  const details = document.createElement("details");
  details.className = "dicom-discovery mt-3";
  const summary = document.createElement("summary");
  summary.textContent = "Saved segmentation files";
  const list = document.createElement("ul");
  list.className = "segmentation-file-list mt-2 mb-0";
  files.forEach(([key, filename]) => {
    const item = document.createElement("li");
    const url = segmentation.file_urls?.[key];
    const name = document.createElement(url ? "a" : "span");
    name.textContent = filename;
    if (url) {
      name.href = url + "?download=1";
      name.setAttribute("download", "");
    }
    item.appendChild(name);
    list.appendChild(item);
  });
  details.append(summary, list);
  container.appendChild(details);
}

function renderMuscleSegmentationNote(segmentation) {
  const note = document.getElementById("muscle-segmentation");
  note.replaceChildren();
  note.classList.toggle("d-none", !segmentation);
  if (!segmentation) return;
  const heading = document.createElement("strong");
  heading.textContent = "MuscleMap segmentation saved for inspection";
  const list = document.createElement("dl");
  const rows = [
    ["Method", `${segmentation.method || ""} · ${segmentation.components || ""} components`],
    ["Saved to", segmentation.directory],
    ["Source selection", segmentation.source_manifest],
  ];
  (segmentation.series || []).forEach((series) => {
    rows.push([
      `Series ${series.series}`,
      `${series.source_filename} · ${(series.mask_voxels || 0).toLocaleString()} segmented voxels`,
    ]);
  });
  rows.forEach(([term, value]) => {
    if (!value) return;
    const dt = document.createElement("dt");
    dt.textContent = term;
    const dd = document.createElement("dd");
    dd.textContent = value;
    if (term === "Saved to" || term === "Source selection") dd.className = "segmentation-note-path";
    list.append(dt, dd);
  });
  note.append(heading, list);
  appendSegmentationFiles(note, segmentation);
}

function renderSegmentationPreviews(modality, segmentation) {
  const container = document.getElementById(modality.toLowerCase() + "-previews");
  container.replaceChildren();
  const keys = modality === "Radiomics"
    ? ["voi_preview"]
    : Object.keys(segmentation?.files || {}).filter((key) => key.endsWith("_preview"));
  keys.forEach((key) => {
    const url = segmentation?.file_urls?.[key];
    if (!url) return;
    const figure = document.createElement("figure");
    figure.className = "segmentation-preview mb-0";
    const image = document.createElement("img");
    image.src = url;
    image.loading = "lazy";
    image.alt = modality === "Radiomics"
      ? "Selected vertebral body VOI overlaid on the sagittal MRI"
      : "MuscleMap muscle segmentation overlaid on the axial MRI";
    const caption = document.createElement("figcaption");
    const link = document.createElement("a");
    link.href = url + "?download=1";
    link.setAttribute("download", "");
    link.textContent = segmentation.files[key];
    caption.appendChild(link);
    image.addEventListener("error", () => {
      image.classList.add("d-none");
      const message = document.createElement("p");
      message.textContent = "Preview unavailable. Inspect the saved segmentation files above.";
      figure.prepend(message);
    }, { once: true });
    figure.append(image, caption);
    container.appendChild(figure);
  });
  container.classList.toggle("d-none", !container.childElementCount);
}

function clearLateFusionExplanation() {
  state.explanationController?.abort();
  state.explanationController = null;
  document.getElementById("explain-section")?.classList.add("d-none");
  document.getElementById("explain-plots")?.replaceChildren();
  const status = document.getElementById("explain-status");
  if (status) status.textContent = "";
}

async function loadLateFusionExplanation(body, prediction) {
  clearLateFusionExplanation();
  // Clinical-only/direct models remain usable, but are not late fusion.
  if (prediction.fusion_strategy !== "late_fusion") return;

  const section = document.getElementById("explain-section");
  const plots = document.getElementById("explain-plots");
  const status = document.getElementById("explain-status");
  const controller = new AbortController();
  state.explanationController = controller;
  status.textContent = "Computing late-fusion SHAP…";
  section.classList.remove("d-none");

  let explanation = null;
  let complete = false;
  const handleEvent = (event) => {
    if (event.event === "error") throw new Error(event.message || "SHAP is unavailable.");
    if (event.event === "start") {
      plots.textContent = "Computing contributions to the final late-fusion probability…";
    } else if (event.event === "explanation") {
      if (
        explanation || event.scope !== "end_to_end" ||
        event.fusion_strategy !== "late_fusion" ||
        !Number.isFinite(event.probability) ||
        Math.abs(event.probability - prediction.probability_subsidence) > 1e-6
      ) throw new Error("SHAP does not match this late-fusion prediction.");
      explanation = event;
      renderLateFusionExplanation(event, plots);
    } else if (event.event === "complete") {
      complete = true;
    }
  };

  try {
    const response = await fetch("/api/explain", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    if (state.explanationController !== controller) return;
    if (!response.ok) {
      const detail = await parseJson(response);
      throw new Error(detail.message || "Late-fusion SHAP is unavailable.");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffered = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (state.explanationController !== controller) return;
      buffered += done ? decoder.decode() : decoder.decode(value, { stream: true });
      let breakAt = buffered.indexOf("\n");
      while (breakAt !== -1) {
        const line = buffered.slice(0, breakAt).trim();
        buffered = buffered.slice(breakAt + 1);
        if (line) handleEvent(JSON.parse(line));
        breakAt = buffered.indexOf("\n");
      }
      if (done) break;
    }
    if (buffered.trim()) handleEvent(JSON.parse(buffered));
    if (!complete || !explanation) throw new Error("The late-fusion SHAP response was interrupted.");
    status.textContent = "Final late-fusion prediction";
  } catch (error) {
    if (controller.signal.aborted || state.explanationController !== controller) return;
    plots.replaceChildren();
    status.textContent = "";
    const failure = document.createElement("div");
    failure.className = "explain-plot-error";
    failure.textContent = `Late-fusion SHAP could not be computed: ${error.message}`;
    plots.appendChild(failure);
  } finally {
    if (state.explanationController === controller) state.explanationController = null;
  }
}

function renderLateFusionExplanation(event, plots) {
  plots.replaceChildren();
  const slot = document.createElement("div");
  slot.className = "explain-plot";
  const heading = document.createElement("div");
  heading.className = "explain-plot-heading";
  heading.textContent = `Late Fusion · ${event.model_name} · ${event.input_modalities.join(" + ")}`;
  const note = document.createElement("p");
  note.className = "explain-plot-note";
  note.textContent =
    `Reference probability ${formatPercent(event.base_value)} → this patient `
    + `${formatPercent(event.probability)}. Red increases the final risk; blue decreases it.`;
  const image = document.createElement("img");
  image.src = `data:image/png;base64,${event.image_png_base64}`;
  image.alt = "End-to-end SHAP waterfall for the final late-fusion subsidence probability";
  slot.append(heading, note, image);
  plots.appendChild(slot);
}

function renderDicomBranchStatus(statuses) {
  const panel = document.getElementById("dicom-branch-status");
  panel.replaceChildren();
  ["Muscle", "Radiomics"].forEach((modality) => {
    const state = statuses[modality];
    if (!state) return;
    const row = document.createElement("div");
    row.className = state.type === "error" ? "text-danger" : state.type === "ready" ? "text-success" : "text-secondary";
    row.textContent = modality + ": " + state.text;
    panel.appendChild(row);
  });
  panel.classList.toggle("d-none", !panel.childElementCount);
}

function renderSavedDicomSeries(saved) {
  const panel = document.getElementById("dicom-saved-series");
  panel.replaceChildren();
  const title = document.createElement("strong");
  title.textContent = "Selected MRI saved for inspection";
  const directory = document.createElement("div");
  directory.className = "segmentation-note-path";
  directory.textContent = saved.directory;
  const manifest = document.createElement("div");
  manifest.className = "segmentation-note-path";
  manifest.textContent = "Selection metadata: " + saved.manifest;
  const list = document.createElement("ul");
  list.className = "mb-0 mt-2";
  (saved.selected || []).forEach((series) => {
    const item = document.createElement("li");
    const name = series.modality === "Muscle" ? "Axial T2 VIEW" : "Sagittal T1 / T1 FLAIR (non-STIR)";
    item.textContent = name + " · " + (series.description || series.protocol || "Unnamed series") +
      " · " + (series.dicom_files || []).length + " source DICOM files" +
      (series.status === "converted" ? "" : " · Conversion failed");
    const files = document.createElement("div");
    files.className = "segmentation-note-path";
    files.textContent = (series.nifti_files || []).join(", ") || "No NIfTI output; retained source DICOMs and selection metadata.";
    item.appendChild(files);
    list.appendChild(item);
  });
  const note = document.createElement("p");
  note.className = "mt-2 mb-0";
  note.textContent = "Files remain on the server even if feature extraction fails. Source DICOMs may contain patient information; they are not publicly served.";
  panel.append(title, directory, manifest, list, note);
  panel.classList.remove("d-none");
}

function selectedDicomFiles() {
  return Array.from(document.getElementById("dicom-files").files || []);
}

function renderDicomSelection() {
  const panel = document.getElementById("dicom-selection");
  const files = selectedDicomFiles();
  if (!files.length) {
    panel.replaceChildren();
    panel.classList.add("d-none");
    return;
  }
  const megabytes = files.reduce((total, file) => total + file.size, 0) / 1048576;
  const details = document.createElement("details");
  details.className = "dicom-discovery mt-1";
  const summary = document.createElement("summary");
  summary.textContent = files.length + " files · " + megabytes.toFixed(1) + " MB";
  const list = document.createElement("div");
  list.className = "ms-3 mt-1 font-monospace";
  const names = files.map((file) => file.name).sort();
  const shown = 500;
  list.textContent = names.slice(0, shown).join(", ") +
    (names.length > shown ? ", … +" + (names.length - shown) + " more" : "");
  details.append(summary, list);
  panel.replaceChildren(details);
  panel.classList.remove("d-none");
}

function renderDicomDiscovery(discovery) {
  const panel = document.getElementById("dicom-discovery");
  const container = document.getElementById("dicom-series");
  container.replaceChildren();
  const table = document.createElement("table");
  table.className = "table table-sm dicom-series-table mb-0";
  const head = table.createTHead().insertRow();
  ["Series", "Plane", "Type", "TR (ms)", "TE (ms)", "TI (ms)", "Decision"].forEach((title) => {
    const cell = document.createElement("th");
    cell.scope = "col";
    cell.textContent = title;
    head.appendChild(cell);
  });
  const body = table.createTBody();
  (discovery.series || []).forEach((series) => {
    const row = body.insertRow();
    const values = [
      series.description || series.protocol || "Unnamed series " + series.series,
      series.plane || "Unknown",
      series.classification || "Not selected",
      series.tr ?? "—", series.te ?? "—", series.ti ?? "—",
      (series.reasons || []).map((reason) => reason.replaceAll("_", " ")).join("; "),
    ];
    values.forEach((value) => {
      const cell = row.insertCell();
      cell.textContent = String(value);
    });
  });
  container.appendChild(table);
  panel.classList.remove("d-none");
}

function renderMuscleFeatures(features, sourceFilenames, segmentation) {
  renderSegmentationPreviews("Muscle", features ? segmentation : null);
  const card = document.getElementById("muscle-features-card");
  const container = document.getElementById("muscle-features");
  const source = document.getElementById("muscle-features-source");
  container.textContent = "";
  source.textContent = "";
  if (!features) {
    card.classList.add("d-none");
    return;
  }
  const names = sourceFilenames || [];
  if (names.length) {
    source.textContent =
      names.length === 1 ? names[0] : `${names.length} series combined`;
    source.title = names.join("\n");
  }

  const table = document.createElement("table");
  table.className = "table muscle-features-table mb-0";
  const head = table.createTHead().insertRow();
  [["#", ""], ["Feature", ""], ["Muscle", ""], ["Value", ""], ["Unit", ""]].forEach(
    ([title], index) => {
      const cell = document.createElement("th");
      cell.scope = "col";
      cell.textContent = title;
      if (index === 3) {
        cell.className = "text-end";
      }
      head.appendChild(cell);
    },
  );

  const body = table.createTBody();
  let index = 0;
  MUSCLE_ROWS.forEach(([code, name]) => {
    ["aVol", "tFat"].forEach((metric) => {
      index += 1;
      const row = body.insertRow();
      const number = document.createElement("th");
      number.scope = "row";
      number.className = "muscle-features-index";
      number.textContent = String(index);
      row.appendChild(number);
      const featureCell = row.insertCell();
      featureCell.className = "muscle-features-code";
      featureCell.textContent = `${metric}_${code}`;
      row.insertCell().textContent = name;
      const valueCell = row.insertCell();
      valueCell.className = "text-end";
      const value = features[`Muscle___${metric}_${code}`];
      valueCell.textContent = typeof value === "number" ? value.toFixed(4) : "—";
      const unitCell = row.insertCell();
      unitCell.className = "muscle-features-unit-cell";
      unitCell.textContent = metric === "aVol" ? "ml/slice" : "%";
    });
  });

  container.appendChild(table);

  const footnote = document.createElement("p");
  footnote.className = "muscle-features-caption mt-3 mb-0";
  footnote.textContent =
    (names.length > 1
      ? `Aggregated across ${names.length} series: ${names.join(", ")}. `
      : "")
    + `${Object.keys(features).length} features, matching the aVol_*/tFat_* columns of `
    + "CSA_muscles_features_new.csv. aVol is total volume divided by total "
    + "segmented slices; tFat is fat plus undefined tissue, weighted by slice count. "
    + "These values are held in your browser and sent with the prediction request.";
  container.appendChild(footnote);

  card.classList.remove("d-none");
}

function setUploadProgress(element, { label, percent, elapsedSeconds }) {
  const determinate = typeof percent === "number";
  const phase = determinate ? "upload" : "work";

  if (!element.querySelector(".progress-bar") || element.dataset.progressPhase !== phase) {
    element.className = "upload-status mt-3 is-pending";
    element.dataset.progressPhase = phase;
    element.textContent = "";

    const heading = document.createElement("div");
    heading.className = "upload-progress-label";
    // Only the stage text is announced; the counter beside it ticks every
    // second and would otherwise flood the aria-live region.
    const stageText = document.createElement("span");
    stageText.className = "upload-progress-text";
    const counter = document.createElement("span");
    counter.className = "upload-progress-meta";
    counter.setAttribute("aria-hidden", "true");
    heading.append(stageText, counter);

    const newTrack = document.createElement("div");
    newTrack.className = "progress upload-progress";
    newTrack.setAttribute("role", "progressbar");
    newTrack.setAttribute("aria-label", label);
    const newBar = document.createElement("div");
    newBar.className = `progress-bar${determinate ? "" : " is-indeterminate"}`;
    newTrack.appendChild(newBar);

    element.appendChild(heading);
    element.appendChild(newTrack);
  }

  const meta = element.querySelector(".upload-progress-meta");
  const track = element.querySelector(".upload-progress");
  const bar = element.querySelector(".progress-bar");
  const stageText = element.querySelector(".upload-progress-text");
  if (stageText.textContent !== label) {
    stageText.textContent = label;
    track.setAttribute("aria-label", label);
  }
  if (determinate) {
    meta.textContent = ` ${percent}%`;
    bar.style.width = `${percent}%`;
    track.setAttribute("aria-valuenow", String(percent));
    track.setAttribute("aria-valuemin", "0");
    track.setAttribute("aria-valuemax", "100");
  } else {
    meta.textContent = elapsedSeconds ? ` ${elapsedSeconds}s` : "";
    track.removeAttribute("aria-valuenow");
  }
}

function showAlert(message, type) {
  elements.globalAlert.className = `alert alert-${type}`;
  elements.globalAlert.textContent = message;
  elements.globalAlert.classList.remove("d-none");
  elements.globalAlert.scrollIntoView({ behavior: "smooth", block: "center" });
}

function hideAlert() {
  elements.globalAlert.classList.add("d-none");
}

function formatApiError(payload) {
  const missing = payload.details?.missing_features;
  if (Array.isArray(missing) && missing.length) {
    return `${payload.message} Missing ${missing.length} feature(s): ${missing.join(", ")}`;
  }
  return payload.message || payload.error || "Prediction failed.";
}

function formatPercent(value) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(2)}%` : "—";
}

function formatNumber(value) {
  return Number.isFinite(value) ? value.toFixed(2) : "—";
}

async function parseJson(response) {
  try {
    return await response.json();
  } catch {
    return { message: `Server returned HTTP ${response.status}.` };
  }
}
