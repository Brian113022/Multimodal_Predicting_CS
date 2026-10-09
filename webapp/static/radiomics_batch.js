// Batch Radiomics page. The server scans the folder and runs the job on a
// background thread; this script renders the patient table, starts the job and
// polls it, so a reload re-attaches to a run that is still going.

const POLL_MS = 1000;
const FOLDER_KEY = "radiomicsBatch.folder";

const STATUS_LABELS = {
  ready: "Ready",
  check: "Check files",
  queued: "Queued",
  running: "Running",
  done: "Done",
  failed: "Failed",
  cancelled: "Cancelled",
};

const state = {
  defaults: null,
  // The last scan: { root, maskPattern, cases, default_output }.
  scan: null,
  // Case id -> { item, checkbox, imageSelect, maskSelect, statusCell }.
  rows: new Map(),
  job: null,
  pollTimer: null,
  // Case ids to tick after the next scan, set by "Retry failed".
  retryIds: null,
};

const $ = (id) => document.getElementById(id);
const elements = {
  alert: $("global-alert"),
  scanForm: $("scan-form"),
  folder: $("folder"),
  maskPattern: $("mask-pattern"),
  scanButton: $("scan-button"),
  patientsCard: $("patients-card"),
  patientsSummary: $("patients-summary"),
  patientsBody: $("patients-body"),
  selectAll: $("select-all"),
  selectReady: $("select-ready"),
  selectNone: $("select-none"),
  selectionTools: $("selection-tools"),
  runCard: $("run-card"),
  outputCsv: $("output-csv"),
  workers: $("workers"),
  workersNote: $("workers-note"),
  label: $("label"),
  normalizeScale: $("normalize-scale"),
  fixedSettings: $("fixed-settings"),
  startButton: $("start-button"),
  cancelButton: $("cancel-button"),
  jobStatus: $("job-status"),
  jobResult: $("job-result"),
};

document.addEventListener("DOMContentLoaded", async () => {
  bindEvents();
  try {
    await loadDefaults();
  } catch (error) {
    showAlert(error.message, "danger");
  }
});

function bindEvents() {
  elements.scanForm.addEventListener("submit", (event) => {
    event.preventDefault();
    scanFolder();
  });
  // The header box also accepts the guesses in "check" rows; the button below
  // it ticks only the rows the scan was sure about.
  elements.selectAll.addEventListener("change", () => {
    const tick = elements.selectAll.checked;
    for (const row of state.rows.values()) {
      row.checkbox.checked = tick && hasValidFiles(row);
    }
    updateSelectionSummary();
  });
  elements.selectReady.addEventListener("click", () => {
    for (const row of state.rows.values()) {
      row.checkbox.checked = row.item.status === "ready" && hasValidFiles(row);
    }
    updateSelectionSummary();
  });
  elements.selectNone.addEventListener("click", () => {
    for (const row of state.rows.values()) {
      row.checkbox.checked = false;
    }
    updateSelectionSummary();
  });
  elements.startButton.addEventListener("click", startJob);
  elements.cancelButton.addEventListener("click", cancelJob);
}

async function loadDefaults() {
  const response = await fetch("/api/radiomics-batch/defaults");
  const payload = await parseJson(response);
  if (!response.ok) {
    throw new Error(payload.message || "Cannot load the Radiomics settings.");
  }
  state.defaults = payload;
  elements.maskPattern.value = payload.mask_pattern;
  elements.workers.value = payload.workers;
  elements.workers.max = payload.max_workers;
  elements.workersNote.textContent =
    `One PyRadiomics process each; this computer has ${payload.max_workers} logical CPUs.`;
  elements.label.value = payload.settings.label ?? 1;
  elements.normalizeScale.value = payload.settings.normalizeScale ?? 1;
  renderFixedSettings(payload);
  if (!payload.radiomics_python) {
    showAlert(
      "No conda environment with both PyRadiomics and SimpleITK was found. "
      + "Start the server with RADIOMICS_PYTHON pointing to that environment's python.",
      "warning",
    );
  }

  const savedFolder = readStorage(FOLDER_KEY);
  if (savedFolder && !elements.folder.value) {
    elements.folder.value = savedFolder;
  }

  if (payload.active_job) {
    const jobResponse = await fetch(`/api/radiomics-batch/jobs/${payload.active_job}`);
    const snapshot = await parseJson(jobResponse);
    if (jobResponse.ok) {
      elements.folder.value = snapshot.root;
      attachJob(snapshot);
    }
  }
}

function renderFixedSettings(defaults) {
  const settings = defaults.settings;
  const entries = [
    ["Filters", defaults.image_types.join(", ")],
    ["Classes", defaults.feature_classes.join(", ")],
    [
      "Resampling",
      settings.resampledPixelSpacing
        ? `${settings.resampledPixelSpacing.join(" × ")} mm, ${settings.interpolator || "sitkBSpline"}`
        : "none",
    ],
    ["Intensity", settings.normalize ? "z-score over the whole image × normalizeScale" : "raw"],
    ["Binning", settings.binCount ? `binCount ${settings.binCount}` : `binWidth ${settings.binWidth ?? 25}`],
    ["Config", defaults.config_path],
  ];
  elements.fixedSettings.replaceChildren();
  for (const [term, value] of entries) {
    const dt = document.createElement("dt");
    dt.textContent = term;
    const dd = document.createElement("dd");
    dd.textContent = value;
    elements.fixedSettings.append(dt, dd);
  }
}

async function scanFolder() {
  const folder = elements.folder.value.trim();
  if (!folder) {
    showAlert("Enter the folder that holds the patient folders.", "warning");
    elements.folder.focus();
    return;
  }
  hideAlert();
  const maskPattern = elements.maskPattern.value.trim();
  elements.scanButton.disabled = true;
  elements.scanButton.textContent = "Scanning…";
  try {
    const response = await fetch("/api/radiomics-batch/scan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ folder, mask_pattern: maskPattern }),
    });
    const payload = await parseJson(response);
    if (!response.ok) {
      showAlert(payload.message || "The folder could not be scanned.", "danger");
      return;
    }
    writeStorage(FOLDER_KEY, folder);
    state.scan = { ...payload, maskPattern };
    renderScan(state.scan);
  } catch (error) {
    showAlert(`The folder could not be scanned: ${error.message}`, "danger");
  } finally {
    elements.scanButton.disabled = isJobRunning();
    elements.scanButton.textContent = "Scan folder";
  }
}

function renderScan(scan) {
  state.rows.clear();
  elements.patientsBody.replaceChildren();
  elements.selectionTools.classList.remove("d-none");
  elements.selectAll.disabled = false;
  const retry = state.retryIds;
  state.retryIds = null;

  for (const item of scan.cases) {
    const tr = document.createElement("tr");

    const checkCell = document.createElement("td");
    checkCell.className = "batch-check-cell";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.className = "form-check-input";
    checkbox.setAttribute("aria-label", `Extract ${item.name}`);
    checkCell.appendChild(checkbox);

    const imageSelect = fileSelect(item.files, item.image, "image");
    const maskSelect = fileSelect(item.files, item.mask, "VOI");
    const statusCell = document.createElement("td");
    const row = { item, checkbox, imageSelect, maskSelect, statusCell, edited: false };

    const imageCell = document.createElement("td");
    imageCell.appendChild(imageSelect);
    const maskCell = document.createElement("td");
    maskCell.appendChild(maskSelect);
    tr.append(checkCell, nameCell(item), groupCell(item), imageCell, maskCell, statusCell);
    elements.patientsBody.appendChild(tr);

    const onFileChange = () => {
      row.edited = true;
      // Choosing the files is the confirmation a "check" row was waiting for.
      checkbox.checked = hasValidFiles(row);
      renderScanStatus(row);
      updateSelectionSummary();
    };
    imageSelect.addEventListener("change", onFileChange);
    maskSelect.addEventListener("change", onFileChange);
    checkbox.addEventListener("change", updateSelectionSummary);

    checkbox.checked = retry
      ? retry.has(item.id) && hasValidFiles(row)
      : item.status === "ready";
    renderScanStatus(row);
    state.rows.set(item.id, row);
  }

  elements.outputCsv.value = scan.default_output;
  elements.patientsCard.classList.remove("d-none");
  elements.runCard.classList.remove("d-none");
  elements.startButton.classList.remove("d-none");
  elements.jobStatus.classList.add("d-none");
  elements.jobResult.classList.add("d-none");
  updateSelectionSummary();
  if (!scan.cases.length) {
    showAlert(
      "No folder below this path contains a .nii or .nii.gz file.",
      "warning",
    );
  }
}

function fileSelect(files, selected, what) {
  const select = document.createElement("select");
  select.className = "form-select form-select-sm batch-file-select";
  select.setAttribute("aria-label", `${what} file`);
  if (!selected) {
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = `Choose the ${what}…`;
    select.appendChild(placeholder);
  }
  for (const file of files) {
    const option = document.createElement("option");
    option.value = file;
    option.textContent = file;
    select.appendChild(option);
  }
  select.value = selected || "";
  select.title = select.value;
  select.addEventListener("change", () => {
    select.title = select.value;
  });
  return select;
}

function nameCell(item) {
  const cell = document.createElement("th");
  cell.scope = "row";
  cell.className = "batch-name";
  cell.textContent = item.name;
  return cell;
}

function groupCell(item) {
  const cell = document.createElement("td");
  cell.className = "text-secondary";
  cell.textContent = item.group || "—";
  return cell;
}

function hasValidFiles(row) {
  const image = row.imageSelect.value;
  const mask = row.maskSelect.value;
  return Boolean(image && mask && image !== mask);
}

function renderScanStatus(row) {
  let status = row.item.status;
  let detail = row.item.issues.join(" ");
  if (row.edited) {
    status = hasValidFiles(row) ? "ready" : "check";
    detail = hasValidFiles(row)
      ? "Files chosen by hand."
      : "Choose two different files: the image and the VOI.";
  }
  row.checkbox.disabled = !hasValidFiles(row);
  if (row.checkbox.disabled) {
    row.checkbox.checked = false;
  }
  setStatusCell(row.statusCell, status, detail);
}

function updateSelectionSummary() {
  const rows = [...state.rows.values()];
  const ticked = rows.filter((row) => row.checkbox.checked).length;
  const selectable = rows.filter(hasValidFiles).length;
  const ready = rows.filter((row) => row.item.status === "ready").length;
  const toCheck = rows.length - ready;
  elements.patientsSummary.textContent =
    `${rows.length} patient folder${rows.length === 1 ? "" : "s"} in ${state.scan?.root ?? ""}`
    + ` · ${ready} ready · ${toCheck} to check · ${ticked} ticked`;
  elements.selectAll.checked = ticked > 0 && ticked === selectable;
  elements.selectAll.indeterminate = ticked > 0 && ticked < selectable;
  elements.startButton.disabled = ticked === 0 || isJobRunning();
  elements.startButton.textContent =
    `Extract radiomics for ${ticked} patient${ticked === 1 ? "" : "s"}`;
}

async function startJob() {
  if (!state.scan) {
    return;
  }
  const cases = [...state.rows.values()]
    .filter((row) => row.checkbox.checked && hasValidFiles(row))
    .map((row) => ({
      id: row.item.id,
      image: row.imageSelect.value,
      mask: row.maskSelect.value,
    }));
  if (!cases.length) {
    showAlert("Tick at least one patient to extract.", "warning");
    return;
  }
  hideAlert();
  elements.startButton.disabled = true;
  try {
    const response = await fetch("/api/radiomics-batch/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        folder: state.scan.root,
        mask_pattern: state.scan.maskPattern,
        cases,
        output_csv: elements.outputCsv.value.trim(),
        workers: elements.workers.value,
        label: elements.label.value,
        normalize_scale: elements.normalizeScale.value,
      }),
    });
    const payload = await parseJson(response);
    if (!response.ok) {
      showAlert(payload.message || "The batch could not start.", "danger");
      elements.startButton.disabled = false;
      return;
    }
    attachJob(payload);
  } catch (error) {
    showAlert(`The batch could not start: ${error.message}`, "danger");
    elements.startButton.disabled = false;
  }
}

async function cancelJob() {
  if (!state.job) {
    return;
  }
  elements.cancelButton.disabled = true;
  elements.cancelButton.textContent = "Cancelling…";
  try {
    const response = await fetch(`/api/radiomics-batch/jobs/${state.job.id}/cancel`, {
      method: "POST",
    });
    if (response.ok) {
      updateJob(await parseJson(response));
    }
  } catch (error) {
    showAlert(`Cancel failed: ${error.message}`, "danger");
    elements.cancelButton.disabled = false;
  }
}

function attachJob(snapshot) {
  state.job = snapshot;
  renderJobTable(snapshot);
  elements.patientsCard.classList.remove("d-none");
  elements.runCard.classList.remove("d-none");
  // A finished run offers "Retry" and "New scan" instead; a scan brings it back.
  elements.startButton.classList.add("d-none");
  elements.outputCsv.value = snapshot.output_csv;
  elements.workers.value = snapshot.workers;
  elements.jobResult.classList.add("d-none");
  updateJob(snapshot);
}

function renderJobTable(snapshot) {
  state.rows.clear();
  elements.patientsBody.replaceChildren();
  elements.selectionTools.classList.add("d-none");
  elements.selectAll.checked = false;
  elements.selectAll.indeterminate = false;
  elements.selectAll.disabled = true;
  for (const item of snapshot.cases) {
    const tr = document.createElement("tr");
    const markCell = document.createElement("td");
    markCell.className = "batch-check-cell";
    const imageCell = document.createElement("td");
    imageCell.className = "batch-file";
    imageCell.textContent = item.image;
    imageCell.title = item.image;
    const maskCell = document.createElement("td");
    maskCell.className = "batch-file";
    maskCell.textContent = item.mask;
    maskCell.title = item.mask;
    const statusCell = document.createElement("td");
    tr.append(markCell, nameCell(item), groupCell(item), imageCell, maskCell, statusCell);
    elements.patientsBody.appendChild(tr);
    state.rows.set(item.id, { item, statusCell, shown: "" });
  }
  elements.patientsSummary.textContent =
    `${snapshot.total} patient${snapshot.total === 1 ? "" : "s"} in this run · ${snapshot.root}`;
}

function updateJob(snapshot) {
  state.job = snapshot;
  for (const item of snapshot.cases) {
    const row = state.rows.get(item.id);
    if (!row || !row.statusCell) {
      continue;
    }
    const detail = jobDetail(item);
    const key = `${item.status}|${detail}`;
    if (row.shown !== key) {
      setStatusCell(row.statusCell, item.status, detail);
      row.shown = key;
    }
  }
  renderJobProgress(snapshot);

  const running = snapshot.state === "running" || snapshot.state === "cancelling";
  setRunningControls(running, snapshot.state);
  if (running) {
    schedulePoll(snapshot.id);
  } else {
    stopPolling();
    renderJobResult(snapshot);
  }
}

function jobDetail(item) {
  switch (item.status) {
    case "running":
      return item.stage || "";
    case "done":
      return [
        `${formatInteger(item.features)} features`,
        item.voxels != null ? `${formatInteger(item.voxels)} VOI voxels` : null,
        item.seconds != null ? `${item.seconds} s` : null,
      ].filter(Boolean).join(" · ");
    case "failed":
    case "cancelled":
      return item.message || "";
    default:
      return "";
  }
}

function renderJobProgress(snapshot) {
  const counts = snapshot.counts || {};
  const done = counts.done || 0;
  const failed = counts.failed || 0;
  const cancelled = counts.cancelled || 0;
  const running = counts.running || 0;
  const finished = done + failed + cancelled;
  const percent = snapshot.total ? Math.round((finished / snapshot.total) * 100) : 0;

  let heading;
  let kind;
  switch (snapshot.state) {
    case "running":
      heading = `Extracting… ${finished} of ${snapshot.total} patients finished`;
      kind = "is-pending";
      break;
    case "cancelling":
      heading = "Cancelling — stopping the running workers…";
      kind = "is-pending";
      break;
    case "finished":
      heading = failed
        ? `Finished with ${failed} failed patient${failed === 1 ? "" : "s"}`
        : `Finished all ${snapshot.total} patients`;
      kind = failed ? "is-pending" : "is-ready";
      break;
    case "cancelled":
      heading = `Cancelled after ${done} patient${done === 1 ? "" : "s"}`;
      kind = "is-pending";
      break;
    default:
      heading = snapshot.message || "The batch stopped.";
      kind = "is-error";
  }
  const active = snapshot.state === "running" || snapshot.state === "cancelling";
  const meta = [
    running ? `${running} running` : null,
    // Once the run ends the heading already names the failures.
    active && failed ? `${failed} failed` : null,
    `${formatDuration(snapshot.elapsed_seconds)} elapsed`,
    snapshot.eta_seconds != null ? `about ${formatDuration(snapshot.eta_seconds)} left` : null,
  ].filter(Boolean).join(" · ");

  const box = elements.jobStatus;
  if (!box.querySelector(".upload-progress")) {
    box.replaceChildren();
    const label = document.createElement("div");
    label.className = "upload-progress-label";
    const text = document.createElement("span");
    text.className = "upload-progress-text";
    // Only the heading is announced; the clock beside it changes every poll.
    const metaSpan = document.createElement("span");
    metaSpan.className = "upload-progress-meta";
    metaSpan.setAttribute("aria-hidden", "true");
    label.append(text, metaSpan);
    const track = document.createElement("div");
    track.className = "progress upload-progress";
    track.setAttribute("role", "progressbar");
    track.setAttribute("aria-label", "Patients finished");
    track.setAttribute("aria-valuemin", "0");
    track.setAttribute("aria-valuemax", "100");
    const bar = document.createElement("div");
    bar.className = "progress-bar";
    track.appendChild(bar);
    const note = document.createElement("div");
    note.className = "batch-note mt-2 d-none";
    box.append(label, track, note);
  }
  box.className = `upload-status mt-3 ${kind}`;
  const text = box.querySelector(".upload-progress-text");
  if (text.textContent !== heading) {
    text.textContent = heading;
  }
  box.querySelector(".upload-progress-meta").textContent = ` · ${meta}`;
  box.querySelector(".upload-progress").setAttribute("aria-valuenow", String(percent));
  box.querySelector(".progress-bar").style.width = `${percent}%`;
  const note = box.querySelector(".batch-note");
  const noteText = snapshot.state === "failed" ? "" : snapshot.message || "";
  if (note.textContent !== noteText) {
    note.textContent = noteText;
  }
  note.classList.toggle("d-none", !noteText);
}

function renderJobResult(snapshot) {
  const box = elements.jobResult;
  box.replaceChildren();
  const counts = snapshot.counts || {};
  const failedIds = snapshot.cases
    .filter((item) => item.status === "failed" || item.status === "cancelled")
    .map((item) => item.id);

  const title = document.createElement("p");
  title.className = "batch-result-title mb-2";
  title.textContent = snapshot.rows_written
    ? `Saved ${snapshot.rows_written} patient${snapshot.rows_written === 1 ? "" : "s"} × `
      + `${formatInteger(snapshot.columns)} features.`
    : "No patient finished, so no feature CSV was written.";
  box.appendChild(title);

  const list = document.createElement("dl");
  list.className = "batch-settings mb-3";
  const files = [
    ["Features", snapshot.rows_written ? snapshot.output_csv : null],
    ["Log", snapshot.log_csv],
    ["Settings", snapshot.settings_json],
  ];
  for (const [term, value] of files) {
    if (!value) {
      continue;
    }
    const dt = document.createElement("dt");
    dt.textContent = term;
    const dd = document.createElement("dd");
    dd.className = "batch-mono";
    dd.textContent = value;
    list.append(dt, dd);
  }
  box.appendChild(list);

  const actions = document.createElement("div");
  actions.className = "d-flex flex-wrap gap-2";
  if (snapshot.rows_written) {
    const download = document.createElement("a");
    download.className = "btn btn-primary";
    download.href = `/api/radiomics-batch/jobs/${snapshot.id}/csv`;
    download.textContent = "Download CSV";
    actions.appendChild(download);
  }
  if (failedIds.length) {
    const retry = document.createElement("button");
    retry.type = "button";
    retry.className = "btn btn-outline-primary";
    retry.textContent = `Retry the ${failedIds.length} unfinished patient${failedIds.length === 1 ? "" : "s"}`;
    retry.addEventListener("click", () => {
      state.retryIds = new Set(failedIds);
      elements.folder.value = snapshot.root;
      scanFolder();
    });
    actions.appendChild(retry);
  }
  const again = document.createElement("button");
  again.type = "button";
  again.className = "btn btn-outline-secondary";
  again.textContent = "New scan";
  again.addEventListener("click", () => {
    elements.folder.value = snapshot.root;
    scanFolder();
  });
  actions.appendChild(again);
  box.appendChild(actions);

  if (counts.failed) {
    const hint = document.createElement("p");
    hint.className = "batch-note mt-3 mb-0";
    hint.textContent = "The reason for each failure is in the table and in the log CSV.";
    box.appendChild(hint);
  }
  box.classList.remove("d-none");
}

function setRunningControls(running, jobState) {
  for (const input of [
    elements.folder,
    elements.maskPattern,
    elements.outputCsv,
    elements.workers,
    elements.label,
    elements.normalizeScale,
  ]) {
    input.disabled = running;
  }
  elements.scanButton.disabled = running;
  elements.cancelButton.classList.toggle("d-none", !running);
  elements.cancelButton.disabled = jobState === "cancelling";
  elements.cancelButton.textContent = jobState === "cancelling" ? "Cancelling…" : "Cancel";
}

function isJobRunning() {
  return state.job?.state === "running" || state.job?.state === "cancelling";
}

function schedulePoll(jobId) {
  stopPolling();
  state.pollTimer = setTimeout(async () => {
    try {
      const response = await fetch(`/api/radiomics-batch/jobs/${jobId}`, { cache: "no-store" });
      const payload = await parseJson(response);
      if (!response.ok) {
        showAlert(payload.message || "Lost track of the batch job.", "warning");
        state.job = null;
        setRunningControls(false);
        return;
      }
      updateJob(payload);
    } catch {
      // The server may be busy or restarting; try again on the next tick.
      schedulePoll(jobId);
    }
  }, POLL_MS);
}

function stopPolling() {
  if (state.pollTimer) {
    clearTimeout(state.pollTimer);
    state.pollTimer = null;
  }
}

function setStatusCell(cell, status, detail) {
  cell.replaceChildren();
  const badge = document.createElement("span");
  badge.className = `batch-status batch-status-${status}`;
  badge.textContent = STATUS_LABELS[status] || status;
  cell.appendChild(badge);
  if (detail) {
    const text = document.createElement("span");
    text.className = "batch-detail";
    text.textContent = detail;
    cell.appendChild(text);
  }
}

function showAlert(message, type) {
  elements.alert.className = `alert alert-${type}`;
  elements.alert.textContent = message;
  elements.alert.classList.remove("d-none");
  elements.alert.scrollIntoView({ behavior: "smooth", block: "center" });
}

function hideAlert() {
  elements.alert.classList.add("d-none");
}

function formatInteger(value) {
  return Number.isFinite(value) ? value.toLocaleString("en-US") : "—";
}

function formatDuration(seconds) {
  if (!Number.isFinite(seconds)) {
    return "—";
  }
  const total = Math.round(seconds);
  if (total < 60) {
    return `${total} s`;
  }
  const minutes = Math.floor(total / 60);
  if (minutes < 60) {
    return `${minutes} min ${String(total % 60).padStart(2, "0")} s`;
  }
  return `${Math.floor(minutes / 60)} h ${String(minutes % 60).padStart(2, "0")} min`;
}

function readStorage(key) {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

function writeStorage(key, value) {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    // Private windows can refuse storage; remembering the folder is optional.
  }
}

async function parseJson(response) {
  try {
    return await response.json();
  } catch {
    return { message: `Server returned HTTP ${response.status}.` };
  }
}
