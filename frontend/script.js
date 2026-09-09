// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function hexToRgb(hex) {
  const h = hex.replace("#", "");
  const full = h.length === 3 ? h.split("").map((c) => c + c).join("") : h;
  const num = parseInt(full, 16);
  return [(num >> 16) & 255, (num >> 8) & 255, num & 255];
}

function mixHex(hexA, hexB, t) {
  const [r1, g1, b1] = hexToRgb(hexA);
  const [r2, g2, b2] = hexToRgb(hexB);
  const r = Math.round(r1 + (r2 - r1) * t);
  const g = Math.round(g1 + (g2 - g1) * t);
  const b = Math.round(b1 + (b2 - b1) * t);
  return `rgb(${r}, ${g}, ${b})`;
}

function sequentialFill(t) {
  // t in [0,1] -> interpolated color between the sequential ramp's light and dark ends
  const light = cssVar("--seq-100");
  const dark = cssVar("--seq-700");
  return mixHex(light, dark, Math.max(0, Math.min(1, t)));
}

function fmtPct(x) {
  return (x * 100).toFixed(1) + "%";
}

function fmtNum(x) {
  return Number(x).toLocaleString();
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// ---------------------------------------------------------------------------
// Shared hover tooltip
// ---------------------------------------------------------------------------

const tooltipEl = document.getElementById("chart-tooltip");

function bindTooltips(root) {
  root.querySelectorAll("[data-tooltip]").forEach((el) => {
    el.addEventListener("mouseenter", () => {
      tooltipEl.innerHTML = el.getAttribute("data-tooltip");
      tooltipEl.hidden = false;
    });
    el.addEventListener("mousemove", (e) => {
      tooltipEl.style.left = e.clientX + "px";
      tooltipEl.style.top = e.clientY + "px";
    });
    el.addEventListener("mouseleave", () => {
      tooltipEl.hidden = true;
    });
  });
}

// ---------------------------------------------------------------------------
// Tab switching
// ---------------------------------------------------------------------------

document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => {
      b.classList.remove("active");
      b.setAttribute("aria-selected", "false");
    });
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    btn.setAttribute("aria-selected", "true");
    document.getElementById(btn.dataset.tab).classList.add("active");
  });
});

// ---------------------------------------------------------------------------
// Meter (accuracy / ROC AUC stat tiles)
// ---------------------------------------------------------------------------

function renderMeter(containerId, ratio) {
  const el = document.getElementById(containerId);
  const color = ratio >= 0.85 ? cssVar("--good") : ratio >= 0.7 ? cssVar("--warning") : cssVar("--critical");
  el.innerHTML = `<div class="meter-fill" style="width:${(ratio * 100).toFixed(1)}%; background:${color};"></div>`;
}

// ---------------------------------------------------------------------------
// Confusion matrix heatmap
// ---------------------------------------------------------------------------

function renderConfusionHeatmap(containerId, labels, matrix) {
  const el = document.getElementById(containerId);
  const max = Math.max(...matrix.flat());
  const total = matrix.flat().reduce((a, b) => a + b, 0);

  const W = 460, H = 250;
  const gridX = 150, gridY = 40, gridW = W - gridX - 20, gridH = H - gridY - 10;
  const cellW = gridW / 2, cellH = gridH / 2;
  const gap = 3;

  let cells = "";
  for (let r = 0; r < 2; r++) {
    for (let c = 0; c < 2; c++) {
      const val = matrix[r][c];
      const t = max > 0 ? val / max : 0;
      const fill = sequentialFill(t);
      const rgbMatch = fill.match(/\d+/g).map(Number);
      const relLum = (0.299 * rgbMatch[0] + 0.587 * rgbMatch[1] + 0.114 * rgbMatch[2]) / 255;
      const textColor = relLum < 0.56 ? "#ffffff" : "#0b0b0b";
      const x = gridX + c * (cellW + gap);
      const y = gridY + r * (cellH + gap);
      const pct = total > 0 ? ((val / total) * 100).toFixed(1) : "0.0";
      const tooltip = `<b>Actual ${labels[r]}</b>, predicted ${labels[c]}<br>${fmtNum(val)} files (${pct}%)`;
      cells += `
        <g data-tooltip="${escapeHtml(tooltip)}" style="cursor:default;">
          <rect x="${x}" y="${y}" width="${cellW}" height="${cellH}" rx="8" fill="${fill}"></rect>
          <text x="${x + cellW / 2}" y="${y + cellH / 2 - 6}" text-anchor="middle" font-size="19" font-weight="700" fill="${textColor}">${fmtNum(val)}</text>
          <text x="${x + cellW / 2}" y="${y + cellH / 2 + 14}" text-anchor="middle" font-size="11" fill="${textColor}" opacity="0.85">${pct}%</text>
        </g>`;
    }
  }

  const colLabels = `
    <text x="${gridX + cellW / 2}" y="${gridY - 14}" text-anchor="middle" font-size="11" fill="var(--text-muted)">Predicted Benign</text>
    <text x="${gridX + cellW + gap + cellW / 2}" y="${gridY - 14}" text-anchor="middle" font-size="11" fill="var(--text-muted)">Predicted Malware</text>`;

  const rowLabels = `
    <text x="${gridX - 12}" y="${gridY + cellH / 2}" text-anchor="end" dominant-baseline="middle" font-size="11" fill="var(--text-muted)">Actual Benign</text>
    <text x="${gridX - 12}" y="${gridY + cellH + gap + cellH / 2}" text-anchor="end" dominant-baseline="middle" font-size="11" fill="var(--text-muted)">Actual Malware</text>`;

  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Confusion matrix heatmap">${colLabels}${rowLabels}${cells}</svg>`;
  bindTooltips(el);
}

// ---------------------------------------------------------------------------
// Grouped bar: precision / recall / f1 by class
// ---------------------------------------------------------------------------

function renderClassMetricsChart(containerId, report) {
  const el = document.getElementById(containerId);
  const metrics = ["precision", "recall", "f1-score"];
  const metricLabels = ["Precision", "Recall", "F1-score"];
  const classes = [
    { key: "0", name: "Benign", color: cssVar("--good") },
    { key: "1", name: "Malware", color: cssVar("--critical") },
  ];

  const W = 460, H = 260;
  const padL = 34, padR = 10, padT = 16, padB = 30;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const groupW = plotW / metrics.length;
  const barW = 26, barGap = 4;

  let gridlines = "", yLabels = "";
  [0, 0.25, 0.5, 0.75, 1].forEach((f) => {
    const y = padT + plotH * (1 - f);
    gridlines += `<line x1="${padL}" y1="${y}" x2="${W - padR}" y2="${y}" stroke="var(--gridline)" stroke-width="1"/>`;
    yLabels += `<text x="${padL - 8}" y="${y + 3}" text-anchor="end" font-size="10" fill="var(--text-muted)">${f}</text>`;
  });

  let bars = "", xLabels = "";
  metrics.forEach((m, i) => {
    const groupCenter = padL + groupW * i + groupW / 2;
    const startX = groupCenter - (classes.length * barW + (classes.length - 1) * barGap) / 2;
    classes.forEach((cls, j) => {
      const val = report[cls.key][m];
      const barH = plotH * val;
      const x = startX + j * (barW + barGap);
      const y = padT + plotH - barH;
      const tooltip = `<b>${cls.name}</b> &mdash; ${metricLabels[i]}<br>${val.toFixed(3)}`;
      bars += `
        <g data-tooltip="${escapeHtml(tooltip)}" style="cursor:default;">
          <rect x="${x}" y="${y}" width="${barW}" height="${barH}" rx="4" fill="${cls.color}"></rect>
          <text x="${x + barW / 2}" y="${y - 6}" text-anchor="middle" font-size="10.5" font-weight="600" fill="var(--text-secondary)">${val.toFixed(2)}</text>
        </g>`;
    });
    xLabels += `<text x="${groupCenter}" y="${H - 8}" text-anchor="middle" font-size="11" fill="var(--text-muted)">${metricLabels[i]}</text>`;
  });

  const legend = classes.map((c) => `<span class="legend-item"><span class="legend-dot" style="background:${c.color}"></span>${c.name}</span>`).join("");

  el.innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Precision, recall and F1 by class">
      ${gridlines}${yLabels}${bars}${xLabels}
    </svg>
    <div class="chart-legend">${legend}</div>`;
  bindTooltips(el);
}

// ---------------------------------------------------------------------------
// Feature importance horizontal bar chart
// ---------------------------------------------------------------------------

function renderFeatureImportanceChart(containerId, importances) {
  const el = document.getElementById(containerId);
  const entries = Object.entries(importances).sort((a, b) => b[1] - a[1]).slice(0, 12);
  const max = Math.max(...entries.map(([, v]) => v));

  const rowH = 26, labelW = 168, padT = 6, padR = 46;
  const W = 620, H = padT + entries.length * rowH + 8;
  const plotW = W - labelW - padR;

  let rows = "";
  entries.forEach(([name, val], i) => {
    const y = padT + i * rowH;
    const barW = max > 0 ? (val / max) * plotW : 0;
    const tooltip = `<b>${escapeHtml(name)}</b><br>importance: ${fmtNum(val)}`;
    rows += `
      <g data-tooltip="${escapeHtml(tooltip)}" style="cursor:default;">
        <text x="${labelW - 10}" y="${y + rowH / 2 + 4}" text-anchor="end" font-size="11.5" fill="var(--text-secondary)">${escapeHtml(name)}</text>
        <rect x="${labelW}" y="${y + 4}" width="${Math.max(barW, 2)}" height="${rowH - 10}" rx="4" fill="var(--series-1)"></rect>
        <text x="${labelW + barW + 8}" y="${y + rowH / 2 + 4}" font-size="11" fill="var(--text-muted)">${fmtNum(val)}</text>
      </g>`;
  });

  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Top feature importances">${rows}</svg>`;
  bindTooltips(el);
}

// ---------------------------------------------------------------------------
// Dashboard
// ---------------------------------------------------------------------------

async function loadDashboard() {
  const statusEl = document.getElementById("dashboard-status");
  const contentEl = document.getElementById("dashboard-content");
  try {
    const res = await fetch("/api/dashboard");
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      statusEl.textContent = body.detail || `Failed to load dashboard (HTTP ${res.status}).`;
      statusEl.classList.add("error");
      return;
    }
    const data = await res.json();
    statusEl.textContent = "";
    contentEl.hidden = false;

    document.getElementById("metric-accuracy").textContent = fmtPct(data.accuracy);
    document.getElementById("metric-auc").textContent = data.roc_auc.toFixed(4);
    document.getElementById("metric-train").textContent = fmtNum(data.n_train);
    document.getElementById("metric-test").textContent = fmtNum(data.n_test);
    renderMeter("meter-accuracy", data.accuracy);
    renderMeter("meter-auc", data.roc_auc);

    renderConfusionHeatmap("confusion-heatmap", data.labels, data.confusion_matrix);
    renderClassMetricsChart("class-metrics-chart", data.classification_report);
    renderFeatureImportanceChart("feature-importance-chart", data.feature_importance);

    renderConfusionTable(data.labels, data.confusion_matrix);
    renderClassificationReportTable(data.classification_report);

    setHeaderStatus(true);
  } catch (err) {
    statusEl.textContent = "Could not reach the API: " + err;
    statusEl.classList.add("error");
    setHeaderStatus(false);
  }
}

function renderConfusionTable(labels, matrix) {
  const table = document.getElementById("confusion-matrix");
  let html = "<caption style='text-align:left;color:var(--text-muted);font-size:0.76rem;margin-bottom:6px;'>Confusion matrix</caption><tr><th></th>";
  labels.forEach((l) => (html += `<th>Predicted ${l}</th>`));
  html += "</tr>";
  matrix.forEach((row, i) => {
    html += `<tr><th>Actual ${labels[i]}</th>`;
    row.forEach((val) => (html += `<td>${fmtNum(val)}</td>`));
    html += "</tr>";
  });
  table.innerHTML = html;
}

function renderClassificationReportTable(report) {
  const table = document.getElementById("classification-report");
  const rows = Object.entries(report).filter(([k]) => k !== "accuracy");
  let html = "<caption style='text-align:left;color:var(--text-muted);font-size:0.76rem;margin-bottom:6px;'>Classification report</caption><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-score</th><th>Support</th></tr>";
  rows.forEach(([key, val]) => {
    if (typeof val !== "object") return;
    html += `<tr><td>${key}</td><td>${val.precision.toFixed(3)}</td><td>${val.recall.toFixed(3)}</td><td>${val["f1-score"].toFixed(3)}</td><td>${fmtNum(val.support)}</td></tr>`;
  });
  table.innerHTML = html;
}

function setHeaderStatus(ok) {
  const el = document.getElementById("header-status");
  if (ok) {
    el.innerHTML = `<span class="dot"></span>Model loaded`;
    el.classList.add("ok");
    el.classList.remove("warn");
  } else {
    el.innerHTML = `<span class="dot"></span>API unreachable`;
    el.classList.add("warn");
    el.classList.remove("ok");
  }
}

// ---------------------------------------------------------------------------
// Live classifier
// ---------------------------------------------------------------------------

const form = document.getElementById("upload-form");
const classifierStatus = document.getElementById("classifier-status");
const progressEl = document.getElementById("upload-progress");
const resultPanel = document.getElementById("result-panel");
const classifyBtn = document.getElementById("classify-btn");
const fileInput = document.getElementById("file-input");
const fileDrop = document.getElementById("file-drop");
const fileDropLabel = document.getElementById("file-drop-label");

fileInput.addEventListener("change", () => {
  fileDropLabel.textContent = fileInput.files.length ? fileInput.files[0].name : "Choose a file or drag it here";
});

["dragover", "dragleave", "drop"].forEach((evt) => {
  fileDrop.addEventListener(evt, (e) => {
    e.preventDefault();
    if (evt === "dragover") fileDrop.classList.add("drag-over");
    else fileDrop.classList.remove("drag-over");
    if (evt === "drop" && e.dataTransfer.files.length) {
      fileInput.files = e.dataTransfer.files;
      fileDropLabel.textContent = e.dataTransfer.files[0].name;
    }
  });
});

async function checkModelStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    if (!data.model_loaded) {
      classifierStatus.textContent =
        "No trained model found. Run backend/train_and_save_model.py first, then reload this page.";
      classifierStatus.classList.add("error");
      classifyBtn.disabled = true;
    }
  } catch (err) {
    classifierStatus.textContent = "Could not reach the API: " + err;
    classifierStatus.classList.add("error");
  }
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!fileInput.files.length) return;

  classifyBtn.disabled = true;
  progressEl.hidden = false;
  resultPanel.hidden = true;
  classifierStatus.textContent = "";
  classifierStatus.classList.remove("error");

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  try {
    const res = await fetch("/api/predict", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
    renderResult(data);
  } catch (err) {
    classifierStatus.textContent = "Prediction failed: " + err.message;
    classifierStatus.classList.add("error");
  } finally {
    progressEl.hidden = true;
    classifyBtn.disabled = false;
  }
});

function renderResult(data) {
  resultPanel.hidden = false;
  const badge = document.getElementById("result-badge");
  badge.textContent = data.prediction;
  badge.className = "result-badge " + (data.prediction === "Malware" ? "malware" : "benign");
  document.getElementById("result-filename").textContent = data.filename;

  const malwarePct = data.malware_probability * 100;
  const benignPct = data.benign_probability * 100;
  const splitBar = document.getElementById("result-split-bar");
  splitBar.innerHTML = `
    <div class="split-bar-seg split-bar-seg--benign" style="width:${benignPct}%">${benignPct >= 12 ? fmtPct(data.benign_probability) : ""}</div>
    <div class="split-bar-seg split-bar-seg--malware" style="width:${malwarePct}%">${malwarePct >= 12 ? fmtPct(data.malware_probability) : ""}</div>`;

  document.getElementById("result-malware-prob").textContent = fmtPct(data.malware_probability);
  document.getElementById("result-benign-prob").textContent = fmtPct(data.benign_probability);
  document.getElementById("result-file-size").textContent = fmtNum(data.file_size) + " B";
  document.getElementById("result-is-pe").textContent = data.is_pe ? "Yes" : "No";
  document.getElementById("result-dlls").textContent = fmtNum(data.features.num_imported_dlls);
  document.getElementById("result-functions").textContent = fmtNum(data.features.num_imported_functions);

  const table = document.getElementById("result-features");
  let html = "<tr><th>Feature</th><th>Value</th></tr>";
  Object.entries(data.features).forEach(([k, v]) => {
    html += `<tr><td>${k}</td><td>${typeof v === "number" ? v.toLocaleString(undefined, { maximumFractionDigits: 4 }) : v}</td></tr>`;
  });
  table.innerHTML = html;
}

loadDashboard();
checkModelStatus();
