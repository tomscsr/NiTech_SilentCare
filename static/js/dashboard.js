/**
 * SilentCare Dashboard - JavaScript
 * Real-time updates via SSE, Chart.js history, alert management.
 */

// ============================================
// State
// ============================================
let isRunning = false;
let historyChart = null;
let eventSource = null;
let audioInterval = null;
const MAX_CHART_POINTS = 60;
const CLASSES = ["DISTRESS", "ANGRY", "ALERT", "CALM"];
const ICONS = { DISTRESS: "!!!", ANGRY: "!!", ALERT: "!", CALM: "--" };
const COLORS = {
    DISTRESS: "#f87171",
    ANGRY: "#fb923c",
    ALERT: "#facc15",
    CALM: "#4ade80",
};

// Chart data buffers
const chartData = {
    labels: [],
    distress: [],
    angry: [],
    alert: [],
    calm: [],
};

// ============================================
// Init
// ============================================
document.addEventListener("DOMContentLoaded", () => {
    initChart();
    initWaveformCanvas();
    loadAudioDevices();
    pollStatus();
    setInterval(pollStatus, 5000);
    loadFeedbackStats();
});

// ============================================
// Chart.js
// ============================================
function initChart() {
    const ctx = document.getElementById("history-chart").getContext("2d");
    historyChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: chartData.labels,
            datasets: [
                {
                    label: "DISTRESS",
                    data: chartData.distress,
                    borderColor: COLORS.DISTRESS,
                    backgroundColor: "rgba(248,113,113,0.06)",
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                    borderWidth: 1.5,
                },
                {
                    label: "ANGRY",
                    data: chartData.angry,
                    borderColor: COLORS.ANGRY,
                    backgroundColor: "rgba(251,146,60,0.06)",
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                    borderWidth: 1.5,
                },
                {
                    label: "ALERT",
                    data: chartData.alert,
                    borderColor: COLORS.ALERT,
                    backgroundColor: "rgba(250,204,21,0.04)",
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                    borderWidth: 1.5,
                },
                {
                    label: "CALM",
                    data: chartData.calm,
                    borderColor: COLORS.CALM,
                    backgroundColor: "rgba(74,222,128,0.06)",
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                    borderWidth: 1.5,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 300 },
            scales: {
                x: {
                    display: true,
                    grid: { color: "rgba(37,40,56,0.5)", lineWidth: 0.5 },
                    ticks: { color: "#5d5f72", maxTicksLimit: 10, font: { size: 10 } },
                    border: { color: "rgba(37,40,56,0.5)" },
                },
                y: {
                    min: 0,
                    max: 1,
                    grid: { color: "rgba(37,40,56,0.5)", lineWidth: 0.5 },
                    ticks: { color: "#5d5f72", font: { size: 10 } },
                    border: { color: "rgba(37,40,56,0.5)" },
                },
            },
            plugins: {
                legend: {
                    labels: { color: "#9496a8", boxWidth: 10, padding: 16, font: { size: 11, weight: 500 } },
                },
            },
        },
    });
}

function addChartPoint(probs) {
    const now = new Date();
    const label = now.toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit", second: "2-digit" });

    chartData.labels.push(label);
    chartData.distress.push(probs[0] || 0);
    chartData.angry.push(probs[1] || 0);
    chartData.alert.push(probs[2] || 0);
    chartData.calm.push(probs[3] || 0);

    if (chartData.labels.length > MAX_CHART_POINTS) {
        chartData.labels.shift();
        chartData.distress.shift();
        chartData.angry.shift();
        chartData.alert.shift();
        chartData.calm.shift();
    }

    historyChart.update("none");
}

// ============================================
// SSE
// ============================================
function connectSSE() {
    if (eventSource) return;
    eventSource = new EventSource("/api/stream");

    eventSource.addEventListener("segment", (e) => {
        const data = JSON.parse(e.data);
        updateDisplay(data);
        addSegmentToList(data);
        refreshStats();
    });

    eventSource.addEventListener("alert", (e) => {
        const data = JSON.parse(e.data);
        addAlertToFeed(data);
        refreshStats();
    });

    eventSource.onerror = () => {
        disconnectSSE();
        setTimeout(() => {
            if (isRunning) connectSSE();
        }, 3000);
    };
}

function disconnectSSE() {
    if (eventSource) {
        eventSource.close();
        eventSource = null;
    }
}

// ============================================
// Display updates
// ============================================
function updateDisplay(result) {
    const cls = result.predicted_class || "CALM";
    const conf = result.confidence || 0;
    const probs = result.fused_probs || [0, 0, 0, 0];
    const audioProbs = result.audio_probs || null;
    const videoProbs = result.video_probs || null;
    const streak = result.streak || { class: "--", count: 0 };

    const display = document.getElementById("emotion-display");
    display.className = "emotion-display emotion-" + cls.toLowerCase();
    document.getElementById("emotion-icon").textContent = ICONS[cls] || "--";
    document.getElementById("emotion-label").textContent = cls;
    document.getElementById("emotion-confidence").textContent = (conf * 100).toFixed(1) + "%";

    document.getElementById("streak-count").textContent = streak.count || 0;
    document.getElementById("streak-class").textContent = streak.class || "--";

    // Fused bars
    updateBar("distress", probs[0]);
    updateBar("angry", probs[1]);
    updateBar("alert", probs[2]);
    updateBar("calm", probs[3]);

    // Audio bars
    if (audioProbs) {
        updateBar("audio-distress", audioProbs[0]);
        updateBar("audio-angry", audioProbs[1]);
        updateBar("audio-alert", audioProbs[2]);
        updateBar("audio-calm", audioProbs[3]);
    } else {
        resetBars("audio");
    }

    // Video bars
    const noFaceEl = document.getElementById("video-no-face");
    const videoContainer = document.getElementById("video-probs-container");
    if (videoProbs) {
        noFaceEl.style.display = "none";
        videoContainer.style.opacity = "1";
        updateBar("video-distress", videoProbs[0]);
        updateBar("video-angry", videoProbs[1]);
        updateBar("video-alert", videoProbs[2]);
        updateBar("video-calm", videoProbs[3]);
    } else {
        noFaceEl.style.display = "flex";
        videoContainer.style.opacity = "0.2";
        resetBars("video");
    }

    addChartPoint(probs);
}

function updateBar(name, value) {
    const pct = ((value || 0) * 100).toFixed(1);
    const bar = document.getElementById("bar-" + name);
    const val = document.getElementById("val-" + name);
    if (bar) bar.style.width = pct + "%";
    if (val) val.textContent = pct + "%";
}

function resetBars(prefix) {
    const names = ["distress", "angry", "alert", "calm"];
    for (const n of names) {
        const bar = document.getElementById("bar-" + prefix + "-" + n);
        const val = document.getElementById("val-" + prefix + "-" + n);
        if (bar) bar.style.width = "0%";
        if (val) val.textContent = "--";
    }
}

// ============================================
// Alert feed
// ============================================
function addAlertToFeed(alert) {
    const feed = document.getElementById("alert-feed");

    const empty = feed.querySelector(".alert-empty");
    if (empty) empty.remove();

    const time = new Date().toLocaleTimeString("fr-FR", {
        hour: "2-digit", minute: "2-digit", second: "2-digit",
    });

    const div = document.createElement("div");
    div.className = `alert-item severity-${alert.severity}`;
    div.innerHTML = `
        <span class="alert-severity ${alert.severity}">${alert.severity}</span>
        <span class="alert-emotion" style="color:${COLORS[alert.emotion]}">${alert.emotion}</span>
        <span class="alert-confidence">${(alert.confidence * 100).toFixed(1)}%</span>
        <span class="alert-time">${time}</span>
        <button class="alert-ack-btn" onclick="acknowledgeAlert(this, ${alert.id || 0})">ACK</button>
        <button class="alert-report-btn" onclick="openReportModal('alert', ${alert.id || 0}, '${alert.emotion}')">Report</button>
    `;

    feed.insertBefore(div, feed.firstChild);

    while (feed.children.length > 50) {
        feed.removeChild(feed.lastChild);
    }

    refreshUnackCount();
}

function acknowledgeAlert(btn, alertId) {
    if (alertId > 0) {
        fetch(`/api/alerts/${alertId}/ack`, { method: "POST" });
    }
    btn.closest(".alert-item").classList.add("acknowledged");
    btn.disabled = true;
    btn.textContent = "OK";
    refreshUnackCount();
}

function refreshUnackCount() {
    const items = document.querySelectorAll(".alert-item:not(.acknowledged)");
    const badge = document.getElementById("unack-count");
    badge.textContent = items.length;
    badge.style.display = items.length > 0 ? "inline" : "none";
}

// ============================================
// Status polling
// ============================================
async function pollStatus() {
    try {
        const resp = await fetch("/api/status");
        const data = await resp.json();

        isRunning = data.running;

        const badge = document.getElementById("status-badge");
        badge.className = "badge " + (isRunning ? "badge-active" : "badge-inactive");
        badge.textContent = isRunning ? "MONITORING" : "INACTIVE";

        document.getElementById("btn-start").disabled = isRunning;
        document.getElementById("btn-stop").disabled = !isRunning;

        const audioEl = document.getElementById("audio-status");
        audioEl.className = "modality-dot " + (data.audio_enabled ? "on" : "off");

        const videoEl = document.getElementById("video-status");
        videoEl.className = "modality-dot " + (data.video_enabled ? "on" : "off");

        if (isRunning && !eventSource) {
            connectSSE();
            startLiveFeeds();
        }
        if (!isRunning && eventSource) {
            disconnectSSE();
            stopLiveFeeds();
        }

        if (data.latest_result) {
            updateDisplay(data.latest_result);
        }

        refreshStats();
    } catch (e) {}
}

async function refreshStats() {
    try {
        const resp = await fetch("/api/stats");
        const data = await resp.json();

        document.getElementById("stat-segments").textContent = data.total_segments || 0;
        document.getElementById("stat-alerts").textContent = data.total_alerts || 0;
        document.getElementById("stat-distress").textContent =
            (data.alerts_by_emotion && data.alerts_by_emotion.DISTRESS) || 0;
        document.getElementById("stat-angry").textContent =
            (data.alerts_by_emotion && data.alerts_by_emotion.ANGRY) || 0;
    } catch (e) {}
}

// ============================================
// Controls
// ============================================
async function startMonitoring() {
    try {
        document.getElementById("btn-start").disabled = true;
        const resp = await fetch("/api/start", { method: "POST" });
        const data = await resp.json();
        if (data.status === "started" || data.status === "already_running") {
            connectSSE();
            startLiveFeeds();
        }
        pollStatus();
    } catch (e) {
        document.getElementById("btn-start").disabled = false;
    }
}

async function stopMonitoring() {
    try {
        document.getElementById("btn-stop").disabled = true;
        await fetch("/api/stop", { method: "POST" });
        disconnectSSE();
        stopLiveFeeds();
        pollStatus();
    } catch (e) {
        document.getElementById("btn-stop").disabled = false;
    }
}

// ============================================
// Live video feed (MJPEG)
// ============================================
function startVideoFeed() {
    const img = document.getElementById("video-feed");
    const overlay = document.getElementById("video-overlay");
    img.src = "/api/video_feed";
    // For MJPEG, the browser handles frames natively.
    // Hide overlay once first frame loads.
    img.onload = () => { overlay.classList.add("hidden"); };
}

function stopVideoFeed() {
    const img = document.getElementById("video-feed");
    img.onload = null;
    img.src = "";
    document.getElementById("video-overlay").classList.remove("hidden");
}

// ============================================
// Live audio waveform (oscilloscope)
// ============================================
function initWaveformCanvas() {
    const canvas = document.getElementById("waveform-canvas");
    // Match CSS rendered size at 2x for sharpness
    const rect = canvas.getBoundingClientRect();
    canvas.width = (rect.width || 600) * 2;
    canvas.height = (rect.height || 200) * 2;
}

function startAudioFeed() {
    if (audioInterval) return;
    audioInterval = setInterval(fetchAndDrawWaveform, 120);
}

function stopAudioFeed() {
    if (audioInterval) {
        clearInterval(audioInterval);
        audioInterval = null;
    }
    document.getElementById("audio-overlay").classList.remove("hidden");
    const canvas = document.getElementById("waveform-canvas");
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

async function fetchAndDrawWaveform() {
    try {
        const resp = await fetch("/api/audio_data");
        const data = await resp.json();

        if (!data.waveform || data.waveform.length === 0) return;

        document.getElementById("audio-overlay").classList.add("hidden");
        drawWaveform(data.waveform);
    } catch (e) {}
}

function drawWaveform(samples) {
    const canvas = document.getElementById("waveform-canvas");
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    const mid = h / 2;

    ctx.clearRect(0, 0, w, h);

    // Grid lines
    ctx.strokeStyle = "rgba(30,33,48,0.5)";
    ctx.lineWidth = 0.5;
    for (let y = 0; y < h; y += h / 4) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
    }

    // Center line
    ctx.strokeStyle = "rgba(37,40,56,0.8)";
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(0, mid);
    ctx.lineTo(w, mid);
    ctx.stroke();

    // Find peak for auto-scaling
    let peak = 0;
    for (let i = 0; i < samples.length; i++) {
        const a = Math.abs(samples[i]);
        if (a > peak) peak = a;
    }
    if (peak < 0.001) peak = 0.001;  // avoid division by ~0
    const scale = (mid - 4) / peak;

    // Waveform line
    ctx.strokeStyle = "#818cf8";
    ctx.lineWidth = 1.5;
    ctx.beginPath();

    const step = w / samples.length;
    for (let i = 0; i < samples.length; i++) {
        const x = i * step;
        const y = mid - samples[i] * scale;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();
}

// ============================================
// Audio device selection
// ============================================
async function loadAudioDevices() {
    try {
        const resp = await fetch("/api/audio_devices");
        const data = await resp.json();
        const select = document.getElementById("mic-select");

        // Clear existing options
        select.innerHTML = '<option value="">-- Microphone --</option>';

        for (const dev of data.devices) {
            const opt = document.createElement("option");
            opt.value = dev.id;
            opt.textContent = dev.name;
            if (data.current !== null && dev.id === data.current) {
                opt.selected = true;
            }
            select.appendChild(opt);
        }
    } catch (e) {}
}

async function changeAudioDevice(deviceId) {
    if (deviceId === "") return;
    try {
        await fetch("/api/audio_devices", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ device_id: parseInt(deviceId) }),
        });
    } catch (e) {}
}

// ============================================
// Live feed management
// ============================================
function startLiveFeeds() {
    startVideoFeed();
    startAudioFeed();
}

function stopLiveFeeds() {
    stopVideoFeed();
    stopAudioFeed();
}

// ============================================
// Segment history list
// ============================================
const recentSegments = [];
const MAX_SEGMENT_LIST = 20;

function addSegmentToList(segment) {
    recentSegments.unshift(segment);
    if (recentSegments.length > MAX_SEGMENT_LIST) {
        recentSegments.pop();
    }
    renderSegmentList();
}

function renderSegmentList() {
    const list = document.getElementById("segment-list");
    if (recentSegments.length === 0) {
        list.innerHTML = '<div class="segment-empty">No segments yet.</div>';
        return;
    }

    list.innerHTML = recentSegments.map(seg => {
        const time = new Date(seg.timestamp * 1000).toLocaleTimeString("fr-FR", {
            hour: "2-digit", minute: "2-digit", second: "2-digit",
        });
        const cls = seg.predicted_class || "CALM";
        const conf = ((seg.confidence || 0) * 100).toFixed(1);
        const segId = seg.segment_id || 0;
        return `
            <div class="segment-item">
                <span class="segment-time">${time}</span>
                <span class="segment-class" style="color:${COLORS[cls] || '#fff'}">${cls}</span>
                <span class="segment-conf">${conf}%</span>
                <button class="segment-report-btn" onclick="openReportModal('segment', ${segId}, '${cls}')">Report</button>
            </div>
        `;
    }).join("");
}

// ============================================
// Report modal
// ============================================
let currentReportContext = null;

function openReportModal(source, id, predictedClass) {
    currentReportContext = { source, id, predictedClass };

    const modal = document.getElementById("report-modal");
    const title = document.getElementById("modal-title");
    const typeSection = document.getElementById("modal-type-section");
    const status = document.getElementById("modal-status");

    status.textContent = "";
    status.className = "modal-status";
    document.getElementById("report-notes").value = "";
    document.getElementById("modal-confirm").disabled = false;

    // Build radio options based on source
    const radioGroup = typeSection.querySelector(".radio-group");
    if (source === "alert") {
        title.textContent = "Report Alert Error";
        radioGroup.innerHTML = `
            <label><input type="radio" name="report-type" value="FALSE_ALERT" checked> False Alert (nothing happened)</label>
            <label><input type="radio" name="report-type" value="WRONG_CLASSIFICATION"> Wrong Classification</label>
        `;
        document.getElementById("correct-class").value = "CALM";
    } else {
        title.textContent = "Report Segment Error";
        const isCalmPrediction = predictedClass === "CALM";
        radioGroup.innerHTML = `
            <label><input type="radio" name="report-type" value="MISSED_DETECTION"${isCalmPrediction ? " checked" : ""}> Missed Detection (should have alerted)</label>
            <label><input type="radio" name="report-type" value="WRONG_CLASSIFICATION"${!isCalmPrediction ? " checked" : ""}> Wrong Classification</label>
        `;
        document.getElementById("correct-class").value = isCalmPrediction ? "DISTRESS" : "CALM";
    }
    typeSection.style.display = "block";

    modal.style.display = "flex";
}

function closeReportModal() {
    document.getElementById("report-modal").style.display = "none";
    currentReportContext = null;
}

async function submitFeedback() {
    if (!currentReportContext) return;

    const correctClass = document.getElementById("correct-class").value;
    const notes = document.getElementById("report-notes").value.trim() || null;
    const status = document.getElementById("modal-status");
    const confirmBtn = document.getElementById("modal-confirm");

    confirmBtn.disabled = true;
    status.textContent = "Saving...";
    status.className = "modal-status saving";

    let url, body;

    const reportType = document.querySelector('input[name="report-type"]:checked').value;

    if (currentReportContext.source === "segment") {
        if (reportType === "MISSED_DETECTION") {
            url = "/api/feedback/missed_detection";
            body = {
                segment_id: currentReportContext.id,
                correct_class: correctClass,
                notes: notes,
            };
        } else {
            url = "/api/feedback/wrong_classification";
            body = {
                segment_id: currentReportContext.id,
                correct_class: correctClass,
                notes: notes,
            };
        }
    } else {
        if (reportType === "FALSE_ALERT") {
            url = "/api/feedback/false_alert";
            body = {
                alert_id: currentReportContext.id,
                correct_class: correctClass,
                notes: notes,
            };
        } else {
            url = "/api/feedback/wrong_classification";
            body = {
                alert_id: currentReportContext.id,
                correct_class: correctClass,
                notes: notes,
            };
        }
    }

    try {
        const resp = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
        const data = await resp.json();

        if (resp.ok) {
            status.textContent = "Saved (ID: " + data.feedback_id + ")";
            status.className = "modal-status success";
            setTimeout(closeReportModal, 1500);
            loadFeedbackStats();
        } else {
            status.textContent = "Error: " + (data.error || "Unknown");
            status.className = "modal-status error";
            confirmBtn.disabled = false;
        }
    } catch (e) {
        status.textContent = "Network error";
        status.className = "modal-status error";
        confirmBtn.disabled = false;
    }
}

// ============================================
// Feedback statistics
// ============================================
async function loadFeedbackStats() {
    try {
        const resp = await fetch("/api/feedback/stats");
        const data = await resp.json();

        document.getElementById("fb-false-alerts").textContent =
            (data.by_type && data.by_type.FALSE_ALERT) || 0;
        document.getElementById("fb-missed").textContent =
            (data.by_type && data.by_type.MISSED_DETECTION) || 0;
        document.getElementById("fb-wrong").textContent =
            (data.by_type && data.by_type.WRONG_CLASSIFICATION) || 0;
        document.getElementById("fb-ready").textContent =
            data.ready_for_training || 0;

        const confList = document.getElementById("fb-confusion-list");
        if (data.confusions && data.confusions.length > 0) {
            confList.innerHTML = data.confusions.slice(0, 3).map(c =>
                `<div class="confusion-item">
                    <span style="color:${COLORS[c.predicted] || '#fff'}">${c.predicted}</span>
                    <span class="confusion-arrow">-></span>
                    <span style="color:${COLORS[c.correct] || '#fff'}">${c.correct}</span>
                    <span class="confusion-count">${c.count}x</span>
                </div>`
            ).join("");
        } else {
            confList.innerHTML = '<div class="confusion-empty">No feedback yet</div>';
        }
    } catch (e) {}
}

function exportFeedbackCSV() {
    window.location.href = "/api/feedback/export";
}
