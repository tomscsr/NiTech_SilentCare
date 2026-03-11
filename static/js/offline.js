/* ==============================================
   SilentCare - Offline Test Page JS
   ============================================== */

// --- State ---
let jobId = null;
let videoUrl = null;
let analysisMode = "realtime";
let eventSource = null;
let pollTimer = null;
let segments = [];
let alerts = [];
let segmentCount = 0;
let alertCount = 0;
let distressCount = 0;
let angryCount = 0;

const COLORS = {
    DISTRESS: "#f87171",
    ANGRY: "#fb923c",
    ALERT: "#facc15",
    CALM: "#4ade80",
};

const ICONS = {
    DISTRESS: "!!",
    ANGRY: "!",
    ALERT: "?!",
    CALM: "OK",
};

// --- Chart setup ---
let historyChart = null;
const chartData = { DISTRESS: [], ANGRY: [], ALERT: [], CALM: [] };
const chartLabels = [];

function initChart() {
    const ctx = document.getElementById("history-chart");
    if (!ctx) return;

    historyChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: chartLabels,
            datasets: [
                { label: "DISTRESS", data: chartData.DISTRESS, borderColor: COLORS.DISTRESS, borderWidth: 1.5, tension: 0.4, pointRadius: 0, fill: false },
                { label: "ANGRY", data: chartData.ANGRY, borderColor: COLORS.ANGRY, borderWidth: 1.5, tension: 0.4, pointRadius: 0, fill: false },
                { label: "ALERT", data: chartData.ALERT, borderColor: COLORS.ALERT, borderWidth: 1.5, tension: 0.4, pointRadius: 0, fill: false },
                { label: "CALM", data: chartData.CALM, borderColor: COLORS.CALM, borderWidth: 1.5, tension: 0.4, pointRadius: 0, fill: false },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 200 },
            scales: {
                x: {
                    ticks: { color: "#5d5f72", font: { size: 10, family: "JetBrains Mono" } },
                    grid: { color: "rgba(37,40,56,0.5)", lineWidth: 0.5 },
                },
                y: {
                    min: 0, max: 1,
                    ticks: { color: "#5d5f72", font: { size: 10, family: "JetBrains Mono" }, stepSize: 0.25 },
                    grid: { color: "rgba(37,40,56,0.5)", lineWidth: 0.5 },
                },
            },
            plugins: {
                legend: {
                    labels: { color: "#9496a8", font: { size: 11 }, boxWidth: 10, padding: 12 },
                },
            },
        },
    });
}

// --- Upload ---
function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    document.getElementById("upload-zone").classList.add("dragover");
}

function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    document.getElementById("upload-zone").classList.remove("dragover");
}

function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    document.getElementById("upload-zone").classList.remove("dragover");

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        uploadFile(files[0]);
    }
}

function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        uploadFile(files[0]);
    }
}

function uploadFile(file) {
    const allowedExts = ["mp4", "avi", "mov", "mkv", "webm"];
    const ext = file.name.split(".").pop().toLowerCase();
    if (!allowedExts.includes(ext)) {
        alert("Unsupported format. Use MP4, AVI, MOV, MKV or WEBM.");
        return;
    }

    const zone = document.getElementById("upload-zone");
    const progress = document.getElementById("upload-progress");
    const progressFill = document.getElementById("upload-progress-fill");
    const progressText = document.getElementById("upload-progress-text");

    progress.classList.add("active");
    progressText.textContent = "Uploading...";
    progressFill.style.width = "0%";

    const formData = new FormData();
    formData.append("file", file);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/offline/upload");

    xhr.upload.onprogress = function(e) {
        if (e.lengthComputable) {
            const pct = Math.round(100 * e.loaded / e.total);
            progressFill.style.width = pct + "%";
            progressText.textContent = `Uploading... ${pct}%`;
        }
    };

    xhr.onload = function() {
        progress.classList.remove("active");

        if (xhr.status === 200) {
            const data = JSON.parse(xhr.responseText);
            jobId = data.job_id;
            videoUrl = data.video_url;

            // Show file info
            zone.classList.add("has-file");
            zone.style.display = "none";
            const info = document.getElementById("file-info");
            info.classList.add("active");
            document.getElementById("file-name").textContent = data.filename;

            const vi = data.info;
            document.getElementById("file-details").textContent =
                `${vi.duration}s | ${vi.width}x${vi.height} | ${vi.fps} fps | ${vi.total_segments} segments` +
                (vi.has_audio ? " | audio" : " | no audio");

            // Show video player
            const player = document.getElementById("video-player");
            const placeholder = document.getElementById("video-placeholder");
            player.src = videoUrl;
            player.style.display = "block";
            placeholder.style.display = "none";

            // Enable analyze button
            document.getElementById("btn-analyze").disabled = false;
        } else {
            let msg = "Upload failed";
            try { msg = JSON.parse(xhr.responseText).error || msg; } catch(e) {}
            alert(msg);
        }
    };

    xhr.onerror = function() {
        progress.classList.remove("active");
        alert("Upload failed: network error");
    };

    xhr.send(formData);
}

function removeFile() {
    jobId = null;
    videoUrl = null;

    // Reset upload zone
    const zone = document.getElementById("upload-zone");
    zone.classList.remove("has-file");
    zone.style.display = "";
    document.getElementById("file-info").classList.remove("active");
    document.getElementById("file-input").value = "";

    // Reset video player
    const player = document.getElementById("video-player");
    player.src = "";
    player.style.display = "none";
    document.getElementById("video-placeholder").style.display = "";

    // Disable analyze
    document.getElementById("btn-analyze").disabled = true;

    // Hide results
    document.getElementById("results-section").style.display = "none";

    // Reset state
    resetResults();
}

// --- Mode selection ---
function setMode(mode) {
    analysisMode = mode;
    document.getElementById("mode-realtime").classList.toggle("active", mode === "realtime");
    document.getElementById("mode-complete").classList.toggle("active", mode === "complete");
}

// --- Analysis control ---
function startAnalysis() {
    if (!jobId) return;

    resetResults();

    // Show results section
    document.getElementById("results-section").style.display = "";
    if (!historyChart) initChart();

    // Show progress
    const progress = document.getElementById("analysis-progress");
    progress.classList.add("active");
    updateAnalysisStatus("running", 0);

    // Update buttons
    document.getElementById("btn-analyze").disabled = true;
    document.getElementById("btn-pause").disabled = false;
    document.getElementById("btn-stop").disabled = false;
    document.getElementById("status-badge").className = "badge badge-active";
    document.getElementById("status-badge").textContent = "Analyzing";

    // Start analysis
    fetch(`/api/offline/analyze/${jobId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode: analysisMode }),
    })
    .then(r => r.json())
    .then(data => {
        if (data.error) {
            alert(data.error);
            resetControls();
            return;
        }

        if (analysisMode === "realtime") {
            startSSE();
        } else {
            startPolling();
        }
    })
    .catch(err => {
        alert("Failed to start analysis: " + err);
        resetControls();
    });
}

function pauseAnalysis() {
    if (!jobId) return;
    fetch(`/api/offline/control/${jobId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "pause" }),
    }).then(() => {
        document.getElementById("btn-pause").disabled = true;
        document.getElementById("btn-pause").style.display = "none";
        document.getElementById("btn-resume").disabled = false;
        document.getElementById("btn-resume").style.display = "";
        document.getElementById("status-badge").textContent = "Paused";
    });
}

function resumeAnalysis() {
    if (!jobId) return;
    fetch(`/api/offline/control/${jobId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "resume" }),
    }).then(() => {
        document.getElementById("btn-resume").disabled = true;
        document.getElementById("btn-resume").style.display = "none";
        document.getElementById("btn-pause").disabled = false;
        document.getElementById("btn-pause").style.display = "";
        document.getElementById("status-badge").textContent = "Analyzing";
    });
}

function stopAnalysis() {
    if (!jobId) return;
    fetch(`/api/offline/control/${jobId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "stop" }),
    }).then(() => {
        stopSSE();
        stopPolling();
        resetControls();
        document.getElementById("status-badge").textContent = "Stopped";
        document.getElementById("status-badge").className = "badge badge-inactive";
    });
}

// --- SSE (realtime mode) ---
function startSSE() {
    stopSSE();
    eventSource = new EventSource(`/api/offline/stream/${jobId}`);

    eventSource.addEventListener("segment", function(e) {
        const data = JSON.parse(e.data);
        handleSegmentResult(data);
    });

    eventSource.addEventListener("alert", function(e) {
        const data = JSON.parse(e.data);
        handleAlertResult(data);
    });

    eventSource.addEventListener("done", function(e) {
        stopSSE();
        onAnalysisComplete();
    });

    eventSource.onerror = function() {
        // Check if analysis is done
        checkStatus();
    };
}

function stopSSE() {
    if (eventSource) {
        eventSource.close();
        eventSource = null;
    }
}

// --- Polling (complete mode) ---
function startPolling() {
    stopPolling();
    pollTimer = setInterval(pollStatus, 1500);
}

function stopPolling() {
    if (pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
    }
}

function pollStatus() {
    if (!jobId) return;

    fetch(`/api/offline/status/${jobId}`)
        .then(r => r.json())
        .then(data => {
            if (data.progress) {
                updateAnalysisStatus(data.status, data.progress.percent, data.progress);
            }

            if (data.status === "complete") {
                stopPolling();
                fetchFullResults();
            } else if (data.status === "error") {
                stopPolling();
                updateAnalysisStatus("error", data.progress ? data.progress.percent : 0);
                resetControls();
            }
        });
}

function checkStatus() {
    if (!jobId) return;
    fetch(`/api/offline/status/${jobId}`)
        .then(r => r.json())
        .then(data => {
            if (data.status === "complete") {
                onAnalysisComplete();
            }
        });
}

function fetchFullResults() {
    fetch(`/api/offline/results/${jobId}`)
        .then(r => r.json())
        .then(data => {
            // Process all segments at once
            if (data.segments) {
                data.segments.forEach(seg => handleSegmentResult(seg));
            }
            if (data.alerts) {
                data.alerts.forEach(a => handleAlertResult(a));
            }
            onAnalysisComplete();
        });
}

function onAnalysisComplete() {
    resetControls();
    document.getElementById("status-badge").textContent = "Complete";
    document.getElementById("status-badge").className = "badge badge-active";

    const statusEl = document.getElementById("analysis-status");
    statusEl.textContent = "Analysis complete";
    statusEl.className = "analysis-status-text complete";
    document.getElementById("analysis-percent").textContent = "100%";
    document.getElementById("analysis-fill").style.width = "100%";
}

// --- Handle results ---
function handleSegmentResult(data) {
    segments.push(data);
    segmentCount++;

    // Update emotion hero
    const cls = data.predicted_class;
    const conf = data.confidence;

    const display = document.getElementById("emotion-display");
    display.className = "emotion-display emotion-" + cls.toLowerCase();
    document.getElementById("emotion-icon").textContent = ICONS[cls] || "--";
    document.getElementById("emotion-label").textContent = cls;
    document.getElementById("emotion-confidence").textContent = (conf * 100).toFixed(1) + "%";

    // Update streak
    if (data.streak) {
        document.getElementById("streak-count").textContent = data.streak.count || 0;
        document.getElementById("streak-class").textContent = data.streak.class || "--";
    }

    // Update probability bars
    if (data.fused_probs) {
        updateProbBars("", data.fused_probs);
    }
    if (data.audio_probs) {
        updateProbBars("audio-", data.audio_probs);
    } else {
        resetProbBars("audio-");
    }
    if (data.video_probs) {
        updateProbBars("video-", data.video_probs);
    } else {
        resetProbBars("video-");
    }

    // Stats
    document.getElementById("stat-segments").textContent = segmentCount;
    if (cls === "DISTRESS") distressCount++;
    if (cls === "ANGRY") angryCount++;
    document.getElementById("stat-distress").textContent = distressCount;
    document.getElementById("stat-angry").textContent = angryCount;

    // Chart
    const timeLabel = formatTime(data.timestamp);
    chartLabels.push(timeLabel);
    if (data.fused_probs) {
        chartData.DISTRESS.push(data.fused_probs[0]);
        chartData.ANGRY.push(data.fused_probs[1]);
        chartData.ALERT.push(data.fused_probs[2]);
        chartData.CALM.push(data.fused_probs[3]);
    }
    if (historyChart) historyChart.update();

    // Update segment list
    addSegmentToList(data);

    // Update progress
    if (data.timestamp !== undefined) {
        // For realtime mode, update progress via status endpoint
    }
}

function handleAlertResult(data) {
    alerts.push(data);
    alertCount++;
    document.getElementById("stat-alerts").textContent = alertCount;

    const countBadge = document.getElementById("alert-count");
    countBadge.textContent = alertCount;
    countBadge.style.display = "";

    addAlertToList(data);
}

function updateProbBars(prefix, probs) {
    const names = ["distress", "angry", "alert", "calm"];
    for (let i = 0; i < 4; i++) {
        const pct = (probs[i] * 100).toFixed(0);
        const bar = document.getElementById(`bar-${prefix}${names[i]}`);
        const val = document.getElementById(`val-${prefix}${names[i]}`);
        if (bar) bar.style.width = pct + "%";
        if (val) val.textContent = pct + "%";
    }
}

function resetProbBars(prefix) {
    const names = ["distress", "angry", "alert", "calm"];
    for (let i = 0; i < 4; i++) {
        const bar = document.getElementById(`bar-${prefix}${names[i]}`);
        const val = document.getElementById(`val-${prefix}${names[i]}`);
        if (bar) bar.style.width = "0%";
        if (val) val.textContent = "--";
    }
}

function addSegmentToList(data) {
    const list = document.getElementById("segment-list");
    // Remove empty message
    const empty = list.querySelector(".segment-empty");
    if (empty) empty.remove();

    const cls = data.predicted_class;
    const color = COLORS[cls] || "#e8e8ed";

    const item = document.createElement("div");
    item.className = "segment-item";
    item.innerHTML = `
        <span class="segment-time">${formatTime(data.timestamp)}</span>
        <span class="segment-class" style="color:${color}">${cls}</span>
        <span class="segment-conf">${(data.confidence * 100).toFixed(0)}%</span>
    `;

    list.prepend(item);

    // Keep max 50 visible
    while (list.children.length > 50) {
        list.removeChild(list.lastChild);
    }
}

function addAlertToList(data) {
    const feed = document.getElementById("alert-feed");
    const empty = feed.querySelector(".alert-empty");
    if (empty) empty.remove();

    const emotion = data.emotion || "UNKNOWN";
    const severity = data.severity || "LOW";
    const conf = data.confidence || 0;
    const color = COLORS[emotion] || "#e8e8ed";

    const item = document.createElement("div");
    item.className = `alert-item severity-${severity}`;
    item.innerHTML = `
        <span class="alert-severity ${severity}">${severity}</span>
        <span class="alert-emotion" style="color:${color}">${emotion}</span>
        <span class="alert-confidence">${(conf * 100).toFixed(0)}%</span>
        <span class="alert-time">${data.consecutive_count || 1}x</span>
    `;

    feed.prepend(item);
}

function updateAnalysisStatus(status, percent, progress) {
    const fill = document.getElementById("analysis-fill");
    const statusEl = document.getElementById("analysis-status");
    const pctEl = document.getElementById("analysis-percent");

    fill.style.width = percent + "%";
    pctEl.textContent = percent.toFixed(0) + "%";

    if (status === "running") {
        const segText = progress ? `Segment ${progress.current_segment}/${progress.total_segments}` : "Running...";
        statusEl.textContent = segText;
        statusEl.className = "analysis-status-text";
    } else if (status === "complete") {
        statusEl.textContent = "Analysis complete";
        statusEl.className = "analysis-status-text complete";
    } else if (status === "error") {
        statusEl.textContent = "Error occurred";
        statusEl.className = "analysis-status-text error";
    } else if (status === "paused") {
        statusEl.textContent = "Paused";
        statusEl.className = "analysis-status-text";
    }
}

// --- Utilities ---
function formatTime(ts) {
    // ts is seconds from start of video
    const secs = Math.floor(ts);
    const mins = Math.floor(secs / 60);
    const s = secs % 60;
    return `${mins}:${String(s).padStart(2, "0")}`;
}

function resetResults() {
    segments = [];
    alerts = [];
    segmentCount = 0;
    alertCount = 0;
    distressCount = 0;
    angryCount = 0;

    // Reset chart
    chartLabels.length = 0;
    chartData.DISTRESS.length = 0;
    chartData.ANGRY.length = 0;
    chartData.ALERT.length = 0;
    chartData.CALM.length = 0;
    if (historyChart) historyChart.update();

    // Reset UI
    document.getElementById("stat-segments").textContent = "0";
    document.getElementById("stat-alerts").textContent = "0";
    document.getElementById("stat-distress").textContent = "0";
    document.getElementById("stat-angry").textContent = "0";
    document.getElementById("emotion-icon").textContent = "--";
    document.getElementById("emotion-label").textContent = "Waiting...";
    document.getElementById("emotion-confidence").textContent = "--";
    document.getElementById("streak-count").textContent = "0";
    document.getElementById("streak-class").textContent = "--";

    document.getElementById("alert-feed").innerHTML = '<div class="alert-empty">No alerts yet.</div>';
    document.getElementById("segment-list").innerHTML = '<div class="segment-empty">No segments yet.</div>';

    const countBadge = document.getElementById("alert-count");
    countBadge.style.display = "none";
    countBadge.textContent = "0";

    resetProbBars("");
    resetProbBars("audio-");
    resetProbBars("video-");
}

function resetControls() {
    document.getElementById("btn-analyze").disabled = !jobId;
    document.getElementById("btn-pause").disabled = true;
    document.getElementById("btn-pause").style.display = "";
    document.getElementById("btn-resume").disabled = true;
    document.getElementById("btn-resume").style.display = "none";
    document.getElementById("btn-stop").disabled = true;
}

function exportResultsCSV() {
    if (segments.length === 0) {
        alert("No results to export.");
        return;
    }

    const header = "timestamp,predicted_class,confidence,distress,angry,alert,calm,audio_distress,audio_angry,audio_alert,audio_calm,video_distress,video_angry,video_alert,video_calm\n";
    let csv = header;

    segments.forEach(seg => {
        const fp = seg.fused_probs || [0, 0, 0, 0];
        const ap = seg.audio_probs || [0, 0, 0, 0];
        const vp = seg.video_probs || [0, 0, 0, 0];
        csv += `${seg.timestamp},${seg.predicted_class},${seg.confidence.toFixed(4)},` +
               `${fp[0].toFixed(4)},${fp[1].toFixed(4)},${fp[2].toFixed(4)},${fp[3].toFixed(4)},` +
               `${ap[0].toFixed(4)},${ap[1].toFixed(4)},${ap[2].toFixed(4)},${ap[3].toFixed(4)},` +
               `${vp[0].toFixed(4)},${vp[1].toFixed(4)},${vp[2].toFixed(4)},${vp[3].toFixed(4)}\n`;
    });

    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "silentcare_offline_results.csv";
    a.click();
    URL.revokeObjectURL(url);
}

// Also poll for progress updates in realtime mode
function startProgressPolling() {
    setInterval(() => {
        if (!jobId) return;
        fetch(`/api/offline/status/${jobId}`)
            .then(r => r.json())
            .then(data => {
                if (data.progress) {
                    updateAnalysisStatus(data.status, data.progress.percent, data.progress);
                }
            })
            .catch(() => {});
    }, 2000);
}

// Override startSSE to also poll progress
const _origStartSSE = startSSE;
startSSE = function() {
    _origStartSSE();
    startProgressPolling();
};
