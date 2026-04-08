/**
 * script.js — Frontend Logic for Music Genre Classifier
 *
 * Handles:
 *  - Drag & drop + click file upload
 *  - Audio playback
 *  - API call to /predict
 *  - Animated confidence ring + genre score bars
 */

// ─── API Base URL ──────────────────────────────────────────────────
const API_BASE = window.location.origin;

// ─── DOM Elements ──────────────────────────────────────────────────
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const fileInfoSection = document.getElementById('fileInfoSection');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
const audioPlayer = document.getElementById('audioPlayer');
const btnRemove = document.getElementById('btnRemove');
const predictSection = document.getElementById('predictSection');
const btnPredict = document.getElementById('btnPredict');
const btnText = document.querySelector('.btn-text');
const btnLoader = document.getElementById('btnLoader');
const resultsSection = document.getElementById('resultsSection');
const predictedGenre = document.getElementById('predictedGenre');
const confidenceValue = document.getElementById('confidenceValue');
const confidenceRing = document.getElementById('confidenceRing');
const scoresChart = document.getElementById('scoresChart');
const chunksAnalyzed = document.getElementById('chunksAnalyzed');
const resultFileName = document.getElementById('resultFileName');
const uploadSection = document.getElementById('uploadSection');

let selectedFile = null;

// ─── Genre Colors ──────────────────────────────────────────────────
const genreColors = {
    blues:     '#6366f1',
    classical: '#8b5cf6',
    country:   '#f59e0b',
    disco:     '#ec4899',
    hiphop:    '#ef4444',
    jazz:      '#14b8a6',
    metal:     '#64748b',
    pop:       '#f97316',
    reggae:    '#22c55e',
    rock:      '#06b6d4'
};

// ─── Utility ───────────────────────────────────────────────────────
function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / 1048576).toFixed(1) + ' MB';
}

function showSection(el) {
    el.classList.remove('hidden');
    el.classList.add('fade-in');
}

function hideSection(el) {
    el.classList.add('hidden');
    el.classList.remove('fade-in');
}

// ─── File Upload Handling ──────────────────────────────────────────
uploadZone.addEventListener('click', () => fileInput.click());

uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('drag-over');
});

uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('drag-over');
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    const files = e.dataTransfer.files;
    if (files.length > 0) handleFile(files[0]);
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) handleFile(e.target.files[0]);
});

function handleFile(file) {
    const allowedTypes = ['.wav', '.mp3', '.flac', '.ogg', '.m4a'];
    const ext = '.' + file.name.split('.').pop().toLowerCase();

    if (!allowedTypes.includes(ext)) {
        alert(`Unsupported file type: ${ext}\nAllowed: ${allowedTypes.join(', ')}`);
        return;
    }

    selectedFile = file;

    // Update UI
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);

    // Audio player
    const audioUrl = URL.createObjectURL(file);
    audioPlayer.src = audioUrl;

    // Show file info + predict button, hide upload zone
    hideSection(uploadSection);
    showSection(fileInfoSection);
    showSection(predictSection);
    hideSection(resultsSection);
}

// ─── Remove File ───────────────────────────────────────────────────
btnRemove.addEventListener('click', () => {
    selectedFile = null;
    fileInput.value = '';
    audioPlayer.src = '';

    hideSection(fileInfoSection);
    hideSection(predictSection);
    hideSection(resultsSection);
    showSection(uploadSection);
});

// ─── Predict ───────────────────────────────────────────────────────
btnPredict.addEventListener('click', async () => {
    if (!selectedFile) return;

    // Loading state
    btnPredict.disabled = true;
    btnText.classList.add('hidden');
    btnLoader.classList.remove('hidden');
    hideSection(resultsSection);

    try {
        const formData = new FormData();
        formData.append('file', selectedFile);

        const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Prediction failed');
        }

        const data = await response.json();
        displayResults(data);

    } catch (error) {
        alert(`Error: ${error.message}`);
        console.error('Prediction error:', error);

    } finally {
        btnPredict.disabled = false;
        btnText.classList.remove('hidden');
        btnLoader.classList.add('hidden');
    }
});

// ─── Display Results ───────────────────────────────────────────────
function displayResults(data) {
    showSection(resultsSection);

    // Predicted genre
    predictedGenre.textContent = data.genre;

    // Confidence ring animation
    const confidencePercent = Math.round(data.confidence * 100);
    const circumference = 2 * Math.PI * 34; // r=34
    const offset = circumference - (data.confidence * circumference);

    confidenceRing.style.strokeDasharray = circumference;
    confidenceRing.style.strokeDashoffset = circumference;

    // Trigger animation after a brief delay
    requestAnimationFrame(() => {
        setTimeout(() => {
            confidenceRing.style.strokeDashoffset = offset;
        }, 100);
    });

    // Animated counter for confidence
    animateCounter(confidenceValue, 0, confidencePercent, 1000);

    // Meta info
    chunksAnalyzed.textContent = data.chunks_analyzed;
    resultFileName.textContent = data.filename;

    // Score bars
    renderScoreBars(data.all_scores, data.genre);
}

function animateCounter(element, start, end, duration) {
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Ease out cubic
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = Math.round(start + (end - start) * eased);

        element.textContent = current + '%';

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}

function renderScoreBars(scores, topGenre) {
    scoresChart.innerHTML = '';

    // Sort by score descending
    const sorted = Object.entries(scores).sort((a, b) => b[1] - a[1]);

    sorted.forEach(([genre, score], index) => {
        const percent = Math.round(score * 100);
        const isTop = genre === topGenre;

        const row = document.createElement('div');
        row.className = 'score-row';
        row.style.animationDelay = `${index * 60}ms`;

        row.innerHTML = `
            <span class="score-label">${genre}</span>
            <div class="score-bar-track">
                <div class="score-bar-fill ${isTop ? 'top-score' : ''}"
                     style="width: 0%;" data-width="${percent}%"></div>
            </div>
            <span class="score-percent">${percent}%</span>
        `;

        scoresChart.appendChild(row);
    });

    // Animate bars after DOM paint
    requestAnimationFrame(() => {
        setTimeout(() => {
            document.querySelectorAll('.score-bar-fill').forEach(bar => {
                bar.style.width = bar.dataset.width;
            });
        }, 150);
    });
}

// ─── Health Check on Load ──────────────────────────────────────────
async function checkHealth() {
    const statusEl = document.getElementById('headerStatus');
    try {
        const res = await fetch(`${API_BASE}/health`);
        const data = await res.json();

        if (data.model_loaded) {
            statusEl.innerHTML = '<span class="status-dot"></span><span>Model Ready</span>';
        } else {
            statusEl.innerHTML = '<span class="status-dot" style="background:#f59e0b;box-shadow:0 0 8px rgba(245,158,11,0.6)"></span><span>No Model</span>';
        }
    } catch {
        statusEl.innerHTML = '<span class="status-dot" style="background:#ef4444;box-shadow:0 0 8px rgba(239,68,68,0.6)"></span><span>Offline</span>';
    }
}

// Run health check when page loads
document.addEventListener('DOMContentLoaded', checkHealth);
