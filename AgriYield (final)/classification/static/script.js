document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll('input[type="range"]').forEach(slider => {
        const num = slider.previousElementSibling.querySelector("input");
        if (num) syncSlider(slider, num.id);
    });

    if (typeof currentPrediction !== "undefined") {
        saveHistory(currentPrediction);
    }

    renderHistory();
});

/* ================= SLIDER SYNC ================= */
function syncSlider(slider, inputId) {
    const num = document.getElementById(inputId);
    num.value = slider.value;

    const percent =
        (slider.value - slider.min) / (slider.max - slider.min) * 100;

    slider.style.background =
        `linear-gradient(90deg, #ffd166 ${percent}%, #555 ${percent}%)`;
}

/* ================= HISTORY STORAGE ================= */
function saveHistory(entry) {
    let history = JSON.parse(localStorage.getItem("predictionHistory")) || [];
    history.unshift(entry);
    history = history.slice(0, 5); // last 5 predictions only
    localStorage.setItem("predictionHistory", JSON.stringify(history));
}

/* ================= HISTORY RENDER ================= */
function renderHistory() {
    const historyDiv = document.getElementById("history");
    const history = JSON.parse(localStorage.getItem("predictionHistory")) || [];

    if (history.length === 0) {
        historyDiv.innerHTML = "<p style='opacity:0.6'>No predictions yet.</p>";
        return;
    }

    historyDiv.innerHTML = history.map(h => `
        <div class="history-entry">
            <div class="history-time">⏱ ${h.time}</div>

            ${h.crops.map(c => `
                <div class="crop-result">
                    <span>${c.name}</span>
                    <strong>${c.confidence}%</strong>
                </div>
                <div class="confidence-bar">
                    <div class="confidence-fill"
                         style="width:${c.confidence}%"></div>
                </div>
            `).join("")}
        </div>
    `).join("");
}

/* ================= CLEAR ================= */
function clearForm() {
    document.getElementById("inputForm").reset();
    localStorage.removeItem("predictionHistory");
    renderHistory();
}
