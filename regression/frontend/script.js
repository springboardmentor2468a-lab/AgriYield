let chart;
// ================= CUSTOM DROPDOWN LOGIC =================
const dropdown = document.getElementById("cropDropdown");
const selected = document.getElementById("selectedCrop");
const menu = document.getElementById("cropMenu");
const hiddenInput = document.getElementById("targetcrop");
const searchInput = document.getElementById("cropSearch");

selected.addEventListener("click", () => {
    menu.style.display = menu.style.display === "block" ? "none" : "block";
});

menu.addEventListener("click", (e) => {
    if (e.target.classList.contains("dropdown-item")) {
        selected.innerText = e.target.innerText;
        hiddenInput.value = e.target.innerText;
        menu.style.display = "none";
    }
});

searchInput.addEventListener("input", () => {
    const value = searchInput.value.toLowerCase();
    document.querySelectorAll(".dropdown-item").forEach(item => {
        item.style.display = item.innerText.includes(value) ? "block" : "none";
    });
});

document.addEventListener("click", (e) => {
    if (!dropdown.contains(e.target)) {
        menu.style.display = "none";
    }
});


document.getElementById("predictForm").addEventListener("submit", async (e) => {
    e.preventDefault();

    // ------------------ COLLECT INPUT DATA ------------------
    const data = {
        N: Number(document.getElementById("N").value),
        P: Number(document.getElementById("P").value),
        K: Number(document.getElementById("K").value),
        temperature: Number(document.getElementById("temperature").value),
        humidity: Number(document.getElementById("humidity").value),
        ph: Number(document.getElementById("ph").value),
        rainfall: Number(document.getElementById("rainfall").value),
        targetcrop: document.getElementById("targetcrop").value
    };

    const aiText = document.getElementById("aiText");
    aiText.innerText = "⏳ Analyzing soil & weather data using AI...";

    // ------------------ API CALL ------------------
    const res = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
    });

    const result = await res.json();

    // ------------------ SUMMARY TEXT ------------------
    aiText.innerText =
        `🤖 Estimated Yield: ${result.estimated_tons} tons/ha
Based on AI productivity index & historical data`;

    // ------------------ GRAPH LOGIC ------------------
    const targetCrop = data.targetcrop.toLowerCase();

    // Take top 5 crops from backend
    let crops = Object.keys(result.graph_data).slice(0, 5);

    // Ensure selected crop is always visible
    if (!crops.includes(targetCrop)) {
        crops[crops.length - 1] = targetCrop;
    }

    const yields = crops.map(crop => result.graph_data[crop]);

    const ctx = document.getElementById("yieldChart").getContext("2d");

    if (chart) chart.destroy();

    chart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: crops.map(c =>
                c.charAt(0).toUpperCase() + c.slice(1)
            ),
            datasets: [{
                data: yields,
                backgroundColor: crops.map(c =>
                    c === targetCrop
                        ? "#22c55e"                 // Selected crop
                        : "rgba(34,197,94,0.25)"    // Other crops
                ),
                borderRadius: 10,
                barThickness: 44
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { display: false },
                title: {
                    display: true,
                    text: "Estimated Yield Analysis (Metric Tons / Hectare)",
                    color: "#e5e7eb",
                    font: { size: 16 }
                },
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            return `${context.parsed.y} tons/ha`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: { color: "#e5e7eb" },
                    grid: { color: "#1f2933" },
                    title: {
                        display: true,
                        text: "Yield (tons / hectare)",
                        color: "#9ca3af"
                    }
                },
                x: {
                    ticks: { color: "#e5e7eb" },
                    grid: { display: false }
                }
            }
        }
    });

    // ------------------ RECOMMENDATION TABLE ------------------
    const tbody = document.getElementById("recommendBody");
    tbody.innerHTML = "";

    result.recommendations.forEach(row => {
        const trendClass =
            row.trend === "High" ? "trend-high" :
            row.trend === "Medium" ? "trend-medium" : "trend-low";

        const bestClass = row.rank === 1 ? "best-row" : "";

        tbody.innerHTML += `
            <tr class="${bestClass}">
                <td>${row.rank}</td>
                <td>${row.crop}</td>
                <td>${row.yield}</td>
                <td><span class="${trendClass}">${row.trend}</span></td>
            </tr>
        `;
    });
});
