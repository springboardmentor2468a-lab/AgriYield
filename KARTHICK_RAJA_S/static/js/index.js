document.addEventListener("DOMContentLoaded", () => {
    const form = document.querySelector(".predict-box");
    const chartCanvas = document.getElementById("productionChart");
    const resultBox = document.querySelector(".result-box");
    const resultText=document.querySelector(".predict-result");
    let chartInstance = null;
    const spinner = document.createElement("div");
    spinner.className = "loading-spinner";
    spinner.textContent = "Predicting...";
    spinner.style.display = "none";
    const errorMsg = document.createElement("div");
    errorMsg.className = "error-msg";
    errorMsg.style.display = "none";
    resultBox.appendChild(spinner);
    resultBox.appendChild(errorMsg);
    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        spinner.style.display = "block";
        errorMsg.style.display = "none";
        if (chartInstance) {
            chartInstance.destroy();
        }
        const formData = new FormData(form);
        const payload = {
            N: Number(formData.get("N")),
            P: Number(formData.get("P")),
            K: Number(formData.get("K")),
            temperature: Number(formData.get("temperature")),
            humidity: Number(formData.get("humidity")),
            ph: Number(formData.get("ph")),
            rainfall: Number(formData.get("rainfall")),
            year: Number(formData.get("year")),
            crop: formData.get("crop")
        };
        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.error || "Prediction failed");
            }
            console.log(data)
            resultText.textContent=data?.predicted_yield
            const labels = Object.keys(data.top_5_recommended_crops);
            const values = Object.values(data.top_5_recommended_crops);
            chartInstance = new Chart(chartCanvas, {
                type: "bar",
                data: {
                    labels: labels.map(c => c.toUpperCase()),
                    datasets: [{
                        label: "Estimated Yield",
                        data: values,
                        backgroundColor: "blue"
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        x: {
                            ticks: {color: "white"},
                            grid: {color: "white"},
                            border: { color: "white"}
                        },

                        y: { beginAtZero: true ,
                        ticks: { color: "white"},
                        grid: {color: "white"},
                        border: { color: "white"}
                        }

                    }
                }
            });

        } catch (err) {
            errorMsg.textContent = err.message;
            errorMsg.style.display = "block";
        } finally {
            spinner.style.display = "none";
        }
    });
});
