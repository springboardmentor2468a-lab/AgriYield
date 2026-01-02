document.addEventListener("DOMContentLoaded", () => {
    const form = document.querySelector(".predict-box");
    const resultBox = document.querySelector(".result-box");
    const resultText = document.querySelector(".predict-result");
    const resultImg = document.getElementById("productionChart");

    // Loading spinner
    const spinner = document.createElement("div");
    spinner.className = "loading-spinner";
    spinner.style.display = "none";

    // Error message
    const errorMsg = document.createElement("div");
    errorMsg.className = "error-msg";
    errorMsg.style.display = "none";

    resultBox.appendChild(spinner);
    resultBox.appendChild(errorMsg);

    // Crop → image mapping
    const result_img = {
        rice: "/static/assets/rice.png",
        maize: "/static/assets/maize.png",
        chickpea: "/static/assets/chipea.png",
        kidneybeans: "/static/assets/beans.png",
        pigeonpeas: "/static/assets/pigen.png",
        mothbeans: "/static/assets/beans.png",
        mungbean: "/static/assets/beans.png",
        blackgram: "/static/assets/beans.png",
        lentil: "/static/assets/lentil.png",
        pomegranate: "/static/assets/pomegranate.png",
        banana: "/static/assets/banana.png",
        mango: "/static/assets/mango.png",
        watermelon: "/static/assets/water.png",
        muskmelon: "/static/assets/melon.png",
        orange: "/static/assets/orange.png",
        papaya: "/static/assets/papaya.png",
        coconut: "/static/assets/coconut.png",
        cotton: "/static/assets/cotton.png",
        jute: "/static/assets/jute.png",
        coffee: "/static/assets/coffe.png"
    };

    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        spinner.style.display = "block";
        errorMsg.style.display = "none";
        resultText.textContent = "Predicting...";
        resultImg.src = "";

        const formData = new FormData(form);

        const payload = {
            N: Number(formData.get("N")),
            P: Number(formData.get("P")),
            K: Number(formData.get("K")),
            temperature: Number(formData.get("temperature")),
            humidity: Number(formData.get("humidity")),
            ph: Number(formData.get("ph")),
            rainfall: Number(formData.get("rainfall")),
        };

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || "Prediction failed");
            }

            const crop = data.result.toLowerCase();
            resultText.textContent = `Recommended Crop: ${data.result}`;

            if (result_img[crop]) {
                resultImg.src = result_img[crop];
            } else {
                resultImg.src = "/static/assets/default.png";
            }

        } catch (err) {
            errorMsg.textContent = err.message;
            errorMsg.style.display = "block";
            resultText.textContent = "Prediction failed";
        } finally {
            spinner.style.display = "none";
        }
    });
});
