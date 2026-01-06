let chart;

// Load crops
fetch("/get-crops")
  .then(res => res.json())
  .then(crops => {
    const select = document.getElementById("selectedCrop");
    crops.forEach(crop => {
      const option = document.createElement("option");
      option.value = crop;
      option.textContent = crop;
      select.appendChild(option);
    });
  });

function predict() {
  const crop = document.getElementById("selectedCrop").value;
  if (!crop) {
    alert("Please select a crop");
    return;
  }

  document.getElementById("yieldResult").innerText = "⏳ Predicting...";
  document.getElementById("cropList").innerHTML = "";

  const data = {
    N: +N.value,
    P: +P.value,
    K: +K.value,
    temperature: +temperature.value,
    humidity: +humidity.value,
    ph: +ph.value,
    rainfall: +rainfall.value,
    selected_crop: crop
  };

  fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data)
  })
  .then(res => res.json())
  .then(res => {
    document.getElementById("yieldResult").innerText =
      `${res.predicted_yield_kg} kg / hectare`;

    const labels = [];
    const values = [];
    const cropList = document.getElementById("cropList");

    res.top_crops.forEach(item => {
      labels.push(item.crop);
      values.push(item.score);

      const li = document.createElement("li");
      li.innerText = `${item.crop} — ${item.score}%`;

      if (item.crop.toLowerCase() === res.selected_crop.toLowerCase()) {
        li.style.background = "#7CFC00";
        li.style.color = "#000";
        li.style.fontWeight = "bold";
      }

      cropList.appendChild(li);
    });

    drawChart(labels, values);
  });
}

function drawChart(labels, values) {
  const ctx = document.getElementById("cropChart");

  if (chart) chart.destroy();

  chart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [{
        data: values,
        backgroundColor: "#7CFC00"
      }]
    },
    options: {
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: "#ffffff" } },
        y: {
          beginAtZero: true,
          ticks: { color: "#ffffff" },
          title: {
            display: true,
            text: "Suitability Score (%)",
            color: "#ffffff"
          }
        }
      }
    }
  });
}
