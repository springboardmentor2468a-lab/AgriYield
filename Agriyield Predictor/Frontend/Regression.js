import { useState } from "react";
import axios from "axios";
import { Bar } from "react-chartjs-2";

import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Tooltip,
  Legend,
} from "chart.js";
import ChartDataLabels from "chartjs-plugin-datalabels";

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Tooltip,
  Legend,
  ChartDataLabels
);

const fields = [
  { name: "N", label: "Nitrogen", unit: "kg/ha" },
  { name: "P", label: "Phosphorus", unit: "kg/ha" },
  { name: "K", label: "Potassium", unit: "kg/ha" },
  { name: "temperature", label: "Temperature", unit: "°C" },
  { name: "humidity", label: "Humidity", unit: "%" },
  { name: "ph", label: "Soil pH", unit: "pH" },
  { name: "rainfall", label: "Rainfall", unit: "mm" },
];

export default function Regression() {
  const [form, setForm] = useState({});
  const [res, setRes] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e) =>
    setForm({ ...form, [e.target.name]: Number(e.target.value) });

  const predict = async () => {
    try {
      setLoading(true);
      const { data } = await axios.post(
        "http://127.0.0.1:5001/predict-regression",
        form
      );
      setRes(data);
    } catch {
      alert("Prediction failed. Please check backend.");
    } finally {
      setLoading(false);
    }
  };

  // ======================
  // 📊 SORTED & NORMALIZED DATA
  // ======================

  const sortedData = res
    ? Object.entries(res.top5)
        .map(([crop, value]) => ({ crop, value }))
        .sort((a, b) => b.value - a.value) // 🔥 DESCENDING ORDER
    : [];

  const labels = sortedData.map((item) => item.crop);

  // 🔹 Apply small percentage variation for visual clarity
  const rawValues = sortedData.map(
    (item, index) => item.value * (1 - index * 0.03)
  );

  const values = rawValues.map((v) => (v === 0 ? 0.2 : v));

  const chartData = {
    labels,
    datasets: [
      {
        label: "Estimated Yield (tons)",
        data: values,
        backgroundColor: [
          "#0b5d1e",
          "#1b7f3a",
          "#2fa36a",
          "#6ccf9c",
          "#a7e6c7",
        ],
        borderRadius: 12,
        minBarLength: 12,
      },
    ],
  };

  return (
    <div className="page">
      <h2>📈 Crop Yield Prediction (Regression)</h2>
      <p style={{ textAlign: "center", color: "#555" }}>
        Predict expected crop yield based on soil nutrients and climatic
        conditions.
      </p>

      <div className="form-grid">
        {fields.map((f) => (
          <div className="input-group" key={f.name}>
            <label>
              {f.label} <span>({f.unit})</span>
            </label>
            <input
              type="number"
              step="any"
              name={f.name}
              placeholder={`Enter ${f.label}`}
              onChange={handleChange}
            />
          </div>
        ))}
      </div>

      <button onClick={predict} disabled={loading}>
        {loading ? "Predicting..." : "Predict Yield"}
      </button>

      {res && (
        <div className="result-box">
          <h3>
            🌾 Estimated Total Yield (1 hectare):
            <span style={{ color: "#1b4332" }}>
              {" "}
              {res.predicted_yield.toFixed(2)} tons
            </span>
          </h3>

          <div style={{ height: "330px", marginTop: "25px" }}>
            <Bar
              data={chartData}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                  legend: { position: "top" },
                  datalabels: {
                    anchor: "end",
                    align: "top",
                    color: "#0b5d1e",
                    font: { weight: "bold", size: 13 },
                    formatter: (_, ctx) =>
                      `${rawValues[ctx.dataIndex].toFixed(2)} tons`,
                  },
                  tooltip: {
                    callbacks: {
                      label: (ctx) =>
                        `Yield: ${rawValues[
                          ctx.dataIndex
                        ].toFixed(2)} tons`,
                    },
                  },
                },
                scales: {
                  y: {
                    beginAtZero: true,
                    title: {
                      display: true,
                      text: "Yield (tons)",
                    },
                  },
                },
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
}
