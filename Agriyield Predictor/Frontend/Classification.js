import { useState } from "react";
import axios from "axios";

// 🌱 Feature config with units & labels
const fields = [
  { name: "N", label: "Nitrogen", unit: "kg/ha" },
  { name: "P", label: "Phosphorus", unit: "kg/ha" },
  { name: "K", label: "Potassium", unit: "kg/ha" },
  { name: "temperature", label: "Temperature", unit: "°C" },
  { name: "humidity", label: "Humidity", unit: "%" },
  { name: "ph", label: "Soil pH", unit: "pH" },
  { name: "rainfall", label: "Rainfall", unit: "mm" },
];

export default function Classification() {
  const [form, setForm] = useState({});
  const [res, setRes] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: Number(e.target.value) });
  };

  const predict = async () => {
    try {
      setLoading(true);
      const { data } = await axios.post(
        "http://127.0.0.1:5001/predict-classification",
        form
      );
      setRes(data);
    } catch {
      alert("Prediction failed. Please check backend.");
    } finally {
      setLoading(false);
    }
  };

  // 🔥 Sort top-5 crops by suitability (descending)
  const sortedTop5 = res
    ? Object.entries(res.top5)
        .sort((a, b) => b[1] - a[1])
        .map(([crop], index) => ({
          rank: index + 1,
          crop,
        }))
    : [];

  return (
    <div className="page">
      <h2>🌾 Crop Recommendation (Classification)</h2>
      <p style={{ textAlign: "center", color: "#555" }}>
        Enter soil nutrients and climate conditions to get the best crop
        recommendation.
      </p>

      {/* 🌱 Input Form */}
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

      {/* 🔘 Button */}
      <button onClick={predict} disabled={loading}>
        {loading ? "Predicting..." : "Predict Best Crop"}
      </button>

      {/* 📋 Result */}
      {res && (
        <div className="result-box">
          <h3>
            🌱 Predicted Best Crop:
            <span style={{ color: "#1b4332" }}>
              {" "}
              {res.prediction.toUpperCase()}
            </span>
          </h3>

          <h4 style={{ marginTop: "15px" }}>🏆 Top 5 Recommended Crops</h4>

          <ul className="top5-list">
            {sortedTop5.map((item) => (
              <li key={item.rank}>
                <strong>{item.rank}.</strong>{" "}
                {item.crop.toUpperCase()}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
