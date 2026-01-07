/**
 * AgroSky AI - Dashboard JavaScript
 * Handles API calls, chart rendering, and UI updates
 */

// Global variables
let yieldChart = null;
const API_URL = 'http://127.0.0.1:5000/predict';
let currentMode = 'yield'; // 'yield' or 'crop'

// Common crop list for generating top 5 (if backend doesn't provide)
const COMMON_CROPS = [
    'Rice', 'Wheat', 'Maize', 'Cotton', 'Sugarcane', 'Potato', 'Tomato',
    'Onion', 'Chilli', 'Banana', 'Mango', 'Apple', 'Grapes', 'Corn',
    'Barley', 'Soybean', 'Peanut', 'Sunflower', 'Mustard', 'Groundnut'
];

/**
 * Handle form submission
 */
function handleSubmit(event) {
    event.preventDefault();
    
    // Get form data
    const formData = {
        nitrogen: parseFloat(document.getElementById('nitrogen').value),
        phosphorus: parseFloat(document.getElementById('phosphorus').value),
        potassium: parseFloat(document.getElementById('potassium').value),
        temperature: parseFloat(document.getElementById('temperature').value),
        humidity: parseFloat(document.getElementById('humidity').value),
        rainfall: parseFloat(document.getElementById('rainfall').value),
        ph: parseFloat(document.getElementById('ph').value)
    };

    // Validate inputs
    if (!validateInputs(formData)) {
        alert('Please fill in all fields with valid values.');
        return;
    }

    // Show loading state
    const submitBtn = document.getElementById('submitBtn');
    const btnText = submitBtn.querySelector('.btn-text');
    const btnLoader = document.getElementById('btnLoader');
    
    btnText.style.display = 'none';
    btnLoader.style.display = 'inline-block';
    submitBtn.disabled = true;

    // Make API call
    fetch(API_URL, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(formData)
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => {
                throw new Error(err.error || `HTTP error! status: ${response.status}`);
            });
        }
        return response.json();
    })
    .then(data => {
        // Check if response has error
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Validate response data
        if (!data.recommended_crop || !data.predicted_yield_kg_per_ha) {
            throw new Error('Invalid response from server. Missing crop or yield data.');
        }
        
        // Process and display results based on mode
        if (currentMode === 'yield') {
            displayYieldResults(data, formData);
            generateYieldInsights(data, formData);
        } else {
            displayCropRecommendation(data, formData);
            generateCropTips(data, formData);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        // Reset button state on error
        btnText.style.display = 'inline-block';
        btnLoader.style.display = 'none';
        submitBtn.disabled = false;
        
        // Show detailed error message
        let errorMsg = 'Error making prediction.\n\n';
        if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
            errorMsg += '❌ Backend server is not running!\n\n';
            errorMsg += 'Please:\n';
            errorMsg += '1. Open terminal/command prompt\n';
            errorMsg += '2. Navigate to project folder\n';
            errorMsg += '3. Run: python app.py\n';
            errorMsg += '4. Wait for "Running on http://127.0.0.1:5000"\n';
            errorMsg += '5. Then try again';
        } else {
            errorMsg += 'Error: ' + error.message + '\n\n';
            errorMsg += 'Please check:\n';
            errorMsg += '- Backend server is running\n';
            errorMsg += '- All model files are present\n';
            errorMsg += '- Input values are valid';
        }
        alert(errorMsg);
    })
    .finally(() => {
        // Reset button state
        btnText.style.display = 'inline-block';
        btnLoader.style.display = 'none';
        submitBtn.disabled = false;
    });
}

/**
 * Validate form inputs
 */
function validateInputs(data) {
    for (let key in data) {
        if (isNaN(data[key]) || data[key] === null || data[key] === undefined) {
            return false;
        }
    }
    
    // Validate pH range
    if (data.ph < 0 || data.ph > 14) {
        alert('Soil pH must be between 0 and 14');
        return false;
    }
    
    return true;
}

/**
 * Display yield prediction results
 */
function displayYieldResults(data, inputData) {
    // Show yield results card and hide placeholder
    document.getElementById('yieldResultsCard').style.display = 'block';
    document.getElementById('cropResultsCard').style.display = 'none';
    document.getElementById('resultsPlaceholder').style.display = 'none';
    document.getElementById('yieldInsightsSection').style.display = 'block';
    document.getElementById('cropTipsSection').style.display = 'none';

    // Update recommended crop
    const recommendedCrop = data.recommended_crop || 'Unknown';
    document.getElementById('recommendedCrop').textContent = recommendedCrop;

    // Update predicted yield
    const predictedYield = data.predicted_yield_kg_per_ha || 0;
    document.getElementById('predictedYield').textContent = predictedYield.toLocaleString('en-IN', {
        maximumFractionDigits: 2
    });

    // Calculate suitability percentage (based on yield)
    const suitabilityPercent = calculateSuitability(predictedYield);
    document.getElementById('suitabilityPercent').textContent = suitabilityPercent;

    // Generate top 5 crops (simulated based on main prediction)
    const topCrops = generateTop5Crops(data, inputData);
    displayTop5Table(topCrops);
    
    // Render chart
    renderChart(topCrops);
}

/**
 * Display crop recommendation results (simple)
 */
function displayCropRecommendation(data, inputData) {
    // Show crop recommendation card and hide placeholder
    document.getElementById('cropResultsCard').style.display = 'block';
    document.getElementById('yieldResultsCard').style.display = 'none';
    document.getElementById('resultsPlaceholder').style.display = 'none';
    document.getElementById('cropTipsSection').style.display = 'block';
    document.getElementById('yieldInsightsSection').style.display = 'none';

    // Update recommended crop
    const recommendedCrop = data.recommended_crop || 'Unknown';
    document.getElementById('cropRecommendedName').textContent = recommendedCrop;

    // Calculate suitability percentage
    const predictedYield = data.predicted_yield_kg_per_ha || 0;
    const suitabilityPercent = calculateSuitability(predictedYield);
    document.getElementById('cropSuitabilityPercent').textContent = suitabilityPercent;
}

/**
 * Calculate suitability percentage based on yield
 */
function calculateSuitability(yield) {
    // Normalize yield to 0-100 scale
    // Assuming typical yield range: 0-10000 kg/ha
    const maxYield = 10000;
    const suitability = Math.min(100, Math.max(0, (yield / maxYield) * 100));
    return Math.round(suitability);
}

/**
 * Generate top 5 crops (simulated)
 * In a real scenario, the backend would provide this
 */
function generateTop5Crops(mainResult, inputData) {
    const crops = [];
    const mainCrop = mainResult.recommended_crop;
    const mainYield = mainResult.predicted_yield_kg_per_ha;
    
    // Add main recommended crop as #1
    crops.push({
        rank: 1,
        name: mainCrop,
        yield: mainYield,
        suitability: calculateSuitability(mainYield)
    });

    // Generate 4 more crops with variations
    // In production, this would come from backend
    const variations = [0.85, 0.75, 0.65, 0.55];
    const otherCrops = COMMON_CROPS.filter(c => c !== mainCrop).slice(0, 4);
    
    variations.forEach((variation, index) => {
        if (otherCrops[index]) {
            const yieldVariation = mainYield * variation;
            crops.push({
                rank: index + 2,
                name: otherCrops[index],
                yield: yieldVariation,
                suitability: calculateSuitability(yieldVariation)
            });
        }
    });

    // Sort by yield (descending) and reassign ranks
    crops.sort((a, b) => b.yield - a.yield);
    crops.forEach((crop, index) => {
        crop.rank = index + 1;
    });

    return crops.slice(0, 5);
}

/**
 * Display top 5 crops in table
 */
function displayTop5Table(crops) {
    const tableBody = document.getElementById('cropsTableBody');
    tableBody.innerHTML = '';

    crops.forEach(crop => {
        const row = document.createElement('tr');
        
        // Highlight first row (best crop)
        if (crop.rank === 1) {
            row.style.backgroundColor = '#f0f9f4';
            row.style.fontWeight = '600';
        }

        row.innerHTML = `
            <td><strong>#${crop.rank}</strong></td>
            <td>${crop.name}</td>
            <td>${crop.yield.toLocaleString('en-IN', { maximumFractionDigits: 2 })} kg/ha</td>
            <td>
                <span style="color: ${getSuitabilityColor(crop.suitability)}; font-weight: 600;">
                    ${crop.suitability}%
                </span>
            </td>
        `;
        
        tableBody.appendChild(row);
    });
}

/**
 * Get color based on suitability percentage
 */
function getSuitabilityColor(percent) {
    if (percent >= 80) return '#27ae60'; // Green
    if (percent >= 60) return '#f39c12'; // Orange
    if (percent >= 40) return '#e67e22'; // Dark orange
    return '#e74c3c'; // Red
}

/**
 * Render bar chart using Chart.js
 */
function renderChart(crops) {
    const ctx = document.getElementById('yieldChart').getContext('2d');
    
    // Destroy existing chart if it exists
    if (yieldChart) {
        yieldChart.destroy();
    }

    // Prepare data
    const labels = crops.map(c => c.name);
    const yields = crops.map(c => c.yield);
    const colors = crops.map(c => {
        const suitability = c.suitability;
        if (suitability >= 80) return '#27ae60';
        if (suitability >= 60) return '#2ecc71';
        if (suitability >= 40) return '#f39c12';
        return '#e67e22';
    });

    // Create new chart
    yieldChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Predicted Yield (kg/ha)',
                data: yields,
                backgroundColor: colors,
                borderColor: '#145a32',
                borderWidth: 2,
                borderRadius: 8,
                barThickness: 'flex',
                maxBarThickness: 60
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        font: {
                            size: 12,
                            weight: 'bold'
                        },
                        color: '#2c3e50'
                    }
                },
                tooltip: {
                    callbacks: {
                        title: function(context) {
                            return context[0].label;
                        },
                        label: function(context) {
                            const crop = crops[context.dataIndex];
                            return [
                                `Predicted Yield: ${context.parsed.y.toLocaleString('en-IN', { maximumFractionDigits: 2 })} kg/ha`,
                                `Suitability: ${crop.suitability}%`
                            ];
                        }
                    },
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    padding: 12,
                    titleFont: {
                        size: 13,
                        weight: 'bold'
                    },
                    bodyFont: {
                        size: 12
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Predicted Yield (kg/ha)',
                        font: {
                            size: 14,
                            weight: 'bold'
                        },
                        color: '#2c3e50'
                    },
                    ticks: {
                        callback: function(value) {
                            return value.toLocaleString('en-IN') + ' kg/ha';
                        },
                        font: {
                            size: 11
                        }
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Crop Name',
                        font: {
                            size: 14,
                            weight: 'bold'
                        },
                        color: '#2c3e50'
                    },
                    ticks: {
                        font: {
                            size: 11
                        }
                    }
                }
            }
        }
    });
}

/**
 * Generate AI insights and tips for yield prediction
 */
function generateYieldInsights(data, inputData) {
    const insightsContent = document.getElementById('yieldInsightsContent');
    const crop = data.recommended_crop;
    const yield = data.predicted_yield_kg_per_ha;
    
    let insights = '';

    // Why this crop was recommended
    insights += `<h3 style="color: #27ae60; margin-bottom: 0.5rem;">Why ${crop} was Recommended:</h3>`;
    insights += `<p>`;
    
    // Analyze soil nutrients
    if (inputData.nitrogen > 50 && inputData.phosphorus > 30 && inputData.potassium > 40) {
        insights += `${crop} thrives in nutrient-rich soil conditions. Your soil shows excellent levels of Nitrogen (${inputData.nitrogen} mg/kg), Phosphorus (${inputData.phosphorus} mg/kg), and Potassium (${inputData.potassium} mg/kg), which are ideal for ${crop} cultivation. `;
    } else {
        insights += `${crop} is well-suited for your current soil nutrient profile. `;
    }

    // Analyze pH
    if (inputData.ph >= 6.0 && inputData.ph <= 7.5) {
        insights += `The soil pH of ${inputData.ph} is within the optimal range for ${crop}, promoting healthy root development and nutrient absorption. `;
    } else {
        insights += `While the soil pH of ${inputData.ph} may require some adjustment, ${crop} can still be cultivated with proper soil management. `;
    }

    // Analyze climate
    if (inputData.temperature >= 20 && inputData.temperature <= 35) {
        insights += `The temperature of ${inputData.temperature}°C is favorable for ${crop} growth. `;
    }
    
    if (inputData.humidity >= 50 && inputData.humidity <= 80) {
        insights += `The humidity level of ${inputData.humidity}% provides adequate moisture for ${crop}. `;
    }

    if (inputData.rainfall >= 500 && inputData.rainfall <= 1500) {
        insights += `The rainfall of ${inputData.rainfall} mm is suitable for ${crop} cultivation.`;
    }

    insights += `</p>`;

    // Why yield is high/low
    insights += `<h3 style="color: #27ae60; margin-top: 1.5rem; margin-bottom: 0.5rem;">Yield Analysis:</h3>`;
    insights += `<p>`;
    
    const suitability = calculateSuitability(yield);
    
    if (suitability >= 80) {
        insights += `The predicted yield of ${yield.toLocaleString('en-IN', { maximumFractionDigits: 2 })} kg/ha indicates <strong>excellent growing conditions</strong>. This high yield is attributed to the optimal combination of soil nutrients, favorable climate conditions, and suitable environmental factors. With proper crop management and timely interventions, you can expect a bountiful harvest.`;
    } else if (suitability >= 60) {
        insights += `The predicted yield of ${yield.toLocaleString('en-IN', { maximumFractionDigits: 2 })} kg/ha suggests <strong>good growing conditions</strong>. While the yield is promising, there's potential for improvement through soil enrichment, irrigation management, and crop-specific care practices.`;
    } else if (suitability >= 40) {
        insights += `The predicted yield of ${yield.toLocaleString('en-IN', { maximumFractionDigits: 2 })} kg/ha indicates <strong>moderate growing conditions</strong>. Consider soil amendments, improved irrigation, and crop-specific fertilizers to enhance yield potential.`;
    } else {
        insights += `The predicted yield of ${yield.toLocaleString('en-IN', { maximumFractionDigits: 2 })} kg/ha suggests that <strong>conditions may need improvement</strong>. Consider soil testing, nutrient supplementation, and consulting with agricultural experts to optimize growing conditions for better yields.`;
    }

    insights += `</p>`;

    // Yield Improvement Tips
    insights += `<h3 style="color: #27ae60; margin-top: 1.5rem; margin-bottom: 0.5rem;">📌 Yield Improvement Tips:</h3>`;
    insights += `<ul style="line-height: 1.8; padding-left: 1.5rem;">`;
    insights += `<li><strong>Soil Management:</strong> Maintain optimal soil pH (6.0-7.5) through regular testing and amendments. Add organic matter to improve soil structure and nutrient retention.</li>`;
    insights += `<li><strong>Nutrient Balance:</strong> Ensure balanced NPK levels. Consider split application of fertilizers during different growth stages for better nutrient uptake.</li>`;
    insights += `<li><strong>Water Management:</strong> Implement efficient irrigation systems. Monitor soil moisture levels and avoid over-watering or under-watering.</li>`;
    insights += `<li><strong>Crop Rotation:</strong> Practice crop rotation to maintain soil fertility and reduce pest/disease buildup.</li>`;
    insights += `<li><strong>Timely Harvesting:</strong> Harvest at the right maturity stage to maximize yield and quality. Monitor crop development closely.</li>`;
    insights += `<li><strong>Pest & Disease Control:</strong> Implement integrated pest management (IPM) strategies. Regular monitoring and early intervention can prevent yield losses.</li>`;
    insights += `<li><strong>Weather Monitoring:</strong> Stay updated with weather forecasts. Protect crops from extreme weather conditions using appropriate measures.</li>`;
    insights += `</ul>`;

    insightsContent.innerHTML = insights;
}

/**
 * Generate tips for crop recommendation
 */
function generateCropTips(data, inputData) {
    const tipsContent = document.getElementById('cropTipsContent');
    const crop = data.recommended_crop;
    
    let tips = '';

    tips += `<h3 style="color: #27ae60; margin-bottom: 0.5rem;">Growing Tips for ${crop}:</h3>`;
    tips += `<ul style="line-height: 1.8; padding-left: 1.5rem;">`;
    
    // General tips based on crop type
    tips += `<li><strong>Soil Preparation:</strong> Prepare the soil well before planting. Ensure proper drainage and soil structure for optimal root development.</li>`;
    tips += `<li><strong>Planting Time:</strong> Plant ${crop} at the recommended time based on your local climate. Consider temperature and rainfall patterns.</li>`;
    tips += `<li><strong>Spacing:</strong> Maintain proper spacing between plants to ensure adequate sunlight, air circulation, and nutrient availability.</li>`;
    tips += `<li><strong>Fertilization:</strong> Apply fertilizers based on soil test results. ${crop} requires balanced nutrition for healthy growth.</li>`;
    tips += `<li><strong>Irrigation:</strong> Provide consistent moisture, especially during critical growth stages. Avoid waterlogging.</li>`;
    tips += `<li><strong>Weed Control:</strong> Keep the field weed-free, especially during early growth stages when crops are most vulnerable.</li>`;
    tips += `<li><strong>Monitoring:</strong> Regularly monitor crop health, watch for signs of pests, diseases, or nutrient deficiencies.</li>`;
    tips += `<li><strong>Harvesting:</strong> Harvest at the right maturity stage for best quality and yield. Handle produce carefully to minimize damage.</li>`;
    tips += `</ul>`;

    tipsContent.innerHTML = tips;
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    // Check if redirected from home page with service selection
    const selectedService = localStorage.getItem('selectedService');
    if (selectedService) {
        currentMode = selectedService; // 'yield' or 'crop'
        localStorage.removeItem('selectedService');
        
        // Update page title and subtitle based on mode
        const pageTitle = document.getElementById('pageTitle');
        const pageSubtitle = document.getElementById('pageSubtitle');
        
        if (currentMode === 'yield') {
            if (pageTitle) pageTitle.textContent = 'Yield Prediction Dashboard';
            if (pageSubtitle) pageSubtitle.textContent = 'Predict crop yield based on soil and climate conditions';
        } else if (currentMode === 'crop') {
            if (pageTitle) pageTitle.textContent = 'Crop Recommendation Dashboard';
            if (pageSubtitle) pageSubtitle.textContent = 'Get AI-powered crop recommendations for your land';
        }
    } else {
        // Default to yield prediction
        currentMode = 'yield';
    }
    
    console.log('Dashboard initialized. Mode:', currentMode);
});

