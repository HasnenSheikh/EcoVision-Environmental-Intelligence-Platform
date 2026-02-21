/**
 * EcoVision Energy Deep Dive JavaScript
 * Handles the actual vs. predicted energy consumption chart with anomaly detection
 */

document.addEventListener('DOMContentLoaded', function() {
    initEnergyTimeseriesChart();
    initTimeFilterListener();
});

/**
 * Initialize the main energy timeseries chart with anomaly highlighting
 */
function initEnergyTimeseriesChart() {
    const ctx = document.getElementById('energyTimeseriesChart');
    if (!ctx) return;
    
    const timestamps = energyData.timestamps || [];
    const actualData = energyData.actual || [];
    const predictedData = energyData.predicted || [];
    const anomalies = energyData.anomalies || [];
    
    // Create anomaly background plugin
    const anomalyBackgroundPlugin = {
        id: 'anomalyBackground',
        beforeDraw: (chart) => {
            const ctx = chart.ctx;
            const chartArea = chart.chartArea;
            const xScale = chart.scales.x;
            const yScale = chart.scales.y;
            
            anomalies.forEach(anomaly => {
                if (anomaly.index >= 0 && anomaly.index < timestamps.length) {
                    // Find the anomaly point's position
                    const xPos = xScale.getPixelForValue(anomaly.index);
                    const xStart = anomaly.index > 0 ? 
                        xScale.getPixelForValue(anomaly.index - 0.5) : chartArea.left;
                    const xEnd = anomaly.index < timestamps.length - 1 ? 
                        xScale.getPixelForValue(anomaly.index + 0.5) : chartArea.right;
                    
                    // Draw red shaded area
                    ctx.save();
                    ctx.fillStyle = 'rgba(220, 53, 69, 0.15)';
                    ctx.fillRect(
                        xStart,
                        chartArea.top,
                        xEnd - xStart,
                        chartArea.bottom - chartArea.top
                    );
                    ctx.restore();
                }
            });
        }
    };
    
    // Create the chart
    const chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: timestamps,
            datasets: [
                {
                    label: 'Actual',
                    data: actualData,
                    borderColor: '#4A90E2',
                    backgroundColor: 'rgba(74, 144, 226, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 4,
                    pointBackgroundColor: '#4A90E2',
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2,
                    pointHoverRadius: 6
                },
                {
                    label: 'Predicted',
                    data: predictedData,
                    borderColor: '#95A5A6',
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    borderDash: [8, 4],
                    fill: false,
                    tension: 0.4,
                    pointRadius: 0,
                    pointHoverRadius: 5,
                    pointBackgroundColor: '#95A5A6'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'bottom',
                    labels: {
                        usePointStyle: true,
                        padding: 20,
                        font: {
                            size: 13,
                            family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto'
                        }
                    }
                },
                tooltip: {
                    enabled: true,
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    padding: 12,
                    titleFont: {
                        size: 14,
                        weight: 'bold'
                    },
                    bodyFont: {
                        size: 13
                    },
                    displayColors: true,
                    callbacks: {
                        title: (tooltipItems) => {
                            return tooltipItems[0].label;
                        },
                        label: (context) => {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            label += Math.round(context.parsed.y) + ' kWh';
                            return label;
                        },
                        afterBody: (tooltipItems) => {
                            // Check if this point is an anomaly
                            const index = tooltipItems[0].dataIndex;
                            const anomaly = anomalies.find(a => a.index === index);
                            if (anomaly) {
                                return [
                                    '',
                                    '⚠️ Anomaly Detected',
                                    `Type: ${anomaly.type}`,
                                    `Severity: ${anomaly.severity.toUpperCase()}`
                                ];
                            }
                            return [];
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        font: {
                            size: 12
                        },
                        maxRotation: 45,
                        minRotation: 0
                    }
                },
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)',
                        drawBorder: false
                    },
                    ticks: {
                        font: {
                            size: 12
                        },
                        callback: function(value) {
                            return value + ' kWh';
                        }
                    },
                    title: {
                        display: true,
                        text: 'Energy Consumption (kWh)',
                        font: {
                            size: 13,
                            weight: 'bold'
                        }
                    }
                }
            }
        },
        plugins: [anomalyBackgroundPlugin]
    });
    
    // Add anomaly labels
    addAnomalyLabels(anomalies, timestamps);
}

/**
 * Add text labels for anomalies above the chart
 */
function addAnomalyLabels(anomalies, timestamps) {
    const chartCard = document.querySelector('.chart-card');
    if (!chartCard || anomalies.length === 0) return;
    
    const existingLabel = chartCard.querySelector('.anomaly-label');
    if (existingLabel) return; // Already added
    
    const labelDiv = document.createElement('div');
    labelDiv.className = 'anomaly-label';
    labelDiv.style.cssText = `
        position: absolute;
        top: 80px;
        left: 50%;
        transform: translateX(-50%);
        background-color: rgba(220, 53, 69, 0.95);
        color: white;
        padding: 8px 16px;
        border-radius: 8px;
        font-size: 0.875rem;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        z-index: 10;
    `;
    
    const primaryAnomaly = anomalies[0];
    const anomalyTime = timestamps[primaryAnomaly.index] || 'Unknown';
    labelDiv.textContent = `Anomaly: ${primaryAnomaly.type} detected at ${anomalyTime}`;
    
    chartCard.style.position = 'relative';
    chartCard.appendChild(labelDiv);
}

/**
 * Time filter dropdown listener
 */
function initTimeFilterListener() {
    const timeFilter = document.querySelector('.time-filter');
    if (!timeFilter) return;
    
    timeFilter.addEventListener('change', function(e) {
        const selectedValue = e.target.value;
        console.log('Time filter changed to:', selectedValue);
        
        // In a real application, this would fetch new data from the API
        // For now, just log the change
        
        // Example API call (commented out):
        // fetchEnergyData(selectedValue);
    });
}

/**
 * Fetch energy data from API (for future implementation)
 */
async function fetchEnergyData(timeRange) {
    try {
        const response = await fetch(`/api/energy-timeseries?range=${timeRange}`);
        const data = await response.json();
        // Update chart with new data
        console.log('Fetched energy data:', data);
    } catch (error) {
        console.error('Error fetching energy data:', error);
    }
}

/**
 * Format number with commas
 */
function formatNumber(num) {
    return new Intl.NumberFormat('en-US').format(num);
}
