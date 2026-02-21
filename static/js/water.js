/**
 * Water Management Dashboard - Visualization Logic
 */

// Initialize Leaflet Map for Water Flow & Leaks
function initWaterFlowMap() {
    // Create map centered on campus (same as dashboard)
    const map = L.map('waterFlowMap', {
        center: [34.0522, -118.2437],
        zoom: 16,
        zoomControl: true,
        scrollWheelZoom: false
    });

    // Add OpenStreetMap tiles
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors',
        maxZoom: 19
    }).addTo(map);

    // Define custom marker icons (matching dashboard style)
    const blueIcon = L.divIcon({
        className: 'custom-marker',
        html: '<div style="background-color: #007bff; width: 30px; height: 30px; border-radius: 50% 50% 50% 0; transform: rotate(-45deg); border: 3px solid white; box-shadow: 0 2px 8px rgba(0,0,0,0.3);"><div style="transform: rotate(45deg); color: white; text-align: center; line-height: 24px; font-size: 16px;">💧</div></div>',
        iconSize: [30, 30],
        iconAnchor: [15, 30]
    });

    const redIcon = L.divIcon({
        className: 'custom-marker',
        html: '<div style="background-color: #DC3545; width: 30px; height: 30px; border-radius: 50% 50% 50% 0; transform: rotate(-45deg); border: 3px solid white; box-shadow: 0 2px 8px rgba(0,0,0,0.3); animation: marker-pulse 2s infinite;"><div style="transform: rotate(45deg); color: white; text-align: center; line-height: 24px; font-size: 16px;">⚠️</div></div>',
        iconSize: [30, 30],
        iconAnchor: [15, 30]
    });

    // Add normal water flow markers using same campus coordinates as dashboard
    const normalLocations = [
        { lat: 34.0522, lng: -118.2437, name: "Dorm 1" },
        { lat: 34.0525, lng: -118.2440, name: "Building A" },
        { lat: 34.0520, lng: -118.2430, name: "Library" },
        { lat: 34.0528, lng: -118.2435, name: "Athletic Center" },
        { lat: 34.0515, lng: -118.2442, name: "Dining Hall" }
    ];

    normalLocations.forEach(location => {
        L.marker([location.lat, location.lng], { icon: blueIcon })
            .addTo(map)
            .bindPopup(`<b>${location.name}</b><br>Normal Flow`);
    });

    // Add critical leak marker (red, pulsing) — Dorm 3 Basement
    const leakLocation = { lat: 34.0518, lng: -118.2445, name: "Dorm 3 Basement" };

    // Add CSS animation for pulsing markers (same as dashboard)
    const style = document.createElement('style');
    style.textContent = `
        @keyframes marker-pulse {
            0%, 100% { transform: rotate(-45deg) scale(1); opacity: 1; }
            50% { transform: rotate(-45deg) scale(1.1); opacity: 0.8; }
        }
    `;
    document.head.appendChild(style);

    L.marker([leakLocation.lat, leakLocation.lng], { icon: redIcon })
        .addTo(map)
        .bindPopup(`<b style="color: #DC3545;">⚠️ ${leakLocation.name}</b><br><strong>CRITICAL LEAK DETECTED</strong><br>Flow Rate: 15 Gal/min`)
        .openPopup();
}

// Initialize Building Usage Bar Chart
function initBuildingUsageChart() {
    const ctx = document.getElementById('buildingUsageChart').getContext('2d');
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: waterData.buildingLabels,
            datasets: [
                {
                    label: 'Building A',
                    data: waterData.buildingDataA,
                    backgroundColor: 'rgba(0, 123, 255, 0.7)',
                    borderColor: 'rgba(0, 123, 255, 1)',
                    borderWidth: 1
                },
                {
                    label: 'Building B',
                    data: waterData.buildingDataB,
                    backgroundColor: 'rgba(40, 167, 69, 0.7)',
                    borderColor: 'rgba(40, 167, 69, 1)',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: function(context) {
                            return context.dataset.label + ': ' + context.parsed.y.toLocaleString() + ' Gal';
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Gallons (Gal)'
                    },
                    ticks: {
                        callback: function(value) {
                            return value.toLocaleString() + ' Gal';
                        }
                    }
                },
                x: {
                    grid: {
                        display: false
                    }
                }
            }
        }
    });
}

// Initialize all visualizations on page load
document.addEventListener('DOMContentLoaded', function() {
    initWaterFlowMap();
    initBuildingUsageChart();
    initWaterTimeseriesChart();
});

/**
 * Initialize the Actual vs. Predicted LSTM water consumption chart
 * with anomaly shading – mirrors the energy page timeseries chart.
 */
function initWaterTimeseriesChart() {
    const ctx = document.getElementById('waterTimeseriesChart');
    if (!ctx) return;

    const timestamps    = waterTimeseriesData.timestamps  || [];
    const actualData    = waterTimeseriesData.actual      || [];
    const predictedData = waterTimeseriesData.predicted   || [];
    const anomalies     = waterTimeseriesData.anomalies   || [];
    let   sliceOffset   = 0;   // tracks current time-filter slice start

    // Plugin: shade anomaly columns in red
    const anomalyBackgroundPlugin = {
        id: 'waterAnomalyBackground',
        beforeDraw: (chart) => {
            const ctx       = chart.ctx;
            const chartArea = chart.chartArea;
            const xScale    = chart.scales.x;
            const visibleLen = chart.data.labels.length;

            anomalies.forEach(anomaly => {
                const relIdx = anomaly.index - sliceOffset;
                if (relIdx < 0 || relIdx >= visibleLen) return;

                const xStart = relIdx > 0
                    ? xScale.getPixelForValue(relIdx - 0.5)
                    : chartArea.left;
                const xEnd = relIdx < visibleLen - 1
                    ? xScale.getPixelForValue(relIdx + 0.5)
                    : chartArea.right;

                ctx.save();
                ctx.fillStyle = 'rgba(220, 53, 69, 0.15)';
                ctx.fillRect(xStart, chartArea.top, xEnd - xStart, chartArea.bottom - chartArea.top);
                ctx.restore();
            });
        }
    };

    const chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: timestamps,
            datasets: [
                {
                    label: 'Actual',
                    data: actualData,
                    borderColor: '#17A2B8',
                    backgroundColor: 'rgba(23, 162, 184, 0.10)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    pointRadius: actualData.map((_, i) =>
                        anomalies.find(a => a.index === i) ? 7 : 3
                    ),
                    pointBackgroundColor: actualData.map((_, i) =>
                        anomalies.find(a => a.index === i) ? '#DC3545' : '#17A2B8'
                    ),
                    pointBorderColor: actualData.map((_, i) =>
                        anomalies.find(a => a.index === i) ? '#fff' : '#fff'
                    ),
                    pointBorderWidth: actualData.map((_, i) =>
                        anomalies.find(a => a.index === i) ? 2 : 2
                    ),
                    pointHoverRadius: 7
                },
                {
                    label: 'Predicted (LSTM)',
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
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: {
                    display: true,
                    position: 'bottom',
                    labels: {
                        usePointStyle: true,
                        padding: 20,
                        font: { size: 13, family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto' }
                    }
                },
                tooltip: {
                    enabled: true,
                    backgroundColor: 'rgba(0,0,0,0.8)',
                    padding: 12,
                    titleFont: { size: 14, weight: 'bold' },
                    bodyFont:  { size: 13 },
                    displayColors: true,
                    callbacks: {
                        label: (context) => {
                            let label = context.dataset.label || '';
                            if (label) label += ': ';
                            label += Math.round(context.parsed.y).toLocaleString() + ' Gal';
                            return label;
                        },
                        afterBody: (tooltipItems) => {
                            const absIdx  = tooltipItems[0].dataIndex + sliceOffset;
                            const anomaly = anomalies.find(a => a.index === absIdx);
                            if (anomaly) {
                                return [
                                    '',
                                    '\u26a0\ufe0f Anomaly Detected',
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
                    grid: { display: false },
                    ticks: { font: { size: 12 }, maxRotation: 45, minRotation: 0 }
                },
                y: {
                    beginAtZero: false,
                    grid: { color: 'rgba(0,0,0,0.05)', drawBorder: false },
                    ticks: {
                        font: { size: 12 },
                        callback: (value) => value.toLocaleString() + ' Gal'
                    },
                    title: {
                        display: true,
                        text: 'Daily Water Consumption (Gallons)',
                        font: { size: 13, weight: 'bold' }
                    }
                }
            }
        },
        plugins: [anomalyBackgroundPlugin]
    });

    // Add anomaly badge if any detected
    addWaterAnomalyBadge(anomalies, timestamps);

    // Helper: build pointRadius / pointBg arrays for a given absolute offset
    function makePointStyles(offset, len) {
        const radii = [];
        const colors = [];
        for (let i = 0; i < len; i++) {
            const absIdx = offset + i;
            const isAnom = anomalies.some(a => a.index === absIdx);
            radii.push(isAnom ? 8 : 3);
            colors.push(isAnom ? '#DC3545' : '#17A2B8');
        }
        return { radii, colors };
    }

    // Time-filter dropdown slices the last N data-points
    const filterEl = document.getElementById('waterTimeFilter');
    if (filterEl) {
        filterEl.addEventListener('change', function() {
            const n         = parseInt(this.value, 10) || 60;
            const total     = timestamps.length;
            const sliceFrom = Math.max(0, total - n);
            sliceOffset     = sliceFrom;   // update closure for tooltip

            const slicedTS  = timestamps.slice(sliceFrom);
            const slicedA   = actualData.slice(sliceFrom);
            const slicedP   = predictedData.slice(sliceFrom);
            const { radii, colors } = makePointStyles(sliceFrom, slicedA.length);

            chart.data.labels                               = slicedTS;
            chart.data.datasets[0].data                    = slicedA;
            chart.data.datasets[0].pointRadius             = radii;
            chart.data.datasets[0].pointBackgroundColor    = colors;
            chart.data.datasets[1].data                    = slicedP;
            chart.update();
        });
    }
}

function addWaterAnomalyBadge(anomalies, timestamps) {
    const card = document.getElementById('waterTimeseriesCard');
    if (!card || anomalies.length === 0) return;
    if (card.querySelector('.anomaly-label')) return;

    const badge = document.createElement('div');
    badge.className = 'anomaly-label';
    badge.style.cssText = [
        'position:absolute', 'top:80px', 'left:50%',
        'transform:translateX(-50%)',
        'background-color:rgba(220,53,69,0.95)', 'color:white',
        'padding:8px 16px', 'border-radius:8px',
        'font-size:0.875rem', 'font-weight:600',
        'box-shadow:0 2px 8px rgba(0,0,0,0.2)', 'z-index:10'
    ].join(';');

    const primary     = anomalies[0];
    const anomalyTime = timestamps[primary.index] || 'Unknown';
    badge.textContent = `Anomaly: ${primary.type} detected at ${anomalyTime}`;

    card.style.position = 'relative';
    card.appendChild(badge);
}
