/**
 * Waste Tracking & Diversion Dashboard - Visualization Logic
 */

// Initialize Diversion Rate Ring Chart
function initDiversionRingChart() {
    const ctx = document.getElementById('diversionRingChart').getContext('2d');
    
    const diversionRate = wasteData.diversionRate;
    const remaining = 100 - diversionRate;
    
    new Chart(ctx, {
        type: 'doughnut',
        data: {
            datasets: [{
                data: [diversionRate, remaining],
                backgroundColor: [
                    'rgba(40, 167, 69, 0.8)',  // Green for diverted
                    'rgba(222, 226, 230, 0.3)'  // Light gray for remainder
                ],
                borderWidth: 0,
                cutout: '75%'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    enabled: false
                }
            }
        }
    });
}

// Initialize Waste Composition Stacked Bar Chart
function initWasteCompositionChart() {
    const ctx = document.getElementById('wasteCompositionChart').getContext('2d');
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: wasteData.compositionLabels,
            datasets: [
                {
                    label: 'Landfill',
                    data: wasteData.landfillData,
                    backgroundColor: 'rgba(140, 146, 172, 0.8)',
                    borderColor: 'rgba(140, 146, 172, 1)',
                    borderWidth: 1
                },
                {
                    label: 'Recycling',
                    data: wasteData.recyclingData,
                    backgroundColor: 'rgba(0, 123, 255, 0.8)',
                    borderColor: 'rgba(0, 123, 255, 1)',
                    borderWidth: 1
                },
                {
                    label: 'Compost',
                    data: wasteData.compostData,
                    backgroundColor: 'rgba(40, 167, 69, 0.8)',
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
                            return context.dataset.label + ': ' + context.parsed.y + ' Tons';
                        }
                    }
                }
            },
            scales: {
                x: {
                    stacked: true,
                    grid: {
                        display: false
                    }
                },
                y: {
                    stacked: true,
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Tons'
                    },
                    ticks: {
                        callback: function(value) {
                            return value + ' Tons';
                        }
                    }
                }
            }
        }
    });
}

// Initialize all visualizations on page load
document.addEventListener('DOMContentLoaded', function() {
    initDiversionRingChart();
    initWasteCompositionChart();
});
