/**
 * EcoVision Dashboard JavaScript
 * Handles interactive charts, maps, and real-time updates
 */

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initSustainabilityGauge();
    initWaterSparkline();
    initCampusMap();
});

/**
 * Initialize Sustainability Score Gauge Chart
 */
function initSustainabilityGauge() {
    const ctx = document.getElementById('sustainabilityGauge');
    if (!ctx) return;
    
    const score = dashboardData.score;
    const maxScore = 100;
    const percentage = (score / maxScore) * 100;
    
    // Determine color based on score
    let gaugeColor = '#28A745'; // Green
    if (score < 50) {
        gaugeColor = '#DC3545'; // Red
    } else if (score < 75) {
        gaugeColor = '#FFC107'; // Yellow
    }
    
    new Chart(ctx, {
        type: 'doughnut',
        data: {
            datasets: [{
                data: [score, maxScore - score],
                backgroundColor: [gaugeColor, '#E9ECEF'],
                borderWidth: 0,
                circumference: 180,
                rotation: 270
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '75%',
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

/**
 * Initialize Water Usage Sparkline
 */
function initWaterSparkline() {
    const ctx = document.getElementById('waterSparkline');
    if (!ctx) return;
    
    const data = dashboardData.waterSparkline || [];
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array(data.length).fill(''),
            datasets: [{
                data: data,
                borderColor: '#28A745',
                backgroundColor: 'rgba(40, 167, 69, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4,
                pointRadius: 0,
                pointHoverRadius: 4,
                pointBackgroundColor: '#28A745'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    enabled: true,
                    displayColors: false,
                    callbacks: {
                        title: () => '',
                        label: (context) => {
                            const value = context.parsed.y;
                            return `${(value / 1000).toFixed(1)}k Gal`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    display: false
                },
                y: {
                    display: false
                }
            },
            interaction: {
                intersect: false,
                mode: 'index'
            }
        }
    });
}

/**
 * Initialize Campus Resource Map with Leaflet
 */
function initCampusMap() {
    const mapElement = document.getElementById('campusMap');
    if (!mapElement) return;
    
    // Create map centered on campus (default coordinates)
    const map = L.map('campusMap', {
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
    
    // Define custom marker icons
    const normalIcon = L.divIcon({
        className: 'custom-marker',
        html: '<div style="background-color: #28A745; width: 30px; height: 30px; border-radius: 50% 50% 50% 0; transform: rotate(-45deg); border: 3px solid white; box-shadow: 0 2px 8px rgba(0,0,0,0.3);"><div style="transform: rotate(45deg); color: white; text-align: center; line-height: 24px; font-size: 16px;">📍</div></div>',
        iconSize: [30, 30],
        iconAnchor: [15, 30]
    });
    
    const criticalIcon = L.divIcon({
        className: 'custom-marker',
        html: '<div style="background-color: #DC3545; width: 30px; height: 30px; border-radius: 50% 50% 50% 0; transform: rotate(-45deg); border: 3px solid white; box-shadow: 0 2px 8px rgba(0,0,0,0.3); animation: marker-pulse 2s infinite;"><div style="transform: rotate(45deg); color: white; text-align: center; line-height: 24px; font-size: 16px;">⚠️</div></div>',
        iconSize: [30, 30],
        iconAnchor: [15, 30]
    });
    
    const warningIcon = L.divIcon({
        className: 'custom-marker',
        html: '<div style="background-color: #FFC107; width: 30px; height: 30px; border-radius: 50% 50% 50% 0; transform: rotate(-45deg); border: 3px solid white; box-shadow: 0 2px 8px rgba(0,0,0,0.3);"><div style="transform: rotate(45deg); color: white; text-align: center; line-height: 24px; font-size: 16px;">⚡</div></div>',
        iconSize: [30, 30],
        iconAnchor: [15, 30]
    });
    
    // Add markers from data
    const markers = dashboardData.mapMarkers || [];
    markers.forEach(markerData => {
        let icon = normalIcon;
        if (markerData.status === 'critical') {
            icon = criticalIcon;
        } else if (markerData.status === 'warning') {
            icon = warningIcon;
        }
        
        const marker = L.marker([markerData.lat, markerData.lon], { icon: icon })
            .addTo(map)
            .bindPopup(`
                <div style="font-family: Arial, sans-serif;">
                    <strong style="font-size: 14px;">${markerData.label}</strong><br>
                    <span style="color: #6C757D; font-size: 12px;">Status: ${markerData.status}</span>
                </div>
            `);
    });
    
    // Add CSS animation for pulsing markers
    const style = document.createElement('style');
    style.textContent = `
        @keyframes marker-pulse {
            0%, 100% {
                transform: rotate(-45deg) scale(1);
                opacity: 1;
            }
            50% {
                transform: rotate(-45deg) scale(1.1);
                opacity: 0.8;
            }
        }
    `;
    document.head.appendChild(style);
}

/**
 * API Helper Functions
 */
async function fetchDashboardStats() {
    try {
        const response = await fetch('/api/stats');
        const data = await response.json();
        updateDashboardMetrics(data);
    } catch (error) {
        console.error('Error fetching dashboard stats:', error);
    }
}

function updateDashboardMetrics(data) {
    // Update metrics dynamically if needed
    console.log('Dashboard metrics:', data);
}

/**
 * Real-time Alert Updates (optional WebSocket integration)
 */
function initRealTimeUpdates() {
    // Poll for new alerts every 30 seconds
    setInterval(async () => {
        try {
            const response = await fetch('/api/alerts');
            const data = await response.json();
            updateAlertsList(data.alerts);
        } catch (error) {
            console.error('Error fetching alerts:', error);
        }
    }, 30000);
}

function updateAlertsList(alerts) {
    const anomalyList = document.querySelector('.anomaly-list');
    if (!anomalyList) return;
    
    // Update notification badge
    const badge = document.querySelector('.notification-badge');
    if (badge) {
        badge.textContent = alerts.length;
    }
    
    // Update alert count
    const alertNumber = document.querySelector('.alert-number');
    if (alertNumber) {
        alertNumber.textContent = alerts.length;
    }
}

/**
 * Search Functionality
 */
document.querySelector('.search-input')?.addEventListener('input', function(e) {
    const searchTerm = e.target.value.toLowerCase();
    console.log('Searching for:', searchTerm);
    // Implement search logic here
});

/**
 * Utility Functions
 */
function formatNumber(num) {
    return new Intl.NumberFormat('en-US').format(num);
}

function formatCurrency(num) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 0
    }).format(num);
}

// Initialize real-time updates (optional)
// initRealTimeUpdates();


// ══════════════════════════════════════════════════════════════
//  NOTIFICATION PANEL
// ══════════════════════════════════════════════════════════════
(function initNotificationPanel() {
    const bell       = document.getElementById('notificationBell');
    const panel      = document.getElementById('notifPanel');
    const body       = document.getElementById('notifBody');
    const badge      = document.getElementById('notifBadge');
    const countBadge = document.getElementById('notifCountBadge');
    const markAllBtn = document.getElementById('markAllRead');

    if (!bell || !panel) return;

    let isOpen    = false;
    let allAlerts = [];

    /* ── toggle on bell click ── */
    bell.addEventListener('click', (e) => {
        e.stopPropagation();
        isOpen = !isOpen;
        panel.style.display = isOpen ? 'flex' : 'none';
        if (isOpen) loadAlerts();
    });

    /* ── close on outside click ── */
    document.addEventListener('click', (e) => {
        if (isOpen && !bell.contains(e.target)) {
            isOpen = false;
            panel.style.display = 'none';
        }
    });

    /* ── mark all read ── */
    markAllBtn?.addEventListener('click', async (e) => {
        e.stopPropagation();
        await Promise.all(allAlerts.map(a => acknowledgeAlert(a.id)));
        allAlerts = [];
        body.innerHTML = renderEmpty();
        updateBadge(0);
    });

    /* ── fetch and render ── */
    async function loadAlerts() {
        body.innerHTML = '<div class="notif-loading"><span class="material-icons" style="font-size:1.3rem;vertical-align:middle;margin-right:6px;animation:spin 1s linear infinite">refresh</span>Loading…</div>';
        try {
            const res   = await fetch('/api/alerts?limit=50');
            const data  = await res.json();
            allAlerts   = data.alerts || [];
            updateBadge(allAlerts.length);
            body.innerHTML = renderAlerts(allAlerts);
            attachAckButtons();
        } catch (err) {
            body.innerHTML = '<div class="notif-loading">⚠️ Failed to load alerts.</div>';
        }
    }

    function renderAlerts(alerts) {
        if (!alerts.length) return renderEmpty();

        const LABELS = {
            critical: '🔴 Critical',
            high:     '🟠 High Priority',
            medium:   '🟡 Medium',
            low:      '🟢 Informational',
        };

        // group by severity (ordered)
        const order  = ['critical', 'high', 'medium', 'low'];
        const groups = {};
        alerts.forEach(a => {
            const s = a.severity || 'low';
            (groups[s] = groups[s] || []).push(a);
        });

        let html = '';
        for (const sev of order) {
            const items = groups[sev];
            if (!items || !items.length) continue;
            html += `<div class="notif-group-label">${LABELS[sev]} <span style="opacity:.6">(${items.length})</span></div>`;
            items.forEach(a => { html += renderItem(a); });
        }
        return html;
    }

    function renderItem(a) {
        const icon     = resourceEmoji(a.resource);
        const title    = formatTitle(a.alert_type, a.resource);
        const building = a.building ? `<span>${escHtml(a.building)}</span><span class="meta-dot"></span>` : '';
        const sigma    = a.sigma != null
            ? `<span class="meta-sigma">${Number(a.sigma).toFixed(2)}&sigma;</span><span class="meta-dot"></span>`
            : '';
        const actual   = a.actual != null
            ? `<span>actual ${formatVal(a.actual, a.resource)}</span><span class="meta-dot"></span>`
            : '';
        const time     = `<span>${relativeTime(a.created_at || a.timestamp)}</span>`;

        return `
        <div class="notif-item" data-id="${a.id}">
            <div class="notif-icon severity-${a.severity}">${icon}</div>
            <div class="notif-content">
                <div class="notif-item-title">${escHtml(title)}</div>
                <div class="notif-item-meta">${building}${sigma}${actual}${time}</div>
            </div>
            <button class="notif-ack-btn" data-id="${a.id}" title="Acknowledge">&#10003;</button>
        </div>`;
    }

    function renderEmpty() {
        return `
        <div class="notif-empty">
            <span class="material-icons">check_circle_outline</span>
            All clear! No active alerts.
        </div>`;
    }

    function attachAckButtons() {
        body.querySelectorAll('.notif-ack-btn').forEach(btn => {
            btn.addEventListener('click', async (e) => {
                e.stopPropagation();
                const id   = btn.dataset.id;
                btn.disabled = true;
                btn.textContent = '…';
                await acknowledgeAlert(id);
                const item = body.querySelector(`.notif-item[data-id="${id}"]`);
                if (item) {
                    item.style.opacity    = '0.4';
                    item.style.transition = 'opacity 0.2s';
                    setTimeout(() => item.remove(), 220);
                }
                allAlerts = allAlerts.filter(a => String(a.id) !== String(id));
                updateBadge(allAlerts.length);
                setTimeout(() => {
                    cleanEmptyGroups();
                    if (!allAlerts.length) body.innerHTML = renderEmpty();
                }, 250);
            });
        });
    }

    async function acknowledgeAlert(id) {
        try {
            await fetch(`/api/anomalies/${id}/acknowledge`, { method: 'POST' });
        } catch (_) { /* silent */ }
    }

    function cleanEmptyGroups() {
        body.querySelectorAll('.notif-group-label').forEach(label => {
            const next = label.nextElementSibling;
            if (!next || next.classList.contains('notif-group-label')) label.remove();
        });
    }

    function updateBadge(count) {
        const n = Math.max(0, count);
        if (badge)      { badge.textContent      = n; badge.style.display = n > 0 ? '' : 'none'; }
        if (countBadge) { countBadge.textContent = n; }
    }

    /* ── helpers ─────────────────────────────────────────── */
    function resourceEmoji(resource) {
        return { energy: '⚡', water: '💧', waste: '♻️' }[resource] || '🔔';
    }

    function formatTitle(alertType, resource) {
        const res  = resource  ? resource.charAt(0).toUpperCase() + resource.slice(1) : '';
        const type = alertType ? alertType.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()) : 'Anomaly';
        return `${res} ${type}`;
    }

    function formatVal(val, resource) {
        const n = Number(val).toFixed(1);
        if (resource === 'energy') return `${n} kWh`;
        if (resource === 'water')  return `${n} gal`;
        if (resource === 'waste')  return `${n} lbs`;
        return n;
    }

    function relativeTime(iso) {
        if (!iso) return '';
        const diff = Date.now() - new Date(iso).getTime();
        const m    = Math.floor(diff / 60000);
        if (m <  2)  return 'just now';
        if (m < 60)  return `${m}m ago`;
        const h = Math.floor(m / 60);
        if (h < 24)  return `${h}h ago`;
        return `${Math.floor(h / 24)}d ago`;
    }

    function escHtml(str) {
        return String(str).replace(/[&<>"']/g,
            c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
    }

    /* ── spin keyframe for loading icon ── */
    const style = document.createElement('style');
    style.textContent = '@keyframes spin { to { transform: rotate(360deg); } }';
    document.head.appendChild(style);
})();
