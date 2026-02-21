/**
 * AI Insights Dashboard - Visualization & Interaction Logic
 *
 * Simulation model (two-factor):
 *
 *  1. investmentCapacity  — how much reduction the budget can physically deliver
 *                           (diminishing-returns curve,  max ≈ 56 % at $200k)
 *  2. fundingAdequacy      — whether budget is proportionate to the stated target
 *                           ($2 200 needed per 1 % reduction)
 *
 *  achievedReduction = min(investmentCapacity, target) × executionFactor
 *  executionFactor    = 0.85 + 0.15 × fundingAdequacy   (85 %–100 % real-world execution)
 */

// ─── Campus baseline constants ────────────────────────────
const BASELINE_CO2      = 580;   // tonnes CO₂e / yr
const BASELINE_ENERGY_$ = 87000; // $/yr energy spend
const COST_PER_PCT      = 2200;  // $ needed per 1 % reduction target
const MAX_CAPACITY      = 56;    // max % achievable with any budget (physical ceiling)

// ─── Core simulation engine ───────────────────────────────
function runSimulation(budget, targetPct) {
    // 1. Investment capacity (exponential saturation curve)
    //    $0 → 0%  |  $65k → 33%  |  $130k → 47%  |  $200k → 54%
    const investmentCapacity = MAX_CAPACITY * (1 - Math.exp(-budget / 65000));

    // 2. Is the budget adequate for the stated target?
    const requiredBudget  = targetPct * COST_PER_PCT;
    const fundingAdequacy = requiredBudget > 0
        ? Math.min(1, budget / requiredBudget)
        : (budget > 0 ? 1 : 0);   // target=0 → nothing to fund

    // 3. What can realistically be achieved?
    const rawAchievable    = Math.min(investmentCapacity, targetPct);
    const executionFactor  = 0.85 + 0.15 * fundingAdequacy;
    const achievedReduction = rawAchievable * executionFactor;

    // 4. Derived metrics
    const co2Saved      = (achievedReduction / 100) * BASELINE_CO2;          // tonnes/yr
    const annualSavings = (achievedReduction / 100) * BASELINE_ENERGY_$;      // $/yr
    const payback       = annualSavings > 0 ? budget / annualSavings : null;  // yrs
    const gap           = Math.max(0, targetPct - achievedReduction);         // shortfall %

    // 5. Confidence tier
    let confidence, confColor;
    if      (fundingAdequacy >= 0.85) { confidence = 'High';   confColor = '#28A745'; }
    else if (fundingAdequacy >= 0.45) { confidence = 'Medium'; confColor = '#FFC107'; }
    else                              { confidence = 'Low';    confColor = '#DC3545'; }

    // 6. Budget utilisation advice
    const additionalNeeded = Math.max(0, requiredBudget - budget);

    return {
        achievedReduction, investmentCapacity, fundingAdequacy,
        co2Saved, annualSavings, payback, gap,
        confidence, confColor, additionalNeeded, requiredBudget,
    };
}

// ─── SVG Gauge with live needle ───────────────────────────
function renderGauge(achievedPct, targetPct) {
    const wrapper = document.getElementById('carbonGaugeWrapper');
    if (!wrapper) return;

    const cx = 150, cy = 155, R = 125, r = 78;

    // Convert reduction % (0-100) to gauge angle
    // 0% → 180° (left)  |  50% → 270° (top)  |  100% → 360° (right)
    function pctToAngle(pct) { return 180 + (Math.min(100, Math.max(0, pct)) / 100) * 180; }

    function ptAtAngle(deg, radius) {
        const rad = deg * Math.PI / 180;
        return [(cx + radius * Math.cos(rad)).toFixed(2),
                (cy + radius * Math.sin(rad)).toFixed(2)];
    }

    function arcPath(startDeg, endDeg, color) {
        const large = Math.abs(endDeg - startDeg) > 180 ? 1 : 0;
        const [ox, oy] = ptAtAngle(startDeg, R);  const [ix, iy] = ptAtAngle(startDeg, r);
        const [ox2, oy2] = ptAtAngle(endDeg, R);  const [ix2, iy2] = ptAtAngle(endDeg, r);
        return `<path d="M ${ox} ${oy} A ${R} ${R} 0 ${large} 1 ${ox2} ${oy2} ` +
               `L ${ix2} ${iy2} A ${r} ${r} 0 ${large} 0 ${ix} ${iy} Z" fill="${color}"/>`;
    }

    // Needle for ACHIEVED reduction
    const angle     = pctToAngle(achievedPct);
    const [nx, ny]  = ptAtAngle(angle, R - 10);
    const [bx1, by1]= ptAtAngle(angle + 90, 7);
    const [bx2, by2]= ptAtAngle(angle - 90, 7);
    const needleColor = achievedPct >= 33 ? '#28A745' : achievedPct >= 17 ? '#FFC107' : '#DC3545';

    // Target tick mark
    const tAngle = pctToAngle(targetPct);
    const [tx1, ty1] = ptAtAngle(tAngle, R + 2);
    const [tx2, ty2] = ptAtAngle(tAngle, R - 18);

    wrapper.innerHTML = `
        <svg viewBox="0 0 300 160" width="100%" xmlns="http://www.w3.org/2000/svg">
            <!-- zones -->
            ${arcPath(181, 240, 'rgba(220,53,69,0.80)')}
            ${arcPath(241, 300, 'rgba(255,193,7,0.80)')}
            ${arcPath(301, 359, 'rgba(40,167,69,0.80)')}

            <!-- target tick -->
            ${ targetPct > 0 ? `<line x1="${tx1}" y1="${ty1}" x2="${tx2}" y2="${ty2}"
                stroke="#555" stroke-width="2.5" stroke-dasharray="3,2"/>` : ''}

            <!-- needle -->
            <polygon points="${nx},${ny} ${bx1},${by1} ${bx2},${by2}"
                fill="${needleColor}" opacity="0.92"/>

            <!-- centre cap -->
            <circle cx="${cx}" cy="${cy}" r="8" fill="#fff"
                stroke="${needleColor}" stroke-width="2"/>
        </svg>`;
}

// ─── Sliders (live preview on input) ─────────────────────
function initSliders() {
    const budgetSlider    = document.getElementById('budgetSlider');
    const reductionSlider = document.getElementById('reductionSlider');
    const budgetValue     = document.getElementById('budgetValue');
    const reductionValue  = document.getElementById('reductionValue');
    const needleDisplay   = document.getElementById('gaugeNeedleValue');

    function fmtBudget(v) {
        return v >= 1000 ? '$' + (v / 1000).toFixed(0) + 'k' : '$' + v;
    }

    function updateSliderFill(slider) {
        const pct = ((slider.value - slider.min) / (slider.max - slider.min)) * 100;
        slider.style.setProperty('--fill', pct + '%');
    }

    budgetSlider.addEventListener('input', function () {
        budgetValue.textContent = fmtBudget(parseInt(this.value));
        updateSliderFill(this);
    });

    reductionSlider.addEventListener('input', function () {
        reductionValue.textContent = parseInt(this.value) + '%';
        updateSliderFill(this);
    });

    updateSliderFill(budgetSlider);
    updateSliderFill(reductionSlider);
}

// ─── Run Simulation Button (full results) ─────────────────
function initRunSimButton() {
    const btn     = document.getElementById('runSimBtn');
    const results = document.getElementById('simResults');
    if (!btn) return;

    btn.addEventListener('click', function () {
        const budget = parseInt(document.getElementById('budgetSlider').value);
        const target = parseInt(document.getElementById('reductionSlider').value);

        btn.disabled = true;
        btn.innerHTML = '<span class="material-icons" style="animation:spin 1s linear infinite">sync</span> Simulating…';

        setTimeout(() => {
            const r = runSimulation(budget, target);

            // Update gauge
            renderGauge(r.achievedReduction, target);
            document.getElementById('gaugeNeedleValue').textContent =
                '-' + r.achievedReduction.toFixed(1) + '%';

            // Fill results panel
            if (results) {
                const achieved   = r.achievedReduction.toFixed(1);
                const gap        = r.gap.toFixed(1);
                const co2        = r.co2Saved.toFixed(0);
                const savings    = (r.annualSavings / 1000).toFixed(1);
                const payback    = r.payback != null ? r.payback.toFixed(1) + ' yrs' : 'N/A';
                const funding    = (r.fundingAdequacy * 100).toFixed(0);
                const addlNeeded = r.additionalNeeded > 0
                    ? '$' + (r.additionalNeeded / 1000).toFixed(0) + 'k more needed to fully fund target'
                    : 'Budget is fully sufficient for this target';

                const gapHtml = r.gap > 0.5
                    ? `<span class="sim-gap">&#9888; ${gap}% short of ${target}% target</span>`
                    : `<span class="sim-ontrack">&#10003; On track to meet ${target}% target</span>`;

                results.innerHTML = `
                <div class="sim-achieved-row">
                    <div class="sim-achieved-number" style="color:${r.confColor}">
                        &minus;${achieved}%
                    </div>
                    <div class="sim-achieved-label">
                        Projected Reduction
                        ${gapHtml}
                    </div>
                </div>

                <div class="sim-metrics-grid">
                    <div class="sim-metric">
                        <span class="material-icons sim-metric-icon" style="color:#28A745">co2</span>
                        <div>
                            <div class="sim-metric-val">${co2} t</div>
                            <div class="sim-metric-label">CO₂e saved / yr</div>
                        </div>
                    </div>
                    <div class="sim-metric">
                        <span class="material-icons sim-metric-icon" style="color:#007BFF">savings</span>
                        <div>
                            <div class="sim-metric-val">$${savings}k</div>
                            <div class="sim-metric-label">Annual savings</div>
                        </div>
                    </div>
                    <div class="sim-metric">
                        <span class="material-icons sim-metric-icon" style="color:#FFC107">schedule</span>
                        <div>
                            <div class="sim-metric-val">${payback}</div>
                            <div class="sim-metric-label">Payback period</div>
                        </div>
                    </div>
                    <div class="sim-metric">
                        <span class="material-icons sim-metric-icon" style="color:${r.confColor}">verified</span>
                        <div>
                            <div class="sim-metric-val" style="color:${r.confColor}">${r.confidence}</div>
                            <div class="sim-metric-label">Confidence</div>
                        </div>
                    </div>
                    <div class="sim-metric">
                        <span class="material-icons sim-metric-icon" style="color:#6f42c1">account_balance</span>
                        <div>
                            <div class="sim-metric-val">${funding}%</div>
                            <div class="sim-metric-label">Funding adequacy</div>
                        </div>
                    </div>
                    <div class="sim-metric">
                        <span class="material-icons sim-metric-icon" style="color:#ea580c">bolt</span>
                        <div>
                            <div class="sim-metric-val">${r.investmentCapacity.toFixed(0)}%</div>
                            <div class="sim-metric-label">Investment capacity</div>
                        </div>
                    </div>
                </div>

                <div class="sim-advice" style="border-left-color:${r.confColor}">
                    <span class="material-icons" style="color:${r.confColor};font-size:1rem;vertical-align:middle;margin-right:4px">info</span>
                    ${addlNeeded}.
                    ${ r.gap > 0.5 ? ` To close the <strong>${gap}%</strong> gap, increase budget by <strong>$${(r.additionalNeeded/1000).toFixed(0)}k</strong> or lower target to <strong>${Math.floor(r.achievedReduction)}%</strong>.` : ' Great balance of investment and ambition.' }
                </div>`;

                results.style.display = 'block';
            }

            btn.disabled = false;
            btn.innerHTML = '<span class="material-icons">play_arrow</span> Run Simulation';
        }, 1000);
    });
}

// ─── Roadmap Action Buttons ───────────────────────────────
function initRoadmapButtons() {
    document.querySelectorAll('.btn-approve').forEach(btn => {
        btn.addEventListener('click', function () {
            this.innerHTML = '<span class="material-icons btn-icon">check</span> Approved!';
            this.style.backgroundColor = '#155724';
            this.disabled = true;
        });
    });

    document.querySelectorAll('.btn-simulate').forEach(btn => {
        btn.addEventListener('click', function () {
            document.querySelector('.simulator-card').scrollIntoView({ behavior: 'smooth' });
        });
    });
}

// ─── Spin animation ───────────────────────────────────────
function injectSpinStyle() {
    const style = document.createElement('style');
    style.textContent = `@keyframes spin { from{ transform:rotate(0deg) } to{ transform:rotate(360deg) } }`;
    document.head.appendChild(style);
}

// ─── Init ─────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', function () {
    // Needle starts at far-left (0 % reduction). No value shown until Run is clicked.
    renderGauge(0, 0);
    document.getElementById('gaugeNeedleValue').textContent = '—';

    initSliders();
    initRunSimButton();
    initRoadmapButtons();
    injectSpinStyle();
});
