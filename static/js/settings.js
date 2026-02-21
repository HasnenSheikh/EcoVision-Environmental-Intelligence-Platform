/**
 * Settings Page - Interaction Logic
 */

// ─── Tab Switching ────────────────────────────────────────
function switchTab(name) {
    // Hide all panes
    document.querySelectorAll('.tab-pane').forEach(p => p.style.display = 'none');
    // Deactivate all tabs
    document.querySelectorAll('.settings-tab').forEach(t => t.classList.remove('active'));

    // Show target pane and activate tab
    document.getElementById('pane_' + name).style.display = 'block';
    document.getElementById('tab_' + name).classList.add('active');
}

// ─── File Input Labels ────────────────────────────────────
function bindFileInputs() {
    const fileMap = [
        { inputId: 'upload_energy_logs', statusId: 'status_energy' },
        { inputId: 'upload_water_logs',  statusId: 'status_water'  },
        { inputId: 'upload_waste_logs',  statusId: 'status_waste'  },
        { inputId: 'upload_fuel_logs',   statusId: 'status_fuel'   }
    ];

    fileMap.forEach(({ inputId, statusId }) => {
        const input  = document.getElementById(inputId);
        const status = document.getElementById(statusId);
        if (!input || !status) return;

        input.addEventListener('change', function () {
            if (this.files && this.files.length > 0) {
                status.textContent = this.files[0].name;
                status.classList.add('file-chosen');
            } else {
                status.textContent = 'No file chosen';
                status.classList.remove('file-chosen');
            }
        });
    });
}

// ─── Import All Button ────────────────────────────────────
function bindImportButton() {
    const btn      = document.getElementById('importAllBtn');
    const feedback = document.getElementById('importFeedback');
    if (!btn || !feedback) return;

    btn.addEventListener('click', function () {
        const energy = document.getElementById('upload_energy_logs').files.length;
        const water  = document.getElementById('upload_water_logs').files.length;
        const waste  = document.getElementById('upload_waste_logs').files.length;
        const fuel   = document.getElementById('upload_fuel_logs').files.length;

        const total = energy + water + waste + fuel;

        if (total === 0) {
            showFeedback(feedback, 'error', '⚠️ Please choose at least one file before importing.');
            return;
        }

        // Simulate import
        btn.disabled = true;
        btn.innerHTML = '<span class="material-icons" style="animation:spin 1s linear infinite;">sync</span> Importing...';

        setTimeout(() => {
            btn.disabled = false;
            btn.innerHTML = '<span class="material-icons">cloud_upload</span> Import All Data';
            showFeedback(feedback, 'success',
                `✅ Successfully imported ${total} file${total > 1 ? 's' : ''}. Dashboard data updated.`);
        }, 1500);
    });
}

function showFeedback(el, type, msg) {
    el.className = 'import-feedback ' + type;
    el.textContent = msg;
    el.style.display = 'block';
    setTimeout(() => { el.style.display = 'none'; }, 5000);
}

// ─── Save Buttons ─────────────────────────────────────────
function bindSaveButtons() {
    document.querySelectorAll('.save-btn').forEach(btn => {
        btn.addEventListener('click', function () {
            const original = this.innerHTML;
            this.innerHTML = '<span class="material-icons">check</span> Saved!';
            this.style.backgroundColor = '#155724';
            setTimeout(() => {
                this.innerHTML = original;
                this.style.backgroundColor = '';
            }, 2000);
        });
    });
}

// ─── Spin style for loading icon ─────────────────────────
function injectSpinStyle() {
    const style = document.createElement('style');
    style.textContent = `@keyframes spin { from { transform:rotate(0deg); } to { transform:rotate(360deg); } }`;
    document.head.appendChild(style);
}

// ─── Dark Mode Toggle ─────────────────────────────────────
function bindDarkMode() {
    const toggle = document.getElementById('darkModeToggle');
    if (!toggle) return;

    // Sync toggle UI to current preference (applied by anti-flash script)
    toggle.checked = localStorage.getItem('ecovision-theme') === 'dark';

    toggle.addEventListener('change', function () {
        if (this.checked) {
            document.documentElement.setAttribute('data-theme', 'dark');
            localStorage.setItem('ecovision-theme', 'dark');
        } else {
            document.documentElement.removeAttribute('data-theme');
            localStorage.setItem('ecovision-theme', 'light');
        }
    });
}

// ─── Init ─────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', function () {
    bindFileInputs();
    bindImportButton();
    bindSaveButtons();
    injectSpinStyle();
    bindDarkMode();
});
