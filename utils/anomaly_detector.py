"""
EcoVision – Anomaly Detection Engine
======================================
Phase 4: Residual Analysis → 2σ Thresholding → SQLite Notifications

Three scanners
--------------
  scan_energy()  – LSTM-predicted vs actual hourly campus energy (kWh)
  scan_water()   – Rule-based: pressure drops + flow spikes (hourly)
  scan_waste()   – RF-predicted vs actual daily waste (lbs)

Severity tiers
--------------
  critical  |residual| > 3σ
  high      |residual| > 2σ   (minimum threshold)
  medium    |residual| > 1.5σ (logged but lower priority)

De-duplication
--------------
  Before inserting, the scanner checks whether an 'active' alert already
  exists for the same resource + timestamp bucket to avoid flooding the
  table on repeated scans.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Optional

from utils.db import (
    init_db, bulk_insert_alerts,
    get_active_alerts, get_alert_counts,
    DEFAULT_DB_PATH,
)


# ─────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────

SIGMA_MEDIUM   = 1.5   # log but lower priority
SIGMA_HIGH     = 2.0   # main alert threshold
SIGMA_CRITICAL = 3.0   # urgent

# Rolling window for computing local std (captures recent volatility)
ENERGY_ROLLING_WINDOW = 168   # 1 week in hours
WATER_ROLLING_WINDOW  = 168
WASTE_ROLLING_WINDOW  = 14    # 2 weeks in days

# How far back to scan on each run (keeps scans fast)
ENERGY_SCAN_HOURS  = 48
WATER_SCAN_HOURS   = 48
WASTE_SCAN_DAYS    = 14


# ─────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────

def _severity(sigma_val: float) -> Optional[str]:
    """Map σ magnitude to severity label, or None if below threshold."""
    if sigma_val >= SIGMA_CRITICAL:
        return 'critical'
    if sigma_val >= SIGMA_HIGH:
        return 'high'
    if sigma_val >= SIGMA_MEDIUM:
        return 'medium'
    return None


def _alert_type(residual: float, resource: str) -> str:
    """Choose a human-readable alert type from sign + resource."""
    if resource == 'energy':
        return 'phantom_load'    if residual > 0 else 'unusual_drop'
    if resource == 'water':
        return 'water_spike'     if residual > 0 else 'pressure_drop'
    if resource == 'waste':
        return 'waste_spike'     if residual > 0 else 'waste_drop'
    return 'anomaly'


def _dedup_key(timestamp_str: str, resource: str, metric: str) -> str:
    """
    Coarse deduplication key:  round energy/water to the hour,
    waste to the day.
    """
    ts = pd.Timestamp(timestamp_str)
    if resource in ('energy', 'water'):
        bucket = ts.floor('h').isoformat()
    else:
        bucket = ts.normalize().isoformat()
    return f"{resource}|{metric}|{bucket}"


# ─────────────────────────────────────────────────────────────
#  AnomalyDetector
# ─────────────────────────────────────────────────────────────

class AnomalyDetector:
    """
    Orchestrates all anomaly scans and persists flags to SQLite.

    Parameters
    ----------
    registry : ModelRegistry
        Loaded model registry (from model_trainer.py).
    pipeline : DataPipeline
        Processed feature-engineered data frames.
    db_path  : str
        Path to the SQLite database file.
    """

    def __init__(self, registry, pipeline, db_path: str = DEFAULT_DB_PATH):
        self.registry = registry
        self.pipeline = pipeline
        self.db_path  = db_path

        # Initialise DB schema (no-op if already exists)
        init_db(db_path)

        # Cache of existing active alert keys to skip duplicates
        self._existing_keys: set = set()
        self._refresh_existing_keys()

    # ── dedup cache ─────────────────────────────────────────

    def _refresh_existing_keys(self):
        active = get_active_alerts(limit=5000, db_path=self.db_path)
        self._existing_keys = {
            _dedup_key(a['timestamp'], a['resource'], a['metric'])
            for a in active
        }

    def _is_duplicate(self, ts: str, resource: str, metric: str) -> bool:
        return _dedup_key(ts, resource, metric) in self._existing_keys

    # ── Energy scanner ───────────────────────────────────────

    def scan_energy(self) -> list:
        """
        Compare LSTM-predicted daily kWh → distribute into hourly via
        historical profile → flag residuals > threshold.
        """
        alerts = []

        if self.pipeline is None or self.registry is None:
            return alerts

        try:
            energy_h = self.pipeline.energy_hourly.copy()
            energy_d = self.pipeline.energy_daily.copy()

            # Get hourly predicted vs actual
            pred_df = self.registry.get_hourly_predictions(energy_h, energy_d)
            # pred_df columns: timestamp, actual, predicted

            # Focus on last N hours
            cutoff = pred_df['timestamp'].max() - timedelta(hours=ENERGY_SCAN_HOURS)
            window = pred_df[pred_df['timestamp'] >= cutoff].copy()

            # Compute residuals
            window['residual'] = window['actual'] - window['predicted']

            # Rolling std over longer lookback for stable σ estimate
            window['roll_std'] = (
                window['residual']
                .rolling(ENERGY_ROLLING_WINDOW, min_periods=12)
                .std()
                .bfill()
                .fillna(window['residual'].std())
            )
            window['sigma_val'] = (
                window['residual'].abs() / window['roll_std'].replace(0, np.nan)
            ).fillna(0)

            for _, row in window.iterrows():
                sev = _severity(row['sigma_val'])
                if sev is None:
                    continue
                ts_str = row['timestamp'].isoformat()
                if self._is_duplicate(ts_str, 'energy', 'energy_kwh'):
                    continue

                alert = {
                    'timestamp':  ts_str,
                    'resource':   'energy',
                    'building':   None,
                    'metric':     'energy_kwh',
                    'actual':     round(float(row['actual']),    2),
                    'predicted':  round(float(row['predicted']), 2),
                    'residual':   round(float(row['residual']),  2),
                    'sigma':      round(float(row['sigma_val']), 2),
                    'severity':   sev,
                    'alert_type': _alert_type(row['residual'], 'energy'),
                }
                alerts.append(alert)

        except Exception as e:
            print(f"[AnomalyDetector] Energy scan error: {e}")

        return alerts

    # ── Water scanner ────────────────────────────────────────

    def scan_water(self) -> list:
        """
        Rule-based water anomaly detection:
          • Flow spike:     gallons > mean + 2σ of rolling window
          • Pressure drop:  pressure_psi < mean - 2σ  (indicates possible leak)
        """
        alerts = []

        if self.pipeline is None:
            return alerts

        try:
            water_h = self.pipeline.water_hourly.copy()

            cutoff = water_h['timestamp'].max() - timedelta(hours=WATER_SCAN_HOURS)
            window = water_h[water_h['timestamp'] >= cutoff].copy()

            # Rolling stats over longer lookback
            for col, resource_type in [
                ('water_gallons', 'flow'),
                ('pressure_psi',  'pressure'),
            ]:
                roll_mean = (
                    water_h[col]
                    .rolling(WATER_ROLLING_WINDOW, min_periods=12)
                    .mean()
                    .iloc[-len(window):]
                    .reset_index(drop=True)
                )
                roll_std = (
                    water_h[col]
                    .rolling(WATER_ROLLING_WINDOW, min_periods=12)
                    .std()
                    .iloc[-len(window):]
                    .reset_index(drop=True)
                )

                win_reset = window.reset_index(drop=True)
                win_reset['_mean']    = roll_mean
                win_reset['_std']     = roll_std.replace(0, np.nan).bfill()
                win_reset['_residual']= win_reset[col] - win_reset['_mean']
                win_reset['_sigma']   = (
                    win_reset['_residual'].abs() / win_reset['_std']
                ).fillna(0)

                for _, row in win_reset.iterrows():
                    sev = _severity(row['_sigma'])
                    if sev is None:
                        continue

                    # For pressure: only flag drops (negative residual = potential leak)
                    if col == 'pressure_psi' and row['_residual'] >= 0:
                        continue

                    ts_str = pd.Timestamp(row['timestamp']).isoformat()
                    if self._is_duplicate(ts_str, 'water', col):
                        continue

                    alerts.append({
                        'timestamp':  ts_str,
                        'resource':   'water',
                        'building':   None,
                        'metric':     col,
                        'actual':     round(float(row[col]),          2),
                        'predicted':  round(float(row['_mean']),      2),
                        'residual':   round(float(row['_residual']),  2),
                        'sigma':      round(float(row['_sigma']),     2),
                        'severity':   sev,
                        'alert_type': 'water_spike' if col == 'water_gallons'
                                      else 'pressure_drop',
                    })

        except Exception as e:
            print(f"[AnomalyDetector] Water scan error: {e}")

        return alerts

    # ── Waste scanner ────────────────────────────────────────

    def scan_waste(self) -> list:
        """
        Compare RF-predicted daily waste (lbs) against actuals.
        Flags both overages (waste_spike) and unexpected drops (waste_drop).
        """
        alerts = []

        if self.pipeline is None or self.registry is None:
            return alerts

        try:
            waste_d = self.pipeline.waste_daily.copy()

            cutoff  = waste_d['date'].max() - timedelta(days=WASTE_SCAN_DAYS)
            window  = waste_d[waste_d['date'] >= cutoff].copy()

            # Get RF predictions for the window
            y_pred  = self.registry.predict_waste_daily(window)
            window  = window.reset_index(drop=True)
            window['predicted'] = np.clip(y_pred, 0, None)
            window['residual']  = window['total_waste_lbs'] - window['predicted']

            # Rolling std for σ estimate
            roll_std = (
                waste_d['total_waste_lbs']
                .rolling(WASTE_ROLLING_WINDOW, min_periods=3)
                .std()
                .iloc[-len(window):]
                .reset_index(drop=True)
                .replace(0, np.nan)
                .bfill()
                .fillna(window['total_waste_lbs'].std())
            )
            window['sigma_val'] = (
                window['residual'].abs() / roll_std
            ).fillna(0)

            for _, row in window.iterrows():
                sev = _severity(row['sigma_val'])
                if sev is None:
                    continue
                ts_str = pd.Timestamp(row['date']).isoformat()
                if self._is_duplicate(ts_str, 'waste', 'total_waste_lbs'):
                    continue

                alerts.append({
                    'timestamp':  ts_str,
                    'resource':   'waste',
                    'building':   None,
                    'metric':     'total_waste_lbs',
                    'actual':     round(float(row['total_waste_lbs']), 2),
                    'predicted':  round(float(row['predicted']),       2),
                    'residual':   round(float(row['residual']),        2),
                    'sigma':      round(float(row['sigma_val']),       2),
                    'severity':   sev,
                    'alert_type': _alert_type(row['residual'], 'waste'),
                })

        except Exception as e:
            print(f"[AnomalyDetector] Waste scan error: {e}")

        return alerts

    # ── Full scan orchestrator ───────────────────────────────

    def run_full_scan(self) -> dict:
        """
        Run all three scanners, persist new alerts to SQLite,
        and return a summary dict.
        """
        print("[AnomalyDetector] Starting full scan …")

        energy_alerts = self.scan_energy()
        water_alerts  = self.scan_water()
        waste_alerts  = self.scan_waste()

        all_new = energy_alerts + water_alerts + waste_alerts

        inserted = bulk_insert_alerts(all_new, db_path=self.db_path)
        self._refresh_existing_keys()   # keep dedup cache fresh

        counts = get_alert_counts(db_path=self.db_path)

        summary = {
            'scanned_at':       datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'new_alerts':       inserted,
            'energy_new':       len(energy_alerts),
            'water_new':        len(water_alerts),
            'waste_new':        len(waste_alerts),
            'active_critical':  counts.get('critical', 0),
            'active_high':      counts.get('high',     0),
            'active_medium':    counts.get('medium',   0),
            'active_total':     sum(counts.values()),
        }

        print(
            f"[AnomalyDetector] Scan complete → "
            f"{inserted} new alerts inserted  |  "
            f"active: critical={summary['active_critical']}  "
            f"high={summary['active_high']}  "
            f"medium={summary['active_medium']}"
        )
        return summary

    # ── Convenience getters (consumed by DataLoader / API) ───

    def get_active_alerts(
        self,
        limit:    int  = 50,
        resource: str  = None,
    ) -> list:
        return get_active_alerts(
            limit=limit,
            resource=resource,
            db_path=self.db_path,
        )

    def get_dashboard_alerts(self) -> list:
        """
        Return top-3 active alerts formatted for the dashboard template.
        Maps raw DB rows to the dict shape expected by dashboard.html.
        """
        rows = get_active_alerts(limit=3, db_path=self.db_path)
        if not rows:
            # Fallback so dashboard never renders empty
            return [
                {"severity": "high",   "location": "Science Lab B",
                 "issue": "Phantom Load",  "details": "Model scan pending"},
                {"severity": "medium", "location": "Library",
                 "issue": "Water Spike",   "details": "Model scan pending"},
            ]

        ALERT_LABEL = {
            'phantom_load':  'Phantom Load',
            'unusual_drop':  'Unusual Drop',
            'water_spike':   'Water Spike',
            'pressure_drop': 'Pressure Drop',
            'waste_spike':   'Waste Spike',
            'waste_drop':    'Waste Drop',
        }
        formatted = []
        for r in rows:
            unit  = ('kWh' if r['resource'] == 'energy'
                     else 'Gal' if r['resource'] == 'water'
                     else 'lbs')
            sign  = '+' if r['residual'] > 0 else ''
            formatted.append({
                'severity': r['severity'],
                'location': r.get('building') or r['resource'].title(),
                'issue':    ALERT_LABEL.get(r['alert_type'], r['alert_type']),
                'details':  (
                    f"{sign}{r['residual']:,.0f} {unit}  "
                    f"({r['sigma']:.1f}σ deviation)"
                ),
            })
        return formatted

    def get_counts(self) -> dict:
        return get_alert_counts(db_path=self.db_path)
