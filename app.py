"""
EcoVision - Environmental Intelligence Platform
Flask Backend for Campus Sustainability Dashboard
"""

from flask import Flask, render_template, jsonify, request
import sys
import os

# Add utils to path
sys.path.append(os.path.dirname(__file__))
from utils.data_loader      import DataLoader
from utils.anomaly_detector import AnomalyDetector

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'ecovision-sustainability-2026')

# Initialize data loader
data_loader = DataLoader(dataset_path='dataset')
print("Loading datasets...")
data_loader.load_all_data()
print("Data loaded successfully!")

# Initialize anomaly detector and run first scan
detector = AnomalyDetector(
    registry = data_loader.registry,
    pipeline = data_loader.pipeline,
)
print("Running anomaly scan...")
_scan_summary = detector.run_full_scan()
print(f"Anomaly scan complete: {_scan_summary['new_alerts']} new alerts, "
      f"{_scan_summary['active_total']} active total")

@app.route('/')
def dashboard():
    """Main dashboard view - Campus Overview"""
    
    # Get KPI metrics
    sustainability_score = data_loader.get_sustainability_score()
    energy_cost  = data_loader.get_energy_cost_mtd()
    water_usage  = data_loader.get_water_usage_sparkline()
    alerts       = detector.get_dashboard_alerts()
    map_markers  = data_loader.get_campus_map_markers()
    alert_counts = detector.get_counts()
    total_alerts = sum(alert_counts.values())

    context = {
        "page_title":    "Campus Overview",
        "score":         sustainability_score,
        "energy_cost":   energy_cost['value'],
        "energy_change": energy_cost['change'],
        "energy_status": energy_cost['status'],
        "water_usage":   water_usage['value'],
        "water_sparkline": water_usage['data_points'],
        "alerts_count":  total_alerts,
        "anomalies":     alerts[:3],
        "map_markers":   map_markers
    }
    
    return render_template('dashboard.html', **context)

@app.route('/energy')
def energy():
    """Energy Deep Dive page"""
    
    # Get energy-specific data
    timeseries_data = data_loader.get_energy_timeseries(hours=24)
    energy_metrics = data_loader.get_energy_metrics()
    recommendations = data_loader.get_energy_recommendations()
    
    context = {
        "page_title": "Energy Deep Dive",
        "timestamps": timeseries_data['timestamps'],
        "actual_usage": timeseries_data['actual'],
        "predicted_baseline": timeseries_data['predicted'],
        "anomalies": timeseries_data['anomalies'],
        "today_usage": energy_metrics['today_usage'],
        "month_forecast": energy_metrics['month_forecast'],
        "top_consumer": energy_metrics['top_consumer'],
        "recommendations": recommendations
    }
    
    return render_template('energy.html', **context)

@app.route('/water')
def water():
    """Water Management page"""
    
    # Get water-specific data
    water_metrics = data_loader.get_water_metrics()
    water_map = data_loader.get_water_map_markers()
    building_usage = data_loader.get_water_building_usage()
    alerts = data_loader.get_ai_alerts()

    # Get LSTM timeseries (actual vs predicted + anomalies for chart)
    try:
        ts = data_loader.registry.predict_water_timeseries(
            data_loader.pipeline, n_days=60
        )
    except Exception:
        ts = {'timestamps': [], 'actual': [], 'predicted': [], 'anomalies': [], 'dates_iso': []}

    # Merge DB rule-based water alerts into chart anomalies
    try:
        from utils.db import get_active_alerts
        db_alerts = get_active_alerts(limit=500)
        date_index = {d: i for i, d in enumerate(ts.get('dates_iso', []))}
        existing   = {a['index'] for a in ts['anomalies']}
        for alert in db_alerts:
            if alert.get('resource') != 'water':
                continue
            try:
                import pandas as _pd
                day_str = _pd.Timestamp(alert['timestamp']).strftime('%Y-%m-%d')
                idx = date_index.get(day_str)
                if idx is None or idx in existing:
                    continue
                ts['anomalies'].append({
                    'index':    int(idx),
                    'type':     alert.get('alert_type', 'water_spike')
                                       .replace('_', ' ').title(),
                    'severity': alert.get('severity', 'medium'),
                })
                existing.add(idx)
            except Exception:
                pass
    except Exception:
        pass  # DB injection is best-effort
    
    context = {
        "page_title": "Water Management",
        "daily_consumption": water_metrics['daily_consumption'],
        "active_leaks": water_metrics['active_leaks'],
        "recycled_water": water_metrics['recycled_water'],
        "map_markers": water_map,
        "building_labels": building_usage['labels'],
        "building_data_a": building_usage['building_a'],
        "building_data_b": building_usage['building_b'],
        "alerts_count": detector.get_counts().get('high', 0)
                        + detector.get_counts().get('critical', 0),
        # LSTM timeseries
        "water_ts_timestamps":  ts['timestamps'],
        "water_ts_actual":      ts['actual'],
        "water_ts_predicted":   ts['predicted'],
        "water_ts_anomalies":   ts['anomalies'],
    }
    
    return render_template('water.html', **context)

@app.route('/waste')
def waste():
    """Waste Tracking & Diversion page"""
    
    # Get waste-specific data
    waste_metrics = data_loader.get_waste_metrics()
    composition_data = data_loader.get_waste_composition_data()
    bin_alerts = data_loader.get_bin_fill_alerts()
    alerts = data_loader.get_ai_alerts()
    
    context = {
        "page_title": "Waste Tracking & Diversion",
        "total_waste": waste_metrics['total_waste'],
        "diversion_rate": waste_metrics['diversion_rate'],
        "diversion_rate_value": waste_metrics['diversion_rate_value'],
        "next_pickup": waste_metrics['next_pickup'],
        "composition_labels": composition_data['labels'],
        "landfill_data": composition_data['landfill'],
        "recycling_data": composition_data['recycling'],
        "compost_data": composition_data['compost'],
        "bin_alerts": bin_alerts,
        "alerts_count": sum(detector.get_counts().values())
    }
    
    return render_template('waste.html', **context)

@app.route('/ai-insights')
def ai_insights():
    """AI Insights page"""

    roadmap_rows = data_loader.get_ai_roadmap()
    models       = data_loader.get_ai_models_status()
    alert_counts = detector.get_counts()

    context = {
        "page_title":    "AI Insights",
        "roadmap_rows":  roadmap_rows,
        "models":        models,
        "alerts_count":  sum(alert_counts.values())
    }

    return render_template('ai_insights.html', **context)

@app.route('/settings')
def settings():
    """Platform Settings page"""
    return render_template('settings.html', page_title="Platform Settings")


@app.route('/api/sustainability-score')
def api_sustainability_score():
    """API endpoint for sustainability score"""
    return jsonify({"score": data_loader.get_sustainability_score()})

@app.route('/api/alerts')
def api_alerts():
    """Active alerts from SQLite notifications table."""
    resource = request.args.get('resource')   # optional filter
    limit    = int(request.args.get('limit', 50))
    alerts   = detector.get_active_alerts(limit=limit, resource=resource)
    return jsonify({'alerts': alerts, 'total': len(alerts)})


@app.route('/api/anomalies')
def api_anomalies():
    """Paginated active anomaly alerts with optional severity/resource filter."""
    from utils.db import get_active_alerts, get_alert_counts
    resource = request.args.get('resource')
    limit    = int(request.args.get('limit', 100))
    alerts   = get_active_alerts(limit=limit, resource=resource)
    counts   = get_alert_counts()
    return jsonify({
        'alerts':  alerts,
        'counts':  counts,
        'total':   sum(counts.values()),
    })


@app.route('/api/anomalies/scan', methods=['POST'])
def api_anomalies_scan():
    """Trigger a fresh anomaly scan on demand."""
    summary = detector.run_full_scan()
    return jsonify(summary)


@app.route('/api/anomalies/<int:alert_id>/resolve', methods=['POST'])
def api_resolve_alert(alert_id):
    """Mark a single alert as resolved."""
    from utils.db import resolve_alert
    ok = resolve_alert(alert_id)
    return jsonify({'resolved': ok, 'id': alert_id})


@app.route('/api/anomalies/<int:alert_id>/acknowledge', methods=['POST'])
def api_acknowledge_alert(alert_id):
    """Mark a single alert as acknowledged."""
    from utils.db import acknowledge_alert
    ok = acknowledge_alert(alert_id)
    return jsonify({'acknowledged': ok, 'id': alert_id})

@app.route('/api/energy-cost')
def api_energy_cost():
    """API endpoint for energy cost"""
    return jsonify(data_loader.get_energy_cost_mtd())

@app.route('/api/water-usage')
def api_water_usage():
    """API endpoint for water usage"""
    return jsonify(data_loader.get_water_usage_sparkline())

@app.route('/api/stats')
def api_stats():
    """API endpoint for daily statistics"""
    return jsonify(data_loader.get_daily_stats())

@app.route('/api/pipeline-health')
def api_pipeline_health():
    """Data pipeline health report – row counts, null checks, date ranges."""
    if data_loader.pipeline is None:
        return jsonify({"error": "Pipeline not initialised"}), 503
    return jsonify({
        "status":  "ok" if not data_loader.pipeline.load_errors else "degraded",
        "errors":  data_loader.pipeline.load_errors,
        "datasets": data_loader.pipeline.health_report()
    })


# ════════════════════════════════════════════════════════════
#  PREDICTION ENDPOINTS
# ════════════════════════════════════════════════════════════

@app.route('/predict/energy')
def predict_energy():
    """
    GET /predict/energy?days=7
    Returns next N days of energy (kWh) forecast from the loaded model.

    Response shape:
    {
      "model":    "energy_lstm",
      "days":     7,
      "forecast": [
        { "date": "2026-01-01", "kwh_forecast": 7820.5,
          "kwh_lower": 7200.0, "kwh_upper": 8440.0, "source": "lstm_mlp" },
        ...
      ],
      "meta": { "mape": 15.72, "trained_at": "..." }
    }
    """
    days = min(int(request.args.get('days', 7)), 30)   # cap at 30

    if not data_loader.registry.is_ready():
        return jsonify({"error": "Models not loaded. Run train_models.py first."}), 503

    forecast = data_loader.registry.forecast_energy(data_loader.pipeline, periods=days)
    metrics  = data_loader.registry.metrics.get('energy', {})

    return jsonify({
        "model":    data_loader.registry._energy_name,
        "days":     days,
        "forecast": forecast,
        "meta": {
            "mape":       metrics.get('mape'),
            "mae_kwh":    metrics.get('mae'),
            "trained_at": metrics.get('trained_at'),
        },
    })


@app.route('/predict/water')
def predict_water():
    """
    GET /predict/water?days=7
    Returns next N days of water (gallons) forecast using seasonal rolling model.

    Response shape:
    {
      "model":    "seasonal_rolling",
      "days":     7,
      "forecast": [
        { "date": "2026-01-01", "gallons_forecast": 19800.0,
          "gallons_lower": 17900.0, "gallons_upper": 21700.0,
          "source": "seasonal_rolling" },
        ...
      ]
    }
    """
    days = min(int(request.args.get('days', 7)), 30)

    if data_loader.pipeline is None:
        return jsonify({"error": "Pipeline not initialised"}), 503

    forecast = data_loader.registry.forecast_water(data_loader.pipeline, periods=days)

    return jsonify({
        "model":    "seasonal_rolling",
        "days":     days,
        "forecast": forecast,
        "meta": {
            "note": "Water LSTM in roadmap – using seasonal rolling baseline",
        },
    })


@app.route('/predict/waste')
def predict_waste():
    """
    GET /predict/waste?days=7
    Returns next N days of waste (lbs/day) forecast from Random Forest.

    Response shape:
    {
      "model":    "waste_rf",
      "days":     7,
      "forecast": [
        { "date": "2026-01-01", "lbs_forecast": 820.5,
          "lbs_lower": 697.4, "lbs_upper": 943.6,
          "source": "random_forest" },
        ...
      ],
      "meta": { "mape": 14.51, ... }
    }
    """
    days = min(int(request.args.get('days', 7)), 30)

    if not data_loader.registry.is_ready():
        return jsonify({"error": "Models not loaded. Run train_models.py first."}), 503

    forecast = data_loader.registry.forecast_waste(data_loader.pipeline, periods=days)
    metrics  = data_loader.registry.metrics.get('waste', {})

    return jsonify({
        "model":    "waste_rf",
        "days":     days,
        "forecast": forecast,
        "meta": {
            "mape":       metrics.get('mape'),
            "mae_lbs":    metrics.get('mae'),
            "top_feature": metrics.get('top_feature'),
            "trained_at": metrics.get('trained_at'),
        },
    })


# ════════════════════════════════════════════════════════════
#  STATUS ENDPOINTS
# ════════════════════════════════════════════════════════════

@app.route('/status/score')
def status_score():
    """
    GET /status/score
    Returns current Sustainability Score with a full per-category breakdown.

    Response shape:
    {
      "score": 82,
      "grade": "B+",
      "breakdown": {
        "energy":  { "score": 85, "weight": 0.4, "weighted": 34.0 },
        "water":   { "score": 78, "weight": 0.3, "weighted": 23.4 },
        "waste":   { "score": 82, "weight": 0.3, "weighted": 24.6 }
      },
      "alerts": { "critical": 0, "high": 2, "medium": 7 },
      "models_ready": true,
      "timestamp": "2026-02-20T18:00:00"
    }
    """
    from datetime import datetime as dt

    # Per-category scores derived from real pipeline data
    pipeline = data_loader.pipeline
    scores   = {"energy": 75, "water": 75, "waste": 75}   # fallback defaults

    if pipeline is not None:
        # Energy: compare last-30-day mean to previous-30-day mean
        e = pipeline.energy_daily['energy_kwh']
        if len(e) >= 60:
            recent_e  = e.iloc[-30:].mean()
            baseline_e= e.iloc[-60:-30].mean()
            delta_e   = (baseline_e - recent_e) / baseline_e   # positive = improvement
            scores['energy'] = int(min(100, max(0, 75 + delta_e * 100)))

        # Water: same logic
        w = pipeline.water_daily['water_gallons']
        if len(w) >= 60:
            recent_w  = w.iloc[-30:].mean()
            baseline_w= w.iloc[-60:-30].mean()
            delta_w   = (baseline_w - recent_w) / baseline_w
            scores['water'] = int(min(100, max(0, 75 + delta_w * 100)))

        # Waste: diversion rate (recycled / total) → scale to 0–100
        if 'diversion_rate' in pipeline.waste_daily.columns:
            avg_div = float(pipeline.waste_daily['diversion_rate'].tail(30).mean())
            scores['waste'] = int(min(100, avg_div * 100))

    WEIGHTS = {"energy": 0.4, "water": 0.3, "waste": 0.3}
    weighted_score = sum(scores[k] * WEIGHTS[k] for k in WEIGHTS)

    def grade(s):
        if s >= 90: return "A"
        if s >= 80: return "B+"
        if s >= 70: return "B"
        if s >= 60: return "C+"
        return "C"

    alert_counts = detector.get_counts()

    return jsonify({
        "score": round(weighted_score, 1),
        "grade": grade(weighted_score),
        "breakdown": {
            k: {
                "score":    scores[k],
                "weight":   WEIGHTS[k],
                "weighted": round(scores[k] * WEIGHTS[k], 1),
            }
            for k in WEIGHTS
        },
        "alerts":       alert_counts,
        "models_ready": data_loader.registry.is_ready(),
        "timestamp":    dt.utcnow().isoformat(timespec='seconds') + 'Z',
    })


@app.route('/status/models')
def status_models():
    """
    GET /status/models
    Returns loaded model names, MAPE, training date and readiness flag.

    Response shape:
    {
      "ready": true,
      "models": {
        "energy": { "name": "energy_lstm", "mape": 15.72, "trained_at": "..." },
        "waste":  { "name": "waste_rf",    "mape": 14.51, "trained_at": "..." },
        "water":  { "name": null, "status": "pending" }
      }
    }
    """
    reg = data_loader.registry
    em  = reg.metrics.get('energy', {})
    wm  = reg.metrics.get('waste',  {})

    return jsonify({
        "ready": reg.is_ready(),
        "models": {
            "energy": {
                "name":       reg._energy_name,
                "status":     "active" if reg.energy_model else "not_trained",
                "mape":       em.get('mape'),
                "mae":        em.get('mae'),
                "trained_at": em.get('trained_at'),
                "test_days":  em.get('test_days'),
            },
            "waste": {
                "name":       reg._waste_name,
                "status":     "active" if reg.waste_model else "not_trained",
                "mape":       wm.get('mape'),
                "mae":        wm.get('mae'),
                "trained_at": wm.get('trained_at'),
                "top_feature": wm.get('top_feature'),
                "test_days":  wm.get('test_days'),
            },
            "water": {
                "name":   None,
                "status": "pending",
                "note":   "Water LSTM planned for next phase",
            },
        },
    })


if __name__ == '__main__':
    port  = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV', 'production') != 'production'
    app.run(debug=debug, host='0.0.0.0', port=port)