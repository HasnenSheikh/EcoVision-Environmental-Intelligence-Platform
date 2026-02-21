"""
Data Loading and Processing Utilities for EcoVision Dashboard
Handles CSV ingestion, feature engineering, and KPI calculations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

from utils.data_pipeline  import DataPipeline
from utils.model_trainer  import ModelRegistry


class DataLoader:
    def __init__(self, dataset_path='dataset'):
        self.dataset_path = dataset_path

        # Processed dataframes (set after load_all_data)
        self.energy_df    = None
        self.water_df     = None
        self.waste_df     = None
        self.emission_factors = {}

        # Full feature-engineered frames from pipeline
        self.pipeline : DataPipeline   = None

        # Trained model registry (populated after load_all_data)
        self.registry : ModelRegistry  = ModelRegistry()
        
    def load_all_data(self):
        """Run the full ingestion & engineering pipeline, then expose
        convenient shorthand dataframes for the rest of the loader."""
        self.pipeline = DataPipeline(self.dataset_path)
        ok = self.pipeline.run()

        # Expose raw (but cleaned) hourly frames for backwards-compat
        self.energy_df        = self.pipeline._raw_energy
        self.water_df         = self.pipeline._raw_water
        self.waste_df         = self.pipeline.waste_daily
        self.emission_factors = self.pipeline.emission_factors

        # Print health report
        report = self.pipeline.health_report()
        for name, stats in report.items():
            print(f"  [{name}] {stats}")

        # Load trained models (silently skipped if not yet trained)
        self.registry.load()

        return ok
    
    def get_sustainability_score(self):
        """Calculate overall sustainability score (0-100)"""
        # Simplified calculation based on efficiency metrics
        if self.energy_df is None or self.water_df is None or self.waste_df is None:
            return 82  # Default from mockup
        
        # Calculate sub-scores
        energy_efficiency = 85  # Placeholder
        water_efficiency = 78
        waste_diversion = 82
        
        # Weighted average
        score = int(0.4 * energy_efficiency + 0.3 * water_efficiency + 0.3 * waste_diversion)
        return score
    
    def get_energy_cost_mtd(self):
        """Calculate month-to-date energy cost"""
        if self.energy_df is None:
            return {"value": "$14,250", "change": "+2.4%", "status": "negative"}
        
        # Get current month data
        latest_date = self.energy_df['timestamp'].max()
        current_month = self.energy_df[
            self.energy_df['timestamp'].dt.month == latest_date.month
        ]
        
        total_kwh = current_month['energy_kwh'].sum()
        cost = total_kwh * 0.12  # $0.12 per kWh assumption
        
        # Calculate vs predicted (simplified)
        predicted_cost = cost * 0.976  # Assume 2.4% difference
        change_pct = ((cost - predicted_cost) / predicted_cost) * 100
        
        return {
            "value": f"${cost:,.0f}",
            "change": f"+{change_pct:.1f}%" if change_pct > 0 else f"{change_pct:.1f}%",
            "status": "negative" if change_pct > 0 else "positive"
        }
    
    def get_water_usage_sparkline(self):
        """Get water usage data for sparkline chart"""
        if self.water_df is None:
            return {
                "value": "1.2M Gal",
                "data_points": [1000000, 1100000, 1050000, 1200000, 1150000, 1250000, 1200000]
            }
        
        # Get last 7 days of data
        latest_date = self.water_df['timestamp'].max()
        last_7_days = self.water_df[
            self.water_df['timestamp'] >= (latest_date - timedelta(days=7))
        ]
        
        daily_usage = last_7_days.groupby(last_7_days['timestamp'].dt.date)['water_gallons'].sum()
        
        return {
            "value": f"{daily_usage.sum() / 1e6:.1f}M Gal",
            "data_points": daily_usage.tolist()[-7:]
        }
    
    def get_ai_alerts(self):
        """Detect anomalies and generate alerts"""
        alerts = []
        
        if self.energy_df is not None:
            # Simple anomaly detection: find values > 2 std dev
            recent_energy = self.energy_df.tail(100)
            mean_usage = recent_energy.groupby('building')['energy_kwh'].mean()
            std_usage = recent_energy.groupby('building')['energy_kwh'].std()
            
            for building in recent_energy['building'].unique():
                building_data = recent_energy[recent_energy['building'] == building]
                latest_usage = building_data['energy_kwh'].iloc[-1]
                
                if latest_usage > mean_usage[building] + 2 * std_usage[building]:
                    alerts.append({
                        "severity": "high",
                        "location": building.replace('_', ' '),
                        "issue": "Phantom Load",
                        "details": f"+{((latest_usage - mean_usage[building]) / mean_usage[building] * 100):.0f}% expected usage"
                    })
        
        # Add mock alerts if none detected
        if len(alerts) == 0:
            alerts = [
                {"severity": "high", "location": "Science Lab B", "issue": "Phantom Load", "details": "+40% expected usage"},
                {"severity": "high", "location": "Dorm 4", "issue": "HVAC Leak", "details": "Continuous runtime detected"},
                {"severity": "medium", "location": "Library", "issue": "Water Spike", "details": "Abnormal flow rate"}
            ]
        
        return alerts
    
    def get_campus_map_markers(self):
        """Generate markers for campus map"""
        alerts = self.get_ai_alerts()
        alert_buildings = [a['location'] for a in alerts if a['severity'] == 'high']
        
        markers = [
            {"lat": 34.0522, "lon": -118.2437, "status": "normal", "label": "Dorm 1", "type": "residential"},
            {"lat": 34.0525, "lon": -118.2440, "status": "critical" if "Science Lab B" in alert_buildings else "normal", 
             "label": "Science Lab B", "type": "academic"},
            {"lat": 34.0520, "lon": -118.2430, "status": "warning" if "Library" in alert_buildings else "normal", 
             "label": "Library", "type": "academic"},
            {"lat": 34.0518, "lon": -118.2445, "status": "critical" if "Dorm 4" in alert_buildings else "normal", 
             "label": "Dorm 4", "type": "residential"},
            {"lat": 34.0528, "lon": -118.2435, "status": "normal", "label": "Athletic Center", "type": "athletic"},
            {"lat": 34.0515, "lon": -118.2442, "status": "normal", "label": "Dining Hall", "type": "dining"},
        ]
        
        return markers
    
    def get_daily_stats(self):
        """Get today's key statistics"""
        stats = {
            "energy_today_kwh": 450,
            "water_today_gal": 42000,
            "waste_mtd_tons": 120,
            "diversion_rate": 45
        }
        
        if self.energy_df is not None:
            latest_date = self.energy_df['timestamp'].max()
            today_energy = self.energy_df[
                self.energy_df['timestamp'].dt.date == latest_date.date()
            ]
            stats["energy_today_kwh"] = int(today_energy['energy_kwh'].sum())
        
        if self.water_df is not None:
            latest_date = self.water_df['timestamp'].max()
            today_water = self.water_df[
                self.water_df['timestamp'].dt.date == latest_date.date()
            ]
            stats["water_today_gal"] = int(today_water['water_gallons'].sum())
        
        if self.waste_df is not None:
            mtd_waste = self.waste_df[
                self.waste_df['date'].dt.month == self.waste_df['date'].max().month
            ]
            stats["waste_mtd_tons"] = int(mtd_waste['total_waste_lbs'].sum() / 2000)  # Convert lbs to tons
            total_waste_lbs = mtd_waste['total_waste_lbs'].sum()
            recycled_lbs = mtd_waste['recycled_lbs'].sum()
            stats["diversion_rate"] = int((recycled_lbs / total_waste_lbs) * 100) if total_waste_lbs > 0 else 0
        
        return stats
    
    def get_energy_timeseries(self, hours=24):
        """Get actual vs predicted energy for the time-series chart.
        Uses real Prophet predictions when the model is loaded; otherwise
        falls back to a rolling-mean baseline."""
        if self.energy_df is None:
            return {
                "timestamps": ["12 AM", "1 AM", "2 AM", "3 AM", "4 AM",
                               "5 AM", "6 AM", "7 AM", "8 AM", "9 AM"],
                "actual":    [60, 45, 50, 350, 80, 90, 140, 160, 180, 140],
                "predicted": [80, 50, 60, 110, 90, 100, 130, 140, 150, 130],
                "anomalies": [{"index": 3, "type": "Phantom Load", "severity": "high"}],
                "source":    "mock",
            }

        # ── hourly actual data ──────────────────────────────
        latest_time = self.energy_df['timestamp'].max()
        start_time  = latest_time - timedelta(hours=hours)
        recent      = self.energy_df[self.energy_df['timestamp'] >= start_time].copy()

        hourly = (
            recent
            .groupby(recent['timestamp'].dt.floor('h'))
            .agg(energy_kwh=('energy_kwh', 'sum'))
            .reset_index()
        )

        # ── predicted column ───────────────────────────────
        if self.registry.is_ready() and self.pipeline is not None:
            try:
                pred_df = self.registry.get_hourly_predictions(
                    self.pipeline.energy_hourly,
                    self.pipeline.energy_daily,
                )
                pred_df = pred_df[
                    pred_df['timestamp'] >= start_time
                ].copy()
                # Merge into hourly
                pred_hourly = (
                    pred_df
                    .groupby(pred_df['timestamp'].dt.floor('h'))
                    .agg(predicted=('predicted', 'sum'))
                    .reset_index()
                )
                hourly = hourly.merge(pred_hourly, on='timestamp', how='left')
                hourly['predicted'] = hourly['predicted'].fillna(
                    hourly['energy_kwh'].rolling(3, min_periods=1).mean()
                )
                source = self.registry._energy_name or 'model'
            except Exception:
                hourly['predicted'] = hourly['energy_kwh'].rolling(3, min_periods=1).mean()
                source = 'rolling_mean'
        else:
            hourly['predicted'] = hourly['energy_kwh'].rolling(3, min_periods=1).mean()
            source = 'rolling_mean'

        # ── anomaly detection (2-sigma residual) ───────────
        hourly['deviation'] = (hourly['energy_kwh'] - hourly['predicted']).abs()
        threshold            = hourly['deviation'].std() * 2
        anomalies = [
            {
                "index":    int(i),
                "type":     "Phantom Load" if row['energy_kwh'] > row['predicted']
                            else "Unusual Drop",
                "severity": "high",
            }
            for i, (_, row) in enumerate(hourly.iterrows())
            if row['deviation'] > threshold
        ]

        return {
            "timestamps": [t.strftime('%I %p') for t in hourly['timestamp']],
            "actual":     hourly['energy_kwh'].tolist(),
            "predicted":  hourly['predicted'].tolist(),
            "anomalies":  anomalies,
            "source":     source,
        }
    
    def get_energy_metrics(self):
        """Get energy-specific KPI metrics.
        Month forecast uses Prophet 30-day ahead prediction when available."""
        stats = self.get_daily_stats()

        metrics = {
            "today_usage":    f"{stats['energy_today_kwh']} kWh",
            "month_forecast": "+12% Overage",
            "top_consumer":   "Science Lab B",
        }

        if self.energy_df is not None:
            latest_date  = self.energy_df['timestamp'].max()
            today_data   = self.energy_df[
                self.energy_df['timestamp'].dt.date == latest_date.date()
            ]
            top_building = today_data.groupby('building')['energy_kwh'].sum().idxmax()
            metrics['top_consumer'] = top_building.replace('_', ' ')

        # ── Month forecast via Prophet ──────────────────────
        if (
            self.registry.is_ready()
            and self.pipeline is not None
            and self.registry._energy_name == 'energy_prophet'
        ):
            try:
                fc_df    = self.registry.energy_model.forecast_future(
                    self.pipeline.energy_daily, periods=30
                )
                projected = fc_df['yhat'].sum()
                # Baseline: last 30-day actual average × 30
                actual_30 = (
                    self.pipeline.energy_daily
                    .tail(30)['energy_kwh']
                    .sum()
                )
                overage_pct = ((projected - actual_30) / actual_30) * 100 if actual_30 > 0 else 0
                if overage_pct > 0:
                    metrics['month_forecast'] = f"+{overage_pct:.0f}% Overage"
                else:
                    metrics['month_forecast'] = f"{abs(overage_pct):.0f}% Under Budget"
            except Exception:
                pass   # keep the static fallback

        return metrics
    
    def get_energy_recommendations(self):
        """Generate AI-powered energy recommendations"""
        recommendations = [
            "Optimize HVAC schedule for Dorm 4",
            "Reduce lighting in Science Lab B after 8 PM",
            "Schedule equipment maintenance for Building A",
            "Consider solar panel installation on Gym Roof"
        ]
        
        # Could be enhanced with actual ML-based recommendations
        return recommendations
    
    def get_water_metrics(self):
        """Get water-specific KPI metrics"""
        stats = self.get_daily_stats()
        
        metrics = {
            "daily_consumption": "42k Gal",
            "active_leaks": "1",
            "recycled_water": "15k Gal"
        }
        
        if self.water_df is not None:
            latest_date = self.water_df['timestamp'].max()
            today_data = self.water_df[
                self.water_df['timestamp'].dt.date == latest_date.date()
            ]
            
            total_consumption = today_data['water_gallons'].sum()
            metrics["daily_consumption"] = f"{total_consumption / 1000:.0f}k Gal"
            
            # Mock recycled water calculation (30% assumption)
            recycled = total_consumption * 0.30
            metrics["recycled_water"] = f"{recycled / 1000:.0f}k Gal"
        
        return metrics
    
    def get_water_map_markers(self):
        """Generate markers for water flow map (same campus as dashboard)"""
        markers = [
            {"lat": 34.0522, "lng": -118.2437, "name": "Dorm 1", "status": "normal"},
            {"lat": 34.0525, "lng": -118.2440, "name": "Building A", "status": "normal"},
            {"lat": 34.0520, "lng": -118.2430, "name": "Library", "status": "normal"},
            {"lat": 34.0528, "lng": -118.2435, "name": "Athletic Center", "status": "normal"},
            {"lat": 34.0515, "lng": -118.2442, "name": "Dining Hall", "status": "normal"},
            {"lat": 34.0518, "lng": -118.2445, "name": "Dorm 3 Basement", "status": "critical", "leak": True}
        ]
        return markers
    
    def get_water_building_usage(self):
        """Get water usage by building for last 7 days"""
        # Mock data based on JSON spec
        return {
            "labels": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            "building_a": [2600, 1900, 1500, 2300, 2700, 3300, 2500],
            "building_b": [1400, 3200, 2300, 1600, 1200, 2900, 1200]
        }
    
    def get_waste_metrics(self):
        """Get waste-specific KPI metrics"""
        stats = self.get_daily_stats()
        
        metrics = {
            "total_waste": "120 Tons",
            "diversion_rate": "45%",
            "diversion_rate_value": 45,
            "next_pickup": "Tomorrow"
        }
        
        if self.waste_df is not None:
            # Calculate month-to-date total waste
            latest_date = self.waste_df['date'].max()
            mtd_waste = self.waste_df[
                self.waste_df['date'].dt.month == latest_date.month
            ]
            
            total_waste_lbs = mtd_waste['total_waste_lbs'].sum()
            total_waste_tons = int(total_waste_lbs / 2000)
            metrics["total_waste"] = f"{total_waste_tons} Tons"
            
            # Calculate diversion rate
            recycled_lbs = mtd_waste['recycled_lbs'].sum()
            composted_lbs = mtd_waste.get('composted_lbs', pd.Series([0])).sum()
            diverted = recycled_lbs + composted_lbs
            diversion_rate = int((diverted / total_waste_lbs) * 100) if total_waste_lbs > 0 else 0
            metrics["diversion_rate"] = f"{diversion_rate}%"
            metrics["diversion_rate_value"] = diversion_rate
        
        return metrics
    
    def get_waste_composition_data(self):
        """Get waste composition data for stacked bar chart"""
        # Mock data based on JSON spec
        return {
            "labels": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            "landfill": [15, 20, 20, 22, 28, 30, 42],
            "recycling": [10, 12, 15, 18, 20, 25, 30],
            "compost": [22, 23, 25, 30, 32, 30, 35]
        }
    
    def get_bin_fill_alerts(self):
        """Get bin fill level alerts"""
        # Mock data based on JSON spec
        alerts = [
            {"location": "Cafeteria Bin #2", "status": "95% Full", "severity": "high"},
            {"location": "Library Bin #1", "status": "92% Full", "severity": "high"},
            {"location": "Cafeteria Bin #4", "status": "95% Full", "severity": "high"}
        ]
        return alerts

    def get_ai_roadmap(self):
        """Get AI sustainability roadmap actions"""
        return [
            {
                "intervention": "Install Solar Array (Gym Roof)",
                "cost": "$120k",
                "annual_savings": "$25k",
                "co2e_reduction": "150 Tons",
                "roi": "4.8 Yrs",
                "action_label": "Approve",
                "action_style": "primary"
            },
            {
                "intervention": "Implement Greywater Recycling",
                "cost": "$40k",
                "annual_savings": "$8k",
                "co2e_reduction": "45 Tons",
                "roi": "5.0 Yrs",
                "action_label": "Simulate",
                "action_style": "secondary"
            },
            {
                "intervention": "Upgrade to LED Lighting (All Dorms)",
                "cost": "$15k",
                "annual_savings": "$6k",
                "co2e_reduction": "20 Tons",
                "roi": "2.5 Yrs",
                "action_label": "Approve",
                "action_style": "primary"
            }
        ]

    def get_ai_models_status(self):
        """Get AI model training status and accuracy.
        Returns live metrics from ModelRegistry when models are trained."""
        if self.registry.is_ready():
            return self.registry.get_model_status_cards()

        # Pre-training placeholder cards
        return [
            {"name": "energy_prophet", "status": "Not Trained", "metric": None,
             "trained": "--"},
            {"name": "water_lstm",     "status": "Pending",     "metric": None,
             "trained": "--"},
            {"name": "waste_rf",       "status": "Not Trained", "metric": None,
             "trained": "--"},
        ]
