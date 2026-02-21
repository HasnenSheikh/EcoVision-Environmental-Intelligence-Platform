"""
EcoVision - Data Ingestion & Engineering Pipeline
=================================================
Phase 2: Ingest → Clean → Feature Engineer → Resample

Handles:
  - CSV ingestion for all 4 data sources
  - Missing value imputation
  - Timestamp alignment & resampling (hourly / daily)
  - Lag features  (1h, 24h, 168h for hourly; 1d, 7d for daily)
  - Calendar features (hour, weekday, weekend, school session, month)
  - Rolling-average smoothing for noisy signals
  - CO₂e calculation using emission factors
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime


# ─────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────

# Academic semester months (roughly Sep–Dec and Jan–May)
SCHOOL_MONTHS = {1, 2, 3, 4, 5, 9, 10, 11, 12}

# Rolling window (hours for hourly data, days for daily data)
ROLLING_WINDOW_HOURLY = 3
ROLLING_WINDOW_DAILY  = 7


# ─────────────────────────────────────────────────────────────
#  1. Ingestion
# ─────────────────────────────────────────────────────────────

def load_energy(path: str) -> pd.DataFrame:
    """Load energy_consumption.csv → parse timestamps."""
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_water(path: str) -> pd.DataFrame:
    """Load water_consumption.csv → parse timestamps."""
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_waste(path: str) -> pd.DataFrame:
    """Load waste_generation.csv → parse dates."""
    df = pd.read_csv(path, parse_dates=['date'])
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_transport(path: str) -> pd.DataFrame:
    """Load transport_fuel.csv → parse dates."""
    df = pd.read_csv(path, parse_dates=['date'])
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_emission_factors(path: str) -> dict:
    """Load emission_factors.csv → return dict keyed by resource_type."""
    df = pd.read_csv(path)
    return dict(zip(df['resource_type'], df['co2e_kg_per_unit']))


# ─────────────────────────────────────────────────────────────
#  2. Cleaning
# ─────────────────────────────────────────────────────────────

def clean_hourly(df: pd.DataFrame, ts_col: str, value_cols: list) -> pd.DataFrame:
    """
    Clean hourly dataframe:
      - Drop duplicates on timestamp
      - Reindex to a full continuous hourly range
      - Interpolate missing values (linear) then backfill edges
    """
    df = df.drop_duplicates(subset=[ts_col])
    df = df.set_index(ts_col)

    full_range = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq='h'
    )
    df = df.reindex(full_range)

    # Interpolate numeric columns
    for col in value_cols:
        if col in df.columns:
            df[col] = df[col].interpolate(method='linear').bfill().ffill()

    df.index.name = ts_col
    df.reset_index(inplace=True)
    return df


def clean_daily(df: pd.DataFrame, date_col: str, value_cols: list) -> pd.DataFrame:
    """
    Clean daily dataframe:
      - Drop duplicates on date
      - Reindex to a full continuous daily range
      - Forward-fill then back-fill missing values
    """
    df = df.drop_duplicates(subset=[date_col])
    df = df.set_index(date_col)

    full_range = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq='D'
    )
    df = df.reindex(full_range)

    for col in value_cols:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()

    df.index.name = date_col
    df.reset_index(inplace=True)
    return df


# ─────────────────────────────────────────────────────────────
#  3. Resampling
# ─────────────────────────────────────────────────────────────

def resample_energy_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate all buildings to campus-level hourly totals."""
    hourly = (
        df.groupby(pd.Grouper(key='timestamp', freq='h'))
          .agg(
              energy_kwh=('energy_kwh', 'sum'),
              temperature_f=('temperature_f', 'mean'),
              occupancy=('occupancy', 'sum')
          )
          .reset_index()
    )
    return hourly


def resample_energy_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate all buildings to campus-level daily totals."""
    daily = (
        df.groupby(pd.Grouper(key='timestamp', freq='D'))
          .agg(
              energy_kwh=('energy_kwh', 'sum'),
              temperature_f=('temperature_f', 'mean'),
              occupancy=('occupancy', 'sum')
          )
          .reset_index()
          .rename(columns={'timestamp': 'date'})
    )
    return daily


def resample_water_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate all zones to campus-level hourly totals."""
    hourly = (
        df.groupby(pd.Grouper(key='timestamp', freq='h'))
          .agg(
              water_gallons=('water_gallons', 'sum'),
              pressure_psi=('pressure_psi', 'mean')
          )
          .reset_index()
    )
    return hourly


def resample_water_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate all zones to campus-level daily totals."""
    daily = (
        df.groupby(pd.Grouper(key='timestamp', freq='D'))
          .agg(
              water_gallons=('water_gallons', 'sum'),
              pressure_psi=('pressure_psi', 'mean')
          )
          .reset_index()
          .rename(columns={'timestamp': 'date'})
    )
    return daily


def resample_transport_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate all vehicle types to campus-level daily totals."""
    daily = (
        df.groupby(pd.Grouper(key='date', freq='D'))
          .agg(
              fuel_gallons=('fuel_gallons', 'sum'),
              trips_count=('trips_count', 'sum'),
              miles_driven=('miles_driven', 'sum')
          )
          .reset_index()
    )
    return daily


# ─────────────────────────────────────────────────────────────
#  4. Smoothing
# ─────────────────────────────────────────────────────────────

def smooth_rolling(series: pd.Series, window: int) -> pd.Series:
    """Apply centered rolling mean; fill edges with original values."""
    smoothed = series.rolling(window=window, center=True, min_periods=1).mean()
    return smoothed


# ─────────────────────────────────────────────────────────────
#  5. Calendar Feature Engineering
# ─────────────────────────────────────────────────────────────

def add_calendar_features_hourly(df: pd.DataFrame, ts_col: str = 'timestamp') -> pd.DataFrame:
    """Add time-based features to an hourly dataframe."""
    df = df.copy()
    dt = df[ts_col]

    df['hour']             = dt.dt.hour
    df['day_of_week']      = dt.dt.dayofweek          # 0=Mon … 6=Sun
    df['day_name']         = dt.dt.day_name()
    df['month']            = dt.dt.month
    df['week_of_year']     = dt.dt.isocalendar().week.astype(int)
    df['is_weekend']       = (df['day_of_week'] >= 5).astype(int)
    df['is_school_session']= df['month'].isin(SCHOOL_MONTHS).astype(int)

    # Business-hours flag (8 AM – 6 PM weekdays)
    df['is_business_hours']= (
        (df['hour'] >= 8) & (df['hour'] < 18) & (df['is_weekend'] == 0)
    ).astype(int)

    return df


def add_calendar_features_daily(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """Add time-based features to a daily dataframe."""
    df = df.copy()
    dt = df[date_col]

    df['day_of_week']       = dt.dt.dayofweek
    df['day_name']          = dt.dt.day_name()
    df['month']             = dt.dt.month
    df['week_of_year']      = dt.dt.isocalendar().week.astype(int)
    df['is_weekend']        = (df['day_of_week'] >= 5).astype(int)
    df['is_school_session'] = df['month'].isin(SCHOOL_MONTHS).astype(int)
    df['quarter']           = dt.dt.quarter

    return df


# ─────────────────────────────────────────────────────────────
#  6. Lag Feature Engineering
# ─────────────────────────────────────────────────────────────

def add_lag_features_hourly(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Add lag features for hourly data:
      lag_1h   : 1 hour ago
      lag_24h  : same hour yesterday
      lag_168h : same hour last week
      rolling_mean_3h  : 3-hour rolling average
      rolling_mean_24h : 24-hour rolling average
    """
    df = df.copy()
    df[f'{col}_lag_1h']          = df[col].shift(1)
    df[f'{col}_lag_24h']         = df[col].shift(24)
    df[f'{col}_lag_168h']        = df[col].shift(168)
    df[f'{col}_rolling_mean_3h'] = df[col].rolling(3,  min_periods=1).mean()
    df[f'{col}_rolling_mean_24h']= df[col].rolling(24, min_periods=1).mean()

    # Fill NaNs in lag columns with the rolling mean as a reasonable fallback
    for lag_col in [f'{col}_lag_1h', f'{col}_lag_24h', f'{col}_lag_168h']:
        df[lag_col] = df[lag_col].fillna(df[f'{col}_rolling_mean_24h'])

    return df


def add_lag_features_daily(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Add lag features for daily data:
      lag_1d  : yesterday
      lag_7d  : same day last week
      rolling_mean_7d : 7-day rolling average
    """
    df = df.copy()
    df[f'{col}_lag_1d']           = df[col].shift(1)
    df[f'{col}_lag_7d']           = df[col].shift(7)
    df[f'{col}_rolling_mean_7d']  = df[col].rolling(7, min_periods=1).mean()

    for lag_col in [f'{col}_lag_1d', f'{col}_lag_7d']:
        df[lag_col] = df[lag_col].fillna(df[f'{col}_rolling_mean_7d'])

    return df


# ─────────────────────────────────────────────────────────────
#  7. CO₂e Calculation
# ─────────────────────────────────────────────────────────────

def calculate_co2e(
    energy_kwh: float,
    water_gal: float,
    landfill_lbs: float,
    recycled_lbs: float,
    emission_factors: dict
) -> dict:
    """
    Calculate CO₂e (kg) for each resource category.
    Returns individual and total values.
    """
    ef = emission_factors

    co2_energy   = energy_kwh   * ef.get('Electricity',     0.43)
    co2_water    = water_gal    * ef.get('Water',            0.00095)
    co2_landfill = landfill_lbs * ef.get('Waste_Landfill',   0.544)
    co2_recycled = recycled_lbs * ef.get('Waste_Recycled',  -0.5)   # negative = avoided

    return {
        'co2e_energy_kg':   round(co2_energy,   2),
        'co2e_water_kg':    round(co2_water,     2),
        'co2e_landfill_kg': round(co2_landfill,  2),
        'co2e_recycled_kg': round(co2_recycled,  2),
        'co2e_total_kg':    round(co2_energy + co2_water + co2_landfill + co2_recycled, 2)
    }


# ─────────────────────────────────────────────────────────────
#  8. Master Pipeline
# ─────────────────────────────────────────────────────────────

class DataPipeline:
    """
    Orchestrates the full ingestion → clean → engineer cycle.
    Exposes ready-to-model dataframes as attributes.
    """

    def __init__(self, dataset_path: str = 'dataset'):
        self.dataset_path = dataset_path
        self.emission_factors: dict = {}

        # Raw
        self._raw_energy    : pd.DataFrame = None
        self._raw_water     : pd.DataFrame = None
        self._raw_waste     : pd.DataFrame = None
        self._raw_transport : pd.DataFrame = None

        # Processed (hourly)
        self.energy_hourly  : pd.DataFrame = None
        self.water_hourly   : pd.DataFrame = None

        # Processed (daily)
        self.energy_daily   : pd.DataFrame = None
        self.water_daily    : pd.DataFrame = None
        self.waste_daily    : pd.DataFrame = None
        self.transport_daily: pd.DataFrame = None

        # Pipeline health
        self.load_errors: list = []

    # ── 8a. Run full pipeline ─────────────────────────────────
    def run(self) -> bool:
        """Execute all pipeline stages. Returns True if successful."""
        self._ingest()
        self._clean()
        self._resample()
        self._feature_engineer()
        ok = len(self.load_errors) == 0
        if ok:
            print("[Pipeline] ✅ All stages completed successfully.")
        else:
            print(f"[Pipeline] ⚠️  Completed with errors: {self.load_errors}")
        return ok

    # ── 8b. Ingest ────────────────────────────────────────────
    def _ingest(self):
        dp = self.dataset_path
        try:
            self._raw_energy    = load_energy   (os.path.join(dp, 'energy_consumption.csv'))
            print(f"[Pipeline] Loaded energy    : {len(self._raw_energy):,} rows")
        except Exception as e:
            self.load_errors.append(f"energy: {e}")

        try:
            self._raw_water     = load_water    (os.path.join(dp, 'water_consumption.csv'))
            print(f"[Pipeline] Loaded water     : {len(self._raw_water):,} rows")
        except Exception as e:
            self.load_errors.append(f"water: {e}")

        try:
            self._raw_waste     = load_waste    (os.path.join(dp, 'waste_generation.csv'))
            print(f"[Pipeline] Loaded waste     : {len(self._raw_waste):,} rows")
        except Exception as e:
            self.load_errors.append(f"waste: {e}")

        try:
            self._raw_transport = load_transport(os.path.join(dp, 'transport_fuel.csv'))
            print(f"[Pipeline] Loaded transport : {len(self._raw_transport):,} rows")
        except Exception as e:
            self.load_errors.append(f"transport: {e}")

        try:
            self.emission_factors = load_emission_factors(
                os.path.join(dp, 'emission_factors.csv')
            )
            print(f"[Pipeline] Loaded emission factors: {self.emission_factors}")
        except Exception as e:
            self.load_errors.append(f"emission_factors: {e}")

    # ── 8c. Clean ─────────────────────────────────────────────
    def _clean(self):
        if self._raw_energy is not None:
            self._raw_energy = clean_hourly(
                self._raw_energy, 'timestamp',
                ['energy_kwh', 'temperature_f', 'occupancy']
            )

        if self._raw_water is not None:
            self._raw_water = clean_hourly(
                self._raw_water, 'timestamp',
                ['water_gallons', 'pressure_psi']
            )

        if self._raw_waste is not None:
            self._raw_waste = clean_daily(
                self._raw_waste, 'date',
                ['total_waste_lbs', 'recycled_lbs', 'landfill_lbs', 'population_count']
            )

        if self._raw_transport is not None:
            self._raw_transport = clean_daily(
                self._raw_transport, 'date',
                ['fuel_gallons', 'trips_count', 'miles_driven']
            )
        print("[Pipeline] Cleaning complete.")

    # ── 8d. Resample ──────────────────────────────────────────
    def _resample(self):
        if self._raw_energy is not None:
            self.energy_hourly = resample_energy_hourly(self._raw_energy)
            self.energy_daily  = resample_energy_daily (self._raw_energy)

        if self._raw_water is not None:
            self.water_hourly = resample_water_hourly(self._raw_water)
            self.water_daily  = resample_water_daily (self._raw_water)

        if self._raw_waste is not None:
            self.waste_daily = self._raw_waste.copy()   # already daily

        if self._raw_transport is not None:
            self.transport_daily = resample_transport_daily(self._raw_transport)

        print("[Pipeline] Resampling complete.")

    # ── 8e. Feature Engineering ───────────────────────────────
    def _feature_engineer(self):
        # Energy hourly
        if self.energy_hourly is not None:
            df = add_calendar_features_hourly(self.energy_hourly)
            df = add_lag_features_hourly(df, 'energy_kwh')
            df['energy_kwh_smoothed'] = smooth_rolling(df['energy_kwh'], ROLLING_WINDOW_HOURLY)
            self.energy_hourly = df

        # Energy daily
        if self.energy_daily is not None:
            df = add_calendar_features_daily(self.energy_daily, date_col='date')
            df = add_lag_features_daily(df, 'energy_kwh')
            df['energy_kwh_smoothed'] = smooth_rolling(df['energy_kwh'], ROLLING_WINDOW_DAILY)
            self.energy_daily = df

        # Water hourly
        if self.water_hourly is not None:
            df = add_calendar_features_hourly(self.water_hourly)
            df = add_lag_features_hourly(df, 'water_gallons')
            df['water_gallons_smoothed'] = smooth_rolling(df['water_gallons'], ROLLING_WINDOW_HOURLY)
            self.water_hourly = df

        # Water daily
        if self.water_daily is not None:
            df = add_calendar_features_daily(self.water_daily, date_col='date')
            df = add_lag_features_daily(df, 'water_gallons')
            df['water_gallons_smoothed'] = smooth_rolling(df['water_gallons'], ROLLING_WINDOW_DAILY)
            self.water_daily = df

        # Waste daily
        if self.waste_daily is not None:
            df = add_calendar_features_daily(self.waste_daily, date_col='date')
            df = add_lag_features_daily(df, 'total_waste_lbs')
            df = add_lag_features_daily(df, 'recycled_lbs')
            df['diversion_rate'] = (df['recycled_lbs'] / df['total_waste_lbs'].replace(0, np.nan)).fillna(0)
            df['waste_smoothed']  = smooth_rolling(df['total_waste_lbs'], ROLLING_WINDOW_DAILY)
            self.waste_daily = df

        # Transport daily
        if self.transport_daily is not None:
            df = add_calendar_features_daily(self.transport_daily, date_col='date')
            df = add_lag_features_daily(df, 'fuel_gallons')
            self.transport_daily = df

        print("[Pipeline] Feature engineering complete.")

    # ── 8f. Data Health Report ────────────────────────────────
    def health_report(self) -> dict:
        """Return a summary of dataset completeness and key stats."""
        report = {}

        def _stats(df, name, date_col, val_col):
            if df is None:
                return {'status': 'not_loaded'}
            nulls = df[val_col].isna().sum()
            return {
                'status':     'ok' if nulls == 0 else 'has_nulls',
                'rows':       len(df),
                'nulls':      int(nulls),
                'date_range': f"{df[date_col].min().date()} → {df[date_col].max().date()}",
                'mean':       round(df[val_col].mean(), 2),
                'std':        round(df[val_col].std(),  2)
            }

        report['energy_hourly']   = _stats(self.energy_hourly,   'energy_hourly',
                                           'timestamp', 'energy_kwh')
        report['energy_daily']    = _stats(self.energy_daily,    'energy_daily',
                                           'date', 'energy_kwh')
        report['water_hourly']    = _stats(self.water_hourly,    'water_hourly',
                                           'timestamp', 'water_gallons')
        report['water_daily']     = _stats(self.water_daily,     'water_daily',
                                           'date', 'water_gallons')
        report['waste_daily']     = _stats(self.waste_daily,     'waste_daily',
                                           'date', 'total_waste_lbs')
        report['transport_daily'] = _stats(self.transport_daily, 'transport_daily',
                                           'date', 'fuel_gallons')

        return report
