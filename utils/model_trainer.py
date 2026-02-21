"""
EcoVision - Baseline Modeling
==============================
Phase 3: Train/Test Split  →  Prophet (Energy)  →  Random Forest (Waste)
         →  MAPE Evaluation  →  LSTM fallback if Prophet underperforms

Train set : Jan – Oct 2025  (TRAIN_CUTOFF = '2025-10-31')
Test  set : Nov – Dec 2025  (~61 days)

Outputs (saved to  models/ ):
    energy_prophet.pkl  or  energy_lstm.pkl
    waste_rf.pkl
    training_metrics.json
"""

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────

MODELS_DIR             = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
TRAIN_CUTOFF           = '2025-10-31'   # Train ≤ this date, Test > this date
LSTM_FALLBACK_THRESH   = 10.0           # Switch to LSTM if Prophet MAPE > 10 %
SCHOOL_MONTHS          = {1, 2, 3, 4, 5, 9, 10, 11, 12}


# ─────────────────────────────────────────────────────────────
#  1.  Shared Utilities
# ─────────────────────────────────────────────────────────────

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (%)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask   = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def split_train_test(df: pd.DataFrame, date_col: str, cutoff: str = TRAIN_CUTOFF):
    """Return (train, test) split on cutoff date."""
    cutoff_dt = pd.Timestamp(cutoff)
    return (
        df[df[date_col] <= cutoff_dt].copy(),
        df[df[date_col] >  cutoff_dt].copy(),
    )


def ensure_models_dir():
    os.makedirs(MODELS_DIR, exist_ok=True)


def save_model(obj, name: str) -> str:
    ensure_models_dir()
    path = os.path.join(MODELS_DIR, f'{name}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f"    ✅ Saved  →  {os.path.basename(path)}")
    return path


def load_model(name: str):
    path = os.path.join(MODELS_DIR, f'{name}.pkl')
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_metrics(metrics: dict):
    ensure_models_dir()
    path = os.path.join(MODELS_DIR, 'training_metrics.json')
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"    ✅ Metrics →  {os.path.basename(path)}")


def load_metrics() -> dict:
    path = os.path.join(MODELS_DIR, 'training_metrics.json')
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────
#  2.  Model A – Prophet Energy Forecaster
# ─────────────────────────────────────────────────────────────

class ProphetEnergyForecaster:
    """
    Facebook Prophet model for daily campus energy (kWh) forecasting.

    Extra regressors layered on top of built-in weekly / yearly seasonality:
        temperature_f     – HVAC demand correlates strongly with temperature
        is_weekend        – campus consumption drops Sat / Sun
        is_school_session – semester vs summer significantly shifts baseline
    """

    REGRESSORS = ['temperature_f', 'is_weekend', 'is_school_session']

    def __init__(
        self,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
    ):
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.model   = None
        self.metrics : dict = {}

    # ── helpers ─────────────────────────────────────────────

    def _to_prophet(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = ['date', 'energy_kwh'] + self.REGRESSORS
        return df[cols].rename(columns={'date': 'ds', 'energy_kwh': 'y'})

    def _fill_future_regressors(
        self, future: pd.DataFrame, ref_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Populate regressors for future dates using historical monthly averages."""
        future = future.copy()
        monthly_temp = ref_df.groupby('month')['temperature_f'].mean()
        future['month'] = future['ds'].dt.month
        future['temperature_f'] = (
            future['month'].map(monthly_temp).fillna(ref_df['temperature_f'].mean())
        )
        future['is_weekend']        = (future['ds'].dt.dayofweek >= 5).astype(int)
        future['is_school_session'] = future['ds'].dt.month.isin(SCHOOL_MONTHS).astype(int)
        return future

    # ── public API ──────────────────────────────────────────

    def fit(self, train_df: pd.DataFrame) -> 'ProphetEnergyForecaster':
        from prophet import Prophet

        self.model = Prophet(
            changepoint_prior_scale = self.changepoint_prior_scale,
            seasonality_prior_scale = self.seasonality_prior_scale,
            daily_seasonality  = False,
            weekly_seasonality = True,
            yearly_seasonality = True,
        )
        for reg in self.REGRESSORS:
            self.model.add_regressor(reg)

        self.model.fit(self._to_prophet(train_df))
        print(f"    [Prophet] Fitted on {len(train_df):,} training days.")
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict for rows that already contain regressor columns."""
        future = self._to_prophet(df).drop(columns=['y'])
        fc = self.model.predict(future)
        fc['yhat'] = np.clip(fc['yhat'], 0, None)
        return fc[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    def evaluate(self, test_df: pd.DataFrame) -> dict:
        fc     = self.predict(test_df)
        y_true = test_df['energy_kwh'].values
        y_pred = fc['yhat'].values

        self.metrics = {
            'model':     'Prophet',
            'mape':      round(mape(y_true, y_pred), 2),
            'mae':       round(mae(y_true,  y_pred), 2),
            'rmse':      round(rmse(y_true, y_pred), 2),
            'test_days': int(len(test_df)),
            'trained_at': datetime.now().isoformat(timespec='seconds'),
        }
        return self.metrics

    def forecast_future(self, full_df: pd.DataFrame, periods: int = 30) -> pd.DataFrame:
        """n-day ahead forecast beyond the last date in full_df."""
        future = self.model.make_future_dataframe(periods=periods, freq='D')
        future = self._fill_future_regressors(future, full_df)
        fc = self.model.predict(future)
        fc['yhat'] = np.clip(fc['yhat'], 0, None)
        last_date = full_df['date'].max()
        return (
            fc[fc['ds'] > last_date][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            .reset_index(drop=True)
        )

    def hourly_profile(self, energy_hourly_df: pd.DataFrame) -> np.ndarray:
        """
        Compute the average fraction of daily kWh that falls in each hour (0-23).
        Used to distribute a daily forecast into 24 hourly estimates.
        """
        df = energy_hourly_df.copy()
        df['hour'] = df['timestamp'].dt.hour
        df['date'] = df['timestamp'].dt.date

        daily_total   = df.groupby('date')['energy_kwh'].transform('sum')
        df['fraction'] = df['energy_kwh'] / daily_total.replace(0, np.nan)

        profile = df.groupby('hour')['fraction'].mean().values   # shape (24,)
        # Normalise so fractions sum to 1
        total = profile.sum()
        return profile / total if total > 0 else np.ones(24) / 24


# ─────────────────────────────────────────────────────────────
#  3.  Model B – Random Forest Waste Forecaster
# ─────────────────────────────────────────────────────────────

class RandomForestWasteForecaster:
    """
    Random Forest regressor for daily waste generation prediction.

    Input features (all available after DataPipeline.run()):
        Calendar    – month, day_of_week, is_weekend, is_school_session, quarter
        Population  – population_count  (key external driver of waste volume)
        Lag / roll  – lag_1d, lag_7d, rolling_mean_7d  (auto-regressive signal)
    """

    TARGET   = 'total_waste_lbs'
    FEATURES = [
        'month', 'day_of_week', 'is_weekend', 'is_school_session', 'quarter',
        'population_count',
        'total_waste_lbs_lag_1d',
        'total_waste_lbs_lag_7d',
        'total_waste_lbs_rolling_mean_7d',
    ]

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth:    int = 10,
        random_state: int = 42,
    ):
        from sklearn.ensemble import RandomForestRegressor
        self.rf = RandomForestRegressor(
            n_estimators = n_estimators,
            max_depth    = max_depth,
            random_state = random_state,
            n_jobs       = -1,
        )
        self.metrics:           dict = {}
        self.feature_importance: dict = {}

    # ── helpers ─────────────────────────────────────────────

    def _get_features(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.FEATURES].copy()
        return X.fillna(X.median())

    # ── public API ──────────────────────────────────────────

    def fit(self, train_df: pd.DataFrame) -> 'RandomForestWasteForecaster':
        X_train = self._get_features(train_df)
        y_train = train_df[self.TARGET].values

        self.rf.fit(X_train, y_train)
        self.feature_importance = dict(
            sorted(
                zip(self.FEATURES, self.rf.feature_importances_),
                key=lambda kv: -kv[1],
            )
        )
        top3 = list(self.feature_importance.keys())[:3]
        print(f"    [RF-Waste] Fitted on {len(train_df):,} training days.  Top-3 features: {top3}")
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self._get_features(df)
        return np.clip(self.rf.predict(X), 0, None)

    def evaluate(self, test_df: pd.DataFrame) -> dict:
        y_true = test_df[self.TARGET].values
        y_pred = self.predict(test_df)

        self.metrics = {
            'model':       'RandomForest',
            'mape':        round(mape(y_true, y_pred), 2),
            'mae':         round(mae(y_true,  y_pred), 2),
            'rmse':        round(rmse(y_true, y_pred), 2),
            'test_days':   int(len(test_df)),
            'top_feature': list(self.feature_importance.keys())[0]
                           if self.feature_importance else 'N/A',
            'trained_at':  datetime.now().isoformat(timespec='seconds'),
        }
        return self.metrics


# ─────────────────────────────────────────────────────────────
#  4.  Model A-Fallback – LSTM Energy Forecaster
# ─────────────────────────────────────────────────────────────

class LSTMEnergyForecaster:
    """
    Sliding-window LSTM for daily energy forecasting.
    Auto-selected when Prophet MAPE > LSTM_FALLBACK_THRESH.

    Requires : tensorflow >= 2.x   (pip install tensorflow)
    Fallback  : sklearn MLPRegressor if TensorFlow is not installed.
    """

    def __init__(
        self,
        window_size: int = 14,   # lookback in days
        units:       int = 64,
        epochs:      int = 80,
        batch_size:  int = 32,
        patience:    int = 10,
    ):
        self.window_size = window_size
        self.units       = units
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.patience    = patience

        self.model   = None
        self.scaler  = None
        self.backend = None     # 'keras' or 'mlp'
        self.metrics : dict = {}

    # ── data prep ───────────────────────────────────────────

    def _scale(self, series: np.ndarray) -> np.ndarray:
        from sklearn.preprocessing import MinMaxScaler
        self.scaler = MinMaxScaler()
        return self.scaler.fit_transform(series.reshape(-1, 1)).flatten()

    def _make_windows(self, scaled: np.ndarray):
        X, y = [], []
        for i in range(self.window_size, len(scaled)):
            X.append(scaled[i - self.window_size: i])
            y.append(scaled[i])
        return np.array(X), np.array(y)

    # ── keras LSTM ──────────────────────────────────────────

    def _build_keras(self, series: np.ndarray):
        # TF 2.16+ ships Keras as a standalone package
        from keras.models    import Sequential
        from keras.layers    import LSTM, Dense, Dropout
        from keras.callbacks import EarlyStopping

        scaled  = self._scale(series)
        X, y    = self._make_windows(scaled)
        X       = X.reshape(X.shape[0], X.shape[1], 1)

        model = Sequential([
            LSTM(self.units, return_sequences=True, input_shape=(self.window_size, 1)),
            Dropout(0.2),
            LSTM(self.units // 2),
            Dropout(0.2),
            Dense(1),
        ])
        model.compile(optimizer='adam', loss='mse')
        cb = EarlyStopping(monitor='loss', patience=self.patience, restore_best_weights=True)
        model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size,
                  callbacks=[cb], verbose=0)

        self.model   = model
        self.backend = 'keras'
        print(f"    [LSTM-Keras] Trained  window={self.window_size}d  units={self.units}")

    # ── MLP fallback ────────────────────────────────────────

    def _build_mlp(self, series: np.ndarray):
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import MinMaxScaler

        self.scaler = MinMaxScaler()
        scaled = self.scaler.fit_transform(series.reshape(-1, 1)).flatten()
        X, y   = self._make_windows(scaled)

        mlp = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            max_iter=800,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
        )
        mlp.fit(X, y)
        self.model   = mlp
        self.backend = 'mlp'
        print("    [LSTM-MLP] TensorFlow not available → MLPRegressor fallback trained.")

    # ── public API ──────────────────────────────────────────

    def fit(self, train_df: pd.DataFrame) -> 'LSTMEnergyForecaster':
        series = train_df['energy_kwh'].values.astype(float)
        try:
            import keras  # noqa: F401  (standalone, TF >= 2.16)
            self._build_keras(series)
        except ImportError:
            self._build_mlp(series)
        return self

    def _predict_steps(self, seed_series: np.ndarray, n_steps: int) -> np.ndarray:
        """Auto-regressive multi-step prediction from a seed window."""
        scaled = self.scaler.transform(seed_series.reshape(-1, 1)).flatten()
        window = list(scaled[-self.window_size:])
        preds  = []

        for _ in range(n_steps):
            x = np.array(window[-self.window_size:])
            if self.backend == 'keras':
                val = float(self.model.predict(x.reshape(1, self.window_size, 1), verbose=0)[0, 0])
            else:
                val = float(self.model.predict(x.reshape(1, -1))[0])
            preds.append(val)
            window.append(val)

        return self.scaler.inverse_transform(
            np.array(preds).reshape(-1, 1)
        ).flatten()

    def evaluate(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
        seed   = train_df['energy_kwh'].values.astype(float)
        y_true = test_df['energy_kwh'].values.astype(float)
        y_pred = np.clip(self._predict_steps(seed, len(test_df)), 0, None)

        self.metrics = {
            'model':     f'LSTM ({self.backend})',
            'mape':      round(mape(y_true, y_pred), 2),
            'mae':       round(mae(y_true,  y_pred), 2),
            'rmse':      round(rmse(y_true, y_pred), 2),
            'test_days': int(len(test_df)),
            'backend':   self.backend,
            'trained_at': datetime.now().isoformat(timespec='seconds'),
        }
        return self.metrics

    def forecast_future(self, full_df: pd.DataFrame, periods: int = 30) -> pd.DataFrame:
        seed  = full_df['energy_kwh'].values.astype(float)
        preds = np.clip(self._predict_steps(seed, periods), 0, None)
        last  = full_df['date'].max()
        dates = pd.date_range(start=last + pd.Timedelta(days=1), periods=periods, freq='D')
        return pd.DataFrame({'ds': dates, 'yhat': preds})


# ─────────────────────────────────────────────────────────────
#  4b. Model D – LSTM Water Forecaster
# ─────────────────────────────────────────────────────────────

class LSTMWaterForecaster:
    """
    Sliding-window LSTM for daily water consumption forecasting.
    Mirrors LSTMEnergyForecaster architecture, targeting water_gallons.
    Fallback: sklearn MLPRegressor when TensorFlow is unavailable.
    """

    def __init__(
        self,
        window_size: int = 14,
        units:       int = 64,
        epochs:      int = 80,
        batch_size:  int = 32,
        patience:    int = 10,
    ):
        self.window_size = window_size
        self.units       = units
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.patience    = patience

        self.model   = None
        self.scaler  = None
        self.backend = None
        self.metrics : dict = {}

    # ── data prep ───────────────────────────────────────────

    def _scale(self, series: np.ndarray) -> np.ndarray:
        from sklearn.preprocessing import MinMaxScaler
        self.scaler = MinMaxScaler()
        return self.scaler.fit_transform(series.reshape(-1, 1)).flatten()

    def _make_windows(self, scaled: np.ndarray):
        X, y = [], []
        for i in range(self.window_size, len(scaled)):
            X.append(scaled[i - self.window_size: i])
            y.append(scaled[i])
        return np.array(X), np.array(y)

    # ── keras LSTM ──────────────────────────────────────────

    def _build_keras(self, series: np.ndarray):
        from keras.models    import Sequential
        from keras.layers    import LSTM, Dense, Dropout
        from keras.callbacks import EarlyStopping

        scaled  = self._scale(series)
        X, y    = self._make_windows(scaled)
        X       = X.reshape(X.shape[0], X.shape[1], 1)

        model = Sequential([
            LSTM(self.units, return_sequences=True, input_shape=(self.window_size, 1)),
            Dropout(0.2),
            LSTM(self.units // 2),
            Dropout(0.2),
            Dense(1),
        ])
        model.compile(optimizer='adam', loss='mse')
        cb = EarlyStopping(monitor='loss', patience=self.patience, restore_best_weights=True)
        model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size,
                  callbacks=[cb], verbose=0)

        self.model   = model
        self.backend = 'keras'
        print(f"    [LSTM-Water-Keras] Trained  window={self.window_size}d  units={self.units}")

    # ── MLP fallback ────────────────────────────────────────

    def _build_mlp(self, series: np.ndarray):
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing  import MinMaxScaler

        self.scaler = MinMaxScaler()
        scaled = self.scaler.fit_transform(series.reshape(-1, 1)).flatten()
        X, y   = self._make_windows(scaled)

        mlp = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            max_iter=800,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
        )
        mlp.fit(X, y)
        self.model   = mlp
        self.backend = 'mlp'
        print("    [LSTM-Water-MLP] TensorFlow not available → MLPRegressor fallback.")

    # ── public API ──────────────────────────────────────────

    def fit(self, train_df: pd.DataFrame) -> 'LSTMWaterForecaster':
        series = train_df['water_gallons'].values.astype(float)
        try:
            import keras  # noqa: F401
            self._build_keras(series)
        except ImportError:
            self._build_mlp(series)
        return self

    def _predict_steps(self, seed_series: np.ndarray, n_steps: int) -> np.ndarray:
        """Auto-regressive multi-step prediction from a seed window."""
        scaled = self.scaler.transform(seed_series.reshape(-1, 1)).flatten()
        window = list(scaled[-self.window_size:])
        preds  = []

        for _ in range(n_steps):
            x = np.array(window[-self.window_size:])
            if self.backend == 'keras':
                val = float(self.model.predict(x.reshape(1, self.window_size, 1), verbose=0)[0, 0])
            else:
                val = float(self.model.predict(x.reshape(1, -1))[0])
            preds.append(val)
            window.append(val)

        return self.scaler.inverse_transform(
            np.array(preds).reshape(-1, 1)
        ).flatten()

    def evaluate(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
        seed   = train_df['water_gallons'].values.astype(float)
        y_true = test_df['water_gallons'].values.astype(float)
        y_pred = np.clip(self._predict_steps(seed, len(test_df)), 0, None)

        self.metrics = {
            'model':      f'LSTM-Water ({self.backend})',
            'mape':       round(mape(y_true, y_pred), 2),
            'mae':        round(mae(y_true,  y_pred), 2),
            'rmse':       round(rmse(y_true, y_pred), 2),
            'test_days':  int(len(test_df)),
            'backend':    self.backend,
            'trained_at': datetime.now().isoformat(timespec='seconds'),
        }
        return self.metrics

    def predict_historical(self, full_df: pd.DataFrame) -> np.ndarray:
        """
        Aligned in-sample predictions for every row in full_df.
        The first `window_size` rows echo the actual values as warm-up.
        """
        series = full_df['water_gallons'].values.astype(float)
        ws = self.window_size
        if len(series) <= ws:
            return series.copy()
        preds = np.clip(self._predict_steps(series[:ws], len(series) - ws), 0, None)
        return np.concatenate([series[:ws], preds])

    def forecast_future(self, full_df: pd.DataFrame, periods: int = 30) -> pd.DataFrame:
        seed  = full_df['water_gallons'].values.astype(float)
        preds = np.clip(self._predict_steps(seed, periods), 0, None)
        last  = full_df['date'].max()
        dates = pd.date_range(start=last + pd.Timedelta(days=1), periods=periods, freq='D')
        return pd.DataFrame({'ds': dates, 'yhat': preds})


# ─────────────────────────────────────────────────────────────
#  5.  Orchestrator – train_all_models()
# ─────────────────────────────────────────────────────────────

def train_all_models(pipeline) -> dict:
    """
    Train Prophet + Random Forest from a live DataPipeline instance.
    Automatically evaluates LSTM fallback if Prophet MAPE > threshold.
    Saves all models and a metrics JSON to  models/ .

    Returns:
        dict with keys 'energy' and 'waste', each containing
        'model_name', 'model_obj', and 'metrics'.
    """
    results     = {}
    all_metrics = {}

    # ── A. Energy – Prophet ──────────────────────────────────
    print("\n━━━  Model A  │  Prophet  │  Energy (Daily kWh)  ━━━")
    energy_df          = pipeline.energy_daily.dropna(subset=['energy_kwh'])
    train_e, test_e    = split_train_test(energy_df, 'date')
    print(f"    Train: {len(train_e):,} days  │  Test: {len(test_e):,} days")

    prophet       = ProphetEnergyForecaster()
    prophet.fit(train_e)
    prophet_metrics = prophet.evaluate(test_e)
    print(
        f"    MAPE = {prophet_metrics['mape']} %   "
        f"MAE = {prophet_metrics['mae']:,.0f} kWh   "
        f"RMSE = {prophet_metrics['rmse']:,.0f}"
    )

    energy_model      = prophet
    energy_model_name = 'energy_prophet'

    # ── Fallback: LSTM if Prophet underperforms ───────────────
    if prophet_metrics['mape'] > LSTM_FALLBACK_THRESH:
        print(f"\n    ⚠  Prophet MAPE {prophet_metrics['mape']} % > threshold {LSTM_FALLBACK_THRESH} %")
        print("       → Training LSTM fallback …\n")
        lstm = LSTMEnergyForecaster()
        lstm.fit(train_e)
        lstm_metrics = lstm.evaluate(train_e, test_e)
        print(
            f"    LSTM  MAPE = {lstm_metrics['mape']} %   "
            f"MAE = {lstm_metrics['mae']:,.0f} kWh"
        )
        if lstm_metrics['mape'] < prophet_metrics['mape']:
            energy_model      = lstm
            energy_model_name = 'energy_lstm'
            prophet_metrics   = lstm_metrics
            print("    ✅ LSTM wins – using LSTM as production energy model.")
        else:
            print("    Prophet still wins – keeping Prophet as production model.")

    save_model(energy_model, energy_model_name)
    results['energy']      = {'model_name': energy_model_name,
                               'model_obj':  energy_model,
                               'metrics':    prophet_metrics}
    all_metrics['energy']  = prophet_metrics

    # ── B. Waste – Random Forest ─────────────────────────────
    print("\n━━━  Model B  │  Random Forest  │  Waste (Daily lbs)  ━━━")
    waste_df           = pipeline.waste_daily.dropna(subset=['total_waste_lbs'])
    train_w, test_w    = split_train_test(waste_df, 'date')
    print(f"    Train: {len(train_w):,} days  │  Test: {len(test_w):,} days")

    rf          = RandomForestWasteForecaster()
    rf.fit(train_w)
    rf_metrics  = rf.evaluate(test_w)
    print(
        f"    MAPE = {rf_metrics['mape']} %   "
        f"MAE = {rf_metrics['mae']:,.0f} lbs   "
        f"RMSE = {rf_metrics['rmse']:,.0f}   "
        f"Top feature: {rf_metrics['top_feature']}"
    )

    save_model(rf, 'waste_rf')
    results['waste']     = {'model_name': 'waste_rf',
                             'model_obj':  rf,
                             'metrics':    rf_metrics}
    all_metrics['waste'] = rf_metrics

    # ── C. Water – LSTM ──────────────────────────────────────
    print("\n━━━  Model C  │  LSTM  │  Water (Daily Gallons)  ━━━")
    water_df           = pipeline.water_daily.dropna(subset=['water_gallons'])
    train_wt, test_wt  = split_train_test(water_df, 'date')
    print(f"    Train: {len(train_wt):,} days  │  Test: {len(test_wt):,} days")

    water_lstm  = LSTMWaterForecaster()
    water_lstm.fit(train_wt)
    wl_metrics  = water_lstm.evaluate(train_wt, test_wt)
    print(
        f"    MAPE = {wl_metrics['mape']} %   "
        f"MAE = {wl_metrics['mae']:,.0f} gal   "
        f"RMSE = {wl_metrics['rmse']:,.0f}"
    )

    save_model(water_lstm, 'water_lstm')
    results['water']     = {'model_name': 'water_lstm',
                             'model_obj':  water_lstm,
                             'metrics':    wl_metrics}
    all_metrics['water'] = wl_metrics

    # ── Save combined metrics JSON ───────────────────────────
    all_metrics['trained_at'] = datetime.now().isoformat(timespec='seconds')
    save_metrics(all_metrics)

    # ── Summary table ────────────────────────────────────────
    _units = {'energy': 'kWh', 'waste': 'lbs', 'water': 'gal'}
    print("\n" + "━" * 55)
    print("  TRAINING SUMMARY")
    print("━" * 55)
    print(f"  {'Target':10s}  {'Model':22s}  {'MAPE':>8s}  {'MAE':>10s}")
    print("  " + "-" * 51)
    for key, val in results.items():
        m    = val['metrics']
        unit = _units.get(key, '')
        print(
            f"  {key:10s}  {val['model_name']:22s}  "
            f"{m['mape']:>7.2f}%  {m['mae']:>8,.0f} {unit}"
        )
    print("━" * 55)

    return results


# ─────────────────────────────────────────────────────────────
#  6.  ModelRegistry  –  serve-time inference helper
# ─────────────────────────────────────────────────────────────

class ModelRegistry:
    """
    Loaded once at Flask startup by DataLoader.
    Provides prediction helpers consumed by the API layer.
    """

    def __init__(self):
        self.energy_model      = None
        self.waste_model       = None
        self.water_lstm_model  = None
        self._energy_name      = None
        self._waste_name       = None
        self._water_lstm_name  = None
        self.metrics           = {}

    # ── loading ─────────────────────────────────────────────

    def load(self) -> 'ModelRegistry':
        """Try to load the best saved model for each target."""
        for name in ('energy_prophet', 'energy_lstm'):
            m = load_model(name)
            if m is not None:
                self.energy_model = m
                self._energy_name = name
                break

        self.waste_model  = load_model('waste_rf')
        self._waste_name  = 'waste_rf' if self.waste_model else None

        self.water_lstm_model = load_model('water_lstm')
        self._water_lstm_name = 'water_lstm' if self.water_lstm_model else None

        self.metrics = load_metrics()

        print(
            f"[ModelRegistry] energy={self._energy_name}  "
            f"waste={self._waste_name}  water={self._water_lstm_name}"
        )
        return self

    def is_ready(self) -> bool:
        return self.energy_model is not None and self.waste_model is not None

    # ── energy predictions ───────────────────────────────────

    def predict_energy_daily(self, energy_daily_df: pd.DataFrame) -> np.ndarray:
        """
        Return predicted daily kWh aligned row-by-row with energy_daily_df.
        Works for both Prophet and LSTM backends.
        """
        if self.energy_model is None:
            return np.full(len(energy_daily_df), np.nan)

        if self._energy_name == 'energy_prophet':
            fc = self.energy_model.predict(energy_daily_df)
            return fc['yhat'].values

        # LSTM: seed from historical + predict forward
        n     = len(energy_daily_df)
        ws    = self.energy_model.window_size
        seed  = energy_daily_df['energy_kwh'].values.astype(float)
        preds = self.energy_model._predict_steps(seed[:ws], max(n - ws, 1))
        # Pad beginning with actuals
        return np.concatenate([seed[:ws], preds])[:n]

    def get_hourly_predictions(
        self,
        energy_hourly_df: pd.DataFrame,
        energy_daily_df:  pd.DataFrame,
        date_str:         str = None,
    ) -> pd.DataFrame:
        """
        Distribute daily Prophet forecast into 24 hourly estimates using
        the historical hourly fraction profile.

        Returns DataFrame with columns: timestamp, actual, predicted
        """
        if self.energy_model is None or not hasattr(self.energy_model, 'hourly_profile'):
            # Fallback: rolling mean
            df = energy_hourly_df.copy()
            df['predicted'] = df['energy_kwh'].rolling(3, min_periods=1).mean()
            return df[['timestamp', 'energy_kwh', 'predicted']].rename(
                columns={'energy_kwh': 'actual'}
            )

        profile = self.energy_model.hourly_profile(energy_hourly_df)   # (24,)

        # Get the daily forecast value for each calendar day
        fc_df = energy_daily_df.copy()
        fc_df['yhat'] = self.predict_energy_daily(fc_df)
        fc_map = dict(zip(fc_df['date'].dt.date, fc_df['yhat']))

        result = energy_hourly_df.copy()
        result = result.rename(columns={'energy_kwh': 'actual'})
        result['date_key'] = result['timestamp'].dt.date
        result['hour']     = result['timestamp'].dt.hour
        result['predicted'] = result.apply(
            lambda row: fc_map.get(row['date_key'], row['actual'])
                        * profile[row['hour']],
            axis=1,
        )
        return result[['timestamp', 'actual', 'predicted']]

    # ── waste predictions ────────────────────────────────────

    def predict_waste_daily(self, waste_daily_df: pd.DataFrame) -> np.ndarray:
        if self.waste_model is None:
            return np.full(len(waste_daily_df), np.nan)
        return self.waste_model.predict(waste_daily_df)

    # ── water timeseries (actual vs. predicted + anomalies) ──────

    def _rolling_water_baseline(self, actual: np.ndarray, window: int = 7) -> np.ndarray:
        """Simple rolling-mean baseline when no model is available."""
        result = np.empty_like(actual)
        for i in range(len(actual)):
            start    = max(0, i - window)
            result[i] = float(np.mean(actual[start:i+1]))
        return result

    def predict_water_timeseries(self, pipeline, n_days: int = 60) -> dict:
        """
        Return last `n_days` of actual vs. LSTM-predicted daily water
        consumption as a JSON-serialisable dict for the chart.

        Returns:
            {
                timestamps : [str, ...],
                actual     : [float, ...],
                predicted  : [float, ...],
                anomalies  : [{index, type, severity}, ...]
            }
        """
        water_daily = pipeline.water_daily.copy()
        df          = water_daily.tail(n_days).reset_index(drop=True)
        actual      = df['water_gallons'].values.astype(float)
        dates       = df['date'].dt.strftime('%b %d').tolist()

        if self.water_lstm_model is not None:
            try:
                ws   = self.water_lstm_model.window_size
                tail = water_daily.tail(n_days + ws).reset_index(drop=True)
                all_preds = self.water_lstm_model.predict_historical(tail)
                predicted = np.clip(all_preds[-n_days:], 0, None)
            except Exception as e:
                print(f"[predict_water_timeseries] LSTM error: {e}")
                predicted = self._rolling_water_baseline(actual)
        else:
            predicted = self._rolling_water_baseline(actual)

        # Anomaly detection: 2-sigma on absolute deviations (same method as energy)
        deviations = np.abs(actual - predicted)
        threshold  = deviations.std() * 2.0
        residuals  = actual - predicted
        anomalies  = []
        for i, (dev, res) in enumerate(zip(deviations, residuals)):
            if dev > threshold:
                sigma = float(dev / (deviations.std() + 1e-9))
                anomalies.append({
                    'index':    int(i),
                    'type':     'High Spike'   if res > 0 else 'Unusual Drop',
                    'severity': 'critical'     if sigma >= 3 else ('high' if sigma >= 2 else 'medium'),
                })

        return {
            'timestamps': dates,
            'actual':     [round(float(v), 1) for v in actual],
            'predicted':  [round(float(v), 1) for v in predicted],
            'anomalies':  anomalies,
            'dates_iso':  df['date'].dt.strftime('%Y-%m-%d').tolist(),
        }

    # ── Forward forecasts (consumed by /predict/* endpoints) ─

    def forecast_energy(self, pipeline, periods: int = 7) -> list:
        """
        Return `periods`-day ahead energy forecast as a list of dicts:
            date, kwh_forecast, kwh_lower, kwh_upper, source

        Works for both LSTM and Prophet backends.
        """
        energy_daily = pipeline.energy_daily
        last_date    = energy_daily['date'].max()

        if self.energy_model is None:
            # Fallback: repeat the last 7-day rolling average
            avg = float(energy_daily['energy_kwh'].tail(7).mean())
            return [
                {
                    'date':         (last_date + pd.Timedelta(days=i+1)).strftime('%Y-%m-%d'),
                    'kwh_forecast': round(avg, 1),
                    'kwh_lower':    round(avg * 0.90, 1),
                    'kwh_upper':    round(avg * 1.10, 1),
                    'source':       'rolling_mean',
                }
                for i in range(periods)
            ]

        try:
            if self._energy_name == 'energy_prophet':
                fc = self.energy_model.forecast_future(energy_daily, periods=periods)
                result = []
                for _, row in fc.iterrows():
                    result.append({
                        'date':         pd.Timestamp(row['ds']).strftime('%Y-%m-%d'),
                        'kwh_forecast': round(float(row['yhat']),       1),
                        'kwh_lower':    round(float(row['yhat_lower']), 1),
                        'kwh_upper':    round(float(row['yhat_upper']), 1),
                        'source':       'prophet',
                    })
                return result

            else:  # LSTM / MLP
                fc = self.energy_model.forecast_future(energy_daily, periods=periods)
                avg_std = float(energy_daily['energy_kwh'].std())
                result  = []
                for _, row in fc.iterrows():
                    yhat = float(row['yhat'])
                    result.append({
                        'date':         pd.Timestamp(row['ds']).strftime('%Y-%m-%d'),
                        'kwh_forecast': round(yhat, 1),
                        'kwh_lower':    round(max(0, yhat - avg_std * 0.5), 1),
                        'kwh_upper':    round(yhat + avg_std * 0.5, 1),
                        'source':       f'lstm_{self.energy_model.backend}',
                    })
                return result

        except Exception as e:
            print(f"[ModelRegistry] forecast_energy error: {e}")
            return []

    def forecast_waste(self, pipeline, periods: int = 7) -> list:
        """
        Iterative autoregressive Random Forest forecast for waste (lbs/day).
        Each prediction feeds back as a lag feature for the next day.
        """
        waste_daily  = pipeline.waste_daily.copy()
        last_date    = waste_daily['date'].max()

        if self.waste_model is None:
            avg = float(waste_daily['total_waste_lbs'].tail(7).mean())
            return [
                {
                    'date':         (last_date + pd.Timedelta(days=i+1)).strftime('%Y-%m-%d'),
                    'lbs_forecast': round(avg, 1),
                    'lbs_lower':    round(avg * 0.85, 1),
                    'lbs_upper':    round(avg * 1.15, 1),
                    'source':       'rolling_mean',
                }
                for i in range(periods)
            ]

        try:
            SCHOOL_MONTHS = {1, 2, 3, 4, 5, 9, 10, 11, 12}
            # Seed lag values from the tail of real data
            lag_1  = float(waste_daily['total_waste_lbs'].iloc[-1])
            lag_7  = float(waste_daily['total_waste_lbs'].iloc[-7])
            roll7  = float(waste_daily['total_waste_lbs'].tail(7).mean())

            result     = []
            predictions: list = []

            for i in range(periods):
                future_date = last_date + pd.Timedelta(days=i+1)
                feat = {
                    'month':                           future_date.month,
                    'day_of_week':                     future_date.dayofweek,
                    'is_weekend':                      int(future_date.dayofweek >= 5),
                    'is_school_session':               int(future_date.month in SCHOOL_MONTHS),
                    'quarter':                         future_date.quarter,
                    'population_count':                float(waste_daily['population_count'].tail(7).mean()),
                    'total_waste_lbs_lag_1d':          lag_1,
                    'total_waste_lbs_lag_7d':          lag_7,
                    'total_waste_lbs_rolling_mean_7d': roll7,
                }
                row_df = pd.DataFrame([feat])
                yhat   = float(np.clip(self.waste_model.rf.predict(row_df)[0], 0, None))

                result.append({
                    'date':         future_date.strftime('%Y-%m-%d'),
                    'lbs_forecast': round(yhat, 1),
                    'lbs_lower':    round(max(0, yhat * 0.85), 1),
                    'lbs_upper':    round(yhat * 1.15, 1),
                    'source':       'random_forest',
                })

                # Slide lag window forward
                predictions.append(yhat)
                lag_1 = yhat
                lag_7 = predictions[-7] if len(predictions) >= 7 else lag_7
                roll7 = float(np.mean(predictions[-7:]))

            return result

        except Exception as e:
            print(f"[ModelRegistry] forecast_waste error: {e}")
            return []

    def forecast_water(self, pipeline, periods: int = 7) -> list:
        """
        Water forecast: uses LSTM model if available, else falls back to
        seasonal rolling-mean extrapolation.
        """
        water_daily = pipeline.water_daily.copy()
        last_date   = water_daily['date'].max()

        # ── LSTM path ──────────────────────────────────────────
        if self.water_lstm_model is not None:
            try:
                fc      = self.water_lstm_model.forecast_future(water_daily, periods=periods)
                avg_std = float(water_daily['water_gallons'].std())
                result  = []
                for _, row in fc.iterrows():
                    yhat = float(row['yhat'])
                    result.append({
                        'date':             pd.Timestamp(row['ds']).strftime('%Y-%m-%d'),
                        'gallons_forecast': round(yhat, 1),
                        'gallons_lower':    round(max(0, yhat - avg_std * 0.5), 1),
                        'gallons_upper':    round(yhat + avg_std * 0.5, 1),
                        'source':           f'lstm_{self.water_lstm_model.backend}',
                    })
                return result
            except Exception as e:
                print(f"[ModelRegistry] forecast_water LSTM error: {e}")

        # ── seasonal rolling fallback ────────────────────────────
        try:
            recent  = water_daily.tail(28).copy()
            recent['dow'] = recent['date'].dt.dayofweek
            dow_avg     = recent.groupby('dow')['water_gallons'].mean()
            global_avg  = float(water_daily['water_gallons'].tail(7).mean())
            global_std  = float(water_daily['water_gallons'].tail(28).std())

            result = []
            for i in range(periods):
                future_date = last_date + pd.Timedelta(days=i+1)
                dow  = future_date.dayofweek
                yhat = float(dow_avg.get(dow, global_avg))
                result.append({
                    'date':             future_date.strftime('%Y-%m-%d'),
                    'gallons_forecast': round(yhat, 1),
                    'gallons_lower':    round(max(0, yhat - global_std * 0.5), 1),
                    'gallons_upper':    round(yhat + global_std * 0.5, 1),
                    'source':           'seasonal_rolling',
                })
            return result
        except Exception as e:
            print(f"[ModelRegistry] forecast_water error: {e}")
            return []

    # ── UI status helper ─────────────────────────────────────

    def get_model_status_cards(self) -> list:
        """
        Returns a list of model-status dicts consumed by the AI Insights page.
        Merges saved training metrics with live load status.
        """
        energy_m = self.metrics.get('energy', {})
        waste_m  = self.metrics.get('waste',  {})
        water_m  = self.metrics.get('water',  {})

        cards = [
            {
                'name':    self._energy_name or 'energy_prophet',
                'status':  'Active'   if self.energy_model else 'Not Trained',
                'metric':  f"MAPE {energy_m.get('mape', '--')} %"
                           if energy_m else None,
                'trained': energy_m.get('trained_at', '--'),
            },
            {
                'name':    self._water_lstm_name or 'water_lstm',
                'status':  'Active'  if self.water_lstm_model else 'Pending',
                'metric':  f"MAPE {water_m.get('mape', '--')} %"
                           if water_m else None,
                'trained': water_m.get('trained_at', '--'),
            },
            {
                'name':    'waste_rf',
                'status':  'Active'   if self.waste_model else 'Not Trained',
                'metric':  f"MAPE {waste_m.get('mape', '--')} %"
                           if waste_m else None,
                'trained': waste_m.get('trained_at', '--'),
            },
        ]
        return cards
