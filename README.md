# 🌍 EcoVision — Environmental Intelligence Platform

> AI-powered environmental intelligence dashboard. Tracks energy, water, waste & transport with LSTM/Random Forest forecasting, anomaly detection, real-time alerts, and AI-generated insights. Built with Flask, TensorFlow, Prophet, scikit-learn & Chart.js. Deploy-ready for Render.

---

## ✨ Features

| Module | Highlights |
|---|---|
| **Dashboard** | Sustainability score, live KPI cards, cost overview, AI alert bell |
| **Energy** | LSTM time-series forecast, actual vs. predicted chart, anomaly markers |
| **Water** | LSTM forecast, 2-sigma anomaly detection, red-dot anomaly overlays |
| **Waste** | Random Forest forecasting, waste-stream breakdown, trend analysis |
| **AI Insights** | GPT-style contextual insights, recommendations, severity scoring |
| **Settings** | Theme toggle (dark / light), notification preferences |

### Core Capabilities
- **LSTM forecasting** for energy and water consumption (MAPE ~6%)
- **Random Forest** model for waste stream prediction
- **Prophet** integration for seasonal decomposition
- **2-sigma anomaly detection** — flags phantom loads, HVAC leaks, water spikes
- **SQLite alert database** — rule-based injection, acknowledge & resolve workflow
- **Real-time notification bell** with unread badge
- **Dark mode** across all 6 pages
- **Render deployment** ready (gunicorn + persistent disk + conditional model training)

---

## 🛠️ Tech Stack

**Backend**
- Python 3.11 · Flask 3.1 · Gunicorn
- TensorFlow / Keras (LSTM models)
- scikit-learn (Random Forest, preprocessing)
- Prophet (seasonal forecasting)
- Pandas · NumPy · SciPy
- SQLite (alerts database)

**Frontend**
- Bootstrap 5 · HTML5 · CSS3
- Chart.js (time-series, bar, doughnut charts)
- Vanilla JavaScript (ES6+)

**Deployment**
- Render (Web Service + Persistent Disk)

---

## 📁 Project Structure

```
EcoVision/
├── app.py                    # Flask app — all routes & API endpoints
├── train_models.py           # Standalone model training script
├── requirements.txt          # All Python dependencies
│
├── dataset/                  # Raw CSV data files
│   ├── energy_consumption.csv
│   ├── water_consumption.csv
│   ├── waste_generation.csv
│   ├── transport_fuel.csv
│   └── emission_factors.csv
│
├── models/                   # Trained model artifacts (.pkl)
│   ├── energy_lstm.pkl
│   ├── water_lstm.pkl
│   └── waste_rf.pkl
│
├── utils/                    # Core Python modules
│   ├── data_loader.py        # Dataset ingestion & model registry
│   ├── data_pipeline.py      # Feature engineering & health checks
│   ├── model_trainer.py      # LSTM + RF training logic
│   ├── anomaly_detector.py   # 2-sigma anomaly detection
│   └── db.py                 # SQLite alert CRUD
│
├── templates/                # Jinja2 HTML templates
│   ├── dashboard.html
│   ├── energy.html
│   ├── water.html
│   ├── waste.html
│   ├── ai_insights.html
│   └── settings.html
│
├── static/
│   ├── css/
│   │   ├── style.css         # Global + dark mode styles
│   │   └── energy.css        # (per-page CSS)
│   └── js/
│       ├── dashboard.js
│       └── energy.js         # (per-page JS)
│
├── data/                     # Runtime data (SQLite DB lives here)
│   └── .gitkeep
│
├── render.yaml               # Render deployment config
├── Procfile                  # gunicorn start command
├── runtime.txt               # Python 3.11.9
├── build.sh                  # Render build script
└── .env.example              # Environment variable reference
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- pip

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/YOUR_USERNAME/ecovision.git
cd ecovision

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

```bash
cp .env.example .env
# Edit .env and set your SECRET_KEY
```

### 4. Train Models

```bash
python train_models.py
```

> Trains Energy LSTM, Water LSTM, and Waste Random Forest models and saves them to `models/`. Takes ~2–5 minutes.

### 5. Run the App

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000)

---

## 🌐 Deploy to Render

This project is fully configured for [Render](https://render.com).

1. Push your repo to GitHub (ensure `models/*.pkl` files are committed)
2. Go to **render.com → New → Web Service**
3. Connect your GitHub repo — Render auto-detects `render.yaml`
4. Click **Create Web Service**

Render will:
- Install all dependencies from `requirements.txt`
- Run `build.sh` (skips model training if `.pkl` files already exist)
- Start the app with `gunicorn --workers 1 --threads 4 --timeout 120`
- Mount a persistent disk at `/data` for SQLite

> **Recommended plan:** Standard ($7/month) — TensorFlow inference requires ~512 MB RAM.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Dashboard |
| GET | `/energy` | Energy analytics page |
| GET | `/water` | Water analytics page |
| GET | `/waste` | Waste analytics page |
| GET | `/ai-insights` | AI insights page |
| GET | `/settings` | Settings page |
| GET | `/api/energy/forecast` | LSTM energy forecast (JSON) |
| GET | `/api/water/forecast` | LSTM water forecast (JSON) |
| GET | `/api/waste/forecast` | RF waste forecast (JSON) |
| GET | `/api/alerts` | Active alerts list |
| POST | `/api/alerts/<id>/resolve` | Resolve an alert |
| POST | `/api/alerts/<id>/acknowledge` | Acknowledge an alert |
| GET | `/api/health` | Data pipeline health report |

---

## 🤖 ML Models

| Model | Algorithm | Target | MAPE |
|---|---|---|---|
| Energy LSTM | LSTM (TensorFlow/Keras) | kWh consumption | ~6% |
| Water LSTM | LSTM (TensorFlow/Keras) | m³ consumption | ~6% |
| Waste RF | Random Forest (scikit-learn) | kg waste generated | — |

Anomaly detection uses a **2-sigma absolute deviation** method — values beyond 2 standard deviations from the rolling mean are flagged and highlighted in charts with red markers.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/YourFeature`
3. Commit your changes: `git commit -m 'Add YourFeature'`
4. Push to the branch: `git push origin feature/YourFeature`
5. Open a Pull Request

---

## 📄 License

MIT License — open for academic and personal use.

---

**Built with 💚 for a sustainable future**

