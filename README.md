# SLT AI Module

> **Python Flask AI microservice** for the SLT After-Service Issue Management & Workforce Optimization System.
>
> Provides fault volume forecasting (Prophet), geographic demand clustering (K-Means), and nearest-technician route optimisation (Dijkstra + Haversine) via a REST API consumed by the Spring Boot backend and React admin portal.

---

## Table of Contents

- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Quick Start — Local Development](#quick-start--local-development)
- [Quick Start — Docker](#quick-start--docker)
- [Environment Variables](#environment-variables)
- [API Reference](#api-reference)
- [Running Tests](#running-tests)
- [Project Structure](#project-structure)
- [Models](#models)
- [Troubleshooting](#troubleshooting)

---

## Architecture

```
Admin Portal (React.js) ──────┐
                              │  HTTP/REST
Spring Boot API (:8080) ──────┼──────────► Flask AI Module (:5000)
                              │                    │
                              └────────────────────┤
                                                   │  SQLAlchemy / PyMySQL
                                              MySQL 8.0 (:3306)
                                         slt_fieldops_db
```

The AI module is a **separate microservice**. It runs on port `5000` alongside the Spring Boot backend. When MySQL is unavailable (development/demo mode), all endpoints automatically fall back to seeded synthetic data — the API always returns a valid response.

---

## Prerequisites

| Requirement | Version | Notes |
|------------|---------|-------|
| Python | 3.9 – 3.11 | 3.11 recommended |
| pip | ≥ 23.0 | `pip install --upgrade pip` |
| MySQL | 8.0+ | Optional — synthetic fallback available |
| Docker | 24+ | Only for Docker deployment |
| docker-compose | 2.x | Only for Docker deployment |

---

## Quick Start — Local Development

### 1. Clone and enter the AI module directory

```bash
cd slt-ai-module
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> ⚠️ **Prophet / pystan** compiles C++ code on first install — this takes **5–10 minutes**. This is normal.

### 4. Configure environment

```bash
cp .env.example .env   # or copy the .env template
```

Edit `.env` and set your database credentials:

```dotenv
DB_HOST=localhost
DB_PORT=3306
DB_NAME=slt_fieldops_db
DB_USER=root
DB_PASSWORD=yourpassword
```

If you don't have MySQL running, leave it as-is — the module will use synthetic data automatically.

### 5. Start the Flask server

```bash
python app.py
```

You should see:

```
====================================================
  SLT AI Module — Starting Flask Server
  Host:    0.0.0.0:5000
  Debug:   True
  DB:      localhost:3306/slt_fieldops_db
====================================================
✅ Database connected: localhost:3306/slt_fieldops_db
 * Running on http://0.0.0.0:5000
```

### 6. Verify it's working

```bash
curl http://localhost:5000/api/ai/health/ping
# {"status": "ok", "ts": "2026-04-20T09:15:30Z"}

curl http://localhost:5000/api/ai/health
# Full health status with DB counts and model readiness

curl "http://localhost:5000/api/ai/predictions?horizon=7"
# 7-day Prophet forecast

curl "http://localhost:5000/api/ai/optimize-route?lat=6.9271&lng=79.8612"
# Nearest technicians to Colombo
```

---

## Quick Start — Docker

### Start all services (full stack)

```bash
# From the project root (where docker-compose.yml lives)
docker-compose up -d

# Check status
docker-compose ps

# Tail AI module logs
docker-compose logs -f ai-module
```

### Start only AI module + database

```bash
docker-compose up -d mysql ai-module
```

### Rebuild after code changes

```bash
docker-compose up -d --build ai-module
```

### Run tests inside the container

```bash
docker-compose exec ai-module python -m pytest tests/ -v
```

### Stop everything

```bash
docker-compose down          # keep volumes
docker-compose down -v       # remove volumes (DELETES DATABASE)
```

---

## Environment Variables

All variables are in `.env`. Copy `.env.example` to get started.

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_ENV` | `development` | `development` or `production` |
| `FLASK_DEBUG` | `true` | Enable Flask debug mode |
| `FLASK_PORT` | `5000` | Port to listen on |
| `DB_HOST` | `localhost` | MySQL hostname |
| `DB_PORT` | `3306` | MySQL port |
| `DB_NAME` | `slt_fieldops_db` | Database name |
| `DB_USER` | `root` | MySQL username |
| `DB_PASSWORD` | `1234` | MySQL password |
| `MODEL_DIR` | `./models/saved` | Where trained models are saved |
| `FORECAST_HORIZON_DAYS` | `30` | Default Prophet forecast horizon |
| `FORECAST_MIN_HISTORY_DAYS` | `90` | Minimum days of data to train Prophet |
| `KMEANS_N_CLUSTERS` | `5` | Number of K-Means clusters |
| `ROUTE_SEARCH_RADIUS_KM` | `50` | Max technician search radius (km) |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `LOG_FILE` | `./logs/ai_module.log` | Log file path |

---

## API Reference

Base URL: `http://localhost:5000/api/ai`

### Health

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Full health: DB status, model readiness, uptime |
| `GET` | `/health/ping` | Liveness probe — always 200 if Flask is running |
| `GET` | `/health/ready` | Readiness — returns 503 if no models importable |

### Predictions (Prophet)

| Method | Endpoint | Query Params | Description |
|--------|----------|--------------|-------------|
| `GET` | `/predictions` | `horizon=30`, `category=BROADBAND`, `format=summary` | Fault volume forecast |
| `GET` | `/predictions/categories` | `horizon=14` | Per-category forecasts |
| `GET` | `/predictions/history` | `days=90` | Raw historical data only |
| `POST` | `/predictions/retrain` | — | Force Prophet retraining |

### Clusters (K-Means)

| Method | Endpoint | Query Params | Description |
|--------|----------|--------------|-------------|
| `GET` | `/clusters` | `n_clusters=5`, `days=180`, `category=FIBER` | Geographic fault clusters |
| `GET` | `/clusters/map` | `n_clusters=5` | Clusters with SVG pixel coordinates |
| `GET` | `/clusters/heatmap` | `resolution=50` | Grid density for Leaflet heatmap |
| `POST` | `/clusters/retrain` | — | Force K-Means refit |

### Route Optimisation (Dijkstra)

| Method | Endpoint | Query Params | Description |
|--------|----------|--------------|-------------|
| `GET` | `/optimize-route` | `lat=6.93&lng=79.86&limit=5` | **Required.** Nearest technicians |
| `GET` | `/optimize-route/nearby` | `lat=&lng=&radius=20` | Radius-based lookup (no Dijkstra) |
| `POST` | `/optimize-route/batch` | — | Batch assign: multiple faults at once |

**Batch request body:**
```json
{
  "faults": [
    { "fault_id": 1, "lat": 6.9271, "lng": 79.8612, "priority": "HIGH" },
    { "fault_id": 2, "lat": 7.2906, "lng": 80.6337, "priority": "MEDIUM" }
  ]
}
```

### Dashboard

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/dashboard` | All widgets in one request |
| `GET` | `/dashboard/summary` | Lightweight KPI cards only |
| `GET` / `POST` | `/dashboard/classify?description=...` | Predict category + priority from text |
| `POST` | `/dashboard/retrain` | Retrain all models at once |

### Standard Response Format

```json
{
  "success": true,
  "data":    { "...": "..." },
  "message": "30-day fault volume forecast",
  "timestamp": "2026-04-20T09:15:30.000000+00:00"
}
```

Error responses:
```json
{
  "success":   false,
  "error":     "Latitude 51.5 out of Sri Lanka bounds [5.9, 9.9]",
  "timestamp": "2026-04-20T09:15:30.000000+00:00"
}
```

---

## Running Tests

```bash
# All tests (155+ cases, no DB required)
python -m pytest tests/ -v

# Individual test files
python -m pytest tests/test_forecasting.py -v   # 45 tests
python -m pytest tests/test_clustering.py  -v   # 35 tests
python -m pytest tests/test_routes.py      -v   # 75 tests

# Filter by test name keyword
python -m pytest tests/ -k "classify" -v
python -m pytest tests/ -k "dijkstra or haversine" -v

# Quick summary (no verbose output)
python -m pytest tests/ -q --tb=short

# Stop on first failure
python -m pytest tests/ -x

# With coverage report
pip install pytest-cov
python -m pytest tests/ --cov=. --cov-report=html
open htmlcov/index.html
```

All tests run in **synthetic data mode** — `conftest.py` patches `is_db_available()` to `False` so no MySQL connection is needed.

---

## Project Structure

```
slt-ai-module/
│
├── app.py                      Flask application factory + route registration
├── config.py                   Config class, DB engine, logging setup
├── requirements.txt            All Python dependencies (pinned versions)
├── .env                        Environment variables (DO NOT COMMIT)
├── .env.example                Template — safe to commit
├── Dockerfile                  Multi-stage Docker build
├── .gitignore                  Excludes .env, *.pkl, logs/
├── README.md                   This file
│
├── data/                       Data access layer
│   ├── __init__.py
│   ├── data_extractor.py       MySQL queries (8 methods, parameterised)
│   ├── data_cleaner.py         Time-series + GPS cleaning pipeline
│   ├── feature_engineer.py     SL holidays, monsoon flags, Haversine graph
│   └── synthetic_data.py       Seeded fallback data generator
│
├── models/                     ML model layer
│   ├── __init__.py
│   ├── forecasting.py          Prophet time-series forecaster
│   ├── clustering.py           K-Means geographic clusterer
│   ├── classifier.py           TF-IDF + LogReg fault classifier
│   ├── route_optimizer.py      Dijkstra + Haversine router
│   └── saved/                  Trained model .pkl files (git-ignored)
│       └── .gitkeep
│
├── routes/                     Flask Blueprint endpoints
│   ├── __init__.py             register_blueprints(app) helper
│   ├── health.py               GET /health, /health/ping, /health/ready
│   ├── predictions.py          GET /predictions, POST /retrain
│   ├── clusters.py             GET /clusters, /map, /heatmap
│   ├── route.py                GET /optimize-route, /nearby, /batch
│   └── dashboard.py            GET /dashboard, POST /classify
│
├── utils/                      Shared utilities
│   ├── __init__.py
│   ├── db_connector.py         DBConnector class + query helpers
│   ├── validators.py           Input validation (16 functions)
│   └── formatters.py           Response formatting (18 functions)
│
├── tests/                      Test suite
│   ├── __init__.py
│   ├── conftest.py             Shared fixtures, DB mock, test client
│   ├── test_forecasting.py     45 tests — DataCleaner, FeatureEngineer, Prophet
│   ├── test_clustering.py      35 tests — GPS cleaner, KMeans, annotations
│   └── test_routes.py          75 tests — all 19 endpoints
│
└── logs/                       Runtime logs (git-ignored)
    └── .gitkeep
```

---

## Models

### Prophet Forecaster (`models/forecasting.py`)

- **Algorithm:** Facebook Prophet with multiplicative seasonality
- **Regressors:** `is_holiday`, `is_weekend`, `is_sw_monsoon`, `is_ne_monsoon`, `rolling_mean_7`, `fault_lag_7`
- **Target accuracy:** ≥ 85% (measured by MAPE on 30-day holdout)
- **Persistence:** `models/saved/prophet_model.pkl` — reloaded on restart
- **Fallback:** Linear extrapolation when Prophet library not installed

### K-Means Clusterer (`models/clustering.py`)

- **Algorithm:** scikit-learn KMeans (k=5)
- **Initialisation:** District GPS coordinates as warm-start seeds (Colombo, Kandy, Galle, Jaffna, Batticaloa)
- **Labels:** Nearest district name via Haversine
- **Risk levels:** HIGH (≥25% of faults), MEDIUM (≥15%), LOW (<15%)
- **Persistence:** `models/saved/kmeans_model.pkl`

### Fault Classifier (`models/classifier.py`)

- **Algorithm:** TF-IDF (1–2 ngrams) + Logistic Regression
- **Tasks:** Category (5 classes) + Priority (3 classes)
- **Training:** Auto-trains from MySQL `faults` table on startup; synthetic fallback
- **Fallback:** Keyword matching when scikit-learn not installed

### Dijkstra Router (`models/route_optimizer.py`)

- **Algorithm:** Dijkstra's shortest path with binary min-heap (`heapq`)
- **Edge weights:** Haversine distance (km) × status penalty factor
- **Status penalties:** `AVAILABLE×1.0` → `IN_PROGRESS×2.0` → `OFFLINE×99.0`
- **Speed model:** 30 km/h urban (within 15 km of Colombo), 50 km/h rural

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'prophet'`

```bash
pip install prophet pystan
# Prophet compiles C++ — allow 5–10 minutes
```

### `Database not reachable` on startup

This is **normal** if MySQL isn't running. The module falls back to synthetic data automatically. You'll see:
```
⚠️  Database not available: ...
   AI module will use synthetic data as fallback.
```

To connect to MySQL, update `.env` with your credentials and restart.

### `FileNotFoundError: models/saved/`

```bash
mkdir -p models/saved logs
```

### Prophet takes too long to start

On first startup, Prophet trains on your fault data. This takes ~30–60 seconds. Subsequent starts load the saved pickle file in ~2 seconds.

### Tests fail with `ImportError`

Make sure you're running from the project root with the virtual environment activated:
```bash
cd slt-ai-module
source .venv/bin/activate
python -m pytest tests/ -v
```

### Port 5000 already in use (macOS)

macOS Monterey+ uses port 5000 for AirPlay. Either:
```bash
# Change the port in .env
FLASK_PORT=5001

# Or disable AirPlay Receiver in System Preferences
```

---

## Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Web framework | Flask | 3.0.3 |
| CORS | flask-cors | 4.0.1 |
| Database ORM | SQLAlchemy | 2.0.30 |
| MySQL driver | PyMySQL | 1.1.1 |
| Data processing | pandas | 2.2.2 |
| Numerics | numpy | 1.26.4 |
| Forecasting | Prophet | 1.1.5 |
| ML | scikit-learn | 1.4.2 |
| Geospatial | geopy | 2.4.1 |
| Production server | Gunicorn | (latest) |

---

## Author

**Tharuka Liyanaarachchi**
Registration: LLCSX25311
Module: FC6P01 — Final Year Project
Sri Lanka Telecom After-Service Issue Management & Workforce Optimization System
