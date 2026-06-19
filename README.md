# Intraday Trading Intelligence & Supervision Platform

An institutional-grade, real-time intraday trading intelligence and trade supervision platform for Indian equities (NSE). The platform ingests live market data streams, calculates microstructure features, and runs a two-phase machine learning pipeline (Ensemble Entry + PyTorch Temporal Transformer Exit) to generate probabilistic trade recommendations and live exit telemetry on a real-time WebSocket-driven dashboard.

---

## 🖥️ System Architecture & Data Flow

The system operates as a distributed real-time pipeline split into ingestion, modeling, and visualization layers.

```
                   ┌──────────────────────────────┐
                   │   Angel One SmartAPI (NSE)   │
                   └──────────────┬───────────────┘
                                  │ (Live WebSockets & REST)
                                  ▼
┌───────────────────────────────────────────────────────────────────┐
│                       BACKEND (app.py)                            │
│                                                                   │
│  ┌────────────────────────┐         ┌──────────────────────────┐  │
│  │   Orderflow Harvester  │         │  Incremental Cache       │  │
│  │  (5-level depth Snaps) │         │  (300 candles, -90% API) │  │
│  └───────────┬────────────┘         └────────────┬─────────────┘  │
│              │ (Parquet Logs)                    │
│              ▼                                   ▼
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                     Inference Engine                        │  │
│  │                                                             │  │
│  │  [Phase 1: Entry Ensemble]      [Phase 2: Exit Engine]      │  │
│  │  - LightGBM + XGBoost           - PyTorch Transformer       │  │
│  │  - 71 indicator feature store   - LightGBM Exit Policy      │  │
│  └───────────────────────────────────────────────┬─────────────┘  │
└──────────────────────────────────────────────────┼────────────────┘
                                                   │ (Flask-SocketIO / Eventlet)
                                                   ▼
┌───────────────────────────────────────────────────────────────────┐
│                       FRONTEND (React)                            │
│                                                                   │
│   - Live Setup Quality Gauge      - Signal Active Indicators     │
│   - Real-time Exit Telemetry      - Active Trade Supervision (UI)│
└───────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Key Architectural Features

### 1. Ingestion & Order Book Microstructure Ingestion
*   **SnapQuote WebSocket Client:** Integrates with Angel One SmartAPI to capture live tick updates and 5-level market depth (LTP, bid/ask quantities, spreads) for **6 priority symbols** (`RELIANCE`, `TVSMOTOR`, `LT`, `TITAN`, `SIEMENS`, `TATAELXSI`).
*   **microstructure Aggregation:** Calculates order flow imbalances, depth imbalance, spread, weighted mid-price, and tick-by-tick price impact in real time ([orderflow_harvester.py](file:///d:/Model/Human_ML/orderflow_harvester.py)).
*   **Production Lifecycle:** Built for 1-year unattended operation with automatic TOTP daily re-authentication, market calendar gating (weekends/holidays), rotating log systems, and disk space alerts.

### 2. Feature Store (71 Features)
The feature engineering module ([feature_engineering.py](file:///d:/Model/Human_ML/feature_engineering.py)) computes exactly **71 technical indicators** across 7 groups on a 300-candle rolling window:
1.  **Price & Return Structure:** Log returns (1m, 3m, 5m), candle body/wick percentages, multi-timeframe returns (15m, 30m), and price momentum.
2.  **Trend & Market Structure:** EMAs (9, 21, 50), normalised slopes, distances, swing highs/lows, and **intraday Fibonacci retracement lines** (0.382, 0.618, 0.786).
3.  **Momentum Indicators:** RSI (14-period + slope), Stochastic ($K$/$D$), MACD histogram, and rolling momentum consistency.
4.  **Volatility Context:** ATR (14, 5m), Bollinger Band positioning, rolling volatility percentile, and volatility regime classification.
5.  **Volume & Participation:** Volume z-scores, volume ratio, cumulative VWAP, VWAP distance, OBV delta, volume price correlation, and volume spike flags.
6.  **Session & Time Context:** Cyclic time encoding (sin/cos of time of day and day of week), session open-range, lunch-time, and close-window flags.
7.  **Higher-Timeframe Bias:** 15m trend direction, daily high/low boundaries, and gap detection.

### 3. Model Engine (Two-Phase ML Pipeline)

#### Phase 1: Directional Entry signals
*   **Models:** An ensemble of symbol-specific **LightGBM** and **XGBoost** classifiers.
*   **Tuning:** Auto-parameterised using **Optuna** and validated via Walk-Forward Validation (TimeSeriesSplit) to prevent leakage.
*   **Signal Aggregation:** Predictions are combined using a learned ensemble weight:
    $$\text{long\_prob} = w_{\text{lgb}} \cdot P_{\text{lgb\_long}} + (1 - w_{\text{lgb}}) \cdot P_{\text{xgb\_long}}$$
*   **Targeting:** Symmetrical labeling requiring price to reach a **+1.5 ATR target** before hitting a **-1.0 ATR stop** within a 15-candle window.

#### Phase 2: Sequential Exit Policy
*   **Models:** PyTorch **Temporal Transformer** (`ExitTransformer` with causal multi-head attention) paired with a **LightGBM** exit policy model.
*   **Inputs:** Tracks sequential trade context (step count, current ATR, price path from entry in ATR units, MFE/MAE, volatility expansion, pullback depth, and momentum decay).
*   **Latency Execution:** Runs on CPU to guarantee low-latency, deterministic inference on 1-minute candle closes.

### 4. Real-time Telemetry Dashboard
*   **Server Concurrency:** Powered by Flask-SocketIO utilizing the **Eventlet** monkey-patched asynchronous loop.
*   **API Optimization:** Employs an **incremental candle caching layer** that fetches historical candles once at startup, appending subsequent closes in memory. This decreases broker API overhead by **~90%**.
*   **WebSocket Stream:** Emits real-time setup quality (entry probability), trade status, and active trade metrics (**PnL, Max Favorable Excursion [MFE], Max Adverse Excursion [MAE]** in ATR units).
*   **Frontend Client:** Built with **React (Vite, TS)** and styled using a curated palette via Tailwind CSS and shadcn components.

---

## 🛠️ File Structure & Responsibilities

| File / Directory | Role |
| :--- | :--- |
| `app.py` | Main server file; orchestrates WebSocket connections, background threads, and cached candle states. |
| `inference.py` | Phase-1 inference pipeline (calculates feature store, runs LightGBM/XGBoost ensembles). |
| `live_exit_engine.py` | Shared Phase-2 exit manager loading the PyTorch Transformer and LightGBM exit models. |
| `feature_engineering.py` | Central module calculating the 71-feature store. |
| `orderflow_harvester.py` | Standalone script collecting tick data and orderbook snaps from the live WebSocket. |
| `exit_transformer.py` | PyTorch network architecture and training code for the Temporal Transformer. |
| `train_universal.py` | Walk-Forward Optuna training pipeline for entry ensemble models. |
| `trade_simulator.py` | End-to-end trading system simulation under modeled market frictions and slippage. |
| `backtest.py` | Backtests historical execution pipelines. |
| `Web/` | React frontend application. |

---

## ⚙️ Setup & Running Instructions

### Prerequisites
1.  **Python 3.10+**
2.  **Node.js / Bun** (for React UI)
3.  **Angel One Developer Credentials** (API_KEY, CLIENT_ID, PASSWORD, TOTP_SECRET)

### Installation
1.  Clone the repository:
    ```bash
    git clone https://github.com/SuryaPranav2k5/Intraday-Trading-Intelligence-Platform.git
    cd Intraday-Trading-Intelligence-Platform
    ```
2.  Install Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Configure API keys in the `.env` files within both the root directory and the `SmartApi` directory:
    ```env
    API_KEY="your_api_key"
    CLIENT_ID="your_client_id"
    PASSWORD="your_password"
    TOTP_SECRET="your_totp_secret_key"
    ```

---

### Executing the Platform

#### Step 1: Start Ingestion & Microstructure Collection
Run the harvester in a persistent background screen to record orderbook snap metrics:
```bash
python orderflow_harvester.py
```

#### Step 2: Start the Backend Server
This launches the Flask-SocketIO thread. It automatically authenticates, caches ~300 candles for warm-up, and listens for the 1-minute updates.
```bash
python app.py
```

#### Step 3: Run the Web Dashboard
Navigate to the frontend folder, install the packages, and start the development server:
```bash
cd Web
npm install   # or bun install
npm run dev   # or bun run dev
```
Open `http://localhost:3000` to view the live dashboard interface.
