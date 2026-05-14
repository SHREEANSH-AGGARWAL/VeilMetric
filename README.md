# VeilMetric

**Privacy-preserving portfolio risk analytics using Fully Homomorphic Encryption.**

VeilMetric is a research project that predicts multi-asset portfolio performance using an ensemble of XGBoost regression agents — and then runs the *exact same inference* on encrypted data via Concrete-ML (Zama). The results are compared side-by-side to quantify the accuracy cost of client-side encryption.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                       DATA PIPELINE                                 │
│                                                                     │
│  data_ingestion.py    →  market_log_returns.csv   (10yr log rets)   │
│  User_Archetype_*.py  →  user_training_data.csv   (synthetic users) │
│  Moving_Window_Converter.py  →  enhanced_training_data.csv          │
│                                (weights + context → targets)        │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
          dynamic_XGBoost.py             FHE.py
          (plaintext training)       (Concrete-ML compilation)
                    │                         │
           agent_*.json              fhe_vault/*/
          (3 model files)         (serialized FHE circuits)
                    │                         │
                    └────────────┬────────────┘
                                 │
                            main.py
                     (FastAPI inference server)
                                 │
                   research_dashboard.html
                    (browser-based frontend)
```

### Components

| File | Purpose |
|------|---------|
| `data_ingestion.py` | Downloads 10 years of daily close data for 6 assets + INR/USD FX from yfinance. Computes log returns. |
| `Moving_Window_Converter.py` | Pairs each synthetic portfolio with a random 4-year market window. Extracts Nifty 50 trailing momentum and volatility as context features. Computes target return, drawdown, and volatility. |
| `dynamic_XGBoost.py` | Trains three plaintext XGBoost agents (Return, Drawdown, Volatility) on the enhanced dataset. |
| `FHE.py` | Trains three Concrete-ML quantized XGBoost agents and compiles them into FHE circuits. Exports deployment artifacts to `fhe_vault/`. |
| `main.py` | FastAPI server that loads both model sets at startup, accepts portfolio weights via API, runs dual-engine inference, and computes derived analytics (Jensen's Alpha, Sharpe, Calmar, CVaR, etc.). |
| `research_dashboard.html` | Single-page glassmorphism dashboard with Chart.js visualizations, animated loading sequence, and an explanatory modal. |

---

## Model Specifications

### Standard XGBoost (3 agents)
- 150 trees, max depth 6, learning rate 0.1
- `tree_method='hist'` for fast histogram-based training
- Trained on full-precision features

### Concrete-ML FHE XGBoost (3 agents)
- 30 trees, max depth 4, 5-bit quantization
- Compiled into FHE circuits via Concrete-ML by Zama
- Inference runs entirely on encrypted data — the server never sees plaintext inputs

### Feature Vector (8 dimensions)
| # | Feature | Source |
|---|---------|--------|
| 1–6 | Portfolio weights (Gold, Silver, BTC, ETH, Nifty 50, Nippon India) | User input |
| 7 | Context Momentum | Trailing 1-year Nifty 50 cumulative log return |
| 8 | Context Volatility | Trailing 1-year Nifty 50 annualized volatility |

### Prediction Targets (per agent)
| Agent | Target | Horizon |
|-------|--------|---------|
| Return | Total portfolio return | 4 years |
| Drawdown | Maximum peak-to-trough drawdown | 4 years |
| Volatility | Annualized portfolio volatility | 4 years |

---

## Derived Analytics

All metrics below are computed dynamically from the three agent outputs — nothing is hardcoded.

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Jensen's Alpha** | `R_portfolio − [R_f + β(R_market − R_f)]` | CAPM excess return; positive = outperforming |
| **Sharpe Ratio** | `(R_annual − R_f) / σ` | Risk-adjusted return; > 1.0 = good |
| **Calmar Ratio** | `R_total / MaxDrawdown` | Return per unit of drawdown |
| **CVaR (95%)** | `min(Drawdown × 1.3, 1.0)` | Tail-risk proxy |
| **Upside Capture** | `(R_portfolio / R_nifty) × 100` | Benchmark-relative performance |
| **Beta** | `σ_portfolio / σ_nifty` | Systematic risk exposure |
| **Wealth Projection** | `Wealth × (1 + R_total)` | Terminal portfolio value |

---

## Setup

### Prerequisites
- **Python 3.10** (required — `concrete-ml` does not support 3.11+)
- Conda (recommended) or Docker

### Installation

```bash
git clone https://github.com/SHREEANSH-AGGARWAL/VeilMetric.git
cd VeilMetric

# Create and activate the conda environment
conda create -n veilmetric python=3.10 -y
conda activate veilmetric

# Install dependencies
pip install -r requirements.txt
```

### Running the Full Pipeline (first time only)

```bash
# 1. Download market data
python data_ingestion.py

# 2. Generate enhanced training data
python Moving_Window_Converter.py

# 3. Train standard XGBoost agents
python dynamic_XGBoost.py

# 4. Compile FHE circuits (takes several minutes)
python FHE.py
```

### Starting the Server

```bash
conda activate veilmetric
python main.py
```

The dashboard will be available at **http://localhost:8000**.

### Docker (alternative)

```bash
# Build and run (pre-trained models must already exist in the repo)
docker build -t veilmetric .
docker run -p 8000:8000 veilmetric
```

---

## API Reference

### `GET /api/health`
Returns loaded engine status.

```json
{
  "status": "ok",
  "engines": {
    "standard": ["return", "drawdown", "volatility"],
    "fhe": ["return", "drawdown", "volatility"]
  }
}
```

### `GET /api/context`
Returns live Nifty 50 market context (momentum + volatility).

### `POST /api/predict/benchmark`
Runs dual-engine inference and returns full analytics.

**Request body:**
```json
{
  "weights": {
    "Gold": 0.20,
    "Silver": 0.05,
    "Bitcoin": 0.10,
    "Ethereum": 0.05,
    "Nifty": 0.40,
    "Nippon": 0.20
  },
  "wealth": 100000
}
```

**Response structure:**
```json
{
  "status": "success",
  "agents": {
    "return":     { "standard": 0.42, "fhe": 0.41, "delta": -0.01 },
    "drawdown":   { "standard": 0.18, "fhe": 0.19, "delta": 0.01 },
    "volatility": { "standard": 0.22, "fhe": 0.21, "delta": -0.01 }
  },
  "analytics": {
    "standard": { "jensens_alpha": 5.2, "sharpe_ratio": 1.4, "..." : "..." },
    "fhe":      { "jensens_alpha": 4.9, "sharpe_ratio": 1.3, "..." : "..." }
  },
  "market": { "context_momentum": 0.12, "context_vol": 0.18 },
  "input":  { "weights": { "..." : "..." }, "wealth": 100000 }
}
```

---

## Dashboard

The research dashboard provides:

- **Dual-engine comparison table** — raw predictions from both engines with accuracy deltas
- **8 analytics cards** — Jensen's Alpha, Sharpe, Calmar, CVaR, Wealth Projection, Max Drawdown (USD), Upside Capture, Expected Return
- **4 interactive charts** — Radar (risk profile), Bar (FHE deltas), Donut (allocation), Line (48-month wealth trajectory)
- **Animated loading overlay** — step-by-step progress with audio feedback
- **Explanatory modal** — plain-English descriptions of every metric for non-technical users

---

## Project Structure

```
VeilMetric/
├── main.py                      # FastAPI inference server
├── research_dashboard.html      # Frontend dashboard
├── data_ingestion.py            # Market data download
├── Moving_Window_Converter.py   # Feature engineering
├── dynamic_XGBoost.py           # Standard model training
├── FHE.py                       # FHE circuit compilation
├── requirements.txt             # Pinned dependencies
├── Dockerfile                   # Container build
├── .dockerignore                # Docker exclusions
├── .python-version              # Pins Python 3.10
├── agent_return.json            # Trained standard model
├── agent_drawdown.json          # Trained standard model
├── agent_volatility.json        # Trained standard model
├── fhe_vault/                   # FHE deployment artifacts
│   ├── return/
│   ├── drawdown/
│   └── volatility/
└── archives/                   # Archived experimental scripts
```

---

## License

This project is for academic and research purposes.
