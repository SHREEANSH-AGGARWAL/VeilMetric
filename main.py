import logging
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from concrete.ml.deployment import FHEModelServer, FHEModelClient
import xgboost as xgb
import numpy as np

log = logging.getLogger("veilmetric")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
app = FastAPI(title="VeilMetric Research Engine", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

log.info("Initializing VeilMetric Dual-Engine...")

# ─────────────────────────────────────────────
# 1. LOAD STANDARD XGBOOST MODELS
# ─────────────────────────────────────────────
standard_models = {}
for agent in ["return", "drawdown", "volatility"]:
    model = xgb.XGBRegressor()
    try:
        model.load_model(f"agent_{agent}.json")
        standard_models[agent] = model
        log.info("Loaded standard model: %s", agent)
    except Exception:
        log.warning("Could not load agent_%s.json", agent)

# ─────────────────────────────────────────────
# 2. LOAD FHE VAULTS
# ─────────────────────────────────────────────
VAULT_BASE = "./fhe_vault"
fhe_clients = {}
fhe_servers = {}

for agent in ["return", "drawdown", "volatility"]:
    vault_path = os.path.join(VAULT_BASE, agent)
    if os.path.exists(vault_path):
        try:
            fhe_servers[agent] = FHEModelServer(vault_path)
            client = FHEModelClient(vault_path)
            client.generate_private_and_evaluation_keys()
            fhe_clients[agent] = client
            log.info("Loaded FHE vault: %s", agent)
        except Exception as e:
            log.warning("FHE vault %s failed to load: %s", agent, e)

# Feature order must match training column order
WEIGHT_KEYS = ["Gold", "Silver", "Bitcoin", "Ethereum", "Nifty", "Nippon"]


# ─────────────────────────────────────────────
# 3. PYDANTIC MODELS
# ─────────────────────────────────────────────
class PortfolioRequest(BaseModel):
    weights: dict   # {"Gold": 0.20, "Silver": 0.05, ...}
    wealth: float = 100000.0


# ─────────────────────────────────────────────
# 4. ANALYTICS ENGINE
# ─────────────────────────────────────────────
def compute_analytics(ret: float, drawdown: float, vol: float,
                       context_momentum: float, context_vol: float,
                       wealth: float) -> dict:
    """Derive portfolio analytics from model-predicted return, drawdown, and volatility."""
    RISK_FREE_4YR    = 0.16   # 4 % annual × 4 years
    RISK_FREE_ANNUAL = 0.04

    # ── Beta proxy (vol ratio) ──────────────────
    beta = (vol / context_vol) if context_vol > 0 else 1.0

    # ── Jensen's Alpha (CAPM) ───────────────────
    expected_capm   = RISK_FREE_4YR + beta * (context_momentum - RISK_FREE_4YR)
    jensens_alpha   = ret - expected_capm

    # ── Sharpe Ratio (annualised) ───────────────
    annual_ret = (1 + max(ret, -0.9999)) ** (1.0 / 4.0) - 1
    sharpe     = (annual_ret - RISK_FREE_ANNUAL) / vol if vol > 0 else 0.0

    # ── Calmar Ratio ────────────────────────────
    calmar = ret / drawdown if drawdown > 0 else 0.0

    # ── CVaR proxy @ 95 % ───────────────────────
    cvar_proxy = min(drawdown * 1.3, 1.0)

    # ── Risk / Reward ────────────────────────────
    risk_reward = ret / vol if vol > 0 else 0.0

    # ── Safety Score (relative vol) ─────────────
    safety_score = (1.0 - vol / context_vol) * 100 if context_vol > 0 else 0.0

    # ── Upside Capture vs Nifty ─────────────────
    upside_capture = (ret / context_momentum) * 100 if context_momentum != 0 else 0.0

    # ── Wealth projections ───────────────────────
    wealth_projection = wealth * (1 + ret)
    max_drawdown_usd  = wealth * drawdown

    return {
        "jensens_alpha":     round(jensens_alpha * 100, 4),
        "sharpe_ratio":      round(sharpe, 4),
        "calmar_ratio":      round(calmar, 4),
        "cvar_proxy":        round(cvar_proxy * 100, 4),
        "risk_reward":       round(risk_reward, 4),
        "safety_score":      round(safety_score, 4),
        "upside_capture":    round(upside_capture, 4),
        "wealth_projection": round(wealth_projection, 2),
        "max_drawdown_usd":  round(max_drawdown_usd, 2),
        "beta":              round(beta, 4),
        "annual_return":     round(annual_ret * 100, 4),
    }


# ─────────────────────────────────────────────
# 5. ROUTES
# ─────────────────────────────────────────────
@app.get("/")
@app.get("/dashboard")
async def serve_dashboard():
    return FileResponse("research_dashboard.html")


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "engines": {
            "standard": list(standard_models.keys()),
            "fhe":      list(fhe_clients.keys()),
        }
    }


@app.get("/api/context")
async def get_market_context():
    """Return live Nifty 50 market context for the frontend."""
    try:
        from Moving_Window_Converter import get_live_market_context
        momentum, vol = get_live_market_context()
        return {
            "status":           "ok",
            "context_momentum": round(float(momentum), 6),
            "context_vol":      round(float(vol), 6),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Context fetch failed: {str(e)}")


@app.post("/api/predict/benchmark")
async def predict_benchmark(payload: PortfolioRequest):
    try:
        # ── Step 1: live market context ─────────
        from Moving_Window_Converter import get_live_market_context
        context_momentum, context_vol = get_live_market_context()

        # ── Step 2: build feature vector ────────
        weight_vals    = [float(payload.weights.get(k, 0.0)) for k in WEIGHT_KEYS]
        feature_vector = weight_vals + [float(context_momentum), float(context_vol)]
        X = np.array([feature_vector])

        # ── Step 3: run all 6 models ─────────────
        agents_raw = {}
        for agent in ["return", "drawdown", "volatility"]:
            agents_raw[agent] = {"standard": None, "fhe": None, "delta": None}

            # Standard XGBoost
            if agent in standard_models:
                std_val = float(standard_models[agent].predict(X)[0])
                agents_raw[agent]["standard"] = round(std_val, 6)

            # FHE XGBoost
            if agent in fhe_clients:
                client   = fhe_clients[agent]
                server   = fhe_servers[agent]
                enc      = client.quantize_encrypt_serialize(X)
                keys     = client.get_serialized_evaluation_keys()
                enc_pred = server.run(enc, keys)
                dec      = client.deserialize_decrypt_dequantize(enc_pred)
                fhe_val  = float(np.array(dec).flatten()[0])
                agents_raw[agent]["fhe"] = round(fhe_val, 6)

                std_v = agents_raw[agent]["standard"]
                agents_raw[agent]["delta"] = round(fhe_val - std_v, 6) if std_v is not None else None

        # ── Step 4: analytics for BOTH engines ───
        def _safe_analytics(agent_key_ret, agent_key_dd, agent_key_vol, engine):
            r  = agents_raw[agent_key_ret][engine]
            dd = agents_raw[agent_key_dd][engine]
            v  = agents_raw[agent_key_vol][engine]
            if any(x is None for x in [r, dd, v]):
                return {}
            return compute_analytics(r, dd, v, context_momentum, context_vol, payload.wealth)

        standard_analytics = _safe_analytics("return", "drawdown", "volatility", "standard")
        fhe_analytics      = _safe_analytics("return", "drawdown", "volatility", "fhe")

        return {
            "status": "success",
            "agents": agents_raw,
            "market": {
                "context_momentum": round(float(context_momentum), 6),
                "context_vol":      round(float(context_vol), 6),
            },
            "analytics": {
                "standard": standard_analytics,
                "fhe":      fhe_analytics,
            },
            "input": {
                "weights": payload.weights,
                "wealth":  payload.wealth,
            },
        }

    except Exception as e:
        log.error("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)