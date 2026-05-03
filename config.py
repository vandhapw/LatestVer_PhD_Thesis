# ============================================================
# CENTRAL CONFIGURATION — single source of truth
# ============================================================
# All sensor bounds, thresholds, connection strings, and model
# names live here. Files MUST import from this module instead
# of redefining values inline.
#
# Sensor bounds and thresholds are DERIVED from the data on
# first run (see derive_sensor_bounds.py) and persisted to
# sensor_bounds_derived.json. This eliminates the previous
# practice of hardcoding "domain knowledge" that was actually
# memorized from the dataset.
# ============================================================

import os
import json
from pathlib import Path

# ─────────────────────────────────────────────
# 0. SECRETS — loaded from environment
# ─────────────────────────────────────────────
# Load .env if present (no third-party dependency: minimal parser)
_env_path = Path(__file__).with_name(".env")
if _env_path.exists():
    for line in _env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

OLLAMA_API_KEY   = os.environ.get("OLLAMA_API_KEY", "")
MONGODB_URI      = os.environ.get("MONGODB_URI", "")
MONGO_DB         = os.environ.get("MONGO_DB", "server_db")
MONGO_COLLECTION = os.environ.get("MONGO_COLLECTION", "simulated_smart_manufactured_data")
KAFKA_BROKERS    = [b.strip() for b in os.environ.get("KAFKA_BROKERS", "").split(",") if b.strip()]
KAFKA_TOPIC      = os.environ.get("KAFKA_TOPIC", "simulated_digitaltwin_manufactured")
OLLAMA_MODEL     = os.environ.get("OLLAMA_MODEL", "glm-5.1:cloud")

if not OLLAMA_API_KEY:
    print("[config] WARNING: OLLAMA_API_KEY not set. Copy .env.example to .env and fill it in.")
if not MONGODB_URI:
    print("[config] WARNING: MONGODB_URI not set. Copy .env.example to .env and fill it in.")

# ─────────────────────────────────────────────
# 1. SENSOR FEATURE LIST (canonical order)
# ─────────────────────────────────────────────
SENSOR_FEATURES = ["temperature", "vibration", "humidity", "pressure", "energy_consumption"]

# Aliases used in some collections / event payloads
FEATURE_ALIASES = {
    "energy": "energy_consumption",
}

# ─────────────────────────────────────────────
# 2. RUL THRESHOLDS (operational policy, disclosed)
# ─────────────────────────────────────────────
# These are policy thresholds, not learned from data.
# RUL_LOW = below this → urgent maintenance
# RUL_MODERATE = below this → monitor more closely
RUL_LOW       = 50
RUL_MODERATE  = 100

# Risk severity bands (policy)
RISK_HIGH_AT    = 0.6
RISK_MEDIUM_AT  = 0.3

# ─────────────────────────────────────────────
# 3. SENSOR BOUNDS — derived from data, not hardcoded
# ─────────────────────────────────────────────
_BOUNDS_FILE = Path(__file__).with_name("sensor_bounds_derived.json")


def load_sensor_bounds():
    """
    Load sensor bounds from sensor_bounds_derived.json.
    The file is produced by derive_sensor_bounds.py — run that script
    once before the first pipeline run.

    Returns:
        dict like {"temperature": {"min": 60.1, "max": 99.7, "p05": ..., "p95": ...}, ...}
    """
    if not _BOUNDS_FILE.exists():
        return None
    return json.loads(_BOUNDS_FILE.read_text(encoding="utf-8"))


def get_normalization_range(feature: str):
    """
    Return (lo, hi) range for normalizing a sensor reading.
    Uses the 5th–95th percentile from the derived bounds file
    (robust against outliers). Falls back to (0, 100) if bounds
    are missing — the caller should warn or run derive_sensor_bounds.py.
    """
    bounds = load_sensor_bounds()
    if bounds is None:
        return (0.0, 100.0)
    feat = FEATURE_ALIASES.get(feature, feature)
    b = bounds.get(feat)
    if b is None:
        return (0.0, 100.0)
    return (b["p05"], b["p95"])


def get_warn_crit_thresholds():
    """
    Return reasoning-engine warn/crit thresholds.

    Disclosure: these thresholds are derived from the 90th and 99th
    percentile of historical sensor readings (computed by
    derive_sensor_bounds.py). They are NOT learned end-to-end and
    should NOT be evaluated against ground-truth labels that were
    themselves produced using the same percentile cutoffs — that
    would be circular. See derive_sensor_bounds.py header for
    details.
    """
    bounds = load_sensor_bounds()
    if bounds is None:
        # Fallback: no bounds derived yet → return permissive defaults
        # so the pipeline still runs, but flag everything as MEDIUM at most.
        return {
            f: {"warn": float("inf"), "crit": float("inf")} for f in SENSOR_FEATURES
        }
    out = {}
    for feat in SENSOR_FEATURES:
        b = bounds.get(feat)
        if b is None:
            out[feat] = {"warn": float("inf"), "crit": float("inf")}
        else:
            out[feat] = {"warn": b["p90"], "crit": b["p99"]}
    return out
