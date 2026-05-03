#!/usr/bin/env python3
"""
LAYER 2 — Agentic AI Multi-Agent Orchestrator
=============================================
Menerima streaming real-time data (simulasi dari Layer 1 T-GCN/Kafka)
dan menjalankan 9-agent pipeline untuk menghasilkan
DAFTAR PRIORITAS MAINTENANCE yang valid.

Input:  sensor_window per machine (dari T-GCN forecast atau Kafka stream)
Output: per-machine decision (P1 immediate_shutdown, P2 inspect,
        P3 schedule_maintenance, P4 monitor)

Constraint dari supervisor:
- Agentic AI output = PRIORITY ONLY (tidak estimasi durasi)
- Tidak perlu latih ulang (gunakan model/bounds yang sudah ada)
- Hindari overclaim dan pelanggaran etika penelitian
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np

CSV_PATH = Path(__file__).parent / "smart_manufacturing_data.csv"
BOUNDS_PATH = Path(__file__).parent / "sensor_bounds_derived.json"
SENSOR_FEATURES = ["temperature", "vibration", "humidity", "pressure", "energy_consumption"]

# ─────────────────────────────────────────────
# 1. LOAD DERIVED BOUNDS (dari train split)
# ─────────────────────────────────────────────

def load_bounds():
    if BOUNDS_PATH.exists():
        with open(BOUNDS_PATH) as f:
            return json.load(f)
    # Derive on the fly if missing
    df = pd.read_csv(CSV_PATH)
    train_df = df[df["machine_id"] % 2 == 0].copy()
    bounds = {}
    for feat in SENSOR_FEATURES:
        vals = sorted(train_df[feat].dropna().tolist())
        bounds[feat] = {
            "p05": vals[int(len(vals)*0.05)],
            "p50": vals[int(len(vals)*0.50)],
            "p90": vals[int(len(vals)*0.90)],
            "p99": vals[int(len(vals)*0.99)],
        }
    with open(BOUNDS_PATH, "w") as f:
        json.dump(bounds, f, indent=2)
    return bounds


def percentile(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    idx = (len(sorted_vals) - 1) * p / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return float(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac)


# ─────────────────────────────────────────────
# 2. CLASSICAL RISK SCORING
# ─────────────────────────────────────────────

def compute_classical_risk(row, bounds):
    risk = 0.0
    for feat in SENSOR_FEATURES:
        val = row[feat]
        p90 = bounds[feat]["p90"]
        p99 = bounds[feat]["p99"]
        if val >= p99:
            risk += 0.35
        elif val >= p90:
            risk += 0.15
    rul = row.get("predicted_remaining_life", 250)
    if rul < 50:
        risk += 0.30
    elif rul < 100:
        risk += 0.10
    if row.get("anomaly_flag", 0) == 1:
        risk += 0.20
    if row.get("downtime_risk", 0.0) > 0.6:
        risk += 0.15
    return min(risk, 1.0)


def classify_risk(risk_score):
    if risk_score >= 0.75:
        return "CRITICAL"
    elif risk_score >= 0.60:
        return "HIGH"
    elif risk_score >= 0.30:
        return "MEDIUM"
    else:
        return "LOW"


# ─────────────────────────────────────────────
# 3. QUANTUM-INSPIRED HEURISTIC LAYER
# ─────────────────────────────────────────────

def compute_heuristic(row, bounds):
    per_sensor = {}
    for feat in SENSOR_FEATURES:
        val = row[feat]
        lo = bounds[feat]["p05"]
        hi = bounds[feat]["p99"]
        norm = (val - lo) / (hi - lo) if hi != lo else 0.5
        norm = max(0.0, min(1.0, norm))
        dist = abs(norm - 0.5) * 2
        per_sensor[feat] = dist

    weights = [0.30, 0.30, 0.15, 0.15, 0.10]
    avg_dist = sum(w * per_sensor[f] for w, f in zip(weights, SENSOR_FEATURES))
    score = 1.0 - math.exp(-3 * avg_dist)

    if score < -0.20:
        label = "QUANTUM-ANOMALY"
    elif score < 0.30:
        label = "QUANTUM-WARN"
    elif score < 0.60:
        label = "QUANTUM-MARGINAL"
    else:
        label = "QUANTUM-NORMAL"

    return score, label, per_sensor


# ─────────────────────────────────────────────
# 4. 9-AGENT DECISION PIPELINE
# ─────────────────────────────────────────────

def run_nine_agent_pipeline(machine_id, row, bounds):
    """
    Simulasi 9-agent sequential pipeline.
    Output: priority only (P1-P4) + action
    """
    classical_risk = compute_classical_risk(row, bounds)
    risk_class = classify_risk(classical_risk)
    h_score, h_label, per_sensor = compute_heuristic(row, bounds)

    # Agent 3: Planning (base decision)
    if classical_risk >= 0.75 or h_label == "QUANTUM-ANOMALY":
        action, priority = "immediate_shutdown", "P1"
    elif classical_risk >= 0.60 or h_label == "QUANTUM-WARN":
        action, priority = "inspect", "P2"
    elif classical_risk >= 0.30:
        action, priority = "schedule_maintenance", "P3"
    else:
        action, priority = "monitor", "P4"

    # Agent 7: Self-Reflection — override check
    override = False
    if h_label in ("QUANTUM-ANOMALY", "QUANTUM-WARN") and risk_class == "LOW":
        override = True
        action, priority = "inspect", "P2"

    return {
        "machine_id": int(machine_id),
        "classical_risk": round(classical_risk, 4),
        "risk_class": risk_class,
        "heuristic_score": round(h_score, 4),
        "heuristic_label": h_label,
        "action": action,
        "priority": priority,
        "override": override,
        "per_sensor_dist": {k: round(v, 4) for k, v in per_sensor.items()},
    }


# ─────────────────────────────────────────────
# 5. ORCHESTRATOR — Process all test machines
# ─────────────────────────────────────────────

def run_layer2_orchestrator(df=None, n_machines=25):
    """
    Jalankan Layer 2 pada test machines.
    Return: list of machine decisions dengan priority valid.
    """
    if df is None:
        df = pd.read_csv(CSV_PATH)

    bounds = load_bounds()

    # Test split: machine_id ganjil (sesuai convention sebelumnya)
    test_df = df[df["machine_id"] % 2 == 1].copy()
    test_machines = sorted(test_df["machine_id"].unique())[:n_machines]

    print(f"[LAYER 2] Processing {len(test_machines)} test machines: {test_machines}")

    results = []
    for mid in test_machines:
        machine_df = test_df[test_df["machine_id"] == mid]
        latest = machine_df.iloc[-1]
        result = run_nine_agent_pipeline(mid, latest, bounds)
        results.append(result)
        print(f"  Machine {mid}: risk={result['classical_risk']} ({result['risk_class']}), "
              f"heuristic={result['heuristic_label']}, action={result['action']}, priority={result['priority']}")
        if result["override"]:
            print(f"    ⚡ OVERRIDE: heuristic escalation triggered")

    # Summary
    overrides = sum(1 for r in results if r["override"])
    actions = defaultdict(int)
    priorities = defaultdict(int)
    for r in results:
        actions[r["action"]] += 1
        priorities[r["priority"]] += 1

    print(f"\n[LAYER 2] Summary:")
    print(f"  Overrides: {overrides}/{len(results)}")
    print(f"  Actions: {dict(actions)}")
    print(f"  Priorities: {dict(priorities)}")

    # Save output untuk Layer 3
    out_path = Path(__file__).parent / "layer2_maintenance_priority.json"
    with open(out_path, "w") as f:
        json.dump({
            "layer": 2,
            "n_machines": len(results),
            "overrides": overrides,
            "actions": dict(actions),
            "priorities": dict(priorities),
            "machines": results,
        }, f, indent=2)
    print(f"\n[LAYER 2] Output saved to {out_path}")

    return results


if __name__ == "__main__":
    run_layer2_orchestrator()
