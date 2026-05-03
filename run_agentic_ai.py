#!/usr/bin/env python3
"""
AGENTIC AI — MULTI-AGENT SYSTEM EXECUTION SCRIPT
================================================
Menjalankan pipeline agentic AI untuk deteksi anomali dan predictive maintenance
dengan membaca CSV secara langsung tanpa MongoDB.

Pipeline:
  1. Data Ingestion (CSV)
  2. Sensor Bounds Derivation
  3. Reasoning Engine (Classical Risk Scoring)
  4. Multi-Agent Orchestrator (9 LLM Agents via Ollama)
  5. Heuristic Layer (Quantum-Inspired Classical)
  6. Decision & Reporting

Output:
  - Per-machine risk assessment
  - Agent decisions (monitor/inspect/schedule_maintenance)
  - Heuristic override triggers
  - Comparative analysis
"""

import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
CSV_PATH = Path(__file__).parent / "smart_manufacturing_data.csv"
SENSOR_FEATURES = ["temperature", "vibration", "humidity", "pressure", "energy_consumption"]
RUL_LOW = 50
RUL_MODERATE = 100
RISK_HIGH_AT = 0.6
RISK_MEDIUM_AT = 0.3

# ─────────────────────────────────────────────
# 1. DATA INGESTION
# ─────────────────────────────────────────────

def load_data(csv_path):
    """Load manufacturing data from CSV."""
    df = pd.read_csv(csv_path)
    print(f"[DATA] Loaded {len(df)} records from {csv_path.name}")
    print(f"[DATA] Machines: {df['machine_id'].nunique()}")
    print(f"[DATA] Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    return df


def split_train_test(df):
    """Split by machine_id parity (even=train, odd=test) per derive_sensor_bounds.py."""
    train_df = df[df["machine_id"] % 2 == 0].copy()
    test_df = df[df["machine_id"] % 2 == 1].copy()
    train_ids = sorted(train_df["machine_id"].unique())
    test_ids = sorted(test_df["machine_id"].unique())
    print(f"[SPLIT] Train machines ({len(train_ids)}): {train_ids[:5]}...")
    print(f"[SPLIT] Test machines  ({len(test_ids)}): {test_ids[:5]}...")
    return train_df, test_df, train_ids, test_ids


# ─────────────────────────────────────────────
# 2. SENSOR BOUNDS DERIVATION
# ─────────────────────────────────────────────

def percentile(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    if p <= 0:
        return float(sorted_vals[0])
    if p >= 100:
        return float(sorted_vals[-1])
    idx = (len(sorted_vals) - 1) * p / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return float(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac)


def derive_sensor_bounds(train_df):
    """Derive sensor bounds from training split."""
    bounds = {}
    for feat in SENSOR_FEATURES:
        vals = sorted(train_df[feat].dropna().tolist())
        bounds[feat] = {
            "min": min(vals),
            "max": max(vals),
            "p05": percentile(vals, 5),
            "p50": percentile(vals, 50),
            "p90": percentile(vals, 90),
            "p95": percentile(vals, 95),
            "p99": percentile(vals, 99),
        }
    bounds["train_machine_ids"] = sorted(train_df["machine_id"].unique().tolist())
    bounds["test_machine_ids"] = sorted(train_df["machine_id"].unique().tolist())

    # Save to JSON
    out_path = Path(__file__).parent / "sensor_bounds_derived.json"
    with open(out_path, "w") as f:
        json.dump(bounds, f, indent=2)
    print(f"[BOUNDS] Saved sensor bounds to {out_path}")
    return bounds


# ─────────────────────────────────────────────
# 3. REASONING ENGINE (Classical Risk Scoring)
# ─────────────────────────────────────────────

def get_warn_crit_thresholds(bounds):
    """Return reasoning-engine warn/crit thresholds from bounds."""
    out = {}
    for feat in SENSOR_FEATURES:
        b = bounds.get(feat, {})
        out[feat] = {
            "warn": b.get("p90", float("inf")),
            "crit": b.get("p99", float("inf")),
        }
    return out


def compute_risk_score(row, thresholds):
    """Compute classical risk score from sensor readings."""
    risk = 0.0

    # Sensor threshold crossings
    for feat in SENSOR_FEATURES:
        val = row[feat]
        th = thresholds.get(feat, {})
        if val >= th.get("crit", float("inf")):
            risk += 0.35
        elif val >= th.get("warn", float("inf")):
            risk += 0.15

    # RUL penalty
    rul = row.get("predicted_remaining_life", 250)
    if rul < RUL_LOW:
        risk += 0.30
    elif rul < RUL_MODERATE:
        risk += 0.10

    # Anomaly flag
    if row.get("anomaly_flag", 0) == 1:
        risk += 0.20

    # Downtime risk
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
# 4. QUANTUM-INSPIRED HEURISTIC LAYER
# ─────────────────────────────────────────────

def energy_distance_anomaly_score(row, bounds):
    """
    Classical heuristic mirroring Pauli-Z multi-sensor encoding.
    Computes normalized L2 distance from median per feature.
    """
    dists = []
    for feat in SENSOR_FEATURES:
        val = row[feat]
        b = bounds.get(feat, {})
        lo = b.get("p05", 0)
        hi = b.get("p95", 100)
        if hi == lo:
            norm = 0.5
        else:
            norm = (val - lo) / (hi - lo)
        norm = max(0.0, min(1.0, norm))
        # Distance from "safe" zone (norm ~ 0.5 is neutral)
        dist = abs(norm - 0.5) * 2  # 0=safe, 1=extreme
        dists.append(dist)

    # Weighted composite (temperature & vibration weighted higher)
    weights = [0.30, 0.30, 0.15, 0.15, 0.10]
    avg_dist = sum(w * d for w, d in zip(weights, dists))

    # Map to quantum-like labels
    score = 1.0 - math.exp(-3 * avg_dist)

    if score < -0.20:
        label = "QUANTUM-ANOMALY"
    elif score < 0.30:
        label = "QUANTUM-WARN"
    elif score < 0.60:
        label = "QUANTUM-MARGINAL"
    else:
        label = "QUANTUM-NORMAL"

    return {
        "score": score,
        "label": label,
        "per_sensor_dists": dict(zip(SENSOR_FEATURES, dists)),
    }


def random_constrained_search(machine_scores, n_samples=200, max_simultaneous=5):
    """
    Classical heuristic mirroring QUBO priority ranking.
    Greedy seed + random 1-bit-flip local search.
    """
    import random
    random.seed(42)

    machines = sorted(machine_scores.keys(), key=lambda m: machine_scores[m], reverse=True)
    best_energy = float("inf")
    best_selection = set()

    for _ in range(n_samples):
        # Greedy seed: select top machines up to max_simultaneous
        selection = set(machines[:random.randint(1, max_simultaneous)])
        # Random flips
        for _ in range(random.randint(0, 3)):
            if random.random() < 0.5 and len(selection) > 0:
                selection.remove(random.choice(list(selection)))
            elif len(selection) < max_simultaneous:
                remaining = [m for m in machines if m not in selection]
                if remaining:
                    selection.add(random.choice(remaining))

        # Compute energy (simplified QUBO)
        energy = 0.0
        for m in selection:
            energy -= machine_scores[m]  # reward high risk
        energy += 0.75 * len(selection)  # penalty for over-selection

        if energy < best_energy:
            best_energy = energy
            best_selection = selection

    return {
        "selected": sorted(best_selection),
        "energy": best_energy,
        "ranking": machines,
    }


# ─────────────────────────────────────────────
# 5. MULTI-AGENT ORCHESTRATOR (Simulated)
# ─────────────────────────────────────────────

def simulate_agent_decision(machine_id, classical_risk, risk_class, heuristic_result):
    """
    Simulate the 9-agent LLM pipeline decision.
    In production this would call Ollama API; here we simulate the logic.
    """
    decisions = []

    # Agent 1: Monitoring
    if risk_class in ("HIGH", "CRITICAL"):
        decisions.append("monitoring: HIGH risk detected")

    # Agent 2: Diagnosis
    if heuristic_result["label"] in ("QUANTUM-ANOMALY", "QUANTUM-WARN"):
        top_sensor = max(heuristic_result["per_sensor_dists"], key=heuristic_result["per_sensor_dists"].get)
        decisions.append(f"diagnosis: abnormal {top_sensor} pattern")

    # Agent 3: Planning
    if risk_class == "CRITICAL" or heuristic_result["label"] == "QUANTUM-ANOMALY":
        action = "immediate_shutdown"
        priority = "P1"
    elif risk_class == "HIGH" or heuristic_result["label"] == "QUANTUM-WARN":
        action = "inspect"
        priority = "P2"
    elif risk_class == "MEDIUM":
        action = "schedule_maintenance"
        priority = "P3"
    else:
        action = "monitor"
        priority = "P4"

    # Agent 7: Self-Reflection — override check
    override = False
    if heuristic_result["label"] in ("QUANTUM-ANOMALY", "QUANTUM-WARN") and risk_class == "LOW":
        override = True
        action = "inspect"
        priority = "P2"
        decisions.append("override: heuristic escalation triggered")

    return {
        "machine_id": machine_id,
        "classical_risk": classical_risk,
        "risk_class": risk_class,
        "heuristic_score": heuristic_result["score"],
        "heuristic_label": heuristic_result["label"],
        "action": action,
        "priority": priority,
        "override": override,
        "agent_reasoning": decisions,
    }


# ─────────────────────────────────────────────
# 6. MAIN PIPELINE
# ─────────────────────────────────────────────

def run_agentic_pipeline():
    print("=" * 80)
    print("AGENTIC AI — MULTI-AGENT SYSTEM EXECUTION")
    print("QASAMAP Ver13 — Classical Heuristic Implementation")
    print("=" * 80)

    # Step 1: Load data
    df = load_data(CSV_PATH)
    train_df, test_df, train_ids, test_ids = split_train_test(df)

    # Step 2: Derive bounds
    bounds = derive_sensor_bounds(train_df)
    thresholds = get_warn_crit_thresholds(bounds)

    # Step 3: Process test machines (sample: 7 machines like in thesis)
    test_machines = test_df["machine_id"].unique()[:7]
    print(f"\n[PIPELINE] Processing {len(test_machines)} test machines: {list(test_machines)}")

    results = []
    machine_scores = {}

    for mid in test_machines:
        machine_df = test_df[test_df["machine_id"] == mid]
        # Take latest event for the machine
        latest = machine_df.iloc[-1]

        # Classical reasoning
        risk = compute_risk_score(latest, thresholds)
        risk_class = classify_risk(risk)

        # Heuristic layer
        heuristic = energy_distance_anomaly_score(latest, bounds)

        # Agentic decision
        decision = simulate_agent_decision(mid, risk, risk_class, heuristic)
        results.append(decision)
        machine_scores[mid] = heuristic["score"]

        print(f"\n  Machine {mid}:")
        print(f"    Classical Risk: {risk:.4f} ({risk_class})")
        print(f"    Heuristic Score: {heuristic['score']:.4f} ({heuristic['label']})")
        print(f"    Action: {decision['action']} | Priority: {decision['priority']}")
        if decision['override']:
            print(f"    *** OVERRIDE TRIGGERED ***")

    # Step 4: QUBO Priority Ranking
    print("\n" + "=" * 80)
    print("QUBO PRIORITY RANKING")
    print("=" * 80)
    qubo = random_constrained_search(machine_scores)
    print(f"Selected machines: {qubo['selected']}")
    print(f"QUBO energy: {qubo['energy']:.4f}")
    print(f"Full ranking: {qubo['ranking']}")

    # Step 5: Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    overrides = sum(1 for r in results if r["override"])
    actions = defaultdict(int)
    for r in results:
        actions[r["action"]] += 1

    print(f"Total machines processed: {len(results)}")
    print(f"Heuristic overrides: {overrides} ({overrides/len(results)*100:.1f}%)")
    print(f"Actions taken:")
    for action, count in sorted(actions.items()):
        print(f"  {action}: {count}")

    # Save results
    out_path = Path(__file__).parent / "agentic_execution_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": str(pd.Timestamp.now()),
            "n_machines": len(results),
            "overrides": overrides,
            "actions": dict(actions),
            "qubo": qubo,
            "per_machine": results,
        }, f, indent=2, default=str)
    print(f"\n[SAVE] Results written to {out_path}")

    return results


if __name__ == "__main__":
    run_agentic_pipeline()
