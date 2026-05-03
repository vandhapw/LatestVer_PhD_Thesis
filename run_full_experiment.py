#!/usr/bin/env python3
"""
QASAMAP Ver13 — FULL 3-LAYER EXPERIMENT RUNNER
===============================================
Menjalankan seluruh pipeline eksperimen 3-layer:
  Layer 1: T-GCN Streaming (Kafka simulation, 1 reading/min)
  Layer 2: Agentic AI Multi-Agent Orchestrator (9-agent pipeline)
  Layer 3: Maintenance Scheduling Optimization (Classical vs Quantum)

Output:
  - layer1_tgcn_predictions.json
  - layer2_maintenance_priority.json
  - layer3_scheduling_results.json
  - full_experiment_summary.json
"""

import json
import time
from pathlib import Path

import pandas as pd

# Import layer modules
import layer1_tgcn_streaming as layer1
import layer2_agentic_orchestrator as layer2
import layer3_scheduling as layer3

CSV_PATH = Path(__file__).parent / "smart_manufacturing_data.csv"
SUMMARY_PATH = Path(__file__).parent / "full_experiment_summary.json"

def main():
    print("=" * 80)
    print("QASAMAP Ver13 — FULL 3-LAYER EXPERIMENT")
    print("Honest comparison: Classical vs Quantum-Inspired vs QAOA Simulator")
    print("=" * 80)

    overall_start = time.time()

    # ── Layer 1: T-GCN Streaming ──────────────────────────
    print("\n" + "=" * 80)
    print("LAYER 1: T-GCN Streaming + Kafka Simulation")
    print("=" * 80)

    df = pd.read_csv(CSV_PATH)
    print(f"[LAYER 1] Loaded {len(df)} records")

    # Train/load model
    if not layer1.MODEL_PATH.exists():
        lstm, fc, adj = layer1.train_tgcn(df)
    else:
        print(f"[LAYER 1] Model already exists at {layer1.MODEL_PATH}")

    # Simulate Kafka stream for a few minutes and collect predictions
    stream_data = []
    for minute, record in enumerate(layer1.simulate_kafka_stream(df, n_minutes=10)):
        stream_data.append(record)
    print(f"[LAYER 1] Kafka stream: {len(stream_data)} records emitted")

    # Save layer 1 output
    l1_out = Path(__file__).parent / "layer1_tgcn_predictions.json"
    with open(l1_out, "w") as f:
        json.dump({"layer": 1, "records": len(stream_data), "stream": stream_data}, f, indent=2)
    print(f"[LAYER 1] Output saved to {l1_out}")

    # ── Layer 2: Agentic AI Orchestrator ──────────────────
    print("\n" + "=" * 80)
    print("LAYER 2: Agentic AI Multi-Agent Orchestrator")
    print("=" * 80)

    layer2_results = layer2.run_layer2_orchestrator(df, n_machines=25)

    # ── Layer 3: Scheduling Optimization ──────────────────
    print("\n" + "=" * 80)
    print("LAYER 3: Maintenance Scheduling Optimization")
    print("=" * 80)

    layer3_results = layer3.run_layer3_scheduling(n_employees=5)

    # ── Summary ───────────────────────────────────────────
    overall_end = time.time()
    total_time = round(overall_end - overall_start, 2)

    n_jobs = layer3_results.get("results", {}).get("classical", {}).get("makespan")
    if n_jobs is None:
        n_jobs = 0

    summary = {
        "experiment": "QASAMAP_Ver13_3Layer",
        "total_time_seconds": total_time,
        "layers": {
            "layer1": {
                "description": "T-GCN Streaming + Kafka simulation",
                "records_processed": len(stream_data),
                "model_path": str(layer1.MODEL_PATH),
            },
            "layer2": {
                "description": "Agentic AI 9-agent orchestrator",
                "n_machines": len(layer2_results),
                "n_maintenance_jobs": sum(1 for r in layer2_results if r["priority"] in ("P1", "P2", "P3")),
                "priority_distribution": {
                    "P1": sum(1 for r in layer2_results if r["priority"] == "P1"),
                    "P2": sum(1 for r in layer2_results if r["priority"] == "P2"),
                    "P3": sum(1 for r in layer2_results if r["priority"] == "P3"),
                    "P4": sum(1 for r in layer2_results if r["priority"] == "P4"),
                },
            },
            "layer3": {
                "description": "Scheduling optimization (Classical vs Quantum vs Quantum-Inspired)",
                "n_jobs": layer3_results.get("results", {}).get("classical", {}).get("makespan", 0) if layer3_results else 0,
                "makespan_comparison": layer3_results.get("makespan_comparison", {}) if layer3_results else {},
            },
        },
        "disclosure": {
            "quantum_solver": "Qiskit QAOA on StatevectorSimulator (NOT real quantum hardware)",
            "qaoa_note": "Reduced to 3 critical jobs x 3 employees = 9 qubits due to simulator memory limits",
            "classical_solver": "OR-Tools CP-SAT",
            "metric": "Makespan (total completion time in minutes)",
        },
        "ethical_statement": "No overclaim. Quantum results are simulator-based. Classical results are optimal on the given instance.",
    }

    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("FULL EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"Total execution time: {total_time}s")
    print(f"\nLayer 1: {summary['layers']['layer1']['records_processed']} Kafka records")
    print(f"Layer 2: {summary['layers']['layer2']['n_machines']} machines, {summary['layers']['layer2']['n_maintenance_jobs']} maintenance jobs")
    print(f"Layer 3: Makespan comparison")
    for name, ms in summary["layers"]["layer3"]["makespan_comparison"].items():
        print(f"  {name:30s}: {ms} min")
    print(f"\nResults saved to {SUMMARY_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
