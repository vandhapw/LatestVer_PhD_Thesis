# ============================================================
# DERIVE SENSOR BOUNDS FROM DATA
# ============================================================
# Reads the MongoDB collection, splits machines into a TRAINING
# subset (deterministic by machine_id mod 2 == 0), computes per-
# feature percentiles (5/50/90/95/99), and writes the result to
# sensor_bounds_derived.json.
#
# Why a training subset?
#   The bounds file is consumed by the reasoning engine and by
#   normalization in every other script. If we computed bounds
#   on the full dataset and then evaluated a baseline against
#   labels derived from those same bounds, we would have circular
#   evaluation. The training subset (~25 machines) is used to
#   derive bounds and tune any thresholds; the held-out subset
#   (the other ~25 machines) is what reports of detection
#   performance MUST be computed on.
#
# Run this script ONCE before any pipeline:
#   python derive_sensor_bounds.py
# ============================================================

import json
import sys
import statistics
from pathlib import Path
from pymongo import MongoClient

from config import (
    MONGODB_URI, MONGO_DB, MONGO_COLLECTION, SENSOR_FEATURES,
)


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


def main():
    if not MONGODB_URI:
        print("[ERROR] MONGODB_URI not set in environment / .env")
        sys.exit(1)

    print(f"[derive] connecting to {MONGO_DB}.{MONGO_COLLECTION}")
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=10_000)
    col = client[MONGO_DB][MONGO_COLLECTION]

    # Identify the training split: machine_id even
    print("[derive] determining train/test split (machine_id mod 2)")
    distinct_machines = sorted(col.distinct("machine_id"))
    train_ids = [m for m in distinct_machines if m % 2 == 0]
    test_ids  = [m for m in distinct_machines if m % 2 == 1]
    print(f"[derive] train machines: {len(train_ids)} | test machines: {len(test_ids)}")
    print(f"[derive]   first 5 train: {train_ids[:5]}")
    print(f"[derive]   first 5 test:  {test_ids[:5]}")

    # Sample bounded number of readings per training machine to keep this fast
    SAMPLES_PER_MACHINE = 200
    feature_values = {f: [] for f in SENSOR_FEATURES}

    for mid in train_ids:
        cursor = col.find({"machine_id": mid}).sort("timestamp", -1).limit(SAMPLES_PER_MACHINE)
        for doc in cursor:
            for f in SENSOR_FEATURES:
                v = doc.get(f)
                if v is None:
                    # Some collections store energy as energy_consumption already; skip
                    continue
                if isinstance(v, (int, float)):
                    feature_values[f].append(float(v))

    bounds = {}
    for feat, vals in feature_values.items():
        if not vals:
            print(f"[derive] WARNING: no values for {feat} — skipping")
            continue
        vals_sorted = sorted(vals)
        bounds[feat] = {
            "n_samples": len(vals),
            "min":   round(vals_sorted[0], 3),
            "p05":   round(percentile(vals_sorted, 5), 3),
            "p50":   round(percentile(vals_sorted, 50), 3),
            "p90":   round(percentile(vals_sorted, 90), 3),
            "p95":   round(percentile(vals_sorted, 95), 3),
            "p99":   round(percentile(vals_sorted, 99), 3),
            "max":   round(vals_sorted[-1], 3),
            "mean":  round(statistics.mean(vals), 3),
            "stdev": round(statistics.pstdev(vals), 3),
        }
        print(f"[derive] {feat:<20} n={len(vals):>6} "
              f"min={bounds[feat]['min']:.2f} p90={bounds[feat]['p90']:.2f} "
              f"p99={bounds[feat]['p99']:.2f} max={bounds[feat]['max']:.2f}")

    out = {
        "_disclosure": (
            "Sensor bounds derived from a TRAINING subset of machines "
            "(machine_id even). Test machines (odd) MUST be used for "
            "any reported detection metrics. p90/p99 are used as the "
            "warn/crit thresholds — these are NOT learned end-to-end."
        ),
        "train_machine_ids": train_ids,
        "test_machine_ids":  test_ids,
        **bounds,
    }

    out_path = Path(__file__).with_name("sensor_bounds_derived.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    client.close()
    print(f"\n[derive] wrote {out_path}")


if __name__ == "__main__":
    main()
