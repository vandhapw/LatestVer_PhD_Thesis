# ============================================================
# SLIDING-WINDOW PER-MACHINE ANOMALY DETECTOR
# ============================================================
# Complements fleet_relative_detector.py (which scores cross-machine)
# with per-machine TEMPORAL scoring.
#
# Architecture: reuses the multi-tier buffer concept already proven in
# kafka_realtime_consumer.MultiTierBuffer (Tier 1=15, Tier 2=120 etc.)
# but operates BATCH-style by pulling the latest N records per machine
# from MongoDB. Honest: this is NOT a streaming detector (no Kafka
# subscription); it samples whatever the producer has written so far
# and applies the same z-score logic the real consumer would.
#
# Why this exists now (and not before the producer patch):
#   The producer used to emit STATIC values per machine across cycles.
#   Per-machine z-score was always 0 for every feature → useless. After
#   `_remote_kafka_producer_patched.py` removed `np.random.seed(m)` and
#   the SensorAligner noise floor, per-machine variance is non-zero, so
#   temporal anomaly detection is suddenly viable for non-anchor
#   machines. Anchor (M-40) also varies after the SensorAligner fix.
#
# Two scoring modes:
#   1. point_zscore  — for each feature, z = (current - mean) / std
#                      computed on the first WINDOW-1 readings (history),
#                      then evaluated against the latest reading.
#                      A machine is flagged if |z| > threshold for ANY
#                      feature. Mirrors kafka_realtime_consumer's
#                      SpikeDetector (z=3.0 default).
#   2. cumulative    — number of features that crossed the z threshold
#                      in the entire window, normalized to [0, 1].
#                      Robust to single-feature noise spikes.
#
# Decision threshold is derived FROM TRAIN-FLEET history (no peeking
# at the test split) — see derive_window_threshold below.
# ============================================================

import math
import statistics
from typing import List, Dict

from config import (
    MONGODB_URI, MONGO_DB, MONGO_COLLECTION, SENSOR_FEATURES,
)


# ─────────────────────────────────────────────
# Per-machine score computation
# ─────────────────────────────────────────────

def _split_history_and_current(events_chrono):
    """Split a chronological list into (history, current_event)."""
    if len(events_chrono) < 2:
        return events_chrono, None
    return events_chrono[:-1], events_chrono[-1]


def point_zscore(events_chrono, *, z_clip=10.0):
    """
    Returns dict per feature: {value, mean, std, z, abs_z}.
    Empty std → z=0 (still nothing to detect).
    """
    history, current = _split_history_and_current(events_chrono)
    if current is None:
        return {}
    out = {}
    for feat in SENSOR_FEATURES:
        # Some events use "energy" instead of "energy_consumption"
        history_vals = [
            e.get(feat, e.get("energy" if feat == "energy_consumption" else feat, 0))
            for e in history
        ]
        history_vals = [v for v in history_vals if isinstance(v, (int, float))]
        cur_val = current.get(feat, current.get("energy" if feat == "energy_consumption" else feat, 0))
        if not isinstance(cur_val, (int, float)) or len(history_vals) < 3:
            continue
        m = statistics.mean(history_vals)
        s = statistics.pstdev(history_vals) if len(history_vals) > 1 else 0.0
        if s <= 1e-9:
            z = 0.0
        else:
            z = (cur_val - m) / s
            z = max(-z_clip, min(z_clip, z))
        out[feat] = {
            "value":   round(cur_val, 4),
            "mean":    round(m, 4),
            "std":     round(s, 4),
            "z":       round(z, 3),
            "abs_z":   round(abs(z), 3),
        }
    return out


def cumulative_z_excess(events_chrono, *, threshold=2.0):
    """
    Across the whole window, count how many feature-readings have
    |z|>threshold (where z is computed against an expanding-window
    mean/std excluding the current point). Returns ratio in [0, 1].
    """
    if len(events_chrono) < 4:
        return 0.0
    n = len(events_chrono)
    n_features = len(SENSOR_FEATURES)
    excess_count = 0
    total_count  = 0
    for i in range(3, n):
        prefix = events_chrono[:i]
        cur = events_chrono[i]
        z_map = point_zscore(prefix + [cur])
        for feat, info in z_map.items():
            total_count += 1
            if info["abs_z"] > threshold:
                excess_count += 1
    return excess_count / max(total_count, 1)


# ─────────────────────────────────────────────
# Per-machine evaluator
# ─────────────────────────────────────────────

def detect_one_machine(events_chrono, *, z_threshold=2.5):
    """
    Combine point_zscore on the latest reading + cumulative_z_excess
    over the whole window. Returns a dict with both signals and a
    composite_score in [0, 1].
    """
    z_map = point_zscore(events_chrono)
    if not z_map:
        return {
            "n_history":   max(len(events_chrono) - 1, 0),
            "max_abs_z":   0.0,
            "max_z_feature": None,
            "z_per_feature": {},
            "cumulative_excess_ratio": 0.0,
            "composite_score": 0.0,
            "_disclosure": "insufficient history (<3 prior points)",
        }
    max_feat, max_info = max(z_map.items(), key=lambda kv: kv[1]["abs_z"])
    cum = cumulative_z_excess(events_chrono, threshold=z_threshold * 0.8)
    # Combine: clip max_abs_z to [0, 4] and scale to [0, 1]
    point_signal = min(max_info["abs_z"] / 4.0, 1.0)
    composite = 0.7 * point_signal + 0.3 * cum
    return {
        "n_history":   len(events_chrono) - 1,
        "max_abs_z":   max_info["abs_z"],
        "max_z_feature": max_feat,
        "z_per_feature": z_map,
        "cumulative_excess_ratio": round(cum, 4),
        "composite_score": round(composite, 4),
    }


# ─────────────────────────────────────────────
# Threshold derivation from training fleet
# ─────────────────────────────────────────────

def derive_window_threshold(history_per_machine_train, *,
                            target_quantile=0.5, z_threshold=2.5):
    """
    Compute per-machine composite scores on the TRAINING fleet, return
    the cutoff at target_quantile. Apply this cutoff to the TEST fleet
    via SlidingWindowDetector(score_threshold=...).
    """
    scores = []
    for mid, events_chrono in history_per_machine_train.items():
        r = detect_one_machine(events_chrono, z_threshold=z_threshold)
        scores.append(r["composite_score"])
    if not scores:
        return 1.0
    scores.sort()
    idx = min(int(len(scores) * target_quantile), len(scores) - 1)
    return float(scores[idx])


# ─────────────────────────────────────────────
# Top-level detector class (for plug-in into comparative_study)
# ─────────────────────────────────────────────

class SlidingWindowDetector:
    """
    Per-machine temporal anomaly detector. Pulls latest WINDOW
    snapshots per machine from MongoDB; computes z-score against the
    machine's own recent history.

    Plug-in pattern matches FleetRelativePipeline:
        det = SlidingWindowDetector(window=15)
        det.fit(train_history_per_machine)
        results = det.analyze(test_history_per_machine)
    """
    def __init__(self, window=15, z_threshold=2.5):
        self.window      = window
        self.z_threshold = z_threshold
        self.score_threshold = None

    def fit(self, train_history_per_machine):
        self.score_threshold = derive_window_threshold(
            train_history_per_machine,
            target_quantile=0.5,
            z_threshold=self.z_threshold,
        )
        return self

    def analyze(self, test_history_per_machine, current_snapshots=None):
        """
        test_history_per_machine: dict {machine_id: [oldest, ..., newest]}
        current_snapshots: optional dict {machine_id: event} — the
            event used as "current" (defaults to last in the chrono list).

        Returns: dict {machine_id: {composite_score, is_attention_needed,
                                    z_per_feature, ...}}
        """
        if self.score_threshold is None:
            raise RuntimeError("Call fit() with training history first.")
        results = {}
        for mid, chrono in test_history_per_machine.items():
            if current_snapshots and mid in current_snapshots:
                chrono = chrono + [current_snapshots[mid]]
            r = detect_one_machine(chrono, z_threshold=self.z_threshold)
            r["is_attention_needed"] = r["composite_score"] >= self.score_threshold
            r["_decision_cutoff"]    = round(self.score_threshold, 4)
            results[mid] = r
        return results


# ─────────────────────────────────────────────
# History loader (MongoDB → per-machine chronological lists)
# ─────────────────────────────────────────────

def load_history_per_machine(machine_ids, window=15):
    """
    Load latest `window` snapshots per machine from MongoDB,
    returned in chronological order (oldest first).
    """
    from pymongo import MongoClient
    client = MongoClient(MONGODB_URI)
    col = client[MONGO_DB][MONGO_COLLECTION]
    out = {}
    for mid in machine_ids:
        cursor = col.find({"machine_id": mid}).sort("timestamp", -1).limit(window)
        rows = list(cursor)
        rows.reverse()  # oldest first
        if rows:
            out[mid] = rows
    client.close()
    return out


# ─────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    from config import load_sensor_bounds

    bounds = load_sensor_bounds()
    train_ids = bounds["train_machine_ids"]
    test_ids  = bounds["test_machine_ids"]
    WINDOW = 12  # enough for ~10 history + 1 current

    print(f"\n{'='*80}")
    print(f"  SLIDING-WINDOW DETECTOR — self-test")
    print(f"  window={WINDOW}, z_threshold=2.5")
    print(f"{'='*80}\n")

    print("[1/3] Loading TRAIN history...")
    train_hist = load_history_per_machine(train_ids, window=WINDOW)
    print(f"  loaded for {len(train_hist)} train machines")

    print("[2/3] Loading TEST history...")
    test_hist = load_history_per_machine(test_ids, window=WINDOW)
    print(f"  loaded for {len(test_hist)} test machines")

    print("[3/3] Fitting detector and scoring test split...")
    det = SlidingWindowDetector(window=WINDOW, z_threshold=2.5)
    det.fit(train_hist)
    print(f"  derived score_threshold from train fleet: {det.score_threshold:.4f}")

    results = det.analyze(test_hist)

    # Build ground truth from latest snapshot's labels
    gt = {}
    for mid, chrono in test_hist.items():
        last = chrono[-1]
        gt[mid] = (last.get("machine_status_label") in ("Warning", "Fault")
                   or last.get("failure_type", "Normal") != "Normal"
                   or last.get("anomaly_flag", 0) == 1
                   or last.get("maintenance_required", 0) == 1)

    print(f"\n  {'M':>4} {'n_hist':>6} {'maxZ':>6} {'maxZfeat':>20} {'cumX':>5} "
          f"{'composite':>9} {'flagged':>7}  ground_truth")
    print(f"  {'-'*82}")
    for mid in sorted(results):
        r = results[mid]
        gt_label = ""
        last = test_hist[mid][-1]
        if gt[mid]:
            gt_label = f"  GT+({last.get('failure_type','?')})"
        print(f"  M-{mid:>2} {r['n_history']:>6} {r['max_abs_z']:>6.2f} "
              f"{(r.get('max_z_feature') or '-'):>20} "
              f"{r['cumulative_excess_ratio']:>5.2f} "
              f"{r['composite_score']:>9.3f} {str(r['is_attention_needed']):>7}{gt_label}")

    tp = fp = fn = tn = 0
    for mid, r in results.items():
        pred, actual = r["is_attention_needed"], gt[mid]
        if pred and actual: tp += 1
        elif pred and not actual: fp += 1
        elif not pred and actual: fn += 1
        else: tn += 1
    total = max(tp + fp + fn + tn, 1)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)
    print(f"\n  Detection vs digital-twin GT (test split):")
    print(f"    TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"    precision={prec:.4f}  recall={rec:.4f}  f1={f1:.4f}  "
          f"accuracy={(tp+tn)/total:.4f}")
