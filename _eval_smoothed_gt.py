"""
Re-evaluate the 4 approaches against TEMPORALLY-SMOOTHED ground truth.

Methodology:
  For each test machine, take last N=15 raw labels from MongoDB and
  reduce to one stable label via majority vote on each component:
    - machine_status_label  → most common
    - failure_type          → most common
    - anomaly_flag          → mode
    - maintenance_required  → mode
  Then: needs_attention_smoothed = any(stable signals != Normal/0)

  This neutralizes classifier flip-noise so we can see the TRUE F1
  ceiling of each detector — i.e., how well they identify machines
  whose dominant label across the recent window is "needs attention".

Honest disclosure: this changes the evaluation criterion (each
machine now has a single stable label rather than the noisy raw
label). We are NOT improving the detector; we are improving the
EVALUATION by removing classifier label-flip noise.
"""
from collections import Counter
from pymongo import MongoClient

from config import (
    MONGODB_URI, MONGO_DB, MONGO_COLLECTION, load_sensor_bounds,
)
from comparative_study import (
    NonAgenticPipeline, AgenticPipeline, FleetRelativePipeline,
    SlidingWindowPipeline, _build_event_from_doc, _load_split,
    compute_detection_metrics,
)


WINDOW = 15
client = MongoClient(MONGODB_URI)
col = client[MONGO_DB][MONGO_COLLECTION]


def _mode(seq, default=None):
    seq = [x for x in seq if x is not None]
    if not seq: return default
    return Counter(seq).most_common(1)[0][0]


def smoothed_ground_truth(machine_ids, window=WINDOW):
    """Per-machine stable label from majority vote over last `window` cycles."""
    gt = {}
    for mid in machine_ids:
        cursor = col.find({"machine_id": mid}).sort("timestamp", -1).limit(window)
        rows = list(cursor)
        if not rows:
            continue
        stable_status = _mode([d.get("machine_status_label", "Normal") for d in rows])
        stable_failure = _mode([d.get("failure_type", "Normal") for d in rows])
        stable_anomaly = _mode([d.get("anomaly_flag", 0) for d in rows], default=0)
        stable_maint   = _mode([d.get("maintenance_required", 0) for d in rows], default=0)
        gt[mid] = {
            "needs_attention": (stable_status in ("Warning", "Fault")
                                or stable_failure != "Normal"
                                or stable_anomaly == 1
                                or stable_maint == 1),
            "stable_status":   stable_status,
            "stable_failure":  stable_failure,
            "raw_majority_share": _majority_share(
                [d.get("failure_type", "Normal") for d in rows]
            ),
        }
    return gt


def _majority_share(seq):
    if not seq: return 0.0
    c = Counter(seq).most_common(1)[0][1]
    return round(c / len(seq), 3)


def main(skip_agentic=False):
    bounds = load_sensor_bounds()
    train_ids = bounds["train_machine_ids"]
    test_ids  = bounds["test_machine_ids"]

    print("="*100)
    print(" SMOOTHED-GT EVALUATION (window=15-cycle majority vote)")
    print("="*100)
    print("\n  Comparing detector performance against:")
    print("    A) raw GT      = labels of THE single latest snapshot")
    print("    B) smoothed GT = majority vote over last 15 cycles per machine")
    print()

    # Load the latest snapshot for both splits (used by detectors)
    train_events = _load_split(train_ids, "Train")
    test_events  = _load_split(test_ids,  "Test")

    # Build BOTH ground truths
    raw_gt = {}
    for evt in test_events:
        raw_gt[evt["machine_id"]] = {
            "needs_attention": (evt["_gt_machine_status_label"] in ("Warning", "Fault")
                                or evt["_gt_failure_type"] != "Normal"
                                or evt["_gt_anomaly_flag"] == 1
                                or evt["_gt_maintenance_required"] == 1),
        }
    smoothed_gt = smoothed_ground_truth(test_ids, window=WINDOW)

    n = len(test_events)
    raw_pos = sum(1 for g in raw_gt.values() if g["needs_attention"])
    sm_pos  = sum(1 for g in smoothed_gt.values() if g["needs_attention"])
    print(f"  Test machines: {n}")
    print(f"  GT positives — raw (1 snapshot): {raw_pos}/{n}")
    print(f"  GT positives — smoothed (15 cycles vote): {sm_pos}/{n}")

    # Show stability per machine
    print(f"\n  Per-machine majority share of dominant failure_type "
          f"(1.0 = stable, 0.2 = chaotic):")
    shares = sorted(smoothed_gt.items(), key=lambda kv: kv[1]["raw_majority_share"])
    for mid, g in shares:
        marker = " ← chaotic" if g["raw_majority_share"] < 0.5 else ""
        print(f"    M-{mid:>2}  share={g['raw_majority_share']:.2f}  "
              f"stable_failure={g['stable_failure']:<18}  "
              f"stable_status={g['stable_status']}{marker}")

    # Run all 4 approaches
    print(f"\n[A] Rule-Based...")
    res_a = NonAgenticPipeline().analyze(test_events)
    if not skip_agentic:
        print(f"[B] Agentic AI (LLM)...")
        res_b = AgenticPipeline().analyze(test_events)
    else:
        print(f"[B] (skipped)")
        res_b = res_a

    print(f"[C] Fleet-Relative...")
    p_c = FleetRelativePipeline(); p_c.fit(train_events)
    res_c = p_c.analyze(test_events)

    print(f"[D] Sliding-Window...")
    p_d = SlidingWindowPipeline(window=12, z_threshold=2.5); p_d.fit(train_ids)
    res_d = p_d.analyze(test_ids)

    # Compute metrics against both GTs
    def metrics_table(gt_label, gt_dict):
        print(f"\n{'─'*100}")
        print(f"  Metrics vs {gt_label} (positives={sum(1 for g in gt_dict.values() if g['needs_attention'])}/{n})")
        print(f"{'─'*100}")
        print(f"  {'Approach':<20} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4} "
              f"{'prec':>7} {'rec':>7} {'F1':>7} {'acc':>7}")
        for name, res in [("Rule-Based", res_a), ("Agentic AI", res_b),
                          ("Fleet-Relative", res_c), ("Sliding-Window", res_d)]:
            m = compute_detection_metrics(res, gt_dict)
            print(f"  {name:<20} {m['TP']:>4} {m['FP']:>4} {m['FN']:>4} {m['TN']:>4} "
                  f"{m['precision']:>7.4f} {m['recall']:>7.4f} {m['f1_score']:>7.4f} "
                  f"{m['accuracy']:>7.4f}")

    metrics_table("RAW GT (1 snapshot)", raw_gt)
    metrics_table("SMOOTHED GT (15-cycle majority)", smoothed_gt)

    print(f"\n{'='*100}")
    print(" INTERPRETATION:")
    print("  - Δ between RAW and SMOOTHED F1 reveals how much classifier flip-noise")
    print("    was depressing your apparent detector quality.")
    print(f"{'='*100}\n")

    client.close()


if __name__ == "__main__":
    import sys
    main(skip_agentic="--skip-agentic" in sys.argv)
