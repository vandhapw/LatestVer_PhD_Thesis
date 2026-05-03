"""
Diagnose why F1=0 on the post-audit test split.
Hypotheses to test:
  H1. Distribution shift: train vs test sensor distributions differ
  H2. Label-snapshot mismatch: GT labels are set from trajectory/sliding-window
      patterns upstream, but we evaluate single snapshots
  H3. Per-machine baselines vary widely; fleet-wide P90 is a poor threshold
"""
import json
import statistics
from pymongo import MongoClient
from config import (
    MONGODB_URI, MONGO_DB, MONGO_COLLECTION, SENSOR_FEATURES,
    load_sensor_bounds, get_warn_crit_thresholds,
)

bounds = load_sensor_bounds()
test_ids = bounds["test_machine_ids"]
train_ids = bounds["train_machine_ids"]
thresholds = get_warn_crit_thresholds()

client = MongoClient(MONGODB_URI)
col = client[MONGO_DB][MONGO_COLLECTION]

# ── 1. Load latest snapshot per test machine ──
test_snapshots = {}
for mid in test_ids:
    doc = col.find_one({"machine_id": mid}, sort=[("timestamp", -1)])
    if doc:
        test_snapshots[mid] = doc

print("="*80)
print(" DIAGNOSIS: WHY F1=0 ON POST-AUDIT TEST SPLIT")
print("="*80)

# ── 2. For each test machine flagged as needs_attention, show: ──
#       latest sensor readings vs the warn/crit thresholds
print(f"\n{'─'*80}")
print(" GROUND-TRUTH POSITIVES (test machines that NEED attention per Digital Twin)")
print(f"{'─'*80}")

n_pos = 0
for mid, doc in test_snapshots.items():
    needs = (doc.get("machine_status_label") in ("Warning", "Fault")
             or doc.get("failure_type", "Normal") != "Normal"
             or doc.get("anomaly_flag", 0) == 1
             or doc.get("maintenance_required", 0) == 1)
    if needs:
        n_pos += 1
        # Compare each sensor against threshold
        crosses = []
        for feat, thr in thresholds.items():
            v = doc.get(feat, 0)
            if isinstance(v, (int, float)) and v >= thr["warn"]:
                crosses.append(f"{feat}={v:.2f}>={thr['warn']:.2f}")
        rul = doc.get("predicted_remaining_life", 200)
        rul_low = "RUL_LOW" if rul < 50 else ""
        print(f"  M-{mid:>3} | status={doc.get('machine_status_label','?'):<8} "
              f"failure={doc.get('failure_type','?'):<18} "
              f"anomaly_flag={doc.get('anomaly_flag',0)} "
              f"maint={doc.get('maintenance_required',0)} | "
              f"temp={doc.get('temperature',0):.1f} vib={doc.get('vibration',0):.1f} "
              f"rul={rul:.0f}")
        if crosses or rul_low:
            print(f"        → DETECTABLE by threshold: {crosses + ([rul_low] if rul_low else [])}")
        else:
            print(f"        → UNDETECTABLE by current thresholds (sensor-only snapshot looks normal)")
print(f"\n  Total ground-truth positives in test split: {n_pos}")

# ── 3. Sensor distribution: train vs test (latest snapshots) ──
print(f"\n{'─'*80}")
print(" H1: TRAIN vs TEST DISTRIBUTION SHIFT (latest snapshots)")
print(f"{'─'*80}")

def sensor_stats(machine_ids):
    out = {f: [] for f in SENSOR_FEATURES}
    for mid in machine_ids:
        doc = col.find_one({"machine_id": mid}, sort=[("timestamp", -1)])
        if doc:
            for f in SENSOR_FEATURES:
                v = doc.get(f, 0)
                if isinstance(v, (int, float)):
                    out[f].append(v)
    return out

train_stats = sensor_stats(train_ids)
test_stats  = sensor_stats(test_ids)

print(f"  {'feature':<22} {'train n':>8} {'train mean':>12} {'train p95':>11} "
      f"{'test n':>8} {'test mean':>11} {'test p95':>10}")
for f in SENSOR_FEATURES:
    tr = sorted(train_stats[f])
    te = sorted(test_stats[f])
    if tr and te:
        tr_mean = statistics.mean(tr)
        tr_p95  = tr[int(len(tr)*0.95)] if tr else 0
        te_mean = statistics.mean(te)
        te_p95  = te[int(len(te)*0.95)] if te else 0
        print(f"  {f:<22} {len(tr):>8} {tr_mean:>12.2f} {tr_p95:>11.2f} "
              f"{len(te):>8} {te_mean:>11.2f} {te_p95:>10.2f}")

# ── 4. H2: How are ground-truth labels actually set? ──
#       Pull all readings of one ground-truth-positive machine and look
#       at the sensor trajectory around the time the label was set.
print(f"\n{'─'*80}")
print(" H2: LABEL-SNAPSHOT MISMATCH — trajectory of a labelled machine")
print(f"{'─'*80}")

flagged = [mid for mid, doc in test_snapshots.items()
           if doc.get("failure_type", "Normal") != "Normal"
           or doc.get("anomaly_flag", 0) == 1]
if flagged:
    sample_mid = flagged[0]
    print(f"  Trajectory for M-{sample_mid} (last 10 readings):")
    cursor = col.find({"machine_id": sample_mid}).sort("timestamp", -1).limit(10)
    rows = list(cursor)
    rows.reverse()
    for r in rows:
        ts = r.get("timestamp")
        ts_str = ts.strftime("%Y-%m-%d %H:%M") if hasattr(ts, "strftime") else str(ts)
        print(f"    {ts_str} | temp={r.get('temperature',0):>6.2f} "
              f"vib={r.get('vibration',0):>6.2f} energy={r.get('energy_consumption',0):>5.2f} | "
              f"status={r.get('machine_status_label','?'):<8} "
              f"failure={r.get('failure_type','?'):<18} "
              f"anomaly_flag={r.get('anomaly_flag',0)} "
              f"rul={r.get('predicted_remaining_life',0):>6.1f}")

# ── 5. H3: Per-machine baseline variance ──
print(f"\n{'─'*80}")
print(" H3: PER-MACHINE BASELINE VARIANCE (do machines have different normal ranges?)")
print(f"{'─'*80}")
sample_mids = list(test_snapshots.keys())[:6]
for mid in sample_mids:
    cursor = col.find({"machine_id": mid}).sort("timestamp", -1).limit(100)
    temps = [d.get("temperature", 0) for d in cursor if isinstance(d.get("temperature"), (int, float))]
    if temps:
        m = statistics.mean(temps)
        s = statistics.pstdev(temps) if len(temps) > 1 else 0
        print(f"  M-{mid:>3}  temp(last 100): mean={m:.2f}  stdev={s:.2f}  "
              f"min={min(temps):.2f}  max={max(temps):.2f}")

client.close()
