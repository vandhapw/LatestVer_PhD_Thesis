"""
Test if the digital-twin's failure_type / status / anomaly_flag classifiers
are unrobust — i.e., do labels flip wildly for tiny input perturbations?

Methodology:
  For each machine in the test split, pull last N (≥30) cycles from
  MongoDB. For each cycle pair, compute:
    Δsensor = L2 distance between consecutive sensor vectors (normalized)
    Δlabel  = 1 if any of {failure_type, status, anomaly_flag} changed,
              else 0
  Then for each machine:
    label_entropy = entropy of label distribution (Shannon)
    sensor_stdev  = stdev of normalized sensor vector
    flip_rate     = fraction of consecutive cycles where label changed
    sensitivity   = average ratio of Δlabel / Δsensor (jittery classifier)

  Robustness verdict:
    - HIGH flip_rate + LOW sensor_stdev = UNROBUST classifier
    - HIGH flip_rate + HIGH sensor_stdev = expected (input drives output)
    - LOW flip_rate = robust OR labels never change (check sensor_stdev)
"""
import math
import statistics
from pymongo import MongoClient
from collections import Counter

from config import (
    MONGODB_URI, MONGO_DB, MONGO_COLLECTION, SENSOR_FEATURES,
    load_sensor_bounds, get_normalization_range,
)

bounds = load_sensor_bounds()
test_ids = bounds["test_machine_ids"]

client = MongoClient(MONGODB_URI)
col = client[MONGO_DB][MONGO_COLLECTION]

WINDOW = 30  # last 30 cycles per machine

def normalized_vector(doc):
    v = []
    for f in SENSOR_FEATURES:
        val = doc.get(f, doc.get("energy" if f == "energy_consumption" else f, 0))
        if not isinstance(val, (int, float)):
            v.append(0.0); continue
        lo, hi = get_normalization_range(f)
        v.append((val - lo) / (hi - lo) if hi > lo else 0.0)
    return v

def shannon_entropy(seq):
    counts = Counter(seq)
    total = sum(counts.values())
    return -sum((c/total) * math.log2(c/total) for c in counts.values() if c > 0)

print("="*100)
print(" CLASSIFIER ROBUSTNESS DIAGNOSIS")
print(f" Looking at last {WINDOW} cycles per test machine")
print("="*100)
print(f"\n  {'M':>4} {'n':>4} {'sensor_sd':>10} {'fail_H':>7} {'stat_H':>7} {'anom_H':>7} "
      f"{'flip_rate':>10} {'sens_per_dz':>12}  failure_types_seen")
print(f"  {'-'*112}")

aggregate = {"low_var_low_flip": 0, "low_var_high_flip": 0,
             "high_var_low_flip": 0, "high_var_high_flip": 0}
all_flip_rates = []
all_sensor_sds = []

for mid in test_ids:
    cursor = col.find({"machine_id": mid}).sort("timestamp", -1).limit(WINDOW)
    docs = list(cursor)
    docs.reverse()  # oldest first
    if len(docs) < 5:
        continue

    # Sensor variance (norm L2 across vectors)
    vecs = [normalized_vector(d) for d in docs]
    sensor_means = [statistics.mean(col_vals) for col_vals in zip(*vecs)]
    sensor_sd_per_feat = []
    for k in range(len(SENSOR_FEATURES)):
        col_k = [v[k] for v in vecs]
        sd = statistics.pstdev(col_k) if len(col_k) > 1 else 0.0
        sensor_sd_per_feat.append(sd)
    sensor_sd_mean = statistics.mean(sensor_sd_per_feat)

    # Label sequences
    fail_seq = [d.get("failure_type", "?") for d in docs]
    stat_seq = [d.get("machine_status_label", "?") for d in docs]
    anom_seq = [d.get("anomaly_flag", 0) for d in docs]

    # Entropy of each label
    fail_H = shannon_entropy(fail_seq)
    stat_H = shannon_entropy(stat_seq)
    anom_H = shannon_entropy(anom_seq)

    # Flip rate: fraction of consecutive cycles where ANY label changed
    flips = 0
    sensitivity_sum = 0.0
    sensitivity_count = 0
    for i in range(1, len(docs)):
        any_flip = (fail_seq[i] != fail_seq[i-1]
                    or stat_seq[i] != stat_seq[i-1]
                    or anom_seq[i] != anom_seq[i-1])
        if any_flip:
            flips += 1
        # Δsensor
        dz = math.sqrt(sum((vecs[i][k] - vecs[i-1][k])**2 for k in range(len(vecs[i]))))
        if dz > 1e-9:
            sensitivity_sum += int(any_flip) / dz
            sensitivity_count += 1
    flip_rate = flips / max(len(docs) - 1, 1)
    sensitivity = sensitivity_sum / max(sensitivity_count, 1)

    fail_types = sorted(set(fail_seq))

    print(f"  M-{mid:>2} {len(docs):>4} {sensor_sd_mean:>10.5f} "
          f"{fail_H:>7.3f} {stat_H:>7.3f} {anom_H:>7.3f} "
          f"{flip_rate:>10.3f} {sensitivity:>12.2f}  {fail_types}")

    all_flip_rates.append(flip_rate)
    all_sensor_sds.append(sensor_sd_mean)

    # Bucket
    var_high = sensor_sd_mean > 0.02
    flip_high = flip_rate > 0.3
    if var_high and flip_high:    aggregate["high_var_high_flip"] += 1
    elif var_high:                 aggregate["high_var_low_flip"] += 1
    elif flip_high:                aggregate["low_var_high_flip"] += 1
    else:                          aggregate["low_var_low_flip"] += 1

print(f"\n{'='*100}")
print(" AGGREGATE (cutoff: sensor_sd>0.02 = HIGH var, flip_rate>0.3 = HIGH flip)")
print(f"{'='*100}")
for label, count in aggregate.items():
    print(f"  {label:>22}: {count} machines")

print(f"\n  Mean sensor_sd:   {statistics.mean(all_sensor_sds):.5f}")
print(f"  Mean flip_rate:   {statistics.mean(all_flip_rates):.3f}")
print(f"\n  INTERPRETATION:")
n = aggregate["low_var_high_flip"]
n2 = aggregate["high_var_high_flip"]
if n > 0:
    print(f"  → {n} machines with LOW variance but HIGH label flips")
    print(f"     → CLASSIFIER IS UNROBUST: tiny noise flips labels")
if n2 > 0:
    print(f"  → {n2} machines with HIGH variance + HIGH label flips")
    print(f"     → expected: input changes drive output changes")
n3 = aggregate["high_var_low_flip"]
if n3 > 0:
    print(f"  → {n3} machines with HIGH variance but LOW label flips")
    print(f"     → robust classifier OR labels saturated to a single class")

client.close()
