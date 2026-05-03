"""Quick precision-recall sweep for fleet_relative_detector on test split."""
import json
from pymongo import MongoClient
from config import MONGODB_URI, MONGO_DB, MONGO_COLLECTION, load_sensor_bounds
from fleet_relative_detector import fleet_relative_score

bounds = load_sensor_bounds()
test_ids = bounds["test_machine_ids"]
client = MongoClient(MONGODB_URI)
col = client[MONGO_DB][MONGO_COLLECTION]
cursor = col.find({"machine_id": {"$in": test_ids}}).sort("timestamp", -1).limit(400)
latest = {}
for d in cursor:
    if d["machine_id"] not in latest:
        latest[d["machine_id"]] = d

events = []
for mid in sorted(latest):
    d = latest[mid]
    events.append({
        "machine_id":  mid,
        "temperature": d.get("temperature", 0),
        "vibration":   d.get("vibration", 0),
        "humidity":    d.get("humidity", 0),
        "pressure":    d.get("pressure", 0),
        "energy_consumption": d.get("energy_consumption", 0),
        "energy":      d.get("energy_consumption", 0),
        "predicted_remaining_life": d.get("predicted_remaining_life", 200),
        "_gt_failure_type":         d.get("failure_type", "Normal"),
        "_gt_machine_status_label": d.get("machine_status_label", "Normal"),
        "_gt_anomaly_flag":         d.get("anomaly_flag", 0),
        "_gt_maintenance_required": d.get("maintenance_required", 0),
    })
client.close()

gt = {}
for e in events:
    gt[e["machine_id"]] = (e["_gt_machine_status_label"] in ("Warning", "Fault")
                            or e["_gt_failure_type"] != "Normal"
                            or e["_gt_anomaly_flag"] == 1
                            or e["_gt_maintenance_required"] == 1)

results = fleet_relative_score(events, decision_quantile=0.0)  # all flagged, get composite_scores
scores = [(r["machine_id"], r["composite_score"]) for r in results]
scores.sort(key=lambda x: -x[1])

print(f"\n{'='*72}")
print(f"  Precision-Recall sweep (composite_score threshold)")
print(f"{'='*72}")
print(f"  {'thresh':>7} {'predicted+':>12} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4} "
      f"{'prec':>7} {'rec':>7} {'F1':>7} {'acc':>7}")
print(f"  {'-'*72}")
for thresh in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80]:
    tp = fp = fn = tn = 0
    for mid, s in scores:
        pred = s >= thresh
        actual = gt[mid]
        if pred and actual:    tp += 1
        elif pred and not actual: fp += 1
        elif not pred and actual: fn += 1
        else: tn += 1
    n_pred = tp + fp
    total = tp + fp + fn + tn
    prec = tp / max(tp+fp, 1) if (tp+fp) else 0
    rec  = tp / max(tp+fn, 1) if (tp+fn) else 0
    f1   = 2*prec*rec / max(prec+rec, 1e-9) if (prec+rec) else 0
    acc  = (tp+tn)/total if total else 0
    print(f"  {thresh:>7.2f} {n_pred:>12} {tp:>4} {fp:>4} {fn:>4} {tn:>4} "
          f"{prec:>7.4f} {rec:>7.4f} {f1:>7.4f} {acc:>7.4f}")
