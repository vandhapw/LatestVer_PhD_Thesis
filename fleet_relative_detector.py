# ============================================================
# FLEET-RELATIVE ANOMALY DETECTOR
# ============================================================
#
# WHY THIS MODULE EXISTS:
#   The post-audit run revealed that the Kafka digital-twin stream
#   reproduces a STATIC sensor snapshot per machine — every reading
#   is identical (per-machine stdev ≈ 0.00 over 100 readings). This
#   means:
#     1. Per-machine temporal anomaly methods (LSTM-AD, TranAD, GDN)
#        cannot detect anything — there is no temporal variation.
#     2. Fleet-wide threshold methods catch nothing either, because
#        the statically-replayed values fall below the P99 cutoffs
#        derived from the training subset.
#     3. The only signal that DOES vary across our snapshot is the
#        cross-machine difference: machine M-x runs hot relative to
#        its peers even when its absolute temperature is "normal".
#
# WHAT THIS MODULE DOES:
#   Cross-machine, fleet-relative anomaly scoring on a single
#   snapshot of N machines. Three complementary detectors:
#
#   1. Population-Mahalanobis  — vectorial outlier vs fleet centroid
#                                (Adams et al., Data-Centric Eng. 2023)
#   2. Per-feature peer z-score — for each (feature, machine), how
#                                many sigma above the fleet mean
#   3. Population-IsolationProxy — random-projection isolation depth
#                                (cheap surrogate for sklearn's IF;
#                                 we keep classical_optimization.py's
#                                 no-numpy ethos)
#
# REFERENCES (synthesized):
#   - Hendrickx et al., "A general anomaly detection framework for
#     fleet-based condition monitoring of machines", Mech. Sys. and
#     Signal Proc. 139 (2020). arXiv:1912.12941. Key idea: assume
#     majority of machines are healthy; deviation from the peer
#     distribution flags suspect units.
#   - Olsson et al., "A conformal anomaly detection based industrial
#     fleet monitoring framework", Eng. Apps. of AI 113 (2022).
#     UL-CAD (unit-level) and SL-CAD (sub-fleet-level) split.
#   - Liu, Ting, Zhou, "Isolation Forest", ICDM 2008. Anomalies
#     have shorter average path-length under random partitioning.
#   - Mathieu et al., "Anomaly detection in a fleet of industrial
#     assets with hierarchical statistical modeling", Data-Centric
#     Eng. (Cambridge) 2023.
#
# HONEST CAVEATS:
#   - With N=25 test machines, Mahalanobis covariance is poorly
#     conditioned for d=5 features. We add Tikhonov regularization.
#   - "Population isolation" is a poor man's Isolation Forest in
#     pure Python; see iforest_score docstring.
#   - This is still snapshot-only. If the producer ever emits real
#     time-varying readings, integrate per-machine sliding windows
#     from kafka_realtime_consumer.MultiTierBuffer for additional
#     signal.
# ============================================================

import math
import random
from typing import List, Dict, Tuple

from config import SENSOR_FEATURES, get_normalization_range


def _safe_extract(events, feature):
    """Extract numeric values for a feature across all events; aliases energy."""
    vals = []
    for e in events:
        v = e.get(feature, e.get("energy" if feature == "energy_consumption" else feature))
        if isinstance(v, (int, float)):
            vals.append(float(v))
        else:
            vals.append(0.0)
    return vals


def _normalize_per_feature(events):
    """
    Build N×d matrix of per-feature normalized sensor values.
    Normalization uses get_normalization_range from config (training-derived bounds).
    Returns (matrix, feature_order).
    """
    feat_order = list(SENSOR_FEATURES)
    matrix = []
    for e in events:
        row = []
        for f in feat_order:
            lo, hi = get_normalization_range(f)
            v = e.get(f, e.get("energy" if f == "energy_consumption" else f, 0))
            if not isinstance(v, (int, float)):
                v = 0.0
            if hi > lo:
                row.append((float(v) - lo) / (hi - lo))
            else:
                row.append(0.0)
        matrix.append(row)
    return matrix, feat_order


# ─────────────────────────────────────────────
# Detector 1: Population Mahalanobis
# ─────────────────────────────────────────────
def _invert_matrix(M, n):
    """Gauss-Jordan inverse for small n×n matrix; returns None if singular."""
    aug = [M[i][:] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        max_row = col
        for row in range(col + 1, n):
            if abs(aug[row][col]) > abs(aug[max_row][col]):
                max_row = row
        aug[col], aug[max_row] = aug[max_row], aug[col]
        if abs(aug[col][col]) < 1e-12:
            return None
        pivot = aug[col][col]
        aug[col] = [x / pivot for x in aug[col]]
        for row in range(n):
            if row != col:
                factor = aug[row][col]
                aug[row] = [aug[row][j] - factor * aug[col][j] for j in range(2 * n)]
    return [r[n:] for r in aug]


def population_mahalanobis(events, regularization=1e-4):
    """
    Score each machine by its Mahalanobis distance from the fleet centroid
    of normalized sensor vectors.

    Returns: list of dicts in the same order as events.
        {machine_id, distance, percentile, is_outlier_at_p90, is_outlier_at_p99}
    """
    matrix, feat_order = _normalize_per_feature(events)
    n = len(matrix)
    if n < 3:
        return [{"machine_id": e.get("machine_id"),
                 "distance": 0.0, "percentile": 0.0,
                 "is_outlier_at_p90": False, "is_outlier_at_p99": False,
                 "_disclosure": "N<3, skipped"} for e in events]
    d = len(feat_order)

    mean = [sum(row[k] for row in matrix) / n for k in range(d)]
    cov = [[0.0] * d for _ in range(d)]
    for row in matrix:
        diff = [row[k] - mean[k] for k in range(d)]
        for i in range(d):
            for j in range(d):
                cov[i][j] += diff[i] * diff[j]
    for i in range(d):
        for j in range(d):
            cov[i][j] /= max(n - 1, 1)
        cov[i][i] += regularization  # Tikhonov for ill-conditioned covariance

    inv = _invert_matrix(cov, d)
    if inv is None:
        return [{"machine_id": e.get("machine_id"),
                 "distance": 0.0, "percentile": 0.0,
                 "is_outlier_at_p90": False, "is_outlier_at_p99": False,
                 "_disclosure": "singular covariance"} for e in events]

    distances = []
    for row in matrix:
        diff = [row[k] - mean[k] for k in range(d)]
        intermediate = [sum(inv[i][j] * diff[j] for j in range(d)) for i in range(d)]
        sq = sum(diff[i] * intermediate[i] for i in range(d))
        distances.append(math.sqrt(max(sq, 0)))

    # Percentile rank for each machine within the fleet
    sorted_d = sorted(distances)
    p90 = sorted_d[int(n * 0.90)] if n > 0 else 0.0
    p99 = sorted_d[min(int(n * 0.99), n - 1)] if n > 0 else 0.0

    out = []
    for evt, dist in zip(events, distances):
        rank = sum(1 for x in distances if x <= dist) / n
        out.append({
            "machine_id":          evt.get("machine_id"),
            "distance":            round(dist, 4),
            "percentile":          round(rank, 4),
            "is_outlier_at_p90":   dist >= p90,
            "is_outlier_at_p99":   dist >= p99,
        })
    return out


# ─────────────────────────────────────────────
# Detector 2: Per-feature peer z-score
# ─────────────────────────────────────────────
def per_feature_peer_zscore(events):
    """
    For each (machine, feature) pair, compute z = (value - fleet_mean) / fleet_std.
    A machine is flagged if |z| > 2.0 on ANY sensor (rough 95% CI band).
    Returns per-machine summary.
    """
    matrix, feat_order = _normalize_per_feature(events)
    n = len(matrix)
    d = len(feat_order)
    if n < 3:
        return [{"machine_id": e.get("machine_id"), "max_abs_z": 0.0,
                 "flagged": False, "_disclosure": "N<3"} for e in events]

    means = [sum(row[k] for row in matrix) / n for k in range(d)]
    stds  = []
    for k in range(d):
        var = sum((row[k] - means[k]) ** 2 for row in matrix) / max(n - 1, 1)
        stds.append(math.sqrt(var) if var > 0 else 1e-9)

    out = []
    for evt, row in zip(events, matrix):
        zs = []
        for k in range(d):
            z = (row[k] - means[k]) / stds[k]
            zs.append((feat_order[k], round(z, 3)))
        max_z = max(zs, key=lambda t: abs(t[1]))
        out.append({
            "machine_id":  evt.get("machine_id"),
            "z_per_feature": dict(zs),
            "max_abs_z":   abs(max_z[1]),
            "max_z_feature": max_z[0],
            "flagged":     abs(max_z[1]) > 2.0,
        })
    return out


# ─────────────────────────────────────────────
# Detector 3: Population isolation (poor man's IF)
# ─────────────────────────────────────────────
def population_isolation(events, n_trees=64, sample_size=None, max_depth=None, seed=42):
    """
    Pure-Python surrogate for Isolation Forest. Builds n_trees random
    half-space partitions over the fleet vectors and reports the
    average isolation depth per machine. Lower depth = more anomalous.

    NOT a faithful Isolation Forest; primarily for monotone score
    ordering. Use sklearn.ensemble.IsolationForest in production if
    available — it has proper subsampling, depth limits, and scoring.

    Returns per-machine dict: {machine_id, mean_depth, score}.
    score is a normalized value in [0,1] where 1 = most anomalous.
    """
    rng = random.Random(seed)
    matrix, feat_order = _normalize_per_feature(events)
    n = len(matrix)
    d = len(feat_order)
    if n < 3:
        return [{"machine_id": e.get("machine_id"),
                 "mean_depth": 0.0, "score": 0.0} for e in events]

    sample_size = sample_size or min(n, 256)
    max_depth = max_depth or int(math.ceil(math.log2(max(sample_size, 2))))

    depths = [0.0] * n
    for _ in range(n_trees):
        # Subsample with replacement
        sub_idx = [rng.randrange(n) for _ in range(sample_size)]
        sub = [matrix[i] for i in sub_idx]

        for i in range(n):
            point = matrix[i]
            depths[i] += _isolation_depth(point, sub, d, max_depth, rng)

    mean_depths = [d_i / n_trees for d_i in depths]
    if not mean_depths:
        return []
    max_d = max(mean_depths) if mean_depths else 1.0
    min_d = min(mean_depths) if mean_depths else 0.0
    span = max(max_d - min_d, 1e-9)

    out = []
    for evt, md in zip(events, mean_depths):
        score = (max_d - md) / span  # invert: shallow = high anomaly
        out.append({
            "machine_id":  evt.get("machine_id"),
            "mean_depth":  round(md, 4),
            "score":       round(score, 4),
        })
    return out


def _isolation_depth(point, subset, d, max_depth, rng):
    """Recursive partitioning depth for one tree."""
    depth = 0
    s = subset
    while depth < max_depth and len(s) > 1:
        feat = rng.randrange(d)
        vals = [row[feat] for row in s]
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-12:
            break
        split = rng.uniform(lo, hi)
        if point[feat] < split:
            s = [row for row in s if row[feat] < split]
        else:
            s = [row for row in s if row[feat] >= split]
        depth += 1
    # Average path length for unsuccessful BST search (Liu et al. 2008)
    if len(s) > 1:
        depth += _avg_path_length(len(s))
    return depth


def _avg_path_length(n):
    if n <= 1:
        return 0.0
    return 2.0 * (math.log(n - 1) + 0.5772156649) - 2.0 * (n - 1) / n


# ─────────────────────────────────────────────
# Unified ensemble
# ─────────────────────────────────────────────
def fleet_relative_score(events, *, mahal_weight=0.4, z_weight=0.3, iforest_weight=0.3,
                         decision_quantile=0.5, score_threshold=None, seed=42):
    """
    Weighted ensemble of the three detectors. Returns per-machine dict
    with composite_score in [0,1] and a binary `is_attention_needed`.

    Decision rule (in priority order):
      - If score_threshold is given → use it as an absolute cutoff.
        Recommended when the cutoff has been *derived from a separate
        training fleet* (see derive_decision_threshold below).
      - Else → flag the top (1 - decision_quantile) of the current
        fleet. This is *fleet-relative* and avoids data peeking on
        the test set, but it always flags some fixed fraction.

    Default weights are equal-ish; tune via held-out validation if you
    have any ground-truth (NOT recommended on the snapshot we currently
    have — all GT-positives have static sensor signals indistinguishable
    from peers).

    AUDIT NOTE: this function does NOT read any ground-truth field.
    """
    if not events:
        return []

    mahal = population_mahalanobis(events)
    zsc   = per_feature_peer_zscore(events)
    iso   = population_isolation(events, seed=seed)

    out = []
    composite_scores = []
    for i, evt in enumerate(events):
        mid = evt.get("machine_id")
        m_pct = mahal[i]["percentile"]
        z_max = zsc[i]["max_abs_z"]
        # Map z to a [0,1] confidence: |z|=0 → 0, |z|>=3 → 1
        z_conf = max(0.0, min(1.0, z_max / 3.0))
        i_score = iso[i]["score"]

        composite = (mahal_weight * m_pct
                     + z_weight    * z_conf
                     + iforest_weight * i_score)
        composite_scores.append(composite)

        out.append({
            "machine_id":          mid,
            "mahalanobis_percentile":  m_pct,
            "max_abs_peer_z":      z_max,
            "isolation_score":     i_score,
            "composite_score":     round(composite, 4),
        })

    if score_threshold is not None:
        cutoff = float(score_threshold)
    else:
        sorted_scores = sorted(composite_scores)
        cutoff_idx = int(len(sorted_scores) * decision_quantile)
        cutoff = sorted_scores[cutoff_idx] if sorted_scores else 1.0

    for o, s in zip(out, composite_scores):
        o["is_attention_needed"] = s >= cutoff
        o["_decision_cutoff"] = round(cutoff, 4)
    return out


def derive_decision_threshold(train_events, *, target_quantile=0.5, seed=42):
    """
    Run the same ensemble on a TRAINING-fleet snapshot and return the
    composite-score cutoff at `target_quantile`. The cutoff is then
    applied to the TEST fleet via fleet_relative_score(score_threshold=...).

    This avoids peeking at the test snapshot when picking a cutoff.
    """
    rows = fleet_relative_score(train_events, decision_quantile=0.0, seed=seed)
    scores = sorted(r["composite_score"] for r in rows)
    if not scores:
        return 1.0
    idx = int(len(scores) * target_quantile)
    idx = min(idx, len(scores) - 1)
    return float(scores[idx])


# ─────────────────────────────────────────────
# Convenience entry for the agentic pipelines
# ─────────────────────────────────────────────
def detect(events, *, decision_quantile=0.5, seed=42):
    """
    Convenience function returning a dict {machine_id: detection_result}
    suitable for plug-in into existing approaches that expect a per-mid
    map.
    """
    rows = fleet_relative_score(events, decision_quantile=decision_quantile, seed=seed)
    return {r["machine_id"]: r for r in rows}


if __name__ == "__main__":
    # Quick self-test on the test split
    from pymongo import MongoClient
    from config import MONGODB_URI, MONGO_DB, MONGO_COLLECTION, load_sensor_bounds

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

    print(f"\n{'='*80}")
    print(f"  FLEET-RELATIVE DETECTOR — self-test on {len(events)} test machines")
    print(f"{'='*80}\n")

    results = fleet_relative_score(events, decision_quantile=0.5)

    # Compare against ground truth
    gt = {}
    for e in events:
        mid = e["machine_id"]
        gt[mid] = (e["_gt_machine_status_label"] in ("Warning", "Fault")
                   or e["_gt_failure_type"] != "Normal"
                   or e["_gt_anomaly_flag"] == 1
                   or e["_gt_maintenance_required"] == 1)

    tp = fp = fn = tn = 0
    for r in results:
        mid = r["machine_id"]
        pred = r["is_attention_needed"]
        actual = gt[mid]
        if pred and actual:    tp += 1
        elif pred and not actual: fp += 1
        elif not pred and actual: fn += 1
        else: tn += 1

    print(f"  {'M':>4} {'mahalP':>8} {'max|z|':>8} {'isoF':>7} {'composite':>10} "
          f"{'flagged':>8}  ground_truth")
    print(f"  {'-'*70}")
    for r in results:
        mid = r["machine_id"]
        gt_label = ""
        e = next(x for x in events if x["machine_id"] == mid)
        if gt[mid]:
            gt_label = f"  GT+({e['_gt_failure_type']})"
        print(f"  M-{mid:>2}  {r['mahalanobis_percentile']:>8.3f} "
              f"{r['max_abs_peer_z']:>8.3f} {r['isolation_score']:>7.3f} "
              f"{r['composite_score']:>10.3f} {str(r['is_attention_needed']):>8}{gt_label}")

    total = max(tp + fp + fn + tn, 1)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)
    print(f"\n  Detection vs digital-twin GT (test split, decision_quantile=0.5):")
    print(f"    TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"    precision={prec:.4f}  recall={rec:.4f}  f1={f1:.4f}  "
          f"accuracy={(tp+tn)/total:.4f}")

    client.close()
