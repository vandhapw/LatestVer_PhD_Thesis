# ============================================================
# CLASSICAL OPTIMIZATION MODULE
# ============================================================
# 5 classical algorithms that mirror the 5 quantum integration
# points. Each can later be replaced by its quantum counterpart
# (VQC, QGNN, QAOA, Quantum Kernel, Quantum Annealing).
#
# Integration Point 1: Multivariate anomaly    (→ future VQC)
# Integration Point 2: Graph correlation       (→ future QGNN)
# Integration Point 3: Scheduling optimization (→ future QAOA)
# Integration Point 4: Consensus validation    (→ future Q-Kernel)
# Integration Point 5: Threshold optimization  (→ future Q-Annealing)
#
# All implementations are pure Python — no numpy, scipy, sklearn.
# ============================================================

import math
import random
import statistics
from collections import defaultdict

random.seed(42)  # disclosed; reproducibility seed for SA / NM optimizers

# AUDIT NOTE: NORM_RANGES previously hardcoded — now derived from
# the training split via config.get_normalization_range. The local
# fallback table is kept ONLY for the case where the bounds file is
# missing, and is identical to the previous hardcoded values so that
# behavior does not silently change.
try:
    from config import (
        SENSOR_FEATURES as _CFG_SENSOR_FEATURES,
        get_normalization_range as _cfg_get_normalization_range,
        load_sensor_bounds as _cfg_load_sensor_bounds,
    )
    SENSOR_FEATURES = _CFG_SENSOR_FEATURES
    _HAVE_CFG = True
except ImportError:
    SENSOR_FEATURES = ["temperature", "vibration", "humidity", "pressure", "energy_consumption"]
    _HAVE_CFG = False

_FALLBACK_NORM_RANGES = {
    "temperature":        (60, 100),
    "vibration":          (30, 90),
    "humidity":           (40, 80),
    "pressure":           (1.0, 5.0),
    "energy_consumption": (1.0, 5.0),
}

# Backward-compat alias — modules importing NORM_RANGES still work.
NORM_RANGES = _FALLBACK_NORM_RANGES


# ─────────────────────────────────────────────
# HELPER: Linear algebra primitives (pure Python)
# ─────────────────────────────────────────────

def normalize_feature(value, feature):
    if _HAVE_CFG:
        try:
            lo, hi = _cfg_get_normalization_range(feature)
        except Exception:
            lo, hi = NORM_RANGES.get(feature, (0, 1))
    else:
        lo, hi = NORM_RANGES.get(feature, (0, 1))
    return max(0.0, min(1.0, (value - lo) / (hi - lo))) if hi > lo else 0.0


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def vec_sub(a, b):
    return [x - y for x, y in zip(a, b)]


def vec_norm(v):
    return math.sqrt(sum(x * x for x in v))


def mat_mul_vec(M, v):
    return [dot(row, v) for row in M]


def identity_matrix(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def outer_product(a, b):
    return [[ai * bj for bj in b] for ai in a]


def mat_add(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def mat_scale(A, s):
    return [[A[i][j] * s for j in range(len(A[0]))] for i in range(len(A))]


def cosine_similarity(a, b):
    d = dot(a, b)
    na = vec_norm(a)
    nb = vec_norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return d / (na * nb)


# ═══════════════════════════════════════════════
# POINT 1: MULTIVARIATE ANOMALY DETECTOR
# Classical mirror of VQC
# ═══════════════════════════════════════════════

class MultivariateAnomalyDetector:
    """
    Mahalanobis distance-based multivariate anomaly detection.
    Detects anomalies that only appear when multiple features
    are considered jointly — exactly what VQC does via entanglement.

    Classical approach: compute covariance matrix from history,
    then Mahalanobis distance of new point from centroid.
    High distance = anomaly.
    """

    def __init__(self, threshold_percentile=95):
        self.threshold_pct = threshold_percentile
        self.history = {}  # machine_id → list of feature vectors

    def update(self, machine_id, event):
        """Add event to history for this machine."""
        vec = [normalize_feature(event.get(f, 0), f) for f in SENSOR_FEATURES]
        if machine_id not in self.history:
            self.history[machine_id] = []
        self.history[machine_id].append(vec)
        # Keep last 120 points (same as Tier 2)
        if len(self.history[machine_id]) > 120:
            self.history[machine_id] = self.history[machine_id][-120:]

    def detect(self, machine_id, event):
        """
        Compute Mahalanobis distance for this event.
        Returns anomaly score (0-1) and whether it exceeds threshold.
        """
        hist = self.history.get(machine_id, [])
        if len(hist) < 10:
            return {"anomaly_score": 0.0, "is_anomaly": False,
                    "method": "mahalanobis", "reason": "insufficient_history"}

        d = len(SENSOR_FEATURES)
        current = [normalize_feature(event.get(f, 0), f) for f in SENSOR_FEATURES]

        # Compute mean vector
        mean = [0.0] * d
        for vec in hist:
            for i in range(d):
                mean[i] += vec[i]
        mean = [m / len(hist) for m in mean]

        # Compute covariance matrix
        cov = [[0.0] * d for _ in range(d)]
        for vec in hist:
            diff = vec_sub(vec, mean)
            for i in range(d):
                for j in range(d):
                    cov[i][j] += diff[i] * diff[j]
        n = len(hist)
        cov = [[cov[i][j] / (n - 1) for j in range(d)] for i in range(d)]

        # Add small regularization to avoid singular matrix
        for i in range(d):
            cov[i][i] += 1e-6

        # Invert covariance (using Gauss-Jordan for small 5x5 matrix)
        cov_inv = self._invert_matrix(cov, d)
        if cov_inv is None:
            return {"anomaly_score": 0.0, "is_anomaly": False,
                    "method": "mahalanobis", "reason": "singular_covariance"}

        # Mahalanobis distance: sqrt((x-μ)ᵀ Σ⁻¹ (x-μ))
        diff = vec_sub(current, mean)
        intermediate = mat_mul_vec(cov_inv, diff)
        mahal_sq = dot(diff, intermediate)
        mahal = math.sqrt(max(0, mahal_sq))

        # Compute threshold from historical distances
        hist_distances = []
        for vec in hist:
            h_diff = vec_sub(vec, mean)
            h_int = mat_mul_vec(cov_inv, h_diff)
            h_dist = math.sqrt(max(0, dot(h_diff, h_int)))
            hist_distances.append(h_dist)

        hist_distances.sort()
        idx = int(len(hist_distances) * self.threshold_pct / 100)
        idx = min(idx, len(hist_distances) - 1)
        threshold = hist_distances[idx] if hist_distances else 3.0

        # Normalize score to 0-1 range
        score = min(1.0, mahal / (threshold * 1.5)) if threshold > 0 else 0.0
        is_anomaly = mahal > threshold

        # Find top contributing features
        contributions = []
        for i, feat in enumerate(SENSOR_FEATURES):
            feat_contrib = abs(diff[i] * intermediate[i])
            contributions.append((feat, round(feat_contrib, 4)))
        contributions.sort(key=lambda x: -x[1])

        return {
            "anomaly_score": round(score, 4),
            "mahalanobis_distance": round(mahal, 4),
            "threshold": round(threshold, 4),
            "is_anomaly": is_anomaly,
            "top_contributors": contributions[:3],
            "method": "mahalanobis",
        }

    def _invert_matrix(self, M, n):
        """Gauss-Jordan matrix inversion for small n×n matrix."""
        # Augment with identity
        aug = [M[i][:] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        for col in range(n):
            # Find pivot
            max_row = col
            for row in range(col + 1, n):
                if abs(aug[row][col]) > abs(aug[max_row][col]):
                    max_row = row
            aug[col], aug[max_row] = aug[max_row], aug[col]
            if abs(aug[col][col]) < 1e-10:
                return None  # Singular
            # Scale pivot row
            pivot = aug[col][col]
            aug[col] = [x / pivot for x in aug[col]]
            # Eliminate
            for row in range(n):
                if row != col:
                    factor = aug[row][col]
                    aug[row] = [aug[row][j] - factor * aug[col][j] for j in range(2 * n)]
        return [row[n:] for row in aug]


# ═══════════════════════════════════════════════
# POINT 2: GRAPH-BASED CORRELATION
# Classical mirror of QGNN
# ═══════════════════════════════════════════════

class GraphCorrelationEngine:
    """
    Graph-based inter-machine correlation analysis using
    message passing and spectral-inspired clustering.
    Mirrors QGNN's quantum message passing.

    Uses actual T-GCN topology (anchor_machine, graph_distance,
    influence_score) from Kafka data.
    """

    def __init__(self, n_iterations=3):
        self.n_iterations = n_iterations

    def analyze(self, events, reasonings):
        """
        Build graph from events, run message passing, detect clusters.
        Returns correlation analysis with propagation risk.
        """
        n = len(events)
        if n < 2:
            return {"clusters": [], "max_propagation_risk": 0, "cascade_warning": False}

        # Build adjacency from T-GCN features
        adj = defaultdict(list)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                ei, ej = events[i], events[j]
                same_anchor = ei.get("anchor_machine") == ej.get("anchor_machine")
                close = abs(ei.get("graph_distance", 0) - ej.get("graph_distance", 0)) < 2
                infl = max(ei.get("influence_score", 0), ej.get("influence_score", 0))
                if same_anchor or (close and infl > 0.3):
                    weight = 0.5 + infl * 0.5
                    adj[i].append((j, weight))

        # Initialize node features (normalized sensor + risk)
        node_feats = []
        for i, evt in enumerate(events):
            feats = [normalize_feature(evt.get(f, 0), f) for f in SENSOR_FEATURES]
            feats.append(reasonings[i].get("risk_score", 0))
            node_feats.append(feats)

        d = len(node_feats[0])

        # Message passing rounds
        for mp_round in range(self.n_iterations):
            new_feats = []
            for i in range(n):
                neighbors = adj.get(i, [])
                if not neighbors:
                    new_feats.append(node_feats[i][:])
                    continue
                agg = [0.0] * d
                total_w = 0.0
                for j, w in neighbors:
                    for k in range(d):
                        agg[k] += node_feats[j][k] * w
                    total_w += w
                if total_w > 0:
                    agg = [a / total_w for a in agg]
                # Combine: 70% self + 30% neighborhood (non-linear)
                combined = []
                for k in range(d):
                    val = 0.7 * node_feats[i][k] + 0.3 * agg[k]
                    combined.append(max(0, val))  # ReLU
                new_feats.append(combined)
            node_feats = new_feats

        # Compute graph anomaly scores (deviation after message passing)
        graph_scores = []
        for i in range(n):
            original = [normalize_feature(events[i].get(f, 0), f) for f in SENSOR_FEATURES]
            original.append(reasonings[i].get("risk_score", 0))
            deviation = sum(abs(node_feats[i][k] - original[k]) for k in range(d))
            graph_scores.append(round(deviation / d, 4))

        # Cluster detection via similarity grouping
        clusters = self._detect_clusters(events, node_feats, graph_scores, reasonings)

        max_prop = max((c["propagation_risk"] for c in clusters), default=0)
        cascade = max_prop > 0.5

        return {
            "clusters": clusters,
            "max_propagation_risk": round(max_prop, 4),
            "cascade_warning": cascade,
            "graph_scores": {events[i]["machine_id"]: graph_scores[i] for i in range(n)},
            "n_edges": sum(len(v) for v in adj.values()) // 2,
            "method": "classical_graph_message_passing",
        }

    def _detect_clusters(self, events, node_feats, graph_scores, reasonings):
        """Simple similarity-based clustering on post-message-passing features."""
        n = len(events)
        visited = [False] * n
        clusters = []
        SIM_THRESHOLD = 0.85

        for i in range(n):
            if visited[i]:
                continue
            # Only start clusters from machines with some signal
            if reasonings[i].get("risk_score", 0) < 0.1 and graph_scores[i] < 0.05:
                continue
            cluster_members = [i]
            visited[i] = True
            for j in range(i + 1, n):
                if visited[j]:
                    continue
                sim = cosine_similarity(node_feats[i], node_feats[j])
                if sim >= SIM_THRESHOLD:
                    cluster_members.append(j)
                    visited[j] = True

            if len(cluster_members) >= 2:
                machine_ids = [f"M-{events[k]['machine_id']}" for k in cluster_members]
                risks = [reasonings[k].get("risk_score", 0) for k in cluster_members]
                root_idx = cluster_members[risks.index(max(risks))]

                # Determine cluster type from dominant failure
                failure_types = [events[k].get("failure_type", "Normal") for k in cluster_members]
                ft_counts = defaultdict(int)
                for ft in failure_types:
                    if ft != "Normal":
                        ft_counts[ft] += 1
                dominant = max(ft_counts, key=ft_counts.get) if ft_counts else "unknown"
                type_map = {"Overheating": "thermal", "Electrical Fault": "electrical",
                            "Vibration Issue": "vibration", "Pressure Drop": "pressure"}
                cluster_type = type_map.get(dominant, "combined")

                prop_risk = min(1.0, sum(risks) / len(risks) + len(cluster_members) * 0.05)

                clusters.append({
                    "machines": machine_ids,
                    "type": cluster_type,
                    "propagation_risk": round(prop_risk, 4),
                    "root_machine": f"M-{events[root_idx]['machine_id']}",
                    "size": len(cluster_members),
                })

        clusters.sort(key=lambda c: -c["propagation_risk"])
        return clusters


# ═══════════════════════════════════════════════
# POINT 3: SCHEDULING OPTIMIZATION
# Classical mirror of QAOA
# ═══════════════════════════════════════════════

class SchedulingOptimizer:
    """
    Simulated Annealing for maintenance scheduling optimization.
    Mirrors QAOA's combinatorial optimization.

    Problem: select which machines to schedule for maintenance,
    minimizing total cost while respecting constraints.

    Variables: x_i ∈ {0,1} for each machine (schedule or defer)
    Objective: minimize Σ(downtime_i * x_i) - Σ(risk_i * x_i * w_risk)
    Constraints: Σx_i ≤ max_simultaneous, working hours only
    """

    def __init__(self, max_simultaneous=5, max_iterations=2000, initial_temp=10.0):
        self.max_sim = max_simultaneous
        self.max_iter = max_iterations
        self.init_temp = initial_temp

    def optimize(self, machines_data):
        """
        Find optimal maintenance schedule.
        machines_data: list of {machine_id, risk_score, predicted_rul, failure_type, ...}
        Returns: schedule with selected machines and optimization metrics.
        """
        n = len(machines_data)
        if n == 0:
            return {"schedule": [], "total_cost": 0, "method": "simulated_annealing"}

        risks = [m.get("risk_score", 0) for m in machines_data]
        ruls  = [m.get("predicted_rul", 200) for m in machines_data]
        max_rul = max(ruls) if ruls else 1
        urgencies = [(max_rul - r) / max_rul for r in ruls]

        # Estimated downtime per machine (hours, based on failure type)
        downtime_map = {"Overheating": 3.0, "Electrical Fault": 4.0,
                        "Vibration Issue": 2.5, "Pressure Drop": 2.0, "Normal": 1.0}
        downtimes = [downtime_map.get(m.get("failure_type", "Normal"), 2.0)
                     for m in machines_data]

        # Cost per hour downtime (arbitrary unit)
        cost_per_hour = 150.0

        def cost_function(solution):
            n_sched = sum(solution)
            # Constraint penalty
            penalty = max(0, n_sched - self.max_sim) ** 2 * 1000.0
            # Downtime cost (minimize)
            downtime_cost = sum(downtimes[i] * solution[i] * cost_per_hour for i in range(n))
            # Risk benefit (maximize → negate)
            risk_benefit = sum(risks[i] * solution[i] * 500.0 for i in range(n))
            # Urgency benefit
            urgency_benefit = sum(urgencies[i] * solution[i] * 300.0 for i in range(n))
            return downtime_cost - risk_benefit - urgency_benefit + penalty

        # Initialize: schedule top-K by risk
        current = [0] * n
        indexed = sorted(range(n), key=lambda i: -risks[i])
        for i in indexed[:self.max_sim]:
            if risks[i] > 0.05:  # only schedule if some risk
                current[i] = 1

        current_cost = cost_function(current)
        best = current[:]
        best_cost = current_cost

        # Simulated annealing
        for iteration in range(self.max_iter):
            temp = self.init_temp * (1 - iteration / self.max_iter)
            if temp < 0.01:
                break

            # Generate neighbor: flip one random bit
            neighbor = current[:]
            flip_idx = random.randint(0, n - 1)
            neighbor[flip_idx] = 1 - neighbor[flip_idx]

            neighbor_cost = cost_function(neighbor)
            delta = neighbor_cost - current_cost

            # Accept or reject
            if delta < 0:
                current = neighbor
                current_cost = neighbor_cost
            else:
                acceptance = math.exp(-delta / temp) if temp > 0 else 0
                if random.random() < acceptance:
                    current = neighbor
                    current_cost = neighbor_cost

            if current_cost < best_cost:
                best = current[:]
                best_cost = current_cost

        # Build schedule from best solution
        schedule = []
        base_date = datetime.date.today() + datetime.timedelta(days=1)
        hour_slot = 8  # start at 08:00
        for i in range(n):
            if best[i] == 1:
                m = machines_data[i]
                mid = m.get("machine_id", i + 1)
                ft = m.get("failure_type", "Normal")
                maint_type = "corrective" if ft != "Normal" else "preventive"
                if risks[i] >= 0.6:
                    priority = "P1"
                elif risks[i] >= 0.3:
                    priority = "P2"
                elif urgencies[i] > 0.5:
                    priority = "P3"
                else:
                    priority = "P4"

                schedule.append({
                    "machine_id": f"M-{mid}",
                    "maintenance_type": maint_type,
                    "scheduled_date": str(base_date),
                    "scheduled_time": f"{hour_slot:02d}:00-{hour_slot + int(downtimes[i]):02d}:00",
                    "priority": priority,
                    "estimated_downtime_hours": round(downtimes[i], 1),
                    "risk_score": round(risks[i], 3),
                    "urgency": round(urgencies[i], 3),
                })
                hour_slot += int(math.ceil(downtimes[i]))
                if hour_slot >= 18:
                    base_date += datetime.timedelta(days=1)
                    hour_slot = 8

        schedule.sort(key=lambda s: (-{"P1": 4, "P2": 3, "P3": 2, "P4": 1}[s["priority"]],))

        total_downtime = sum(s["estimated_downtime_hours"] for s in schedule)
        total_cost = total_downtime * cost_per_hour

        # Compute improvement vs greedy baseline
        greedy_solution = [0] * n
        greedy_indexed = sorted(range(n), key=lambda i: -risks[i])
        for i in greedy_indexed[:self.max_sim]:
            greedy_solution[i] = 1
        greedy_cost = cost_function(greedy_solution)

        improvement = ((greedy_cost - best_cost) / abs(greedy_cost) * 100) if greedy_cost != 0 else 0

        return {
            "schedule": schedule,
            "total_downtime_hours": round(total_downtime, 1),
            "total_cost": round(total_cost, 2),
            "machines_scheduled": sum(best),
            "sa_iterations": self.max_iter,
            "sa_best_cost": round(best_cost, 2),
            "greedy_cost": round(greedy_cost, 2),
            "improvement_vs_greedy_pct": round(improvement, 2),
            "method": "simulated_annealing",
        }


# ═══════════════════════════════════════════════
# POINT 4: CONSENSUS VALIDATION
# Classical mirror of Quantum Kernel Voting
# ═══════════════════════════════════════════════

class ConsensusValidator:
    """
    Ensemble agreement validation using cosine similarity
    between agent output vectors. Mirrors quantum kernel
    fidelity-based voting.

    Converts each agent's output to a numeric vector,
    computes pairwise similarity, determines agreement.
    """

    RISK_MAP = {"LOW": 0.1, "MEDIUM": 0.5, "HIGH": 0.8, "CRITICAL": 1.0}
    ACTION_MAP = {"monitor": 0.1, "inspect": 0.3,
                  "schedule_maintenance": 0.7, "immediate_shutdown": 1.0}

    def validate(self, monitoring, diagnosis, planning, reasoning):
        """
        Compute agreement between agent outputs.
        Returns consensus decision with reliability score.
        """
        # Convert each agent output to numeric vector
        vec_monitoring = self._vectorize_monitoring(monitoring)
        vec_diagnosis = self._vectorize_diagnosis(diagnosis)
        vec_planning = self._vectorize_planning(planning)
        vec_reasoning = self._vectorize_reasoning(reasoning)

        vectors = {
            "monitoring": vec_monitoring,
            "diagnosis": vec_diagnosis,
            "planning": vec_planning,
            "reasoning": vec_reasoning,
        }

        # Compute pairwise similarities
        agents = list(vectors.keys())
        similarities = {}
        total_sim = 0.0
        n_pairs = 0
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                sim = cosine_similarity(vectors[agents[i]], vectors[agents[j]])
                similarities[f"{agents[i]}_vs_{agents[j]}"] = round(sim, 4)
                total_sim += sim
                n_pairs += 1

        avg_agreement = total_sim / n_pairs if n_pairs > 0 else 0.0

        # Determine consensus risk level (weighted vote)
        risk_votes = []
        risk_mon = self._extract_risk(monitoring)
        risk_rsn = reasoning.get("severity", "LOW")
        risk_plan = self._infer_risk_from_action(planning)
        risk_votes = [risk_mon, risk_rsn, risk_plan]

        risk_scores = [self.RISK_MAP.get(r, 0.1) for r in risk_votes]
        avg_risk_score = sum(risk_scores) / len(risk_scores)

        if avg_risk_score >= 0.7:
            agreed_risk = "HIGH"
        elif avg_risk_score >= 0.4:
            agreed_risk = "MEDIUM"
        else:
            agreed_risk = "LOW"

        # Determine consensus action
        action_plan = self._extract_action(planning)
        agreed_action = action_plan if action_plan else "monitor"

        # Disagreement detection
        risk_spread = max(risk_scores) - min(risk_scores)
        has_disagreement = risk_spread > 0.4 or avg_agreement < 0.6

        # Reliability = agreement * (1 - spread)
        reliability = avg_agreement * (1 - risk_spread * 0.5)

        return {
            "agreed_risk": agreed_risk,
            "agreed_action": agreed_action,
            "agreement_ratio": round(avg_agreement, 4),
            "has_disagreement": has_disagreement,
            "reliability_score": round(max(0, min(1, reliability)), 4),
            "pairwise_similarities": similarities,
            "risk_votes": risk_votes,
            "method": "cosine_similarity_consensus",
        }

    def _vectorize_monitoring(self, m):
        risk = self.RISK_MAP.get(
            m.get("risk_level", m.get("risk", "LOW")), 0.1)
        anomaly = 1.0 if m.get("anomaly_detected", False) else 0.0
        conf = m.get("confidence", 0.5)
        return [risk, anomaly, conf, risk * conf]

    def _vectorize_diagnosis(self, d):
        cause_map = {"normal": 0.0, "unknown": 0.2, "lubrication": 0.4,
                     "bearing_wear": 0.6, "electrical": 0.7, "overheating": 0.9}
        cause = cause_map.get(d.get("probable_cause", d.get("cause", "unknown")), 0.3)
        urgency = min(1.0, d.get("urgency_hours", 72) / 72)
        return [cause, 1.0 - urgency, cause * (1.0 - urgency), 0.5]

    def _vectorize_planning(self, p):
        action = self.ACTION_MAP.get(
            p.get("action", p.get("recommended_action", "monitor")), 0.1)
        priority_map = {"P1": 1.0, "P2": 0.7, "P3": 0.4, "P4": 0.1}
        priority = priority_map.get(p.get("priority", "P4"), 0.1)
        notify = 1.0 if p.get("notify_supervisor", False) else 0.0
        return [action, priority, notify, action * priority]

    def _vectorize_reasoning(self, r):
        risk = r.get("risk_score", 0)
        sev_map = {"LOW": 0.1, "MEDIUM": 0.5, "HIGH": 0.9}
        sev = sev_map.get(r.get("severity", "LOW"), 0.1)
        n_flags = min(1.0, len(r.get("flags", [])) / 5)
        return [risk, sev, n_flags, risk * sev]

    def _extract_risk(self, m):
        return m.get("risk_level", m.get("risk", "LOW"))

    def _extract_action(self, p):
        return p.get("action", p.get("recommended_action", "monitor"))

    def _infer_risk_from_action(self, p):
        action = self._extract_action(p)
        action_to_risk = {"monitor": "LOW", "inspect": "MEDIUM",
                          "schedule_maintenance": "HIGH",
                          "immediate_shutdown": "CRITICAL"}
        return action_to_risk.get(action, "LOW")


# ═══════════════════════════════════════════════
# POINT 5: THRESHOLD OPTIMIZATION
# Classical mirror of Quantum Annealing
# ═══════════════════════════════════════════════

class ThresholdOptimizer:
    """
    Nelder-Mead simplex optimization for reasoning engine thresholds.
    Mirrors quantum annealing's landscape search.

    Optimizes 8 threshold parameters (4 features × 2 levels)
    to minimize combined false positive + false negative rate
    using historical ground truth from Tier 4 buffer.
    """

    def __init__(self, max_iterations=500):
        self.max_iter = max_iterations

    def optimize(self, events_history, current_thresholds):
        """
        Find optimal thresholds using Nelder-Mead on historical data.
        events_history: list of events with ground truth fields.
        current_thresholds: dict of current {feature: {warn, crit}}.
        Returns: optimized thresholds and improvement metrics.
        """
        if len(events_history) < 20:
            return {
                "optimized": False,
                "reason": "insufficient_history",
                "n_events": len(events_history),
                "method": "nelder_mead",
            }

        # Build ground truth: machine needs attention if any DT flag is set
        gt = {}
        for evt in events_history:
            mid = evt.get("machine_id")
            needs = (evt.get("machine_status", 0) == 1
                     or evt.get("failure_type", "Normal") != "Normal"
                     or evt.get("anomaly_flag", 0) == 1
                     or evt.get("maintenance_required", 0) == 1)
            gt[mid] = needs

        # Flatten thresholds to 8-dimensional vector
        features_order = list(current_thresholds.keys())
        initial = []
        for feat in features_order:
            initial.append(current_thresholds[feat]["warn"])
            initial.append(current_thresholds[feat]["crit"])

        d = len(initial)

        def objective(params):
            """Compute FP + FN rate for given threshold params."""
            # Reconstruct thresholds
            thr = {}
            for i, feat in enumerate(features_order):
                w = params[i * 2]
                c = params[i * 2 + 1]
                # Enforce warn < crit
                if w >= c:
                    c = w + 1.0
                thr[feat] = {"warn": w, "crit": c}

            # AUDIT NOTE: previous predictor used anomaly_flag /
            # machine_status / failure_type as features — those are
            # ground truth. The objective function below now scores
            # using ONLY the threshold-based component, which is what
            # we are actually trying to tune.
            fp = fn = tp = tn = 0
            for evt in events_history:
                mid = evt.get("machine_id")
                actual = gt.get(mid, False)

                score = 0.0
                for feat, t in thr.items():
                    val = evt.get(feat, 0)
                    if not isinstance(val, (int, float)):
                        continue
                    if val >= t["crit"]:
                        score += 0.35
                    elif val >= t["warn"]:
                        score += 0.15

                predicted = score >= 0.3  # MEDIUM threshold

                if predicted and actual:
                    tp += 1
                elif predicted and not actual:
                    fp += 1
                elif not predicted and actual:
                    fn += 1
                else:
                    tn += 1

            total = tp + fp + fn + tn
            if total == 0:
                return 1.0
            # Weighted: FN is 2× worse than FP (missing a failure is worse)
            fp_rate = fp / total
            fn_rate = fn / total
            return fp_rate + 2.0 * fn_rate

        # Nelder-Mead simplex
        best_params, best_cost = self._nelder_mead(objective, initial, d)

        # Reconstruct optimized thresholds
        optimized = {}
        for i, feat in enumerate(features_order):
            w = round(best_params[i * 2], 2)
            c = round(best_params[i * 2 + 1], 2)
            if w >= c:
                c = w + 1.0
            optimized[feat] = {"warn": w, "crit": c}

        # Compute improvement
        initial_cost = objective(initial)
        improvement = ((initial_cost - best_cost) / initial_cost * 100) if initial_cost > 0 else 0

        return {
            "optimized": True,
            "thresholds": optimized,
            "initial_cost": round(initial_cost, 4),
            "optimized_cost": round(best_cost, 4),
            "improvement_pct": round(improvement, 2),
            "n_events_used": len(events_history),
            "iterations": self.max_iter,
            "method": "nelder_mead",
        }

    def _nelder_mead(self, func, x0, d, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
        """Simplified Nelder-Mead for d dimensions."""
        # Initialize simplex
        simplex = [x0[:]]
        step = [max(0.5, abs(x) * 0.1) for x in x0]
        for i in range(d):
            point = x0[:]
            point[i] += step[i]
            simplex.append(point)

        for iteration in range(self.max_iter):
            # Evaluate and sort
            scored = [(func(s), s) for s in simplex]
            scored.sort(key=lambda x: x[0])
            simplex = [s[1] for s in scored]
            costs = [s[0] for s in scored]

            best_cost = costs[0]
            worst_cost = costs[-1]
            second_worst = costs[-2]

            # Convergence check
            if abs(worst_cost - best_cost) < 1e-8:
                break

            # Centroid (excluding worst)
            centroid = [0.0] * d
            for i in range(d):
                for s in simplex[:-1]:
                    centroid[i] += s[i]
                centroid[i] /= d

            # Reflection
            worst = simplex[-1]
            reflected = [centroid[i] + alpha * (centroid[i] - worst[i]) for i in range(d)]
            f_reflected = func(reflected)

            if f_reflected < second_worst and f_reflected >= best_cost:
                simplex[-1] = reflected
            elif f_reflected < best_cost:
                # Expansion
                expanded = [centroid[i] + gamma * (reflected[i] - centroid[i]) for i in range(d)]
                f_expanded = func(expanded)
                simplex[-1] = expanded if f_expanded < f_reflected else reflected
            else:
                # Contraction
                contracted = [centroid[i] + rho * (worst[i] - centroid[i]) for i in range(d)]
                f_contracted = func(contracted)
                if f_contracted < worst_cost:
                    simplex[-1] = contracted
                else:
                    # Shrink
                    best_point = simplex[0]
                    for i in range(1, len(simplex)):
                        simplex[i] = [best_point[j] + sigma * (simplex[i][j] - best_point[j])
                                      for j in range(d)]

        scored = [(func(s), s) for s in simplex]
        scored.sort(key=lambda x: x[0])
        return scored[0][1], scored[0][0]


# ═══════════════════════════════════════════════
# UNIFIED INTERFACE
# ═══════════════════════════════════════════════

class ClassicalOptimizationEngine:
    """
    Unified interface for all 5 classical optimization algorithms.
    Drop-in replacement path: swap each method with its quantum
    counterpart when quantum hardware becomes available.

    Usage in pipeline:
        engine = ClassicalOptimizationEngine()

        # Per-event (Point 1): alongside SpikeDetector
        engine.multivariate_detector.update(mid, event)
        result = engine.multivariate_detector.detect(mid, event)

        # Micro-batch (Point 4): after LLM agents
        consensus = engine.validate_consensus(monitoring, diagnosis, planning, reasoning)

        # Daily (Point 2): Correlation Agent enhancement
        correlation = engine.analyze_graph(events, reasonings)

        # Daily (Point 3): Scheduling optimization
        schedule = engine.optimize_schedule(machines_data)

        # Daily (Point 5): Learning Loop threshold tuning
        new_thresholds = engine.optimize_thresholds(history, current_thresholds)
    """

    def __init__(self):
        self.multivariate_detector = MultivariateAnomalyDetector()
        self.graph_engine = GraphCorrelationEngine(n_iterations=3)
        self.scheduler = SchedulingOptimizer(max_simultaneous=5)
        self.consensus = ConsensusValidator()
        self.threshold_opt = ThresholdOptimizer(max_iterations=500)

    def detect_multivariate_anomaly(self, machine_id, event):
        """Point 1: multivariate anomaly detection (→ future VQC)."""
        self.multivariate_detector.update(machine_id, event)
        return self.multivariate_detector.detect(machine_id, event)

    def analyze_graph(self, events, reasonings):
        """Point 2: graph correlation analysis (→ future QGNN)."""
        return self.graph_engine.analyze(events, reasonings)

    def optimize_schedule(self, machines_data):
        """Point 3: maintenance scheduling (→ future QAOA)."""
        return self.scheduler.optimize(machines_data)

    def validate_consensus(self, monitoring, diagnosis, planning, reasoning):
        """Point 4: consensus validation (→ future Quantum Kernel)."""
        return self.consensus.validate(monitoring, diagnosis, planning, reasoning)

    def optimize_thresholds(self, events_history, current_thresholds):
        """Point 5: threshold optimization (→ future Quantum Annealing)."""
        return self.threshold_opt.optimize(events_history, current_thresholds)


# ─────────────────────────────────────────────
# Import datetime for scheduling
# ─────────────────────────────────────────────
import datetime
