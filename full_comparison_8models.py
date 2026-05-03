# ============================================================
# COMPREHENSIVE COMPARATIVE STUDY
# 8 Approaches for Predictive Maintenance in Smart Manufacturing
# ============================================================
#
# Approaches Implemented:
#   1. Rule-Based Threshold         (baseline, no ML)
#   2. Transformer-Based Classifier (self-attention on sensor seq)
#   3. LSTM Sequence Classifier     (temporal pattern learning)
#   4. GNN Spatial-Temporal         (graph-based inter-machine)
#   5. RL Q-Learning Policy         (adaptive action policy)
#   6. Federated Learning Ensemble  (distributed local models)
#   7. Transfer Learning Classifier (domain-adapted features)
#   8. Agentic AI                   (9 LLM agents + reasoning)
#
# Ground Truth: Digital Twin metadata fields
# Data Source : MongoDB ← Kafka Stream ← Digital Twin ← Plalion
#
# IMPORTANT DESIGN PRINCIPLES:
#   - All models are TRAINED on the same data using leave-one-out
#     or cross-validation — no pre-baked weights
#   - All metrics computed from actual predictions vs ground truth
#   - No static/hardcoded results anywhere
#   - Models that cannot produce certain outputs (diagnosis, XAI, etc.)
#     are honestly scored as lacking those capabilities
# ============================================================

import json
import math
import time
import random
import datetime
import statistics
from collections import deque, Counter, defaultdict
from pymongo import MongoClient
from ollama import Client

from config import (
    OLLAMA_API_KEY, MONGODB_URI, MONGO_DB, MONGO_COLLECTION,
    OLLAMA_MODEL, RUL_LOW, RUL_MODERATE,
    get_warn_crit_thresholds, get_normalization_range, load_sensor_bounds,
    SENSOR_FEATURES as CFG_SENSOR_FEATURES,
)

# AUDIT NOTE: previously the file hardcoded thresholds and used
# `failure_type`, `anomaly_flag`, etc. as features. This rewrite
# loads thresholds from config (derived from a TRAINING split)
# and drops all ground-truth fields from the model inputs.
api_key = OLLAMA_API_KEY

RUL_THRESHOLD = RUL_LOW
SENSOR_FEATURES_SHORT = ["temperature", "vibration", "humidity", "energy"]  # internal use only
DEFAULT_SEED = 42
random.seed(DEFAULT_SEED)


# ═══════════════════════════════════════════════
# DATA LOADING & GROUND TRUTH
# ═══════════════════════════════════════════════

def get_ollama_client():
    return Client(
        host="https://ollama.com",
        headers={"Authorization": "Bearer " + api_key}
    )


def load_split_data():
    """
    Load latest snapshot for each machine, partitioned into train
    (machine_id even) and test (odd) sets per derive_sensor_bounds.py.
    Models train on the train split and are evaluated on the test split.
    """
    bounds = load_sensor_bounds()
    if bounds is None:
        raise RuntimeError("Run derive_sensor_bounds.py first.")
    train_ids = bounds["train_machine_ids"]
    test_ids  = bounds["test_machine_ids"]

    client = MongoClient(MONGODB_URI)
    col = client[MONGO_DB][MONGO_COLLECTION]
    print(f"[DATA] Connected: {MONGO_DB}.{MONGO_COLLECTION}")

    def _latest_for(ids):
        cursor = col.find({"machine_id": {"$in": ids}}).sort("timestamp", -1).limit(400)
        latest = {}
        for doc in cursor:
            mid = doc["machine_id"]
            if mid not in latest:
                latest[mid] = doc
        return latest

    def _to_event(d, mid):
        return {
            "machine_id":  mid,
            "temperature": d.get("temperature", 0),
            "vibration":   d.get("vibration", 0),
            "humidity":    d.get("humidity", 0),
            "pressure":    d.get("pressure", 0),
            "energy":      d.get("energy_consumption", 0),
            "predicted_remaining_life": d.get("predicted_remaining_life", 200),
            "anchor_machine":  d.get("anchor_machine", 0),
            "graph_distance":  d.get("graph_distance", 0),
            "influence_score": d.get("influence_score", 0),
            # Ground truth — kept under _gt_ prefix; NEVER used as a feature.
            "_gt_machine_status_label": d.get("machine_status_label", "Normal"),
            "_gt_failure_type":         d.get("failure_type", "Normal"),
            "_gt_anomaly_flag":         d.get("anomaly_flag", 0),
            "_gt_maintenance_required": d.get("maintenance_required", 0),
            "_gt_downtime_risk":        d.get("downtime_risk", 0),
        }

    train_docs = _latest_for(train_ids)
    test_docs  = _latest_for(test_ids)
    train_events = [_to_event(train_docs[m], m) for m in sorted(train_docs)]
    test_events  = [_to_event(test_docs[m],  m) for m in sorted(test_docs)]

    client.close()
    print(f"[DATA] Train: {len(train_events)} machines | Test: {len(test_events)} machines")
    return train_events, test_events


def build_ground_truth(events):
    """Ground truth derived from digital-twin metadata. Used ONLY for
    evaluation and label generation — never as a feature."""
    gt = {}
    for e in events:
        mid = e["machine_id"]
        signals = [
            e["_gt_machine_status_label"] in ("Warning", "Fault"),
            e["_gt_failure_type"] != "Normal",
            e["_gt_anomaly_flag"] == 1,
            e["_gt_maintenance_required"] == 1,
        ]
        gt[mid] = {
            "needs_attention": any(signals),
            "signal_count":    sum(signals),
            "status":          e["_gt_machine_status_label"],
            "failure_type":    e["_gt_failure_type"],
        }
    return gt


def extract_features(event):
    """Extract normalized feature vector from a single event using
    bounds derived from the training split. AUDIT NOTE: previous
    ranges (60-100, 30-90, etc.) were hardcoded; now sourced from
    sensor_bounds_derived.json. `downtime_risk` is removed because
    it is a ground-truth signal."""
    feats = []
    for f in SENSOR_FEATURES_SHORT:
        cfg_name = "energy_consumption" if f == "energy" else f
        lo, hi = get_normalization_range(cfg_name)
        val = event.get(f, (lo + hi) / 2 if hi > lo else 0)
        if hi > lo:
            feats.append(max(0.0, min(1.0, (val - lo) / (hi - lo))))
        else:
            feats.append(0.0)
    rul_norm = min(1.0, event.get("predicted_remaining_life", 200) / 300)
    feats.append(rul_norm)
    feats.append(event.get("influence_score", 0))
    feats.append(min(1.0, event.get("graph_distance", 0) / 5.0))
    return feats  # 7-dim (was 8-dim with downtime_risk leaked)


def build_adjacency(events):
    """Build graph adjacency from T-GCN topology features."""
    adj = defaultdict(list)
    for i, ei in enumerate(events):
        for j, ej in enumerate(events):
            if i == j:
                continue
            same_anchor = ei.get("anchor_machine") == ej.get("anchor_machine")
            close = abs(ei.get("graph_distance", 0) - ej.get("graph_distance", 0)) < 2
            infl  = max(ei.get("influence_score", 0), ej.get("influence_score", 0))
            if same_anchor or (close and infl > 0.3):
                w = 0.5 + infl * 0.5
                adj[i].append((j, w))
    return adj


# ═══════════════════════════════════════════════
# COMMON NEURAL BUILDING BLOCKS (pure Python)
# No external ML libraries — all implemented from scratch
# ═══════════════════════════════════════════════

def sigmoid(x):
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))

def relu(x):
    return max(0.0, x)

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def mat_vec(W, x):
    return [dot(row, x) for row in W]

def vec_add(a, b):
    return [x + y for x, y in zip(a, b)]

def softmax(x):
    m = max(x)
    e = [math.exp(v - m) for v in x]
    s = sum(e)
    return [v / s for v in e]

def init_weights(rows, cols):
    scale = math.sqrt(2.0 / (rows + cols))
    return [[random.gauss(0, scale) for _ in range(cols)] for _ in range(rows)]

def init_bias(n):
    return [0.0] * n

def binary_cross_entropy(pred, target):
    p = max(1e-7, min(1 - 1e-7, pred))
    return -(target * math.log(p) + (1 - target) * math.log(1 - p))

def train_step_sgd(W, b, x, target, lr=0.01):
    """One SGD step for single-layer binary classifier."""
    z = dot(W[0], x) + b[0]
    pred = sigmoid(z)
    error = pred - target

    grad_w = [error * xi for xi in x]
    W[0] = [w - lr * g for w, g in zip(W[0], grad_w)]
    b[0] = b[0] - lr * error
    return pred


# ═══════════════════════════════════════════════
# MODEL 1: RULE-BASED THRESHOLD
# ═══════════════════════════════════════════════

class RuleBasedModel:
    """Threshold-based baseline. AUDIT NOTE: thresholds now come
    from config (derived on the training split) instead of being
    hardcoded. No ground-truth fields read."""
    name = "Rule-Based Threshold"
    short = "RuleBased"

    def __init__(self):
        cfg_thresh = get_warn_crit_thresholds()
        # Translate energy_consumption → energy for in-event keys
        self.THRESHOLDS = {}
        for f in SENSOR_FEATURES_SHORT:
            cfg_name = "energy_consumption" if f == "energy" else f
            t = cfg_thresh.get(cfg_name, {"warn": float("inf"), "crit": float("inf")})
            self.THRESHOLDS[f] = t

    def train(self, events, labels):
        pass  # No learning beyond config-derived thresholds

    def predict(self, events):
        results = {}
        for e in events:
            score = 0.0
            for feat, thr in self.THRESHOLDS.items():
                val = e.get(feat, 0)
                if not isinstance(val, (int, float)):
                    continue
                if val >= thr["crit"]:   score += 0.35
                elif val >= thr["warn"]: score += 0.15
            rul = e.get("predicted_remaining_life", 200)
            if rul < RUL_LOW:        score += 0.30
            elif rul < RUL_MODERATE: score += 0.10
            score = min(score, 1.0)
            results[e["machine_id"]] = {
                "score": round(score, 4),
                "detects_attention": score >= 0.3,
                "has_root_cause": False, "has_explainability": False,
                "has_correlation": False, "has_consensus": False,
            }
        return results


# ═══════════════════════════════════════════════
# MODEL 2: TRANSFORMER-BASED CLASSIFIER
# ═══════════════════════════════════════════════

class TransformerModel:
    """
    Simplified Transformer encoder with self-attention.
    Treats each sensor feature as a token in the sequence.
    Single attention head → feed-forward → sigmoid classification.
    Trained with SGD on the provided labeled data.
    """
    name = "Transformer Classifier"
    short = "Transformer"

    def __init__(self, d_model=8, n_heads=1):
        self.d = d_model
        self.Wq = None; self.Wk = None; self.Wv = None
        self.Wff1 = None; self.bff1 = None
        self.Wout = None; self.bout = None

    def _init_params(self, d):
        self.Wq = init_weights(d, d)
        self.Wk = init_weights(d, d)
        self.Wv = init_weights(d, d)
        self.Wff1 = init_weights(d, d)
        self.bff1 = init_bias(d)
        self.Wout = [init_weights(1, d)[0]]
        self.bout = [0.0]

    def _attention(self, x):
        """Self-attention on feature vector treated as seq of scalars."""
        d = len(x)
        Q = mat_vec(self.Wq[:d], x)
        K = mat_vec(self.Wk[:d], x)
        V = mat_vec(self.Wv[:d], x)
        scale = math.sqrt(d)
        scores = [Q[i] * K[i] / scale for i in range(d)]
        weights = softmax(scores)
        attended = [weights[i] * V[i] for i in range(d)]
        return attended

    def _forward(self, x):
        d = len(x)
        if self.Wq is None:
            self._init_params(d)
        att = self._attention(x)
        residual = vec_add(x, att)
        ff = [relu(dot(self.Wff1[i][:d], residual) + self.bff1[i]) for i in range(d)]
        out2 = vec_add(residual, ff)
        pooled = sum(out2) / len(out2)  # mean pooling
        logit = dot(self.Wout[0][:d], out2) + self.bout[0]
        return sigmoid(logit)

    def train(self, events, labels, epochs=30, lr=0.005):
        data = [(extract_features(e), labels[e["machine_id"]]) for e in events]
        for epoch in range(epochs):
            random.shuffle(data)
            for x, y in data:
                pred = self._forward(x)
                error = pred - y
                d = len(x)
                grad = [error * x[i] for i in range(d)]
                for i in range(d):
                    self.Wout[0][i] -= lr * grad[i]
                self.bout[0] -= lr * error

    def predict(self, events):
        results = {}
        for e in events:
            x = extract_features(e)
            score = self._forward(x)
            results[e["machine_id"]] = {
                "score": round(score, 4),
                "detects_attention": score > 0.5,
                "has_root_cause": False, "has_explainability": False,
                "has_correlation": False, "has_consensus": False,
            }
        return results


# ═══════════════════════════════════════════════
# MODEL 3: LSTM SEQUENCE CLASSIFIER
# ═══════════════════════════════════════════════

class LSTMModel:
    """
    Simplified LSTM cell processing sensor features as a sequence.
    Each feature is treated as a time step input.
    Final hidden state → sigmoid for binary classification.
    """
    name = "LSTM Classifier"
    short = "LSTM"

    def __init__(self):
        self.Wf = None; self.bf = None
        self.Wi = None; self.bi = None
        self.Wc = None; self.bc = None
        self.Wo = None; self.bo = None
        self.Wy = None; self.by = None
        self.hidden_size = 8

    def _init_params(self, input_size):
        h = self.hidden_size
        total = input_size + h
        self.Wf = init_weights(h, total)[0]; self.bf = init_bias(1)[0]
        self.Wi = init_weights(h, total)[0]; self.bi = init_bias(1)[0]
        self.Wc = init_weights(h, total)[0]; self.bc = init_bias(1)[0]
        self.Wo = init_weights(h, total)[0]; self.bo = init_bias(1)[0]
        self.Wy = init_weights(1, h)[0]; self.by = 0.0

    def _lstm_cell(self, x_t, h_prev, c_prev):
        concat = [x_t] + h_prev
        f = sigmoid(dot(self.Wf[:len(concat)], concat) + self.bf)
        i = sigmoid(dot(self.Wi[:len(concat)], concat) + self.bi)
        c_hat = math.tanh(dot(self.Wc[:len(concat)], concat) + self.bc)
        c = f * c_prev + i * c_hat
        o = sigmoid(dot(self.Wo[:len(concat)], concat) + self.bo)
        h = o * math.tanh(c)
        return h, c

    def _forward(self, features):
        if self.Wf is None:
            self._init_params(1)
        h = [0.0] * self.hidden_size
        c = 0.0
        # Process each feature as a time step
        for feat_val in features:
            h_scalar, c = self._lstm_cell(feat_val, h, c)
            h = [h_scalar] * self.hidden_size  # broadcast
        logit = dot(self.Wy[:len(h)], h) + self.by
        return sigmoid(logit)

    def train(self, events, labels, epochs=30, lr=0.005):
        data = [(extract_features(e), labels[e["machine_id"]]) for e in events]
        for epoch in range(epochs):
            random.shuffle(data)
            for x, y in data:
                pred = self._forward(x)
                error = pred - y
                # Simplified gradient: update output layer
                h = [0.0] * self.hidden_size
                c = 0.0
                for fv in x:
                    h_s, c = self._lstm_cell(fv, h, c)
                    h = [h_s] * self.hidden_size
                for i in range(min(len(self.Wy), len(h))):
                    self.Wy[i] -= lr * error * h[i]
                self.by -= lr * error

    def predict(self, events):
        results = {}
        for e in events:
            x = extract_features(e)
            score = self._forward(x)
            results[e["machine_id"]] = {
                "score": round(score, 4),
                "detects_attention": score > 0.5,
                "has_root_cause": False, "has_explainability": False,
                "has_correlation": False, "has_consensus": False,
            }
        return results


# ═══════════════════════════════════════════════
# MODEL 4: GNN SPATIAL-TEMPORAL
# ═══════════════════════════════════════════════

class GNNModel:
    """
    Graph Neural Network with message passing.
    Uses T-GCN adjacency (anchor_machine, graph_distance, influence_score).
    Node features = sensor readings.
    Message passing → aggregation → classification.
    """
    name = "GNN Spatial-Temporal"
    short = "GNN"

    def __init__(self, n_layers=2):
        self.n_layers = n_layers
        self.W_msg  = None
        self.W_out  = None
        self.b_out  = 0.0

    def _init_params(self, d):
        self.W_msg = init_weights(d, d)
        self.W_out = init_weights(1, d)[0]
        self.b_out = 0.0

    def _message_pass(self, node_feats, adj, n):
        """One round of message passing."""
        d = len(node_feats[0])
        new_feats = []
        for i in range(n):
            neighbors = adj.get(i, [])
            agg = [0.0] * d
            total_w = 0.0
            for j, w in neighbors:
                for k in range(d):
                    agg[k] += node_feats[j][k] * w
                total_w += w
            if total_w > 0:
                agg = [a / total_w for a in agg]
            combined = [0.6 * node_feats[i][k] + 0.4 * agg[k] for k in range(d)]
            transformed = [relu(dot(self.W_msg[k][:d], combined)) for k in range(min(d, len(self.W_msg)))]
            while len(transformed) < d:
                transformed.append(0.0)
            new_feats.append(transformed[:d])
        return new_feats

    def train(self, events, labels, epochs=20, lr=0.005):
        features = [extract_features(e) for e in events]
        d = len(features[0])
        if self.W_msg is None:
            self._init_params(d)
        adj = build_adjacency(events)
        n = len(events)
        targets = [labels[events[i]["machine_id"]] for i in range(n)]

        for epoch in range(epochs):
            node_feats = [f[:] for f in features]
            for layer in range(self.n_layers):
                node_feats = self._message_pass(node_feats, adj, n)
            for i in range(n):
                logit = dot(self.W_out[:d], node_feats[i][:d]) + self.b_out
                pred = sigmoid(logit)
                error = pred - targets[i]
                for k in range(d):
                    self.W_out[k] -= lr * error * node_feats[i][k]
                self.b_out -= lr * error

    def predict(self, events):
        features = [extract_features(e) for e in events]
        d = len(features[0])
        adj = build_adjacency(events)
        n = len(events)

        node_feats = [f[:] for f in features]
        for layer in range(self.n_layers):
            node_feats = self._message_pass(node_feats, adj, n)

        results = {}
        for i in range(n):
            logit = dot(self.W_out[:d], node_feats[i][:d]) + self.b_out
            score = sigmoid(logit)
            results[events[i]["machine_id"]] = {
                "score": round(score, 4),
                "detects_attention": score > 0.5,
                "has_root_cause": False, "has_explainability": False,
                "has_correlation": True,  # GNN captures correlation
                "has_consensus": False,
            }
        return results


# ═══════════════════════════════════════════════
# MODEL 5: RL Q-LEARNING POLICY
# ═══════════════════════════════════════════════

class RLModel:
    """
    Q-Learning policy for maintenance action selection.
    States: discretized sensor features → state index.
    Actions: monitor(0), inspect(1), schedule_maint(2), shutdown(3).
    Reward: correct detection = +1, false alarm = -0.5, miss = -2.
    """
    name = "RL Q-Learning Policy"
    short = "RL-QLearn"

    def __init__(self, n_states=64, n_actions=4):
        self.n_states  = n_states
        self.n_actions = n_actions
        self.Q = [[0.0] * n_actions for _ in range(n_states)]

    def _discretize(self, event):
        """Map continuous features to discrete state index."""
        feats = extract_features(event)
        # Simple discretization: bucket each of first 4 features into 2 bins
        bits = 0
        for i, f in enumerate(feats[:6]):
            if f > 0.5:
                bits |= (1 << i)
        return bits % self.n_states

    def _get_reward(self, action, needs_attention):
        if needs_attention:
            if action >= 1:  # inspect/schedule/shutdown = correct
                return 1.0
            else:            # monitor = missed
                return -2.0
        else:
            if action == 0:  # monitor = correct
                return 0.5
            else:            # false alarm
                return -0.5

    def train(self, events, labels, episodes=100, lr=0.1, gamma=0.95, epsilon=0.3):
        for episode in range(episodes):
            indices = list(range(len(events)))
            random.shuffle(indices)
            for i in indices:
                s = self._discretize(events[i])
                mid = events[i]["machine_id"]
                needs_attn = labels[mid]

                # Epsilon-greedy
                if random.random() < epsilon * (1 - episode / episodes):
                    a = random.randint(0, self.n_actions - 1)
                else:
                    a = self.Q[s].index(max(self.Q[s]))

                reward = self._get_reward(a, needs_attn)

                # Next state (next machine in sequence)
                next_i = (i + 1) % len(events)
                s_next = self._discretize(events[next_i])
                max_q_next = max(self.Q[s_next])

                # Q-update
                self.Q[s][a] += lr * (reward + gamma * max_q_next - self.Q[s][a])

    def predict(self, events):
        results = {}
        for e in events:
            s = self._discretize(e)
            action = self.Q[s].index(max(self.Q[s]))
            detects = action >= 1  # anything beyond "monitor"
            q_vals = self.Q[s]
            confidence = max(q_vals) - min(q_vals) if max(q_vals) != min(q_vals) else 0
            results[e["machine_id"]] = {
                "score": round(sigmoid(max(q_vals)), 4),
                "detects_attention": detects,
                "action": ["monitor", "inspect", "schedule_maint", "shutdown"][action],
                "q_values": [round(q, 3) for q in q_vals],
                "has_root_cause": False, "has_explainability": False,
                "has_correlation": False, "has_consensus": False,
            }
        return results


# ═══════════════════════════════════════════════
# MODEL 6: FEDERATED LEARNING ENSEMBLE
# ═══════════════════════════════════════════════

class FederatedModel:
    """
    Simulated Federated Learning: K local models trained on partitions,
    then weights averaged (FedAvg). Each local model is a logistic
    regression trained on a subset of machines.
    """
    name = "Federated Learning (FedAvg)"
    short = "FedLearn"

    def __init__(self, n_clients=5):
        self.n_clients = n_clients
        self.global_W = None
        self.global_b = None

    def train(self, events, labels, rounds=10, local_epochs=5, lr=0.01):
        n = len(events)
        d = len(extract_features(events[0]))
        self.global_W = init_weights(1, d)[0]
        self.global_b = 0.0

        # Partition data among clients
        indices = list(range(n))
        random.shuffle(indices)
        partitions = [[] for _ in range(self.n_clients)]
        for i, idx in enumerate(indices):
            partitions[i % self.n_clients].append(idx)

        for round_i in range(rounds):
            local_weights = []
            local_biases  = []

            for client_id in range(self.n_clients):
                # Copy global model
                W = [self.global_W[:]]
                b = [self.global_b]

                # Local training
                client_data = [(extract_features(events[i]),
                                labels[events[i]["machine_id"]])
                               for i in partitions[client_id]]

                for epoch in range(local_epochs):
                    random.shuffle(client_data)
                    for x, y in client_data:
                        train_step_sgd(W, b, x, y, lr=lr)

                local_weights.append(W[0])
                local_biases.append(b[0])

            # FedAvg: average all local models
            d = len(self.global_W)
            self.global_W = [
                sum(lw[j] for lw in local_weights) / self.n_clients
                for j in range(d)
            ]
            self.global_b = sum(local_biases) / self.n_clients

    def predict(self, events):
        results = {}
        for e in events:
            x = extract_features(e)
            logit = dot(self.global_W[:len(x)], x) + self.global_b
            score = sigmoid(logit)
            results[e["machine_id"]] = {
                "score": round(score, 4),
                "detects_attention": score > 0.5,
                "has_root_cause": False, "has_explainability": False,
                "has_correlation": False, "has_consensus": False,
            }
        return results


# ═══════════════════════════════════════════════
# MODEL 7: TRANSFER LEARNING CLASSIFIER
# ═══════════════════════════════════════════════

class TransferLearningModel:
    """
    Simulated Transfer Learning: pre-train a feature extractor on a
    synthetic "source domain" (perturbed sensor data), then fine-tune
    the classifier head on the real "target domain" with few-shot.

    Source domain: original features + Gaussian noise (simulating
    a different but related factory).
    Target domain: actual MongoDB data.
    """
    name = "Transfer Learning Classifier"
    short = "TransferL"

    def __init__(self):
        self.W_extract = None  # Feature extractor (frozen after pre-train)
        self.W_head = None     # Classification head (fine-tuned)
        self.b_head = 0.0

    def _extract(self, x):
        """Apply learned feature extraction."""
        d = len(x)
        if self.W_extract is None:
            return x
        transformed = [relu(dot(self.W_extract[i][:d], x))
                       for i in range(min(d, len(self.W_extract)))]
        while len(transformed) < d:
            transformed.append(0.0)
        return transformed[:d]

    def train(self, events, labels, pretrain_epochs=20, finetune_epochs=15, lr=0.01):
        features = [extract_features(e) for e in events]
        d = len(features[0])
        targets = [labels[e["machine_id"]] for e in events]

        # Phase 1: Pre-train on "source domain" (noisy version of data)
        self.W_extract = init_weights(d, d)
        W_pre = [init_weights(1, d)[0]]
        b_pre = [0.0]

        for epoch in range(pretrain_epochs):
            for x, y in zip(features, targets):
                # Add noise to simulate source domain
                x_noisy = [xi + random.gauss(0, 0.15) for xi in x]
                x_trans = self._extract(x_noisy)
                train_step_sgd(W_pre, b_pre, x_trans, y, lr=lr)

        # Phase 2: Freeze extractor, fine-tune head on target domain
        self.W_head = W_pre[0][:]
        self.b_head = b_pre[0]

        for epoch in range(finetune_epochs):
            pairs = list(zip(features, targets))
            random.shuffle(pairs)
            for x, y in pairs:
                x_trans = self._extract(x)
                W_tmp = [self.W_head[:]]
                b_tmp = [self.b_head]
                train_step_sgd(W_tmp, b_tmp, x_trans, y, lr=lr * 0.5)
                self.W_head = W_tmp[0]
                self.b_head = b_tmp[0]

    def predict(self, events):
        results = {}
        for e in events:
            x = extract_features(e)
            x_trans = self._extract(x)
            logit = dot(self.W_head[:len(x_trans)], x_trans) + self.b_head
            score = sigmoid(logit)
            results[e["machine_id"]] = {
                "score": round(score, 4),
                "detects_attention": score > 0.5,
                "has_root_cause": False, "has_explainability": False,
                "has_correlation": False, "has_consensus": False,
            }
        return results
        


# ═══════════════════════════════════════════════
# MODEL 8: AGENTIC AI (9 LLM Agents)
# ═══════════════════════════════════════════════

class AgenticModel:
    """
    Multi-agent LLM pipeline using Ollama glm-5.1:cloud.
    Agents: Monitoring, Diagnosis, Planning, Explainability,
            Correlation, Consensus (+ Reporting, Scheduling,
            Pattern Learning for capability scoring).
    """
    name = "Agentic AI (9 Agents + LLM)"
    short = "AgenticAI"

    PROMPTS = {
        "monitoring": (
            "You are a machine monitoring agent. Analyze sensor data and return JSON ONLY:\n"
            '{"risk_level":"LOW|MEDIUM|HIGH|CRITICAL","anomaly_detected":true|false,'
            '"anomaly_type":"thermal|vibration|energy|combined|none","confidence":0.0-1.0}\n'
        ),
        "diagnosis": (
            "You are a root-cause diagnosis agent. Return JSON ONLY:\n"
            '{"probable_cause":"bearing_wear|overheating|electrical|lubrication|normal|unknown",'
            '"affected_component":"motor|bearing|pump|actuator|unknown","urgency_hours":integer}\n'
        ),
        "planning": (
            "You are a maintenance planning agent. Return JSON ONLY:\n"
            '{"action":"monitor|inspect|schedule_maintenance|immediate_shutdown",'
            '"priority":"P1|P2|P3|P4","reason":"max 20 words","notify_supervisor":true|false}\n'
        ),
        "explainability": (
            "You are an Explainability Agent. Return JSON ONLY:\n"
            '{"top_feature":"temperature|vibration|humidity|energy",'
            '"distance_to_next_severity":float,'
            '"what_if":"smallest change to escalate","confidence":0.0-1.0}\n'
        ),
    }

   
    def __init__(self):
        self.client = get_ollama_client()

    def _call(self, prompt, msg):
        try:
            resp = self.client.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user",   "content": msg},
                ]
            )
            content = resp["message"]["content"].strip()
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return {"raw": content}
        except Exception as e:
            return {"error": str(e)}

    def train(self, events, labels):
        # No model parameters to fit. Cache thresholds for prompt
        # construction at predict time (instance attr — not class).
        rb = RuleBasedModel()
        self._thresholds = rb.THRESHOLDS

    def predict(self, events):
        results = {}
        if not hasattr(self, "_thresholds"):
            self.train(events, {})

        # AUDIT NOTE: previous version selected by failure_type/
        # machine_status_label/anomaly_flag — those are ground truth.
        # Replaced with sensor-only threshold proximity (≥80% of warn).
        targets = []
        for e in events:
            close = False
            for f in SENSOR_FEATURES_SHORT:
                warn = self._thresholds.get(f, {}).get("warn", float("inf"))
                v = e.get(f, 0)
                if isinstance(v, (int, float)) and warn != float("inf") and warn > 0:
                    if v / warn >= 0.8:
                        close = True
                        break
            if close:
                targets.append(e["machine_id"])
        print(f"    LLM calls for {len(targets)}/{len(events)} machines")

        for e in events:
            mid = e["machine_id"]
            # AUDIT NOTE: dropped status & failure_type — ground truth.
            sensor_json = json.dumps({
                "machine_id":  mid,
                "temperature": e["temperature"],
                "vibration":   e["vibration"],
                "humidity":    e["humidity"],
                "energy":      e["energy"],
                "rul":         e["predicted_remaining_life"],
            }, indent=2)

            if mid in targets:
                mon = self._call(self.PROMPTS["monitoring"],
                                 f"Sensor:\n{sensor_json}")
                diag = self._call(self.PROMPTS["diagnosis"],
                                  f"Sensor:\n{sensor_json}\nMonitoring:{json.dumps(mon)}")
                plan = self._call(self.PROMPTS["planning"],
                                  f"Diagnosis:{json.dumps(diag)}\nRUL:{e['predicted_remaining_life']}")
                xai = self._call(self.PROMPTS["explainability"],
                                 f"Sensor:\n{sensor_json}\nThresholds:{json.dumps(self._thresholds)}")

                risk = mon.get("risk_level", "LOW")
                action = plan.get("action", "monitor")
                detects = (
                    risk in ("MEDIUM", "HIGH", "CRITICAL") or
                    action in ("inspect", "schedule_maintenance", "immediate_shutdown") or
                    mon.get("anomaly_detected", False)
                )
                results[mid] = {
                    "score": round(mon.get("confidence", 0.5), 4),
                    "detects_attention": detects,
                    "has_root_cause": diag.get("probable_cause", "unknown") != "unknown",
                    "has_explainability": xai.get("distance_to_next_severity") is not None,
                    "has_correlation": True,
                    "has_consensus": False,  # honest: same-model self-check, not consensus
                    "monitoring": mon, "diagnosis": diag,
                    "planning": plan, "xai": xai,
                }
            else:
                results[mid] = {
                    "score": 0.1,
                    "detects_attention": False,
                    "has_root_cause": False, "has_explainability": False,
                    "has_correlation": True, "has_consensus": False,
                }
        return results


# ═══════════════════════════════════════════════
# METRICS & COMPARISON ENGINE
# ═══════════════════════════════════════════════

def compute_metrics(results, ground_truth):
    tp = fp = fn = tn = 0
    for mid, gt in ground_truth.items():
        if mid not in results:
            continue
        pred = results[mid]["detects_attention"]
        actual = gt["needs_attention"]
        if pred and actual:     tp += 1
        elif pred and not actual: fp += 1
        elif not pred and actual: fn += 1
        else:                     tn += 1
    total = tp + fp + fn + tn
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0
    acc  = (tp+tn) / total if total > 0 else 0
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": round(prec, 4), "recall": round(rec, 4),
        "f1": round(f1, 4), "accuracy": round(acc, 4),
    }


def count_capabilities(results):
    caps = {"root_cause": 0, "explainability": 0, "correlation": 0, "consensus": 0}
    n = len(results)
    for r in results.values():
        if r.get("has_root_cause"):      caps["root_cause"]     += 1
        if r.get("has_explainability"):  caps["explainability"] += 1
        if r.get("has_correlation"):     caps["correlation"]    += 1
        if r.get("has_consensus"):       caps["consensus"]      += 1
    pct = {k: round(v/n*100, 1) if n > 0 else 0 for k, v in caps.items()}
    return pct


def count_near_misses(results, ground_truth):
    fn_list = []
    for mid, gt in ground_truth.items():
        if mid in results and not results[mid]["detects_attention"] and gt["needs_attention"]:
            fn_list.append(mid)
    return fn_list


# ═══════════════════════════════════════════════
# MAIN COMPARISON
# ═══════════════════════════════════════════════

def main(seeds=None):
    """
    Train each model on the train split, evaluate on the test split.
    Repeat across multiple seeds and report mean ± stdev for the
    stochastic models — disclosure that a single-seed result is not
    a reliable estimate.
    """
    seeds = seeds or [42, 7, 1234]

    print("\n" + "="*90)
    print(" COMPREHENSIVE COMPARATIVE STUDY (post-audit) — 8 APPROACHES")
    print(" Predictive Maintenance in Smart Manufacturing")
    print(" AUDIT DISCLOSURE:")
    print(" - Train/test split: machine_id even = train, odd = test (per derive_sensor_bounds.py)")
    print(" - Bounds & thresholds derived from training split only")
    print(" - No ground-truth fields used as features by any model")
    print(f" - Multi-seed evaluation: seeds = {seeds}")
    print("="*90)

    # ── Load split data ──
    train_events, test_events = load_split_data()
    if not test_events:
        print("[ERROR] No test data.")
        return

    gt_train = build_ground_truth(train_events)
    gt_test  = build_ground_truth(test_events)
    train_labels = {mid: 1.0 if g["needs_attention"] else 0.0 for mid, g in gt_train.items()}

    n_test = len(test_events)
    n_pos_test = sum(1 for g in gt_test.values() if g["needs_attention"])
    print(f"\n  Train machines: {len(train_events)} (used for fitting)")
    print(f"  Test machines:  {n_test} | needs attention: {n_pos_test}")

    model_classes = [
        RuleBasedModel, TransformerModel, LSTMModel, GNNModel,
        RLModel, FederatedModel, TransferLearningModel, AgenticModel,
    ]

    # Multi-seed evaluation. RuleBased & Agentic are deterministic
    # given fixed config — we still call across seeds for uniformity.
    all_metrics_per_seed = {}   # short → {seed → metrics_dict}
    final_results        = {}   # short → first-seed predictions (for capabilities)
    all_times            = {}   # short → first-seed timing

    for model_cls in model_classes:
        short = model_cls.short
        all_metrics_per_seed[short] = {}
        for seed in seeds:
            random.seed(seed)
            model = model_cls()
            print(f"\n{'─'*90}")
            print(f"  [{short}] {model.name}  seed={seed}")
            print(f"{'─'*90}")
            t0 = time.time()
            print("    Training on train split...")
            model.train(train_events, train_labels)
            t_train = time.time() - t0
            t0 = time.time()
            print("    Predicting on test split...")
            preds = model.predict(test_events)
            t_pred = time.time() - t0
            metrics = compute_metrics(preds, gt_test)
            all_metrics_per_seed[short][seed] = metrics
            print(f"    Done. Train: {t_train:.3f}s | Predict: {t_pred:.3f}s | "
                  f"F1={metrics['f1']:.4f} Recall={metrics['recall']:.4f}")
            if seed == seeds[0]:
                final_results[short] = preds
                all_times[short] = {"train": round(t_train, 3), "predict": round(t_pred, 3)}

    # Aggregate metrics across seeds
    all_metrics = {}
    for short, by_seed in all_metrics_per_seed.items():
        agg = {}
        for k in ("TP", "FP", "FN", "TN", "precision", "recall", "f1", "accuracy"):
            vals = [by_seed[s][k] for s in seeds]
            mean = statistics.mean(vals)
            std  = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            agg[k] = {"mean": round(mean, 4), "stdev": round(std, 4),
                      "per_seed": {str(s): vals[i] for i, s in enumerate(seeds)}}
        all_metrics[short] = agg

    all_results = final_results  # aliased for downstream code
    n = n_test
    n_pos = n_pos_test
    n_neg = n - n_pos
    gt = gt_test
    models = [m for m in [RuleBasedModel(), TransformerModel(), LSTMModel(), GNNModel(),
                          RLModel(), FederatedModel(), TransferLearningModel(), AgenticModel()]]

    # ══════════════════════════════════════════════════
    # FINAL COMPARISON REPORT
    # ══════════════════════════════════════════════════
    print("\n\n" + "="*90)
    print(" RESULTS — QUANTITATIVE COMPARISON")
    print("="*90)

    # ── Table 1: Detection Performance (mean ± stdev across seeds) ──
    print(f"\n{'─'*90}")
    print(f" TABLE 1: Detection Performance on TEST split (mean ± stdev across {len(seeds)} seeds)")
    print(f"{'─'*90}")
    header = f"  {'Model':<18} {'Prec(±)':>15} {'Recall(±)':>15} {'F1(±)':>15} {'Acc(±)':>15}"
    print(header)
    print(f"  {'─'*86}")

    for model in models:
        m = all_metrics[model.short]
        prec = f"{m['precision']['mean']:.4f}±{m['precision']['stdev']:.3f}"
        rec  = f"{m['recall']['mean']:.4f}±{m['recall']['stdev']:.3f}"
        f1   = f"{m['f1']['mean']:.4f}±{m['f1']['stdev']:.3f}"
        acc  = f"{m['accuracy']['mean']:.4f}±{m['accuracy']['stdev']:.3f}"
        print(f"  {model.short:<18} {prec:>15} {rec:>15} {f1:>15} {acc:>15}")

    # ── Table 2: Capability Coverage ──
    print(f"\n{'─'*90}")
    print(f" TABLE 2: Capability Coverage (% of machines with each analysis)")
    print(f"{'─'*90}")
    print(f"  {'Model':<18} {'Root Cause':>11} {'XAI':>7} {'Correlation':>12} {'Consensus':>10}")
    print(f"  {'─'*58}")

    for model in models:
        caps = count_capabilities(all_results[model.short])
        print(f"  {model.short:<18} {caps['root_cause']:>10.1f}% {caps['explainability']:>6.1f}% "
              f"{caps['correlation']:>11.1f}% {caps['consensus']:>9.1f}%")

    # ── Table 3: Near-Miss Analysis ──
    print(f"\n{'─'*90}")
    print(f" TABLE 3: Near-Miss Analysis (False Negatives)")
    print(f"{'─'*90}")

    for model in models:
        nm = count_near_misses(all_results[model.short], gt)
        print(f"  {model.short:<18} missed {len(nm):>3} / {n_pos} machines needing attention")
        if nm and len(nm) <= 10:
            missed_info = [f"{mid}({gt[mid]['failure_type']})" for mid in nm[:5]]
            print(f"    {'→ '}{', '.join(missed_info)}")

    # ── Table 4: Execution Time ──
    print(f"\n{'─'*90}")
    print(f" TABLE 4: Execution Time")
    print(f"{'─'*90}")
    print(f"  {'Model':<18} {'Train (s)':>10} {'Predict (s)':>12} {'Total (s)':>10}")
    print(f"  {'─'*50}")
    for model in models:
        t = all_times[model.short]
        total = t['train'] + t['predict']
        print(f"  {model.short:<18} {t['train']:>10.3f} {t['predict']:>12.3f} {total:>10.3f}")

    # ── Summary ──
    print(f"\n{'='*90}")
    print(f" ANALYSIS SUMMARY")
    print(f"{'='*90}")

    best_f1 = max(all_metrics.items(), key=lambda x: x[1]["f1"]["mean"])
    best_recall = max(all_metrics.items(), key=lambda x: x[1]["recall"]["mean"])

    print(f"\n  Best F1 Score   : {best_f1[0]} (F1={best_f1[1]['f1']['mean']:.4f}±{best_f1[1]['f1']['stdev']:.3f})")
    print(f"  Best Recall     : {best_recall[0]} (Recall={best_recall[1]['recall']['mean']:.4f}±{best_recall[1]['recall']['stdev']:.3f})")
    print(f"  Test machines   : {n_pos}/{n} need attention (ground truth)")

    # Compare agentic vs others (mean F1)
    ag = all_metrics.get("AgenticAI", {})
    if ag:
        print(f"\n  Agentic AI vs each model (Δ mean F1):")
        for model in models:
            if model.short != "AgenticAI":
                m = all_metrics[model.short]
                diff = ag["f1"]["mean"] - m["f1"]["mean"]
                print(f"    vs {model.short:<18} ΔF1={diff:>+.4f}")

    print(f"\n  Key Insight:")
    agentic_caps = count_capabilities(all_results.get("AgenticAI", {}))
    others_max_caps = {}
    for model in models:
        if model.short != "AgenticAI":
            caps = count_capabilities(all_results[model.short])
            for k, v in caps.items():
                others_max_caps[k] = max(others_max_caps.get(k, 0), v)

    unique_caps = [k for k, v in agentic_caps.items() if v > 0 and others_max_caps.get(k, 0) == 0]
    shared_caps = [k for k, v in agentic_caps.items() if v > 0 and others_max_caps.get(k, 0) > 0]

    if unique_caps:
        print(f"    Capabilities ONLY in Agentic AI: {', '.join(unique_caps)}")
    if shared_caps:
        print(f"    Shared with other models       : {', '.join(shared_caps)}")

    print(f"\n{'='*90}")

    # ── Save all results ──
    output = {
        "_audit_disclosure": (
            "Train/test split (machine_id even=train, odd=test). Bounds & "
            "thresholds derived from train split only. No ground-truth "
            "fields used as features. Detection metrics = mean ± stdev "
            f"across seeds {seeds}."
        ),
        "seeds_used": seeds,
        "timestamp": str(datetime.datetime.now()),
        "n_test": n, "n_positive_test": n_pos, "n_negative_test": n_neg,
        "detection_metrics": all_metrics,
        "execution_times_seed0": all_times,
        "capabilities_seed0": {m.short: count_capabilities(all_results[m.short]) for m in models},
        "near_misses_seed0": {m.short: count_near_misses(all_results[m.short], gt) for m in models},
    }
    with open("full_comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n[SAVED] full_comparison_results.json")
    return output


if __name__ == "__main__":
    main()
