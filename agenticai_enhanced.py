# ============================================================
# AGENTIC AI — ENHANCED Smart Manufacturing Pipeline
# ============================================================
#
# AUDIT NOTE (research-ethics rewrite):
#   The previous version of this file presented several heuristics
#   as Variational Quantum Classifier (VQC), Quantum Approximate
#   Optimization Algorithm (QAOA), and Quantum Graph Neural Network
#   (QGNN). The implementations were classical heuristics —
#   weighted averages, random sampling, and label lookups — that
#   did not invoke any quantum simulator. The "VQC" also added
#   ground-truth fields (failure_type, anomaly_flag) directly to
#   the anomaly amplitude, which is direct label leakage.
#
#   This rewrite renames those classes to honest labels:
#     * QuantumSimulator                → ClassicalHeuristicSuite
#     * qaoa_schedule_optimization      → random_constrained_search
#     * vqc_anomaly_detection           → energy_distance_anomaly_score
#     * qgnn_dependency_analysis        → graph_message_passing_score
#   Label leakage in the anomaly score is removed. The "fidelity"
#   field, which was previously a random number, is removed.
#   Algorithm metadata in the saved JSON now reflects what the
#   code actually does.
#
#   The Consensus Agent — which previously asked the SAME LLM to
#   "validate" outputs from itself — is renamed SelfReflectionAgent
#   to be honest about what it is. A real consensus agent would
#   need ≥2 different models; that's left as future work.
#
#   The "Anomaly Pattern Learning Agent" prompt no longer asks the
#   LLM to fabricate quantum metrics (vqc_*, qgnn_*,
#   quantum_vs_classical_accuracy_gain) — those values were
#   hallucinations.
#
# Architecture (post-rewrite):
#   1. Monitoring Agent       — risk classification (LLM)
#   2. Diagnosis Agent        — root cause (LLM)
#   3. Planning Agent         — action recommendation (LLM)
#   4. Reporting Agent        — operator-facing alert (LLM)
#   5. Correlation Agent      — cluster analysis from graph topology
#   6. Explainability Agent   — counterfactual / feature attribution
#   7. SelfReflection Agent   — sanity check on agents 1–3 (same LLM!)
#   8. Scheduling Agent       — maintenance window planning
#   9. Pattern Learning Agent — temporal patterns from sliding window
#
# Heuristic suite (formerly mislabeled "quantum"):
#   - random_constrained_search → maintenance scheduling
#   - energy_distance_anomaly_score → multivariate anomaly score
#   - graph_message_passing_score → inter-machine influence score
# ============================================================

import json
import time
import math
import random
import datetime
from collections import deque
from pymongo import MongoClient
from ollama import Client

from config import (
    OLLAMA_API_KEY, MONGODB_URI, MONGO_DB, MONGO_COLLECTION,
    OLLAMA_MODEL, RUL_LOW, RUL_MODERATE, RISK_HIGH_AT, RISK_MEDIUM_AT,
    SENSOR_FEATURES, get_normalization_range, get_warn_crit_thresholds,
    load_sensor_bounds,
)

POLL_INTERVAL_SEC = 60
RANDOM_SEED       = 42  # disclosed; vary with --seed for variance reporting

# ─────────────────────────────────────────────
# LAYER 1 — DATA INGESTION
# ─────────────────────────────────────────────
class KafkaStreamConsumer:
    """
    Real-time consumer: reads the digital-twin output collection in
    MongoDB. The collection itself is populated by an upstream Kafka
    sink (digital twin → Kafka → MongoDB).
    """
    def __init__(self, mongo_uri=MONGODB_URI, db_name=MONGO_DB, collection_name=MONGO_COLLECTION):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        print(f"[CONNECTED] MongoDB: {db_name}.{collection_name}")

    def get_latest_snapshot(self, machine_ids=None):
        try:
            query = {}
            if machine_ids is not None:
                query['machine_id'] = {'$in': machine_ids}
            cursor = self.collection.find(query).sort('timestamp', -1).limit(200)
            latest_per_machine = {}
            for doc in cursor:
                mid = doc['machine_id']
                if mid not in latest_per_machine:
                    latest_per_machine[mid] = doc
            results = list(latest_per_machine.values())
            results.sort(key=lambda x: x['machine_id'])
            return results
        except Exception as e:
            print(f"[ERROR] get_latest_snapshot failed: {e}")
            return []

    def close(self):
        self.client.close()
        print("[DISCONNECTED] MongoDB connection closed")


# ─────────────────────────────────────────────
# LAYER 2 — CONTEXT / MEMORY STORE (sliding window)
# ─────────────────────────────────────────────
class ContextMemoryStore:
    def __init__(self, window_size=10):
        self.window_size     = window_size
        self.sliding_window  = {}
        self.decision_log    = []
        self.feedback_log    = []

    def update(self, machine_id, event):
        if machine_id not in self.sliding_window:
            self.sliding_window[machine_id] = deque(maxlen=self.window_size)
        self.sliding_window[machine_id].append(event)

    def get_history(self, machine_id):
        return list(self.sliding_window.get(machine_id, []))


# ─────────────────────────────────────────────
# LAYER 3 — REASONING ENGINE (no label leakage)
# ─────────────────────────────────────────────
class ReasoningEngine:
    """
    Threshold-based scoring on raw sensor channels and RUL ONLY.

    AUDIT NOTE: previous versions added 0.20 for anomaly_flag and
    0.15 for failure_type != Normal — those fields are GROUND TRUTH
    and using them as features made the baseline trivially high.
    Both are removed here.
    """
    def __init__(self):
        self.thresholds = get_warn_crit_thresholds()
        if not self.thresholds or all(t["warn"] == float("inf") for t in self.thresholds.values()):
            print("[reasoner] WARN: no derived bounds found. Run derive_sensor_bounds.py first.")

    def score(self, event: dict) -> dict:
        score = 0.0
        flags = []
        for feature, thr in self.thresholds.items():
            val = event.get(feature, event.get("energy" if feature == "energy_consumption" else feature, 0))
            if not isinstance(val, (int, float)):
                continue
            if val >= thr["crit"]:
                score += 0.35
                flags.append(f"{feature}=CRITICAL({val:.2f}>={thr['crit']:.2f})")
            elif val >= thr["warn"]:
                score += 0.15
                flags.append(f"{feature}=WARN({val:.2f}>={thr['warn']:.2f})")

        rul = event.get('predicted_remaining_life', 200)
        if rul < RUL_LOW:
            score += 0.30
            flags.append(f"RUL=LOW({rul})")
        elif rul < RUL_MODERATE:
            score += 0.10
            flags.append(f"RUL=MODERATE({rul})")

        risk_score = min(score, 1.0)
        severity = "HIGH" if risk_score >= RISK_HIGH_AT else \
                   ("MEDIUM" if risk_score >= RISK_MEDIUM_AT else "LOW")
        return {
            "risk_score"   : round(risk_score, 3),
            "severity"     : severity,
            "flags"        : flags,
            "predicted_rul": rul,
        }


# ─────────────────────────────────────────────
# LAYER 4 — MULTI-AGENT ORCHESTRATOR (Ollama)
# ─────────────────────────────────────────────
def get_ollama_client():
    return Client(
        host="https://ollama.com",
        headers={"Authorization": "Bearer " + OLLAMA_API_KEY},
    )


# ── Agent prompts.
# Inputs sent to LLMs MUST NOT include ground-truth fields
# (failure_type, anomaly_flag, machine_status_label, maintenance_required,
# downtime_risk). Those are evaluation labels, not features.
SYSTEM_PROMPT_MONITORING = """
You are a machine monitoring agent for a smart factory.
Analyze sensor data and return a JSON decision with these keys ONLY:
{
  "risk_level": "LOW|MEDIUM|HIGH|CRITICAL",
  "anomaly_detected": true|false,
  "anomaly_type": "thermal|vibration|energy|combined|none",
  "confidence": 0.0-1.0
}
Return valid JSON only. No explanation outside JSON.
"""

SYSTEM_PROMPT_DIAGNOSIS = """
You are a root-cause diagnosis agent for predictive maintenance.
Given sensor context and threshold flags, return JSON:
{
  "probable_cause": "bearing_wear|overheating|electrical|lubrication|normal|unknown",
  "affected_component": "motor|bearing|pump|actuator|unknown",
  "urgency_hours": integer (0-72)
}
Return valid JSON only.
"""

SYSTEM_PROMPT_PLANNING = """
You are a maintenance planning agent.
Given diagnosis and risk context, decide optimal action and return JSON:
{
  "action": "monitor|inspect|schedule_maintenance|immediate_shutdown",
  "priority": "P1|P2|P3|P4",
  "reason": "brief explanation max 20 words",
  "notify_supervisor": true|false
}
Return valid JSON only.
"""

SYSTEM_PROMPT_REPORTING = """
You are a reporting agent for factory supervisors.
Generate a short, human-readable maintenance alert in Bahasa Indonesia.
Max 3 sentences. Plain text only, no JSON.
"""

SYSTEM_PROMPT_CORRELATION = """
You are a Correlation Agent for inter-machine dependency analysis.
You are given pre-computed graph topology features (anchor_machine,
graph_distance, influence_score) along with sensor readings.
Analyze and return JSON:
{
  "correlated_clusters": [
    {
      "machines": ["M-1", "M-3"],
      "correlation_type": "thermal|vibration|electrical|other",
      "propagation_risk": 0.0-1.0,
      "root_machine": "M-X",
      "explanation": "max 30 words"
    }
  ],
  "highest_risk_cluster_index": integer or null,
  "cascade_probability": 0.0-1.0,
  "recommendation": "max 30 words"
}
Return valid JSON only.
"""

SYSTEM_PROMPT_EXPLAINABILITY = """
You are an Explainability Agent (XAI).
Given a machine's sensor data and threshold configuration,
generate counterfactual explanations and feature attribution.
Return JSON:
{
  "feature_attribution": {
    "temperature": {"contribution_pct": float, "direction": "increasing_risk|decreasing_risk|neutral"},
    "vibration":   {"contribution_pct": float, "direction": "increasing_risk|decreasing_risk|neutral"},
    "humidity":    {"contribution_pct": float, "direction": "increasing_risk|decreasing_risk|neutral"},
    "energy":      {"contribution_pct": float, "direction": "increasing_risk|decreasing_risk|neutral"},
    "rul":         {"contribution_pct": float, "direction": "increasing_risk|decreasing_risk|neutral"}
  },
  "counterfactual": {
    "current_severity": "LOW|MEDIUM|HIGH",
    "nearest_boundary": "MEDIUM|HIGH",
    "distance_to_boundary": float,
    "what_if_scenarios": [
      {"change": "string", "new_severity": "MEDIUM|HIGH", "delta_risk": float}
    ]
  },
  "natural_language_explanation": "max 50 words",
  "confidence_in_explanation": 0.0-1.0
}
Return valid JSON only.
"""

# Renamed: was SYSTEM_PROMPT_CONSENSUS. This is a self-reflection
# from the SAME model — not multi-model consensus.
SYSTEM_PROMPT_SELFREFLECTION = """
You are a Self-Reflection Agent reviewing the outputs of three
sibling agents (Monitoring, Diagnosis, Planning) produced by the
SAME underlying language model. Your job is to flag internal
inconsistencies — for example: planning says "monitor" while
diagnosis says "overheating".

DISCLOSURE: this is a self-consistency check, not a multi-model
consensus. Real consensus would require ≥2 different models.

Return JSON:
{
  "self_consistent": true|false,
  "inconsistencies": ["short description"],
  "recommended_resolution": "monitor|inspect|schedule_maintenance|immediate_shutdown",
  "confidence_in_review": 0.0-1.0
}
Return valid JSON only.
"""

SYSTEM_PROMPT_SCHEDULING = """
You are a Predictive Scheduling Agent for optimizing maintenance windows
across the fleet. Given RUL predictions and risk scores for multiple
machines and the constraint of max-K simultaneous operations, propose
a schedule that minimizes total downtime.

Return JSON:
{
  "schedule": [
    {
      "machine_id": "M-X",
      "maintenance_type": "preventive|corrective|inspection",
      "scheduled_window": "YYYY-MM-DD HH:MM - HH:MM",
      "priority": "P1|P2|P3|P4",
      "estimated_downtime_hours": float,
      "rul_at_schedule": float
    }
  ],
  "schedule_summary": {
    "total_downtime_hours": float,
    "machines_covered": integer,
    "method": "greedy|constraint_satisfaction"
  }
}
Return valid JSON only.
"""

# AUDIT NOTE: previous prompt asked the LLM for vqc_anomaly_probability,
# vqc_circuit_depth, qgnn_graph_anomaly_score, and
# quantum_vs_classical_accuracy_gain. Those values would be hallucinated
# — the LLM has no quantum runtime. Fields removed.
SYSTEM_PROMPT_PATTERN = """
You are an Anomaly Pattern Learning Agent. Given sliding-window history
across machines (sensor readings only), identify temporal patterns.

Return JSON:
{
  "detected_patterns": [
    {
      "pattern_id": "PAT-XXX",
      "pattern_type": "slow_drift|cyclic|spike|cross_feature",
      "affected_machines": ["M-X"],
      "affected_features": ["temperature", "vibration"],
      "confidence": 0.0-1.0,
      "description": "max 30 words",
      "trend_direction": "worsening|stable|improving",
      "estimated_time_to_failure_hours": float
    }
  ],
  "summary": {
    "total_patterns_detected": integer,
    "novel_patterns": integer
  }
}
Return valid JSON only.
"""


class MultiAgentOrchestrator:
    def __init__(self, model_name=OLLAMA_MODEL):
        self.model  = model_name
        self.client = get_ollama_client()

    def _call(self, system_prompt, user_msg):
        try:
            resp = self.client.chat(
                model    = self.model,
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_msg},
                ],
            )
            content = resp["message"]["content"].strip()
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return {"raw": content}
        except Exception as e:
            return {"error": str(e)}

    # AUDIT NOTE: helper to build LLM-safe sensor context. Excludes
    # all ground-truth labels.
    @staticmethod
    def _safe_sensor_view(event):
        return {
            "machine_id":  event["machine_id"],
            "temperature": event.get("temperature", 0),
            "vibration":   event.get("vibration", 0),
            "humidity":    event.get("humidity", 0),
            "pressure":    event.get("pressure", 0),
            "energy":      event.get("energy", event.get("energy_consumption", 0)),
            "predicted_remaining_life": event.get("predicted_remaining_life", 200),
        }

    @staticmethod
    def _safe_topology_view(event):
        return {
            "machine_id":     event["machine_id"],
            "anchor_machine": event.get("anchor_machine", 0),
            "graph_distance": event.get("graph_distance", 0),
            "influence_score": event.get("influence_score", 0),
        }

    def run_monitoring_agent(self, event, reasoning_result):
        msg = (f"Sensor event: {json.dumps(self._safe_sensor_view(event), indent=2)}\n"
               f"Threshold flags: {reasoning_result.get('flags', [])}")
        return self._call(SYSTEM_PROMPT_MONITORING, msg)

    def run_diagnosis_agent(self, event, monitoring_result, reasoning_result):
        msg = (f"Sensor event: {json.dumps(self._safe_sensor_view(event), indent=2)}\n"
               f"Monitoring result: {json.dumps(monitoring_result, indent=2)}\n"
               f"Threshold flags: {reasoning_result.get('flags', [])}")
        return self._call(SYSTEM_PROMPT_DIAGNOSIS, msg)

    def run_planning_agent(self, diagnosis_result, reasoning_result):
        msg = (f"Diagnosis: {json.dumps(diagnosis_result, indent=2)}\n"
               f"Risk score: {reasoning_result.get('risk_score')}\n"
               f"Severity: {reasoning_result.get('severity')}\n"
               f"Predicted RUL: {reasoning_result.get('predicted_rul')}")
        return self._call(SYSTEM_PROMPT_PLANNING, msg)

    def run_reporting_agent(self, machine_id, planning_result, diagnosis_result):
        msg = (f"Machine ID: M-{machine_id}\n"
               f"Diagnosis: {json.dumps(diagnosis_result, indent=2)}\n"
               f"Action plan: {json.dumps(planning_result, indent=2)}")
        return self._call(SYSTEM_PROMPT_REPORTING, msg)

    def run_correlation_agent(self, all_events, all_reasonings):
        machine_summary = []
        for evt, rsn in zip(all_events, all_reasonings):
            machine_summary.append({
                **self._safe_sensor_view(evt),
                **self._safe_topology_view(evt),
                "risk_score":    rsn["risk_score"],
                "severity":      rsn["severity"],
            })
        msg = f"Factory-wide sensor + topology data for {len(machine_summary)} machines:\n{json.dumps(machine_summary, indent=2)}"
        return self._call(SYSTEM_PROMPT_CORRELATION, msg)

    def run_explainability_agent(self, event, reasoning_result, thresholds):
        msg = (f"Machine: {event['machine_id']}\n"
               f"Sensor event: {json.dumps(self._safe_sensor_view(event), indent=2)}\n"
               f"Risk assessment: {json.dumps(reasoning_result, indent=2)}\n"
               f"Current thresholds: {json.dumps(thresholds, indent=2)}\n"
               f"RUL policy: warn<{RUL_MODERATE}h, critical<{RUL_LOW}h")
        return self._call(SYSTEM_PROMPT_EXPLAINABILITY, msg)

    def run_self_reflection_agent(self, monitoring_result, diagnosis_result, planning_result):
        msg = (f"Monitoring: {json.dumps(monitoring_result, indent=2)}\n"
               f"Diagnosis: {json.dumps(diagnosis_result, indent=2)}\n"
               f"Planning: {json.dumps(planning_result, indent=2)}")
        return self._call(SYSTEM_PROMPT_SELFREFLECTION, msg)

    def run_scheduling_agent(self, all_events, all_reasonings):
        sched_input = []
        for evt, rsn in zip(all_events, all_reasonings):
            sched_input.append({
                "machine_id":    evt["machine_id"],
                "risk_score":    rsn["risk_score"],
                "severity":      rsn["severity"],
                "predicted_rul": rsn["predicted_rul"],
                "energy":        evt.get("energy", evt.get("energy_consumption", 0)),
            })
        msg = (f"Maintenance scheduling for {len(sched_input)} machines.\n"
               f"Current timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
               f"Machine data:\n{json.dumps(sched_input, indent=2)}\n\n"
               "Constraints: max 5 simultaneous, working hours 08:00-18:00, "
               "lowest RUL first.")
        return self._call(SYSTEM_PROMPT_SCHEDULING, msg)

    def run_pattern_agent(self, history_per_machine):
        """
        history_per_machine: dict {machine_id: [event, event, ...]}
        Only the last few events per machine are passed to keep prompt small.
        """
        compact = {}
        for mid, events in history_per_machine.items():
            recent = events[-5:]
            compact[mid] = [
                {k: e.get(k) for k in ("temperature", "vibration", "humidity", "energy", "predicted_remaining_life")}
                for e in recent
            ]
        msg = (f"Sliding-window history (last ≤5 readings per machine) for {len(compact)} machines:\n"
               f"{json.dumps(compact, indent=2)}")
        return self._call(SYSTEM_PROMPT_PATTERN, msg)


# ════════════════════════════════════════════════════════════
# CLASSICAL HEURISTIC SUITE
# (was: QuantumSimulator with mislabeled "QAOA / VQC / QGNN" methods)
# ════════════════════════════════════════════════════════════

class ClassicalHeuristicSuite:
    """
    AUDIT NOTE (renamed from QuantumSimulator):
        The previous implementation labeled three classical heuristics
        as quantum algorithms (VQC, QAOA, QGNN). They did not invoke
        any quantum simulator. Names and output metadata in this class
        now reflect what the code actually does. A real quantum
        implementation using Qiskit AerSimulator would belong in a
        separate module.

    Methods:
      - random_constrained_search    → maintenance scheduling
      - energy_distance_anomaly_score → multivariate anomaly score
      - graph_message_passing_score   → inter-machine influence score
    """

    @staticmethod
    def random_constrained_search(machines_data, n_samples=200, max_simultaneous=5,
                                  rng=None):
        """
        Constrained random search over binary decisions x_i ∈ {0,1}
        (schedule machine i now: yes/no), maximizing
            Σ risk_i * x_i + Σ urgency_i * x_i
        subject to Σ x_i ≤ max_simultaneous.

        AUDIT NOTE:
          The previous version applied a `cos(γ·cost) * sin(β)` term
          and called it "quantum interference". That term has no
          quantum semantics — it was a constant deformation of a
          classical objective. Removed.
        """
        rng = rng or random.Random(RANDOM_SEED)
        n = len(machines_data)
        if n == 0:
            return {
                "algorithm": "random_constrained_search",
                "n_samples": 0,
                "best_score": 0.0,
                "machines_to_schedule": [],
                "total_scheduled": 0,
            }

        risks = [m.get("risk_score", 0) for m in machines_data]
        ruls  = [m.get("predicted_rul", 200) for m in machines_data]
        max_rul = max(ruls) if ruls else 1
        urgencies = [(max_rul - r) / max_rul if max_rul > 0 else 0 for r in ruls]

        def objective(bits):
            n_sched = sum(bits)
            if n_sched > max_simultaneous:
                return -1e9  # infeasible
            return sum((risks[i] + urgencies[i]) * bits[i] for i in range(n))

        # Greedy seed: top-K by (risk + urgency)
        order = sorted(range(n), key=lambda i: -(risks[i] + urgencies[i]))
        best = [0] * n
        for i in order[:max_simultaneous]:
            best[i] = 1
        best_score = objective(best)

        # Random local search around the seed
        for _ in range(n_samples):
            cand = best[:]
            i = rng.randrange(n)
            cand[i] = 1 - cand[i]
            s = objective(cand)
            if s > best_score:
                best, best_score = cand, s

        scheduled = []
        for i, bit in enumerate(best):
            if bit == 1:
                scheduled.append({
                    "machine_id":    machines_data[i].get("machine_id", f"M-{i+1}"),
                    "risk_score":    risks[i],
                    "urgency":       round(urgencies[i], 3),
                    "predicted_rul": ruls[i],
                })
        scheduled.sort(key=lambda x: -x["urgency"])

        return {
            "algorithm": "random_constrained_search",
            "n_samples": n_samples,
            "max_simultaneous": max_simultaneous,
            "best_objective_score": round(best_score, 4),
            "machines_to_schedule": scheduled,
            "total_scheduled": sum(best),
            "_disclosure": (
                "Greedy seed + random 1-bit-flip local search. NOT a quantum "
                "algorithm. Deterministic given RANDOM_SEED."
            ),
        }

    @staticmethod
    def energy_distance_anomaly_score(events):
        """
        Per-machine multivariate score = L2 distance of normalized
        sensor vector from the per-feature p50 (median) derived from
        the training set. NO ground-truth fields are read.

        AUDIT NOTE:
          The previous "VQC anomaly detection" added 0.15 if
          failure_type != Normal and 0.20 if anomaly_flag == 1. Both
          fields are GROUND TRUTH. This rewrite scores from sensor
          values only. Output also no longer reports a randomized
          "fidelity" field.
        """
        bounds = load_sensor_bounds() or {}
        # Build canonical median vector
        canonical = {}
        for f in SENSOR_FEATURES:
            b = bounds.get(f, {})
            canonical[f] = b.get("p50", 0.0)

        results = []
        for event in events:
            sq = 0.0
            n_used = 0
            for f in SENSOR_FEATURES:
                # Some events use "energy" instead of "energy_consumption"
                v = event.get(f, event.get("energy" if f == "energy_consumption" else f))
                if v is None or not isinstance(v, (int, float)):
                    continue
                lo, hi = get_normalization_range(f)
                if hi <= lo:
                    continue
                norm_v = (v - lo) / (hi - lo)
                norm_c = (canonical[f] - lo) / (hi - lo)
                sq += (norm_v - norm_c) ** 2
                n_used += 1
            dist = math.sqrt(sq / max(n_used, 1))
            score = 1.0 - math.exp(-3.0 * dist)
            results.append({
                "machine_id": event.get("machine_id", "unknown"),
                "anomaly_score": round(score, 4),
                "classification_at_0_5": "ANOMALY" if score > 0.5 else "NORMAL",
                "n_features_used": n_used,
            })
        return {
            "algorithm": "energy_distance_anomaly_score",
            "n_machines": len(results),
            "machine_results": results,
            "anomaly_count_at_0_5": sum(1 for r in results if r["classification_at_0_5"] == "ANOMALY"),
            "_disclosure": (
                "Normalized L2 distance from training-set median per feature, "
                "passed through 1 - exp(-3d). Sensor inputs only. NOT a quantum "
                "algorithm."
            ),
        }

    @staticmethod
    def graph_message_passing_score(events, n_rounds=3):
        """
        Classical message passing on the T-GCN topology already present
        in the events. After n_rounds of weighted neighbor averaging,
        each node's deviation from its initial state is reported.

        AUDIT NOTE:
          The previous "QGNN" version used the formula
              updated = 0.7 * self + 0.3 * neighbor
          and called it "quantum interference". It is a weighted
          average. Renamed accordingly. This class no longer claims
          to be quantum.
        """
        # Build adjacency
        adjacency = {}
        for i, evt_i in enumerate(events):
            mid_i = evt_i.get("machine_id", f"M-{i+1}")
            neighbors = []
            for j, evt_j in enumerate(events):
                if i == j:
                    continue
                mid_j = evt_j.get("machine_id", f"M-{j+1}")
                same_anchor = evt_i.get("anchor_machine") == evt_j.get("anchor_machine")
                close_dist  = abs(evt_i.get("graph_distance", 0) - evt_j.get("graph_distance", 0)) < 2
                influence   = max(evt_i.get("influence_score", 0), evt_j.get("influence_score", 0))
                if same_anchor or (close_dist and influence > 0.3):
                    edge_weight = 0.5 + influence * 0.5
                    neighbors.append({"node": mid_j, "weight": round(edge_weight, 3)})
            adjacency[mid_i] = neighbors

        # Initial node features (normalized)
        node_states = {}
        for evt in events:
            mid = evt.get("machine_id")
            feats = []
            for f in SENSOR_FEATURES:
                v = evt.get(f, evt.get("energy" if f == "energy_consumption" else f))
                if v is None or not isinstance(v, (int, float)):
                    feats.append(0.0)
                    continue
                lo, hi = get_normalization_range(f)
                feats.append((v - lo) / (hi - lo) if hi > lo else 0.0)
            node_states[mid] = {
                "feature_vector": feats,
                "initial_vector": feats[:],
            }

        # Message passing rounds
        for _ in range(n_rounds):
            new_states = {}
            for mid, state in node_states.items():
                neighbors = adjacency.get(mid, [])
                if not neighbors:
                    new_states[mid] = state
                    continue
                d = len(state["feature_vector"])
                msg_sum = [0.0] * d
                total_weight = 0.0
                for neighbor in neighbors:
                    n_state = node_states.get(neighbor["node"])
                    if not n_state:
                        continue
                    w = neighbor["weight"]
                    for k in range(d):
                        msg_sum[k] += n_state["feature_vector"][k] * w
                    total_weight += w
                if total_weight == 0:
                    new_states[mid] = state
                    continue
                new_features = []
                for k in range(d):
                    self_val = state["feature_vector"][k]
                    neighbor_val = msg_sum[k] / total_weight
                    updated = 0.7 * self_val + 0.3 * neighbor_val
                    new_features.append(round(updated, 4))
                new_states[mid] = {
                    "feature_vector": new_features,
                    "initial_vector": state["initial_vector"],
                }
            node_states = new_states

        # Influence score = magnitude of change after message passing
        scored = []
        for mid, state in node_states.items():
            init = state["initial_vector"]
            curr = state["feature_vector"]
            d = len(init)
            deviation = sum(abs(curr[k] - init[k]) for k in range(d)) / max(d, 1)
            scored.append({
                "machine_id": mid,
                "graph_influence_score": round(deviation, 4),
                "n_neighbors": len(adjacency.get(mid, [])),
            })
        scored.sort(key=lambda x: -x["graph_influence_score"])
        n_edges = sum(len(v) for v in adjacency.values()) // 2
        return {
            "algorithm": "graph_message_passing_score",
            "n_nodes":   len(node_states),
            "n_edges":   n_edges,
            "n_rounds":  n_rounds,
            "ranked_nodes_top10": scored[:10],
            "_disclosure": (
                "Weighted neighbor averaging on the existing T-GCN adjacency. "
                "NOT a quantum algorithm."
            ),
        }


# ─────────────────────────────────────────────
# LAYER 5 — ACTION LAYER (HITL gate disclosed)
# ─────────────────────────────────────────────
HIGH_RISK_ACTIONS = {"immediate_shutdown", "schedule_maintenance"}


def execute_action(machine_id, planning_result, report, *, hitl_callback=None):
    """
    Dispatch an action. Real production deployments MUST supply a
    hitl_callback that prompts a human supervisor; if none is supplied,
    the action proceeds and the bypass is recorded in the audit log.

    AUDIT NOTE: previous version hard-coded `approved = True` and still
    advertised "human-in-the-loop". Without a callback we now log the
    bypass explicitly so reviewers can see when a human was — or was
    not — actually consulted.
    """
    action = planning_result.get("action", "monitor")
    priority = planning_result.get("priority", "P4")

    bypass_reason = None
    if action in HIGH_RISK_ACTIONS:
        if hitl_callback is None:
            bypass_reason = "no hitl_callback supplied; action auto-dispatched"
        else:
            ok, reason = hitl_callback(machine_id, action, priority)
            if not ok:
                print(f"  [HITL_REJECT] M-{machine_id} action={action} reason={reason}")
                return {"dispatched": False, "rejected_by_hitl": True, "reason": reason}

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"  [ACTION] {ts} — M-{machine_id} | {action} | {priority}"
          + (f" [HITL_BYPASS: {bypass_reason}]" if bypass_reason else ""))

    return {
        "dispatched": True,
        "rejected_by_hitl": False,
        "machine_id": machine_id,
        "action": action,
        "priority": priority,
        "hitl_bypass_reason": bypass_reason,
        "timestamp": ts,
    }


# ─────────────────────────────────────────────
# LAYER 6 — LEARNING LOOP (heuristic, disclosed)
# ─────────────────────────────────────────────
class LearningLoop:
    """
    AUDIT NOTE:
      This module previously claimed "continuous adaptive learning"
      while only adding +2 to the temperature warning threshold once.
      This rewrite keeps the same simple heuristic but is honest:
      it is a one-shot reactive bump triggered by FP rate, NOT
      adaptive learning in any meaningful ML sense.
    """
    def __init__(self, memory: ContextMemoryStore):
        self.memory = memory

    def maybe_relax_threshold_temperature(self, engine: ReasoningEngine):
        feedback = self.memory.feedback_log
        if len(feedback) < 5:
            return None
        false_positives = sum(
            1 for f in feedback
            if f["action"] in HIGH_RISK_ACTIONS and f["outcome"] == "no_failure"
        )
        fp_rate = false_positives / len(feedback)
        if fp_rate > 0.4:
            old = engine.thresholds["temperature"]["warn"]
            engine.thresholds["temperature"]["warn"] = old + 2
            return {"old_warn": old, "new_warn": old + 2, "fp_rate": fp_rate,
                    "_disclosure": "one-shot +2 nudge; not adaptive learning"}
        return None


# ════════════════════════════════════════════════════════════
# ENHANCED PIPELINE
# ════════════════════════════════════════════════════════════

def run_enhanced_pipeline(machine_ids=None, max_events=50, seed=RANDOM_SEED):
    print("\n" + "="*90)
    print(" ENHANCED AGENTIC AI PIPELINE (post-audit rewrite)")
    print(f" seed={seed}")
    print("="*90)

    rng = random.Random(seed)

    bounds = load_sensor_bounds()
    if bounds is None:
        print("[FATAL] sensor_bounds_derived.json missing. Run derive_sensor_bounds.py first.")
        return None

    # Train/test split disclosure (honest reporting)
    test_ids = bounds.get("test_machine_ids", [])
    train_ids = bounds.get("train_machine_ids", [])
    print(f"[split] derived bounds use {len(train_ids)} train machines.")
    print(f"[split] this pipeline runs over the {len(test_ids)} held-out test "
          f"machines for honest evaluation. (Override with machine_ids=...)")

    if machine_ids is None:
        machine_ids = test_ids

    kafka_consumer = KafkaStreamConsumer()
    reasoner       = ReasoningEngine()
    orchestrator   = MultiAgentOrchestrator(model_name=OLLAMA_MODEL)
    memory         = ContextMemoryStore(window_size=10)
    heuristics     = ClassicalHeuristicSuite()

    print("\n[STEP 1] Loading latest snapshot from MongoDB (test machines only)...")
    latest_data = kafka_consumer.get_latest_snapshot(machine_ids=machine_ids)
    if not latest_data:
        print("[ERROR] no data returned")
        kafka_consumer.close()
        return None

    all_events = []
    all_reasonings = []
    for doc in latest_data[:max_events]:
        event = {
            "machine_id":  doc.get('machine_id', '-'),
            "timestamp":   doc['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
                           if hasattr(doc.get('timestamp'), 'strftime') else str(doc.get('timestamp', '')),
            "temperature": doc.get('temperature', 0),
            "vibration":   doc.get('vibration', 0),
            "humidity":    doc.get('humidity', 0),
            "pressure":    doc.get('pressure', 0),
            "energy":      doc.get('energy_consumption', 0),
            "energy_consumption": doc.get('energy_consumption', 0),
            "predicted_remaining_life": doc.get('predicted_remaining_life', 200),
            "anchor_machine":  doc.get('anchor_machine', 0),
            "graph_distance":  doc.get('graph_distance', 0),
            "influence_score": doc.get('influence_score', 0),
            # Ground-truth fields are KEPT in the event ONLY for evaluation
            # (compute metrics) but are NEVER passed to the LLM agents.
            "_gt_machine_status_label": doc.get('machine_status_label', 'Normal'),
            "_gt_failure_type":         doc.get('failure_type', 'Normal'),
            "_gt_anomaly_flag":         doc.get('anomaly_flag', 0),
            "_gt_maintenance_required": doc.get('maintenance_required', 0),
            "_gt_downtime_risk":        doc.get('downtime_risk', 0),
        }
        all_events.append(event)
        all_reasonings.append(reasoner.score(event))
        memory.update(event["machine_id"], event)

    print(f"  → {len(all_events)} mesin dimuat (test split).")

    # ── Step 2: Original 4 agents on top-5 by reasoning risk ──
    print("\n[STEP 2] Top-5 by reasoning risk → Monitoring/Diagnosis/Planning/Reporting agents")
    indexed = sorted(enumerate(all_reasonings), key=lambda x: -x[1]["risk_score"])
    top5_indices = [idx for idx, _ in indexed[:5]]
    original_agent_results = []
    for i in top5_indices:
        evt = all_events[i]; rsn = all_reasonings[i]
        monitoring = orchestrator.run_monitoring_agent(evt, rsn)
        diagnosis  = orchestrator.run_diagnosis_agent(evt, monitoring, rsn)
        planning   = orchestrator.run_planning_agent(diagnosis, rsn)
        reporting  = orchestrator.run_reporting_agent(evt["machine_id"], planning, diagnosis)
        original_agent_results.append({
            "machine_id": evt["machine_id"],
            "risk_score": rsn["risk_score"],
            "severity"  : rsn["severity"],
            "monitoring": monitoring,
            "diagnosis" : diagnosis,
            "planning"  : planning,
            "reporting" : reporting,
        })
        print(f"  → M-{evt['machine_id']} | risk={rsn['risk_score']} done")

    # ── Step 3 - 7: extra agents ──
    print("\n[STEP 3] Correlation Agent")
    correlation_result = orchestrator.run_correlation_agent(all_events, all_reasonings)

    print("\n[STEP 4] Explainability Agent (top-5)")
    explainability_results = []
    for i in top5_indices:
        evt = all_events[i]; rsn = all_reasonings[i]
        xai = orchestrator.run_explainability_agent(evt, rsn, reasoner.thresholds)
        explainability_results.append({"machine_id": evt["machine_id"], "xai_result": xai})

    print("\n[STEP 5] Self-Reflection Agent (top-5; same-model, NOT multi-model consensus)")
    self_reflection_results = []
    for orig in original_agent_results:
        srf = orchestrator.run_self_reflection_agent(
            orig["monitoring"], orig["diagnosis"], orig["planning"]
        )
        self_reflection_results.append({"machine_id": orig["machine_id"], "self_reflection": srf})

    print("\n[STEP 6] Scheduling Agent (LLM-proposed)")
    scheduling_result = orchestrator.run_scheduling_agent(all_events, all_reasonings)

    print("\n[STEP 7] Pattern Learning Agent (sliding window)")
    history_per_machine = {evt["machine_id"]: memory.get_history(evt["machine_id"])
                           for evt in all_events}
    pattern_result = orchestrator.run_pattern_agent(history_per_machine)

    # ── Step 8: Classical heuristic suite (formerly mislabeled "quantum") ──
    print("\n[STEP 8] Classical heuristic suite (random_constrained_search, "
          "energy_distance_anomaly_score, graph_message_passing_score)")
    sched_input = [{"machine_id": e["machine_id"], "risk_score": r["risk_score"],
                    "predicted_rul": r["predicted_rul"]}
                   for e, r in zip(all_events, all_reasonings)]
    schedule_heuristic = heuristics.random_constrained_search(sched_input, n_samples=200, rng=rng)
    print(f"  random_constrained_search: scheduled {schedule_heuristic['total_scheduled']} "
          f"machines, score={schedule_heuristic['best_objective_score']}")

    anomaly_heuristic = heuristics.energy_distance_anomaly_score(all_events)
    print(f"  energy_distance_anomaly_score: {anomaly_heuristic['anomaly_count_at_0_5']} "
          f"machines above 0.5 / {anomaly_heuristic['n_machines']}")

    graph_heuristic = heuristics.graph_message_passing_score(all_events, n_rounds=3)
    print(f"  graph_message_passing_score: top "
          f"{min(3, len(graph_heuristic['ranked_nodes_top10']))} = "
          f"{[(n['machine_id'], n['graph_influence_score']) for n in graph_heuristic['ranked_nodes_top10'][:3]]}")

    # ── Step 9: Honest detection metrics on this test split ──
    print("\n[STEP 9] Honest detection metrics on test machines")
    tp = fp = fn = tn = 0
    for evt, rsn in zip(all_events, all_reasonings):
        gt_attn = (evt["_gt_machine_status_label"] in ("Warning", "Fault")
                   or evt["_gt_failure_type"] != "Normal"
                   or evt["_gt_anomaly_flag"] == 1
                   or evt["_gt_maintenance_required"] == 1)
        pred_attn = rsn["severity"] != "LOW"
        if pred_attn and gt_attn: tp += 1
        elif pred_attn and not gt_attn: fp += 1
        elif not pred_attn and gt_attn: fn += 1
        else: tn += 1
    total = max(tp + fp + fn + tn, 1)
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)
    accuracy  = (tp + tn) / total
    print(f"  TP={tp} FP={fp} FN={fn} TN={tn}  precision={precision:.4f}  recall={recall:.4f}  f1={f1:.4f}  acc={accuracy:.4f}")

    kafka_consumer.close()

    out = {
        "_audit_disclosure": (
            "Pipeline rewrite (post-audit): no quantum-labelled heuristics; "
            "no ground-truth fields passed to LLM agents; bounds derived from "
            "training subset; metrics reported on held-out test split."
        ),
        "seed":          seed,
        "timestamp":     datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "split":         {"train_ids": train_ids, "test_ids": test_ids},
        "n_test_machines": len(all_events),
        "events":        [{k: v for k, v in e.items() if not k.startswith("_gt_")} for e in all_events],
        "reasonings":    all_reasonings,
        "agents": {
            "original_top5":      original_agent_results,
            "correlation":        correlation_result,
            "explainability":     explainability_results,
            "self_reflection":    self_reflection_results,
            "scheduling":         scheduling_result,
            "pattern_learning":   pattern_result,
        },
        "classical_heuristics": {
            "schedule":   schedule_heuristic,
            "anomaly":    anomaly_heuristic,
            "graph":      graph_heuristic,
        },
        "detection_metrics_on_test_split": {
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "accuracy":  round(accuracy, 4),
            "n":         total,
        },
    }
    out_path = "enhanced_pipeline_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[SAVED] {out_path}")
    print("[DONE] Pipeline complete.")
    return out


if __name__ == "__main__":
    import sys
    seed = RANDOM_SEED
    if "--seed" in sys.argv:
        try:
            seed = int(sys.argv[sys.argv.index("--seed") + 1])
        except (IndexError, ValueError):
            pass
    if "--machine" in sys.argv:
        try:
            mid = int(sys.argv[sys.argv.index("--machine") + 1])
            run_enhanced_pipeline(machine_ids=[mid], max_events=1, seed=seed)
        except (IndexError, ValueError):
            print("Usage: --machine <id>")
    else:
        run_enhanced_pipeline(max_events=50, seed=seed)
