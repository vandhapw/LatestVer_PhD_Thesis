# ============================================================
# AGENTIC AI — KAFKA STREAM INTEGRATION (post-audit rewrite)
# ============================================================
#
# AUDIT NOTE:
#   The previous version was titled "QUANTUM AGENTIC AI". The
#   "Layer 2 quantum encoding" is a weighted sum of normalized
#   sensor readings (Σ w_i * x_i) that was labeled "Pauli-Z
#   composite expectation". The "Layer 5 QAOA scheduler" was
#   `np.argsort(-cost_matrix)` (greedy sort) inside a function
#   named `_run_qaoa_optimization`. Neither invoked Qiskit even
#   when Qiskit was importable.
#
#   This rewrite renames the components to honest labels:
#     - PauliZRiskEncoder         → WeightedSensorRiskEncoder
#     - QAOAMaintenanceOptimizer  → GreedyMaintenanceScheduler
#     - HybridReasoningEngine     → BlendedReasoningEngine
#     - QuantumLearningLoop       → ReactiveThresholdNudger
#     - "quantum_risk_score"      → "weighted_sensor_score"
#     - "quantum_label"           → "weighted_sensor_label"
#     - "qaoa_slot"               → "greedy_priority_slot"
#   The `algorithm` field in saved JSON is updated accordingly.
#
#   SENSOR_BOUNDS hardcode is removed; values come from
#   sensor_bounds_derived.json via config.get_normalization_range.
#
#   The blend weight is renamed from QUANTUM_WEIGHT → BLEND_WEIGHT
#   to reflect what it actually controls (a convex combination of
#   two classical scores).
#
#   A real Qiskit-based implementation is left as future work and
#   would belong in a separate module.
# ============================================================

import json
import time
import datetime
import numpy as np
from collections import deque
from pymongo import MongoClient
from ollama import Client

from config import (
    OLLAMA_API_KEY, MONGODB_URI, MONGO_DB, MONGO_COLLECTION,
    OLLAMA_MODEL, RUL_LOW, RUL_MODERATE, RISK_HIGH_AT, RISK_MEDIUM_AT,
    SENSOR_FEATURES, get_normalization_range, get_warn_crit_thresholds,
    load_sensor_bounds,
)

api_key = OLLAMA_API_KEY  # backward-compat for downstream uses

# AUDIT NOTE: Qiskit imports removed. The previous QISKIT_AVAILABLE
# branch never produced quantum behavior; keeping the imports just
# implied capability the code does not use.
QISKIT_AVAILABLE = False

POLL_INTERVAL_SEC    = 60
RISK_THRESHOLD       = RISK_HIGH_AT  # use central policy
RUL_THRESHOLD        = RUL_LOW

# Blend weight between classical threshold score and weighted-sensor
# score. Renamed from QUANTUM_WEIGHT to be honest — both inputs are
# classical.
BLEND_WEIGHT         = 0.4
QUANTUM_WEIGHT       = BLEND_WEIGHT  # backward-compat alias

# ─────────────────────────────────────────────
# LAYER 1: DATA INGESTION (Kafka Stream via MongoDB)
# ─────────────────────────────────────────────
class KafkaStreamConsumer:
    """
    Real-time consumer that fetches latest predictions from MongoDB.
    MongoDB receives data from Kafka stream (digital twin output).
    """
    def __init__(self, mongo_uri=MONGODB_URI, db_name=MONGO_DB, collection_name=MONGO_COLLECTION):
        self.client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10000)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        
        # Test connection
        try:
            self.collection.count_documents({})
            print(f"[CONNECTED] MongoDB: {mongo_uri}")
            print(f"[CONNECTED] Collection: {collection_name}")
        except Exception as e:
            print(f"[ERROR] MongoDB connection failed: {e}")
            raise
    
    def get_latest_snapshot(self, machine_ids=None):
        """Get latest prediction for all machines or specific machines."""
        try:
            query = {}
            if machine_ids:
                query['machine_id'] = {'$in': machine_ids}
            
            # Simple approach: get latest docs using find and sort
            cursor = self.collection.find(query).sort('timestamp', -1).limit(250)
            
            # Get latest per machine
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
            import traceback
            traceback.print_exc()
            return []
    
    def stream_realtime(self, machine_ids=None, poll_interval=POLL_INTERVAL_SEC):
        """Generator that polls MongoDB every N seconds for new data."""
        last_timestamps = {}
        
        while True:
            try:
                latest_data = self.get_latest_snapshot(machine_ids)
                
                for doc in latest_data:
                    machine_id = doc['machine_id']
                    current_ts = doc['timestamp']
                    
                    # Only yield if this is newer than what we've seen
                    if machine_id not in last_timestamps or current_ts > last_timestamps[machine_id]:
                        last_timestamps[machine_id] = current_ts
                        
                        # Transform to event format
                        event = {
                            "machine_id": machine_id,
                            "timestamp": current_ts.strftime("%Y-%m-%d %H:%M:%S") if hasattr(current_ts, 'strftime') else str(current_ts),
                            "temperature": float(doc['temperature']),
                            "vibration": float(doc['vibration']),
                            "humidity": float(doc['humidity']),
                            "pressure": float(doc['pressure']),
                            "energy": float(doc['energy_consumption']),
                            "source": doc.get('source', 'KAFKA_STREAM'),
                            "machine_status": int(doc.get('machine_status', 0)),
                            "machine_status_label": doc.get('machine_status_label', 'Normal'),
                            "anomaly_flag": int(doc.get('anomaly_flag', 0)),
                            "anomaly_flag_label": doc.get('anomaly_flag_label', 'Normal'),
                            "failure_type": doc.get('failure_type', 'Normal'),
                            "downtime_risk": float(doc.get('downtime_risk', 0)),
                            "maintenance_required": int(doc.get('maintenance_required', 0)),
                            "predicted_remaining_life": float(doc.get('predicted_remaining_life', 200)),
                            "anchor_machine": int(doc.get('anchor_machine', 40)),
                            "graph_distance": int(doc.get('graph_distance', 0)),
                            "influence_score": float(doc.get('influence_score', 0))
                        }
                        
                        yield event
                
                time.sleep(poll_interval)
                
            except Exception as e:
                print(f"[ERROR] Stream polling failed: {e}")
                time.sleep(5)
    
    def close(self):
        self.client.close()
        print("[DISCONNECTED] MongoDB connection closed")

# ─────────────────────────────────────────────
# LAYER 2: WEIGHTED SENSOR ENCODING (renamed from "Quantum Encoding")
# ─────────────────────────────────────────────
class WeightedSensorRiskEncoder:
    """
    AUDIT NOTE (renamed from QuantumSensorEncoder):
        Computes a normalized weighted sum of 5 sensor channels in
        [-1, 1] and maps to a [0, 1] risk score. The previous version
        called the weighted sum "composite Pauli-Z expectation" and
        the score "quantum_risk_score". No quantum operation occurs.
        Bounds are now derived from data via config.
    """
    def __init__(self):
        self.simulator = None
        # Domain weights — disclosed as a heuristic, not learned.
        self.weights = {
            'temperature': 0.30,
            'vibration':   0.25,
            'humidity':    0.15,
            'pressure':    0.15,
            'energy':      0.15,
        }

    def normalize_sensor(self, value: float, sensor_type: str) -> float:
        """Normalize to [-1, 1] using p05/p95 bounds from training set."""
        cfg_name = "energy_consumption" if sensor_type == "energy" else sensor_type
        min_val, max_val = get_normalization_range(cfg_name)
        if max_val <= min_val:
            return 0.0
        normalized = 2 * (value - min_val) / (max_val - min_val) - 1
        return float(np.clip(normalized, -1, 1))

    def encode_event(self, event: dict) -> dict:
        sensors = {
            'temperature': event['temperature'],
            'vibration':   event['vibration'],
            'humidity':    event['humidity'],
            'pressure':    event['pressure'],
            'energy':      event.get('energy', event.get('energy_consumption', 0)),
        }
        normalized = {k: self.normalize_sensor(v, k) for k, v in sensors.items()}
        composite = sum(normalized[k] * self.weights[k] for k in normalized)
        weighted_score = (composite + 1) / 2

        rul = event.get('predicted_remaining_life', 200)
        if rul < RUL_THRESHOLD:
            weighted_score = min(1.0, weighted_score + 0.3)
        elif rul < RUL_MODERATE:
            weighted_score = min(1.0, weighted_score + 0.15)

        return {
            'normalized_sensors':         normalized,
            'composite_score':            composite,
            'weighted_sensor_score':      round(weighted_score, 4),
            'weighted_sensor_label':      self._label(composite),
            'priority_rank':              None,    # set by GreedyMaintenanceScheduler
            'greedy_priority_slot':       None,
            'greedy_priority_slot_label': 'Not scheduled',
            '_disclosure': "Classical weighted sum + RUL bump. NOT quantum.",
        }

    def _label(self, composite_score: float) -> str:
        if composite_score < -0.5:
            return "STABLE"
        elif composite_score < 0:
            return "SLIGHT_DEVIATION"
        elif composite_score < 0.5:
            return "MODERATE_RISK"
        else:
            return "CRITICAL_STATE"


class GreedyMaintenanceScheduler:
    """
    AUDIT NOTE (renamed from QAOAMaintenanceOptimizer):
        Sorts machines by a cost score and groups them into
        time-of-day slots. The previous version used the same
        np.argsort logic inside `_run_qaoa_optimization` and labeled
        the output algorithm as "QAOA". Renamed to reflect what it
        actually does. A real QAOA implementation belongs in a
        separate module.
    """
    def __init__(self, max_concurrent=3):
        self.max_concurrent = max_concurrent

    def build_cost_matrix(self, events: list) -> np.ndarray:
        """
        Build cost matrix for maintenance scheduling.
        Higher cost = higher priority for maintenance.
        """
        costs = []
        for event in events:
            cost = 0.0
            
            # Base cost from RUL (inverse relationship)
            rul = event.get('predicted_remaining_life', 200)
            rul_cost = max(0, (200 - rul) / 200)
            cost += rul_cost * 0.4
            
            # Cost from failure type
            failure_type = event.get('failure_type', 'Normal')
            if failure_type != 'Normal':
                cost += 0.3
            
            # Cost from anomaly flag
            if event.get('anomaly_flag', 0) == 1:
                cost += 0.2
            
            # Cost from downtime risk
            cost += event.get('downtime_risk', 0) * 0.1
            
            costs.append(min(1.0, cost))
        
        return np.array(costs)
    
    def optimize_schedule(self, events: list, max_concurrent=3) -> dict:
        """Greedy schedule by cost. Honest label: greedy_sort, not QAOA."""
        if len(events) == 0:
            return {}
        cost_matrix = self.build_cost_matrix(events)
        schedule = self._greedy_sort(cost_matrix, max_concurrent)
        result = {}
        for i, event in enumerate(events):
            mid = event['machine_id']
            result[mid] = {
                'priority_rank':              schedule[i]['rank'],
                'greedy_priority_slot':       schedule[i]['slot'],
                'greedy_priority_slot_label': schedule[i]['slot_label'],
                'cost_score':                 float(cost_matrix[i]),
            }
        result['_algorithm']  = 'greedy_sort_by_cost'
        result['_disclosure'] = "np.argsort on a hand-tuned cost vector. NOT QAOA."
        return result

    def _greedy_sort(self, cost_matrix: np.ndarray, max_concurrent: int) -> list:
        n = len(cost_matrix)
        sorted_indices = np.argsort(-cost_matrix)
        rank_of = {int(idx): rank for rank, idx in enumerate(sorted_indices)}
        schedule = []
        for i in range(n):
            rank = rank_of[i]
            slot = rank // max_concurrent
            if slot == 0:   slot_label = "IMMEDIATE (0-4h)"
            elif slot == 1: slot_label = "SOON (4-12h)"
            elif slot == 2: slot_label = "TODAY (12-24h)"
            else:           slot_label = "LATER (>24h)"
            schedule.append({
                'rank':           rank + 1,
                'slot':           slot,
                'slot_label':     slot_label,
                'machine_index':  i,
            })
        return schedule


# Backward-compat alias (so legacy importers don't break)
QuantumSensorEncoder      = WeightedSensorRiskEncoder
QAOAMaintenanceOptimizer  = GreedyMaintenanceScheduler


# ─────────────────────────────────────────────
# LAYER 3: BLENDED REASONING (renamed from "Hybrid Quantum")
# ─────────────────────────────────────────────
class BlendedReasoningEngine:
    """
    AUDIT NOTE (renamed from HybridReasoningEngine):
        Convex blend of two CLASSICAL scores: a threshold-based risk
        and a weighted-sensor risk. Previously labeled the second
        score "quantum risk". Both come from classical computation.

    AUDIT NOTE (label leakage):
        Previous version added 0.20 for anomaly_flag and 0.15 for
        failure_type — both are ground-truth fields. Removed.
    """
    def __init__(self, blend_weight=BLEND_WEIGHT):
        self.blend_weight = blend_weight
        self.threshold_weight = 1.0 - blend_weight
        self.thresholds = get_warn_crit_thresholds()
    
    def score(self, event: dict, sensor_context: dict) -> dict:
        """Calculate blended risk score from threshold + weighted-sensor."""
        threshold_score = self._threshold_score(event)
        ws_score = sensor_context.get('weighted_sensor_score',
                    sensor_context.get('quantum_risk_score', 0.0))  # backward-compat
        blended_score = (self.threshold_weight * threshold_score['risk_score']
                         + self.blend_weight * ws_score)
        if blended_score >= RISK_HIGH_AT:   severity = "HIGH"
        elif blended_score >= RISK_MEDIUM_AT: severity = "MEDIUM"
        else:                                 severity = "LOW"

        all_flags = threshold_score['flags'].copy()
        composite = sensor_context.get('composite_score',
                    sensor_context.get('composite_z', 0))
        if composite > 0.5:
            all_flags.append(f"WEIGHTED_SENSOR_CRITICAL(score={composite:.3f})")

        return {
            'threshold_risk':       round(threshold_score['risk_score'], 3),
            'weighted_sensor_risk': round(ws_score, 3),
            'blended_risk':         round(blended_score, 3),
            'severity':             severity,
            'flags':                all_flags,
            'predicted_rul':        event.get('predicted_remaining_life', 200),
            # Backward-compat aliases
            'classical_risk':       round(threshold_score['risk_score'], 3),
            'quantum_risk':         round(ws_score, 3),
            'hybrid_risk':          round(blended_score, 3),
        }

    def _threshold_score(self, event: dict) -> dict:
        """
        Threshold-based risk on sensors + RUL ONLY.
        AUDIT NOTE: previous version added 0.20 for anomaly_flag and
        0.15 for failure_type — both are GROUND TRUTH. Removed.
        """
        score = 0.0
        flags = []
        for feature, thr in self.thresholds.items():
            val = event.get(feature, event.get("energy" if feature == "energy_consumption" else feature, 0))
            if not isinstance(val, (int, float)) or thr["warn"] == float("inf"):
                continue
            if val >= thr["crit"]:
                score += 0.35
                flags.append(f"{feature}=CRITICAL({val:.2f})")
            elif val >= thr["warn"]:
                score += 0.15
                flags.append(f"{feature}=WARN({val:.2f})")
        rul = event.get('predicted_remaining_life', 200)
        if rul < RUL_LOW:
            score += 0.30
            flags.append(f"RUL=LOW({rul})")
        elif rul < RUL_MODERATE:
            score += 0.10
            flags.append(f"RUL=MODERATE({rul})")
        return {'risk_score': min(score, 1.0), 'flags': flags}

    def should_blend_override(self, threshold_severity: str, sensor_context: dict) -> tuple:
        """If threshold scoring says LOW but the weighted-sensor score
        signals emerging risk, escalate to inspect."""
        if threshold_severity != "LOW":
            return False, ""
        composite = sensor_context.get('composite_score',
                    sensor_context.get('composite_z', 0))
        ws_score  = sensor_context.get('weighted_sensor_score',
                    sensor_context.get('quantum_risk_score', 0))
        if composite > 0.6:
            return True, f"Weighted-sensor composite high (score={composite:.3f})"
        if ws_score > 0.7:
            return True, f"Weighted-sensor risk high ({ws_score:.2f})"
        return False, ""

    # Backward-compat alias for legacy callers
    should_quantum_override = should_blend_override


# Backward-compat alias
HybridReasoningEngine = BlendedReasoningEngine

# ─────────────────────────────────────────────
# LAYER 4: QUANTUM-ENHANCED MULTI-AGENT ORCHESTRATOR
# ─────────────────────────────────────────────
def get_ollama_client():
    return Client(
        host="https://ollama.com",
        headers={"Authorization": "Bearer " + api_key}
    )

SYSTEM_PROMPT_MONITORING = """
You are a machine monitoring agent for a smart factory.
Inputs you receive include sensor readings, threshold-derived flags,
and a weighted-sensor risk score (a classical heuristic). Return JSON
with these keys ONLY:
{
  "risk_level": "LOW|MEDIUM|HIGH|CRITICAL",
  "anomaly_detected": true|false,
  "anomaly_type": "thermal|vibration|energy|combined|none",
  "confidence": 0.0-1.0
}
Return valid JSON only.
"""

SYSTEM_PROMPT_DIAGNOSIS = """
You are a root-cause diagnosis agent.
Given sensor context, threshold flags, and a weighted-sensor risk
score, return JSON:
{
  "probable_cause": "bearing_wear|overheating|electrical|lubrication|normal|unknown",
  "affected_component": "motor|bearing|pump|actuator|unknown",
  "urgency_hours": integer (0-72)
}
Return valid JSON only.
"""

SYSTEM_PROMPT_PLANNING = """
You are a maintenance planning agent.
Given diagnosis, risk context, and a greedy-priority slot, decide
optimal action:
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
Generate a short maintenance alert in Bahasa Indonesia.
Max 3 sentences. Plain text only.
"""


def _safe_event_view(event):
    """Return the LLM-safe subset of an event (no ground-truth fields)."""
    return {
        "machine_id":  event.get("machine_id"),
        "timestamp":   event.get("timestamp"),
        "temperature": event.get("temperature"),
        "vibration":   event.get("vibration"),
        "humidity":    event.get("humidity"),
        "pressure":    event.get("pressure"),
        "energy":      event.get("energy", event.get("energy_consumption")),
        "predicted_remaining_life": event.get("predicted_remaining_life"),
        "anchor_machine":  event.get("anchor_machine"),
        "graph_distance":  event.get("graph_distance"),
        "influence_score": event.get("influence_score"),
    }


class BlendedAgentOrchestrator:
    """
    AUDIT NOTE (renamed from QuantumMultiAgentOrchestrator):
        Same class — but the agent prompts no longer claim quantum
        sensors / quantum measurements / QAOA schedule. Inputs to
        the LLM are filtered through `_safe_event_view` to ensure
        no ground-truth fields are passed in.
    """
    def __init__(self, model_name=OLLAMA_MODEL):
        self.model = model_name
        self.client = get_ollama_client()

    def _call(self, system_prompt, user_msg):
        try:
            resp = self.client.chat(
                model=self.model,
                messages=[
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

    def run_monitoring_agent(self, event, reasoning_result, sensor_context):
        msg = (f"Sensor event: {json.dumps(_safe_event_view(event), indent=2)}\n"
               f"Threshold risk: {json.dumps(reasoning_result, indent=2)}\n"
               f"Weighted-sensor context:\n"
               f"  composite_score: {sensor_context.get('composite_score', 0):.4f}\n"
               f"  weighted_sensor_score: {sensor_context.get('weighted_sensor_score', 0):.4f}\n"
               f"  label: {sensor_context.get('weighted_sensor_label', 'UNKNOWN')}\n"
               f"  greedy_priority_slot: {sensor_context.get('greedy_priority_slot_label', 'N/A')}")
        return self._call(SYSTEM_PROMPT_MONITORING, msg)

    def run_diagnosis_agent(self, event, monitoring_result, reasoning_result, sensor_context):
        msg = (f"Sensor event: {json.dumps(_safe_event_view(event), indent=2)}\n"
               f"Monitoring result: {json.dumps(monitoring_result, indent=2)}\n"
               f"Threshold flags: {reasoning_result.get('flags', [])}\n"
               f"Weighted-sensor composite: {sensor_context.get('composite_score', 0):.4f}")
        return self._call(SYSTEM_PROMPT_DIAGNOSIS, msg)

    def run_planning_agent(self, diagnosis_result, reasoning_result, sensor_context):
        msg = (f"Diagnosis: {json.dumps(diagnosis_result, indent=2)}\n"
               f"Blended risk score: {reasoning_result.get('blended_risk', reasoning_result.get('hybrid_risk', 0))}\n"
               f"Severity: {reasoning_result.get('severity', 'UNKNOWN')}\n"
               f"Predicted RUL: {reasoning_result.get('predicted_rul', 200)}h\n"
               f"Greedy priority slot: {sensor_context.get('greedy_priority_slot_label', 'N/A')}")
        return self._call(SYSTEM_PROMPT_PLANNING, msg)

    def run_reporting_agent(self, machine_id, planning_result, diagnosis_result, event, sensor_context):
        msg = (f"Machine ID: M-{machine_id}\n"
               f"Diagnosis: {json.dumps(diagnosis_result, indent=2)}\n"
               f"Action plan: {json.dumps(planning_result, indent=2)}")
        return self._call(SYSTEM_PROMPT_REPORTING, msg)


# Backward-compat alias
QuantumMultiAgentOrchestrator = BlendedAgentOrchestrator

# ─────────────────────────────────────────────
# LAYER 5: ACTION LAYER WITH QUANTUM CONTEXT
# ─────────────────────────────────────────────
HIGH_RISK_ACTIONS = {"immediate_shutdown", "schedule_maintenance"}

def execute_action(machine_id, planning_result, report, quantum_context=None):
    action = planning_result.get("action", "monitor")
    priority = planning_result.get("priority", "P4")
    notify = planning_result.get("notify_supervisor", False)
    quantum_driven = planning_result.get("quantum_driven", False)
    
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*70}")
    print(f"[ACTION] {ts} — Machine M-{machine_id}")
    print(f"  Action     : {action}")
    print(f"  Priority   : {priority}")
    print(f"  Quantum    : {'Yes (Q-overridden)' if quantum_driven else 'No'}")
    print(f"  Report     : {report.get('raw', report) if isinstance(report, dict) else report}")
    print(f"{'='*70}")
    
    if action == "immediate_shutdown":
        print(f"  >> [CMMS] Emergency stop triggered for M-{machine_id}")
        print(f"  >> [ALERT] SMS/email sent to maintenance team")
    elif action == "schedule_maintenance":
        print(f"  >> [CMMS] Work order created for M-{machine_id} ({priority})")
        slot = (quantum_context.get('greedy_priority_slot_label') or
                quantum_context.get('qaoa_slot_label') or 'N/A') if quantum_context else 'N/A'
        print(f"  >> [GREEDY-SCHED] Slot: {slot}")
    elif action == "inspect":
        if quantum_driven:
            print(f"  >> [BLEND-OVERRIDE] Inspection triggered by weighted-sensor signal")
        print(f"  >> [SCHEDULER] Inspection scheduled for M-{machine_id} within 4h")
    elif action == "monitor":
        print(f"  >> [MONITOR] Enhanced monitoring activated for M-{machine_id}")
    if notify:
        print(f"  >> [NOTIFY] Supervisor alert sent")


# ─────────────────────────────────────────────
# LAYER 6: REACTIVE THRESHOLD NUDGER (renamed from "Quantum Learning Loop")
# ─────────────────────────────────────────────
class ReactiveThresholdNudger:
    """
    AUDIT NOTE (renamed from QuantumLearningLoop):
        This class previously claimed "quantum-classical retraining
        pipeline". It has no retraining and no ML at all — it just
        nudges the blend weight up by 0.05 once when a heuristic
        accuracy proxy crosses 60%. Renamed to be honest. Backward-
        compat alias `QuantumLearningLoop` is kept.
    """
    def __init__(self):
        self.feedback_log = []

    def log_feedback(self, machine_id, action, outcome, threshold_risk, weighted_sensor_risk):
        self.feedback_log.append({
            'machine_id':            machine_id,
            'action':                action,
            'outcome':               outcome,
            'threshold_risk':        threshold_risk,
            'weighted_sensor_risk':  weighted_sensor_risk,
            'timestamp':             str(datetime.datetime.now()),
        })

    def maybe_nudge_blend_weight(self, engine):
        if len(self.feedback_log) < 10:
            return None
        wins = sum(1 for f in self.feedback_log
                   if f['outcome'] == 'failure_detected'
                   and f['weighted_sensor_risk'] > f['threshold_risk'])
        proxy_accuracy = wins / len(self.feedback_log)
        if proxy_accuracy > 0.6:
            old = engine.blend_weight
            new = min(0.6, old + 0.05)
            engine.blend_weight = new
            engine.threshold_weight = 1.0 - new
            print(f"[NUDGE] proxy_accuracy={proxy_accuracy:.1%} → blend_weight {old:.2f} → {new:.2f}")
            return {"old": old, "new": new, "_disclosure": "one-shot +0.05 nudge; not learning"}
        return None

    def trigger_retrain(self):
        print(f"[NUDGE] {len(self.feedback_log)} feedback records collected. "
              f"Retraining is NOT implemented in this module.")


# Backward-compat alias
QuantumLearningLoop = ReactiveThresholdNudger


# ─────────────────────────────────────────────
# MAIN PIPELINE (renamed from "QUANTUM" pipeline)
# ─────────────────────────────────────────────
def run_blended_agentic_pipeline(machine_ids=None, poll_interval=POLL_INTERVAL_SEC, max_events=None):
    """
    Real-time agentic pipeline using a blended threshold + weighted-sensor
    risk score. AUDIT NOTE: previous name `run_quantum_agentic_pipeline`
    suggested quantum computation that does not occur. Renamed.
    """
    print("\n" + "="*90)
    print(" BLENDED AGENTIC AI PIPELINE — REAL-TIME (post-audit rewrite)")
    print("="*90)
    print(f" Collection:        {MONGO_DB}.{MONGO_COLLECTION}")
    print(f" Poll interval:     {poll_interval}s")
    print(f" Machines:          {machine_ids if machine_ids else 'ALL'}")
    print(f" Blend weight:      {BLEND_WEIGHT} (threshold weight: {1-BLEND_WEIGHT})")
    print(" AUDIT NOTE:        no Qiskit / no quantum simulator used.")
    print("="*90 + "\n")

    kafka_consumer = KafkaStreamConsumer()
    encoder        = WeightedSensorRiskEncoder()
    scheduler      = GreedyMaintenanceScheduler()
    reasoner       = BlendedReasoningEngine()
    orchestrator   = BlendedAgentOrchestrator()
    nudger         = ReactiveThresholdNudger()

    event_count = 0
    sched_cycle = 0
    latest_events = {}

    try:
        for event in kafka_consumer.stream_realtime(machine_ids, poll_interval):
            event_count += 1
            machine_id = event['machine_id']
            latest_events[machine_id] = event

            print(f"\n[INGEST] {event['timestamp']} — M-{machine_id}")
            print(f"  Temp: {event['temperature']:.1f}°C | Vib: {event['vibration']:.1f} | "
                  f"Humi: {event['humidity']:.1f}% | Energy: {event['energy']:.1f} kWh | "
                  f"RUL: {event['predicted_remaining_life']:.1f}h")

            # LAYER 2: Weighted sensor encoding
            sensor_context = encoder.encode_event(event)
            print(f"  [WEIGHTED-SENSOR] composite={sensor_context['composite_score']:+.4f} | "
                  f"score={sensor_context['weighted_sensor_score']:.3f} | "
                  f"label={sensor_context['weighted_sensor_label']}")

            # LAYER 3: Blended risk scoring
            reasoning = reasoner.score(event, sensor_context)
            print(f"  [RISK] Threshold: {reasoning['threshold_risk']:.3f} | "
                  f"WeightedSensor: {reasoning['weighted_sensor_risk']:.3f} | "
                  f"Blended: {reasoning['blended_risk']:.3f} ({reasoning['severity']})")

            override, reason_str = reasoner.should_blend_override(reasoning['severity'], sensor_context)
            if override:
                print(f"  [BLEND-OVERRIDE] {reason_str}")

            if reasoning['severity'] == "LOW" and not override:
                print("  [SKIP] LOW risk + no override → passive monitoring")
                nudger.log_feedback(machine_id, "monitor", "no_failure",
                                    reasoning['threshold_risk'], reasoning['weighted_sensor_risk'])
                if max_events and event_count >= max_events:
                    break
                continue

            print("  [AGENT] Running monitoring/diagnosis/planning/reporting...")
            monitoring = orchestrator.run_monitoring_agent(event, reasoning, sensor_context)
            diagnosis  = orchestrator.run_diagnosis_agent(event, monitoring, reasoning, sensor_context)
            planning   = orchestrator.run_planning_agent(diagnosis, reasoning, sensor_context)
            if override and planning.get("action") == "monitor":
                planning["action"]   = "inspect"
                planning["priority"] = "P3"
                planning["override_driven"] = True
                planning["reason"] = (planning.get("reason", "") + f" | override: {reason_str}").strip(" |")
            report = orchestrator.run_reporting_agent(machine_id, planning, diagnosis, event, sensor_context)
            execute_action(machine_id, planning, report, sensor_context)

            nudger.log_feedback(machine_id, planning.get("action", "monitor"),
                                "no_failure",
                                reasoning['threshold_risk'], reasoning['weighted_sensor_risk'])

            sched_cycle += 1
            if sched_cycle % 10 == 0 and len(latest_events) >= 10:
                print(f"\n[GREEDY-SCHED] Running scheduler for {len(latest_events)} machines...")
                schedule = scheduler.optimize_schedule(list(latest_events.values()))
                top = sorted(((mid, s) for mid, s in schedule.items() if isinstance(s, dict)),
                             key=lambda x: x[1]['priority_rank'])[:5]
                for mid, s in top:
                    print(f"  M-{mid}: rank #{s['priority_rank']} | {s['greedy_priority_slot_label']}")

            if max_events and event_count >= max_events:
                print(f"\n[DONE] reached max events: {max_events}")
                break

    except KeyboardInterrupt:
        print(f"\n[STOP] interrupted. Processed {event_count} events.")
    finally:
        nudger.maybe_nudge_blend_weight(reasoner)
        nudger.trigger_retrain()
        kafka_consumer.close()
        print("\n" + "="*90)
        print(f"[SUMMARY] events processed: {event_count} | feedback rows: {len(nudger.feedback_log)} | "
              f"final blend_weight: {reasoner.blend_weight:.2f}")
        print("="*90)
    return nudger


# Backward-compat alias
run_quantum_agentic_pipeline = run_blended_agentic_pipeline

# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    print("\n" + "="*90)
    print(" BLENDED AGENTIC AI — KAFKA STREAM INTEGRATION (post-audit)")
    print("="*90)
    print("\nUsage:")
    print("  python quantum_agenticai_kafka.py              # all machines, blended scoring")
    print("  python quantum_agenticai_kafka.py --machine 40 # single machine")
    print("  python quantum_agenticai_kafka.py --threshold-only  # disable weighted-sensor blend")
    print("="*90 + "\n")

    if len(sys.argv) > 1:
        if sys.argv[1] in ("--threshold-only", "--classical"):
            print("[MODE] Threshold-only mode (blend weight = 0)")
            BLEND_WEIGHT = 0.0
            run_blended_agentic_pipeline(max_events=50)
        elif sys.argv[1] == "--machine" and len(sys.argv) > 2:
            machine_id = int(sys.argv[2])
            print(f"[MODE] Real-time monitoring for Machine {machine_id}")
            run_blended_agentic_pipeline(machine_ids=[machine_id], max_events=10)
        else:
            print("[MODE] Real-time monitoring for ALL machines")
            run_blended_agentic_pipeline(max_events=50)
    else:
        print("[MODE] Real-time monitoring for ALL machines")
        run_blended_agentic_pipeline(max_events=50)
