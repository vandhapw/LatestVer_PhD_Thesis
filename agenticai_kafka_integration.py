# ============================================================
# AGENTIC AI — Smart Manufacturing with REAL-TIME KAFKA STREAMING
# ============================================================
# Integration with Digital Twin Kafka Stream
# 
# Data Flow:
#   Plalion Sensor → Kafka → Digital Twin → Kafka → MongoDB
#                                            ↓
#                                    This Agentic AI System
#
# Layers:
#  1. Data Ingestion (REAL MongoDB query from Kafka stream)
#  2. Context / Memory Store
#  3. Reasoning Engine (T-GCN forecast + risk scoring)
#  4. Multi-Agent Orchestrator (Ollama cloud)
#  5. Action Layer
#  6. Learning Loop
# ============================================================

import json
import time
import datetime
import threading
from collections import deque
from pymongo import MongoClient
from ollama import Client

from config import (
    OLLAMA_API_KEY, MONGODB_URI, MONGO_DB, MONGO_COLLECTION, KAFKA_TOPIC,
    OLLAMA_MODEL, RUL_LOW, RISK_HIGH_AT, get_warn_crit_thresholds,
)

api_key = OLLAMA_API_KEY
RISK_THRESHOLD    = RISK_HIGH_AT
RUL_THRESHOLD     = RUL_LOW
POLL_INTERVAL_SEC = 60

# ─────────────────────────────────────────────
# LAYER 1 — DATA INGESTION (REAL KAFKA STREAM VIA MONGODB)
# ─────────────────────────────────────────────
class KafkaStreamConsumer:
    """
    Real-time consumer that fetches latest predictions from MongoDB.
    MongoDB receives data from Kafka stream (digital twin output).
    
    Architecture:
    Plalion Device → Kafka Producer → Kafka → Digital Twin → Kafka → MongoDB ← This Consumer
    """
    def __init__(self, mongo_uri=MONGODB_URI, db_name=MONGO_DB, collection_name=MONGO_COLLECTION):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        print(f"[CONNECTED] MongoDB: {mongo_uri}")
        print(f"[CONNECTED] Collection: {collection_name}")
    
    def get_latest_snapshot(self, machine_ids=None):
        """
        Get latest prediction for all machines or specific machines.
        Returns list of latest documents sorted by timestamp.
        """
        try:
            query = {}
            if machine_ids:
                query['machine_id'] = {'$in': machine_ids}
            
            # Simple approach: get latest docs using find and sort
            # Group by machine_id in Python to avoid aggregation issues
            cursor = self.collection.find(query).sort('timestamp', -1).limit(100)
            
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
    
    def get_latest_for_machine(self, machine_id):
        """Get latest prediction for a specific machine."""
        doc = self.collection.find_one(
            {'machine_id': machine_id},
            sort=[('processed_at', -1)]
        )
        return doc
    
    def stream_realtime(self, machine_ids=None, poll_interval=POLL_INTERVAL_SEC):
        """
        Generator that polls MongoDB every N seconds for new data.
        Simulates Kafka consumer behavior but reads from MongoDB (faster).
        """
        last_timestamps = {}
        
        while True:
            try:
                latest_data = self.get_latest_snapshot(machine_ids)
                
                for doc in latest_data:
                    machine_id = doc['machine_id']
                    current_ts = doc['processed_at']
                    
                    # Only yield if this is newer than what we've seen
                    if machine_id not in last_timestamps or current_ts > last_timestamps[machine_id]:
                        last_timestamps[machine_id] = current_ts
                        
                        # Transform to event format expected by agentic pipeline
                        event = {
                            "machine_id": machine_id,
                            "timestamp": doc['timestamp'].strftime("%Y-%m-%d %H:%M:%S") if hasattr(doc['timestamp'], 'strftime') else str(doc['timestamp']),
                            "temperature": doc['temperature'],
                            "vibration": doc['vibration'],
                            "humidity": doc['humidity'],
                            "pressure": doc['pressure'],
                            "energy": doc['energy_consumption'],
                            "source": doc.get('source', 'KAFKA_STREAM'),
                            # Additional fields from digital twin
                            "machine_status": doc.get('machine_status', 0),
                            "machine_status_label": doc.get('machine_status_label', 'Normal'),
                            "anomaly_flag": doc.get('anomaly_flag', 0),
                            "anomaly_flag_label": doc.get('anomaly_flag_label', 'Normal'),
                            "failure_type": doc.get('failure_type', 'Normal'),
                            "downtime_risk": doc.get('downtime_risk', 0),
                            "maintenance_required": doc.get('maintenance_required', 0),
                            "predicted_remaining_life": doc.get('predicted_remaining_life', 200),
                            "anchor_machine": doc.get('anchor_machine', 40),
                            "graph_distance": doc.get('graph_distance', 0),
                            "influence_score": doc.get('influence_score', 0)
                        }
                        
                        yield event
                
                # Wait for next polling cycle
                time.sleep(poll_interval)
                
            except Exception as e:
                print(f"[ERROR] Stream polling failed: {e}")
                time.sleep(5)
    
    def close(self):
        self.client.close()
        print("[DISCONNECTED] MongoDB connection closed")

# ─────────────────────────────────────────────
# LAYER 2 — CONTEXT / MEMORY STORE
# ─────────────────────────────────────────────
class ContextMemoryStore:
    """
    Menyimpan:
    - sliding_window: histori n event terakhir per mesin
    - machine_profile: profil normal per mesin
    - decision_log: audit trail semua keputusan agent
    - feedback_log: hasil aksi untuk learning loop
    """
    def __init__(self, window_size=10):
        self.window_size    = window_size
        self.sliding_window = {}    # {machine_id: deque(events)}
        self.machine_profile= {}    # {machine_id: {mean_temp, ...}}
        self.decision_log   = []    # audit trail
        self.feedback_log   = []    # untuk learning loop

    def update(self, machine_id, event):
        if machine_id not in self.sliding_window:
            self.sliding_window[machine_id] = deque(maxlen=self.window_size)
        self.sliding_window[machine_id].append(event)

    def get_context(self, machine_id):
        history = list(self.sliding_window.get(machine_id, []))
        return {
            "machine_id"   : machine_id,
            "event_count"  : len(history),
            "latest"       : history[-1] if history else {},
            "history_last3": history[-3:] if len(history) >= 3 else history
        }

    def log_decision(self, machine_id, event, risk_score, action, reason):
        entry = {
            "timestamp"  : event.get("timestamp"),
            "machine_id" : machine_id,
            "risk_score" : risk_score,
            "action"     : action,
            "reason"     : reason
        }
        self.decision_log.append(entry)
        print(f"[AUDIT] M-{machine_id} | action={action} | risk={risk_score:.2f}")
        return entry

    def log_feedback(self, machine_id, action, outcome, delta_rul):
        self.feedback_log.append({
            "machine_id": machine_id,
            "action"    : action,
            "outcome"   : outcome,
            "delta_rul" : delta_rul,
            "logged_at" : str(datetime.datetime.now())
        })

# ─────────────────────────────────────────────
# LAYER 3 — REASONING ENGINE
# ─────────────────────────────────────────────
class ReasoningEngine:
    """
    Menghitung risk_score dari pembacaan sensor + RUL.
    AUDIT NOTE: previously added anomaly_flag (+0.20) and
    failure_type (+0.15) — both ground truth. Removed.
    Thresholds now sourced from config (derived from training split).
    """
    def __init__(self):
        cfg = get_warn_crit_thresholds()
        # Map energy_consumption → energy used in event payloads
        self.THRESHOLDS = {}
        for k, v in cfg.items():
            self.THRESHOLDS["energy" if k == "energy_consumption" else k] = v

    def score(self, event: dict, predicted_rul: int = 200) -> dict:
        score = 0.0
        flags = []
        for feature, thr in self.THRESHOLDS.items():
            val = event.get(feature, 0)
            if not isinstance(val, (int, float)) or thr["warn"] == float("inf"):
                continue
            if val >= thr["crit"]:
                score += 0.35
                flags.append(f"{feature}=CRITICAL({val:.2f})")
            elif val >= thr["warn"]:
                score += 0.15
                flags.append(f"{feature}=WARN({val:.2f})")

        rul = event.get('predicted_remaining_life', predicted_rul)
        if rul < RUL_THRESHOLD:
            score += 0.30
            flags.append(f"RUL=LOW({rul})")
        elif rul < 100:
            score += 0.10
            flags.append(f"RUL=MODERATE({rul})")

        risk_score = min(score, 1.0)
        severity   = "HIGH" if risk_score >= 0.6 else ("MEDIUM" if risk_score >= 0.3 else "LOW")

        return {
            "risk_score":   round(risk_score, 3),
            "severity":     severity,
            "flags":        flags,
            "predicted_rul": rul,
        }

# ─────────────────────────────────────────────
# LAYER 4 — MULTI-AGENT ORCHESTRATOR (Ollama)
# ─────────────────────────────────────────────
def get_ollama_client():
    return Client(
        host   = "https://ollama.com",
        headers= {"Authorization": "Bearer " + api_key}
    )

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
Given sensor context and risk flags, return JSON:
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

class MultiAgentOrchestrator:
    def __init__(self, model_name=OLLAMA_MODEL):
        self.model = model_name
        self.client= get_ollama_client()

    def _call(self, system_prompt, user_msg):
        try:
            resp = self.client.chat(
                model   = self.model,
                messages= [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_msg}
                ]
            )
            content = resp["message"]["content"].strip()
            # Parse JSON jika bukan reporting agent
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return {"raw": content}
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def _safe_event(event):
        """LLM-safe view of an event — strips ground-truth fields."""
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

    def run_monitoring_agent(self, event, reasoning_result):
        msg = f"""
Sensor event: {json.dumps(self._safe_event(event), indent=2)}
Risk assessment: {json.dumps(reasoning_result, indent=2)}
"""
        return self._call(SYSTEM_PROMPT_MONITORING, msg)

    def run_diagnosis_agent(self, event, monitoring_result, reasoning_result):
        msg = f"""
Sensor event: {json.dumps(self._safe_event(event), indent=2)}
Monitoring result: {json.dumps(monitoring_result, indent=2)}
Risk flags: {reasoning_result.get('flags', [])}
"""
        return self._call(SYSTEM_PROMPT_DIAGNOSIS, msg)

    def run_planning_agent(self, diagnosis_result, reasoning_result):
        msg = f"""
Diagnosis: {json.dumps(diagnosis_result, indent=2)}
Risk score: {reasoning_result.get('risk_score')}
Severity: {reasoning_result.get('severity')}
Predicted RUL: {reasoning_result.get('predicted_rul')}
"""
        return self._call(SYSTEM_PROMPT_PLANNING, msg)

    def run_reporting_agent(self, machine_id, planning_result, diagnosis_result):
        msg = f"""
Machine ID: M-{machine_id}
Diagnosis: {json.dumps(diagnosis_result, indent=2)}
Action plan: {json.dumps(planning_result, indent=2)}
"""
        return self._call(SYSTEM_PROMPT_REPORTING, msg)

# ─────────────────────────────────────────────
# LAYER 5 — ACTION LAYER
# ─────────────────────────────────────────────
HIGH_RISK_ACTIONS = {"immediate_shutdown", "schedule_maintenance"}

def execute_action(machine_id, planning_result, report, memory: ContextMemoryStore, require_approval=True):
    action   = planning_result.get("action", "monitor")
    priority = planning_result.get("priority", "P4")
    notify   = planning_result.get("notify_supervisor", False)

    # Human-in-the-loop untuk aksi berisiko tinggi
    if action in HIGH_RISK_ACTIONS and require_approval:
        print(f"\n[APPROVAL REQUIRED] M-{machine_id}: action={action} priority={priority}")
        print(f"  Reason: {planning_result.get('reason','')}")
        # Dalam production: kirim ke approval workflow / Slack / CMMS
        # Simulasi: auto-approve untuk demo
        approved = True
        if not approved:
            print(f"[REJECTED] Action dibatalkan oleh supervisor.")
            return

    # Eksekusi aksi
    _dispatch_action(machine_id, action, priority, notify, report)

def _dispatch_action(machine_id, action, priority, notify, report):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*60}")
    print(f"[ACTION] {ts} — Machine M-{machine_id}")
    print(f"  Action   : {action}")
    print(f"  Priority : {priority}")
    print(f"  Report   : {report.get('raw', report) if isinstance(report, dict) else report}")
    print(f"{'='*60}")

    if action == "immediate_shutdown":
        print(f"  >> [CMMS] Emergency stop triggered for M-{machine_id}")
        print(f"  >> [ALERT] SMS/email sent to maintenance team")

    elif action == "schedule_maintenance":
        print(f"  >> [CMMS] Work order created for M-{machine_id} ({priority})")
        print(f"  >> [ERP]  Spare part check initiated")

    elif action == "inspect":
        print(f"  >> [SCHEDULER] Inspection scheduled for M-{machine_id} within 4h")

    elif action == "monitor":
        print(f"  >> [MONITOR] Enhanced monitoring activated for M-{machine_id}")

    if notify:
        print(f"  >> [NOTIFY] Supervisor alert sent")

# ─────────────────────────────────────────────
# LAYER 6 — LEARNING LOOP
# ─────────────────────────────────────────────
class LearningLoop:
    """
    Mengumpulkan feedback dari action outcomes.
    Dalam implementasi penuh:
    - Bandingkan prediksi RUL vs actual downtime
    - Kirim data ke pipeline retraining T-GCN
    - Update threshold di ReasoningEngine
    """
    def __init__(self, memory: ContextMemoryStore):
        self.memory = memory

    def update_thresholds(self, engine: ReasoningEngine):
        """
        Contoh: jika terlalu banyak false positives,
        naikkan threshold warning.
        """
        feedback = self.memory.feedback_log
        if len(feedback) < 5:
            return  # tidak cukup data

        false_positives = sum(1 for f in feedback
                              if f["action"] in HIGH_RISK_ACTIONS
                              and f["outcome"] == "no_failure")
        fp_rate = false_positives / len(feedback)

        if fp_rate > 0.4:
            # Longgarkan threshold temperature warning
            engine.THRESHOLDS["temperature"]["warn"] += 2
            print(f"[LEARN] FP rate={fp_rate:.1%} → temp warn threshold naik ke "
                  f"{engine.THRESHOLDS['temperature']['warn']}")

    def trigger_retrain(self):
        """Kirim feedback log ke pipeline retraining T-GCN."""
        print(f"[LEARN] {len(self.memory.feedback_log)} feedback records dikirim ke retraining pipeline")
        # Dalam production: kirim ke MLflow / Airflow / Ray Train

# ─────────────────────────────────────────────
# MAIN AGENTIC PIPELINE (REAL-TIME KAFKA STREAM)
# ─────────────────────────────────────────────
def run_agentic_pipeline_realtime(machine_ids=None, poll_interval=POLL_INTERVAL_SEC, max_events=50):
    """
    Real-time agentic pipeline — menampilkan data sensor terbaru 50 mesin.
    Hanya menampilkan data mentah tanpa reasoning, threshold, atau status.
    
    Args:
        machine_ids: List of machine IDs to monitor (None = all 50 machines)
        poll_interval: Seconds between MongoDB polls (default 60s)
        max_events: Maximum events to process (default 50)
    """
    kafka_consumer = KafkaStreamConsumer()

    # Ambil snapshot terbaru langsung dari MongoDB (tanpa streaming)
    latest_data = kafka_consumer.get_latest_snapshot(machine_ids)

    print("\n" + "="*90)
    print(" DATA SENSOR TERBARU — 50 MESIN")
    print("="*90)
    print(f"{'No':<4} {'Machine':<10} {'Timestamp':<22} {'Temp(°C)':<10} {'Vib':<8} "
          f"{'Humi(%)':<9} {'Press':<8} {'Energy(kWh)':<13} {'RUL(h)':<8}")
    print("-"*90)

    for i, doc in enumerate(latest_data[:max_events], 1):
        machine_id = doc.get('machine_id', '-')
        timestamp  = doc.get('timestamp', '-')
        if hasattr(timestamp, 'strftime'):
            timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")

        temp    = doc.get('temperature', 0)
        vib     = doc.get('vibration', 0)
        humi    = doc.get('humidity', 0)
        press   = doc.get('pressure', 0)
        energy  = doc.get('energy_consumption', 0)
        rul     = doc.get('predicted_remaining_life', 0)

        print(f"{i:<4} {machine_id:<10} {str(timestamp):<22} {temp:<10.1f} {vib:<8.1f} "
              f"{humi:<9.1f} {press:<8.2f} {energy:<13.1f} {rul:<8.1f}")

    print("-"*90)
    print(f" Total: {len(latest_data[:max_events])} mesin")
    print("="*90)

    kafka_consumer.close()
    return latest_data

# ─────────────────────────────────────────────
# LEGACY FUNCTION (for backward compatibility)
# ─────────────────────────────────────────────
def run_agentic_pipeline(machine_id, tgcn_predictions):
    """
    Legacy function using simulated T-GCN data.
    Use run_agentic_pipeline_realtime() for real Kafka stream.
    """
    print("\n" + "="*70)
    print(f" AGENTIC AI PIPELINE — Machine M-{machine_id} (SIMULATED MODE)")
    print("="*70)

    memory      = ContextMemoryStore(window_size=10)
    reasoner    = ReasoningEngine()
    orchestrator= MultiAgentOrchestrator(model_name=OLLAMA_MODEL)
    learner     = LearningLoop(memory)

    for event in kafka_consumer_simulation(machine_id, tgcn_predictions):
        print(f"\n[INGEST] {event['timestamp']} — M-{machine_id}")

        # L2: update memory
        memory.update(machine_id, event)
        context = memory.get_context(machine_id)

        # L3: reasoning engine
        rul = 180  # ambil dari checkpoint atau kolom predicted_remaining_life
        reasoning = reasoner.score(event, predicted_rul=rul)
        print(f"  Risk: {reasoning['risk_score']} ({reasoning['severity']}) | Flags: {reasoning['flags']}")

        # Skip agent calls jika LOW risk → hemat resource
        if reasoning["severity"] == "LOW":
            print("  [SKIP] LOW risk — monitoring only")
            memory.log_decision(machine_id, event, reasoning["risk_score"], "monitor", "auto-LOW")
            continue

        # L4: multi-agent orchestration
        monitoring = orchestrator.run_monitoring_agent(event, reasoning)
        print(f"  [MONITOR AGENT] {monitoring}")

        diagnosis  = orchestrator.run_diagnosis_agent(event, monitoring, reasoning)
        print(f"  [DIAGNOSIS AGENT] {diagnosis}")

        planning   = orchestrator.run_planning_agent(diagnosis, reasoning)
        print(f"  [PLANNING AGENT] {planning}")

        report     = orchestrator.run_reporting_agent(machine_id, planning, diagnosis)
        print(f"  [REPORT AGENT] {report}")

        # L5: action layer
        execute_action(machine_id, planning, report, memory)

        # L2: log decision audit
        memory.log_decision(machine_id, event, reasoning["risk_score"],
                            planning.get("action","monitor"),
                            planning.get("reason",""))

        # L6: learning loop feedback (simulasi outcome)
        memory.log_feedback(machine_id,
                            action   = planning.get("action","monitor"),
                            outcome  = "no_failure",   # dari ground truth monitoring
                            delta_rul= 0)

    # L6: update thresholds setelah batch event
    learner.update_thresholds(reasoner)
    learner.trigger_retrain()

    print("\n[DONE] Pipeline selesai.")
    print(f"[AUDIT LOG] {len(memory.decision_log)} decisions recorded.")
    print(f"[FEEDBACK]  {len(memory.feedback_log)} feedback records.")
    return memory

# Old simulation function (kept for backward compatibility)
def kafka_consumer_simulation(machine_id: int, tgcn_predictions: list):
    for pred in tgcn_predictions:
        yield {
            "machine_id"  : machine_id,
            "timestamp"   : pred["timestamp"],
            "temperature" : pred["temp"],
            "vibration"   : pred["vib"],
            "humidity"    : pred["hum"],
            "pressure"    : pred["press"],
            "energy"      : pred["energy"],
            "source"      : pred.get("source", "SENSOR+IMP")
        }
        time.sleep(0.05)

# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    
    print("\n" + "="*80)
    print(" AGENTIC AI — DATA SENSOR 50 MESIN (KAFKA STREAM via MongoDB)")
    print("="*80)

    # Parse command line args
    if len(sys.argv) > 1 and sys.argv[1] == "--machine" and len(sys.argv) > 2:
        machine_id = int(sys.argv[2])
        print(f"[MODE] Menampilkan data mesin {machine_id}")
        run_agentic_pipeline_realtime(machine_ids=[machine_id], max_events=1)
    else:
        print("[MODE] Menampilkan data terbaru seluruh 50 mesin")
        run_agentic_pipeline_realtime(max_events=50)