# ============================================================
# KAFKA REAL-TIME CONSUMER WITH MULTI-TIER SLIDING WINDOW
# ============================================================
#
# Kafka Topic: simulated_digitaltwin_manufactured
# Broker: 10.247.166.188:9092
#
# Actual Kafka Fields (verified):
#   machine_id (int)              - Machine ID 1-50
#   timestamp (str)               - Sensor timestamp
#   temperature (float)           - Machine temperature
#   vibration (float)             - Vibration level
#   humidity (float)              - Humidity reading
#   pressure (float)              - Pressure reading
#   energy_consumption (float)    - Energy usage
#   machine_status (int)          - 0=Normal, 1=Fault
#   anomaly_flag (int)            - 0=Normal, 1=Anomaly
#   predicted_remaining_life (float) - Predicted RUL
#   failure_type (str)            - Failure classification
#   downtime_risk (int)           - 0 or 1
#   maintenance_required (int)    - 0 or 1
#   source (str)                  - "PROPAGATED" or "T-GCN"
#   sensor_input (obj)            - Raw Plalion sensor mapping
#   processed_at (str)            - Processing timestamp
#
# ============================================================

import json
import math
import time
import signal
import datetime
import statistics
import threading
from collections import deque, defaultdict

from kafka import KafkaConsumer
from ollama import Client

from config import (
    OLLAMA_API_KEY, KAFKA_BROKERS as CFG_KAFKA_BROKERS, KAFKA_TOPIC as CFG_KAFKA_TOPIC,
    OLLAMA_MODEL, RUL_LOW, SENSOR_FEATURES, get_warn_crit_thresholds,
)

api_key           = OLLAMA_API_KEY
KAFKA_BROKERS     = CFG_KAFKA_BROKERS or ["localhost:9092"]
KAFKA_TOPIC       = CFG_KAFKA_TOPIC
KAFKA_GROUP_ID    = "agentic_ai_consumer_group"
RUL_THRESHOLD     = RUL_LOW
STATUS_MAP        = {0: "Normal", 1: "Fault"}

TIER1_SIZE       = 15
TIER2_SIZE       = 120
TIER3_SIZE       = 288
TIER4_SIZE       = 336
TIER3_DOWNSAMPLE = 5
TIER4_DOWNSAMPLE = 60

MICROBATCH_INTERVAL_SEC = 600
SCHEDULED_INTERVAL_SEC  = 86400
SPIKE_ZSCORE_THRESHOLD  = 3.0

# AUDIT NOTE: thresholds derived from training split, not hardcoded.
THRESHOLDS = get_warn_crit_thresholds()


# ========== SECTION 1: MULTI-TIER BUFFER ==========

class MultiTierBuffer:
    def __init__(self, machine_id):
        self.machine_id = machine_id
        self.tier1 = deque(maxlen=TIER1_SIZE)
        self.tier2 = deque(maxlen=TIER2_SIZE)
        self.tier3 = deque(maxlen=TIER3_SIZE)
        self.tier4 = deque(maxlen=TIER4_SIZE)
        self._t3_acc = []
        self._t4_acc = []
        self.total_events = 0
        self.last_event   = None

    def append(self, event):
        self.total_events += 1
        self.last_event = event
        self.tier1.append(event)
        self.tier2.append(event)
        self._t3_acc.append(event)
        if len(self._t3_acc) >= TIER3_DOWNSAMPLE:
            self.tier3.append(self._avg(self._t3_acc))
            self._t3_acc = []
        self._t4_acc.append(event)
        if len(self._t4_acc) >= TIER4_DOWNSAMPLE:
            self.tier4.append(self._avg(self._t4_acc))
            self._t4_acc = []

    def _avg(self, events):
        out = {"machine_id": self.machine_id, "n_samples": len(events)}
        for feat in SENSOR_FEATURES:
            vals = [e.get(feat, 0) for e in events]
            out[feat] = round(sum(vals) / len(vals), 3)
        ruls = [e.get("predicted_remaining_life", 0) for e in events]
        out["predicted_remaining_life"] = round(sum(ruls) / len(ruls), 2)
        last = events[-1]
        out["timestamp"]            = last.get("timestamp")
        out["machine_status"]       = last.get("machine_status", 0)
        out["failure_type"]         = last.get("failure_type", "Normal")
        out["anomaly_flag"]         = last.get("anomaly_flag", 0)
        out["downtime_risk"]        = last.get("downtime_risk", 0)
        out["maintenance_required"] = last.get("maintenance_required", 0)
        out["source"]               = last.get("source", "")
        return out

    def get_tier(self, n):
        return list({1: self.tier1, 2: self.tier2, 3: self.tier3, 4: self.tier4}[n])

    def stats(self):
        return {"machine_id": self.machine_id, "total_events": self.total_events,
                "tier1": f"{len(self.tier1)}/{TIER1_SIZE}", "tier2": f"{len(self.tier2)}/{TIER2_SIZE}",
                "tier3": f"{len(self.tier3)}/{TIER3_SIZE}", "tier4": f"{len(self.tier4)}/{TIER4_SIZE}"}


class BufferManager:
    def __init__(self):
        self.buffers = {}
        self.lock = threading.Lock()

    def ingest(self, event):
        mid = event.get("machine_id")
        if mid is None: return
        with self.lock:
            if mid not in self.buffers:
                self.buffers[mid] = MultiTierBuffer(mid)
            self.buffers[mid].append(event)

    def get_buffer(self, mid):
        with self.lock: return self.buffers.get(mid)

    def all_machine_ids(self):
        with self.lock: return list(self.buffers.keys())

    def get_latest_events(self):
        with self.lock:
            return [b.last_event for b in sorted(self.buffers.values(), key=lambda b: b.machine_id) if b.last_event]

    def summary(self):
        with self.lock:
            return [b.stats() for b in sorted(self.buffers.values(), key=lambda b: b.machine_id)]


# ========== SECTION 2: SPIKE DETECTOR ==========

class SpikeDetector:
    def check(self, buffer_mgr, event):
        mid = event.get("machine_id")
        buf = buffer_mgr.get_buffer(mid)
        if buf is None or len(buf.tier1) < 5: return None
        data = list(buf.tier1)
        spikes = []
        for feat in SENSOR_FEATURES:
            history = [e.get(feat, 0) for e in data[:-1]]
            current = event.get(feat, 0)
            if len(history) < 3: continue
            mean = statistics.mean(history)
            std = statistics.stdev(history) if len(history) > 1 else 0.001
            if std < 0.001: continue
            z = (current - mean) / std
            if abs(z) >= SPIKE_ZSCORE_THRESHOLD:
                spikes.append({"feature": feat, "current": round(current, 2),
                               "mean": round(mean, 2), "z_score": round(z, 2),
                               "direction": "HIGH" if z > 0 else "LOW"})
        return {"machine_id": mid, "spikes": spikes} if spikes else None


# ========== SECTION 3: REASONING ENGINE ==========

class ReasoningEngine:
    def __init__(self):
        self.thresholds = dict(THRESHOLDS)

    def score(self, event):
        score = 0.0
        flags = []
        for feat, thr in self.thresholds.items():
            val = event.get(feat, 0)
            if val >= thr["crit"]:   score += 0.35; flags.append(f"{feat}=CRIT({val:.1f})")
            elif val >= thr["warn"]: score += 0.15; flags.append(f"{feat}=WARN({val:.1f})")
        rul = event.get("predicted_remaining_life", 200)
        if rul < RUL_THRESHOLD:   score += 0.30; flags.append(f"RUL=LOW({rul:.0f})")
        elif rul < 100:           score += 0.10; flags.append(f"RUL=MOD({rul:.0f})")
        if event.get("anomaly_flag", 0) == 1:        score += 0.20; flags.append("ANOMALY")
        if event.get("machine_status", 0) == 1:      score += 0.15; flags.append("STATUS=FAULT")
        ft = event.get("failure_type", "Normal")
        if ft != "Normal":                            score += 0.15; flags.append(f"FAIL={ft}")
        if event.get("downtime_risk", 0) == 1:       score += 0.10; flags.append("DOWNTIME_RISK")
        if event.get("maintenance_required", 0) == 1: score += 0.10; flags.append("MAINT_REQ")
        risk = min(score, 1.0)
        sev = "HIGH" if risk >= 0.6 else ("MEDIUM" if risk >= 0.3 else "LOW")
        return {"risk_score": round(risk, 3), "severity": sev, "flags": flags, "predicted_rul": rul}


# ========== SECTION 4: LLM INTERFACE ==========

def get_ollama_client():
    return Client(host="https://ollama.com", headers={"Authorization": "Bearer " + api_key})

def llm_call(client, system_prompt, user_msg):
    try:
        resp = client.chat(model=OLLAMA_MODEL, messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_msg}])
        content = resp["message"]["content"].strip()
        try: return json.loads(content)
        except json.JSONDecodeError: return {"raw": content}
    except Exception as e: return {"error": str(e)}

PROMPT_MONITORING = (
    "You are a machine monitoring agent for smart manufacturing. "
    "Analyze sensor data (temperature, vibration, humidity, pressure, energy_consumption) "
    "and digital twin flags (machine_status: 0=Normal/1=Fault, anomaly_flag, failure_type, "
    "downtime_risk, maintenance_required). Return JSON ONLY:\n"
    '{"risk_level":"LOW|MEDIUM|HIGH|CRITICAL","anomaly_detected":true|false,'
    '"anomaly_type":"thermal|vibration|energy|combined|none",'
    '"trend":"stable|degrading|improving","confidence":0.0-1.0}\n')

PROMPT_DIAGNOSIS = (
    "You are a root-cause diagnosis agent. Given sensor data and digital twin flags, "
    "determine cause. Return JSON ONLY:\n"
    '{"probable_cause":"bearing_wear|overheating|electrical|lubrication|normal|unknown",'
    '"affected_component":"motor|bearing|pump|actuator|unknown",'
    '"urgency_hours":integer,"evidence":"max 20 words"}\n')

PROMPT_PLANNING = (
    "You are a maintenance planning agent. Return JSON ONLY:\n"
    '{"action":"monitor|inspect|schedule_maintenance|immediate_shutdown",'
    '"priority":"P1|P2|P3|P4","reason":"max 20 words","notify_supervisor":true|false}\n')

PROMPT_REPORTING = (
    "You are a reporting agent. Generate a short maintenance alert in Bahasa Indonesia. "
    "Max 3 sentences. Plain text only.\n")

PROMPT_CORRELATION = (
    "You are a Correlation Agent. Given sensor data from multiple machines, identify "
    "correlated failure clusters. Return JSON ONLY:\n"
    '{"clusters":[{"machines":["M-X"],"type":"thermal|electrical|vibration",'
    '"propagation_risk":0.0-1.0,"root_machine":"M-X"}],'
    '"max_propagation_risk":float,"cascade_warning":true|false}\n')

PROMPT_EXPLAINABILITY = (
    "You are an Explainability Agent (XAI). Return JSON ONLY:\n"
    '{"top_feature":"temperature|vibration|humidity|energy_consumption",'
    '"contribution_pct":float,"distance_to_next_severity":float,'
    '"what_if":"smallest change to escalate","confidence":0.0-1.0}\n')

PROMPT_CONSENSUS = (
    "You are a Consensus Agent. Return JSON ONLY:\n"
    '{"agreed_risk":"LOW|MEDIUM|HIGH","agreed_action":"monitor|inspect|'
    'schedule_maintenance|immediate_shutdown","agreement_ratio":float,'
    '"has_disagreement":true|false,"reliability_score":0.0-1.0}\n')

PROMPT_SCHEDULING = (
    "You are a Predictive Scheduling Agent. Return JSON ONLY:\n"
    '{"schedule":[{"machine_id":"M-X","maintenance_type":"preventive|corrective|'
    'predictive|inspection","scheduled_date":"YYYY-MM-DD","priority":"P1|P2|P3|P4",'
    '"estimated_downtime_hours":float}],"total_downtime_hours":float,"machines_scheduled":int}\n')

PROMPT_ANOMALY_PATTERN = (
    "You are an Anomaly Pattern Learning Agent. Return JSON ONLY:\n"
    '{"patterns":[{"type":"slow_drift|cyclic|spike|novel","machines":["M-X"],'
    '"features":["temperature"],"confidence":0.0-1.0,"trend":"worsening|stable|improving",'
    '"ttf_hours":float}],"total_patterns":int,"novel_count":int,"model_update_needed":true|false}\n')


# ========== SECTION 5: TREND COMPUTATION ==========

def compute_trend(buffer_data, feature):
    vals = [e.get(feature, 0) for e in buffer_data]
    n = len(vals)
    if n < 3: return {"mean": 0, "stdev": 0, "slope": 0, "direction": "insufficient", "n": n, "pct_warn": 0}
    mean = statistics.mean(vals)
    std = statistics.stdev(vals)
    x_mean = (n - 1) / 2.0
    num = sum((i - x_mean) * (v - mean) for i, v in enumerate(vals))
    den = sum((i - x_mean) ** 2 for i in range(n))
    slope = num / den if den != 0 else 0
    direction = "stable"
    if abs(slope) > std * 0.01:
        direction = "increasing" if slope > 0 else "decreasing"
    thr = THRESHOLDS.get(feature, {})
    warn = thr.get("warn", float('inf'))
    pct_warn = round(vals[-1] / warn * 100, 1) if warn > 0 and warn != float('inf') else 0
    return {"mean": round(mean, 2), "stdev": round(std, 3), "slope": round(slope, 5),
            "direction": direction, "latest": round(vals[-1], 2), "n": n, "pct_warn": pct_warn}


# ========== SECTION 6: MICRO-BATCH TRIGGER (10 min) ==========

class MicroBatchTrigger:
    def __init__(self):
        self.client = get_ollama_client()
        self.reasoner = ReasoningEngine()
        self.run_count = 0

    def execute(self, buffer_mgr):
        self.run_count += 1
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        machine_ids = buffer_mgr.all_machine_ids()
        n = len(machine_ids)
        print(f"\n{'='*80}")
        print(f"  MICRO-BATCH #{self.run_count} - {ts} | {n} machines | Tier 2")
        print(f"{'='*80}")
        if n == 0:
            print("  [SKIP] No machines in buffer.")
            return {}

        latest = buffer_mgr.get_latest_events()
        reasonings = {e["machine_id"]: self.reasoner.score(e) for e in latest}

        trends = {}
        for mid in machine_ids:
            buf = buffer_mgr.get_buffer(mid)
            if buf is None: continue
            t2 = buf.get_tier(2)
            if len(t2) < 5: continue
            trends[mid] = {f: compute_trend(t2, f) for f in SENSOR_FEATURES}

        targets = []
        for evt in latest:
            mid = evt["machine_id"]
            rsn = reasonings.get(mid, {})
            has_signal = (evt.get("machine_status", 0) == 1
                or evt.get("failure_type", "Normal") != "Normal"
                or evt.get("anomaly_flag", 0) == 1
                or evt.get("maintenance_required", 0) == 1
                or evt.get("downtime_risk", 0) == 1
                or rsn.get("severity") in ("MEDIUM", "HIGH"))
            near_thr = any(t.get("pct_warn", 0) >= 90 for t in trends.get(mid, {}).values())
            degrading = any(t.get("direction") == "increasing" and t.get("slope", 0) > 0.01
                           for t in trends.get(mid, {}).values())
            if has_signal or near_thr or degrading:
                targets.append(mid)

        print(f"  [SELECT] {len(targets)}/{n} machines for LLM")

        results = {}
        for mid in targets:
            evt = next((e for e in latest if e["machine_id"] == mid), None)
            if evt is None: continue
            rsn = reasonings.get(mid, {})
            trn = trends.get(mid, {})

            sensor_str = json.dumps({k: evt.get(k) for k in [
                "machine_id", "temperature", "vibration", "humidity", "pressure",
                "energy_consumption", "machine_status", "failure_type", "anomaly_flag",
                "downtime_risk", "maintenance_required", "predicted_remaining_life"]}, indent=2)
            trend_str = json.dumps({f: {"slope": trn[f]["slope"], "direction": trn[f]["direction"],
                "pct_warn": trn[f]["pct_warn"]} for f in trn}, indent=2) if trn else "{}"
            risk_str = json.dumps(rsn)

            monitoring = llm_call(self.client, PROMPT_MONITORING,
                f"Sensor:\n{sensor_str}\nTrend(2h):\n{trend_str}\nRisk:{risk_str}")
            diagnosis = llm_call(self.client, PROMPT_DIAGNOSIS,
                f"Sensor:\n{sensor_str}\nMonitoring:{json.dumps(monitoring)}\nTrend:\n{trend_str}")
            planning = llm_call(self.client, PROMPT_PLANNING,
                f"Diagnosis:{json.dumps(diagnosis)}\nRisk:{rsn.get('risk_score')}\n"
                f"Severity:{rsn.get('severity')}\nRUL:{evt['predicted_remaining_life']}")
            xai = llm_call(self.client, PROMPT_EXPLAINABILITY,
                f"M-{mid}\nSensor:\n{sensor_str}\nTrend:\n{trend_str}\n"
                f"Thresholds:{json.dumps(THRESHOLDS)}\nRisk:{risk_str}")
            consensus = llm_call(self.client, PROMPT_CONSENSUS,
                f"Monitoring:{json.dumps(monitoring)}\nDiagnosis:{json.dumps(diagnosis)}\n"
                f"Planning:{json.dumps(planning)}")

            results[mid] = {"reasoning": rsn, "trends": trn, "monitoring": monitoring,
                "diagnosis": diagnosis, "planning": planning, "xai": xai, "consensus": consensus}
            print(f"    M-{mid:<3} risk={monitoring.get('risk_level','?'):<8} "
                  f"cause={diagnosis.get('probable_cause','?'):<15} "
                  f"action={planning.get('action','?')}")

        for mid in machine_ids:
            if mid not in results:
                results[mid] = {"reasoning": reasonings.get(mid, {}), "action": "monitor"}

        print(f"  [DONE] {len(targets)} LLM calls\n")
        return results


# ========== SECTION 7: SCHEDULED DAILY TRIGGER ==========

class ScheduledTrigger:
    def __init__(self):
        self.client = get_ollama_client()
        self.reasoner = ReasoningEngine()
        self.run_count = 0

    def execute(self, buffer_mgr):
        self.run_count += 1
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        machine_ids = buffer_mgr.all_machine_ids()
        n = len(machine_ids)
        print(f"\n{'#'*80}")
        print(f"  DAILY TRIGGER #{self.run_count} - {ts} | {n} machines")
        print(f"{'#'*80}")
        if n == 0:
            print("  [SKIP] No machines.")
            return {}

        latest = buffer_mgr.get_latest_events()
        all_e = []
        all_r = []
        for evt in latest:
            rsn = self.reasoner.score(evt)
            all_e.append(evt)
            all_r.append(rsn)
        print(f"  [1] {len(all_e)} machines scored")

        print(f"  [2] Correlation Agent...")
        corr_in = [{"machine_id": e["machine_id"], "temperature": e["temperature"],
            "vibration": e["vibration"], "machine_status": e["machine_status"],
            "failure_type": e["failure_type"], "anomaly_flag": e["anomaly_flag"],
            "downtime_risk": e["downtime_risk"]} for e in all_e]
        correlation = llm_call(self.client, PROMPT_CORRELATION,
            f"Factory ({n} machines):\n{json.dumps(corr_in, indent=2)}")

        print(f"  [3] Anomaly Pattern Agent (Tier 3)...")
        pat_in = []
        for mid in machine_ids:
            buf = buffer_mgr.get_buffer(mid)
            t3 = buf.get_tier(3) if buf else []
            if len(t3) < 3: continue
            pat_in.append({"machine_id": mid, "n_24h": len(t3),
                "trends": {f: compute_trend(t3, f) for f in SENSOR_FEATURES},
                "failure_type": buf.last_event.get("failure_type", "Normal") if buf.last_event else "Normal"})
        patterns = llm_call(self.client, PROMPT_ANOMALY_PATTERN,
            f"24h trends ({len(pat_in)} machines):\n{json.dumps(pat_in, indent=2)}")

        print(f"  [4] Scheduling Agent...")
        sched_in = [{"machine_id": e["machine_id"], "risk_score": r["risk_score"],
            "severity": r["severity"], "predicted_rul": r["predicted_rul"],
            "failure_type": e["failure_type"], "machine_status": e["machine_status"]}
            for e, r in zip(all_e, all_r)]
        scheduling = llm_call(self.client, PROMPT_SCHEDULING,
            f"Schedule {n} machines on {datetime.date.today()}.\n"
            f"Max 5 simultaneous, 08:00-18:00.\n{json.dumps(sched_in, indent=2)}")

        print(f"  [5] Daily report...")
        risk_dist = {"HIGH": sum(1 for r in all_r if r["severity"]=="HIGH"),
            "MEDIUM": sum(1 for r in all_r if r["severity"]=="MEDIUM"),
            "LOW": sum(1 for r in all_r if r["severity"]=="LOW")}
        report = llm_call(self.client, PROMPT_REPORTING,
            f"Summary {datetime.date.today()}: {n} machines, risk={json.dumps(risk_dist)}\n"
            f"Correlation:{json.dumps(correlation)}\nPatterns:{json.dumps(patterns)}\n"
            f"Schedule:{json.dumps(scheduling)}")

        print(f"  [6] Baseline drift (Tier 4)...")
        drift = {}
        for mid in machine_ids:
            buf = buffer_mgr.get_buffer(mid)
            t4 = buf.get_tier(4) if buf else []
            if len(t4) < 10: continue
            for feat in SENSOR_FEATURES:
                t = compute_trend(t4, feat)
                if t["direction"] == "increasing" and t["slope"] > 0.005:
                    drift.setdefault(feat, []).append({"machine_id": mid, "slope": t["slope"]})
        print(f"    Drift: {list(drift.keys()) if drift else 'none'}")

        result = {"timestamp": ts, "run": self.run_count, "correlation": correlation,
            "patterns": patterns, "scheduling": scheduling, "report": report,
            "risk_distribution": risk_dist, "drift": drift}
        print(f"  [DONE] Daily #{self.run_count}\n")
        return result


# ========== SECTION 8: MAIN LOOP ==========

def parse_kafka_message(msg):
    """msg.value is already dict (json deserializer). Normalize fields."""
    try:
        data = msg.value
        if not isinstance(data, dict) or data.get("machine_id") is None:
            return None
        status = data.get("machine_status", 0)
        return {
            "machine_id":               data["machine_id"],
            "timestamp":                data.get("timestamp", ""),
            "temperature":              data.get("temperature", 0),
            "vibration":                data.get("vibration", 0),
            "humidity":                 data.get("humidity", 0),
            "pressure":                 data.get("pressure", 0),
            "energy_consumption":       data.get("energy_consumption", 0),
            "machine_status":           status,
            "machine_status_label":     STATUS_MAP.get(status, "Normal"),
            "anomaly_flag":             data.get("anomaly_flag", 0),
            "predicted_remaining_life": data.get("predicted_remaining_life", 200),
            "failure_type":             data.get("failure_type", "Normal"),
            "downtime_risk":            data.get("downtime_risk", 0),
            "maintenance_required":     data.get("maintenance_required", 0),
            "source":                   data.get("source", ""),
            "processed_at":             data.get("processed_at", ""),
        }
    except Exception as e:
        print(f"  [PARSE ERROR] {e}")
        return None


def run_pipeline():
    print("\n" + "="*80)
    print("  AGENTIC AI - KAFKA REAL-TIME CONSUMER")
    print("  Multi-Tier Sliding Window + Dual Trigger")
    print("="*80)
    print(f"  Kafka   : {KAFKA_BROKERS[0]}")
    print(f"  Topic   : {KAFKA_TOPIC}")
    print(f"  Group   : {KAFKA_GROUP_ID}")
    print(f"  Micro   : every {MICROBATCH_INTERVAL_SEC//60} min")
    print(f"  Daily   : every {SCHEDULED_INTERVAL_SEC//3600} hr")
    print(f"  Tiers   : T1={TIER1_SIZE} T2={TIER2_SIZE} T3={TIER3_SIZE} T4={TIER4_SIZE}")
    print(f"  LLM     : {OLLAMA_MODEL}")
    print("="*80 + "\n")

    buffer_mgr    = BufferManager()
    spike_det     = SpikeDetector()
    micro_trigger = MicroBatchTrigger()
    daily_trigger = ScheduledTrigger()

    print("[KAFKA] Connecting...")
    try:
        consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers=KAFKA_BROKERS,
            group_id=KAFKA_GROUP_ID,
            auto_offset_reset='latest',
            enable_auto_commit=True,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        )
        print(f"[KAFKA] Connected: {KAFKA_BROKERS[0]} / {KAFKA_TOPIC}\n")
    except Exception as e:
        print(f"[KAFKA ERROR] {e}")
        return

    last_micro = time.time()
    last_daily = time.time()
    event_count = 0
    spike_count = 0
    running = True
    first_logged = False

    def on_signal(sig, frame):
        nonlocal running
        print("\n[SIGNAL] Shutting down...")
        running = False
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    print("[STREAM] Listening...\n")

    while running:
        try:
            batch = consumer.poll(timeout_ms=3000, max_records=200)

            for tp, messages in batch.items():
                for msg in messages:
                    event = parse_kafka_message(msg)
                    if event is None: continue

                    event_count += 1
                    mid = event["machine_id"]

                    if not first_logged:
                        first_logged = True
                        print(f"  OK FIRST EVENT - data flowing!")
                        print(f"    machine_id              : {mid}")
                        print(f"    machine_status          : {event['machine_status']} ({event['machine_status_label']})")
                        print(f"    temperature             : {event['temperature']}")
                        print(f"    vibration               : {event['vibration']}")
                        print(f"    humidity                : {event['humidity']}")
                        print(f"    pressure                : {event['pressure']}")
                        print(f"    energy_consumption      : {event['energy_consumption']}")
                        print(f"    failure_type            : {event['failure_type']}")
                        print(f"    anomaly_flag            : {event['anomaly_flag']}")
                        print(f"    predicted_remaining_life: {event['predicted_remaining_life']}")
                        print(f"    downtime_risk           : {event['downtime_risk']}")
                        print(f"    maintenance_required    : {event['maintenance_required']}")
                        print(f"    source                  : {event['source']}")
                        print(f"    timestamp               : {event['timestamp']}")
                        print(f"    processed_at            : {event['processed_at']}")
                        print(f"    Kafka keys              : {list(msg.value.keys())}")
                        print()

                    buffer_mgr.ingest(event)

                    spike = spike_det.check(buffer_mgr, event)
                    if spike:
                        spike_count += 1
                        feats = ", ".join(f"{s['feature']}={s['current']}(z={s['z_score']})" for s in spike["spikes"])
                        print(f"  SPIKE M-{mid}: {feats}")

                    if event_count % 50 == 0:
                        nm = len(buffer_mgr.all_machine_ids())
                        remain = max(0, MICROBATCH_INTERVAL_SEC - (time.time() - last_micro))
                        print(f"  [PROGRESS] {event_count} events | {nm} machines | "
                              f"{spike_count} spikes | next micro-batch {remain:.0f}s")

            now = time.time()
            if now - last_micro >= MICROBATCH_INTERVAL_SEC:
                result = micro_trigger.execute(buffer_mgr)
                last_micro = now
                fname = f"microbatch_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"
                with open(fname, "w") as f: json.dump(result, f, indent=2, default=str)
                print(f"  [SAVED] {fname}")

            if now - last_daily >= SCHEDULED_INTERVAL_SEC:
                result = daily_trigger.execute(buffer_mgr)
                last_daily = now
                fname = f"daily_{datetime.datetime.now():%Y%m%d}.json"
                with open(fname, "w") as f: json.dump(result, f, indent=2, default=str)
                print(f"  [SAVED] {fname}")

            if not batch: time.sleep(1)

        except KeyboardInterrupt:
            running = False
        except Exception as e:
            print(f"  [ERROR] {e}")
            time.sleep(5)

    consumer.close()
    print(f"\n{'='*80}")
    print(f"  SHUTDOWN SUMMARY")
    print(f"{'='*80}")
    print(f"  Events   : {event_count}")
    print(f"  Spikes   : {spike_count}")
    print(f"  Micro    : {micro_trigger.run_count} runs")
    print(f"  Daily    : {daily_trigger.run_count} runs")
    print(f"  Machines : {len(buffer_mgr.all_machine_ids())}")
    print(f"\n  Buffer occupancy:")
    for s in buffer_mgr.summary():
        print(f"    M-{s['machine_id']:<3} total={s['total_events']:<6} "
              f"T1={s['tier1']} T2={s['tier2']} T3={s['tier3']} T4={s['tier4']}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    run_pipeline()