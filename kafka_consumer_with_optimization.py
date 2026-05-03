# ============================================================
# KAFKA CONSUMER + CLASSICAL OPTIMIZATION + AGENTIC AI
# ============================================================
# Integrates classical_optimization.py at 5 points in the pipeline:
#
#   Per-event:  Point 1 — Mahalanobis anomaly (alongside z-score)
#   Micro-batch: Point 4 — Cosine consensus (alongside LLM consensus)
#   Daily:      Point 2 — Graph correlation (feeds Correlation Agent)
#               Point 3 — SA scheduling (feeds Scheduling Agent)
#               Point 5 — Nelder-Mead thresholds (feeds Learning Loop)
#
# Later, each classical algorithm can be swapped for its quantum
# counterpart: VQC, QGNN, QAOA, Q-Kernel, Q-Annealing.
# ============================================================

import json, math, time, signal, datetime, statistics, threading, random
from collections import deque, defaultdict
from kafka import KafkaConsumer
from ollama import Client
from classical_optimization import ClassicalOptimizationEngine, SENSOR_FEATURES as OPT_FEATURES

from config import (
    OLLAMA_API_KEY, KAFKA_BROKERS as CFG_KAFKA_BROKERS, KAFKA_TOPIC as CFG_KAFKA_TOPIC,
    OLLAMA_MODEL, RUL_LOW, SENSOR_FEATURES, get_warn_crit_thresholds,
)
api_key           = OLLAMA_API_KEY
KAFKA_BROKERS     = CFG_KAFKA_BROKERS or ["localhost:9092"]
KAFKA_TOPIC       = CFG_KAFKA_TOPIC
KAFKA_GROUP_ID    = "agentic_ai_optimized_group"
RUL_THRESHOLD     = RUL_LOW
STATUS_MAP        = {0: "Normal", 1: "Fault"}
TIER1_SIZE, TIER2_SIZE, TIER3_SIZE, TIER4_SIZE = 15, 120, 288, 336
TIER3_DS, TIER4_DS = 5, 60
MICROBATCH_SEC = 600
SCHEDULED_SEC  = 86400
SPIKE_Z = 3.0
# AUDIT NOTE: thresholds derived from training split, not hardcoded.
THRESHOLDS = get_warn_crit_thresholds()


# ═══════ BUFFER (same as v3) ═══════
class MultiTierBuffer:
    def __init__(self, mid):
        self.machine_id = mid
        self.tier1 = deque(maxlen=TIER1_SIZE); self.tier2 = deque(maxlen=TIER2_SIZE)
        self.tier3 = deque(maxlen=TIER3_SIZE); self.tier4 = deque(maxlen=TIER4_SIZE)
        self._t3a = []; self._t4a = []; self.total = 0; self.last = None
    def append(self, e):
        self.total += 1; self.last = e; self.tier1.append(e); self.tier2.append(e)
        self._t3a.append(e)
        if len(self._t3a) >= TIER3_DS:
            self.tier3.append(self._avg(self._t3a)); self._t3a = []
        self._t4a.append(e)
        if len(self._t4a) >= TIER4_DS:
            self.tier4.append(self._avg(self._t4a)); self._t4a = []
    def _avg(self, evts):
        o = {"machine_id": self.machine_id, "n": len(evts)}
        for f in SENSOR_FEATURES: o[f] = round(sum(e.get(f,0) for e in evts)/len(evts), 3)
        o["predicted_remaining_life"] = round(sum(e.get("predicted_remaining_life",0) for e in evts)/len(evts), 2)
        for k in ["timestamp","machine_status","failure_type","anomaly_flag","downtime_risk",
                   "maintenance_required","source","anchor_machine","graph_distance","influence_score"]:
            o[k] = evts[-1].get(k, 0)
        return o
    def get_tier(self, n): return list({1:self.tier1,2:self.tier2,3:self.tier3,4:self.tier4}[n])
    def stats(self):
        return {"mid": self.machine_id, "total": self.total,
                "t1": f"{len(self.tier1)}/{TIER1_SIZE}", "t2": f"{len(self.tier2)}/{TIER2_SIZE}",
                "t3": f"{len(self.tier3)}/{TIER3_SIZE}", "t4": f"{len(self.tier4)}/{TIER4_SIZE}"}

class BufferManager:
    def __init__(self): self.buffers = {}; self.lock = threading.Lock()
    def ingest(self, e):
        mid = e.get("machine_id")
        if mid is None: return
        with self.lock:
            if mid not in self.buffers: self.buffers[mid] = MultiTierBuffer(mid)
            self.buffers[mid].append(e)
    def get_buffer(self, mid):
        with self.lock: return self.buffers.get(mid)
    def all_ids(self):
        with self.lock: return list(self.buffers.keys())
    def latest_events(self):
        with self.lock:
            return [b.last for b in sorted(self.buffers.values(), key=lambda b: b.machine_id) if b.last]
    def summary(self):
        with self.lock:
            return [b.stats() for b in sorted(self.buffers.values(), key=lambda b: b.machine_id)]


# ═══════ SPIKE DETECTOR + MULTIVARIATE (Point 1) ═══════
class EnhancedSpikeDetector:
    """Combined: classical z-score + Mahalanobis multivariate (Point 1)."""
    def __init__(self, opt_engine):
        self.opt = opt_engine

    def check(self, buffer_mgr, event):
        mid = event.get("machine_id")
        buf = buffer_mgr.get_buffer(mid)

        # ── Classical z-score (univariate) ──
        z_spikes = []
        if buf and len(buf.tier1) >= 5:
            data = list(buf.tier1)
            for feat in SENSOR_FEATURES:
                hist = [e.get(feat,0) for e in data[:-1]]
                cur = event.get(feat,0)
                if len(hist) < 3: continue
                mean = statistics.mean(hist)
                std = statistics.stdev(hist) if len(hist)>1 else 0.001
                if std < 0.001: continue
                z = (cur - mean) / std
                if abs(z) >= SPIKE_Z:
                    z_spikes.append({"feature": feat, "value": round(cur,2), "z": round(z,2)})

        # ── Point 1: Mahalanobis multivariate ──
        mv_result = self.opt.detect_multivariate_anomaly(mid, event)

        # Combine: flag if EITHER detects anomaly
        has_spike = len(z_spikes) > 0
        has_mv = mv_result.get("is_anomaly", False)

        if has_spike or has_mv:
            return {
                "machine_id": mid,
                "z_score_spikes": z_spikes,
                "multivariate": mv_result,
                "detection_source": ("both" if has_spike and has_mv
                                     else "z_score" if has_spike else "mahalanobis"),
            }
        return None


# ═══════ REASONING ENGINE ═══════
class ReasoningEngine:
    """
    AUDIT NOTE: previous version added 0.20 for anomaly_flag, 0.15
    for machine_status, 0.15 for failure_type, 0.10 each for
    downtime_risk and maintenance_required — all five are ground
    truth in the digital-twin metadata. Removed.
    """
    def __init__(self): self.thresholds = dict(THRESHOLDS)
    def score(self, e):
        s = 0.0; flags = []
        for f, t in self.thresholds.items():
            v = e.get(f, 0)
            if not isinstance(v, (int, float)) or t["warn"] == float("inf"):
                continue
            if v >= t["crit"]: s += 0.35; flags.append(f"{f}=CRIT({v:.2f})")
            elif v >= t["warn"]: s += 0.15; flags.append(f"{f}=WARN({v:.2f})")
        rul = e.get("predicted_remaining_life", 200)
        if rul < RUL_THRESHOLD: s += 0.30; flags.append(f"RUL=LOW({rul:.0f})")
        elif rul < 100: s += 0.10
        risk = min(s, 1.0)
        sev = "HIGH" if risk>=0.6 else ("MEDIUM" if risk>=0.3 else "LOW")
        return {"risk_score": round(risk,3), "severity": sev, "flags": flags, "predicted_rul": rul}


# ═══════ LLM INTERFACE (same as v3) ═══════
def get_ollama(): return Client(host="https://ollama.com", headers={"Authorization": "Bearer "+api_key})

def llm_call(client, prompt, msg):
    try:
        r = client.chat(model=OLLAMA_MODEL, messages=[{"role":"system","content":prompt},{"role":"user","content":msg}])
        c = r["message"]["content"].strip()
        if c.startswith("```"):
            c = "\n".join(l for l in c.split("\n") if not l.strip().startswith("```")).strip()
        try: return json.loads(c)
        except: return {"raw": c}
    except Exception as e: return {"error": str(e)}

def extract(result, *keys):
    for k in keys:
        if k in result: return result[k]
    return None

P_MON = ("You are a machine monitoring agent. You MUST respond with ONLY JSON:\n"
    '{"risk_level":"LOW|MEDIUM|HIGH|CRITICAL","anomaly_detected":true|false,'
    '"anomaly_type":"thermal|vibration|energy|combined|none","confidence":0.0-1.0}\n'
    "Raw JSON only, no markdown.")
P_DIAG = ("You are a root-cause diagnosis agent. You MUST respond with ONLY JSON:\n"
    '{"probable_cause":"bearing_wear|overheating|electrical|lubrication|normal|unknown",'
    '"affected_component":"motor|bearing|pump|actuator|unknown","urgency_hours":24}\n'
    "Raw JSON only.")
P_PLAN = ("You are a maintenance planning agent. You MUST respond with ONLY JSON:\n"
    '{"action":"monitor|inspect|schedule_maintenance|immediate_shutdown",'
    '"priority":"P1|P2|P3|P4","reason":"max 20 words","notify_supervisor":false}\n'
    "Raw JSON only.")
P_XAI = ("You are an Explainability Agent. You MUST respond with ONLY JSON:\n"
    '{"top_feature":"temperature","contribution_pct":45.0,'
    '"distance_to_next_severity":1.6,"what_if":"smallest change to escalate","confidence":0.8}\n'
    "Raw JSON only.")
P_REPORT = "Generate a short maintenance alert in Bahasa Indonesia. Max 3 sentences."
P_CORR = ("You are a Correlation Agent. Given classical graph analysis results AND sensor data, "
    "provide interpretation. You MUST respond with ONLY JSON:\n"
    '{"interpretation":"description of clusters and risks","recommended_actions":["action1"],'
    '"cascade_risk_assessment":"low|medium|high"}\n'
    "Raw JSON only.")
P_SCHED = ("You are a Scheduling Agent. Given classical optimization results, "
    "add context and reasoning. You MUST respond with ONLY JSON:\n"
    '{"assessment":"evaluation of schedule","adjustments":["adjustment1"],'
    '"human_readable_summary":"Bahasa Indonesia summary"}\n'
    "Raw JSON only.")
P_PATTERN = ("You are an Anomaly Pattern Learning Agent. You MUST respond with ONLY JSON:\n"
    '{"patterns":[{"type":"slow_drift","machines":["M-1"],"features":["temperature"],'
    '"confidence":0.7,"trend":"worsening","ttf_hours":168}],'
    '"total_patterns":1,"model_update_needed":false}\nRaw JSON only.')


# ═══════ TREND COMPUTATION (same) ═══════
def compute_trend(data, feat):
    vals = [e.get(feat,0) for e in data]; n = len(vals)
    if n < 3: return {"mean":0,"stdev":0,"slope":0,"direction":"insufficient","n":n,"pct_warn":0}
    mean = statistics.mean(vals); std = statistics.stdev(vals)
    xm = (n-1)/2.0
    num = sum((i-xm)*(v-mean) for i,v in enumerate(vals))
    den = sum((i-xm)**2 for i in range(n))
    slope = num/den if den else 0
    d = "stable"
    if abs(slope) > std*0.01: d = "increasing" if slope>0 else "decreasing"
    thr = THRESHOLDS.get(feat,{}); w = thr.get("warn", float('inf'))
    pct = round(vals[-1]/w*100,1) if w>0 and w!=float('inf') else 0
    return {"mean":round(mean,2),"stdev":round(std,3),"slope":round(slope,5),
            "direction":d,"latest":round(vals[-1],2),"n":n,"pct_warn":pct}


# ═══════ MICRO-BATCH TRIGGER (Point 1 + Point 4 integrated) ═══════
class MicroBatchTrigger:
    def __init__(self, opt_engine):
        self.client = get_ollama()
        self.reasoner = ReasoningEngine()
        self.opt = opt_engine
        self.run_count = 0

    def execute(self, buffer_mgr):
        self.run_count += 1
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        mids = buffer_mgr.all_ids(); n = len(mids)
        print(f"\n{'='*80}")
        print(f"  MICRO-BATCH #{self.run_count} - {ts} | {n} machines")
        print(f"{'='*80}")
        if n == 0: print("  [SKIP]"); return {}

        latest = buffer_mgr.latest_events()
        reasonings = {e["machine_id"]: self.reasoner.score(e) for e in latest}
        sev = {"HIGH":0,"MEDIUM":0,"LOW":0}
        for r in reasonings.values(): sev[r["severity"]] = sev.get(r["severity"],0)+1
        print(f"  [REASONING] H={sev['HIGH']} M={sev['MEDIUM']} L={sev['LOW']}")

        trends = {}
        for mid in mids:
            buf = buffer_mgr.get_buffer(mid)
            if buf is None: continue
            t2 = buf.get_tier(2)
            if len(t2) < 5: continue
            trends[mid] = {f: compute_trend(t2, f) for f in SENSOR_FEATURES}

        # ── Point 1: Multivariate anomaly on all machines ──
        mv_anomalies = {}
        for evt in latest:
            mid = evt["machine_id"]
            mv = self.opt.detect_multivariate_anomaly(mid, evt)
            if mv.get("is_anomaly"):
                mv_anomalies[mid] = mv
        if mv_anomalies:
            print(f"  [MAHALANOBIS] {len(mv_anomalies)} multivariate anomalies detected:")
            for mid, mv in list(mv_anomalies.items())[:5]:
                top = mv.get("top_contributors", [])
                top_str = ", ".join(f"{t[0]}({t[1]})" for t in top[:2])
                print(f"    M-{mid:<3} score={mv['anomaly_score']:.3f} top=[{top_str}]")
        else:
            print(f"  [MAHALANOBIS] No multivariate anomalies")

        # Select machines for LLM
        targets = []
        for evt in latest:
            mid = evt["machine_id"]
            rsn = reasonings.get(mid, {})
            has_signal = (evt.get("machine_status",0)==1 or evt.get("failure_type","Normal")!="Normal"
                or evt.get("anomaly_flag",0)==1 or evt.get("maintenance_required",0)==1
                or evt.get("downtime_risk",0)==1 or rsn.get("severity") in ("MEDIUM","HIGH"))
            near = any(t.get("pct_warn",0)>=90 for t in trends.get(mid,{}).values())
            degrad = any(t.get("direction")=="increasing" and t.get("slope",0)>0.01
                        for t in trends.get(mid,{}).values())
            mv_flag = mid in mv_anomalies  # Point 1 integration
            if has_signal or near or degrad or mv_flag:
                targets.append(mid)

        print(f"  [SELECT] {len(targets)}/{n} machines for LLM")

        results = {}
        for idx, mid in enumerate(targets):
            evt = next((e for e in latest if e["machine_id"]==mid), None)
            if evt is None: continue
            rsn = reasonings.get(mid, {}); trn = trends.get(mid, {})

            sensor_str = json.dumps({k: evt.get(k) for k in [
                "machine_id","temperature","vibration","humidity","pressure",
                "energy_consumption","machine_status","failure_type","anomaly_flag",
                "downtime_risk","maintenance_required","predicted_remaining_life",
                "influence_score","graph_distance","anchor_machine"]}, indent=2)
            trend_str = json.dumps({f:{"slope":trn[f]["slope"],"dir":trn[f]["direction"],
                "pct":trn[f]["pct_warn"]} for f in trn}, indent=2) if trn else "{}"

            # Include Mahalanobis result if available
            mv_info = ""
            if mid in mv_anomalies:
                mv_info = f"\nMahalanobis anomaly: {json.dumps(mv_anomalies[mid])}"

            monitoring = llm_call(self.client, P_MON,
                f"Sensor:\n{sensor_str}\nTrend(2h):\n{trend_str}\nReasoning:{json.dumps(rsn)}{mv_info}")
            diagnosis = llm_call(self.client, P_DIAG,
                f"Sensor:\n{sensor_str}\nMonitoring:{json.dumps(monitoring)}")
            planning = llm_call(self.client, P_PLAN,
                f"Diagnosis:{json.dumps(diagnosis)}\nRisk:{rsn.get('risk_score')}\n"
                f"Severity:{rsn.get('severity')}\nRUL:{evt['predicted_remaining_life']}")
            xai = llm_call(self.client, P_XAI,
                f"M-{mid}\nSensor:\n{sensor_str}\nThresholds:{json.dumps(THRESHOLDS)}\nReasoning:{json.dumps(rsn)}")

            # ── Point 4: Classical consensus validation ──
            classical_consensus = self.opt.validate_consensus(monitoring, diagnosis, planning, rsn)

            results[mid] = {
                "reasoning": rsn, "trends": trn, "monitoring": monitoring,
                "diagnosis": diagnosis, "planning": planning, "xai": xai,
                "classical_consensus": classical_consensus,
                "multivariate_anomaly": mv_anomalies.get(mid),
            }

            risk = extract(monitoring,"risk_level","risk") or "?"
            cause = extract(diagnosis,"probable_cause","cause") or "?"
            action = extract(planning,"action") or "?"
            cons_risk = classical_consensus.get("agreed_risk", "?")
            reliability = classical_consensus.get("reliability_score", 0)
            print(f"    [{idx+1}/{len(targets)}] M-{mid:<3} "
                  f"risk={str(risk):<8} cause={str(cause):<15} action={str(action):<22} "
                  f"consensus={cons_risk}(r={reliability:.2f})")

        for mid in mids:
            if mid not in results:
                results[mid] = {"reasoning": reasonings.get(mid,{}), "action": "monitor"}

        print(f"  [DONE] {len(targets)} LLM + {len(mv_anomalies)} Mahalanobis + {len(targets)} consensus\n")
        return results


# ═══════ DAILY TRIGGER (Points 2, 3, 5 integrated) ═══════
class ScheduledTrigger:
    def __init__(self, opt_engine):
        self.client = get_ollama()
        self.reasoner = ReasoningEngine()
        self.opt = opt_engine
        self.run_count = 0

    def execute(self, buffer_mgr):
        self.run_count += 1
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        mids = buffer_mgr.all_ids(); n = len(mids)
        print(f"\n{'#'*80}")
        print(f"  DAILY TRIGGER #{self.run_count} - {ts} | {n} machines")
        print(f"{'#'*80}")
        if n == 0: return {}

        latest = buffer_mgr.latest_events()
        all_e, all_r = [], []
        for evt in latest:
            all_e.append(evt); all_r.append(self.reasoner.score(evt))
        print(f"  [1/6] Reasoning: {len(all_e)} machines scored")

        # ── Point 2: Classical graph correlation (→ future QGNN) ──
        print(f"  [2/6] Graph correlation (classical message passing)...")
        graph_result = self.opt.analyze_graph(all_e, all_r)
        n_clusters = len(graph_result.get("clusters", []))
        max_prop = graph_result.get("max_propagation_risk", 0)
        cascade = graph_result.get("cascade_warning", False)
        print(f"    Clusters: {n_clusters} | Max propagation: {max_prop:.3f} | Cascade: {cascade}")
        for c in graph_result.get("clusters", [])[:3]:
            print(f"    [{c['type']}] {', '.join(c['machines'][:5])} → root={c['root_machine']} risk={c['propagation_risk']:.3f}")

        # LLM Correlation Agent interprets graph results
        corr_interpretation = llm_call(self.client, P_CORR,
            f"Classical graph analysis results:\n{json.dumps(graph_result, indent=2)}")
        print(f"    LLM interpretation: {json.dumps(corr_interpretation)[:150]}...")

        # ── Point 3: Scheduling optimization (→ future QAOA) ──
        print(f"  [3/6] Scheduling optimization (simulated annealing)...")
        sched_input = [{"machine_id": e["machine_id"], "risk_score": r["risk_score"],
            "predicted_rul": r["predicted_rul"], "failure_type": e.get("failure_type","Normal")}
            for e, r in zip(all_e, all_r)]
        sched_result = self.opt.optimize_schedule(sched_input)
        print(f"    Scheduled: {sched_result['machines_scheduled']} machines | "
              f"Downtime: {sched_result['total_downtime_hours']}h | "
              f"Cost: ${sched_result['total_cost']:.0f}")
        print(f"    SA improvement vs greedy: {sched_result['improvement_vs_greedy_pct']:.1f}%")
        for s in sched_result.get("schedule", [])[:3]:
            print(f"    {s['machine_id']} {s['priority']} {s['maintenance_type']} "
                  f"{s['scheduled_time']} ({s['estimated_downtime_hours']}h)")

        # LLM Scheduling Agent adds context
        sched_interpretation = llm_call(self.client, P_SCHED,
            f"SA optimization results:\n{json.dumps(sched_result, indent=2)}")

        # Anomaly Pattern Agent (Tier 3)
        print(f"  [4/6] Anomaly Pattern Agent (Tier 3)...")
        pat_in = []
        for mid in mids:
            buf = buffer_mgr.get_buffer(mid)
            t3 = buf.get_tier(3) if buf else []
            if len(t3) < 3: continue
            pat_in.append({"machine_id": mid, "n_24h": len(t3),
                "trends": {f: compute_trend(t3, f) for f in SENSOR_FEATURES},
                "failure_type": buf.last.get("failure_type","Normal") if buf.last else "Normal"})
        patterns = llm_call(self.client, P_PATTERN,
            f"24h trends ({len(pat_in)} machines):\n{json.dumps(pat_in, indent=2)}")

        # Daily report
        print(f"  [5/6] Daily report...")
        risk_dist = {"HIGH": sum(1 for r in all_r if r["severity"]=="HIGH"),
            "MEDIUM": sum(1 for r in all_r if r["severity"]=="MEDIUM"),
            "LOW": sum(1 for r in all_r if r["severity"]=="LOW")}
        report = llm_call(self.client, P_REPORT,
            f"Summary {datetime.date.today()}: {n} machines, risk={json.dumps(risk_dist)}\n"
            f"Graph: {n_clusters} clusters, cascade={cascade}\n"
            f"Schedule: {sched_result['machines_scheduled']} machines, ${sched_result['total_cost']:.0f}\n"
            f"Patterns: {json.dumps(patterns)}")

        # ── Point 5: Threshold optimization (→ future Q-Annealing) ──
        print(f"  [6/6] Threshold optimization (Nelder-Mead)...")
        all_history = []
        for mid in mids:
            buf = buffer_mgr.get_buffer(mid)
            if buf: all_history.extend(buf.get_tier(4))
        thr_result = self.opt.optimize_thresholds(all_history, self.reasoner.thresholds)
        if thr_result.get("optimized"):
            impr = thr_result["improvement_pct"]
            print(f"    Improvement: {impr:.1f}% (cost {thr_result['initial_cost']:.4f} → {thr_result['optimized_cost']:.4f})")
            print(f"    New thresholds: {json.dumps(thr_result['thresholds'])}")
            # Apply optimized thresholds to reasoning engine
            if impr > 1.0:  # only apply if meaningful improvement
                self.reasoner.thresholds = thr_result["thresholds"]
                print(f"    Applied to reasoning engine.")
        else:
            print(f"    {thr_result.get('reason', 'not optimized')}")

        result = {"timestamp": ts, "graph_correlation": graph_result,
            "graph_interpretation": corr_interpretation,
            "scheduling": sched_result, "scheduling_interpretation": sched_interpretation,
            "patterns": patterns, "report": report, "risk_distribution": risk_dist,
            "threshold_optimization": thr_result}
        print(f"  [DONE] Daily #{self.run_count}\n")
        return result


# ═══════ KAFKA PARSE (same as v3) ═══════
def parse(msg):
    try:
        d = msg.value
        if not isinstance(d, dict) or d.get("machine_id") is None: return None
        st = d.get("machine_status", 0)
        return {"machine_id": d["machine_id"], "timestamp": d.get("timestamp",""),
            "temperature": d.get("temperature",0), "vibration": d.get("vibration",0),
            "humidity": d.get("humidity",0), "pressure": d.get("pressure",0),
            "energy_consumption": d.get("energy_consumption",0),
            "machine_status": st, "machine_status_label": d.get("machine_status_label", STATUS_MAP.get(st,"Normal")),
            "anomaly_flag": d.get("anomaly_flag",0), "predicted_remaining_life": d.get("predicted_remaining_life",200),
            "failure_type": d.get("failure_type","Normal"), "downtime_risk": d.get("downtime_risk",0),
            "maintenance_required": d.get("maintenance_required",0), "source": d.get("source",""),
            "anchor_machine": d.get("anchor_machine",0), "graph_distance": d.get("graph_distance",0),
            "influence_score": d.get("influence_score",0), "processed_at": d.get("processed_at","")}
    except Exception as e:
        print(f"  [PARSE ERROR] {e}"); return None


# ═══════ MAIN PIPELINE ═══════
def run():
    print("\n" + "="*80)
    print("  AGENTIC AI + CLASSICAL OPTIMIZATION")
    print("  Kafka Consumer with 5-point optimization integration")
    print("="*80)
    print(f"  Kafka      : {KAFKA_BROKERS[0]} / {KAFKA_TOPIC}")
    print(f"  LLM        : {OLLAMA_MODEL}")
    print(f"  Micro-batch: every {MICROBATCH_SEC//60} min (Point 1: Mahalanobis + Point 4: Consensus)")
    print(f"  Daily      : every {SCHEDULED_SEC//3600} hr (Point 2: Graph + Point 3: SA + Point 5: Nelder-Mead)")
    print(f"  Tiers      : T1={TIER1_SIZE} T2={TIER2_SIZE} T3={TIER3_SIZE} T4={TIER4_SIZE}")
    print("="*80 + "\n")

    # Initialize all components
    opt_engine    = ClassicalOptimizationEngine()
    buffer_mgr    = BufferManager()
    spike_det     = EnhancedSpikeDetector(opt_engine)
    micro_trigger = MicroBatchTrigger(opt_engine)
    daily_trigger = ScheduledTrigger(opt_engine)

    print("[KAFKA] Connecting...")
    try:
        consumer = KafkaConsumer(KAFKA_TOPIC, bootstrap_servers=KAFKA_BROKERS,
            group_id=KAFKA_GROUP_ID, auto_offset_reset='latest', enable_auto_commit=True,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')))
        print(f"[KAFKA] Connected\n")
    except Exception as e:
        print(f"[KAFKA ERROR] {e}"); return

    last_micro = time.time(); last_daily = time.time()
    evt_count = 0; spike_count = 0; mv_count = 0
    running = True; first = False

    def stop(s, f):
        nonlocal running; print("\n[SIGNAL] Stopping..."); running = False
    signal.signal(signal.SIGINT, stop); signal.signal(signal.SIGTERM, stop)

    print("[STREAM] Listening...\n")

    while running:
        try:
            batch = consumer.poll(timeout_ms=3000, max_records=200)
            for tp, msgs in batch.items():
                for msg in msgs:
                    event = parse(msg)
                    if event is None: continue
                    evt_count += 1; mid = event["machine_id"]

                    if not first:
                        first = True
                        print(f"  OK data flowing! M-{mid} temp={event['temperature']:.1f} "
                              f"status={event['machine_status']} fail={event['failure_type']}")
                        print(f"    Keys: {list(msg.value.keys())}\n")

                    buffer_mgr.ingest(event)

                    # Per-event: z-score + Mahalanobis (Point 1)
                    spike = spike_det.check(buffer_mgr, event)
                    if spike:
                        src = spike["detection_source"]
                        if spike["z_score_spikes"]:
                            spike_count += 1
                        if spike.get("multivariate", {}).get("is_anomaly"):
                            mv_count += 1
                        z_str = ", ".join(f"{s['feature']}(z={s['z']})" for s in spike["z_score_spikes"])
                        mv_s = spike.get("multivariate",{}).get("anomaly_score",0)
                        print(f"  ALERT M-{mid} [{src}] z=[{z_str}] mahal={mv_s:.3f}")

                    if evt_count % 50 == 0:
                        nm = len(buffer_mgr.all_ids())
                        remain = max(0, MICROBATCH_SEC - (time.time()-last_micro))
                        print(f"  [PROGRESS] {evt_count} events | {nm} machines | "
                              f"spikes={spike_count} mahal={mv_count} | next micro {remain:.0f}s")

            now = time.time()
            if now - last_micro >= MICROBATCH_SEC:
                r = micro_trigger.execute(buffer_mgr); last_micro = now
                fn = f"microbatch_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"
                with open(fn,"w") as f: json.dump(r, f, indent=2, default=str)
                print(f"  [SAVED] {fn}")

            if now - last_daily >= SCHEDULED_SEC:
                r = daily_trigger.execute(buffer_mgr); last_daily = now
                fn = f"daily_{datetime.datetime.now():%Y%m%d}.json"
                with open(fn,"w") as f: json.dump(r, f, indent=2, default=str)
                print(f"  [SAVED] {fn}")

            if not batch: time.sleep(1)
        except KeyboardInterrupt: running = False
        except Exception as e: print(f"  [ERROR] {e}"); time.sleep(5)

    consumer.close()
    print(f"\n{'='*80}")
    print(f"  SHUTDOWN: {evt_count} events, {spike_count} spikes, {mv_count} mahalanobis")
    print(f"  Micro: {micro_trigger.run_count} | Daily: {daily_trigger.run_count} | Machines: {len(buffer_mgr.all_ids())}")
    for s in buffer_mgr.summary()[:10]:
        print(f"    M-{s['mid']:<3} total={s['total']:<6} {s['t1']} {s['t2']} {s['t3']} {s['t4']}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    run()
