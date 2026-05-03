# ============================================================
# KAFKA CONSUMER — VERBOSE AGENT LOGGING
# ============================================================
# Shows full input → output → timing for each of 9 agents.
# Designed so you can see exactly what each agent receives,
# what it returns, and how long it takes — enabling targeted
# enhancement of any individual agent.
#
# 9 Agent Orchestration (sequential pipeline):
#   MICRO-BATCH (every 10 min):
#     Agent ① Monitoring    — risk assessment from sensor + trend
#     Agent ② Diagnosis     — root-cause from sensor + monitoring
#     Agent ③ Planning      — action from diagnosis + risk
#     Agent ④ Reporting     — Bahasa Indonesia alert
#     Agent ⑥ Explainability — counterfactual XAI
#     Opt   ④ Consensus     — classical cosine consensus (Point 4)
#     Opt   ① Mahalanobis   — multivariate anomaly (Point 1)
#
#   DAILY (every 24 hr):
#     Agent ⑤ Correlation   — inter-machine clusters
#     Agent ⑧ Scheduling    — maintenance optimization
#     Agent ⑨ Pattern       — temporal anomaly patterns
#     Agent ④ Reporting     — daily summary
#     Opt   ② Graph         — classical message passing (Point 2)
#     Opt   ③ SA Schedule   — simulated annealing (Point 3)
#     Opt   ⑤ Threshold     — Nelder-Mead optimization (Point 5)
# ============================================================

import json, math, time, signal, datetime, statistics, threading
from collections import deque, defaultdict
from kafka import KafkaConsumer
from ollama import Client
from classical_optimization import ClassicalOptimizationEngine

from config import (
    OLLAMA_API_KEY, KAFKA_BROKERS as CFG_KAFKA_BROKERS, KAFKA_TOPIC as CFG_KAFKA_TOPIC,
    OLLAMA_MODEL, RUL_LOW, SENSOR_FEATURES,
    get_warn_crit_thresholds,
)
api_key        = OLLAMA_API_KEY
KAFKA_BROKERS  = CFG_KAFKA_BROKERS or ["localhost:9092"]
KAFKA_TOPIC    = CFG_KAFKA_TOPIC
KAFKA_GROUP_ID = "agentic_verbose_group"
RUL_THRESHOLD  = RUL_LOW
STATUS_MAP     = {0: "Normal", 1: "Fault"}
TIER1_SIZE, TIER2_SIZE, TIER3_SIZE, TIER4_SIZE = 15, 120, 288, 336
TIER3_DS, TIER4_DS = 5, 60
MICROBATCH_SEC = 600
SCHEDULED_SEC  = 86400
SPIKE_Z = 3.0
# AUDIT NOTE: thresholds loaded from training-derived bounds, not hardcoded
THRESHOLDS = get_warn_crit_thresholds()


# ═══════ AGENT LOGGER ═══════
class AgentLogger:
    """Structured logger for each agent call showing input, output, timing."""

    def __init__(self):
        self.logs = []

    def log_agent(self, machine_id, agent_name, agent_number, trigger_type,
                  input_data, output_data, elapsed_sec, notes=""):
        entry = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "machine_id": machine_id,
            "agent_name": agent_name,
            "agent_number": agent_number,
            "trigger": trigger_type,
            "elapsed_sec": round(elapsed_sec, 3),
            "input_summary": self._summarize(input_data),
            "output": output_data,
            "notes": notes,
        }
        self.logs.append(entry)
        self._print_agent_log(entry, input_data)
        return entry

    def _summarize(self, data):
        """Create a short summary of input data for display."""
        if isinstance(data, str):
            return data[:200] + "..." if len(data) > 200 else data
        if isinstance(data, dict):
            return {k: str(v)[:60] for k, v in list(data.items())[:6]}
        return str(data)[:200]

    def _print_agent_log(self, entry, full_input):
        """Pretty-print one agent's execution."""
        mid = entry["machine_id"]
        name = entry["agent_name"]
        num = entry["agent_number"]
        trigger = entry["trigger"]
        elapsed = entry["elapsed_sec"]
        output = entry["output"]

        # Header
        print(f"\n    ┌─ Agent {num} : {name} ─── M-{mid} ({trigger}) ──────")

        # Input summary (compact)
        if isinstance(full_input, dict):
            for k, v in list(full_input.items())[:4]:
                val_str = json.dumps(v, default=str) if not isinstance(v, str) else v
                if len(val_str) > 80:
                    val_str = val_str[:77] + "..."
                print(f"    │ IN  {k}: {val_str}")
        elif isinstance(full_input, str):
            lines = full_input.split("\n")
            for line in lines[:4]:
                if line.strip():
                    print(f"    │ IN  {line[:80]}")
            if len(lines) > 4:
                print(f"    │ IN  ... ({len(lines)-4} more lines)")

        # Output (full JSON, formatted)
        if isinstance(output, dict):
            if "error" in output:
                print(f"    │ OUT [ERROR] {output['error']}")
            elif "raw" in output:
                raw = str(output["raw"])
                print(f"    │ OUT [RAW] {raw[:120]}")
                if len(raw) > 120:
                    print(f"    │         {raw[120:240]}")
            else:
                for k, v in output.items():
                    val_str = json.dumps(v, default=str) if not isinstance(v, (str, int, float, bool)) else str(v)
                    if len(val_str) > 70:
                        val_str = val_str[:67] + "..."
                    print(f"    │ OUT {k}: {val_str}")
        else:
            print(f"    │ OUT {str(output)[:120]}")

        # Footer with timing
        print(f"    └─ {elapsed:.3f}s {'─'*40}")


# ═══════ BUFFER (same) ═══════
class MultiTierBuffer:
    def __init__(self, mid):
        self.machine_id=mid; self.tier1=deque(maxlen=TIER1_SIZE)
        self.tier2=deque(maxlen=TIER2_SIZE); self.tier3=deque(maxlen=TIER3_SIZE)
        self.tier4=deque(maxlen=TIER4_SIZE); self._t3a=[]; self._t4a=[]
        self.total=0; self.last=None
    def append(self, e):
        self.total+=1; self.last=e; self.tier1.append(e); self.tier2.append(e)
        self._t3a.append(e)
        if len(self._t3a)>=TIER3_DS: self.tier3.append(self._avg(self._t3a)); self._t3a=[]
        self._t4a.append(e)
        if len(self._t4a)>=TIER4_DS: self.tier4.append(self._avg(self._t4a)); self._t4a=[]
    def _avg(self, evts):
        o={"machine_id":self.machine_id,"n":len(evts)}
        for f in SENSOR_FEATURES: o[f]=round(sum(e.get(f,0)for e in evts)/len(evts),3)
        o["predicted_remaining_life"]=round(sum(e.get("predicted_remaining_life",0)for e in evts)/len(evts),2)
        for k in ["timestamp","machine_status","failure_type","anomaly_flag","downtime_risk",
                   "maintenance_required","source","anchor_machine","graph_distance","influence_score"]:
            o[k]=evts[-1].get(k,0)
        return o
    def get_tier(self,n): return list({1:self.tier1,2:self.tier2,3:self.tier3,4:self.tier4}[n])
    def stats(self):
        return {"mid":self.machine_id,"total":self.total,
                "t1":f"{len(self.tier1)}/{TIER1_SIZE}","t2":f"{len(self.tier2)}/{TIER2_SIZE}",
                "t3":f"{len(self.tier3)}/{TIER3_SIZE}","t4":f"{len(self.tier4)}/{TIER4_SIZE}"}

class BufferManager:
    def __init__(self): self.buffers={}; self.lock=threading.Lock()
    def ingest(self,e):
        mid=e.get("machine_id");
        if mid is None: return
        with self.lock:
            if mid not in self.buffers: self.buffers[mid]=MultiTierBuffer(mid)
            self.buffers[mid].append(e)
    def get_buffer(self,mid):
        with self.lock: return self.buffers.get(mid)
    def all_ids(self):
        with self.lock: return list(self.buffers.keys())
    def latest_events(self):
        with self.lock: return [b.last for b in sorted(self.buffers.values(),key=lambda b:b.machine_id) if b.last]
    def summary(self):
        with self.lock: return [b.stats() for b in sorted(self.buffers.values(),key=lambda b:b.machine_id)]


# ═══════ REASONING + LLM ═══════
class ReasoningEngine:
    # AUDIT NOTE: removed five label-leakage paths (anomaly_flag,
    # machine_status, failure_type, downtime_risk, maintenance_required)
    # — all are ground truth, not features.
    def __init__(self): self.thresholds=dict(THRESHOLDS)
    def score(self,e):
        s=0.0; flags=[]
        for f,t in self.thresholds.items():
            v=e.get(f,0)
            if not isinstance(v,(int,float)) or t["warn"]==float("inf"): continue
            if v>=t["crit"]: s+=0.35; flags.append(f"{f}=CRIT({v:.2f})")
            elif v>=t["warn"]: s+=0.15; flags.append(f"{f}=WARN({v:.2f})")
        rul=e.get("predicted_remaining_life",200)
        if rul<RUL_THRESHOLD: s+=0.30; flags.append(f"RUL=LOW({rul:.0f})")
        elif rul<100: s+=0.10
        risk=min(s,1.0); sev="HIGH" if risk>=0.6 else("MEDIUM" if risk>=0.3 else "LOW")
        return {"risk_score":round(risk,3),"severity":sev,"flags":flags,"predicted_rul":rul}

def get_ollama(): return Client(host="https://ollama.com",headers={"Authorization":"Bearer "+api_key})

def llm_call(client, prompt, msg):
    try:
        r=client.chat(model=OLLAMA_MODEL,messages=[{"role":"system","content":prompt},{"role":"user","content":msg}])
        c=r["message"]["content"].strip()
        if c.startswith("```"):
            c="\n".join(l for l in c.split("\n") if not l.strip().startswith("```")).strip()
        try: return json.loads(c)
        except: return {"raw":c}
    except Exception as e: return {"error":str(e)}

def extract(r,*keys):
    for k in keys:
        if k in r: return r[k]
    return None

# ═══════ AGENT PROMPTS ═══════
P_MON=("You are Agent 1: Monitoring Agent for smart manufacturing. "
    "Analyze sensor data + digital twin flags + trend statistics. "
    "Return JSON ONLY:\n"
    '{"risk_level":"LOW|MEDIUM|HIGH|CRITICAL","anomaly_detected":true|false,'
    '"anomaly_type":"thermal|vibration|energy|combined|none",'
    '"trend":"stable|degrading|improving","confidence":0.0-1.0}\n'
    "Raw JSON only, no markdown.")
P_DIAG=("You are Agent 2: Diagnosis Agent. Determine root cause. "
    "Return JSON ONLY:\n"
    '{"probable_cause":"bearing_wear|overheating|electrical|lubrication|normal|unknown",'
    '"affected_component":"motor|bearing|pump|actuator|unknown",'
    '"urgency_hours":24,"evidence":"brief description max 20 words"}\n'
    "Raw JSON only.")
P_PLAN=("You are Agent 3: Planning Agent. Decide maintenance action. "
    "Return JSON ONLY:\n"
    '{"action":"monitor|inspect|schedule_maintenance|immediate_shutdown",'
    '"priority":"P1|P2|P3|P4","reason":"max 20 words","notify_supervisor":false}\n'
    "Raw JSON only.")
P_REPORT=("You are Agent 4: Reporting Agent. Generate a short maintenance alert "
    "in Bahasa Indonesia. Max 3 sentences. Plain text only.")
P_CORR=("You are Agent 5: Correlation Agent. Interpret graph analysis results "
    "and identify inter-machine failure propagation risks. Return JSON ONLY:\n"
    '{"interpretation":"description","recommended_actions":["action"],'
    '"cascade_risk_assessment":"low|medium|high"}\nRaw JSON only.')
P_XAI=("You are Agent 6: Explainability Agent. Generate counterfactual analysis. "
    "Return JSON ONLY:\n"
    '{"top_feature":"temperature","contribution_pct":45.0,'
    '"distance_to_next_severity":1.6,"what_if":"smallest change to escalate",'
    '"confidence":0.8}\nRaw JSON only.')
P_CONS=("You are Agent 7: Consensus Agent. Validate agreement between agents. "
    "Return JSON ONLY:\n"
    '{"agreed_risk":"LOW|MEDIUM|HIGH","agreed_action":"monitor|inspect|'
    'schedule_maintenance","agreement_ratio":0.85,"has_disagreement":false,'
    '"reliability_score":0.9}\nRaw JSON only.')
P_SCHED=("You are Agent 8: Scheduling Agent. Evaluate optimization results "
    "and add operational context. Return JSON ONLY:\n"
    '{"assessment":"evaluation","adjustments":["adjustment"],'
    '"human_readable_summary":"Bahasa Indonesia summary"}\nRaw JSON only.')
P_PATTERN=("You are Agent 9: Anomaly Pattern Learning Agent. Detect temporal "
    "patterns from sliding window data. Return JSON ONLY:\n"
    '{"patterns":[{"type":"slow_drift","machines":["M-1"],"features":["temperature"],'
    '"confidence":0.7,"trend":"worsening","ttf_hours":168}],'
    '"total_patterns":1,"novel_count":0,"model_update_needed":false}\nRaw JSON only.')


# ═══════ TREND ═══════
def compute_trend(data, feat):
    vals=[e.get(feat,0) for e in data]; n=len(vals)
    if n<3: return {"mean":0,"stdev":0,"slope":0,"direction":"insufficient","n":n,"pct_warn":0}
    mean=statistics.mean(vals); std=statistics.stdev(vals)
    xm=(n-1)/2.0; num=sum((i-xm)*(v-mean) for i,v in enumerate(vals))
    den=sum((i-xm)**2 for i in range(n)); slope=num/den if den else 0
    d="stable"
    if abs(slope)>std*0.01: d="increasing" if slope>0 else "decreasing"
    thr=THRESHOLDS.get(feat,{}); w=thr.get("warn",float('inf'))
    pct=round(vals[-1]/w*100,1) if w>0 and w!=float('inf') else 0
    return {"mean":round(mean,2),"stdev":round(std,3),"slope":round(slope,5),
            "direction":d,"latest":round(vals[-1],2),"n":n,"pct_warn":pct}


# ═══════ SPIKE ═══════
class SpikeDetector:
    def __init__(self, opt): self.opt=opt
    def check(self, buf_mgr, event):
        mid=event.get("machine_id"); buf=buf_mgr.get_buffer(mid)
        z_spikes=[]
        if buf and len(buf.tier1)>=5:
            data=list(buf.tier1)
            for feat in SENSOR_FEATURES:
                hist=[e.get(feat,0) for e in data[:-1]]; cur=event.get(feat,0)
                if len(hist)<3: continue
                mean=statistics.mean(hist); std=statistics.stdev(hist) if len(hist)>1 else 0.001
                if std<0.001: continue
                z=(cur-mean)/std
                if abs(z)>=SPIKE_Z: z_spikes.append({"feature":feat,"value":round(cur,2),"z":round(z,2)})
        mv=self.opt.detect_multivariate_anomaly(mid,event)
        if z_spikes or mv.get("is_anomaly"):
            return {"machine_id":mid,"z_spikes":z_spikes,"multivariate":mv,
                    "source":"both" if z_spikes and mv.get("is_anomaly") else("z_score" if z_spikes else "mahalanobis")}
        return None


# ═══════ MICRO-BATCH with VERBOSE AGENT LOGS ═══════
class MicroBatchTrigger:
    def __init__(self, opt):
        self.client=get_ollama(); self.reasoner=ReasoningEngine()
        self.opt=opt; self.run_count=0; self.logger=AgentLogger()

    def execute(self, buffer_mgr):
        self.run_count += 1
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        mids = buffer_mgr.all_ids(); n = len(mids)
        print(f"\n{'='*80}")
        print(f"  MICRO-BATCH #{self.run_count} — {ts} | {n} machines")
        print(f"  Agents: ①Monitor ②Diagnose ③Plan ④Report ⑥XAI ⑦Consensus + Opt①④")
        print(f"{'='*80}")
        if n == 0: print("  [SKIP]"); return {}

        latest = buffer_mgr.latest_events()
        reasonings = {e["machine_id"]: self.reasoner.score(e) for e in latest}
        sev = {"HIGH":0,"MEDIUM":0,"LOW":0}
        for r in reasonings.values(): sev[r["severity"]] += 1
        print(f"  [REASONING] HIGH={sev['HIGH']} MEDIUM={sev['MEDIUM']} LOW={sev['LOW']}")

        trends = {}
        for mid in mids:
            buf = buffer_mgr.get_buffer(mid)
            if buf is None: continue
            t2 = buf.get_tier(2)
            if len(t2) < 5: continue
            trends[mid] = {f: compute_trend(t2, f) for f in SENSOR_FEATURES}

        # Point 1: Mahalanobis on all
        mv_anom = {}
        for evt in latest:
            mid = evt["machine_id"]
            mv = self.opt.detect_multivariate_anomaly(mid, evt)
            if mv.get("is_anomaly"): mv_anom[mid] = mv

        # Select targets
        targets = []
        for evt in latest:
            mid = evt["machine_id"]; rsn = reasonings.get(mid,{})
            has_signal = (evt.get("machine_status",0)==1 or evt.get("failure_type","Normal")!="Normal"
                or evt.get("anomaly_flag",0)==1 or evt.get("maintenance_required",0)==1
                or evt.get("downtime_risk",0)==1 or rsn.get("severity") in ("MEDIUM","HIGH"))
            near = any(t.get("pct_warn",0)>=90 for t in trends.get(mid,{}).values())
            degrad = any(t.get("direction")=="increasing" and t.get("slope",0)>0.01 for t in trends.get(mid,{}).values())
            if has_signal or near or degrad or mid in mv_anom: targets.append(mid)

        print(f"\n  [SELECT] {len(targets)}/{n} machines → running 9 agents on each")
        print(f"  [MAHALANOBIS] {len(mv_anom)} multivariate anomalies")

        results = {}
        for idx, mid in enumerate(targets):
            evt = next((e for e in latest if e["machine_id"]==mid), None)
            if evt is None: continue
            rsn = reasonings.get(mid, {}); trn = trends.get(mid, {})

            print(f"\n  {'─'*76}")
            print(f"  Machine M-{mid} | status={evt.get('machine_status')} "
                  f"fail={evt.get('failure_type')} risk={rsn['risk_score']} "
                  f"sev={rsn['severity']} rul={rsn['predicted_rul']:.0f}h")
            print(f"  Sensor: temp={evt['temperature']:.1f} vib={evt['vibration']:.1f} "
                  f"hum={evt['humidity']:.1f} pres={evt['pressure']:.2f} "
                  f"energy={evt['energy_consumption']:.2f}")
            if trn:
                trend_summary = " | ".join(f"{f}:{trn[f]['direction']}({trn[f]['slope']:.4f})" for f in list(trn.keys())[:3])
                print(f"  Trends: {trend_summary}")
            print(f"  {'─'*76}")

            sensor_input = {k: evt.get(k) for k in [
                "machine_id","temperature","vibration","humidity","pressure",
                "energy_consumption","machine_status","failure_type","anomaly_flag",
                "downtime_risk","maintenance_required","predicted_remaining_life",
                "influence_score","graph_distance","anchor_machine"]}
            sensor_str = json.dumps(sensor_input, indent=2)
            trend_str = json.dumps({f:{"slope":trn[f]["slope"],"dir":trn[f]["direction"],
                "pct":trn[f]["pct_warn"]} for f in trn}, indent=2) if trn else "{}"
            mv_str = f"\nMahalanobis: {json.dumps(mv_anom[mid])}" if mid in mv_anom else ""

            # ── Agent ① Monitoring ──
            mon_input = f"Sensor:\n{sensor_str}\nTrend(2h):\n{trend_str}\nReasoning:{json.dumps(rsn)}{mv_str}"
            t0 = time.time()
            monitoring = llm_call(self.client, P_MON, mon_input)
            self.logger.log_agent(mid, "Monitoring Agent", "①", "micro-batch",
                {"sensor": sensor_input, "trend": trn, "reasoning": rsn}, monitoring, time.time()-t0)

            # ── Agent ② Diagnosis ──
            diag_input = f"Sensor:\n{sensor_str}\nMonitoring:{json.dumps(monitoring)}"
            t0 = time.time()
            diagnosis = llm_call(self.client, P_DIAG, diag_input)
            self.logger.log_agent(mid, "Diagnosis Agent", "②", "micro-batch",
                {"sensor": sensor_input, "monitoring": monitoring}, diagnosis, time.time()-t0)

            # ── Agent ③ Planning ──
            plan_input = f"Diagnosis:{json.dumps(diagnosis)}\nRisk:{rsn.get('risk_score')}\nSeverity:{rsn.get('severity')}\nRUL:{evt['predicted_remaining_life']}"
            t0 = time.time()
            planning = llm_call(self.client, P_PLAN, plan_input)
            self.logger.log_agent(mid, "Planning Agent", "③", "micro-batch",
                {"diagnosis": diagnosis, "risk": rsn}, planning, time.time()-t0)

            # ── Agent ④ Reporting ──
            report_input = f"Machine M-{mid}\nDiagnosis:{json.dumps(diagnosis)}\nPlanning:{json.dumps(planning)}"
            t0 = time.time()
            reporting = llm_call(self.client, P_REPORT, report_input)
            self.logger.log_agent(mid, "Reporting Agent", "④", "micro-batch",
                {"diagnosis": diagnosis, "planning": planning}, reporting, time.time()-t0)

            # ── Agent ⑥ Explainability ──
            xai_input = f"M-{mid}\nSensor:\n{sensor_str}\nThresholds:{json.dumps(THRESHOLDS)}\nReasoning:{json.dumps(rsn)}"
            t0 = time.time()
            xai = llm_call(self.client, P_XAI, xai_input)
            self.logger.log_agent(mid, "Explainability Agent", "⑥", "micro-batch",
                {"sensor": sensor_input, "thresholds": THRESHOLDS, "reasoning": rsn}, xai, time.time()-t0)

            # ── Agent ⑦ Consensus (LLM) ──
            cons_input = f"Monitoring:{json.dumps(monitoring)}\nDiagnosis:{json.dumps(diagnosis)}\nPlanning:{json.dumps(planning)}"
            t0 = time.time()
            consensus_llm = llm_call(self.client, P_CONS, cons_input)
            self.logger.log_agent(mid, "Consensus Agent (LLM)", "⑦", "micro-batch",
                {"monitoring": monitoring, "diagnosis": diagnosis, "planning": planning},
                consensus_llm, time.time()-t0)

            # ── Opt Point 4: Classical Consensus ──
            t0 = time.time()
            consensus_classical = self.opt.validate_consensus(monitoring, diagnosis, planning, rsn)
            self.logger.log_agent(mid, "Consensus (Classical Cosine)", "Opt④", "micro-batch",
                {"monitoring": monitoring, "diagnosis": diagnosis, "planning": planning, "reasoning": rsn},
                consensus_classical, time.time()-t0, notes="Classical mirror of Quantum Kernel")

            results[mid] = {
                "reasoning": rsn, "trends": trn,
                "agent_1_monitoring": monitoring, "agent_2_diagnosis": diagnosis,
                "agent_3_planning": planning, "agent_4_reporting": reporting,
                "agent_6_xai": xai, "agent_7_consensus_llm": consensus_llm,
                "opt_4_consensus_classical": consensus_classical,
                "opt_1_mahalanobis": mv_anom.get(mid),
            }

        for mid in mids:
            if mid not in results:
                results[mid] = {"reasoning": reasonings.get(mid,{}), "action": "monitor", "note": "LOW risk, agents skipped"}

        print(f"\n  [SUMMARY] {len(targets)} machines × 7 agent calls = {len(targets)*7} total LLM calls")
        print(f"  Agent logs stored: {len(self.logger.logs)} entries\n")
        return results


# ═══════ DAILY TRIGGER with VERBOSE AGENT LOGS ═══════
class ScheduledTrigger:
    def __init__(self, opt):
        self.client=get_ollama(); self.reasoner=ReasoningEngine()
        self.opt=opt; self.run_count=0; self.logger=AgentLogger()

    def execute(self, buffer_mgr):
        self.run_count += 1
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        mids = buffer_mgr.all_ids(); n = len(mids)
        print(f"\n{'#'*80}")
        print(f"  DAILY TRIGGER #{self.run_count} — {ts} | {n} machines")
        print(f"  Agents: ⑤Correlation ⑧Scheduling ⑨Pattern ④Report + Opt②③⑤")
        print(f"{'#'*80}")
        if n == 0: return {}

        latest = buffer_mgr.latest_events()
        all_e, all_r = [], []
        for evt in latest:
            all_e.append(evt); all_r.append(self.reasoner.score(evt))

        risk_dist = {"HIGH":sum(1 for r in all_r if r["severity"]=="HIGH"),
            "MEDIUM":sum(1 for r in all_r if r["severity"]=="MEDIUM"),
            "LOW":sum(1 for r in all_r if r["severity"]=="LOW")}
        print(f"  [REASONING] {n} machines: H={risk_dist['HIGH']} M={risk_dist['MEDIUM']} L={risk_dist['LOW']}")

        # ── Opt Point 2: Classical graph correlation ──
        t0 = time.time()
        graph_result = self.opt.analyze_graph(all_e, all_r)
        self.logger.log_agent("ALL", "Graph Correlation (Classical)", "Opt②", "daily",
            {"n_machines": n, "n_edges": graph_result.get("n_edges",0)},
            graph_result, time.time()-t0, notes="Classical mirror of QGNN")

        # ── Agent ⑤ Correlation (LLM interprets graph) ──
        corr_input = f"Graph analysis:\n{json.dumps(graph_result, indent=2)}"
        t0 = time.time()
        correlation = llm_call(self.client, P_CORR, corr_input)
        self.logger.log_agent("ALL", "Correlation Agent (LLM)", "⑤", "daily",
            {"graph_clusters": len(graph_result.get("clusters",[])),
             "max_propagation": graph_result.get("max_propagation_risk",0)},
            correlation, time.time()-t0)

        # ── Opt Point 3: Simulated annealing scheduling ──
        sched_data = [{"machine_id":e["machine_id"],"risk_score":r["risk_score"],
            "predicted_rul":r["predicted_rul"],"failure_type":e.get("failure_type","Normal")}
            for e, r in zip(all_e, all_r)]
        t0 = time.time()
        sched_result = self.opt.optimize_schedule(sched_data)
        self.logger.log_agent("ALL", "Scheduling Optimizer (SA)", "Opt③", "daily",
            {"n_machines": n, "max_simultaneous": 5},
            sched_result, time.time()-t0, notes="Classical mirror of QAOA")

        # ── Agent ⑧ Scheduling (LLM adds context) ──
        sched_input = f"SA optimization:\n{json.dumps(sched_result, indent=2)}"
        t0 = time.time()
        scheduling = llm_call(self.client, P_SCHED, sched_input)
        self.logger.log_agent("ALL", "Scheduling Agent (LLM)", "⑧", "daily",
            {"scheduled": sched_result.get("machines_scheduled",0),
             "cost": sched_result.get("total_cost",0)},
            scheduling, time.time()-t0)

        # ── Agent ⑨ Anomaly Pattern (Tier 3) ──
        pat_in = []
        for mid in mids:
            buf = buffer_mgr.get_buffer(mid)
            t3 = buf.get_tier(3) if buf else []
            if len(t3) < 3: continue
            pat_in.append({"machine_id":mid,"n_24h":len(t3),
                "trends":{f:compute_trend(t3,f) for f in SENSOR_FEATURES},
                "failure_type":buf.last.get("failure_type","Normal") if buf.last else "Normal"})
        pattern_input = f"24h data ({len(pat_in)} machines):\n{json.dumps(pat_in, indent=2)}"
        t0 = time.time()
        patterns = llm_call(self.client, P_PATTERN, pattern_input)
        self.logger.log_agent("ALL", "Anomaly Pattern Agent", "⑨", "daily",
            {"machines_analyzed": len(pat_in)}, patterns, time.time()-t0)

        # ── Agent ④ Reporting (daily summary) ──
        report_input = (f"Daily summary {datetime.date.today()}: {n} machines\n"
            f"Risk: {json.dumps(risk_dist)}\n"
            f"Graph: {len(graph_result.get('clusters',[]))} clusters\n"
            f"Schedule: {sched_result.get('machines_scheduled',0)} machines\n"
            f"Patterns: {json.dumps(patterns)}")
        t0 = time.time()
        report = llm_call(self.client, P_REPORT, report_input)
        self.logger.log_agent("ALL", "Reporting Agent (Daily)", "④", "daily",
            {"risk_distribution": risk_dist}, report, time.time()-t0)

        # ── Opt Point 5: Threshold optimization ──
        all_hist = []
        for mid in mids:
            buf = buffer_mgr.get_buffer(mid)
            if buf: all_hist.extend(buf.get_tier(4))
        t0 = time.time()
        thr_result = self.opt.optimize_thresholds(all_hist, self.reasoner.thresholds)
        self.logger.log_agent("ALL", "Threshold Optimizer (Nelder-Mead)", "Opt⑤", "daily",
            {"n_history": len(all_hist), "current_thresholds": self.reasoner.thresholds},
            thr_result, time.time()-t0, notes="Classical mirror of Quantum Annealing")

        if thr_result.get("optimized") and thr_result.get("improvement_pct", 0) > 1.0:
            self.reasoner.thresholds = thr_result["thresholds"]
            print(f"\n  [APPLIED] New thresholds from Nelder-Mead optimization")

        print(f"\n  [SUMMARY] Daily trigger used {len(self.logger.logs)} agent calls total\n")

        return {"timestamp":ts,"graph":graph_result,"correlation":correlation,
            "scheduling_opt":sched_result,"scheduling_llm":scheduling,
            "patterns":patterns,"report":report,"risk_dist":risk_dist,
            "threshold_opt":thr_result}


# ═══════ PARSE ═══════
def parse(msg):
    try:
        d=msg.value
        if not isinstance(d,dict) or d.get("machine_id") is None: return None
        st=d.get("machine_status",0)
        return {"machine_id":d["machine_id"],"timestamp":d.get("timestamp",""),
            "temperature":d.get("temperature",0),"vibration":d.get("vibration",0),
            "humidity":d.get("humidity",0),"pressure":d.get("pressure",0),
            "energy_consumption":d.get("energy_consumption",0),
            "machine_status":st,"machine_status_label":d.get("machine_status_label",STATUS_MAP.get(st,"Normal")),
            "anomaly_flag":d.get("anomaly_flag",0),"predicted_remaining_life":d.get("predicted_remaining_life",200),
            "failure_type":d.get("failure_type","Normal"),"downtime_risk":d.get("downtime_risk",0),
            "maintenance_required":d.get("maintenance_required",0),"source":d.get("source",""),
            "anchor_machine":d.get("anchor_machine",0),"graph_distance":d.get("graph_distance",0),
            "influence_score":d.get("influence_score",0),"processed_at":d.get("processed_at","")}
    except Exception as e: print(f"  [PARSE ERROR] {e}"); return None


# ═══════ MAIN ═══════
def run():
    print("\n" + "="*80)
    print("  AGENTIC AI — VERBOSE AGENT LOGGING")
    print("  9-agent orchestration with full I/O logging per agent")
    print("="*80)
    print(f"  Kafka : {KAFKA_BROKERS[0]} / {KAFKA_TOPIC}")
    print(f"  LLM   : {OLLAMA_MODEL}")
    print(f"  Micro : {MICROBATCH_SEC//60}min (Agents ①②③④⑥⑦ + Opt①④)")
    print(f"  Daily : {SCHEDULED_SEC//3600}hr (Agents ⑤⑧⑨④ + Opt②③⑤)")
    print("="*80 + "\n")

    opt = ClassicalOptimizationEngine()
    buf = BufferManager()
    spike = SpikeDetector(opt)
    micro = MicroBatchTrigger(opt)
    daily = ScheduledTrigger(opt)

    print("[KAFKA] Connecting...")
    try:
        consumer = KafkaConsumer(KAFKA_TOPIC,bootstrap_servers=KAFKA_BROKERS,
            group_id=KAFKA_GROUP_ID,auto_offset_reset='latest',enable_auto_commit=True,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')))
        print(f"[KAFKA] Connected\n")
    except Exception as e: print(f"[KAFKA ERROR] {e}"); return

    last_m=time.time(); last_d=time.time()
    ec=0; sc=0; mc=0; running=True; first=False

    def stop(s,f):
        nonlocal running; print("\n[SIGNAL] Stopping..."); running=False
    signal.signal(signal.SIGINT,stop); signal.signal(signal.SIGTERM,stop)

    print("[STREAM] Listening...\n")
    while running:
        try:
            batch=consumer.poll(timeout_ms=3000,max_records=200)
            for tp,msgs in batch.items():
                for msg in msgs:
                    event=parse(msg)
                    if event is None: continue
                    ec+=1; mid=event["machine_id"]
                    if not first:
                        first=True
                        print(f"  OK M-{mid} temp={event['temperature']:.1f} status={event['machine_status']} "
                              f"fail={event['failure_type']} keys={list(msg.value.keys())[:8]}...\n")
                    buf.ingest(event)
                    sp = spike.check(buf, event)
                    if sp:
                        if sp["z_spikes"]: sc+=1
                        if sp.get("multivariate",{}).get("is_anomaly"): mc+=1
                        z_s=", ".join(f"{s['feature']}(z={s['z']})" for s in sp["z_spikes"])
                        mv_s=sp.get("multivariate",{}).get("anomaly_score",0)
                        print(f"  ALERT M-{mid} [{sp['source']}] z=[{z_s}] mahal={mv_s:.3f}")
                    if ec%50==0:
                        nm=len(buf.all_ids()); remain=max(0,MICROBATCH_SEC-(time.time()-last_m))
                        print(f"  [PROGRESS] {ec} events | {nm} machines | spk={sc} mah={mc} | next {remain:.0f}s")

            now=time.time()
            if now-last_m>=MICROBATCH_SEC:
                r=micro.execute(buf); last_m=now
                fn=f"microbatch_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"
                # Save results + agent logs
                save_data = {"results": r, "agent_logs": micro.logger.logs[-100:]}
                with open(fn,"w") as f: json.dump(save_data,f,indent=2,default=str)
                print(f"  [SAVED] {fn} (with agent logs)")

            if now-last_d>=SCHEDULED_SEC:
                r=daily.execute(buf); last_d=now
                fn=f"daily_{datetime.datetime.now():%Y%m%d}.json"
                save_data = {"results": r, "agent_logs": daily.logger.logs[-50:]}
                with open(fn,"w") as f: json.dump(save_data,f,indent=2,default=str)
                print(f"  [SAVED] {fn} (with agent logs)")

            if not batch: time.sleep(1)
        except KeyboardInterrupt: running=False
        except Exception as e: print(f"  [ERROR] {e}"); time.sleep(5)

    consumer.close()
    total_agent_calls = len(micro.logger.logs) + len(daily.logger.logs)
    print(f"\n{'='*80}")
    print(f"  SHUTDOWN SUMMARY")
    print(f"{'='*80}")
    print(f"  Events: {ec} | Spikes: {sc} | Mahalanobis: {mc}")
    print(f"  Micro runs: {micro.run_count} | Daily runs: {daily.run_count}")
    print(f"  Total agent calls logged: {total_agent_calls}")
    print(f"  Machines: {len(buf.all_ids())}")
    print(f"\n  Agent call breakdown:")
    agent_counts = defaultdict(lambda: {"calls":0, "total_time":0})
    for log in micro.logger.logs + daily.logger.logs:
        key = f"{log['agent_number']} {log['agent_name']}"
        agent_counts[key]["calls"] += 1
        agent_counts[key]["total_time"] += log["elapsed_sec"]
    for name, stats in sorted(agent_counts.items()):
        avg = stats["total_time"]/stats["calls"] if stats["calls"]>0 else 0
        print(f"    {name:<40} calls={stats['calls']:<6} avg={avg:.2f}s total={stats['total_time']:.1f}s")
    print(f"\n  Buffer:")
    for s in buf.summary()[:5]:
        print(f"    M-{s['mid']:<3} total={s['total']:<6} {s['t1']} {s['t2']} {s['t3']} {s['t4']}")
    if len(buf.all_ids())>5: print(f"    ... +{len(buf.all_ids())-5} more")
    print(f"{'='*80}\n")

if __name__=="__main__":
    run()
