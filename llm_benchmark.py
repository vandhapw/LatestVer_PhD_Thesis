"""
============================================================
LLM MODEL BENCHMARK FOR AGENTIC AI PIPELINE
============================================================
Tests 3 models across 6 agent roles using REAL sensor data
from MongoDB/Kafka, measuring 7 performance dimensions.

Models:
  1. glm-5.1:cloud        (current production model)
  2. qwen3.5:397b-cloud   (large parameter alternative)
  3. deepseek-v4-pro:cloud (reasoning-optimized alternative)

Agents Tested (6 of 9 — those requiring LLM reasoning):
  ① Monitoring Agent    — risk assessment
  ② Diagnosis Agent     — root-cause analysis
  ③ Planning Agent      — action recommendation
  ④ Reporting Agent     — natural language alert
  ⑥ Explainability Agent — counterfactual XAI
  ⑤ Correlation Agent   — inter-machine dependency

Metrics (7 dimensions):
  M1. JSON parse success rate (structural reliability)
  M2. Schema compliance rate (correct field names)
  M3. Reasoning quality score (domain-appropriate answers)
  M4. Response latency (seconds per call)
  M5. Cross-agent consistency (diagnosis matches planning)
  M6. Temporal consistency (same input → same output)
  M7. Cost efficiency (quality per dollar)

Data: Real sensor data from MongoDB (same as production pipeline)
============================================================
"""

import json
import time
import datetime
import statistics
from collections import defaultdict
from pymongo import MongoClient
from ollama import Client

from config import (
    OLLAMA_API_KEY, MONGODB_URI, MONGO_DB, MONGO_COLLECTION,
    get_warn_crit_thresholds,
)
api_key = OLLAMA_API_KEY

MODELS = [
    {"name": "glm-5.1:cloud",        "short": "GLM-5.1",    "cost_in": 0.20, "cost_out": 0.80},
    {"name": "qwen3.5:397b-cloud",   "short": "Qwen3.5",    "cost_in": 0.30, "cost_out": 1.20},
    {"name": "deepseek-v4-pro:cloud", "short": "DeepSeek-V4","cost_in": 0.25, "cost_out": 1.00},
]

# Test on 5 machines: mix of Normal, Warning/Fault, and near-threshold
TEST_MACHINE_IDS = [1, 5, 9, 37, 50]
N_CONSISTENCY_RUNS = 3  # repeat each call N times for consistency check

# AUDIT NOTE: thresholds now come from config (derived from training split).
THRESHOLDS = get_warn_crit_thresholds()


# ═══════════════════════════════════════════════
# AGENT PROMPTS + EXPECTED SCHEMAS
# ═══════════════════════════════════════════════

AGENTS = {
    "monitoring": {
        "number": "①",
        "name": "Monitoring Agent",
        "prompt": (
            "You are Agent 1: Monitoring Agent for smart manufacturing. "
            "Analyze sensor data and digital twin flags. "
            "You MUST respond with ONLY a JSON object, no other text:\n"
            '{"risk_level":"LOW|MEDIUM|HIGH|CRITICAL","anomaly_detected":true|false,'
            '"anomaly_type":"thermal|vibration|energy|combined|none","confidence":0.0-1.0}\n'
            "IMPORTANT: respond with raw JSON only, no markdown, no explanation."
        ),
        "expected_keys": {"risk_level", "anomaly_detected", "anomaly_type", "confidence"},
        "valid_values": {
            "risk_level": {"LOW", "MEDIUM", "HIGH", "CRITICAL"},
            "anomaly_type": {"thermal", "vibration", "energy", "combined", "none"},
        },
    },
    "diagnosis": {
        "number": "②",
        "name": "Diagnosis Agent",
        "prompt": (
            "You are Agent 2: Diagnosis Agent for predictive maintenance. "
            "Determine root cause from sensor data. "
            "You MUST respond with ONLY a JSON object:\n"
            '{"probable_cause":"bearing_wear|overheating|electrical|lubrication|normal|unknown",'
            '"affected_component":"motor|bearing|pump|actuator|unknown",'
            '"urgency_hours":24,"evidence":"brief description max 20 words"}\n'
            "Raw JSON only."
        ),
        "expected_keys": {"probable_cause", "affected_component", "urgency_hours"},
        "valid_values": {
            "probable_cause": {"bearing_wear", "overheating", "electrical", "lubrication", "normal", "unknown"},
            "affected_component": {"motor", "bearing", "pump", "actuator", "unknown"},
        },
    },
    "planning": {
        "number": "③",
        "name": "Planning Agent",
        "prompt": (
            "You are Agent 3: Planning Agent. Decide maintenance action. "
            "You MUST respond with ONLY a JSON object:\n"
            '{"action":"monitor|inspect|schedule_maintenance|immediate_shutdown",'
            '"priority":"P1|P2|P3|P4","reason":"max 20 words","notify_supervisor":false}\n'
            "Raw JSON only."
        ),
        "expected_keys": {"action", "priority", "reason"},
        "valid_values": {
            "action": {"monitor", "inspect", "schedule_maintenance", "immediate_shutdown"},
            "priority": {"P1", "P2", "P3", "P4"},
        },
    },
    "reporting": {
        "number": "④",
        "name": "Reporting Agent",
        "prompt": (
            "You are Agent 4: Reporting Agent. Generate a maintenance alert "
            "in Bahasa Indonesia. Max 3 sentences. Plain text only, no JSON."
        ),
        "expected_keys": set(),  # free text
        "valid_values": {},
    },
    "explainability": {
        "number": "⑥",
        "name": "Explainability Agent",
        "prompt": (
            "You are Agent 6: Explainability Agent (XAI). "
            "Generate counterfactual analysis. "
            "You MUST respond with ONLY a JSON object:\n"
            '{"top_feature":"temperature|vibration|humidity|energy_consumption",'
            '"contribution_pct":45.0,"distance_to_next_severity":1.6,'
            '"what_if":"smallest change to escalate severity","confidence":0.8}\n'
            "Raw JSON only."
        ),
        "expected_keys": {"top_feature", "distance_to_next_severity", "what_if", "confidence"},
        "valid_values": {
            "top_feature": {"temperature", "vibration", "humidity", "energy_consumption",
                            "energy", "pressure", "rul", "predicted_remaining_life"},
        },
    },
    "correlation": {
        "number": "⑤",
        "name": "Correlation Agent",
        "prompt": (
            "You are Agent 5: Correlation Agent. Analyze inter-machine dependencies. "
            "You MUST respond with ONLY a JSON object:\n"
            '{"clusters":[{"machines":["M-1","M-3"],"type":"thermal|electrical|vibration",'
            '"propagation_risk":0.5,"root_machine":"M-1"}],'
            '"max_propagation_risk":0.5,"cascade_warning":false}\n'
            "Raw JSON only."
        ),
        "expected_keys": {"clusters", "max_propagation_risk", "cascade_warning"},
        "valid_values": {},
    },
}


# ═══════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════

def load_test_data():
    """Load real sensor data from MongoDB for test machines."""
    print("[DATA] Loading from MongoDB...")
    client = MongoClient(MONGODB_URI)
    db = client[MONGO_DB]
    col = db[MONGO_COLLECTION]

    events = {}
    for mid in TEST_MACHINE_IDS:
        doc = col.find_one({"machine_id": mid}, sort=[("timestamp", -1)])
        if doc:
            events[mid] = {
                "machine_id": mid,
                "temperature": doc.get("temperature", 0),
                "vibration": doc.get("vibration", 0),
                "humidity": doc.get("humidity", 0),
                "pressure": doc.get("pressure", 0),
                "energy_consumption": doc.get("energy_consumption", 0),
                "machine_status": doc.get("machine_status", 0),
                "failure_type": doc.get("failure_type", "Normal"),
                "anomaly_flag": doc.get("anomaly_flag", 0),
                "predicted_remaining_life": doc.get("predicted_remaining_life", 200),
                "downtime_risk": doc.get("downtime_risk", 0),
                "maintenance_required": doc.get("maintenance_required", 0),
                "influence_score": doc.get("influence_score", 0),
                "graph_distance": doc.get("graph_distance", 0),
                "anchor_machine": doc.get("anchor_machine", 0),
            }
    client.close()
    print(f"[DATA] Loaded {len(events)} machines: {list(events.keys())}")
    for mid, e in events.items():
        print(f"  M-{mid:<3} temp={e['temperature']:.1f} vib={e['vibration']:.1f} "
              f"status={e['machine_status']} fail={e['failure_type']}")
    return events


def build_user_message(agent_key, event, prev_outputs=None):
    """Build the user message for each agent, mimicking production pipeline."""
    sensor_str = json.dumps({
        "machine_id": event["machine_id"],
        "temperature": event["temperature"],
        "vibration": event["vibration"],
        "humidity": event["humidity"],
        "pressure": event["pressure"],
        "energy_consumption": event["energy_consumption"],
        "machine_status": event["machine_status"],
        "failure_type": event["failure_type"],
        "anomaly_flag": event["anomaly_flag"],
        "predicted_remaining_life": event["predicted_remaining_life"],
        "downtime_risk": event["downtime_risk"],
        "maintenance_required": event["maintenance_required"],
    }, indent=2)

    if agent_key == "monitoring":
        return f"Sensor data:\n{sensor_str}"
    elif agent_key == "diagnosis":
        mon = prev_outputs.get("monitoring", {}) if prev_outputs else {}
        return f"Sensor:\n{sensor_str}\nMonitoring result: {json.dumps(mon)}"
    elif agent_key == "planning":
        diag = prev_outputs.get("diagnosis", {}) if prev_outputs else {}
        return (f"Diagnosis: {json.dumps(diag)}\n"
                f"Risk score: 0.15\nSeverity: LOW\n"
                f"RUL: {event['predicted_remaining_life']}")
    elif agent_key == "reporting":
        diag = prev_outputs.get("diagnosis", {}) if prev_outputs else {}
        plan = prev_outputs.get("planning", {}) if prev_outputs else {}
        return f"Machine M-{event['machine_id']}\nDiagnosis: {json.dumps(diag)}\nPlanning: {json.dumps(plan)}"
    elif agent_key == "explainability":
        return (f"Machine M-{event['machine_id']}\nSensor:\n{sensor_str}\n"
                f"Thresholds: {json.dumps(THRESHOLDS)}")
    elif agent_key == "correlation":
        return f"Factory data (5 machines):\n{sensor_str}"
    return sensor_str


# ═══════════════════════════════════════════════
# LLM CALL WITH METRICS
# ═══════════════════════════════════════════════

def call_model(client, model_name, system_prompt, user_msg):
    """Call LLM and return result + timing + raw content."""
    t0 = time.time()
    try:
        resp = client.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ]
        )
        elapsed = time.time() - t0
        raw = resp["message"]["content"].strip()

        # Strip markdown fences
        clean = raw
        if clean.startswith("```"):
            lines = clean.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            clean = "\n".join(lines).strip()

        # Try JSON parse
        try:
            parsed = json.loads(clean)
            return {"parsed": parsed, "raw": raw, "json_valid": True,
                    "elapsed": round(elapsed, 3), "error": None}
        except json.JSONDecodeError:
            return {"parsed": None, "raw": raw, "json_valid": False,
                    "elapsed": round(elapsed, 3), "error": "json_parse_failed"}

    except Exception as e:
        elapsed = time.time() - t0
        return {"parsed": None, "raw": "", "json_valid": False,
                "elapsed": round(elapsed, 3), "error": str(e)}


# ═══════════════════════════════════════════════
# SCORING FUNCTIONS
# ═══════════════════════════════════════════════

def score_json_validity(result):
    """M1: Did the model return valid JSON?"""
    return 1.0 if result["json_valid"] else 0.0


def score_schema_compliance(result, agent_config):
    """M2: Does JSON contain the expected keys with valid values?"""
    if not result["json_valid"] or result["parsed"] is None:
        return 0.0

    expected = agent_config["expected_keys"]
    if not expected:  # free text agent (reporting)
        return 1.0 if len(result["raw"]) > 10 else 0.0

    parsed = result["parsed"]
    present_keys = set(parsed.keys())

    # Key presence score
    key_score = len(expected & present_keys) / len(expected) if expected else 1.0

    # Value validity score
    valid_values = agent_config["valid_values"]
    value_score = 1.0
    n_checked = 0
    for field, allowed in valid_values.items():
        val = parsed.get(field)
        if val is not None:
            n_checked += 1
            if str(val).lower() not in {v.lower() for v in allowed}:
                value_score -= 1.0 / len(valid_values)

    value_score = max(0, value_score)
    return round((key_score * 0.6 + value_score * 0.4), 4)


def score_reasoning_quality(result, agent_key, event):
    """
    M3: Domain-specific reasoning quality.
    Checks if the answer makes sense given the sensor data.
    NOT a subjective evaluation — uses deterministic rules.
    """
    if not result["json_valid"] or result["parsed"] is None:
        if agent_key == "reporting":
            # For reporting, check text quality
            raw = result["raw"]
            score = 0.0
            if len(raw) > 20: score += 0.3
            if any(c in raw for c in ["mesin", "Mesin", "M-", "suhu", "temperatur"]): score += 0.3
            if len(raw) < 500: score += 0.2  # not too verbose
            if result["error"] is None: score += 0.2
            return round(score, 4)
        return 0.0

    parsed = result["parsed"]
    score = 0.0

    if agent_key == "monitoring":
        # If machine has failure_type != Normal, should detect anomaly
        has_failure = event.get("failure_type", "Normal") != "Normal"
        has_fault = event.get("machine_status", 0) == 1
        anomaly_detected = parsed.get("anomaly_detected", False)
        risk = parsed.get("risk_level", "LOW")

        if has_failure or has_fault:
            if anomaly_detected: score += 0.4
            if risk in ("MEDIUM", "HIGH", "CRITICAL"): score += 0.3
        else:
            if not anomaly_detected: score += 0.4
            if risk == "LOW": score += 0.3

        conf = parsed.get("confidence", 0)
        if isinstance(conf, (int, float)) and 0 <= conf <= 1: score += 0.3

    elif agent_key == "diagnosis":
        cause = parsed.get("probable_cause", "unknown")
        failure = event.get("failure_type", "Normal")
        # Check cause matches failure type
        match_map = {
            "Overheating": ["overheating"],
            "Electrical Fault": ["electrical"],
            "Vibration Issue": ["bearing_wear", "lubrication"],
            "Pressure Drop": ["lubrication"],
            "Normal": ["normal"],
        }
        expected_causes = match_map.get(failure, ["unknown"])
        if cause in expected_causes: score += 0.5
        elif cause != "unknown": score += 0.2  # at least tried

        urgency = parsed.get("urgency_hours")
        if isinstance(urgency, (int, float)) and 0 < urgency <= 168: score += 0.3

        if parsed.get("affected_component", "unknown") != "unknown": score += 0.2

    elif agent_key == "planning":
        action = parsed.get("action", "monitor")
        has_issue = (event.get("failure_type", "Normal") != "Normal"
                     or event.get("machine_status", 0) == 1)
        if has_issue:
            if action in ("inspect", "schedule_maintenance", "immediate_shutdown"): score += 0.5
        else:
            if action == "monitor": score += 0.5

        priority = parsed.get("priority", "")
        if priority in ("P1", "P2", "P3", "P4"): score += 0.25
        if parsed.get("reason") and len(str(parsed["reason"])) > 5: score += 0.25

    elif agent_key == "explainability":
        top = parsed.get("top_feature")
        if top: score += 0.25
        dist = parsed.get("distance_to_next_severity")
        if isinstance(dist, (int, float)) and dist > 0: score += 0.25
        whatif = parsed.get("what_if", "")
        if whatif and len(str(whatif)) > 10: score += 0.25
        conf = parsed.get("confidence", 0)
        if isinstance(conf, (int, float)) and 0 <= conf <= 1: score += 0.25

    elif agent_key == "correlation":
        clusters = parsed.get("clusters", [])
        if isinstance(clusters, list): score += 0.3
        if len(clusters) > 0: score += 0.3
        prop = parsed.get("max_propagation_risk")
        if isinstance(prop, (int, float)) and 0 <= prop <= 1: score += 0.2
        if "cascade_warning" in parsed: score += 0.2

    return round(min(1.0, score), 4)


def score_consistency(results_list):
    """
    M6: Temporal consistency — given same input, does model give same answer?
    Compares N runs of the same prompt.
    """
    if len(results_list) < 2:
        return 1.0

    # For JSON outputs: compare key fields across runs
    valid = [r["parsed"] for r in results_list if r["json_valid"] and r["parsed"]]
    if len(valid) < 2:
        return 0.0

    # Count how many runs agree on each field
    all_keys = set()
    for v in valid:
        all_keys.update(v.keys())

    agreements = 0
    total = 0
    for key in all_keys:
        values = [str(v.get(key, "")) for v in valid]
        total += 1
        # Most common value
        from collections import Counter
        mc = Counter(values).most_common(1)[0][1]
        agreements += mc / len(values)

    return round(agreements / total, 4) if total > 0 else 0.0


# ═══════════════════════════════════════════════
# MAIN BENCHMARK
# ═══════════════════════════════════════════════

def run_benchmark():
    print("\n" + "=" * 80)
    print("  LLM MODEL BENCHMARK FOR AGENTIC AI PIPELINE")
    print("  Testing 3 models × 6 agents × 5 machines × 3 runs")
    print("=" * 80)

    # Load data
    events = load_test_data()
    if not events:
        print("[ERROR] No data loaded.")
        return

    client = Client(
        host="https://ollama.com",
        headers={"Authorization": "Bearer " + api_key}
    )

    # Track all results
    all_results = {}  # model → agent → machine → [runs]
    agent_keys = ["monitoring", "diagnosis", "planning", "reporting", "explainability", "correlation"]

    total_calls = len(MODELS) * len(agent_keys) * len(events) * N_CONSISTENCY_RUNS
    call_count = 0

    for model in MODELS:
        model_name = model["name"]
        model_short = model["short"]
        all_results[model_short] = {}

        print(f"\n{'━' * 80}")
        print(f"  MODEL: {model_name}")
        print(f"{'━' * 80}")

        for agent_key in agent_keys:
            agent = AGENTS[agent_key]
            all_results[model_short][agent_key] = {}

            print(f"\n  Agent {agent['number']} {agent['name']}:")

            for mid in TEST_MACHINE_IDS:
                event = events[mid]
                runs = []

                for run_idx in range(N_CONSISTENCY_RUNS):
                    call_count += 1

                    # Build input (for sequential agents, use first run's output as prev)
                    prev = {}
                    if run_idx == 0 or agent_key in ("monitoring", "reporting", "explainability", "correlation"):
                        prev = {}
                    else:
                        # Use previous agent outputs from this model's first run
                        if "monitoring" in all_results[model_short]:
                            mon_runs = all_results[model_short]["monitoring"].get(mid, [])
                            if mon_runs and mon_runs[0]["parsed"]:
                                prev["monitoring"] = mon_runs[0]["parsed"]
                        if "diagnosis" in all_results[model_short]:
                            diag_runs = all_results[model_short]["diagnosis"].get(mid, [])
                            if diag_runs and diag_runs[0]["parsed"]:
                                prev["diagnosis"] = diag_runs[0]["parsed"]

                    user_msg = build_user_message(agent_key, event, prev)
                    result = call_model(client, model_name, agent["prompt"], user_msg)

                    # Compute per-call metrics
                    result["m1_json"] = score_json_validity(result)
                    result["m2_schema"] = score_schema_compliance(result, agent)
                    result["m3_reasoning"] = score_reasoning_quality(result, agent_key, event)

                    runs.append(result)

                    status = "✓" if result["json_valid"] else "✗"
                    print(f"    [{call_count}/{total_calls}] M-{mid:<3} run{run_idx+1} "
                          f"{status} {result['elapsed']:.1f}s "
                          f"json={result['m1_json']:.0f} schema={result['m2_schema']:.2f} "
                          f"reasoning={result['m3_reasoning']:.2f}")

                all_results[model_short][agent_key][mid] = runs

    # ═══════════════════════════════════════════════
    # COMPUTE AGGREGATE METRICS
    # ═══════════════════════════════════════════════

    print("\n\n" + "=" * 80)
    print("  BENCHMARK RESULTS")
    print("=" * 80)

    model_metrics = {}

    for model in MODELS:
        ms = model["short"]
        metrics = {
            "m1_json_rate": [],
            "m2_schema_rate": [],
            "m3_reasoning_avg": [],
            "m4_latency_avg": [],
            "m4_latency_p95": [],
            "m6_consistency": [],
            "per_agent": {},
        }

        for agent_key in agent_keys:
            agent_m1, agent_m2, agent_m3, agent_m4 = [], [], [], []
            agent_consistency = []

            for mid in TEST_MACHINE_IDS:
                runs = all_results[ms][agent_key].get(mid, [])
                for r in runs:
                    agent_m1.append(r["m1_json"])
                    agent_m2.append(r["m2_schema"])
                    agent_m3.append(r["m3_reasoning"])
                    agent_m4.append(r["elapsed"])

                if len(runs) >= 2:
                    agent_consistency.append(score_consistency(runs))

            metrics["m1_json_rate"].extend(agent_m1)
            metrics["m2_schema_rate"].extend(agent_m2)
            metrics["m3_reasoning_avg"].extend(agent_m3)
            metrics["m4_latency_avg"].extend(agent_m4)
            metrics["m6_consistency"].extend(agent_consistency)

            metrics["per_agent"][agent_key] = {
                "m1": round(statistics.mean(agent_m1), 4) if agent_m1 else 0,
                "m2": round(statistics.mean(agent_m2), 4) if agent_m2 else 0,
                "m3": round(statistics.mean(agent_m3), 4) if agent_m3 else 0,
                "m4": round(statistics.mean(agent_m4), 3) if agent_m4 else 0,
                "m6": round(statistics.mean(agent_consistency), 4) if agent_consistency else 0,
                "n_calls": len(agent_m1),
            }

        all_latencies = metrics["m4_latency_avg"]
        all_latencies_sorted = sorted(all_latencies)
        p95_idx = int(len(all_latencies_sorted) * 0.95)

        # Cost estimation
        avg_input_tokens = 800   # estimated from prompt sizes
        avg_output_tokens = 150  # estimated from JSON responses
        total_calls_model = len(metrics["m1_json_rate"])
        cost_total = (avg_input_tokens * model["cost_in"] +
                      avg_output_tokens * model["cost_out"]) / 1_000_000 * total_calls_model

        model_metrics[ms] = {
            "m1_json_rate": round(statistics.mean(metrics["m1_json_rate"]), 4),
            "m2_schema_rate": round(statistics.mean(metrics["m2_schema_rate"]), 4),
            "m3_reasoning_avg": round(statistics.mean(metrics["m3_reasoning_avg"]), 4),
            "m4_latency_avg": round(statistics.mean(all_latencies), 3),
            "m4_latency_p95": round(all_latencies_sorted[p95_idx] if all_latencies_sorted else 0, 3),
            "m6_consistency": round(statistics.mean(metrics["m6_consistency"]), 4) if metrics["m6_consistency"] else 0,
            "m7_cost_total": round(cost_total, 4),
            "m7_quality_per_dollar": round(
                statistics.mean(metrics["m3_reasoning_avg"]) / cost_total if cost_total > 0 else 0, 2),
            "total_calls": total_calls_model,
            "per_agent": metrics["per_agent"],
        }

    # ═══════════════════════════════════════════════
    # PRINT COMPARISON TABLES
    # ═══════════════════════════════════════════════

    # Table 1: Overall comparison
    print(f"\n{'─' * 80}")
    print(f"  TABLE 1: Overall Model Comparison")
    print(f"{'─' * 80}")
    header = (f"  {'Metric':<28} " +
              " ".join(f"{m['short']:>14}" for m in MODELS))
    print(header)
    print(f"  {'─' * 72}")

    metric_labels = [
        ("M1 JSON parse rate", "m1_json_rate", "{:.1%}"),
        ("M2 Schema compliance", "m2_schema_rate", "{:.1%}"),
        ("M3 Reasoning quality", "m3_reasoning_avg", "{:.1%}"),
        ("M4 Latency avg (sec)", "m4_latency_avg", "{:.2f}s"),
        ("M4 Latency p95 (sec)", "m4_latency_p95", "{:.2f}s"),
        ("M6 Temporal consistency", "m6_consistency", "{:.1%}"),
        ("M7 Est. cost (benchmark)", "m7_cost_total", "${:.4f}"),
        ("M7 Quality per dollar", "m7_quality_per_dollar", "{:.1f}"),
    ]

    for label, key, fmt in metric_labels:
        vals = []
        for m in MODELS:
            v = model_metrics[m["short"]][key]
            vals.append(fmt.format(v))
        print(f"  {label:<28} " + " ".join(f"{v:>14}" for v in vals))

    # Find best per metric
    print(f"\n  {'Best model per metric:'}")
    for label, key, _ in metric_labels:
        if "cost" in key.lower() or "latency" in key.lower():
            best = min(MODELS, key=lambda m: model_metrics[m["short"]][key])
        else:
            best = max(MODELS, key=lambda m: model_metrics[m["short"]][key])
        print(f"    {label:<28} → {best['short']}")

    # Table 2: Per-agent breakdown
    print(f"\n{'─' * 80}")
    print(f"  TABLE 2: Per-Agent Performance Breakdown")
    print(f"{'─' * 80}")

    for agent_key in agent_keys:
        agent = AGENTS[agent_key]
        print(f"\n  {agent['number']} {agent['name']}:")
        print(f"  {'Metric':<20} " +
              " ".join(f"{m['short']:>14}" for m in MODELS))
        print(f"  {'─' * 60}")
        for metric, label in [("m1", "JSON rate"), ("m2", "Schema"), ("m3", "Reasoning"),
                               ("m4", "Latency(s)"), ("m6", "Consistency")]:
            vals = []
            for m in MODELS:
                v = model_metrics[m["short"]]["per_agent"][agent_key][metric]
                if metric == "m4":
                    vals.append(f"{v:.2f}s")
                else:
                    vals.append(f"{v:.1%}")
            print(f"  {label:<20} " + " ".join(f"{v:>14}" for v in vals))

    # Table 3: Cross-agent consistency (M5)
    print(f"\n{'─' * 80}")
    print(f"  TABLE 3: Cross-Agent Consistency (does diagnosis match planning?)")
    print(f"{'─' * 80}")

    for m in MODELS:
        ms = m["short"]
        consistent = 0
        total = 0
        for mid in TEST_MACHINE_IDS:
            diag_runs = all_results[ms]["diagnosis"].get(mid, [])
            plan_runs = all_results[ms]["planning"].get(mid, [])
            if not diag_runs or not plan_runs:
                continue
            diag = diag_runs[0]["parsed"] if diag_runs[0]["parsed"] else {}
            plan = plan_runs[0]["parsed"] if plan_runs[0]["parsed"] else {}

            total += 1
            cause = diag.get("probable_cause", "unknown")
            action = plan.get("action", "monitor")

            # Consistency check: serious cause → serious action
            if cause in ("overheating", "electrical", "bearing_wear"):
                if action in ("inspect", "schedule_maintenance", "immediate_shutdown"):
                    consistent += 1
            elif cause in ("normal", "unknown"):
                if action in ("monitor", "inspect"):
                    consistent += 1
            else:
                consistent += 0.5  # partial

        rate = consistent / total if total > 0 else 0
        print(f"  {ms:<18} {rate:.1%} ({consistent:.0f}/{total})")

    # ═══════════════════════════════════════════════
    # SAVE RESULTS
    # ═══════════════════════════════════════════════

    output = {
        "timestamp": str(datetime.datetime.now()),
        "models": [m["name"] for m in MODELS],
        "test_machines": TEST_MACHINE_IDS,
        "n_runs_per_call": N_CONSISTENCY_RUNS,
        "aggregate_metrics": model_metrics,
        "detailed_results": {},
    }

    # Add detailed results (without raw text to keep file manageable)
    for ms in all_results:
        output["detailed_results"][ms] = {}
        for ak in all_results[ms]:
            output["detailed_results"][ms][ak] = {}
            for mid in all_results[ms][ak]:
                output["detailed_results"][ms][ak][mid] = [
                    {
                        "json_valid": r["json_valid"],
                        "elapsed": r["elapsed"],
                        "m1": r["m1_json"],
                        "m2": r["m2_schema"],
                        "m3": r["m3_reasoning"],
                        "parsed": r["parsed"],
                        "error": r["error"],
                    }
                    for r in all_results[ms][ak][mid]
                ]

    fname = "llm_benchmark_results.json"
    with open(fname, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n[SAVED] {fname}")

    # ═══════════════════════════════════════════════
    # INSIGHTS SUMMARY
    # ═══════════════════════════════════════════════

    print(f"\n{'=' * 80}")
    print(f"  INSIGHTS SUMMARY")
    print(f"{'=' * 80}")

    # Rank models
    ranked = sorted(MODELS, key=lambda m: (
        model_metrics[m["short"]]["m3_reasoning_avg"] * 0.35 +
        model_metrics[m["short"]]["m1_json_rate"] * 0.25 +
        model_metrics[m["short"]]["m2_schema_rate"] * 0.15 +
        model_metrics[m["short"]]["m6_consistency"] * 0.15 +
        (1 - min(1, model_metrics[m["short"]]["m4_latency_avg"] / 30)) * 0.10
    ), reverse=True)

    print(f"\n  Weighted ranking (reasoning 35%, JSON 25%, schema 15%, consistency 15%, speed 10%):")
    for i, m in enumerate(ranked):
        ms = m["short"]
        mm = model_metrics[ms]
        composite = (mm["m3_reasoning_avg"] * 0.35 + mm["m1_json_rate"] * 0.25 +
                     mm["m2_schema_rate"] * 0.15 + mm["m6_consistency"] * 0.15 +
                     (1 - min(1, mm["m4_latency_avg"] / 30)) * 0.10)
        print(f"    #{i+1} {ms:<18} composite={composite:.3f} "
              f"(reasoning={mm['m3_reasoning_avg']:.2f} json={mm['m1_json_rate']:.2f} "
              f"latency={mm['m4_latency_avg']:.1f}s)")

    print(f"\n{'=' * 80}\n")

    return output


if __name__ == "__main__":
    run_benchmark()
