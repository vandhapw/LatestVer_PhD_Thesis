# ============================================================
# COMPARATIVE STUDY: Non-Agentic vs Agentic AI
# for Predictive Maintenance in Smart Manufacturing
# ============================================================
#
# AUDIT REWRITE:
#   The previous version of this file passed `failure_type`,
#   `machine_status_label`, and `anomaly_flag` directly into the
#   LLM agent prompts as if they were features. Those fields are
#   GROUND TRUTH in the digital-twin metadata. Letting the agent
#   read them and then evaluating its prediction against the same
#   labels is circular.
#
#   This rewrite:
#     1. Loads thresholds from the central config (derived from
#        a TRAINING split of machines).
#     2. Evaluates BOTH baselines on the held-out TEST split only.
#     3. Removes all ground-truth fields from the LLM agent input.
#     4. Removes ground-truth fields from the rule-based scoring.
#     5. Reports detection metrics with explicit disclosure that
#        this is a single snapshot, not cross-validation.
#
# Ground truth (used ONLY for evaluation, never as a feature):
#   - machine_status_label in (Warning, Fault)
#   - failure_type != Normal
#   - anomaly_flag == 1
#   - maintenance_required == 1
# ============================================================

import json
import datetime
import statistics
from collections import Counter
from pymongo import MongoClient
from ollama import Client

from config import (
    OLLAMA_API_KEY, MONGODB_URI, MONGO_DB, MONGO_COLLECTION,
    OLLAMA_MODEL, RUL_LOW, RUL_MODERATE, RISK_HIGH_AT, RISK_MEDIUM_AT,
    SENSOR_FEATURES, get_warn_crit_thresholds, load_sensor_bounds,
)
from fleet_relative_detector import (
    fleet_relative_score, derive_decision_threshold,
)
from sliding_window_detector import (
    SlidingWindowDetector, load_history_per_machine,
)


# ═══════════════════════════════════════════════════════════
# DATA LOADING — TEST SPLIT ONLY
# ═══════════════════════════════════════════════════════════

def _build_event_from_doc(d, mid):
    return {
        "machine_id":  mid,
        "timestamp":   str(d.get("timestamp", "")),
        "temperature": d.get("temperature", 0),
        "vibration":   d.get("vibration", 0),
        "humidity":    d.get("humidity", 0),
        "pressure":    d.get("pressure", 0),
        "energy_consumption": d.get("energy_consumption", 0),
        "energy":      d.get("energy_consumption", 0),
        "predicted_remaining_life": d.get("predicted_remaining_life", 200),
        "anchor_machine":  d.get("anchor_machine", 0),
        "graph_distance":  d.get("graph_distance", 0),
        "influence_score": d.get("influence_score", 0),
        # Ground-truth fields — kept under _gt_ prefix; NEVER passed to agents.
        "_gt_machine_status_label": d.get("machine_status_label", "Normal"),
        "_gt_failure_type":         d.get("failure_type", "Normal"),
        "_gt_anomaly_flag":         d.get("anomaly_flag", 0),
        "_gt_maintenance_required": d.get("maintenance_required", 0),
        "_gt_downtime_risk":        d.get("downtime_risk", 0),
    }


def _load_split(machine_ids, label):
    client = MongoClient(MONGODB_URI)
    col = client[MONGO_DB][MONGO_COLLECTION]
    cursor = col.find({"machine_id": {"$in": machine_ids}}).sort("timestamp", -1).limit(800)
    latest = {}
    for doc in cursor:
        if doc["machine_id"] not in latest:
            latest[doc["machine_id"]] = doc
    events = [_build_event_from_doc(latest[m], m) for m in sorted(latest)]
    client.close()
    print(f"[DATA] {label}: {len(events)} machines")
    return events


def load_test_data():
    """
    Load latest snapshots for both train and test splits per
    derive_sensor_bounds.py. Returns (train_events, test_events).
    The train split is used to derive a decision threshold for the
    fleet-relative detector; the test split is what every approach
    is evaluated on.
    """
    bounds = load_sensor_bounds()
    if bounds is None:
        raise RuntimeError("sensor_bounds_derived.json missing. Run derive_sensor_bounds.py first.")
    test_ids  = bounds.get("test_machine_ids", [])
    train_ids = bounds.get("train_machine_ids", [])
    if not test_ids or not train_ids:
        raise RuntimeError("Missing train/test_machine_ids in bounds file.")

    print(f"[DATA] Connected to {MONGO_DB}.{MONGO_COLLECTION}")
    train_events = _load_split(train_ids, "Train")
    test_events  = _load_split(test_ids,  "Test")
    print()
    return train_events, test_events


# ═══════════════════════════════════════════════════════════
# GROUND TRUTH (evaluation only)
# ═══════════════════════════════════════════════════════════

def build_ground_truth(events):
    gt = {}
    for evt in events:
        mid = evt["machine_id"]
        signals = [
            evt["_gt_machine_status_label"] in ("Warning", "Fault"),
            evt["_gt_failure_type"] != "Normal",
            evt["_gt_anomaly_flag"] == 1,
            evt["_gt_maintenance_required"] == 1,
        ]
        gt[mid] = {
            "needs_attention": any(signals),
            "signal_count":    sum(signals),
            "status":          evt["_gt_machine_status_label"],
            "failure_type":    evt["_gt_failure_type"],
        }
    return gt


# ═══════════════════════════════════════════════════════════
# APPROACH A — Non-Agentic (rule-based threshold)
# ═══════════════════════════════════════════════════════════

class NonAgenticPipeline:
    """
    Rule-based maintenance pipeline. Uses ONLY thresholds on raw
    sensors and RUL — no ground-truth fields are read.
    """
    def __init__(self):
        self.thresholds = get_warn_crit_thresholds()

    def analyze(self, events):
        results = {}
        for evt in events:
            mid = evt["machine_id"]
            score = 0.0
            flags = []
            for feature, thr in self.thresholds.items():
                # Some events use "energy" instead of "energy_consumption"
                val = evt.get(feature, evt.get("energy" if feature == "energy_consumption" else feature, 0))
                if not isinstance(val, (int, float)):
                    continue
                if val >= thr["crit"]:
                    score += 0.35
                    flags.append(f"{feature}=CRIT({val:.2f})")
                elif val >= thr["warn"]:
                    score += 0.15
                    flags.append(f"{feature}=WARN({val:.2f})")

            rul = evt.get("predicted_remaining_life", 200)
            if rul < RUL_LOW:
                score += 0.30
                flags.append("RUL=LOW")
            elif rul < RUL_MODERATE:
                score += 0.10
                flags.append("RUL=MODERATE")

            risk_score = min(score, 1.0)
            severity = "HIGH" if risk_score >= RISK_HIGH_AT else \
                       ("MEDIUM" if risk_score >= RISK_MEDIUM_AT else "LOW")

            if severity == "HIGH":
                action = "immediate_shutdown"
            elif severity == "MEDIUM":
                action = "schedule_maintenance"
            else:
                action = "monitor"

            results[mid] = {
                "risk_score":         round(risk_score, 4),
                "severity":           severity,
                "action":             action,
                "flags":              flags,
                "detects_attention":  severity != "LOW",
                # Capability flags — honest about what this approach lacks
                "has_root_cause":     False,
                "has_explainability": False,
                "has_correlation":    False,
                "has_consensus":      False,
                "has_scheduling_opt": False,
                "has_pattern_detect": False,
            }
        return results


# ═══════════════════════════════════════════════════════════
# APPROACH B — Agentic AI (no label leakage)
# ═══════════════════════════════════════════════════════════

def get_ollama_client():
    return Client(host="https://ollama.com",
                  headers={"Authorization": "Bearer " + OLLAMA_API_KEY})


def llm_call(client, model, system_prompt, user_msg):
    try:
        resp = client.chat(
            model=model,
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


def safe_sensor_view(evt):
    """Sensor-only view — NO ground-truth fields."""
    return {
        "machine_id":  evt["machine_id"],
        "temperature": evt["temperature"],
        "vibration":   evt["vibration"],
        "humidity":    evt["humidity"],
        "pressure":    evt["pressure"],
        "energy":      evt.get("energy", evt.get("energy_consumption", 0)),
        "rul":         evt["predicted_remaining_life"],
    }


def safe_topology_view(evt):
    return {
        "machine_id":      evt["machine_id"],
        "anchor_machine":  evt["anchor_machine"],
        "graph_distance":  evt["graph_distance"],
        "influence_score": evt["influence_score"],
    }


class AgenticPipeline:
    """
    Multi-agent LLM pipeline. AUDIT NOTE: previous version passed
    failure_type and machine_status_label into the LLM input — those
    are ground truth. Removed.
    """
    PROMPT_MONITORING = (
        "You are a machine monitoring agent for a smart factory. "
        "Analyze sensor data only and return JSON ONLY:\n"
        '{"risk_level":"LOW|MEDIUM|HIGH|CRITICAL",'
        '"anomaly_detected":true|false,'
        '"anomaly_type":"thermal|vibration|energy|combined|none",'
        '"confidence":0.0-1.0}\n'
        "Return valid JSON only."
    )

    PROMPT_DIAGNOSIS = (
        "You are a root-cause diagnosis agent. From sensor patterns and "
        "monitoring output, return JSON ONLY:\n"
        '{"probable_cause":"bearing_wear|overheating|electrical|lubrication|normal|unknown",'
        '"affected_component":"motor|bearing|pump|actuator|unknown",'
        '"urgency_hours":integer}\n'
        "Return valid JSON only."
    )

    PROMPT_PLANNING = (
        "You are a maintenance planning agent. Given diagnosis and risk, "
        "return JSON ONLY:\n"
        '{"action":"monitor|inspect|schedule_maintenance|immediate_shutdown",'
        '"priority":"P1|P2|P3|P4",'
        '"reason":"max 20 words",'
        '"notify_supervisor":true|false}\n'
        "Return valid JSON only."
    )

    PROMPT_EXPLAINABILITY = (
        "You are an Explainability Agent (XAI). Given sensor data, risk score, "
        "and thresholds, return JSON ONLY:\n"
        '{"top_contributing_feature":"temperature|vibration|humidity|energy|rul",'
        '"contribution_pct":float,'
        '"distance_to_next_severity":float,'
        '"what_if":"description of smallest change to escalate severity",'
        '"current_severity":"LOW|MEDIUM|HIGH",'
        '"nearest_boundary":"MEDIUM|HIGH",'
        '"confidence":0.0-1.0}\n'
        "Return valid JSON only."
    )

    PROMPT_CORRELATION = (
        "You are a Correlation Agent. You are given sensor data and graph "
        "topology features (influence_score, graph_distance, anchor_machine) "
        "for multiple machines. Identify clusters and propagation risk. "
        "Return JSON ONLY:\n"
        '{"clusters":[{"machines":["M-X"],"correlation_type":"thermal|electrical|vibration",'
        '"propagation_risk":0.0-1.0,"root_machine":"M-X"}],'
        '"total_clusters":int,'
        '"max_propagation_risk":float}\n'
        "Return valid JSON only."
    )

    # AUDIT NOTE: renamed from PROMPT_CONSENSUS. Same model self-checking
    # is NOT consensus.
    PROMPT_SELFREFLECTION = (
        "You are a Self-Reflection Agent reviewing the outputs of three "
        "sibling agents (Monitoring, Diagnosis, Planning) generated by the "
        "SAME underlying language model. This is a self-consistency check, "
        "NOT multi-model consensus. Return JSON ONLY:\n"
        '{"self_consistent":true|false,'
        '"recommended_action":"monitor|inspect|schedule_maintenance|immediate_shutdown",'
        '"agreement_ratio":float,'
        '"reliability_score":0.0-1.0}\n'
        "Return valid JSON only."
    )

    def __init__(self):
        self.client = get_ollama_client()
        self.model  = OLLAMA_MODEL
        self.thresholds = get_warn_crit_thresholds()

    def _call(self, prompt, msg):
        return llm_call(self.client, self.model, prompt, msg)

    def analyze(self, events):
        n = len(events)
        results = {}

        # ── Phase 1: rule-based reasoning (no label leakage) ──
        baseline = NonAgenticPipeline().analyze(events)

        # ── Phase 2: per-machine threshold proximity (sensor only) ──
        threshold_distances = {}
        for evt in events:
            mid = evt["machine_id"]
            distances = {}
            for feature, thr in self.thresholds.items():
                val = evt.get(feature, evt.get("energy" if feature == "energy_consumption" else feature, 0))
                if isinstance(val, (int, float)) and thr["warn"] != float("inf"):
                    distances[feature] = {
                        "value":           round(val, 2),
                        "warn_threshold":  thr["warn"],
                        "crit_threshold":  thr["crit"],
                        "dist_to_warn":    round(thr["warn"] - val, 2),
                        "pct_to_warn":     round(val / thr["warn"] * 100, 1) if thr["warn"] > 0 else 0,
                    }
            threshold_distances[mid] = distances

        # ── Phase 3: graph adjacency from T-GCN topology ──
        adjacency = {}
        for i, evt_i in enumerate(events):
            mid_i = evt_i["machine_id"]
            neighbors = []
            for j, evt_j in enumerate(events):
                if i == j:
                    continue
                mid_j = evt_j["machine_id"]
                same_anchor = evt_i["anchor_machine"] == evt_j["anchor_machine"]
                close_dist  = abs(evt_i["graph_distance"] - evt_j["graph_distance"]) < 2
                influence   = max(evt_i["influence_score"], evt_j["influence_score"])
                if same_anchor or (close_dist and influence > 0.3):
                    neighbors.append(mid_j)
            adjacency[mid_i] = neighbors

        # ── Phase 4: select machines for LLM analysis ──
        # AUDIT NOTE: previous version selected by failure_type/anomaly_flag,
        # which is label leakage. This version uses threshold proximity only.
        machines_for_agents = []
        for evt in events:
            mid = evt["machine_id"]
            close = any(d.get("pct_to_warn", 0) >= 80
                        for d in threshold_distances[mid].values())
            if close or baseline[mid]["severity"] != "LOW":
                machines_for_agents.append(mid)
        print(f"  [AGENTIC] {len(machines_for_agents)}/{n} machines selected (threshold proximity only)")

        # ── Phase 5: per-machine LLM agents ──
        agent_outputs = {}
        for idx, mid in enumerate(machines_for_agents):
            evt = next(e for e in events if e["machine_id"] == mid)
            base = baseline[mid]
            sensor_str = json.dumps(safe_sensor_view(evt), indent=2)
            risk_str = json.dumps({"risk_score": base["risk_score"],
                                   "severity": base["severity"],
                                   "flags": base["flags"]}, indent=2)
            monitoring = self._call(self.PROMPT_MONITORING,
                                    f"Sensor: {sensor_str}\nThreshold-derived risk: {risk_str}")
            diagnosis = self._call(self.PROMPT_DIAGNOSIS,
                                   f"Sensor: {sensor_str}\nMonitoring: {json.dumps(monitoring)}")
            planning = self._call(self.PROMPT_PLANNING,
                                  f"Diagnosis: {json.dumps(diagnosis)}\n"
                                  f"Risk: {base['risk_score']}  Severity: {base['severity']}\n"
                                  f"RUL: {evt['predicted_remaining_life']}")
            xai = self._call(self.PROMPT_EXPLAINABILITY,
                             f"Sensor: {sensor_str}\nRisk: {risk_str}\n"
                             f"Thresholds: {json.dumps(self.thresholds)}")
            agent_outputs[mid] = {
                "monitoring":      monitoring,
                "diagnosis":       diagnosis,
                "planning":        planning,
                "explainability":  xai,
            }
            print(f"    [{idx+1}/{len(machines_for_agents)}] M-{mid}  agents complete")

        # ── Phase 6: correlation agent (one call, fleet-wide) ──
        corr_input = []
        for evt in events:
            corr_input.append({**safe_sensor_view(evt), **safe_topology_view(evt)})
        correlation = self._call(self.PROMPT_CORRELATION,
                                 f"Machine data for {n} machines:\n{json.dumps(corr_input, indent=2)}")

        # ── Phase 7: self-reflection (per machine that had agents) ──
        self_reflection_results = {}
        for mid, ao in agent_outputs.items():
            self_reflection_results[mid] = self._call(
                self.PROMPT_SELFREFLECTION,
                f"Monitoring: {json.dumps(ao['monitoring'])}\n"
                f"Diagnosis: {json.dumps(ao['diagnosis'])}\n"
                f"Planning: {json.dumps(ao['planning'])}",
            )

        # ── Phase 8: build final per-machine results ──
        for evt in events:
            mid = evt["machine_id"]
            base = baseline[mid]
            has_agents = mid in agent_outputs
            if has_agents:
                ao = agent_outputs[mid]
                srf = self_reflection_results.get(mid, {})
                agent_action = ao["planning"].get("action", base["action"])
                agent_risk   = ao["monitoring"].get("risk_level", base["severity"])
                detects = (
                    agent_risk in ("MEDIUM", "HIGH", "CRITICAL")
                    or agent_action in ("inspect", "schedule_maintenance", "immediate_shutdown")
                )
                results[mid] = {
                    "risk_score":          base["risk_score"],
                    "severity_rule_based": base["severity"],
                    "severity_agentic":    agent_risk,
                    "action":              agent_action,
                    "detects_attention":   detects,
                    "has_root_cause":      ao["diagnosis"].get("probable_cause", "unknown") != "unknown",
                    "has_explainability":  ao["explainability"].get("distance_to_next_severity") is not None,
                    "has_correlation":     True,
                    "has_self_reflection": True,
                    "has_scheduling_opt":  False,  # honest: no separate scheduling agent here
                    "has_pattern_detect":  False,  # honest: no pattern agent here
                    "diagnosis":           ao["diagnosis"],
                    "explainability":      ao["explainability"],
                    "self_reflection":     srf,
                    "correlated_machines": adjacency.get(mid, []),
                    "threshold_distances": threshold_distances[mid],
                }
            else:
                results[mid] = {
                    "risk_score":         base["risk_score"],
                    "severity_rule_based": base["severity"],
                    "severity_agentic":    base["severity"],
                    "action":              base["action"],
                    "detects_attention":   base["detects_attention"],
                    "has_root_cause":      False,
                    "has_explainability":  False,
                    "has_correlation":     True,   # correlation runs fleet-wide
                    "has_self_reflection": False,
                    "has_scheduling_opt":  False,
                    "has_pattern_detect":  False,
                    "correlated_machines": adjacency.get(mid, []),
                    "threshold_distances": threshold_distances[mid],
                }

        for mid in results:
            results[mid]["correlation_summary"] = correlation
        return results


# ═══════════════════════════════════════════════════════════
# APPROACH C — Fleet-Relative (no labels, no LLM)
# ═══════════════════════════════════════════════════════════
#
# Motivation: see fleet_relative_detector.py header. Briefly:
# the digital-twin Kafka stream we evaluate on emits a STATIC sensor
# value per machine (per-machine stdev ≈ 0 over 100 readings), so
# per-machine temporal methods cannot detect anything. The only
# remaining signal is cross-machine: which machines are unusual
# RELATIVE TO their peers in the current snapshot.
#
# References (synthesized in chat history 2026-05-03):
#   - Hendrickx et al., MSSP 2020 (arXiv:1912.12941) — fleet-baseline
#   - Adams et al., Data-Centric Eng. 2023 — hierarchical fleet model
#   - Olsson et al., EAAI 2022 — UL-CAD / SL-CAD conformal split
#   - Liu, Ting, Zhou, ICDM 2008 — Isolation Forest
# ═══════════════════════════════════════════════════════════
class FleetRelativePipeline:
    """
    Cross-machine population scoring. The decision threshold is
    derived from the TRAIN snapshot at quantile=0.5 and applied
    to the TEST snapshot — no peeking.
    """
    def __init__(self):
        self.score_threshold = None

    def fit(self, train_events):
        self.score_threshold = derive_decision_threshold(train_events, target_quantile=0.5)
        print(f"  [Fleet-Relative] derived score_threshold from TRAIN fleet: "
              f"{self.score_threshold:.4f}")

    def analyze(self, events):
        if self.score_threshold is None:
            raise RuntimeError("Call fit(train_events) first.")
        rows = fleet_relative_score(events, score_threshold=self.score_threshold)
        results = {}
        for r in rows:
            mid = r["machine_id"]
            detects = bool(r["is_attention_needed"])
            results[mid] = {
                "risk_score":          r["composite_score"],
                "severity_rule_based": "—",
                "severity_agentic":    "—",
                "action":              "schedule_maintenance" if detects else "monitor",
                "detects_attention":   detects,
                "has_root_cause":      False,
                "has_explainability":  False,
                "has_correlation":     True,   # composite over all peers
                "has_self_reflection": False,
                "has_scheduling_opt":  False,
                "has_pattern_detect":  False,
                "fleet_metrics": {
                    "mahalanobis_percentile": r["mahalanobis_percentile"],
                    "max_abs_peer_z":         r["max_abs_peer_z"],
                    "isolation_score":        r["isolation_score"],
                },
                "decision_cutoff":     r["_decision_cutoff"],
            }
        return results


# ═══════════════════════════════════════════════════════════
# APPROACH D — Per-machine Sliding Window (z-score on own history)
# ═══════════════════════════════════════════════════════════
# Reuses the MultiTierBuffer concept from kafka_realtime_consumer.py
# but operates batch-style — pulls latest WINDOW snapshots per machine
# from MongoDB and runs the same z-score logic the streaming consumer
# would. Per-machine TEMPORAL scoring (vs Approach C cross-machine).
# Fit decision threshold from train fleet at q=0.5 (no peeking).
# ═══════════════════════════════════════════════════════════
class SlidingWindowPipeline:
    def __init__(self, window=12, z_threshold=2.5):
        self.window      = window
        self.z_threshold = z_threshold
        self.detector    = None

    def fit(self, train_machine_ids):
        train_hist = load_history_per_machine(train_machine_ids, window=self.window)
        self.detector = SlidingWindowDetector(window=self.window,
                                              z_threshold=self.z_threshold).fit(train_hist)
        print(f"  [Sliding-Window] derived score_threshold from TRAIN fleet: "
              f"{self.detector.score_threshold:.4f}")

    def analyze(self, test_machine_ids):
        if self.detector is None:
            raise RuntimeError("Call fit(train_ids) first.")
        test_hist = load_history_per_machine(test_machine_ids, window=self.window)
        rows = self.detector.analyze(test_hist)
        results = {}
        for mid, r in rows.items():
            detects = bool(r["is_attention_needed"])
            results[mid] = {
                "risk_score":          r["composite_score"],
                "severity_rule_based": "—",
                "severity_agentic":    "—",
                "action":              "schedule_maintenance" if detects else "monitor",
                "detects_attention":   detects,
                "has_root_cause":      False,
                "has_explainability":  False,
                "has_correlation":     False,
                "has_self_reflection": False,
                "has_scheduling_opt":  False,
                "has_pattern_detect":  True,   # temporal pattern via z-score
                "sliding_window_metrics": {
                    "max_abs_z":           r["max_abs_z"],
                    "max_z_feature":       r.get("max_z_feature"),
                    "cumulative_excess":   r["cumulative_excess_ratio"],
                    "n_history":           r["n_history"],
                },
                "decision_cutoff":     r["_decision_cutoff"],
            }
        return results


# ═══════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════

def compute_detection_metrics(approach_results, ground_truth):
    tp = fp = fn = tn = 0
    for mid, gt in ground_truth.items():
        if mid not in approach_results:
            continue
        pred = approach_results[mid]["detects_attention"]
        actual = gt["needs_attention"]
        if pred and actual:    tp += 1
        elif pred and not actual: fp += 1
        elif not pred and actual: fn += 1
        else: tn += 1
    total = max(tp + fp + fn + tn, 1)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": round(prec, 4),
        "recall":    round(rec, 4),
        "f1_score":  round(f1, 4),
        "accuracy":  round((tp + tn) / total, 4),
        "n":         total,
    }


def compute_capability_metrics(approach_results):
    caps = {
        "root_cause_analysis":      0,
        "explainability":           0,
        "cross_machine_correlation": 0,
        "self_reflection":          0,
        "scheduling_optimization":  0,
        "pattern_detection":        0,
    }
    total = max(len(approach_results), 1)
    for r in approach_results.values():
        if r.get("has_root_cause"):     caps["root_cause_analysis"]      += 1
        if r.get("has_explainability"): caps["explainability"]           += 1
        if r.get("has_correlation"):    caps["cross_machine_correlation"] += 1
        if r.get("has_self_reflection") or r.get("has_consensus"):
            caps["self_reflection"]    += 1
        if r.get("has_scheduling_opt"): caps["scheduling_optimization"]  += 1
        if r.get("has_pattern_detect"): caps["pattern_detection"]        += 1
    pct = {k: round(v / total * 100, 1) for k, v in caps.items()}
    return {"counts": caps, "percentages": pct, "total_machines": total}


# ═══════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════

def print_report(events, gt, results_a, results_b, results_c=None, results_d=None):
    n = len(events)
    gt_attention = sum(1 for g in gt.values() if g["needs_attention"])

    print("\n" + "="*100)
    print(" COMPARATIVE STUDY (post-audit + fleet-relative + sliding-window): 4 approaches")
    print("="*100)
    print("\n  AUDIT DISCLOSURE:")
    print("  - Bounds derived from training split (machine_id even).")
    print("  - This report computed on TEST split (machine_id odd) only.")
    print("  - Ground-truth fields (failure_type, anomaly_flag, etc.) are")
    print("    NEVER passed as features to any approach.")
    print("  - Approach C (Fleet-Relative) and D (Sliding-Window) decision")
    print("    thresholds are derived from train fleet at quantile=0.5 and")
    print("    applied to test (no peeking).")
    print("  - Results are from a single snapshot; not cross-validated.")

    print(f"\n  Test machines:        {n}")
    print(f"  Need attention (GT):  {gt_attention} ({round(gt_attention/n*100,1)}%)")
    print(f"  Status distribution:  {dict(Counter(e['_gt_machine_status_label'] for e in events))}")

    approaches = [
        ("Rule-Based",       results_a),
        ("Agentic AI",       results_b),
        ("Fleet-Relative",   results_c),
        ("Sliding-Window",   results_d),
    ]
    metrics_by_name = {
        name: compute_detection_metrics(res, gt) if res else None
        for name, res in approaches
    }

    name_w = 16
    cell_w = 13
    header = "  " + " ".join(f"{name:>{cell_w}}" for name, _ in approaches)
    print(f"\n  {'Metric':<{name_w}}{header}")
    print("  " + "─" * (name_w + (cell_w + 1) * len(approaches)))
    for metric in ["TP", "FP", "FN", "TN", "precision", "recall", "f1_score", "accuracy"]:
        line = f"  {metric:<{name_w}}"
        for name, _ in approaches:
            m = metrics_by_name[name]
            if m is None:
                line += f"{'—':>{cell_w}} "
            elif isinstance(m[metric], float):
                line += f"{m[metric]:>{cell_w}.4f} "
            else:
                line += f"{m[metric]:>{cell_w}} "
        print(line)

    caps_by_name = {
        name: compute_capability_metrics(res) if res else None
        for name, res in approaches
    }
    print(f"\n  Capability coverage (% of test machines):")
    print(f"  {'Capability':<32}" + " ".join(f"{name:>{cell_w}}" for name, _ in approaches))
    print("  " + "─" * (32 + (cell_w + 1) * len(approaches)))
    if caps_by_name[approaches[0][0]]:
        for cap in caps_by_name[approaches[0][0]]["percentages"]:
            line = f"  {cap:<32}"
            for name, _ in approaches:
                c = caps_by_name[name]
                if c is None:
                    line += f"{'—':>{cell_w}} "
                else:
                    line += f"{c['percentages'][cap]:>{cell_w-1}.1f}% "
            print(line)

    print(f"\n{'='*100}\n")
    return {
        f"metrics_{name.lower().replace('-','_').replace(' ','_')}": metrics_by_name[name]
        for name, _ in approaches if metrics_by_name[name]
    }


def main(skip_agentic=False):
    """If skip_agentic=True, skip Approach B (the slow LLM pipeline) and
    report Approach A vs C only — useful for fast iteration on the
    fleet-relative detector."""
    print("\n" + "="*80)
    print(" LOADING TEST-SPLIT DATA FROM MONGODB")
    print("="*80)

    train_events, test_events = load_test_data()
    if not test_events:
        print("[ERROR] No test data loaded.")
        return

    gt = build_ground_truth(test_events)

    print("="*80)
    print(" APPROACH A: Non-Agentic (rule-based, threshold from training split)")
    print("="*80)
    results_a = NonAgenticPipeline().analyze(test_events)
    print(f"  Completed: {len(results_a)} machines analyzed\n")

    if not skip_agentic:
        print("="*80)
        print(" APPROACH B: Agentic AI (sensor-only inputs, no label leakage)")
        print("="*80)
        results_b = AgenticPipeline().analyze(test_events)
        print(f"  Completed: {len(results_b)} machines analyzed\n")
    else:
        print("[SKIP] Approach B (Agentic AI) skipped per request.")
        # Build a no-op result mirroring rule-based for downstream code.
        results_b = {mid: dict(r) for mid, r in results_a.items()}

    print("="*80)
    print(" APPROACH C: Fleet-Relative (Mahalanobis + peer-z + isolation; no LLM, no labels)")
    print("="*80)
    pipeline_c = FleetRelativePipeline()
    pipeline_c.fit(train_events)
    results_c = pipeline_c.analyze(test_events)
    print(f"  Completed: {len(results_c)} machines analyzed\n")

    print("="*80)
    print(" APPROACH D: Sliding-Window (per-machine z-score on own history; no LLM, no labels)")
    print("="*80)
    bounds = load_sensor_bounds()
    train_ids = bounds["train_machine_ids"]
    test_ids  = bounds["test_machine_ids"]
    pipeline_d = SlidingWindowPipeline(window=12, z_threshold=2.5)
    pipeline_d.fit(train_ids)
    results_d = pipeline_d.analyze(test_ids)
    print(f"  Completed: {len(results_d)} machines analyzed\n")

    comparison = print_report(test_events, gt, results_a, results_b, results_c, results_d)

    output = {
        "_audit_disclosure": (
            "All approaches evaluated on held-out test split (machine_id odd). "
            "Bounds derived from training split. No ground-truth fields passed "
            "to LLM agents or any feature pipeline. Approach C and D decision "
            "thresholds derived from train fleet (no peeking). Single-snapshot "
            "results, not cross-validated."
        ),
        "timestamp":   str(datetime.datetime.now()),
        "n_machines":  len(test_events),
        "skip_agentic": skip_agentic,
        "comparison":  comparison,
        "raw_events":  [{k: v for k, v in e.items() if not k.startswith("_gt_")} for e in test_events[:5]],
    }
    out_path = "comparison_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"[SAVED] {out_path}")
    return comparison


if __name__ == "__main__":
    import sys
    main(skip_agentic="--skip-agentic" in sys.argv)
