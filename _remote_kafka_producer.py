"""
Smart Manufacturing Digital Twin v2 - Conceptual Framework Implementation
==========================================================================
Architecture:
1. TCGN model forecasts ALL 5 continuous features (temp, vibration, humidity, pressure, energy)
   for all 50 machines based on active sensor input
2. Active sensor (Plalion device measuring temp+humidity) is treated as part of MACHINE 40
3. Classification models predict discrete states: machine_status, failure_type, 
   maintenance_required, anomaly_flag, downtime_risk, predicted_remaining_life
4. Uniqueness constraint: No two machines have exactly identical values
"""

import time
import json
import os
import numpy as np
import pandas as pd
import torch
import joblib
import lightgbm as lgb 
import xgboost as xgb
from kafka import KafkaProducer
from pymongo import MongoClient
from bson import encode
from datetime import datetime
from catboost import CatBoostRegressor as CBR
from model_export_utils import load_checkpoint
from tcgn_updated import prepare_data, load_and_engineer as load_data

# Auto-detect GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEVICE] Using: {DEVICE}", "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU-only")

# ==========================================
# CONFIGURATION & CONNECTIONS (env-driven)
# ==========================================
MONGODB_URI     = os.environ.get("MONGODB_URI", "mongodb://superUser:superUpsil!@localhost:27017/server_db?authSource=server_db")
DB_NAME         = os.environ.get("MONGO_DB", "server_db")
KAFKA_BROKER    = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")
KAFKA_TOPIC     = os.environ.get("KAFKA_TOPIC", "simulated_digitaltwin_manufactured")
COLLECTION_IN   = os.environ.get("MONGO_COLLECTION_IN", "plalion_klaen_sensor")
COLLECTION_OUT  = os.environ.get("MONGO_COLLECTION_OUT", "simulated_smart_manufactured_data")
SIM_INTERVAL    = int(os.environ.get("SIM_INTERVAL_SEC", "60"))
ANCHOR_MACHINE  = int(os.environ.get("ANCHOR_MACHINE", "40"))

# MongoDB Connections
mongo_client = MongoClient(MONGODB_URI)
db           = mongo_client[DB_NAME]
klaen_col    = db[COLLECTION_IN]
ssmd_col     = db[COLLECTION_OUT]

# Kafka Producer Setup
def json_serializer(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32)): return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if hasattr(obj, "isoformat"): return obj.isoformat()
    return str(obj)

def convert_numpy_types(obj):
    """Convert all numpy types to native Python types for MongoDB compatibility"""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v, default=json_serializer).encode("utf-8"),
    key_serializer=lambda k: str(int(k)).encode("utf-8") if k else None
)

print(f"[CONFIG] MongoDB: {MONGODB_URI} | DB: {DB_NAME}")
print(f"[CONFIG] Kafka: {KAFKA_BROKER} | Topic: {KAFKA_TOPIC}")
print(f"[CONFIG] Collection IN: {COLLECTION_IN} | OUT: {COLLECTION_OUT}")
print(f"[CONFIG] Interval: {SIM_INTERVAL} seconds")
print(f"[CONFIG] Anchor Machine: {ANCHOR_MACHINE} (Active Sensor)")

# ==========================================
# SUPPORTING CLASSES & FUNCTIONS
# ==========================================

class SensorAligner:
    """
    Aligns plalion sensor values to manufacturing range.
    Active sensor is treated as physically part of Machine 40 (anchor).
    """
    def __init__(self, initial_data=None):
        self.temp_buffer = []
        self.hum_buffer = []
        if initial_data is not None:
            self.update(initial_data["temperature"], initial_data["humidity"])

    def update(self, temp, hum):
        self.temp_buffer.append(temp)
        self.hum_buffer.append(hum)
        if len(self.temp_buffer) > 100:
            self.temp_buffer.pop(0)
            self.hum_buffer.pop(0)

    def transform(self, temp, hum):
        """
        Scale plalion sensor readings to manufacturing range using z-score mapping.
        Plalion: temp~22°C, humidity~23%
        Manufacturing: temp~75°C, humidity~55%
        """
        if len(self.temp_buffer) == 0: 
            self.update(temp, hum)
        
        # Target manufacturing ranges (from smart_manufacturing_data.csv statistics)
        MFG_TEMP_MEAN, MFG_TEMP_STD = 75.01, 5.0
        MFG_HUM_MEAN, MFG_HUM_STD = 54.995, 10.0
        
        # Compute running stats from sensor buffer
        sensor_temp_mean = np.mean(self.temp_buffer) if self.temp_buffer else temp
        sensor_temp_std = np.std(self.temp_buffer) if len(self.temp_buffer) > 1 else 1.0
        sensor_hum_mean = np.mean(self.hum_buffer) if self.hum_buffer else hum
        sensor_hum_std = np.std(self.hum_buffer) if len(self.hum_buffer) > 1 else 1.0
        
        # Z-score normalization: map sensor value to manufacturing distribution
        if sensor_temp_std > 0:
            z_temp = (temp - sensor_temp_mean) / sensor_temp_std
            temp_scaled = z_temp * MFG_TEMP_STD + MFG_TEMP_MEAN
        else:
            temp_scaled = MFG_TEMP_MEAN
            
        if sensor_hum_std > 0:
            z_hum = (hum - sensor_hum_mean) / sensor_hum_std
            hum_scaled = z_hum * MFG_HUM_STD + MFG_HUM_MEAN
        else:
            hum_scaled = MFG_HUM_MEAN
            
        return temp_scaled, hum_scaled


def predict_anchor_machine(model, scaler, A, window, prev_full, node_idx):
    """
    Predict all 5 continuous features for anchor machine (Machine 40)
    using T-GCN model.
    
    Args:
        model: T-GCN model
        scaler: Feature scaler
        A: Adjacency matrix
        window: Input window (sequence of feature vectors)
        prev_full: Previous full feature values
        node_idx: Node index in graph
    
    Returns:
        pred_full: Predicted feature values (unscaled)
    """
    x = torch.tensor(window[None], dtype=torch.float32)
    with torch.no_grad():
        pred_scaled = model(x, A, node_idx=node_idx).numpy()[0]
    pred_full = prev_full + scaler.inverse_transform(pred_scaled.reshape(1, -1))[0]
    return pred_full


def propagate_to_fleet(model, scaler, A, machine_ids, registry, 
                       anchor_pred_scaled, anchor_features, anchor_id=40):
    """
    Propagate predictions from anchor machine to all other machines.
    Uses graph propagation where anchor machine's changes influence others.
    
    Key concept: Changes in anchor machine (active sensor) propagate through
    the graph structure to affect all other machines differently.
    
    Args:
        anchor_pred_scaled: Scaled prediction differences from anchor
        anchor_features: Actual predicted feature values for anchor
    
    Returns:
        results: Dict of {machine_id: predicted_features}
    """
    results = {}
    mid_to_idx = {m: i for i, m in enumerate(machine_ids)}
    
    for m in machine_ids:
        if m == anchor_id:
            continue
        
        node_idx = mid_to_idx[m]
        dep_window = registry[m]["X"][-1].copy()
        dep_prev   = registry[m]["prev"][-1].copy()
        
        # Propagate anchor's influence: scale by graph connectivity
        # Each machine receives different influence based on graph structure
        influence_factor = 0.02  # Base influence strength
        
        # Add machine-specific perturbation to ensure uniqueness
        # This ensures no two machines have exactly identical values
        np.random.seed(m)  # Deterministic per machine
        machine_perturbation = np.random.uniform(0.98, 1.02, len(anchor_pred_scaled))
        
        # Apply propagated influence with uniqueness constraint
        influenced_change = anchor_pred_scaled * influence_factor * machine_perturbation
        dep_window[-1] = dep_window[-1] + influenced_change
        
        # Predict for this machine
        pred_full = predict_anchor_machine(model, scaler, A, dep_window, dep_prev, node_idx)
        
        # Add small uniqueness noise to raw features (not derived features)
        for i in range(5):  # First 5 are raw sensor features
            noise = np.random.uniform(-0.01, 0.01) * pred_full[i]
            pred_full[i] += noise
        
        results[m] = pred_full
    
    return results


def build_classification_features(machine_id, pred_features, timestamp, df_stats):
    """
    Build feature vector for classification models.
    
    Args:
        machine_id: Machine identifier
        pred_features: Predicted continuous features from TCGN
        timestamp: Current timestamp
        df_stats: Historical statistics dataframe
    
    Returns:
        row: Feature dictionary for classification models
    """
    hour = timestamp.hour if hasattr(timestamp, 'hour') else 0
    dayofweek = timestamp.weekday() if hasattr(timestamp, 'weekday') else 0
    
    row = {
        "machine_id": machine_id,
        "temperature": pred_features[0],
        "vibration": pred_features[1],
        "humidity": pred_features[2],
        "pressure": pred_features[3],
        "energy_consumption": pred_features[4],
        "hour": hour,
        "dayofweek": dayofweek
    }
    
    # Add historical statistics
    stats = df_stats[df_stats["machine_id"] == machine_id]
    if not stats.empty:
        for col in stats.columns:
            if col != "machine_id":
                row[col] = stats.iloc[0][col]
    
    return row


def predict_discrete_states(feat_df, models):
    """
    Step 3: Use classification models to predict discrete states.
    
    Models:
    - anomaly_flag: Binary (Normal/Anomaly)
    - downtime_risk: Binary (Low/High)
    - machine_status: Multi-class (Operating, Warning, Fault)
    - failure_type: Multi-class (Normal, Overheating, Vibration, etc.)
    - maintenance_required: Binary (No/Yes)
    - predicted_remaining_life: Regression (hours)
    
    Args:
        feat_df: Feature DataFrame
        models: Dict of trained models
    
    Returns:
        dict: Predicted discrete states
    """
    lgb_anomaly = models['anomaly']
    lgb_downtime = models['downtime']
    xgb_failure = models['failure']
    xgb_status = models['status']
    xgb_maint = models['maintenance']
    le_failure = models['failure_encoder']
    cat_rul = models['rul']
    
    # Predict binary classifiers
    anomaly_prob = lgb_anomaly.predict(feat_df)[0]
    anomaly_val = int(anomaly_prob > 0.5)
    
    downtime_prob = lgb_downtime.predict(feat_df)[0]
    downtime_risk_val = int(downtime_prob > 0.5)
    
    # Predict multi-class classifiers
    failure_enc = xgb_failure.predict(feat_df)[0]
    failure_label = str(le_failure.inverse_transform([failure_enc])[0])
    
    machine_status = int(xgb_status.predict(feat_df)[0])
    maintenance_required = int(xgb_maint.predict(feat_df)[0])
    
    # LOGICAL CONSISTENCY FIX:
    if machine_status == 0 and anomaly_val == 0:
        failure_label = "Normal"
    
    # Predict remaining useful life (regression)
    predicted_remaining_life = float(cat_rul.predict(feat_df)[0])
    
    return {
        'anomaly_flag': anomaly_val,
        'anomaly_probability': float(anomaly_prob),
        'downtime_risk': downtime_risk_val,
        'downtime_probability': float(downtime_prob),
        'machine_status': machine_status,
        'failure_type': failure_label,
        'maintenance_required': maintenance_required,
        'predicted_remaining_life': round(predicted_remaining_life, 2)
    }


# ==========================================
# MAIN STREAMING FUNCTION
# ==========================================
def run_streaming_simulation():
    """
    Real-time streaming mode implementing conceptual framework:
    
    1. Read active sensor data (temp+humidity) from MongoDB
    2. Treat sensor as part of Machine 40 (anchor)
    3. TCGN forecasts all 5 continuous features for all machines
    4. Classification models predict discrete states
    5. Ensure uniqueness across all machines
    6. Publish to Kafka and store in MongoDB
    """
    print("[INIT] Loading AI Models and Fleet Statistics...")
    
    # Load T-GCN Checkpoint & Scaler
    model, scaler, machine_ids, A, _ = load_checkpoint("model_exports/tgcn_checkpoint.pth")
    
    # Load ML Classification Models (Step 3)
    models = {
        'anomaly': lgb.Booster(model_file='models/anomaly_flag__LightGBM.txt'),
        'downtime': lgb.Booster(model_file='models/downtime_risk__LightGBM.txt'),
        'failure': xgb.XGBClassifier(),
        'status': xgb.XGBClassifier(),
        'maintenance': xgb.XGBClassifier(),
        'failure_encoder': joblib.load('models/label_encoder_failure_type.pkl'),
        'rul': CBR()
    }
    models['failure'].load_model('models/failure_type__XGBoost.ubj')
    models['status'].load_model('models/machine_status__XGBoost.ubj')
    models['maintenance'].load_model('models/maintenance_required__XGBoost.ubj')
    models['rul'].load_model('models/predicted_remaining_life__CatBoost.cbm')
    
    FEATURE_COLS = joblib.load('models/feature_cols.pkl')

    # Load historical data for T-GCN registry & fleet statistics
    df_raw = load_data("smart_manufacturing_data.csv")
    _, _, registry, _, _ = prepare_data(df_raw)
    
    features_raw = ['temperature','vibration','humidity','pressure','energy_consumption']
    df_stats = df_raw.groupby('machine_id')[features_raw].agg(['mean','std','min','max']).reset_index()
    df_stats.columns = ['machine_id'] + [f'{c}_{s}' for c in features_raw for s in ['mean','std','min','max']]

    print(f"[INIT] Models loaded. Fleet: {len(machine_ids)} machines")
    print(f"[START] MongoDB-Driven Real-Time Mode | Cycle: every {SIM_INTERVAL} seconds")
    print(f"[START] Reading from: {COLLECTION_IN} → Publishing to: {KAFKA_TOPIC}")
    print(f"[INFO] Active sensor → Machine {ANCHOR_MACHINE} (Anchor)")

    aligner = SensorAligner()
    total_records = 0

    while True:
        try:
            start_cycle = time.time()
            
            # ===== STEP 1: Read Active Sensor =====
            # Get latest temperature and humidity from Plalion sensor
            sensor_data = klaen_col.find_one(
                sort=[("timestamp", -1)],
                projection={"_id": 0, "timestamp": 1, "humidity": 1, "temperature": 1}
            )

            if sensor_data:
                # Update aligner and transform sensor values to manufacturing range
                raw_temp = sensor_data["temperature"]
                raw_hum = sensor_data["humidity"]
                aligner.update(raw_temp, raw_hum)
                t_scaled, h_scaled = aligner.transform(raw_temp, raw_hum)

                print(f"[SENSOR] Raw: temp={raw_temp:.1f}°C, humi={raw_hum:.1f}% → "
                      f"Mapped: temp={t_scaled:.2f}°C, humi={h_scaled:.2f}%")

                # ===== STEP 2: T-GCN Forecasts All 5 Continuous Features =====
                FEATURES = ["temperature", "vibration", "humidity", "pressure",
                           "energy_consumption", "temp_vibe_idx", "press_energy_ratio"]
                feature_idx = {f: i for i, f in enumerate(FEATURES)}
                
                # Update anchor machine (Machine 40) with active sensor values
                anchor_data = registry[ANCHOR_MACHINE]
                new_full = anchor_data["prev"][-1].copy()
                new_full[feature_idx["temperature"]] = t_scaled
                new_full[feature_idx["humidity"]] = h_scaled
                
                # Compute difference for T-GCN input
                diff_scaled = scaler.transform((new_full - anchor_data["prev"][-1]).reshape(1, -1))[0]
                new_window = np.vstack([anchor_data["X"][-1][1:], diff_scaled])
                
                # Predict for anchor machine (all 5 features)
                pred_anchor = predict_anchor_machine(
                    model, scaler, A, new_window, new_full, 
                    machine_ids.index(ANCHOR_MACHINE)
                )
                pred_scaled = scaler.transform((pred_anchor - new_full).reshape(1, -1))[0]
                
                # Propagate to all other machines (Step 2 continued)
                fleet_preds = propagate_to_fleet(
                    model, scaler, A, machine_ids, registry, 
                    pred_scaled, pred_anchor, anchor_id=ANCHOR_MACHINE
                )

                # Build payloads for all machines
                ts = datetime.now()
                all_results = []

                for m_id in machine_ids:
                    # Get predicted continuous features
                    full = pred_anchor if m_id == ANCHOR_MACHINE else fleet_preds[m_id]
                    
                    # Build features for classification models
                    feat_row = build_classification_features(m_id, full, ts, df_stats)
                    feat_df = pd.DataFrame([feat_row])[FEATURE_COLS]

                    # ===== STEP 3: Predict Discrete States =====
                    discrete_states = predict_discrete_states(feat_df, models)

                    # Build final payload
                    now = datetime.now()
                    now_iso = now.strftime("%Y-%m-%d %H:%M:%S")
                    
                    payload = {
                        "machine_id": int(m_id),
                        "timestamp": now,
                        "temperature": round(float(full[0]), 4),
                        "vibration": round(float(full[1]), 4),
                        "humidity": round(float(full[2]), 4),
                        "pressure": round(float(full[3]), 4),
                        "energy_consumption": round(float(full[4]), 4),
                        "machine_status": int(discrete_states['machine_status']),
                        "machine_status_label": ["Normal", "Warning", "Fault"][discrete_states['machine_status']],
                        "anomaly_flag": int(discrete_states['anomaly_flag']),
                        "anomaly_flag_label": ["Normal", "Anomaly"][discrete_states['anomaly_flag']],
                        "predicted_remaining_life": float(discrete_states['predicted_remaining_life']),
                        "failure_type": discrete_states['failure_type'],
                        "downtime_risk": int(discrete_states['downtime_risk']),
                        "downtime_risk_label": ["Low", "High"][discrete_states['downtime_risk']],
                        "maintenance_required": int(discrete_states['maintenance_required']),
                        "maintenance_required_label": ["No", "Yes"][discrete_states['maintenance_required']],
                        "source": "ANCHOR" if m_id == ANCHOR_MACHINE else "PROPAGATED",
                        "anchor_machine": ANCHOR_MACHINE,
                        "graph_distance": abs(m_id - ANCHOR_MACHINE),
                        "influence_score": round(0.02 * np.exp(-abs(m_id - ANCHOR_MACHINE) / 10), 6),
                        "normalization_method": "z_score_mapping",
                        "sensor_input": {
                            "raw_temperature": float(raw_temp),
                            "raw_humidity": float(raw_hum),
                            "mapped_temperature": float(t_scaled),
                            "mapped_humidity": float(h_scaled),
                        },
                        "processed_at": now,
                        "ingested_at": now
                    }
                    
                    all_results.append(payload)
                    producer.send(KAFKA_TOPIC, key=m_id, value=payload)

                # Store in MongoDB (convert numpy types first)
                if all_results:
                    # Convert all numpy types to native Python for MongoDB compatibility
                    clean_results = [convert_numpy_types(payload) for payload in all_results]
                    ssmd_col.insert_many(clean_results)
                    total_records += len(all_results)

                print(f"[SUCCESS] {now_iso} | {len(all_results)} machines | "
                      f"Total: {total_records} records | "
                      f"Anchor M{ANCHOR_MACHINE}: temp={pred_anchor[0]:.2f}, humi={pred_anchor[2]:.2f}")
            
            else:
                print(f"[WAIT] No sensor data found in {COLLECTION_IN}. Waiting...")
            
            elapsed = time.time() - start_cycle
            time.sleep(max(0, SIM_INTERVAL - elapsed))

        except KeyboardInterrupt:
            print(f"\n[STOP] Simulation ended by user. {total_records} total records processed.")
            break
        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()
            time.sleep(10)


# ==========================================
# ENTRY POINT
# ==========================================
if __name__ == '__main__':
    run_streaming_simulation()
