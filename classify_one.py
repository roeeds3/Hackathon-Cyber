import joblib, json
import numpy as np
from pathlib import Path

MODEL_PATH = Path("classifier model/cyber_rfc_model.joblib")
META_PATH  = Path("classifier model/cyber_rfc_meta.json")

rfc = joblib.load(MODEL_PATH)
with open(META_PATH, "r") as f:
    meta = json.load(f)

STATUS_MAP = meta["STATUS_MAP"]
FEATURE_ORDER = meta["FEATURE_ORDER"]

LABEL_MAP = {0: "OK", 1: "ERROR", 2: "CYBER_ATTACK"}


def compute_severity(record, p_attack=None):
    """
    Compute severity score for a charging station record.
    
    Args:
        record: dict with sensor fields (current, delta_current, voltage, delta_voltage,
                power_w, expected_load, temperature, status_str)
        p_attack: Optional attack probability from ML model (0-1)
    
    Returns:
        float: Severity score (0-100)
    """
    # record is dict with your JSON fields
    I = record["current"]
    dI = record["delta_current"]
    V = record["voltage"]
    dV = record["delta_voltage"]
    P = record["power_w"]
    L = record["expected_load"]
    T = record["temperature"]
    status = record["status_str"]

    # 1) power mismatch
    P_calc = I * V
    m_p = abs(P - P_calc) / max(P, 1)
    s_p = np.clip(m_p / 0.5, 0, 1)

    # 2) load mismatch
    m_l = abs(L - P) / max(L, 1)
    s_l = np.clip(m_l / 0.6, 0, 1)

    # 3) delta spikes
    s_c = np.clip(abs(dI) / 60, 0, 1)
    s_v = np.clip(abs(dV) / 150, 0, 1)

    # 4) temperature danger
    if T < 60:
        s_t = 0
    elif T <= 90:
        s_t = (T - 60) / 30
    else:
        s_t = 1

    # 5) status contradiction
    contradiction = False
    if status == "CHARGING" and (I < 1 or P < 200):
        contradiction = True
    if status in ["IDLE", "OFF"] and (V > 150 or L > 500):
        contradiction = True
    s_s = 1 if contradiction else 0

    # rule-based score
    sev_rule = 100 * np.clip(
        0.25*s_p + 0.25*s_l + 0.2*s_c + 0.15*s_v + 0.1*s_t + 0.05*s_s,
        0, 1
    )

    # if model probability given, do hybrid
    if p_attack is not None:
        sev = 100*p_attack + 20*s_t + 15*s_p
        sev = np.clip(sev, 0, 100)
        return float(sev)

    return float(sev_rule)


def preprocess_record(record_tuple):
    """
    Input tuple order (RAW event):
    (
      charger_id:int,
      current:float,
      delta_current:float,
      voltage:float,
      delta_voltage:float,
      power_w:float,
      expected_load:float/int,
      status_str:str,
      location:(x,y),
      temperature:float
    )

    Returns:
      X  -> numeric features matching training df after dropping charger id & location
      dropped -> dict with charger id & location for passing downstream
    """
    (charger_id, current, delta_current, voltage, delta_voltage,
     power_w, expected_load, status_str, location, temperature) = record_tuple

    # Save dropped fields separately
    dropped = {
        "charger id": charger_id,
        "location": location
    }

    # Map status same as training
    if status_str not in STATUS_MAP:
        raise ValueError(
            f"Unknown status '{status_str}'. Expected one of {list(STATUS_MAP.keys())}"
        )
    status_num = STATUS_MAP[status_str]

    # Build features AFTER dropping charger id & location
    features = {
        "current": current,
        "delta_current": delta_current,
        "voltage": voltage,
        "delta_voltage": delta_voltage,
        "power (W)": power_w,
        "expected_load": expected_load,
        "status": status_num,
        "temperature": temperature
    }

    # Order exactly like training
    X = np.array([[features[col] for col in FEATURE_ORDER]], dtype=float)
    return X, dropped


def predict_one(record_tuple):
    X, dropped = preprocess_record(record_tuple)

    pred_class = int(rfc.predict(X)[0])
    proba = rfc.predict_proba(X)[0]

    result = {
        "predicted_class": pred_class,
        "predicted_label": LABEL_MAP[pred_class],
        "probabilities": {
            "OK(0)": float(proba[0]),
            "ERROR(1)": float(proba[1]),
            "CYBER_ATTACK(2)": float(proba[2]),
        },
        "dropped_fields": dropped
    }
    return result


if __name__ == "__main__":
    # Example usage
    record = (
        999,                 # charger id  (dropped)
        15.2,                # current
        0.5,                 # delta_current
        480.1,               # voltage
        1.1,                 # delta_voltage
        7296.5,              # power (W)
        7300,                # expected_load
        "CHARGING",          # status
        (40.71, -74.01),     # location (dropped)
        35.5                 # temperature
    )

    res = predict_one(record)

    print("\n=== Prediction ===")
    print("Charger ID:", res["dropped_fields"]["charger id"])
    print("Location:", res["dropped_fields"]["location"])
    print("Predicted:", res["predicted_class"], f"({res['predicted_label']})")
    print("Probabilities:", {k: round(v,4) for k,v in res["probabilities"].items()})
