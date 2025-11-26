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
