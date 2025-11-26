import time
import json
import random
from datetime import datetime


class ChargingSessionSimulator:
    def __init__(self, session_id):
        self.session_id = session_id
        self.status = "IDLE"  # מצבים: IDLE, CONNECTED, CHARGING, FINISHING
        self.battery_level = 20  # מתחילים מ-20% סוללה
        self.temp = 25.0  # טמפרטורה התחלתית (צלזיוס)

    def generate_sensor_data(self):
        """
        פונקציה זו מדמה קריאה רגעית מכל הסנסורים בעמדה.
        הערכים משתנים בהתאם לסטטוס הטעינה (מכונת מצבים של IEC 61851).
        """
        data = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "status": self.status,
            "sensors": {}
        }

        # 1. סימולציית Control Pilot (CP) - הלב של התקשורת
        if self.status == "IDLE":
            cp_voltage = 12.0
            pwm_duty_cycle = 0
            current_draw = 0.0
        elif self.status == "CONNECTED":
            cp_voltage = 9.0
            pwm_duty_cycle = 5.0  # Handshake
            current_draw = 0.0
        elif self.status == "CHARGING":
            cp_voltage = 6.0
            pwm_duty_cycle = 50.0  # נניח שזה מתיר 30 אמפר
            # הוספת רעש אקראי (Noise) כדי שזה ייראה כמו סנסור אמיתי
            current_draw = 30.0 + random.uniform(-0.5, 0.5)
            self.battery_level += 0.1  # טעינה מתקדמת
            self.temp += random.uniform(0.05, 0.2)  # הטמפרטורה עולה בטעינה
        elif self.status == "FINISHING":
            cp_voltage = 9.0
            pwm_duty_cycle = 0
            current_draw = 0.0
            self.temp -= 0.1  # מתחיל להתקרר

        # הוספת רעש קטן למתח ה-CP (סנסורים לא מושלמים במציאות)
        final_cp_voltage = cp_voltage + random.uniform(-0.1, 0.1)

        data["sensors"] = {
            "cp_voltage_v": round(final_cp_voltage, 2),
            "pwm_duty_cycle_percent": pwm_duty_cycle,
            "output_voltage_v": 230.0 + random.uniform(-2, 2),  # מתח רשת משתנה קלות
            "output_current_a": round(current_draw, 2),
            "connector_temp_c": round(self.temp, 1),
            "vehicle_soc_percent": min(int(self.battery_level), 100)
        }

        return data


# --- הרצת הסימולציה ---
sim = ChargingSessionSimulator(session_id="SESSION_1024")
log_data = []

print("Starting Charging Simulation...")

# 1. שלב המתנה (Idle)
for _ in range(3):
    log_data.append(sim.generate_sensor_data())

# 2. חיבור הרכב
sim.status = "CONNECTED"
for _ in range(3):
    log_data.append(sim.generate_sensor_data())

# 3. התחלת טעינה (Charging)
sim.status = "CHARGING"
for _ in range(20):  # נניח 20 דגימות זמן של טעינה
    log_data.append(sim.generate_sensor_data())
    if sim.battery_level >= 80:  # סימולציה מקוצרת
        break

# 4. סיום
sim.status = "FINISHING"
for _ in range(3):
    log_data.append(sim.generate_sensor_data())

# שמירה לקובץ
filename = "ev_charging_log_normal.json"
with open(filename, "w") as f:
    json.dump(log_data, f, indent=4)

print(f"Simulation complete. Data saved to {filename}")