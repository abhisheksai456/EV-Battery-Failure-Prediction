import streamlit as st
import requests

st.set_page_config(page_title="EV Battery Failure Predictor", layout="wide")
st.title("EV Battery Failure Prediction")

API_URL = "https://ev-battery-failure-prediction-7m2n.onrender.com/predict"

with st.form("battery_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        capacity_loss_percent = st.number_input("Capacity Loss (%)", 0.0, 100.0, 8.89)
        state_of_health = st.number_input("State of Health (%)", 0.0, 100.0, 91.05)
        battery_health_percent = st.number_input("Battery Health (%)", 0.0, 100.0, 91.11)
        cell_voltage_std = st.number_input("Cell Voltage Std (V)", 0.0, 5.0, 0.01607, format="%.5f")
        aging_score = st.number_input("Aging Score", 0.0, 100.0, 22.5)
    with col2:
        thermal_runaway_risk = st.number_input("Thermal Runaway Risk", 0.0, 100.0, 51.1)
        charging_quality_score = st.number_input("Charging Quality Score", 0.0, 100.0, 53.6)
        voltage_imbalance = st.number_input("Voltage Imbalance", 0.0, 100.0, 16.91)
        charge_efficiency = st.number_input("Charge Efficiency (%)", 0.0, 100.0, 95.42)
        cycle_count = st.number_input("Cycle Count", 0, 10000, 217)
    with col3:
        thermal_health_score = st.number_input("Thermal Health Score", 0.0, 100.0, 61.4)
        odometer_km = st.number_input("Odometer (km)", 0.0, 1000000.0, 78887.0)
        internal_resistance = st.number_input("Internal Resistance", 0.0, 10.0, 0.5412, format="%.4f")
        fleet_or_private = st.selectbox("Usage Type", ["Private", "Fleet"])
        battery_stress_index = st.number_input("Battery Stress Index", 0.0, 100.0, 33.2)

    submitted = st.form_submit_button("Predict Failure Risk")

if submitted:
    payload = {
        "battery_details": {
            "capacity_loss_percent": capacity_loss_percent,
            "state_of_health": state_of_health,
            "battery_health_percent": battery_health_percent,
            "cell_voltage_std": cell_voltage_std,
            "aging_score": aging_score,
            "thermal_runaway_risk": thermal_runaway_risk,
            "charging_quality_score": charging_quality_score,
            "voltage_imbalance": voltage_imbalance,
            "charge_efficiency": charge_efficiency,
            "cycle_count": cycle_count,
            "thermal_health_score": thermal_health_score,
            "odometer_km": odometer_km,
            "internal_resistance": internal_resistance,
            "fleet_or_private": fleet_or_private,
            "battery_stress_index": battery_stress_index,
        }
    }
    with st.spinner("Connecting to prediction server (may take up to 60s on first run)..."):
        try:
            res = requests.post(API_URL, json=payload, timeout=60)
            res.raise_for_status()
            output = res.json()
            if output["prediction"] == 1:
                st.error(f"{output['result']} — Failure Probability: {output['failure_probability']}%")
            else:
                st.success(f"{output['result']} — Failure Probability: {output['failure_probability']}%")
        except requests.exceptions.RequestException as e:
            st.warning(f"API connection error: {e}")
