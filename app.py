
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from streamlit_autorefresh import st_autorefresh

# ==========================================================
# CONFIGURATION
# ==========================================================
st.set_page_config(
    page_title="DC Motor | AI Health Monitor",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_FILE = "dc_motor_four_sensor_model.pkl"
FEATURE_FILE = "dc_motor_four_sensor_features.pkl"

FEATURES = [
    "Temperature_C",
    "Vibration_Level",
    "Voltage_V",
    "Current_A"
]

# ==========================================================
# INDUSTRIAL DASHBOARD STYLE
# ==========================================================
st.markdown("""
<style>
.stApp {
    background: #0b1020;
    color: #e5e7eb;
}
[data-testid="stSidebar"] {
    background: #111827;
}
.block-container {
    padding-top: 1.5rem;
    max-width: 1600px;
}
h1, h2, h3 {
    color: #f8fafc !important;
}
div[data-testid="stMetric"] {
    background: #151e30;
    border: 1px solid #293750;
    padding: 16px;
    border-radius: 12px;
}
div[data-testid="stMetricLabel"] {
    color: #9ca3af;
}
div[data-testid="stMetricValue"] {
    color: #f8fafc;
}
.panel {
    background: #151e30;
    border: 1px solid #293750;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 12px;
}
.good {
    border-left: 5px solid #22c55e;
}
.bad {
    border-left: 5px solid #ef4444;
}
.warning {
    border-left: 5px solid #f59e0b;
}
.muted {
    color: #9ca3af;
    font-size: 13px;
}
div.stButton > button {
    border-radius: 8px;
    min-height: 40px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# MODEL LOADING
# ==========================================================
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_FILE):
        clf = joblib.load(MODEL_FILE)
        cols = (
            joblib.load(FEATURE_FILE)
            if os.path.exists(FEATURE_FILE)
            else FEATURES
        )
        return clf, cols, True

    # Fallback model for demonstration only.
    # Uses synthetic labels, not real fault measurements.
    rng = np.random.default_rng(42)
    n = 6000

    data = pd.DataFrame({
        "Temperature_C": rng.uniform(25, 70, n),
        "Vibration_Level": rng.uniform(0, 12, n),
        "Voltage_V": rng.uniform(3, 12, n),
        "Current_A": rng.uniform(0.1, 3, n)
    })

    y = (
        (data["Temperature_C"] > 50) |
        (data["Vibration_Level"] > 7) |
        (data["Current_A"] > 2.3) |
        (data["Voltage_V"] < 4.5) |
        (data["Voltage_V"] > 8.5)
    ).astype(int)

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    )
    clf.fit(data[FEATURES], y)
    return clf, FEATURES, False


try:
    model, model_features, saved_model_used = load_model()
except Exception as exc:
    st.error(f"Model loading failed: {exc}")
    st.stop()

# ==========================================================
# SESSION STATE
# ==========================================================
if "history" not in st.session_state:
    st.session_state.history = []

if "reading" not in st.session_state:
    st.session_state.reading = {
        "Temperature_C": 32.0,
        "Vibration_Level": 2.0,
        "Voltage_V": 6.0,
        "Current_A": 0.8
    }

if "last_auto_count" not in st.session_state:
    st.session_state.last_auto_count = -1

if "monitoring_active" not in st.session_state:
    st.session_state.monitoring_active = False

# ==========================================================
# SENSOR SIMULATOR
# ==========================================================
def generate_reading(mode):
    rng = np.random.default_rng()

    if mode == "Normal Operation":
        return {
            "Temperature_C": round(rng.uniform(30, 43), 2),
            "Vibration_Level": round(rng.uniform(0.5, 4), 2),
            "Voltage_V": round(rng.uniform(5.2, 7.2), 2),
            "Current_A": round(rng.uniform(0.4, 1.8), 2)
        }

    if mode == "Overheating":
        return {
            "Temperature_C": round(rng.uniform(53, 68), 2),
            "Vibration_Level": round(rng.uniform(1, 5), 2),
            "Voltage_V": round(rng.uniform(5.2, 7.2), 2),
            "Current_A": round(rng.uniform(0.8, 2), 2)
        }

    if mode == "Excessive Vibration":
        return {
            "Temperature_C": round(rng.uniform(32, 45), 2),
            "Vibration_Level": round(rng.uniform(8, 11.5), 2),
            "Voltage_V": round(rng.uniform(5.2, 7.2), 2),
            "Current_A": round(rng.uniform(0.6, 2), 2)
        }

    if mode == "High Current":
        return {
            "Temperature_C": round(rng.uniform(32, 46), 2),
            "Vibration_Level": round(rng.uniform(1, 5), 2),
            "Voltage_V": round(rng.uniform(5.2, 7.2), 2),
            "Current_A": round(rng.uniform(2.5, 3), 2)
        }

    if mode == "Voltage Anomaly":
        return {
            "Temperature_C": round(rng.uniform(32, 45), 2),
            "Vibration_Level": round(rng.uniform(1, 5), 2),
            "Voltage_V": round(rng.choice([rng.uniform(3, 4.4),
                                           rng.uniform(8.6, 12)]), 2),
            "Current_A": round(rng.uniform(0.5, 2), 2)
        }

    if mode == "Combined Fault":
        return {
            "Temperature_C": round(rng.uniform(55, 68), 2),
            "Vibration_Level": round(rng.uniform(8, 11), 2),
            "Voltage_V": round(rng.uniform(3.3, 4.3), 2),
            "Current_A": round(rng.uniform(2.5, 3), 2)
        }

    # Random sensor simulation
    return {
        "Temperature_C": round(rng.uniform(25, 70), 2),
        "Vibration_Level": round(rng.uniform(0, 12), 2),
        "Voltage_V": round(rng.uniform(3, 12), 2),
        "Current_A": round(rng.uniform(0.1, 3), 2)
    }

# ==========================================================
# PREDICTION AND ALERTS
# ==========================================================
def predict(reading):
    row = pd.DataFrame(
        [[reading[f] for f in model_features]],
        columns=model_features
    )

    prediction = int(model.predict(row)[0])
    probs = model.predict_proba(row)[0]
    classes = list(model.classes_)

    normal_score = (
        float(probs[classes.index(0)]) * 100 if 0 in classes else 0
    )
    abnormal_score = (
        float(probs[classes.index(1)]) * 100 if 1 in classes else 0
    )

    return (
        prediction,
        normal_score,
        abnormal_score
    )


def alerts_for(reading):
    alerts = []

    if reading["Temperature_C"] > 50:
        alerts.append("HIGH TEMPERATURE")

    if reading["Vibration_Level"] > 7:
        alerts.append("EXCESSIVE VIBRATION")

    if reading["Current_A"] > 2.3:
        alerts.append("HIGH CURRENT")

    if not 4.5 <= reading["Voltage_V"] <= 8.5:
        alerts.append("VOLTAGE OUT OF PROTOTYPE RANGE")

    return alerts


def calculate_health(reading):
    # Transparent prototype indicator, not a certified safety score.
    score = 100

    if reading["Temperature_C"] > 50:
        score -= min(30, (reading["Temperature_C"] - 50) * 1.5)

    if reading["Vibration_Level"] > 7:
        score -= min(30, (reading["Vibration_Level"] - 7) * 6)

    if reading["Current_A"] > 2.3:
        score -= min(20, (reading["Current_A"] - 2.3) * 25)

    if not 4.5 <= reading["Voltage_V"] <= 8.5:
        score -= 25

    return int(max(0, min(100, score)))


def record_reading(reading, prediction, normal, abnormal, alerts):
    st.session_state.history.append({
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **reading,
        "Health_Score": calculate_health(reading),
        "Predicted_Condition": (
            "Abnormal" if prediction == 1 else "Normal"
        ),
        "Normal_Model_Score_Percent": round(normal, 2),
        "Abnormal_Model_Score_Percent": round(abnormal, 2),
        "Alerts": "; ".join(alerts) if alerts else "None"
    })

# ==========================================================
# SIDEBAR
# ==========================================================
st.sidebar.title("⚙️ MOTOR MONITOR")
st.sidebar.caption("AI condition monitoring platform")

page = st.sidebar.radio(
    "NAVIGATION",
    ["Overview", "Analytics", "Event Logs", "Model Details"]
)

st.sidebar.divider()
st.sidebar.subheader("Monitoring Controls")

mode = st.sidebar.selectbox(
    "Simulation scenario",
    [
        "Normal Operation",
        "Overheating",
        "Excessive Vibration",
        "High Current",
        "Voltage Anomaly",
        "Combined Fault",
        "Random"
    ]
)

auto_monitor = st.sidebar.toggle(
    "Automatic monitoring",
    value=False
)

refresh_seconds = st.sidebar.selectbox(
    "Refresh interval",
    [2, 3, 5],
    index=1
)

st.sidebar.caption(
    "Automatic monitoring generates simulated readings only."
)

if auto_monitor:
    st_autorefresh(
        interval=refresh_seconds * 1000,
        key="motor_auto_refresh"
    )

st.sidebar.divider()

if saved_model_used:
    st.sidebar.success("Trained model loaded")
else:
    st.sidebar.warning("Fallback model active")

st.sidebar.caption("Prototype • Sensor hardware not connected")

# ==========================================================
# AUTOMATIC READING UPDATE
# ==========================================================
if auto_monitor:
    # The refresh counter changes after each scheduled rerun.
    refresh_count = st.session_state.get(
        "motor_auto_refresh", 0
    )

    # Generate a reading once for each refresh.
    if refresh_count != st.session_state.last_auto_count:
        st.session_state.reading = generate_reading(mode)
        st.session_state.last_auto_count = refresh_count

        pred, normal, abnormal = predict(st.session_state.reading)
        current_alerts = alerts_for(st.session_state.reading)

        record_reading(
            st.session_state.reading,
            pred, normal, abnormal, current_alerts
        )

# ==========================================================
# TOP HEADER
# ==========================================================
st.markdown("## DC MOTOR HEALTH MONITOR")
st.caption(
    "Industrial-style condition monitoring • AI/ML classification "
    "• Sensor analytics"
)

reading = st.session_state.reading

# Manual controls only when automatic mode is disabled
if page == "Overview" and not auto_monitor:
    with st.expander("Manual Sensor Control", expanded=True):
        c1, c2, c3, c4 = st.columns(4)

        reading["Temperature_C"] = c1.slider(
            "Temperature (°C)", 25.0, 70.0,
            float(reading["Temperature_C"]), 0.5
        )
        reading["Vibration_Level"] = c2.slider(
            "Vibration", 0.0, 12.0,
            float(reading["Vibration_Level"]), 0.1
        )
        reading["Voltage_V"] = c3.slider(
            "Voltage (V)", 3.0, 12.0,
            float(reading["Voltage_V"]), 0.1
        )
        reading["Current_A"] = c4.slider(
            "Current (A)", 0.1, 3.0,
            float(reading["Current_A"]), 0.1
        )

        st.session_state.reading = reading

        if st.button("Record Current Reading", type="primary"):
            pred, normal, abnormal = predict(reading)
            current_alerts = alerts_for(reading)
            record_reading(
                reading, pred, normal, abnormal, current_alerts
            )
            st.success("Reading recorded.")

# ==========================================================
# COMMON CURRENT STATUS
# ==========================================================
reading = st.session_state.reading
prediction, normal_score, abnormal_score = predict(reading)
current_alerts = alerts_for(reading)
health = calculate_health(reading)

# ==========================================================
# OVERVIEW PAGE
# ==========================================================
if page == "Overview":

    status = "ABNORMAL" if prediction == 1 else "NORMAL"
    status_color = "#ef4444" if prediction == 1 else "#22c55e"

    st.markdown(
        f"""
        <div class="panel {'bad' if prediction else 'good'}">
            <h2 style="color:{status_color};">
                ● MOTOR STATUS: {status}
            </h2>
            <p class="muted">
                {'The model has flagged this reading for investigation.'
                 if prediction else
                 'The model classifies this reading as normal.'}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    a, b, c, d, e = st.columns(5)

    a.metric("Motor Health Score", f"{health}/100")
    b.metric("Temperature", f'{reading["Temperature_C"]:.1f} °C')
    c.metric("Vibration", f'{reading["Vibration_Level"]:.1f}')
    d.metric("Voltage", f'{reading["Voltage_V"]:.1f} V')
    e.metric("Current", f'{reading["Current_A"]:.1f} A')

    st.progress(health / 100)

    left, right = st.columns([1.25, 1])

    with left:
        st.subheader("Sensor Readings")

        sensor_df = pd.DataFrame({
            "Sensor": [
                "Temperature (°C)",
                "Vibration",
                "Voltage (V)",
                "Current (A)"
            ],
            "Reading": [
                reading["Temperature_C"],
                reading["Vibration_Level"],
                reading["Voltage_V"],
                reading["Current_A"]
            ]
        }).set_index("Sensor")

        st.bar_chart(sensor_df)

    with right:
        st.subheader("AI Model Output")

        probability_df = pd.DataFrame({
            "Class": ["Normal", "Abnormal"],
            "Model Score (%)": [normal_score, abnormal_score]
        }).set_index("Class")

        st.bar_chart(probability_df)

        st.caption(
            "Classifier scores are not validated probabilities "
            "of actual motor failure."
        )

    st.subheader("Fault Alerts")

    if current_alerts:
        for alert in current_alerts:
            st.error("⚠️ " + alert)
    elif prediction == 1:
        st.warning(
            "The model flags an abnormal condition, but none of the "
            "configured prototype thresholds triggered an alert."
        )
    else:
        st.success("No configured prototype threshold alerts.")

    st.subheader("Recent Monitoring Events")

    if st.session_state.history:
        recent = pd.DataFrame(
            st.session_state.history[-8:][::-1]
        )
        st.dataframe(recent, use_container_width=True, hide_index=True)
    else:
        st.info(
            "No readings recorded yet. Enable automatic monitoring "
            "or record a manual reading."
        )

    if auto_monitor:
        st.info(
            f"Automatic simulation is running. New readings are "
            f"generated approximately every {refresh_seconds} seconds."
        )

# ==========================================================
# ANALYTICS PAGE
# ==========================================================
elif page == "Analytics":

    st.subheader("Sensor Analytics")

    if not st.session_state.history:
        st.info("Record readings first to populate the analytics.")
    else:
        df = pd.DataFrame(st.session_state.history)

        n1, n2, n3 = st.columns(3)
        n1.metric("Total Readings", len(df))
        n2.metric(
            "Normal Classifications",
            int((df["Predicted_Condition"] == "Normal").sum())
        )
        n3.metric(
            "Abnormal Classifications",
            int((df["Predicted_Condition"] == "Abnormal").sum())
        )

        st.subheader("Sensor Trends Over Time")
        st.line_chart(
            df.set_index("Timestamp")[
                ["Temperature_C", "Vibration_Level"]
            ]
        )

        st.subheader("Electrical Measurements")
        st.line_chart(
            df.set_index("Timestamp")[
                ["Voltage_V", "Current_A"]
            ]
        )

        st.subheader("Health Score Trend")
        st.line_chart(df.set_index("Timestamp")["Health_Score"])

        st.subheader("Condition Distribution")
        st.bar_chart(df["Predicted_Condition"].value_counts())

        st.subheader("Sensor Statistics")
        st.dataframe(
            df[FEATURES].describe().round(2),
            use_container_width=True
        )

# ==========================================================
# EVENT LOGS PAGE
# ==========================================================
elif page == "Event Logs":

    st.subheader("Monitoring Event Logs")

    if not st.session_state.history:
        st.info("No logs available yet.")
    else:
        df = pd.DataFrame(st.session_state.history)

        filter_status = st.selectbox(
            "Filter by classification",
            ["All", "Normal", "Abnormal"]
        )

        shown = df
        if filter_status != "All":
            shown = df[df["Predicted_Condition"] == filter_status]

        st.dataframe(
            shown,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Complete CSV Log",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="dc_motor_health_logs.csv",
            mime="text/csv",
            type="primary"
        )

        if st.button("Clear Session Logs"):
            st.session_state.history = []
            st.rerun()

# ==========================================================
# MODEL DETAILS PAGE
# ==========================================================
elif page == "Model Details":

    st.subheader("AI/ML Model Information")

    st.markdown("""
    <div class="panel">
        <h3>Random Forest Classifier</h3>
        <p>Inputs: Temperature, Vibration, Voltage and Current</p>
        <p>Output: Normal or Abnormal</p>
    </div>
    """, unsafe_allow_html=True)

    if saved_model_used:
        st.success("Your saved four-sensor model is being used.")
    else:
        st.warning(
            "A fallback model is being used because the saved model "
            "file was not found."
        )

    if hasattr(model, "feature_importances_"):
        importance = pd.DataFrame({
            "Sensor": model_features,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)

        st.subheader("Feature Importance")
        st.bar_chart(importance.set_index("Sensor"))

    st.subheader("Prototype Thresholds")

    st.markdown("""
    | Parameter | Prototype alert rule |
    |---|---|
    | Temperature | Above 50 °C |
    | Vibration | Above 7 units |
    | Current | Above 2.3 A |
    | Voltage | Below 4.5 V or above 8.5 V |

    These are experimental assumptions, not universal motor safety limits.
    Use the actual motor rating and calibrated sensor measurements to
    establish appropriate thresholds.
    """)

    st.warning(
        "The original training labels were generated using rules on "
        "synthetic data. Real-world fault detection requires validation "
        "against measured motor data."
    )

# ==========================================================
# FOOTER
# ==========================================================
st.divider()
st.caption(
    "DC Motor Health Monitor | Academic prototype | "
    "Simulated readings — no physical sensors connected"
)