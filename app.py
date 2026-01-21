import streamlit as st
import pickle
import numpy as np

# -------------------------------------------------
# Load trained ML model
# -------------------------------------------------
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error("Model could not be loaded.")
    st.write(e)
    st.stop()

# -------------------------------------------------
# App Title & Description
# -------------------------------------------------
st.title("Rider–Order Compatibility Prediction System")

st.write(
    "This application predicts whether assigning a particular order to a rider "
    "is SAFE or RISKY based on rider workload and order conditions. "
    "It is designed as a decision-support tool for delivery operations."
)

# -------------------------------------------------
# Sidebar Inputs
# -------------------------------------------------
st.sidebar.header("Rider and Order Inputs")

working_hours = st.sidebar.slider(
    "Working hours today", min_value=1, max_value=12, value=6
)

orders_completed = st.sidebar.slider(
    "Orders completed today", min_value=1, max_value=25, value=10
)

past_delays = st.sidebar.slider(
    "Past delay count", min_value=0, max_value=10, value=2
)

order_distance = st.sidebar.slider(
    "Order distance (km)", min_value=1.0, max_value=15.0, value=5.0
)

area_difficulty = st.sidebar.selectbox(
    "Area difficulty (1 = Easy, 3 = Congested)", [1, 2, 3]
)

is_peak_hour = st.sidebar.selectbox(
    "Peak hour", [0, 1]
)

weather_condition = st.sidebar.selectbox(
    "Weather condition (0 = Normal, 1 = Rain)", [0, 1]
)

# -------------------------------------------------
# Feature Engineering
# -------------------------------------------------
fatigue_score = (
    working_hours * 0.4 +
    orders_completed * 0.3 +
    past_delays * 0.3
)

st.subheader("Calculated Rider Fatigue")
st.write(f"Fatigue Score: {fatigue_score:.2f}")

# -------------------------------------------------
# Prediction
# -------------------------------------------------
if st.button("Predict Assignment Risk"):

    X_input = np.array([
        fatigue_score,
        order_distance,
        area_difficulty,
        is_peak_hour,
        weather_condition
    ]).reshape(1, -1)

    risk_probability = model.predict_proba(X_input)[0][1]

    st.subheader("Prediction Result")
    st.write(f"Risk Probability: {risk_probability:.2f}")

    if risk_probability >= 0.4:
        st.error("RISKY ASSIGNMENT")
        st.write(
            "The rider appears to be overloaded. "
            "It may be better to assign a lighter or closer order."
        )
    else:
        st.success("SAFE ASSIGNMENT")
        st.write(
            "The rider should be able to handle this order safely."
        )

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.caption(
    "This system provides decision support only. "
    "Final assignment decisions should consider real-time operational factors."
)
