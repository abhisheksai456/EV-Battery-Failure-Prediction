import os
import streamlit as st
import pandas as pd
import joblib
import requests
from pathlib import Path

# =====================================
# 1. PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="EV Battery Failure Prediction",
    page_icon="🔋",
    layout="wide"
)

st.title("🔋 EV Battery Failure Prediction")

st.write(
    "Enter battery details to predict whether "
    "an EV battery is likely to fail."
)

# =====================================
# 2. LOAD DATA AND MODEL
# =====================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "ev_battery_failure_dataset.csv"
MODEL_PATH = BASE_DIR / "backend" / "model.pkl"

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


df = load_data()
model_data = load_model()

feature_columns = model_data["feature_columns"]

# =====================================
# 3. INPUT FORM
# =====================================

st.subheader("Enter Battery Information")

battery_details = {}

with st.form("battery_form"):

    col1, col2 = st.columns(2)

    for i, feature in enumerate(feature_columns):

        current_col = col1 if i % 2 == 0 else col2

        with current_col:

            if feature not in df.columns:
                st.error(f"Feature missing from dataset: {feature}")
                st.stop()

            if pd.api.types.is_numeric_dtype(df[feature]):

                median_value = df[feature].median()

                if pd.isna(median_value):
                    median_value = 0.0

                value = st.number_input(
                    feature.replace("_", " ").title(),
                    value=float(median_value),
                    step=0.1,
                    format="%.4f"
                )

                battery_details[feature] = value

            else:

                categories = (
                    df[feature]
                    .dropna()
                    .unique()
                    .tolist()
                )

                if len(categories) == 0:
                    st.error(f"No available options for {feature}")
                    st.stop()

                value = st.selectbox(
                    feature.replace("_", " ").title(),
                    options=categories
                )

                battery_details[feature] = value

    submit = st.form_submit_button(
        "Predict Battery Failure",
        use_container_width=True
    )

# =====================================
# 4. PREDICTION
# =====================================

if submit:

    payload = {
        "battery_details": battery_details
    }

    try:

        with st.spinner("Analyzing battery information..."):

            response = requests.post(
                API_URL,
                json=payload,
                timeout=60
            )

        if response.status_code == 200:

            result = response.json()

            st.divider()
            st.subheader("Prediction Result")

            if result["prediction"] == 1:

                st.error("⚠️ Battery Failure Predicted")

            else:

                st.success("✅ No Battery Failure Predicted")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Prediction",
                    result["result"]
                )

            with col2:

                st.metric(
                    "Battery Failure Probability",
                    f'{result["failure_probability"]}%'
                )

            st.progress(
                min(
                    max(result["failure_probability"] / 100, 0.0),
                    1.0
                )
            )

        else:

            st.error(
                f"API Error: {response.status_code}"
            )

            st.write(response.text)

    except requests.exceptions.ConnectionError:

        st.error(
            "Cannot connect to FastAPI. "
            "Please start the backend first."
        )

    except requests.exceptions.RequestException as e:

        st.error(f"Request failed: {e}")

st.divider()

st.caption(
    "Educational ML project using a synthetic EV battery dataset. "
    "Predictions are not a substitute for professional battery diagnostics."
)
