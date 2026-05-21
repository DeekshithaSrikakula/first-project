import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# ===============================
# PAGE SETTINGS
# ===============================
st.set_page_config(
    page_title="Soil Nutrient AI Dashboard",
    page_icon="🌱",
    layout="wide"
)

st.markdown("<h1 style='text-align:center;color:#2E8B57;'>🌱 Soil Nutrient AI Dashboard</h1>", unsafe_allow_html=True)
st.markdown("---")

# ===============================
# LOAD MODEL & DATA
# ===============================
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

model_path = os.path.join(PROJECT_ROOT, "model", "soil_model.pkl")
scaler_path = os.path.join(PROJECT_ROOT, "model", "scaler.pkl")
data_path = os.path.join(PROJECT_ROOT, "data", "soildataset.csv")

model = pickle.load(open(model_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))
df_full = pd.read_csv(data_path)

# ===============================
# SIDEBAR INPUT PANEL
# ===============================
st.sidebar.header("🧪 Enter Soil Parameters")

pH = st.sidebar.slider("Soil pH", 0.0, 14.0, 6.5)
moisture = st.sidebar.slider("Soil Moisture %", 0.0, 100.0, 30.0)
temp = st.sidebar.slider("Temperature °C", 0.0, 50.0, 25.0)
ec = st.sidebar.slider("Electrical Conductivity", 0.0, 5.0, 1.0)
nitrogen = st.sidebar.slider("Nitrogen ppm", 0.0, 1000.0, 400.0)
phosphorus = st.sidebar.slider("Phosphorus ppm", 0.0, 500.0, 50.0)
potassium = st.sidebar.slider("Potassium ppm", 0.0, 1000.0, 300.0)
organic = st.sidebar.slider("Organic Carbon %", 0.0, 5.0, 1.2)

texture = st.sidebar.selectbox("Soil Texture", ["Loamy", "Sandy", "Clay"])

predict_btn = st.sidebar.button("🚀 Predict")

# ===============================
# PREDICTION
# ===============================
prediction_value = None

if predict_btn:

    data = {
        "Soil_pH": [pH],
        "Soil_Moisture_%": [moisture],
        "Soil_Temperature_C": [temp],
        "Electrical_Conductivity_dS_m": [ec],
        "Nitrogen_ppm": [nitrogen],
        "Phosphorus_ppm": [phosphorus],
        "Potassium_ppm": [potassium],
        "Organic_Carbon_%": [organic],
        "Soil_Texture": [texture]
    }

    df = pd.DataFrame(data)

    # Encoding categorical
    df = pd.get_dummies(df, columns=["Soil_Texture"])

    for col in ["Soil_Texture_Loamy", "Soil_Texture_Sandy"]:
        if col not in df:
            df[col] = 0

    # Feature engineering
    df["Moisture_Temp"] = df["Soil_Moisture_%"] * df["Soil_Temperature_C"]
    df["pH_Carbon"] = df["Soil_pH"] * df["Organic_Carbon_%"]

    df = df.reindex(columns=scaler.feature_names_in_, fill_value=0)

    features_scaled = scaler.transform(df)
    prediction_value = model.predict(features_scaled)[0]

# ===============================
# METRIC CARDS
# ===============================
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🌡 Avg Temperature", f"{df_full['Soil_Temperature_C'].mean():.1f} °C")

with col2:
    st.metric("💧 Avg Moisture", f"{df_full['Soil_Moisture_%'].mean():.1f} %")

with col3:
    st.metric("🧪 Avg Nitrogen", f"{df_full['Nitrogen_ppm'].mean():.1f}")

st.markdown("---")

# ===============================
# PREDICTION DISPLAY
# ===============================
if prediction_value is not None:

    st.markdown(
        f"""
        <div style="background-color:#e6ffe6;padding:30px;border-radius:15px;text-align:center;">
            <h2>Predicted Nutrient Quality Score</h2>
            <h1 style="color:#2E8B57;">{prediction_value:.2f}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    # =====================================
    # CROP RECOMMENDATION
    # =====================================
    st.markdown("---")
    st.subheader("🌾 Recommended Crops")

    if prediction_value < 4:
        st.info("Low fertility soil → Grow: Millets, Pulses")

    elif prediction_value < 7:
        st.info("Moderate fertility → Grow: Wheat, Maize")

    elif prediction_value < 10:
        st.success("Good fertility → Grow: Rice, Sugarcane")

    else:
        st.success("Highly fertile soil → Grow: Vegetables, Fruits")

    # =====================================
    # FERTILIZER SUGGESTION
    # =====================================
    st.markdown("---")
    st.subheader("🧪 Fertilizer Recommendation")

    fertilizer_list = []

    if nitrogen < 300:
        fertilizer_list.append("Add Nitrogen fertilizer (Urea)")

    if phosphorus < 40:
        fertilizer_list.append("Add Phosphorus fertilizer (DAP)")

    if potassium < 250:
        fertilizer_list.append("Add Potassium fertilizer (MOP)")

    if organic < 1.0:
        fertilizer_list.append("Add Organic Compost")

    if len(fertilizer_list) == 0:
        st.success("Soil nutrients balanced. No major fertilizer required.")

    else:
        for f in fertilizer_list:
            st.warning(f)

st.markdown("---")

# ===============================
# DASHBOARD TABS
# ===============================
tab1, tab2, tab3 = st.tabs(["📊 Correlation", "📈 Feature Importance", "📉 Distribution"])

# Correlation
with tab1:
    numeric_df = df_full.select_dtypes(include=np.number)
    corr = numeric_df.corr()

    fig = plt.figure()
    plt.imshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.colorbar()

    st.pyplot(fig)

# Feature importance
with tab2:
    importance = model.feature_importances_
    features = scaler.feature_names_in_

    fig2 = plt.figure()
    plt.barh(features, importance)

    st.pyplot(fig2)

# Distribution
with tab3:
    numeric_cols = df_full.select_dtypes(include=np.number).columns
    selected_feature = st.selectbox("Select Parameter", numeric_cols)

    fig3 = plt.figure()
    plt.hist(df_full[selected_feature], bins=20)

    st.pyplot(fig3)
