import joblib
import streamlit as st
import pandas as pd


# ---------------- Load artifacts ----------------
model = joblib.load("model.pkl")
# ohe = joblib.load("ohe.pkl")
scaler = joblib.load("scaler.pkl")
pt = joblib.load("pt.pkl")


def predict_from_raw_input(raw_df: pd.DataFrame):
    df = raw_df.copy()

    # Power transform humidity
    df['humidity'] = pt.transform(df[['humidity']]).ravel()

    # One-hot encode categorical columns
    # ohe_cols = ['crop ID', 'soil_type', 'Seedling Stage']
    # encoded = ohe.transform(df[ohe_cols])

    # encoded_df = pd.DataFrame(
    #     encoded,
    #     columns=ohe.get_feature_names_out(ohe_cols),
    #     index=df.index
    # )

    # df = pd.concat([df.drop(columns=ohe_cols), encoded_df], axis=1)

    # Scale temperature
    df[['temp']] = scaler.transform(df[['temp']])

    # Predict
    prediction = model.predict(df)
    return prediction


# ---------------- Load data for UI ranges ----------------
dataFrame = pd.read_csv("cropdata_updated.csv")

humidity_min = float(dataFrame['humidity'].min())
humidity_max = float(dataFrame['humidity'].max())

moi_min = float(dataFrame['MOI'].min())
moi_max = float(dataFrame['MOI'].max())

# soil_types = sorted(dataFrame['soil_type'].unique())
# crops = sorted(dataFrame['crop ID'].unique())
# SeedlingStages = sorted(dataFrame['Seedling Stage'].unique())


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Crop Result Prediction", layout="centered")

st.title("🌱 Crop Result Prediction System")
st.write("Enter crop and environmental details to get prediction")

# ---- User Inputs ----
# crop_id = st.selectbox(
#     "Crop Name",
#     options=crops
# )

# soil_type = st.selectbox(
#     "Soil Type",
#     options=soil_types
# )

# seedling_stage = st.selectbox(
#     "Seedling Stage",
#     options=SeedlingStages
# )

moi = st.number_input(
    "Moisture (MOI)",
    min_value=moi_min,
    max_value=moi_max,
    value=float((moi_min + moi_max) / 2),
    step=0.01
)

temp = st.number_input(
    "Temperature (°C)",
    value=30.0
)

humidity = st.number_input(
    "Humidity (%)",
    min_value=humidity_min,
    max_value=humidity_max,
    value=float((humidity_min + humidity_max) / 2),
    step=1.0
)

# ---- Predict Button ----
if st.button("Predict Result"):
    input_df = pd.DataFrame({
        'MOI': [moi],
        'temp': [temp],
        'humidity': [humidity]
    })

    prediction = predict_from_raw_input(input_df)

    if prediction[0] == 0:
        result = "🚫 No Need of Irrigation"
    else:
        result = "💧 Irrigation Required"

    st.success(f"Predicted Result: {result}")
