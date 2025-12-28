import joblib
import streamlit as st
import pandas as pd


# load all the things
model = joblib.load("model.pkl")
ohe = joblib.load("ohe.pkl")
scaler = joblib.load("scaler.pkl")
pt = joblib.load("pt.pkl")



def predict_from_raw_input(raw_df: pd.DataFrame):

    df = raw_df.copy()

    # Power transform humidity
    df['humidity'] = pt.transform(df[['humidity']]).ravel()

    # One-hot encode categorical columns
    ohe_cols = ['crop ID', 'soil_type', 'Seedling Stage']
    encoded = ohe.transform(df[ohe_cols])

    encoded_df = pd.DataFrame(
        encoded,
        columns=ohe.get_feature_names_out(ohe_cols),
        index=df.index
    )

    df = pd.concat([df.drop(columns=ohe_cols), encoded_df], axis=1)

    # Scale temperature
    df[['temp']] = scaler.transform(df[['temp']])

    # Predict
    prediction = model.predict(df)

    return prediction



# Streamlit UI
st.set_page_config(page_title="Crop Result Prediction", layout="centered")

st.title("🌱 Crop Result Prediction System")
st.write("Enter crop and environmental details to get prediction")

# ---- User Inputs ----
crop_id = st.text_input("Crop ID", value="C001")

soil_type = st.selectbox(
    "Soil Type",
    options=["loamy", "clay", "sandy"]
)

seedling_stage = st.selectbox(
    "Seedling Stage",
    options=["early", "mid", "late"]
)

moi = st.number_input("MOI", min_value=0.0, max_value=1.0, value=0.45)

temp = st.number_input("Temperature (°C)", value=30.0)

humidity = st.number_input("Humidity (%)", value=75.0)

# ---- Predict Button ----
if st.button("Predict Result"):
    input_df = pd.DataFrame({
        'crop ID': [crop_id],
        'soil_type': [soil_type],
        'Seedling Stage': [seedling_stage],
        'MOI': [moi],
        'temp': [temp],
        'humidity': [humidity]
    })

    prediction = predict_from_raw_input(input_df)
    
    if prediction[0] == 0:
        prediction = "NO Need Of irrigation"
    else:
        prediction = "Need irrigation "
    st.success(f"Predicted Result: {prediction}")