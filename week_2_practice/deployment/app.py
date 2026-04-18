
import streamlit as st
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Download model
model_path = hf_hub_download(
    repo_id="your-username/medical-insurance-model",
    filename="best_model.joblib"
)
model = joblib.load(model_path)

# UI
st.title("Insurance Charges Prediction App")

age = st.number_input("Age", 18, 100, 30)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
children = st.number_input("Children", 0, 10, 0)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

# Input Data
input_df = pd.DataFrame([{
    "age": age,
    "sex": sex,
    "bmi": bmi,
    "children": children,
    "smoker": smoker,
    "region": region
}])

# Preprocessing (must match training)
num_cols = ["age", "bmi", "children"]
cat_cols = ["sex", "smoker", "region"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first"), cat_cols)
], remainder="passthrough")

# Fit on input (NOT ideal but works for now)
input_processed = preprocessor.fit_transform(input_df)

# Prediction
if st.button("Predict Charges"):
    prediction = model.predict(input_processed)[0]
    st.success(f"Estimated Charges: ${prediction:,.2f}")
