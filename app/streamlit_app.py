"""
streamlit_app.py

A simple interactive demo: input restaurant attributes,
get a predicted rating from the trained model.

Run with: streamlit run app/streamlit_app.py
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Zomato Rating Predictor", page_icon="🍽️")

st.title("🍽️ Zomato Restaurant Rating Predictor")
st.write("Enter restaurant attributes to predict its expected rating.")

model = joblib.load("models/best_model.pkl")

online_order = st.selectbox("Online order available?", ["Yes", "No"])
book_table = st.selectbox("Table booking available?", ["Yes", "No"])
cost = st.number_input("Approx cost for two (₹)", min_value=0, value=500)
cuisine_count = st.slider("Number of cuisines served", min_value=1, max_value=8, value=2)
has_dish_liked_info = st.checkbox("Has 'dish liked' data available?", value=True)


location_freq = st.slider("Location popularity (frequency score)", 0.0, 0.1, 0.02)
rest_type_freq = st.slider("Restaurant type popularity (frequency score)", 0.0, 0.3, 0.1)

if st.button("Predict Rating"):
    input_df = pd.DataFrame([{
        "online_order": 1 if online_order == "Yes" else 0,
        "book_table": 1 if book_table == "Yes" else 0,

        "approx_cost(for two people)": cost,
        "location_freq": location_freq,
        "rest_type_freq": rest_type_freq,
        "cuisine_count": cuisine_count,
        "has_dish_liked_info": int(has_dish_liked_info),
    }])

    # Align columns with model's expected feature set (fill missing cuisine/type flags with 0)
    model_features = model.feature_names_in_ if hasattr(model, "feature_names_in_") else input_df.columns
    for col in model_features:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model_features]

    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Rating: ⭐ {prediction:.2f} / 5")
