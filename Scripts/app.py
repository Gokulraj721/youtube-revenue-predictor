# app.py — Streamlit Dashboard for YouTube Revenue Prediction

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ✅ Load model bundle with feature metadata
model_path = "C:/Gokul Important things/Content Monetization Modeler/outputs/models/LinearRegression_model.pkl"
model_bundle = joblib.load(model_path)
model = model_bundle["model"]
expected_features = model_bundle["features"]

# ✅ Page config
st.set_page_config(page_title="YouTube Revenue Predictor", layout="centered")
st.title("📺 YouTube Revenue Predictor")
st.markdown("Enter your video metrics below to estimate ad revenue.")

# ✅ Input fields
views = st.number_input("Views", min_value=0, value=1000)
likes = st.number_input("Likes", min_value=0, value=100)
comments = st.number_input("Comments", min_value=0, value=20)
watch_time = st.number_input("Watch Time (minutes)", min_value=0.0, value=300.0)
video_length = st.number_input("Video Length (minutes)", min_value=0.0, value=10.0)
subscribers = st.number_input("Subscribers", min_value=0, value=5000)

# ✅ Predict button
if st.button("Predict Revenue"):
    # Prepare input as DataFrame with correct column names
    input_data = pd.DataFrame([{
        'views': views,
        'likes': likes,
        'comments': comments,
        'watch_time_minutes': watch_time,
        'video_length_minutes': video_length,
        'subscribers': subscribers
    }])

    # ✅ Reorder columns to match training
    input_data = input_data[expected_features]

    # ✅ Predict log revenue
    log_pred = model.predict(input_data)[0]
    revenue = np.expm1(log_pred)  # Inverse of log1p

    st.success(f"💰 Estimated Ad Revenue: ${revenue:,.2f}")
    st.caption("Prediction based on trained Linear Regression model.")
    
# ✅ Footer
st.markdown("---")
st.markdown(
    "Made by Gokul • [GitHub](https://github.com/Gokulraj721) • "
    "[LinkedIn](https://www.linkedin.com/in/gokulraj-karupaiyasami-7954b02a0)"
)

