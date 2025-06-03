import streamlit as st
import joblib
import numpy as np

# Load model and encoders
model = joblib.load(r'C:\Users\VENKAT\Downloads\iete\selling_price_model.pkl')
label_encoders = joblib.load(r'C:\Users\VENKAT\Downloads\iete\label_encoders.pkl')

st.title("📱 Mobile Selling Price Predictor")

# Input fields
brand = st.selectbox("Brand", label_encoders['Brands'].classes_)
color = st.selectbox("Color", label_encoders['Colors'].classes_)
memory = st.number_input("Memory (in GB)", min_value=1, max_value=32, value=4)
storage = st.number_input("Storage (in GB)", min_value=8, max_value=512, value=64)
camera = st.selectbox("Camera Present?", label_encoders['Camera'].classes_)
rating = st.slider("Rating", 1.0, 5.0, 4.0)
original_price = st.number_input("Original Price (₹)", min_value=1000, max_value=100000, value=10000)
discount = st.number_input("Discount (₹)", min_value=0, max_value=50000, value=1000)
discount_percentage = st.number_input("Discount Percentage (%)", min_value=0.0, max_value=100.0, value=10.0)

# Encode categorical inputs
brand_encoded = label_encoders['Brands'].transform([brand])[0]
color_encoded = label_encoders['Colors'].transform([color])[0]
camera_encoded = label_encoders['Camera'].transform([camera])[0]

# Prepare input
input_data = np.array([[brand_encoded, color_encoded, memory, storage, camera_encoded,
                        rating, original_price, discount, discount_percentage]])

# Predict
if st.button("Predict Selling Price"):
    predicted_price = model.predict(input_data)[0]
    st.success(f"📦 Predicted Selling Price: ₹{int(predicted_price)}")
