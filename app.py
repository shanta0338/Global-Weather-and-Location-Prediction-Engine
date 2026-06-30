import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import joblib
import numpy as np

# Configure the page
st.set_page_config(page_title="Country Predictor", page_icon="🌍", layout="centered")

# Define the exact model architecture from your training script
class LocationClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LocationClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, x):
        return self.network(x)

# Cache the loading of artifacts so it only happens once
@st.cache_resource
def load_assets():
    scaler = joblib.load("scaler.joblib")
    encoder = joblib.load("label_encoder.joblib")
    
    input_dimension = 10  # 10 weather features
    output_dimension = len(encoder.classes_)
    
    model = LocationClassifier(input_dim=input_dimension, output_dim=output_dimension)
    model.load_state_dict(torch.load("location_classifier.pth", map_location=torch.device('cpu')))
    model.eval() # Set model to evaluation mode
    
    return model, scaler, encoder

model, scaler, encoder = load_assets()

st.title("🌍 Global Weather to Country Predictor")
st.write("Enter the current weather conditions to predict the country.")

# Create input fields for the 10 features
col1, col2 = st.columns(2)

with col1:
    temp = st.number_input("Temperature (°C)", value=20.0)
    wind_kph = st.number_input("Wind Speed (kph)", value=15.0)
    wind_degree = st.number_input("Wind Degree", value=180, min_value=0, max_value=360)
    pressure_mb = st.number_input("Pressure (mb)", value=1013.0)
    precip_mm = st.number_input("Precipitation (mm)", value=0.0)

with col2:
    humidity = st.number_input("Humidity (%)", value=50, min_value=0, max_value=100)
    cloud = st.number_input("Cloud Cover", value=20)
    visibility_km = st.number_input("Visibility (km)", value=10.0)
    uv_index = st.number_input("UV Index", value=5.0)
    gust_kph = st.number_input("Gust Speed (kph)", value=20.0)

# Prediction execution
if st.button("Predict Country", type="primary"):
    # Organise inputs into a NumPy array
    raw_inputs = np.array([[
        temp, wind_kph, wind_degree, pressure_mb, precip_mm, 
        humidity, cloud, visibility_km, uv_index, gust_kph
    ]])
    
    # Standardise the features using the loaded scaler
    scaled_inputs = scaler.transform(raw_inputs)
    
    # Convert to PyTorch tensor
    tensor_inputs = torch.tensor(scaled_inputs, dtype=torch.float32)
    
    # Perform inference
    with torch.no_grad():
        outputs = model(tensor_inputs)
        _, predicted_idx = torch.max(outputs, 1)
        
    # Decode the integer prediction back to a country name
    predicted_country = encoder.inverse_transform([predicted_idx.item()])[0]
    
    st.success(f"### Predicted Country: **{predicted_country}**")