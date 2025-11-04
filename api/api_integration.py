#  AI-Driven Smart City Management System – Unified API

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import numpy as np
from io import BytesIO
from PIL import Image
from tensorflow.keras.models import load_model


#  Load All Models

BASE_PATH = "C:/Users/user/Desktop/main/AI_Smart_City/models"

def load_model_safe(path):
    """Safely load model, return None if unavailable"""
    try:
        model = joblib.load(path)
        print(f" Loaded: {os.path.basename(path)}")
        return model
    except Exception as e:
        print(f" Could not load {os.path.basename(path)}: {e}")
        return None

traffic_model = load_model_safe(os.path.join(BASE_PATH, "traffic_model.pkl"))
aqi_model = load_model_safe(os.path.join(BASE_PATH, "aqi_model.pkl"))
energy_model = load_model_safe(os.path.join(BASE_PATH, "energy_model.pkl"))

# CNN Accident Detection
cnn_model_path = os.path.join(BASE_PATH, "cctv_model.h5")
cnn_model = None
if os.path.exists(cnn_model_path):
    try:
        cnn_model = load_model(cnn_model_path)
        print(" Loaded CNN Accident Detection Model")
    except Exception as e:
        print(" Could not load CNN model:", e)
else:
    print(" CNN model not found, skipping emergency detection model load.")


#  FastAPI Configuration

app = FastAPI(
    title="AI-Driven Smart City Management System",
    description=" Predict Traffic •  Forecast AQI •  Optimize Energy •  Detect Accidents",
    version="2.0"
)


#  Input Schemas

class TrafficInput(BaseModel):
    weather_temp: float
    hour: int
    day_of_week: int
    number_of_vehicles: int
    event_flag: int

class AQIInput(BaseModel):
    co_gt: float
    no2_gt: float
    o3_sensor: float

class EnergyInput(BaseModel):
    global_reactive_power: float
    voltage: float
    global_intensity: float
    sub_metering_1: float
    sub_metering_2: float
    sub_metering_3: float


#  Routes

@app.get("/")
def home():
    return {"message": " Smart City AI System is Running Successfully!"}

#TRAFFIC
@app.post("/predict_traffic")
def predict_traffic(data: TrafficInput):
    if traffic_model is None:
        return {"error": "Traffic model not loaded."}
    try:
        df = pd.DataFrame([data.dict()])
        print("\n[Traffic] Incoming Data:\n", df)

        
        df = df.rename(columns={
            'weather_temp': 'temp',
            'number_of_vehicles': 'traffic_peak',
            'event_flag': 'is_weekend'
        })

        
        expected_cols = [
            'temp', 'rain_1h', 'snow_1h', 'clouds_all',
            'hour', 'day', 'month', 'day_of_week',
            'is_weekend', 'traffic_peak'
        ]
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0

        df = df[expected_cols]
        pred = traffic_model.predict(df)
        print(" Prediction Success:", pred)
        return {"Predicted_Traffic_Volume": round(float(pred[0]), 2)}

    except Exception as e:
        print(" Traffic Prediction Error:", e)
        return {"error": str(e)}

# AQI 
@app.post("/predict_aqi")
def predict_aqi(data: AQIInput):
    if aqi_model is None:
        return {"error": "AQI model not loaded."}
    try:
        df = pd.DataFrame([data.dict()])
        pred = aqi_model.predict(df)
        return {"Predicted_AQI": round(float(pred[0]), 2)}
    except Exception as e:
        return {"error": str(e)}

# ENERGY 
@app.post("/predict_energy")
def predict_energy(data: EnergyInput):
    if energy_model is None:
        return {"error": "Energy model not loaded."}
    try:
        df = pd.DataFrame([data.dict()])
        df = df[[
            'global_reactive_power',
            'voltage',
            'global_intensity',
            'sub_metering_1',
            'sub_metering_2',
            'sub_metering_3'
        ]]
        pred = energy_model.predict(df)
        return {"Predicted_Energy_Consumption": round(float(pred[0]), 2)}
    except Exception as e:
        return {"error": str(e)}

#  EMERGENCY DETECTION
@app.post("/predict_incident")
async def predict_incident(file: UploadFile = File(...)):
    if cnn_model is None:
        return {"error": "CNN model not loaded. Please train and save cctv_model.h5 first."}

    try:
        contents = await file.read()
        img = Image.open(BytesIO(contents)).convert("RGB")
        img = img.resize((128, 128))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
        prediction = cnn_model.predict(img_array)
        result = " Accident Detected" if prediction[0][0] > 0.5 else " No Accident Detected"
        return {"Emergency_Status": result}
    except Exception as e:
        return {"error": str(e)}






