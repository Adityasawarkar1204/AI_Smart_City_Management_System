 #  AI-Driven Smart City Management System  
### *(Data Science + AI + FastAPI Integration)*  

**Author:** Aditya Sawarkar  
**Hackathon Project Submission**  
**Tech Stack:** Python • FastAPI • TensorFlow • scikit-learn • Pandas • NumPy  

---

##  Objective  
To build an **AI-powered Smart City Management System** that helps city administrators make **data-driven, predictive, and automated decisions** for:  
-  **Traffic flow optimization**  
-  **Energy consumption forecasting**  
-  **Air Quality (AQI) prediction**  
-  **Emergency detection** from CCTV feeds  

The system integrates multiple ML/DL models into a unified **FastAPI backend** for real-time predictions.

---

## System Architecture  

             ┌───────────────────────────────┐
             │   Raw Datasets (Kaggle)       │
             └────────────┬──────────────────┘
                          │
              Data Preprocessing & Cleaning
                          │
                          ▼
         ┌──────────────────────────────────┐
         │  Feature Engineering (Traffic,    │
         │  Energy, AQI, Emergency)          │
         └──────────────────────────────────┘
                          │
                          ▼
         ┌──────────────────────────────────┐
         │   ML/DL Models                   │
         │   - RandomForest (Traffic)       │
         │   - XGBoost (Energy)             │
         │   - RandomForest (AQI)           │
         │   - CNN (Emergency Detection)    │
         └──────────────────────────────────┘
                          │
                          ▼
         ┌──────────────────────────────────┐
         │   FastAPI Backend (Unified API)  │
         │   - /predict_traffic             │
         │   - /predict_aqi                 │
         │   - /predict_energy              │
         │   - /predict_incident (CNN)      │
         └──────────────────────────────────┘

---

##  Modules Breakdown  

### 1️⃣ **Traffic Flow Prediction**
**Goal:** Predict number of vehicles or congestion level based on time, weather, and event data.  
**Model:** Random Forest Regressor  
**Dataset:** [Metro Interstate Traffic Volume – Kaggle](https://www.kaggle.com/datasets/ulrikthygepedersen/metro-interstate-traffic-volume)  

 **Sample Output:**
```json
{
  "Predicted_Traffic_Volume": 4875.21
}
2️⃣ Energy Consumption Forecasting

Goal: Predict next-day energy demand for city zones.
Model: XGBoost Regressor
Dataset: Household Power Consumption – Kaggle
 Sample Output:

{
  "Predicted_Energy_Consumption": 5.67
}

3️⃣ Air Quality Index (AQI) Forecasting

Goal: Predict AQI based on environmental sensor data.
Model: Random Forest Regressor
Dataset: Air Quality UCI Dataset – Kaggle

 Sample Output:

{
  "Predicted_AQI": 87.12
}

4️⃣ Emergency Detection (CCTV Image Classification)

Goal: Detect whether a CCTV image contains an accident.
Model: Convolutional Neural Network (CNN)
Dataset: Accident Detection from CCTV Footage – Kaggle

 Sample Output:

{
  "Emergency_Status": " Accident Detected"
}


 Model saved as: models/cctv_model.h5

 FastAPI Integration

All ML and DL models are deployed through FastAPI as RESTful endpoints.

▶ Run the API Server
cd C:/Users/user/Desktop/main/AI_Smart_City
uvicorn api.api_integration:app --reload


Then open in browser:
 http://127.0.0.1:8000/docs

 API Endpoints Overview
Endpoint	Input Type	Output Example
/predict_traffic	JSON	{"Predicted_Traffic_Volume": 4875.21}
/predict_aqi	JSON	{"Predicted_AQI": 87.12}
/predict_energy	JSON	{"Predicted_Energy_Consumption": 5.67}
/predict_incident	Image Upload	{"Emergency_Status": " Accident Detected"}
 Folder Structure
AI_Smart_City/
│
├── api/
│   └── api_integration.py
│
├── data/
│   ├── Metro_Interstate_Traffic_Volume.csv
│   ├── AirQualityUCI.csv
│   ├── household_power_consumption.txt
│   └── cctv_incidents/
│       ├── train/
│       │   ├── Accident/
│       │   └── Non-Accident/
│       └── test/
│           ├── Accident/
│           └── Non-Accident/
│
├── models/
│   ├── traffic_model.pkl
│   ├── aqi_model.pkl
│   ├── energy_model.pkl
│   └── cctv_model.h5
│
├── notebooks/
│   ├── 1_data_cleaning.ipynb
│   ├── 2_feature_engineering.ipynb
│   ├── 3_traffic_prediction.ipynb
│   ├── 4_aqi_forecasting.ipynb
│   ├── 5_energy_forecasting.ipynb
│   └── 6_emergency_detection.ipynb
│
└── README.md

 Libraries Used
Category	Libraries
Data Processing	Pandas, NumPy
Machine Learning	scikit-learn, XGBoost
Deep Learning	TensorFlow, Keras
Visualization	Matplotlib, Seaborn
API Framework	FastAPI, Uvicorn
Image Handling	Pillow, python-multipart
🏁 Results

- Cleaned datasets ready for analysis
-Predictive models trained and validated
-Real-time REST API for all AI modules
-CNN model deployed for emergency detection

 Future Scope

Integration with Power BI or Streamlit dashboards

Real-time IoT data streaming

Automated alerts for high-risk zones

LSTM-based time-series forecasting
=======
# AI_Smart_City_Management_System
AI-driven system for predicting traffic, air quality, energy usage, and accident detection using ML &amp; CNN models
>>>>>>> fcedd569a50fc2249aeb987c75db944227702b1d
