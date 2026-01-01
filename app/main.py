from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np
import pandas as pd
import os

app = FastAPI()

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

# Mount static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# Templates (HTML files)
templates = Jinja2Templates(directory=FRONTEND_DIR)

# Load models and helpers from inside app/
logreg = joblib.load(os.path.join(APP_DIR, "logistic_regression_heart_model.joblib"))
dt = joblib.load(os.path.join(APP_DIR, "decision_tree_heart_model.joblib"))
scaler = joblib.load(os.path.join(APP_DIR, "heart_scaler.joblib"))
features = joblib.load(os.path.join(APP_DIR, "heart_features.joblib"))

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
def predict(request: Request, model: str = Form(...),
            age: int = Form(...), trestbps: int = Form(...), chol: int = Form(...),
            thalach: int = Form(...), oldpeak: float = Form(...),
            sex: int = Form(...), cp: int = Form(...), fbs: int = Form(...),
            restecg: int = Form(...), exang: int = Form(...),
            slope: int = Form(...), ca: int = Form(...), thal: int = Form(...)):
    
    # Create input data
    input_data = {
        'age': age, 'trestbps': trestbps, 'chol': chol,
        'thalach': thalach, 'oldpeak': oldpeak,
        'sex': sex, 'cp': cp, 'fbs': fbs, 'restecg': restecg,
        'exang': exang, 'slope': slope, 'ca': ca, 'thal': thal
    }
    
    input_df = pd.DataFrame([input_data])
    
    # One-hot encode exactly like during training
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)
    
    # Match the exact columns from training
    input_encoded = input_encoded.reindex(columns=features, fill_value=0)
    
    features_array = input_encoded.values
    features_scaled = scaler.transform(features_array)
    
    # Predict
    if model == "logistic":
        pred = logreg.predict(features_scaled)[0]
        model_used = "Logistic Regression"
    else:
        pred = dt.predict(features_array)[0]
        model_used = "Decision Tree"
    
    result = "Heart Disease Present" if pred == 1 else "No Heart Disease"
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": result,
        "model_used": model_used
    })
