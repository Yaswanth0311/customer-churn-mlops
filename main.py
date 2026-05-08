from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Load model
model = joblib.load("model.pkl")

# Create FastAPI app
app = FastAPI()

# Input schema
class CustomerData(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: int
    PhoneService: int
    PaperlessBilling: int
    MonthlyCharges: float
    TotalCharges: float

# Home route
@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API"}

# Prediction route
@app.post("/predict")
def predict(data: CustomerData):

    input_data = pd.DataFrame([{
        "gender": data.gender,
        "SeniorCitizen": data.SeniorCitizen,
        "Partner": data.Partner,
        "Dependents": data.Dependents,
        "tenure": data.tenure,
        "PhoneService": data.PhoneService,
        "PaperlessBilling": data.PaperlessBilling,
        "MonthlyCharges": data.MonthlyCharges,
        "TotalCharges": data.TotalCharges
    }])

    prediction = model.predict(input_data)

    return {
        "churn_prediction": int(prediction[0])
    }