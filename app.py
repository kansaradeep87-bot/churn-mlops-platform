
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Create the API application
app = FastAPI(title="Customer Churn Prediction API")

# Load our trained model
model = joblib.load("churn_model.pkl")


# Define the customer information we expect
class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running!"}


@app.post("/predict")
def predict_churn(customer: Customer):
    customer_data = pd.DataFrame([customer.model_dump()])

    prediction = int(model.predict(customer_data)[0])
    probability = float(model.predict_proba(customer_data)[0][1])

    return {
        "prediction": "Churn" if prediction == 1 else "No Churn",
        "churn_probability_percent": round(probability * 100, 2)
    }
