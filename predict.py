
import joblib
import pandas as pd

# Load the model we saved earlier
model = joblib.load("churn_model.pkl")

# Example customer information
customer = pd.DataFrame([{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.0,
    "TotalCharges": 1020.0
}])

# Predict whether this customer may churn
prediction = model.predict(customer)[0]
probability = model.predict_proba(customer)[0][1]

print("Customer churn prediction:")

if prediction == 1:
    print("The model predicts: CHURN")
else:
    print("The model predicts: NO CHURN")

print(f"Estimated churn probability: {probability * 100:.2f}%")
