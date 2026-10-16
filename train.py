
import pandas as pd

file_path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(file_path)

print("Customer dataset successfully loaded!")
print("Total rows and columns:", df.shape)

print("\nFirst 5 customers:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nCustomer churn summary:")
print(df["Churn"].value_counts())

print("\nChurn percentage:")
print(df["Churn"].value_counts(normalize=True) * 100)

print("\nMissing values in each column:")
print(df.isnull().sum())

print("\nData types:")
print(df.dtypes)

# Clean the TotalCharges column
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

print("\nMissing values after cleaning TotalCharges:")
print(df.isnull().sum())

print("\nTotal missing values:")
print(df.isnull().sum().sum())

# Remove rows with missing values
df = df.dropna()

print("\nDataset after removing missing values:")
print("Rows and columns:", df.shape)

print("\nMissing values remaining:")
print(df.isnull().sum().sum())

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Separate customer information (X) and answer (y)
X = df.drop(columns=["Churn", "customerID"])
y = df["Churn"].map({"No": 0, "Yes": 1})

# Identify text columns and number columns
categorical_columns = X.select_dtypes(include=["object"]).columns
numeric_columns = X.select_dtypes(include=["number"]).columns

# Convert text categories into numbers and scale numeric values
preprocessor = ColumnTransformer(
    transformers=[
        ("numbers", StandardScaler(), numeric_columns),
        ("categories", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
    ]
)

# Build the model pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000)),
    ]
)

# Split data: 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train the model
model.fit(X_train, y_train)

# Predict churn for the test data
predictions = model.predict(X_test)

print("\nModel training completed!")
print("Training customers:", len(X_train))
print("Testing customers:", len(X_test))
print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print("\nDetailed report:")
print(classification_report(y_test, predictions, target_names=["No Churn", "Churn"]))

import joblib

# Save the trained model to a file
joblib.dump(model, "churn_model.pkl")

print("\nModel saved successfully as churn_model.pkl")
