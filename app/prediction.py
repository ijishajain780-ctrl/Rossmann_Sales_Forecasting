
import joblib
import pandas as pd
from pathlib import Path


# PROJECT PATHS

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_DIR / "models"


# 1. LOAD TRAINED MODELS

def load_models():

    sales_model_path = MODEL_DIR / "sales_model.joblib"
    customers_model_path = MODEL_DIR / "customers_model.joblib"

    if not sales_model_path.exists():
        raise FileNotFoundError("Sales model file not found.")

    if not customers_model_path.exists():
        raise FileNotFoundError("Customers model file not found.")

    sales_model = joblib.load(sales_model_path)
    customers_model = joblib.load(customers_model_path)

    return sales_model, customers_model


# 2. GENERATE PREDICTIONS

def predict_sales_and_customers(
    df,
    sales_model,
    customers_model
):

    # Get expected model features
    sales_features = list(sales_model.feature_names_in_)
    customers_features = list(customers_model.feature_names_in_)

    # Check missing features
    missing_sales = [
        col for col in sales_features
        if col not in df.columns
    ]

    missing_customers = [
        col for col in customers_features
        if col not in df.columns
    ]

    if missing_sales:
        raise ValueError(
            f"Missing Sales features: {missing_sales}"
        )

    if missing_customers:
        raise ValueError(
            f"Missing Customers features: {missing_customers}"
        )

    # Prepare model inputs in expected order
    X_sales = df[sales_features]
    X_customers = df[customers_features]

    # Generate predictions
    sales_predictions = sales_model.predict(X_sales)

    customers_predictions = customers_model.predict(X_customers)

    # Store predictions
    result_df = df.copy()

    result_df = result_df.sort_values(
        "Date"
        ).reset_index(drop=True)

    result_df["Predicted_Sales"] = sales_predictions
    result_df["Predicted_Customers"] = customers_predictions

    # Avoid negative customer predictions
    result_df["Predicted_Customers"] = (
        result_df["Predicted_Customers"].clip(lower=0)
    )

    # Avoid negative sales predictions
    result_df["Predicted_Sales"] = (
        result_df["Predicted_Sales"].clip(lower=0)
    )

    return result_df
