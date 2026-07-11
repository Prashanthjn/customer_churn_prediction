

import joblib
import pandas as pd

MODEL_PATH = "models/model.joblib"

FEATURE_ORDER = [
    "Age",
    "Gender",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Subscription Type",
    "Contract Length",
    "Total Spend",
    "Last Interaction",
    "Spend_per_Tenure",
    "Calls_per_Usage",
]


def load_model(path=MODEL_PATH):
    return joblib.load(path)


def prepare_features(customer: dict) -> pd.DataFrame:
    row = dict(customer)
    row["Spend_per_Tenure"] = row["Total Spend"] / (row["Tenure"] + 1)
    row["Calls_per_Usage"] = row["Support Calls"] / (row["Usage Frequency"] + 1)

    df = pd.DataFrame([row])
    return df[FEATURE_ORDER]


def predict_single(customer: dict, model=None):
    if model is None:
        model = load_model()

    X = prepare_features(customer)
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    return {"churn": bool(prediction), "churn_probability": float(probability)}


if __name__ == "__main__":
    sample_customer = {
        "Age": 35,
        "Gender": 1,
        "Tenure": 12,
        "Usage Frequency": 15,
        "Support Calls": 3,
        "Payment Delay": 10,
        "Subscription Type": 1,
        "Contract Length": 0,
        "Total Spend": 500,
        "Last Interaction": 5,
    }

    result = predict_single(sample_customer)
    print(result)
