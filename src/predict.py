import mlflow.pyfunc
import pandas as pd

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


def load_model():
    return mlflow.pyfunc.load_model("registered_model")


def prepare_features(customer):
    row = dict(customer)

    row["Spend_per_Tenure"] = row["Total Spend"] / (row["Tenure"] + 1)
    row["Calls_per_Usage"] = row["Support Calls"] / (row["Usage Frequency"] + 1)

    df = pd.DataFrame([row])

    return df[FEATURE_ORDER]


def predict_single(customer, model=None):

    if model is None:
        model = load_model()

    X = prepare_features(customer)

    prediction = model.predict(X)

    return {
        "prediction": int(prediction[0])
    }


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

    print(predict_single(sample_customer))