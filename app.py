from fastapi import FastAPI
from pydantic import BaseModel
import os

# Download model from S3 if it doesn't exist locally
if not os.path.exists("registered_model"):
    import download_from_s3

from src.predict import load_model, predict_single

model = load_model()

app = FastAPI(title="Customer Churn Prediction API")


class Customer(BaseModel):
    Age: int
    Gender: int
    Tenure: int
    Usage_Frequency: int
    Support_Calls: int
    Payment_Delay: int
    Subscription_Type: int
    Contract_Length: int
    Total_Spend: int
    Last_Interaction: int


@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running"
    }


@app.post("/predict")
def predict(customer: Customer):

    customer_dict = {
        "Age": customer.Age,
        "Gender": customer.Gender,
        "Tenure": customer.Tenure,
        "Usage Frequency": customer.Usage_Frequency,
        "Support Calls": customer.Support_Calls,
        "Payment Delay": customer.Payment_Delay,
        "Subscription Type": customer.Subscription_Type,
        "Contract Length": customer.Contract_Length,
        "Total Spend": customer.Total_Spend,
        "Last Interaction": customer.Last_Interaction,
    }

    result = predict_single(customer_dict, model)

    return result