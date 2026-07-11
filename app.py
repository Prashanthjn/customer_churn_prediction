

from fastapi import FastAPI
from pydantic import BaseModel

from src.predict import load_model, predict_single

app = FastAPI(title="Customer Churn Prediction API")

model = load_model()


class Customer(BaseModel):
    Age: int
    Gender: int          # 0 = Female, 1 = Male
    Tenure: int
    Usage_Frequency: int
    Support_Calls: int
    Payment_Delay: int
    Subscription_Type: int   # 0 = Basic, 1 = Premium, 2 = Standard
    Contract_Length: int     # 0 = Annual, 1 = Monthly, 2 = Quarterly
    Total_Spend: int
    Last_Interaction: int


@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running"}


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

    result = predict_single(customer_dict, model=model)
    return result
