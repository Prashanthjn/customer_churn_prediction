# Customer Churn Prediction

An end-to-end ML project takes raw information and turns it into a useful, working application. It starts by cleaning messy data and shaping new features, trains two different models to compare them, uses MLflow to log and select the best version, and finishes by serving that model through a FastAPI web app.

## Project structure

```
data/                   raw + processed CSVs
notebooks/EDA.ipynb      exploration
src/preprocessing.py     cleaning + encoding
src/feature_engineering.py   derived features
src/train.py             trains models, logs to MLflow, saves best model
src/evaluate.py          evaluates the saved model
src/predict.py           prediction helper used by the API
models/model.joblib       saved best model
app.py                    FastAPI app
Dockerfile
requirements.txt
```

## Running locally

```bash
pip install -r requirements.txt

# 1. clean + encode raw data
python src/preprocessing.py

# 2. train models (logs to MLflow, saves best model)
python src/train.py

# 3. check metrics
python src/evaluate.py

# 4. run the API
uvicorn app:app --reload
```

Then hit `POST /predict` with a JSON body like:

```json
{
  "Age": 35,
  "Gender": 1,
  "Tenure": 12,
  "Usage_Frequency": 15,
  "Support_Calls": 3,
  "Payment_Delay": 10,
  "Subscription_Type": 1,
  "Contract_Length": 0,
  "Total_Spend": 500,
  "Last_Interaction": 5
}
```

`Gender`: 0 = Female, 1 = Male
`Subscription_Type`: 0 = Basic, 1 = Premium, 2 = Standard
`Contract_Length`: 0 = Annual, 1 = Monthly, 2 = Quarterly

## MLflow

View experiment runs with:

```bash
cd customer_churn_prediction
mlflow ui 
```

If you start the UI from a different folder, pass the same backend-store-uri so it reads the same tracking database.

## Docker

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```
