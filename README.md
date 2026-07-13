# Customer Churn Prediction - End-to-End MLOps Pipeline

An end-to-end Machine Learning project that predicts customer churn using multiple ML models. The project follows a complete MLOps workflow including data preprocessing, feature engineering, model training, experiment tracking with MLflow, model versioning, AWS S3 model storage, FastAPI deployment, Docker containerization, and automated CI/CD deployment using GitHub Actions and AWS Elastic Beanstalk.

---

# Project Architecture

```
                Raw Dataset
                     │
                     ▼
            Data Preprocessing
                     │
                     ▼
          Feature Engineering
                     │
                     ▼
        Train Multiple Models
                     │
                     ▼
          MLflow Experiment Tracking
                     │
                     ▼
           Select Best Model
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
 Save Joblib Model        Save MLflow Model
(models/model.joblib)      (registered_model/)
        │                         │
        └────────────┬────────────┘
                     ▼
            Upload Model to AWS S3
                     │
                     ▼
              FastAPI Application
                     │
                     ▼
      Download MLflow Model from S3
                     │
                     ▼
               Prediction API
                     │
                     ▼
             Docker Container
                     │
                     ▼
        AWS Elastic Beanstalk
                     │
                     ▼
      GitHub Actions CI/CD Pipeline
```

---

# Tech Stack

- Python
- Scikit-Learn
- Pandas
- MLflow
- FastAPI
- Pydantic
- Docker
- AWS S3
- AWS Elastic Beanstalk
- GitHub Actions
- Boto3

---

# Project Structure

```
customer_churn_prediction/

│
├── data/
│   ├── raw_dataset.csv
│   └── processed_dataset.csv
│
├── notebooks/
│   └── EDA.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
│   └── model.joblib
│
├── registered_model/
│   ├── MLmodel
│   ├── model.skops
│   ├── conda.yaml
│   ├── python_env.yaml
│   └── requirements.txt
│
├── app.py
├── config.py
├── upload_to_s3.py
├── download_from_s3.py
├── Dockerfile
├── requirements.txt
├── .env
└── README.md
```

---

# Features

- Data preprocessing pipeline
- Feature engineering
- Multiple model training
- Automatic best model selection
- MLflow experiment tracking
- MLflow model packaging
- Upload MLflow model to AWS S3
- Download model automatically during API startup
- REST API using FastAPI
- Docker containerization
- AWS Elastic Beanstalk deployment
- Automated CI/CD using GitHub Actions

---

# Running the Project Locally

## 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/customer_churn_prediction.git

cd customer_churn_prediction
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Data Preprocessing

```bash
python src/preprocessing.py
```

---

## 4. Train Models

```bash
python src/train.py
```

This will:

- Train Logistic Regression
- Train Random Forest
- Log experiments to MLflow
- Select the best model
- Save Joblib model
- Create the MLflow Registered Model

---

## 5. Evaluate Model

```bash
python src/evaluate.py
```

---

## 6. Upload Best Model to AWS S3

```bash
python upload_to_s3.py
```

---

## 7. Run the FastAPI Server

```bash
uvicorn app:app --reload
```

API Documentation:

```
http://127.0.0.1:8000/docs
```

---

# Prediction Request

POST `/predict`

```json
{
    "Age":35,
    "Gender":1,
    "Tenure":12,
    "Usage_Frequency":15,
    "Support_Calls":3,
    "Payment_Delay":10,
    "Subscription_Type":1,
    "Contract_Length":0,
    "Total_Spend":500,
    "Last_Interaction":5
}
```

Example Response

```json
{
    "prediction":0
}
```

---

# MLflow

Start MLflow UI

```bash
mlflow ui
```

Open

```
http://127.0.0.1:5000
```

MLflow is used for:

- Experiment Tracking
- Metric Logging
- Parameter Logging
- Model Versioning

---

# Docker

Build Docker Image

```bash
docker build -t churn-app .
```

Run Container

```bash
docker run -p 8000:8000 churn-app
```

---

# AWS S3

The best MLflow model is stored in Amazon S3.

Workflow:

```
Train Model
      │
      ▼
Create registered_model/
      │
      ▼
Upload to S3
      │
      ▼
FastAPI downloads model
      │
      ▼
Prediction
```

---

# CI/CD Pipeline

Deployment is fully automated using GitHub Actions.

Workflow

```
Developer
      │
      ▼
git push
      │
      ▼
GitHub Actions
      │
      ▼
Build Deployment Package
      │
      ▼
Deploy to Elastic Beanstalk
      │
      ▼
Docker Container Starts
      │
      ▼
Download Model from AWS S3
      │
      ▼
Prediction API
```

---

# Deployment

The application is deployed on

- AWS Elastic Beanstalk
- Docker
- FastAPI

Every push to the `main` branch automatically triggers the deployment workflow through GitHub Actions.

---

# Future Improvements

- Model Registry Integration
- Unit Testing
- Monitoring & Logging
- Model Drift Detection
- Automatic Retraining
- Kubernetes Deployment
- CI/CD Quality Gates
