
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

RAW_PATH = DATA_DIR / "customer_churn_dataset.csv"
PROCESSED_PATH = DATA_DIR / "processed_customer_data.csv"

df = pd.read_csv(RAW_PATH)


def load_data(path=RAW_PATH):
    df = pd.read_csv(path)
    return df


def clean_data(df):
    df = df.drop_duplicates()
    df = df.dropna()
    return df


def encode_data(df):
    df["Gender"] = df["Gender"].map({"Female": 0, "Male": 1})

    df["Subscription Type"] = df["Subscription Type"].map(
        {"Basic": 0, "Standard": 2, "Premium": 1}
    )

    df["Contract Length"] = df["Contract Length"].map(
        {"Monthly": 1, "Quarterly": 2, "Annual": 0}
    )

    return df


def run_preprocessing():
    df = load_data()
    df = clean_data(df)
    df = encode_data(df)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Processed data saved to {PROCESSED_PATH}, shape: {df.shape}")
    return df


if __name__ == "__main__":
    run_preprocessing()
