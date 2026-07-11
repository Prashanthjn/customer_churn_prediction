

import pandas as pd

PROCESSED_PATH = "data/processed_customer_data.csv"


def load_processed(path=PROCESSED_PATH):
    return pd.read_csv(path)


def add_features(df):
    # avoid divide by zero on tenure/usage
    df["Spend_per_Tenure"] = df["Total Spend"] / (df["Tenure"] + 1)
    df["Calls_per_Usage"] = df["Support Calls"] / (df["Usage Frequency"] + 1)
    return df


def get_feature_target_split(df):
    drop_cols = ["Churn"]
    if "CustomerID" in df.columns:
        drop_cols.append("CustomerID")

    X = df.drop(columns=drop_cols)
    y = df["Churn"]
    return X, y


if __name__ == "__main__":
    df = load_processed()
    df = add_features(df)
    print(df.head())
    print(df.shape)
