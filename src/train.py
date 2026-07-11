import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from feature_engineering import load_processed, add_features, get_feature_target_split

MODEL_PATH = "models/model.joblib"

mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")
mlflow.set_experiment("customer_churn")


def get_models():
    return {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }


def train_and_log(X_train, X_test, y_train, y_test):
    results = {}

    for name, model in get_models().items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds)

            mlflow.log_param("model_type", name)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)
            mlflow.sklearn.log_model(model, name)

            results[name] = {"model": model, "accuracy": acc, "f1": f1}
            print(f"{name} -> accuracy: {acc:.4f}, f1: {f1:.4f}")

    return results


def pick_best(results):
    best_name = max(results, key=lambda k: results[k]["f1"])
    return best_name, results[best_name]["model"]


def run_training():
    df = load_processed()
    df = add_features(df)
    X, y = get_feature_target_split(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = train_and_log(X_train, X_test, y_train, y_test)
    best_name, best_model = pick_best(results)

    joblib.dump(best_model, MODEL_PATH)
    print(f"\nBest model: {best_name} -> saved to {MODEL_PATH}")

    return best_name, best_model


if __name__ == "__main__":
    run_training()
