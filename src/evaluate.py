

import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

from feature_engineering import load_processed, add_features, get_feature_target_split

MODEL_PATH = "models/model.joblib"


def evaluate():
    df = load_processed()
    df = add_features(df)
    X, y = get_feature_target_split(df)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = joblib.load(MODEL_PATH)
    preds = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, preds))
    print("F1 Score:", f1_score(y_test, preds))
    print("\nClassification Report:\n", classification_report(y_test, preds))


if __name__ == "__main__":
    evaluate()
