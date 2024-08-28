import joblib
from xgboost import XGBClassifier


def train_model():
    X_train, X_val, y_train, y_val = joblib.load("data/processed/cleaned_data.pkl")

    print("XGBoost model training started.")
    xgb_model = XGBClassifier(
        random_state=42, use_label_encoder=False, eval_metric="logloss"
    )
    xgb_model.fit(X_train, y_train)

    joblib.dump(xgb_model, "models/xgb_model.pkl")

    print("XGBoost model trained and saved.")


if __name__ == "__main__":
    train_model()
