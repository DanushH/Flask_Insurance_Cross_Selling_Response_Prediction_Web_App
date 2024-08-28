import joblib
from sklearn.metrics import accuracy_score, classification_report


def evaluate_model():
    model = joblib.load("models/xgb_model.pkl")

    X_train, X_val, y_train, y_val = joblib.load("data/processed/cleaned_data.pkl")

    y_pred = model.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)

    report = classification_report(y_val, y_pred)

    print(f"Validation Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", report)


if __name__ == "__main__":
    evaluate_model()
