import joblib


def predict(input_data):
    model = joblib.load("models/xgb_model.pkl")

    predictions = model.predict(input_data)

    return predictions
