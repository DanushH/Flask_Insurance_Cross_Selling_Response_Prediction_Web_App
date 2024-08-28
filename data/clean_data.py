import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def clean_data():
    df = pd.read_csv("data/raw/train.csv")
    df = df.sample(frac=0.2, random_state=42)

    df = df.drop(columns=["id"])

    df = df.dropna()

    le = LabelEncoder()
    df["Gender"] = le.fit_transform(df["Gender"])
    df["Driving_License"] = le.fit_transform(df["Driving_License"])
    df["Previously_Insured"] = le.fit_transform(df["Previously_Insured"])
    df["Vehicle_Damage"] = le.fit_transform(df["Vehicle_Damage"])

    df = pd.get_dummies(df, columns=["Vehicle_Age"], drop_first=True)

    percentile_99 = df["Annual_Premium"].quantile(0.99)
    df["Annual_Premium"] = df["Annual_Premium"].apply(
        lambda x: percentile_99 if x > percentile_99 else x
    )

    scaler = StandardScaler()
    df[["Age", "Region_Code", "Annual_Premium", "Policy_Sales_Channel", "Vintage"]] = (
        scaler.fit_transform(
            df[
                [
                    "Age",
                    "Region_Code",
                    "Annual_Premium",
                    "Policy_Sales_Channel",
                    "Vintage",
                ]
            ]
        )
    )

    df.columns = (
        df.columns.str.replace(" ", "_")
        .str.replace(">", "gt")
        .str.replace("<", "lt")
        .str.lower()
    )
    print(df.columns)

    X = df.drop("response", axis=1)
    y = df["response"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    joblib.dump((X_train, X_val, y_train, y_val), "data/processed/cleaned_data.pkl")


if __name__ == "__main__":
    clean_data()
