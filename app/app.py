from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pipelines.prediction_pipeline import predict


app = Flask(__name__)


@app.route("/", methods=["GET"])
def index_get():
    return render_template("form.html")


@app.route("/", methods=["POST"])
def index_post():

    data = {
        "Gender": request.form["Gender"],
        "Age": int(request.form["Age"]),
        "Driving_License": request.form["Driving_License"],
        "Region_Code": float(request.form["Region_Code"]),
        "Previously_Insured": request.form["Previously_Insured"],
        "Vehicle_Damage": request.form["Vehicle_Damage"],
        "Annual_Premium": float(request.form["Annual_Premium"]),
        "Policy_Sales_Channel": float(request.form["Policy_Sales_Channel"]),
        "Vintage": int(request.form["Vintage"]),
        "Vehicle_Age": int(request.form["Vehicle_Age"]),
    }

    if data["Driving_License"] == "Yes":
        data["Driving_License"] = 1
    elif data["Driving_License"] == "No":
        data["Driving_License"] = 0

    if data["Previously_Insured"] == "Yes":
        data["Previously_Insured"] = 1
    elif data["Previously_Insured"] == "No":
        data["Previously_Insured"] = 0

    if data["Vehicle_Age"] < 1:
        data["vehicle_age_gt_2_years"] = False
        data["vehicle_age_lt_1_year"] = True
    elif data["Vehicle_Age"] > 2:
        data["vehicle_age_lt_1_year"] = False
        data["vehicle_age_gt_2_years"] = True
    else:
        data["vehicle_age_lt_1_year"] = False
        data["vehicle_age_gt_2_years"] = False

    input_data = pd.DataFrame([data])

    # clean input_data
    le = LabelEncoder()
    input_data["Gender"] = le.fit_transform(input_data["Gender"])
    input_data["Driving_License"] = le.fit_transform(input_data["Driving_License"])
    input_data["Previously_Insured"] = le.fit_transform(
        input_data["Previously_Insured"]
    )
    input_data["Vehicle_Damage"] = le.fit_transform(input_data["Vehicle_Damage"])

    input_data = pd.get_dummies(input_data, columns=["Vehicle_Age"], drop_first=True)

    percentile_99 = input_data["Annual_Premium"].quantile(0.99)
    input_data["Annual_Premium"] = input_data["Annual_Premium"].apply(
        lambda x: percentile_99 if x > percentile_99 else x
    )

    scaler = StandardScaler()
    input_data[
        ["Age", "Region_Code", "Annual_Premium", "Policy_Sales_Channel", "Vintage"]
    ] = scaler.fit_transform(
        input_data[
            [
                "Age",
                "Region_Code",
                "Annual_Premium",
                "Policy_Sales_Channel",
                "Vintage",
            ]
        ]
    )

    input_data.columns = (
        input_data.columns.str.replace(" ", "_")
        .str.replace(">", "gt")
        .str.replace("<", "lt")
        .str.lower()
    )

    prediction = predict(input_data)
    print(f"result:{prediction}")

    return render_template("form.html", prediction=prediction[0])


if __name__ == "__main__":
    app.run(debug=True)
