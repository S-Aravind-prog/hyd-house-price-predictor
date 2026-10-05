from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained ML pipeline
pipe = pickle.load(open("pipe.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # -----------------------------
    # GET INPUTS FROM FORM
    # -----------------------------

    location = request.form["location"]
    property_type = request.form["PropertyType"]
    building_status = request.form["building_status"]

    bhk = int(request.form["BHK"])
    area = float(request.form["area_insqft"])


    # -----------------------------
    # FEATURE ENGINEERING
    # -----------------------------

    area_per_room = area / bhk


    # -----------------------------
    # CREATE INPUT DATAFRAME
    # -----------------------------

    input_data = pd.DataFrame(
        [[
            location,
            property_type,
            building_status,
            bhk,
            area,
            area_per_room
        ]],
        columns=[
            "location",
            "PropertyType",
            "building_status",
            "BHK",
            "area_insqft",
            "AreaPerRoom"
        ]
    )


    # -----------------------------
    # PREDICTION
    # -----------------------------

    prediction_lakhs = float(
        pipe.predict(input_data)[0]
    )


    # Convert Lakhs → Crores

    prediction_crore = prediction_lakhs / 100


    # -----------------------------
    # INDICATIVE RANGE
    # -----------------------------

    lower = prediction_crore * 0.90
    upper = prediction_crore * 1.10


    # -----------------------------
    # PROPERTY CATEGORY
    # -----------------------------

    if prediction_crore >= 2:

        category = "Luxury House"
        emoji = "🏰"

    elif prediction_crore >= 1:

        category = "Premium House"
        emoji = "🏡"

    else:

        category = "Affordable House"
        emoji = "🏠"


    # -----------------------------
    # SEND RESULT TO HTML
    # -----------------------------

    return render_template(
        "index.html",

        prediction=prediction_crore,
        lower=lower,
        upper=upper,

        category=category,
        emoji=emoji,

        selected_location=location,
        selected_property=property_type,
        selected_status=building_status,
        selected_bhk=bhk,
        selected_area=area
    )


if __name__ == "__main__":
    app.run(debug=True)