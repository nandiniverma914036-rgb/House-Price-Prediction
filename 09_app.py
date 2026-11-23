
from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load Machine Learning Model
model = joblib.load("house_price_model.pkl")


@app.route("/")
def home():
    return render_template("index.html", prediction=None)          


@app.route("/predict", methods=["POST"])
def predict():

    area = float(request.form["Area"])
    bedrooms = int(request.form["Bedrooms"])
    bathrooms = int(request.form["Bathrooms"])
    floors = int(request.form["Floors"])
    year_built = int(request.form["YearBuilt"])

    location = request.form["Location"]
    condition = request.form["Condition"]
    garage = request.form["Garage"]

    input_data = pd.DataFrame([{
        "Area": area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Floors": floors,
        "YearBuilt": year_built,
        "Location": location,
        "Condition": condition,
        "Garage": garage
    }])

    prediction = model.predict(input_data)[0]

    return render_template(
        "index.html",
        prediction=round(prediction, 2)
    )


if __name__ == "__main__":
    app.run(debug=True)
