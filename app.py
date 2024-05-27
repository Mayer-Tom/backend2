from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("loan_approval_model.pkl")

app = Flask(__name__)
CORS(app)

# Define the expected feature names
feature_names = [
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Credit_History",
    "Property_Area",
]


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # Extract features from the request
    features = [
        data["self_employed"],
        data["applicant_income"],
        data["coapplicant_income"],
        data["loan_amount"],
        data["past_defaults"],
        data["property_area"],
    ]

    # Create a DataFrame to match the feature names
    features_df = pd.DataFrame([features], columns=feature_names)

    # Make a prediction
    prediction = model.predict(features_df)[0]
    result = "Approved" if prediction == 1 else "Denied"
    return jsonify({"result": result})


if __name__ == "__main__":
    app.run(debug=True)
