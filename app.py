from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("model.pkl")
features = joblib.load("features.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    input_df = pd.DataFrame([data])

    input_df = pd.get_dummies(input_df)

    input_df = input_df.reindex(columns=features, fill_value=0)

    prediction = model.predict(input_df)

    result = "Approved" if prediction[0] == 1 else "Rejected"

    return jsonify({"Loan Status": result})

if __name__ == "__main__":
    app.run(debug=True)