from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load your model and features
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
features_path = os.path.join(os.path.dirname(__file__), "features.pkl")

model = joblib.load(model_path)
features = joblib.load(features_path)

# Home route to render index.html
@app.route("/")
def home():
    return render_template("index.html")

# API route for predictions
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Convert JSON to DataFrame
        input_df = pd.DataFrame([data])

        # One-hot encode
        input_df = pd.get_dummies(input_df)

        # Align with training features
        input_df = input_df.reindex(columns=features, fill_value=0)

        # Predict
        prediction = model.predict(input_df)

        # Convert prediction to human-readable
        result = "Approved" if prediction[0] == 1 else "Rejected"

        return jsonify({"Loan Status": result})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Optional health check for Railway
@app.route("/health")
def health():
    return "App is running!"

# Entry point for Railway / Gunicorn
if __name__ == "__main__":
    # Use 0.0.0.0 and PORT environment variable for Railway
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)