from flask import Flask, request, jsonify, send_from_directory
import os

from ensemble_methods.model import train_sgd, predict_lr
from ensemble_methods.features import extract_features
from ensemble_methods.data import load_dataset, train_test_split

# =========================
# App setup
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_FOLDER = os.path.join(BASE_DIR, "static")

app = Flask(__name__, static_folder=STATIC_FOLDER)

# =========================
# Load and train model ONCE
# =========================
DATA_PATH = os.path.join(BASE_DIR, "dataset.csv")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"dataset.csv not found at {DATA_PATH}")

X, y = load_dataset(DATA_PATH)
X_train, X_test, y_train, y_test = train_test_split(X, y)

weights, bias = train_sgd(X_train, y_train)

print("✅ Model trained successfully")

# =========================
# Serve frontend
# =========================
@app.route("/")
def home():
    return send_from_directory(STATIC_FOLDER, "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(STATIC_FOLDER, path)


# =========================
# Prediction API
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data or "url" not in data:
            return jsonify({"error": "URL is required"}), 400

        url = data["url"].strip()

        if not url:
            return jsonify({"error": "Empty URL provided"}), 400

        # Feature extraction
        features = extract_features(url)

        # Prediction
        prediction = predict_lr([features], weights, bias)[0]

        result = "SAFE ✅" if prediction == 0 else "MALICIOUS ❌"

        return jsonify({
            "url": url,
            "prediction": result
        })

    except Exception as e:
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500


# =========================
# Run server
# =========================
if __name__ == "__main__":
    app.run(debug=True)