"""
Train a Logistic Regression model for URL classification
and save it for Flask backend use.
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from features import extract_features


# =========================
# Paths
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dataset.csv")

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")


# =========================
# Load dataset
# =========================
def load_data():
    df = pd.read_csv(DATA_PATH)

    df = df.dropna(subset=["URL", "Label"])
    df["label_bin"] = (df["Label"].str.lower().str.strip() == "bad").astype(int)

    X = df["URL"].apply(extract_features).tolist()
    y = df["label_bin"].values

    return np.array(X), y


# =========================
# Train model
# =========================
def train_model():
    print("[1] Loading data...")
    X, y = load_data()

    print("[2] Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("[3] Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"[4] Model Accuracy: {accuracy:.4f}")

    print("[5] Saving model...")
    joblib.dump(model, MODEL_PATH)

    print(f"Model saved at: {MODEL_PATH}")


# =========================
# Run training
# =========================
if __name__ == "__main__":
    train_model()