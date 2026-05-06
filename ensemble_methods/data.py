"""
data.py
-------

Handles dataset loading and train/test splitting.
"""

import csv
from ensemble_methods.features import extract_features

# =========================
# Load Dataset
# =========================
def load_dataset(filepath):
    X = []
    y = []

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            url = row["URL"]
            label = row["Label"].strip().lower()

            features = extract_features(url)
            target = 1 if label == "bad" else 0

            X.append(features)
            y.append(target)

    return X, y


# =========================
# Train-Test Split
# =========================
def train_test_split(X, y, test_ratio=0.2):
    split = int(len(X) * (1 - test_ratio))

    X_train = X[:split]
    X_test = X[split:]

    y_train = y[:split]
    y_test = y[split:]

    return X_train, X_test, y_train, y_test