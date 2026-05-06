import math
import random
import pandas as pd
import os

# =========================
# SIGMOID FUNCTION
# =========================
def sigmoid(z):
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    else:
        return math.exp(z) / (1 + math.exp(z))


# =========================
# FEATURE EXTRACTION
# =========================
def extract_features(url):
    return [
        len(url) / 100,
        url.count('.') / 10,
        url.count('/') / 10,
        url.count('@'),
        1 if "https" in url else 0,
        1 if "-" in url else 0
    ]


# =========================
# LOAD DATA (SAFE)
# =========================
file_path = r"C:\Users\acpat\Desktop\Fiona\coursework\dataset.csv"
df = pd.read_csv(file_path)
if not os.path.exists(file_path):
    raise FileNotFoundError("dataset.csv not found. Put it in the same folder or provide full path.")

df = pd.read_csv(file_path)

X = [extract_features(url) for url in df["URL"]]
y = [1 if label == "bad" else 0 for label in df["Label"]]


# =========================
# TRAIN TEST SPLIT
# =========================
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


# =========================
# MODEL 1 (SGD)
# =========================
def train(X, y, epochs=1000, lr=0.001):
    weights = [random.random() for _ in range(len(X[0]))]
    bias = 0

    for _ in range(epochs):
        for i in range(len(X)):
            z = sum(w * x for w, x in zip(weights, X[i])) + bias
            pred = sigmoid(z)
            error = pred - y[i]

            for j in range(len(weights)):
                weights[j] -= lr * error * X[i][j]

            bias -= lr * error

    return weights, bias


def predict(X, weights, bias):
    preds = []
    for x in X:
        z = sum(w * xi for w, xi in zip(weights, x)) + bias
        preds.append(1 if sigmoid(z) >= 0.5 else 0)
    return preds


# =========================
# MODEL 2 (Batch Gradient)
# =========================
def train_lr_v2(X, y, epochs=500, lr=0.001):
    n_features = len(X[0])
    weights = [0.0] * n_features
    bias = 0.0

    for _ in range(epochs):
        dw = [0.0] * n_features
        db = 0.0

        for i in range(len(X)):
            z = sum(w * x for w, x in zip(weights, X[i])) + bias
            pred = sigmoid(z)
            error = pred - y[i]

            for j in range(n_features):
                dw[j] += error * X[i][j]
            db += error

        for j in range(n_features):
            weights[j] -= lr * dw[j] / len(X)
        bias -= lr * db / len(X)

    return weights, bias


def predict_lr_v2(X, weights, bias):
    preds = []
    for x in X:
        z = sum(w * xi for w, xi in zip(weights, x)) + bias
        preds.append(1 if sigmoid(z) >= 0.5 else 0)
    return preds


# =========================
# MODEL 3 (KNN)
# =========================
def euclidean_distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def knn_predict(X_train, y_train, X_test, k=3):
    predictions = []

    for test_point in X_test:
        distances = []

        for i in range(len(X_train)):
            dist = euclidean_distance(test_point, X_train[i])
            distances.append((dist, y_train[i]))

        distances.sort(key=lambda x: x[0])
        neighbors = distances[:k]

        votes = sum(label for _, label in neighbors)
        prediction = 1 if votes > k / 2 else 0

        predictions.append(prediction)

    return predictions


# =========================
# ACCURACY FUNCTION
# =========================
def accuracy(y_true, y_pred):
    return sum(p == y for p, y in zip(y_pred, y_true)) / len(y_true)


# =========================
# TRAIN MODELS
# =========================
weights1, bias1 = train(X_train, y_train)
weights2, bias2 = train_lr_v2(X_train, y_train)


# =========================
# PREDICTIONS
# =========================
pred1 = predict(X_test, weights1, bias1)
pred2 = predict_lr_v2(X_test, weights2, bias2)
pred3 = knn_predict(X_train, y_train, X_test, k=3)


# =========================
# RESULTS
# =========================
print("LR (SGD):", accuracy(y_test, pred1))
print("LR (Batch):", accuracy(y_test, pred2))
print("KNN:", accuracy(y_test, pred3))