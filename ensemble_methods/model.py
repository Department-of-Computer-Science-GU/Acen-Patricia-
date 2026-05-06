"""
model.py
--------

Implements Logistic Regression (SGD & Batch) and KNN from scratch.
"""

import math
import random

# =========================
# Sigmoid function
# =========================
def sigmoid(z):
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    else:
        return math.exp(z) / (1 + math.exp(z))


# =========================
# Logistic Regression (SGD)
# =========================
def train_sgd(X, y, epochs=1000, lr=0.001):
    weights = [random.random() for _ in range(len(X[0]))]
    bias = 0.0

    for _ in range(epochs):
        for i in range(len(X)):
            z = sum(w * x for w, x in zip(weights, X[i])) + bias
            pred = sigmoid(z)
            error = pred - y[i]

            for j in range(len(weights)):
                weights[j] -= lr * error * X[i][j]

            bias -= lr * error

    return weights, bias


# =========================
# Logistic Regression (Batch)
# =========================
def train_batch(X, y, epochs=500, lr=0.001):
    weights = [0.0] * len(X[0])
    bias = 0.0

    for _ in range(epochs):
        dw = [0.0] * len(weights)
        db = 0.0

        for i in range(len(X)):
            z = sum(w * x for w, x in zip(weights, X[i])) + bias
            pred = sigmoid(z)
            error = pred - y[i]

            for j in range(len(weights)):
                dw[j] += error * X[i][j]
            db += error

        for j in range(len(weights)):
            weights[j] -= lr * dw[j] / len(X)

        bias -= lr * db / len(X)

    return weights, bias


# =========================
# Prediction (Logistic Regression)
# =========================
def predict_lr(X, weights, bias):
    preds = []

    for x in X:
        z = sum(w * xi for w, xi in zip(weights, x)) + bias
        preds.append(1 if sigmoid(z) >= 0.5 else 0)

    return preds


# =========================
# KNN
# =========================
def euclidean_distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def knn_predict(X_train, y_train, X_test, k=3):
    preds = []

    for test_point in X_test:
        distances = []

        for i in range(len(X_train)):
            dist = euclidean_distance(test_point, X_train[i])
            distances.append((dist, y_train[i]))

        distances.sort(key=lambda x: x[0])
        neighbors = distances[:k]

        votes = sum(label for _, label in neighbors)
        preds.append(1 if votes > k / 2 else 0)

    return preds