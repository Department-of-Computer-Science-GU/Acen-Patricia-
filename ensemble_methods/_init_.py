"""
url_classifier package
----------------------

URL phishing/scam detection using multiple machine learning models
implemented from scratch (no external ML libraries).

## Modules

features   -- URL → numeric feature extraction (manual feature engineering)
data       -- CSV loading, preprocessing, and train/test splitting
logistic1  -- Logistic Regression (SGD version)
logistic2  -- Logistic Regression (Batch Gradient Descent version)
knn        -- K-Nearest Neighbors (distance-based classification)
predictor  -- Unified prediction interface using all models
evaluate   -- Accuracy calculation and model comparison utilities
main       -- Entry point for training, testing, and running predictions

## Description

This package implements three different machine learning models from scratch:

1. Logistic Regression (Stochastic Gradient Descent)
2. Logistic Regression (Batch Gradient Descent)
3. K-Nearest Neighbors (KNN)

All models are trained on the same dataset and used to classify URLs
as either "good" (legitimate) or "bad" (phishing/malicious).

A combined prediction interface allows comparing outputs from all models
for a given URL.
"""


