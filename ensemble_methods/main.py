"""
main.py
-------

Entry point for training and evaluating URL classification models.
"""

from ensemble_methods.model import train_sgd, train_batch, predict_lr, knn_predict
from ensemble_methods.data import load_dataset, train_test_split
from ensemble_methods.evaluate import print_report


# =========================
# Main Execution
# =========================
def main():

    # Load and split data
    X, y = load_dataset("dataset.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Train models
    w1, b1 = train_sgd(X_train, y_train)
    w2, b2 = train_batch(X_train, y_train)

    # Predictions
    pred1 = predict_lr(X_test, w1, b1)
    pred2 = predict_lr(X_test, w2, b2)
    pred3 = knn_predict(X_train, y_train, X_test, k=3)

    # Individual model evaluation
    print("\n--- Logistic Regression (SGD) ---")
    print_report(y_test, pred1)

    print("\n--- Logistic Regression (Batch) ---")
    print_report(y_test, pred2)

    print("\n--- KNN ---")
    print_report(y_test, pred3)

    # =========================
    # Ensemble (Majority Voting)
    # =========================
    final_predictions = []

    for i in range(len(pred1)):
        votes = pred1[i] + pred2[i] + pred3[i]
        final_predictions.append(1 if votes >= 2 else 0)

    # Final evaluation
    print("\n--- Ensemble Model (Majority Voting) ---")
    print_report(y_test, final_predictions)


# =========================
# Entry Point
# =========================
if __name__ == "__main__":
    main()