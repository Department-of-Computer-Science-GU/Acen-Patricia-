"""
evaluate.py
-----------

Evaluation metrics for classification models.
"""

# =========================
# Confusion Matrix
# =========================
def compute_confusion(y_true, y_pred):
    tp = fp = tn = fn = 0

    for a, p in zip(y_true, y_pred):
        if a == 1 and p == 1:
            tp += 1
        elif a == 0 and p == 1:
            fp += 1
        elif a == 0 and p == 0:
            tn += 1
        elif a == 1 and p == 0:
            fn += 1

    return tp, fp, tn, fn


# =========================
# Accuracy
# =========================
def compute_accuracy(y_true, y_pred):
    return sum(a == p for a, p in zip(y_true, y_pred)) / len(y_true)


# =========================
# Precision
# =========================
def compute_precision(y_true, y_pred):
    tp, fp, _, _ = compute_confusion(y_true, y_pred)
    return tp / (tp + fp) if (tp + fp) != 0 else 0.0


# =========================
# Recall
# =========================
def compute_recall(y_true, y_pred):
    tp, _, _, fn = compute_confusion(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) != 0 else 0.0


# =========================
# F1 Score
# =========================
def compute_f1(y_true, y_pred):
    p = compute_precision(y_true, y_pred)
    r = compute_recall(y_true, y_pred)
    return (2 * p * r) / (p + r) if (p + r) != 0 else 0.0


# =========================
# Full Report
# =========================
def print_report(y_true, y_pred):
    tp, fp, tn, fn = compute_confusion(y_true, y_pred)

    acc = compute_accuracy(y_true, y_pred)
    prec = compute_precision(y_true, y_pred)
    rec = compute_recall(y_true, y_pred)
    f1 = compute_f1(y_true, y_pred)

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("Confusion Matrix:")
    print(f"TP: {tp} | FP: {fp}")
    print(f"FN: {fn} | TN: {tn}")