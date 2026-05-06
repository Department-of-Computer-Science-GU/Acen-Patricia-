# 🔍 URL Classification System (Machine Learning Project)

This project is a machine learning-based system that classifies URLs as **Safe** or **Malicious (Phishing/Bad URLs)** using custom-built models from scratch.

It includes:
- Logistic Regression (SGD & Batch Gradient Descent)
- K-Nearest Neighbors (KNN)
- Ensemble Majority Voting
- Custom feature extraction from URLs
- Evaluation metrics (Accuracy, Precision, Recall, F1-score)
- Simple web UI (HTML, CSS, JavaScript)

## ⚙️ Features Extracted from URLs

The system extracts simple numerical features such as:

- URL length
- Number of dots (.)
- Number of slashes (/)
- Number of special characters (@)
- Presence of HTTPS
- Presence of hyphen (-)

---

## 🧠 Machine Learning Models Used

### 1. Logistic Regression (SGD)
- Updates weights per sample
- Faster for large datasets

### 2. Logistic Regression (Batch)
- Updates weights using full dataset per epoch

### 3. K-Nearest Neighbors (KNN)
- Distance-based classification using Euclidean distance

### 4. Ensemble Model
- Combines predictions using majority voting

---

## 📊 Evaluation Metrics

The system evaluates models using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
## 🚀 How to Run the Project

### 1. Clone the repository
```bash
git clone https://github.com/your-username/url-classifier.git
cd url-classifier
project/
│
├── main.py # Entry point (runs training + evaluation)
├── model.py # ML models (Logistic Regression & KNN)
├── data.py # Dataset loading & splitting
├── features.py # URL feature extraction
├── evaluate.py # Evaluation metrics
│
├── dataset.csv # Dataset (URL, Label)
│
├── static/ # Frontend UI
│ ├── index.html
│ ├── style.css
│ └── app.js
│
└── README.md


