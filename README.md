# 🔐 AI Phishing URL Detection System

A machine learning-based cybersecurity application that analyses website URLs and predicts whether they may be phishing or legitimate.

Built using **Python, Scikit-learn, Pandas, and Streamlit**.

## 🎯 Project Overview

This project aims to identify potentially malicious URLs by analysing their structure and extracting features commonly associated with phishing attacks.

### Key Features

- Phishing URL detection using machine learning
- URL feature extraction and data preprocessing
- Logistic Regression and Random Forest classification
- Model evaluation using accuracy, precision, recall, and F1-score
- Interactive Streamlit dashboard
- Dataset analysis and visualisation

## 🛠️ Technologies Used

- Python 3.12
- Pandas & NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Joblib

## 📊 Dataset

**Source:** [UCI PhiUSIIL Phishing URL Dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset)

The original dataset contains **235,795 URLs**.

After cleaning:

- **235,370** total URLs
- **100,520** phishing URLs
- **134,850** legitimate URLs

## ⚙️ How It Works

1. Load and clean the phishing URL dataset.
2. Extract 13 URL-based features.
3. Train Logistic Regression and Random Forest models.
4. Select the best model using validation F1-score.
5. Evaluate the model on a separate test set.
6. Predict phishing risk through a Streamlit dashboard.

The system analyses URL text without visiting websites.

## 💡 Skills Demonstrated

Cybersecurity | Python | Machine Learning | Data Analysis | Feature Engineering | Streamlit | GitHub
