
from pathlib import Path

import joblib
import pandas as pd

from features import extract_url_features


print("=" * 55)
print("AI PHISHING URL DETECTOR - SYSTEM TEST")
print("=" * 55)


model_path = Path("models/phishing_detector.joblib")

assert model_path.exists(), "Trained model file not found!"

print("\n[PASS] Trained model file exists.")

# Test 2: Load trained model
model = joblib.load(model_path)

assert hasattr(model, "predict_proba"), (
    "Model does not support probability predictions."
)

print("[PASS] Machine learning model loaded successfully.")


test_urls = [
    "https://www.google.com",
    "https://www.microsoft.com",
    "https://www.github.com",
    "http://secure-login-verify-account.example.com/update-password",
    "http://192.0.2.10/login/verify-account"
]


features = pd.DataFrame([
    extract_url_features(url)
    for url in test_urls
])

assert len(features) == len(test_urls), (
    "Feature extraction failed."
)

assert features.notna().all().all(), (
    "Missing values found in extracted features."
)

print("[PASS] URL features extracted successfully.")


predictions = model.predict(features)

phishing_index = list(model.classes_).index(1)
scores = model.predict_proba(features)[:, phishing_index]

assert all(0 <= score <= 1 for score in scores), (
    "Invalid probability score detected."
)

print("[PASS] AI predictions generated successfully.")


print("\n" + "=" * 55)
print("URL TEST RESULTS")
print("=" * 55)

for url, prediction, score in zip(
    test_urls, predictions, scores
):
    result = (
        "Potential Phishing"
        if prediction == 1
        else "Not Flagged"
    )

    print(f"\nURL: {url}")
    print(f"Prediction: {result}")
    print(f"Phishing Score: {score:.2%}")

print("\n" + "=" * 55)
print("SYSTEM TEST COMPLETED SUCCESSFULLY!")
print("=" * 55)
