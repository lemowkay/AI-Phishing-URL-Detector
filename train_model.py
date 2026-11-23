
from pathlib import Path
from urllib.parse import urlsplit
import json

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
)

from features import extract_url_features


# STEP 1: Load the processed dataset
print("Loading processed dataset...")

data_path = Path("data/processed_urls.csv")
df = pd.read_csv(data_path)

# Use only features that can be extracted from a new URL
feature_columns = list(
    extract_url_features("https://example.com").keys()
)

X = df[feature_columns]
y = (df["label"] == 0).astype(int)


# STEP 2: Group URLs by hostname to reduce data leakage
def get_hostname(url):
    url = str(url).strip()

    if "://" not in url:
        url = "http://" + url

    try:
        hostname = urlsplit(url).hostname
        return (hostname or url).lower().removeprefix("www.")
    except ValueError:
        return url.lower()


groups = df["url"].map(get_hostname)


split1 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_val_idx, test_idx = next(
    split1.split(X, y, groups)
)

X_train_val = X.iloc[train_val_idx]
y_train_val = y.iloc[train_val_idx]
groups_train_val = groups.iloc[train_val_idx]

X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx]


split2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=43
)

train_idx, val_idx = next(
    split2.split(X_train_val, y_train_val, groups_train_val)
)

X_train = X_train_val.iloc[train_idx]
y_train = y_train_val.iloc[train_idx]

X_val = X_train_val.iloc[val_idx]
y_val = y_train_val.iloc[val_idx]

print("Training records:", len(X_train))
print("Validation records:", len(X_val))
print("Testing records:", len(X_test))



models = {
   "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=18,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
}


results = []

for name, model in models.items():
    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)
    precision = precision_score(y_val, predictions, zero_division=0)
    recall = recall_score(y_val, predictions, zero_division=0)
    f1 = f1_score(y_val, predictions, zero_division=0)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    results.append({
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })



results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    "f1_score",
    ascending=False
)

best_name = results_df.iloc[0]["model"]
best_model = models[best_name]

print("\nModel comparison:")
print(results_df.to_string(index=False))

print("\nBest model:", best_name)


print("\nRetraining selected model...")
best_model.fit(X_train_val, y_train_val)



test_predictions = best_model.predict(X_test)

test_metrics = {
    "accuracy": float(accuracy_score(y_test, test_predictions)),
    "precision": float(precision_score(y_test, test_predictions)),
    "recall": float(recall_score(y_test, test_predictions)),
    "f1_score": float(f1_score(y_test, test_predictions))
}

print("\nFinal test results:")
for metric, value in test_metrics.items():
    print(f"{metric}: {value:.4f}")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        test_predictions,
        labels=[0, 1],
        target_names=["Legitimate", "Phishing"],
        zero_division=0
    )
)


Path("models").mkdir(exist_ok=True)
Path("charts").mkdir(exist_ok=True)

joblib.dump(
    best_model,
    "models/phishing_detector.joblib"
)

results_df.to_csv(
    "models/model_comparison.csv",
    index=False
)

with open("models/test_metrics.json", "w") as file:
    json.dump(
        {
            "selected_model": best_name,
            **test_metrics
        },
        file,
        indent=4
    )


ConfusionMatrixDisplay.from_predictions(
    y_test,
    test_predictions,
    labels=[0, 1],
    display_labels=["Legitimate", "Phishing"],
    cmap="Blues",
    colorbar=False
)

plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.savefig(
    "charts/confusion_matrix.png",
    dpi=160
)
plt.close()

print("\nModel training completed successfully!")
print("Saved: models/phishing_detector.joblib")
print("Saved: models/model_comparison.csv")
print("Saved: models/test_metrics.json")
print("Saved: charts/confusion_matrix.png")
