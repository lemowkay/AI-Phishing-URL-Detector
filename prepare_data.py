
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from features import extract_url_features



print("Loading phishing dataset...")

data_path = Path("data/phishing_urls.csv")

if not data_path.exists():
    raise FileNotFoundError(
        "Run explore_data.py first to create data/phishing_urls.csv"
    )

df = pd.read_csv(data_path, usecols=["url", "label"])

print("Original records:", len(df))



print("\nCleaning dataset...")

df = df.dropna(subset=["url", "label"]).copy()

df["url"] = df["url"].astype(str).str.strip()
df["label"] = pd.to_numeric(df["label"], errors="coerce")


df = df[df["label"].isin([0, 1])]


df = df[df["url"].str.len() >= 4]


conflicting = df.groupby("url")["label"].transform("nunique") > 1
df = df[~conflicting]


df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)

df["label"] = df["label"].astype(int)

print("Records after cleaning:", len(df))



print("\nExtracting URL features...")

features_df = pd.DataFrame(
    [extract_url_features(url) for url in df["url"]]
)


processed_df = pd.concat(
    [
        df[["url"]],
        features_df,
        df[["label"]]
    ],
    axis=1
)


Path("data").mkdir(exist_ok=True)
Path("charts").mkdir(exist_ok=True)

processed_df.to_csv(
    "data/processed_urls.csv",
    index=False
)

print("\nProcessed dataset preview:")
print(processed_df.head())

print("\nClass distribution:")
print(processed_df["label"].value_counts())

print("\nMissing values:")
print(processed_df.isnull().sum().sum())



processed_df["category"] = processed_df["label"].map({
    0: "Phishing",
    1: "Legitimate"
})

median_lengths = processed_df.groupby(
    "category"
)["url_length"].median()

median_lengths.plot(
    kind="bar",
    title="Median URL Length by Category"
)

plt.ylabel("Median URL Length")
plt.xlabel("URL Category")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    "charts/url_length_by_class.png",
    dpi=160
)

plt.close()

print("\nData preparation completed successfully!")
print("Saved: data/processed_urls.csv")
print("Saved: charts/url_length_by_class.png")
