
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset from your computer
dataset_path = Path("data/PhiUSIIL_Phishing_URL_Dataset.csv")
df = pd.read_csv(dataset_path)

# Keep URL and label columns
df = df[["URL", "label"]].rename(columns={"URL": "url"})

# Remove missing values
df = df.dropna()

# Convert labels into readable categories
df["category"] = df["label"].map({
    0: "Phishing",
    1: "Legitimate"
})

print("Total records:", len(df))
print("\nDataset preview:")
print(df.head())

print("\nClass distribution:")
print(df["category"].value_counts())

# Create folders
Path("data").mkdir(exist_ok=True)
Path("charts").mkdir(exist_ok=True)

# Save data for the next step
df.to_csv("data/phishing_urls.csv", index=False)

# Create chart
df["category"].value_counts().plot(
    kind="bar",
    title="Phishing vs Legitimate URLs"
)

plt.ylabel("Number of URLs")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("charts/url_distribution.png")
plt.show()

print("\nDataset analysis completed successfully!")
