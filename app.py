
import json
import ipaddress
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import pandas as pd
import streamlit as st

from features import extract_url_features


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "phishing_detector.joblib"
METRICS_PATH = BASE_DIR / "models" / "test_metrics.json"
COMPARISON_PATH = BASE_DIR / "models" / "model_comparison.csv"
CHART_PATH = BASE_DIR / "charts" / "confusion_matrix.png"
DATA_PATH = BASE_DIR / "data" / "processed_urls.csv"


st.set_page_config(
    page_title="AI Phishing URL Detector",
    page_icon="🔐",
    layout="wide"
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset_summary():
    df = pd.read_csv(DATA_PATH, usecols=["label"])
    return {
        "total": len(df),
        "phishing": int((df["label"] == 0).sum()),
        "legitimate": int((df["label"] == 1).sum())
    }


def validate_url(url):
    """Validate a URL without connecting to its website."""
    candidate = url.strip()

    if "://" not in candidate:
        candidate = "http://" + candidate

    try:
        parsed = urlsplit(candidate)
        hostname = parsed.hostname or ""
    except ValueError:
        return False

    try:
        ipaddress.ip_address(hostname)
        is_ip = True
    except ValueError:
        is_ip = False

    valid_host = (
        bool(hostname)
        and " " not in hostname
        and ("." in hostname or is_ip)
    )

    return parsed.scheme in ("http", "https") and valid_host



if not MODEL_PATH.exists():
    st.error("Trained model not found. Run train_model.py first.")
    st.stop()

model = load_model()


st.title("🔐 AI Phishing URL Detection System")

st.write(
    "Detect potentially suspicious website links using "
    "Machine Learning and URL feature analysis."
)

st.info(
    "This system analyses URL text only. "
    "It does not open or visit the website."
)

st.divider()


# Navigation tabs
tab1, tab2, tab3 = st.tabs([
    "🔎 URL Detection",
    "📊 Model Performance",
    "📁 Dataset Overview"
])


# TAB 1: URL DETECTION
with tab1:

    st.header("Check a Website URL")

    st.write(
        "Enter a website link below to analyse its structure "
        "and estimate phishing risk."
    )

    user_url = st.text_input(
        "Enter URL",
        placeholder="https://example.com"
    )

    if st.button("Analyse URL", type="primary"):

        if not user_url.strip():
            st.warning("Please enter a URL.")

        elif not validate_url(user_url):
            st.error("Please enter a valid HTTP or HTTPS URL.")

        else:

            # Extract features using the same function
            features = extract_url_features(user_url)

            feature_df = pd.DataFrame([features])

            # Training script defined class 1 as phishing
            phishing_index = list(model.classes_).index(1)

            phishing_score = float(
                model.predict_proba(feature_df)[0][phishing_index]
            )

            st.subheader("Analysis Results")

            st.metric(
                "Estimated Phishing Score",
                f"{phishing_score:.1%}"
            )

            st.progress(phishing_score)

            if phishing_score >= 0.80:

                st.error(
                    "HIGH RISK: The URL has strong phishing-like "
                    "patterns according to the model."
                )

            elif phishing_score >= 0.50:

                st.warning(
                    "POTENTIAL PHISHING: The model flags this URL "
                    "as suspicious."
                )

            else:

                st.info(
                    "NOT FLAGGED: The model did not identify enough "
                    "phishing-like patterns to flag this URL. "
                    "This does NOT guarantee the website is safe."
                )

            st.caption(
                "Scores are model estimates and are not calibrated "
                "guarantees of phishing risk."
            )

            with st.expander("View Extracted URL Features"):
                st.dataframe(
                    feature_df.T.rename(columns={0: "Value"})
                )


# TAB 2: MODEL PERFORMANCE
with tab2:

    st.header("Machine Learning Performance")

    if METRICS_PATH.exists():

        with open(METRICS_PATH, "r") as file:
            metrics = json.load(file)

        st.write(
            "**Selected Model:**",
            metrics.get("selected_model", "Unknown")
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Accuracy",
            f"{metrics.get('accuracy', 0):.2%}"
        )

        col2.metric(
            "Precision",
            f"{metrics.get('precision', 0):.2%}"
        )

        col3.metric(
            "Recall",
            f"{metrics.get('recall', 0):.2%}"
        )

        col4.metric(
            "F1-Score",
            f"{metrics.get('f1_score', 0):.2%}"
        )

    else:
        st.warning("Model metrics file not found.")

    if COMPARISON_PATH.exists():
        st.subheader("Model Comparison")

        comparison_df = pd.read_csv(COMPARISON_PATH)

        st.dataframe(
            comparison_df,
            hide_index=True,
            use_container_width=True
        )

    if CHART_PATH.exists():
        st.subheader("Confusion Matrix")

        st.image(
            str(CHART_PATH),
            caption="Final test-set predictions"
        )


# TAB 3: DATASET OVERVIEW
with tab3:

    st.header("Phishing Dataset Overview")

    if DATA_PATH.exists():

        summary = load_dataset_summary()

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total URLs",
            f"{summary['total']:,}"
        )

        col2.metric(
            "Phishing URLs",
            f"{summary['phishing']:,}"
        )

        col3.metric(
            "Legitimate URLs",
            f"{summary['legitimate']:,}"
        )

        st.write(
            "Dataset: UCI PhiUSIIL Phishing URL Dataset"
        )

        distribution_chart = (
            BASE_DIR / "charts" / "url_distribution.png"
        )

        if distribution_chart.exists():
            st.image(str(distribution_chart))

    else:
        st.warning("Processed dataset not found.")


st.divider()

st.caption(
    "AI Phishing URL Detection System | "
    "Python • Scikit-learn • Streamlit"
)
