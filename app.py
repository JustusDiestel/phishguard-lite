"""Streamlit interface for PhishGuard-Lite."""

from __future__ import annotations

import os
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent / ".cache"
(CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix

from src.classical_models import MAIN_MODEL_PATH, load_model_bundle, predict_url_classical, train_models
from src.feature_extraction import extract_dataset_features
from src.hybrid_detector import HybridPhishingDetector
from src.metrics import classification_metrics
from src.utils import DATA_DIR, MODEL_DIR, read_url_dataset


st.set_page_config(page_title="PhishGuard-Lite", page_icon="🛡️", layout="wide")


def ensure_model() -> bool:
    """Train a small demo model when no saved model exists."""

    if MAIN_MODEL_PATH.exists():
        return True
    sample_path = DATA_DIR / "sample_urls.csv"
    if not sample_path.exists():
        st.error("No trained model found. Run: python train.py")
        return False
    with st.spinner("Training a small demo model from data/sample_urls.csv..."):
        train_models(sample_path, MODEL_DIR)
    return True


def plot_confusion(y_true: list[int], y_pred: list[int]) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1], labels=["Legitimate", "Phishing"])
    ax.set_yticks([0, 1], labels=["Legitimate", "Phishing"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=12)
    fig.tight_layout()
    return fig


def plot_feature_importance() -> plt.Figure | None:
    bundle = load_model_bundle(MAIN_MODEL_PATH)
    model = bundle["model"]
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return None
    ranking = (
        pd.DataFrame({"feature": bundle["feature_names"], "importance": importances})
        .sort_values("importance", ascending=True)
        .tail(12)
    )
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.barh(ranking["feature"], ranking["importance"], color="#2563EB")
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance")
    fig.tight_layout()
    return fig


st.title("PhishGuard-Lite")
st.caption(
    "A lightweight hybrid phishing URL detector using handcrafted features, classical machine learning, "
    "optional pretrained Hugging Face models, and transparent rule-based explanations."
)

if not ensure_model():
    st.stop()

with st.sidebar:
    st.header("Settings")
    use_hf = st.checkbox("Enable Hugging Face model", value=False)
    st.caption("Keep disabled for offline demos. If enabled, the app will try to load a pretrained URL model.")

detector = HybridPhishingDetector(use_hf=use_hf)

tab_single, tab_batch, tab_model = st.tabs(["Single URL", "Batch Evaluation", "Model Details"])

with tab_single:
    st.subheader("Single URL Analysis")
    url = st.text_input("URL", value="https://paypal.verify-account.example-login.test/security/update")
    analyze = st.button("Analyze URL", type="primary")

    if analyze and url.strip():
        result = detector.analyze_url(url.strip())
        col1, col2, col3 = st.columns(3)
        col1.metric("Final prediction", result["prediction"].upper())
        col2.metric("Risk level", result["risk_level"])
        col3.metric("Confidence", f"{result['confidence']:.2f}")

        st.markdown("#### Model comparison")
        scores = pd.DataFrame(
            [
                {"component": "Classical ML", "score": result["classical_score"], "label": result["classical_label"]},
                {"component": "Hugging Face", "score": result["hf_score"], "label": result["hf_label"]},
                {"component": "Rule-based", "score": result["rule_score"], "label": "risk"},
                {"component": "Hybrid final", "score": result["final_score"], "label": result["prediction"]},
            ]
        )
        st.dataframe(scores, use_container_width=True, hide_index=True)
        if not result["hf_available"]:
            st.info("Hugging Face layer is unavailable or disabled; the hybrid score used the fallback formula.")

        st.markdown("#### Explanation")
        for explanation in result["explanations"]:
            st.write(f"- {explanation}")

        st.markdown("#### Extracted feature table")
        st.dataframe(pd.DataFrame([result["features"]]).T.rename(columns={0: "value"}), use_container_width=True)

with tab_batch:
    st.subheader("Batch CSV Evaluation")
    uploaded = st.file_uploader("Upload CSV with columns: url,label", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            temp_path = Path("results/tables/uploaded_dataset.csv")
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(temp_path, index=False)
            clean_df = read_url_dataset(temp_path)
            bundle = load_model_bundle(MAIN_MODEL_PATH)
            y_pred = [int(predict_url_classical(url, bundle)["prediction"]) for url in clean_df["url"]]
            metrics = classification_metrics(clean_df["label"].to_numpy(), y_pred)
            st.dataframe(pd.DataFrame([metrics]), use_container_width=True, hide_index=True)
            st.pyplot(plot_confusion(clean_df["label"].tolist(), y_pred))
        except Exception as exc:
            st.error(f"Could not evaluate dataset: {exc}")
    else:
        st.write("Upload a labeled CSV to evaluate the trained classical model.")

with tab_model:
    st.subheader("Trained Model Details")
    bundle = load_model_bundle(MAIN_MODEL_PATH)
    st.write(f"Main model: **{bundle['model_name']}**")
    st.write(f"Number of features: **{len(bundle['feature_names'])}**")
    fig = plot_feature_importance()
    if fig is not None:
        st.pyplot(fig)

    st.markdown("#### Feature names")
    st.dataframe(pd.DataFrame({"feature": bundle["feature_names"]}), use_container_width=True, hide_index=True)
