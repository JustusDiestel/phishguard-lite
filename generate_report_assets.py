"""Generate figures and tables for the PhishGuard-Lite report."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent / ".cache"
(CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.classical_models import MAIN_MODEL_PATH, load_model_bundle, predict_url_classical, train_models
from evaluate import evaluate_dataset
from src.feature_selection import select_features
from src.utils import DATA_DIR, FIGURES_DIR, MODEL_DIR, REPORT_FIGURES_DIR, TABLES_DIR, ensure_directories, read_url_dataset


def _copy_to_report(path: Path) -> None:
    REPORT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, REPORT_FIGURES_DIR / path.name)


def plot_confusion_matrix(data_path: Path) -> None:
    model_bundle = load_model_bundle(MAIN_MODEL_PATH)
    df = read_url_dataset(data_path)
    predictions = [int(predict_url_classical(url, model_bundle)["prediction"]) for url in df["url"]]
    cm = confusion_matrix(df["label"], predictions, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1], labels=["Legitimate", "Phishing"])
    ax.set_yticks([0, 1], labels=["Legitimate", "Phishing"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=12)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output = FIGURES_DIR / "confusion_matrix.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    _copy_to_report(output)


def plot_feature_importance() -> None:
    bundle = load_model_bundle(MAIN_MODEL_PATH)
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return

    ranking = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=True)
        .tail(12)
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(ranking["feature"], ranking["importance"], color="#3B82F6")
    ax.set_title("Random Forest Feature Importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    output = FIGURES_DIR / "feature_importance.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    _copy_to_report(output)


def plot_feature_selection_comparison(comparison: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(comparison["num_features"], comparison["f1"], marker="o", color="#0F766E", linewidth=2)
    ax.set_title("Feature Selection Comparison")
    ax.set_xlabel("Number of Features")
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output = FIGURES_DIR / "feature_selection_comparison.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    _copy_to_report(output)


def plot_system_architecture_placeholder() -> None:
    """Create a simple architecture diagram placeholder for the report."""

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axis("off")
    boxes = [
        (0.03, 0.62, "Single URL\nor CSV Dataset"),
        (0.24, 0.62, "Feature\nExtraction"),
        (0.45, 0.62, "Classical ML\nRandom Forest"),
        (0.45, 0.25, "Optional HF\nURL Model"),
        (0.67, 0.46, "Hybrid Risk\nScoring"),
        (0.84, 0.46, "Prediction +\nExplanation"),
    ]
    for x_pos, y_pos, label in boxes:
        ax.add_patch(
            plt.Rectangle((x_pos, y_pos), 0.15, 0.18, fill=False, linewidth=1.8, edgecolor="#1F2937")
        )
        ax.text(x_pos + 0.075, y_pos + 0.09, label, ha="center", va="center", fontsize=10)

    arrows = [
        ((0.18, 0.71), (0.24, 0.71)),
        ((0.39, 0.71), (0.45, 0.71)),
        ((0.60, 0.71), (0.67, 0.58)),
        ((0.60, 0.34), (0.67, 0.49)),
        ((0.82, 0.55), (0.84, 0.55)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "linewidth": 1.8})

    ax.set_title("PhishGuard-Lite System Architecture", fontsize=14, pad=12)
    fig.tight_layout()
    output = FIGURES_DIR / "system_architecture_placeholder.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    _copy_to_report(output)


def plot_streamlit_interface_placeholder() -> None:
    """Create a simple interface placeholder for the report."""

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0.05, 0.08), 0.9, 0.82, fill=False, linewidth=1.8, edgecolor="#111827"))
    ax.text(0.08, 0.84, "PhishGuard-Lite", fontsize=15, weight="bold")
    ax.add_patch(plt.Rectangle((0.08, 0.72), 0.64, 0.07, fill=False, edgecolor="#4B5563"))
    ax.text(0.10, 0.745, "URL input", fontsize=9, color="#374151")
    ax.add_patch(plt.Rectangle((0.75, 0.72), 0.14, 0.07, fill=False, edgecolor="#2563EB"))
    ax.text(0.82, 0.755, "Analyze", ha="center", va="center", fontsize=9, color="#2563EB")
    ax.add_patch(plt.Rectangle((0.08, 0.52), 0.25, 0.13, fill=False, edgecolor="#16A34A"))
    ax.text(0.205, 0.585, "Prediction\nRisk Level", ha="center", va="center", fontsize=9)
    ax.add_patch(plt.Rectangle((0.37, 0.52), 0.25, 0.13, fill=False, edgecolor="#9333EA"))
    ax.text(0.495, 0.585, "Model\nScores", ha="center", va="center", fontsize=9)
    ax.add_patch(plt.Rectangle((0.66, 0.52), 0.23, 0.13, fill=False, edgecolor="#EA580C"))
    ax.text(0.775, 0.585, "Explanations", ha="center", va="center", fontsize=9)
    ax.add_patch(plt.Rectangle((0.08, 0.16), 0.81, 0.27, fill=False, edgecolor="#4B5563"))
    ax.text(0.485, 0.295, "Extracted Feature Table / Batch Evaluation", ha="center", va="center", fontsize=10)
    ax.set_title("Streamlit Interface Placeholder", fontsize=14, pad=12)
    fig.tight_layout()
    output = FIGURES_DIR / "streamlit_interface_placeholder.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    _copy_to_report(output)


def create_latency_table(data_path: Path) -> pd.DataFrame:
    bundle = load_model_bundle(MAIN_MODEL_PATH)
    df = read_url_dataset(data_path)
    urls = df["url"].astype(str).tolist()
    classical_latencies = []
    rule_latencies = []

    from src.explanation import rule_based_score
    from src.feature_extraction import extract_url_features

    for url in urls:
        start = time.perf_counter()
        predict_url_classical(url, bundle)
        classical_latencies.append(time.perf_counter() - start)

        start = time.perf_counter()
        rule_based_score(extract_url_features(url))
        rule_latencies.append(time.perf_counter() - start)

    table = pd.DataFrame(
        [
            {
                "component": "Classical ML model",
                "average_latency_seconds": round(sum(classical_latencies) / len(classical_latencies), 6),
                "notes": "Random Forest over handcrafted URL features",
            },
            {
                "component": "Rule-based layer",
                "average_latency_seconds": round(sum(rule_latencies) / len(rule_latencies), 6),
                "notes": "Transparent feature threshold scoring",
            },
            {
                "component": "Hugging Face model",
                "average_latency_seconds": "Not measured by default",
                "notes": "Optional layer; depends on download, hardware, and chosen model",
            },
        ]
    )
    table.to_csv(TABLES_DIR / "latency_comparison_table.csv", index=False)
    return table


def main() -> None:
    ensure_directories()
    data_path = DATA_DIR / "sample_urls.csv"
    if not MAIN_MODEL_PATH.exists():
        train_models(data_path, MODEL_DIR)

    _, model_comparison = train_models(data_path, MODEL_DIR)
    model_comparison.to_csv(TABLES_DIR / "model_comparison_table.csv", index=False)
    feature_comparison, _ = select_features(data_path, MODEL_DIR)
    feature_comparison.to_csv(TABLES_DIR / "feature_selection_results.csv", index=False)
    feature_comparison.to_csv(TABLES_DIR / "feature_selection_comparison.csv", index=False)

    metrics, _ = evaluate_dataset(str(data_path), str(MAIN_MODEL_PATH))
    pd.DataFrame([metrics]).to_csv(TABLES_DIR / "evaluation_metrics.csv", index=False)
    create_latency_table(data_path)

    plot_confusion_matrix(data_path)
    plot_feature_importance()
    plot_feature_selection_comparison(feature_comparison)
    plot_system_architecture_placeholder()
    plot_streamlit_interface_placeholder()
    print("Report assets generated in results/ and report/figures/.")


if __name__ == "__main__":
    main()
