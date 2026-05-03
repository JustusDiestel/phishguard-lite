"""Evaluate PhishGuard-Lite on a labeled URL dataset."""

from __future__ import annotations

import argparse
import time

import pandas as pd

from src.classical_models import MAIN_MODEL_PATH, load_model_bundle, predict_url_classical
from src.feature_extraction import extract_url_features
from src.metrics import classification_metrics
from src.utils import DATA_DIR, TABLES_DIR, ensure_directories, read_url_dataset


def evaluate_dataset(data_path: str, model_path: str = str(MAIN_MODEL_PATH)) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate the saved classical model on a labeled CSV dataset."""

    model_bundle = load_model_bundle(model_path)
    df = read_url_dataset(data_path)
    predictions: list[int] = []
    latencies: list[float] = []

    for url in df["url"].astype(str):
        start = time.perf_counter()
        result = predict_url_classical(url, model_bundle)
        latencies.append(time.perf_counter() - start)
        predictions.append(int(result["prediction"]))

    metrics = classification_metrics(df["label"].to_numpy(), predictions)
    metrics["average_latency_seconds"] = round(float(sum(latencies) / len(latencies)), 6)

    details = pd.DataFrame(
        {
            "url": df["url"],
            "label": df["label"],
            "prediction": predictions,
            "latency_seconds": latencies,
        }
    )
    return metrics, details


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved PhishGuard-Lite model.")
    parser.add_argument("--data", default=str(DATA_DIR / "sample_urls.csv"), help="CSV with url,label columns.")
    parser.add_argument("--model", default=str(MAIN_MODEL_PATH), help="Saved model bundle path.")
    args = parser.parse_args()

    ensure_directories()
    metrics, details = evaluate_dataset(args.data, args.model)
    pd.DataFrame([metrics]).to_csv(TABLES_DIR / "evaluation_metrics.csv", index=False)
    details.to_csv(TABLES_DIR / "evaluation_predictions.csv", index=False)

    print(pd.DataFrame([metrics]).to_string(index=False))


if __name__ == "__main__":
    main()
