"""Train PhishGuard-Lite classical ML models and feature selectors."""

from __future__ import annotations

import argparse

from src.classical_models import train_models
from src.feature_selection import select_features
from src.utils import DATA_DIR, MODEL_DIR, TABLES_DIR, ensure_directories


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PhishGuard-Lite models.")
    parser.add_argument("--data", default=str(DATA_DIR / "sample_urls.csv"), help="CSV with url,label columns.")
    parser.add_argument("--models-dir", default=str(MODEL_DIR), help="Directory for saved models.")
    args = parser.parse_args()

    ensure_directories()
    _, model_comparison = train_models(args.data, args.models_dir)
    feature_comparison, _ = select_features(args.data, args.models_dir)

    model_comparison.to_csv(TABLES_DIR / "model_comparison_table.csv", index=False)
    feature_comparison.to_csv(TABLES_DIR / "feature_selection_results.csv", index=False)

    print("Training complete.")
    print(model_comparison.to_string(index=False))
    print()
    print(feature_comparison.to_string(index=False))


if __name__ == "__main__":
    main()
