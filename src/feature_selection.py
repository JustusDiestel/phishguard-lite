"""Feature selection utilities for PhishGuard-Lite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from src.feature_extraction import FEATURE_NAMES, extract_dataset_features
from src.utils import MODEL_DIR, RANDOM_SEED, ensure_directories, read_url_dataset, write_json


def select_features(
    dataset_path: str | Path,
    output_dir: str | Path = MODEL_DIR,
    top_values: tuple[int, ...] = (20, 10, 5),
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Compare all features with top-k feature subsets and save selected lists."""

    ensure_directories()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = read_url_dataset(dataset_path)
    feature_df = extract_dataset_features(raw_df)
    x_values = feature_df[FEATURE_NAMES]
    y_values = feature_df["label"].astype(int)

    stratify = y_values if y_values.nunique() == 2 and len(y_values) >= 10 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x_values,
        y_values,
        test_size=0.3,
        random_state=RANDOM_SEED,
        stratify=stratify,
    )

    base_model = RandomForestClassifier(
        n_estimators=160,
        max_depth=8,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    base_model.fit(x_train, y_train)

    permutation = permutation_importance(
        base_model,
        x_test,
        y_test,
        n_repeats=15,
        random_state=RANDOM_SEED,
        scoring="f1",
    )
    permutation_ranking = (
        pd.DataFrame(
            {
                "feature": FEATURE_NAMES,
                "permutation_importance": permutation.importances_mean,
            }
        )
        .sort_values("permutation_importance", ascending=False)
        .reset_index(drop=True)
    )

    kbest = SelectKBest(score_func=mutual_info_classif, k="all")
    kbest.fit(x_train, y_train)
    mutual_info_ranking = (
        pd.DataFrame({"feature": FEATURE_NAMES, "mutual_information": kbest.scores_})
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .sort_values("mutual_information", ascending=False)
        .reset_index(drop=True)
    )

    selected: dict[str, list[str]] = {"all_features": FEATURE_NAMES}
    rows = [_evaluate_subset("all_features", FEATURE_NAMES, x_train, x_test, y_train, y_test)]
    for top_k in top_values:
        safe_k = min(top_k, len(FEATURE_NAMES))
        subset = mutual_info_ranking.head(safe_k)["feature"].tolist()
        key = f"top_{safe_k}_mutual_info"
        selected[key] = subset
        rows.append(_evaluate_subset(key, subset, x_train, x_test, y_train, y_test))
        write_json(output_dir / f"selected_features_top_{safe_k}.json", subset)

    write_json(output_dir / "selected_features_all.json", FEATURE_NAMES)
    write_json(output_dir / "feature_selection_summary.json", selected)

    ranking = permutation_ranking.merge(mutual_info_ranking, on="feature", how="outer")
    ranking.to_csv(output_dir / "feature_rankings.csv", index=False)
    comparison = pd.DataFrame(rows)
    return comparison, selected


def _evaluate_subset(
    name: str,
    feature_subset: list[str],
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, float | int | str]:
    model = RandomForestClassifier(
        n_estimators=160,
        max_depth=8,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    model.fit(x_train[feature_subset], y_train)
    predictions = model.predict(x_test[feature_subset])
    return {
        "feature_set": name,
        "num_features": len(feature_subset),
        "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
    }
