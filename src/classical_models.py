"""Training, persistence, and prediction helpers for classical ML models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.feature_extraction import FEATURE_NAMES, extract_dataset_features, extract_url_features
from src.utils import MODEL_DIR, RANDOM_SEED, ensure_directories, read_url_dataset


MAIN_MODEL_PATH = MODEL_DIR / "phishguard_rf.joblib"


def build_models() -> dict[str, Any]:
    """Create lightweight classical ML models."""

    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=160,
            max_depth=8,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        "Linear SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearSVC(class_weight="balanced", random_state=RANDOM_SEED)),
            ]
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_SEED),
    }


def _phishing_scores(model: Any, x_values: pd.DataFrame) -> np.ndarray:
    """Return class-1 scores for models with or without predict_proba."""

    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_values)[:, 1]
    if hasattr(model, "decision_function"):
        decision = model.decision_function(x_values)
        return 1.0 / (1.0 + np.exp(-decision))
    predictions = model.predict(x_values)
    return np.asarray(predictions, dtype=float)


def train_models(
    dataset_path: str | Path,
    model_dir: str | Path = MODEL_DIR,
    selected_features: list[str] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Train and save classical models from a labeled URL CSV file."""

    ensure_directories()
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    raw_df = read_url_dataset(dataset_path)
    feature_df = extract_dataset_features(raw_df)
    feature_names = selected_features or FEATURE_NAMES
    x_values = feature_df[feature_names]
    y_values = feature_df["label"].astype(int)

    stratify = y_values if y_values.nunique() == 2 and len(y_values) >= 10 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x_values,
        y_values,
        test_size=0.3,
        random_state=RANDOM_SEED,
        stratify=stratify,
    )

    trained: dict[str, Any] = {}
    rows: list[dict[str, float | str]] = []
    for model_name, model in build_models().items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        scores = _phishing_scores(model, x_test)
        trained[model_name] = model
        rows.append(
            {
                "model": model_name,
                "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
                "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
                "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
                "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
                "mean_phishing_score": round(float(np.mean(scores)), 4),
            }
        )

        bundle = {
            "model": model,
            "model_name": model_name,
            "feature_names": feature_names,
            "trained_on": str(Path(dataset_path)),
        }
        safe_name = model_name.lower().replace(" ", "_")
        joblib.dump(bundle, model_dir / f"{safe_name}.joblib")

    main_bundle = {
        "model": trained["Random Forest"],
        "model_name": "Random Forest",
        "feature_names": feature_names,
        "trained_on": str(Path(dataset_path)),
    }
    joblib.dump(main_bundle, model_dir / MAIN_MODEL_PATH.name)
    return trained, pd.DataFrame(rows)


def load_model_bundle(model_path: str | Path = MAIN_MODEL_PATH) -> dict[str, Any]:
    """Load a saved model bundle."""

    return joblib.load(model_path)


def predict_url_classical(url: str, model_bundle: dict[str, Any]) -> dict[str, float | str | int]:
    """Predict phishing probability for a single URL with a saved model bundle."""

    features = extract_url_features(url)
    feature_names = model_bundle["feature_names"]
    x_values = pd.DataFrame([{name: features.get(name, 0) for name in feature_names}])
    model = model_bundle["model"]
    score = float(_phishing_scores(model, x_values)[0])
    prediction = int(score >= 0.5)
    return {
        "label": "phishing" if prediction else "legitimate",
        "prediction": prediction,
        "score": round(score, 4),
    }
