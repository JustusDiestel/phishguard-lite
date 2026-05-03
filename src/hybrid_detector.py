"""Hybrid risk scoring engine for PhishGuard-Lite."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.classical_models import MAIN_MODEL_PATH, load_model_bundle, predict_url_classical
from src.explanation import generate_explanations, rule_based_score
from src.feature_extraction import extract_url_features
from src.hf_model import HFResult, HuggingFaceURLClassifier


class HybridPhishingDetector:
    """Combine classical ML, optional Hugging Face inference, and rules."""

    def __init__(
        self,
        model_path: str | Path = MAIN_MODEL_PATH,
        use_hf: bool = False,
        hf_model_names: list[str] | None = None,
    ) -> None:
        self.model_bundle = load_model_bundle(model_path)
        self.hf_classifier = HuggingFaceURLClassifier(model_names=hf_model_names, enabled=use_hf)

    @staticmethod
    def _risk_level(score: float) -> str:
        if score < 0.40:
            return "LOW"
        if score < 0.70:
            return "MEDIUM"
        return "HIGH"

    @staticmethod
    def _combine_scores(classical_score: float, hf_result: HFResult, rule_score: float) -> tuple[float, str]:
        if hf_result.available:
            final_score = 0.40 * classical_score + 0.40 * hf_result.score + 0.20 * rule_score
            formula = "0.40 * classical_ml_score + 0.40 * hf_score + 0.20 * rule_score"
        else:
            final_score = 0.70 * classical_score + 0.30 * rule_score
            formula = "0.70 * classical_ml_score + 0.30 * rule_score"
        return round(final_score, 4), formula

    def analyze_url(self, url: str) -> dict[str, Any]:
        """Return a complete single-URL analysis result."""

        features = extract_url_features(url)
        classical = predict_url_classical(url, self.model_bundle)
        rules = rule_based_score(features)
        hf_result = self.hf_classifier.predict(url)
        final_score, formula = self._combine_scores(float(classical["score"]), hf_result, rules)
        prediction = "phishing" if final_score >= 0.50 else "legitimate"

        return {
            "url": url,
            "prediction": prediction,
            "risk_level": self._risk_level(final_score),
            "confidence": round(final_score if prediction == "phishing" else 1.0 - final_score, 4),
            "final_score": final_score,
            "formula": formula,
            "classical_label": classical["label"],
            "classical_score": classical["score"],
            "hf_label": hf_result.label,
            "hf_score": hf_result.score,
            "hf_available": hf_result.available,
            "hf_model": hf_result.model_name,
            "hf_error": hf_result.error,
            "rule_score": rules,
            "explanations": generate_explanations(url, features),
            "features": features,
        }
