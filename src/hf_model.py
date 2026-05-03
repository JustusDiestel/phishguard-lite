"""Lazy Hugging Face URL classification wrapper with graceful fallback."""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_HF_MODELS = [
    "CrabInHoney/urlbert-tiny-v4-phishing-classifier",
    "CrabInHoney/urlbert-tiny-v4-malicious-url-classifier",
    "Eason918/malicious-url-detector-v2",
    "darshan8950/phishing_url_detection_BERT",
]


@dataclass
class HFResult:
    """Structured result returned by the Hugging Face wrapper."""

    label: str
    score: float
    available: bool
    model_name: str | None = None
    error: str | None = None


class HuggingFaceURLClassifier:
    """Small wrapper around a pretrained Hugging Face text classifier.

    The model is loaded only when ``predict`` is first called. If loading or
    inference fails, callers receive an unavailable result and the rest of the
    application can continue with classical ML and rule-based scoring.
    """

    def __init__(self, model_names: list[str] | None = None, enabled: bool = True) -> None:
        self.model_names = model_names or DEFAULT_HF_MODELS
        self.enabled = enabled
        self._pipeline = None
        self.loaded_model_name: str | None = None
        self.last_error: str | None = None

    def _load(self) -> None:
        if not self.enabled or self._pipeline is not None:
            return
        try:
            from transformers import pipeline
        except Exception as exc:
            self.last_error = f"transformers is unavailable: {exc}"
            self.enabled = False
            return

        for model_name in self.model_names:
            try:
                self._pipeline = pipeline(
                    "text-classification",
                    model=model_name,
                    tokenizer=model_name,
                    truncation=True,
                )
                self.loaded_model_name = model_name
                self.last_error = None
                return
            except Exception as exc:
                self.last_error = f"Could not load {model_name}: {exc}"
        self.enabled = False

    @staticmethod
    def _normalize_result(raw_result: dict[str, object]) -> tuple[str, float]:
        label_text = str(raw_result.get("label", "")).lower()
        confidence = float(raw_result.get("score", 0.0))

        phishing_tokens = ["phish", "malicious", "bad", "unsafe", "label_1", "1"]
        legitimate_tokens = ["legit", "benign", "safe", "good", "label_0", "0"]

        if any(token in label_text for token in phishing_tokens):
            return "phishing", confidence
        if any(token in label_text for token in legitimate_tokens):
            return "legitimate", 1.0 - confidence

        # Unknown labels are treated conservatively as an uncertain phishing score.
        return "unknown", 0.5

    def predict(self, url: str) -> HFResult:
        """Classify a URL string with the pretrained model if available."""

        if not self.enabled:
            return HFResult(label="unavailable", score=0.5, available=False, error="Hugging Face disabled.")

        self._load()
        if self._pipeline is None:
            return HFResult(
                label="unavailable",
                score=0.5,
                available=False,
                model_name=self.loaded_model_name,
                error=self.last_error,
            )

        try:
            result = self._pipeline(url)
            raw_result = result[0] if isinstance(result, list) else result
            label, phishing_score = self._normalize_result(raw_result)
            return HFResult(
                label=label,
                score=round(float(max(0.0, min(phishing_score, 1.0))), 4),
                available=True,
                model_name=self.loaded_model_name,
            )
        except Exception as exc:
            self.last_error = str(exc)
            return HFResult(
                label="unavailable",
                score=0.5,
                available=False,
                model_name=self.loaded_model_name,
                error=str(exc),
            )
