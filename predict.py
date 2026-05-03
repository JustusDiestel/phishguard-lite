"""Command-line single URL analysis for PhishGuard-Lite."""

from __future__ import annotations

import argparse
import json

from src.hybrid_detector import HybridPhishingDetector


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze one URL with PhishGuard-Lite.")
    parser.add_argument("url", help="URL to analyze.")
    parser.add_argument("--use-hf", action="store_true", help="Enable pretrained Hugging Face model layer.")
    args = parser.parse_args()

    detector = HybridPhishingDetector(use_hf=args.use_hf)
    result = detector.analyze_url(args.url)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
