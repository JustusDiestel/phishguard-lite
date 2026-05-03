"""Human-readable explanations for PhishGuard-Lite predictions."""

from __future__ import annotations

from src.feature_extraction import parse_url


BRAND_DOMAINS = {
    "paypal": "paypal.com",
    "google": "google.com",
    "microsoft": "microsoft.com",
    "apple": "apple.com",
    "amazon": "amazon.com",
}


def _brand_misuse_messages(url: str, features: dict[str, float]) -> list[str]:
    parsed = parse_url(url)
    messages: list[str] = []
    registered_domain = parsed.registered_domain.lower()
    for brand, official_domain in BRAND_DOMAINS.items():
        if features.get(f"contains_{brand}", 0) and registered_domain != official_domain:
            messages.append(
                f"URL contains the brand name '{brand}' outside the official domain {official_domain}."
            )
    return messages


def generate_explanations(url: str, features: dict[str, float]) -> list[str]:
    """Generate short explanations based on activated URL features."""

    explanations: list[str] = []

    if features.get("url_length", 0) > 90:
        explanations.append("URL is unusually long.")
    if features.get("num_hyphens", 0) >= 3:
        explanations.append("Domain or path contains many hyphens.")
    if features.get("num_digits", 0) >= 8:
        explanations.append("URL contains many digits, which can indicate obfuscation.")
    if features.get("subdomain_count", 0) >= 3:
        explanations.append("URL uses many subdomains.")
    if features.get("has_https", 0) == 0:
        explanations.append("URL does not use HTTPS.")
    if features.get("has_ip_address", 0):
        explanations.append("URL contains an IP address instead of a normal domain.")
    if features.get("has_at_symbol", 0):
        explanations.append("URL contains an @ symbol, which can hide the true destination.")
    if features.get("has_double_slash_redirect", 0):
        explanations.append("URL path or query contains a double-slash redirect pattern.")
    if features.get("has_suspicious_tld", 0):
        explanations.append("URL uses a top-level domain often observed in abuse datasets.")
    if features.get("has_url_shortener", 0):
        explanations.append("URL appears to use a URL shortener.")
    if features.get("entropy_score", 0) > 4.5:
        explanations.append("URL has unusually high character entropy.")

    credential_keywords = ["login", "verify", "secure", "account", "update", "bank"]
    for keyword in credential_keywords:
        if features.get(f"contains_{keyword}", 0):
            explanations.append(f"URL contains credential- or account-related keyword: {keyword}.")

    explanations.extend(_brand_misuse_messages(url, features))

    if not explanations:
        explanations.append("No strong suspicious URL indicators were triggered.")
    return explanations


def rule_based_score(features: dict[str, float]) -> float:
    """Compute a transparent phishing risk score from handcrafted indicators."""

    score = 0.0
    weights = {
        "has_ip_address": 0.18,
        "has_at_symbol": 0.16,
        "has_double_slash_redirect": 0.14,
        "has_suspicious_tld": 0.11,
        "has_url_shortener": 0.10,
        "contains_login": 0.07,
        "contains_verify": 0.07,
        "contains_secure": 0.05,
        "contains_account": 0.05,
        "contains_update": 0.05,
        "contains_bank": 0.08,
    }
    for feature_name, weight in weights.items():
        score += weight if features.get(feature_name, 0) else 0.0

    if features.get("url_length", 0) > 90:
        score += 0.08
    if features.get("num_hyphens", 0) >= 3:
        score += 0.06
    if features.get("num_digits", 0) >= 8:
        score += 0.06
    if features.get("subdomain_count", 0) >= 3:
        score += 0.06
    if features.get("entropy_score", 0) > 4.5:
        score += 0.06
    if features.get("has_https", 0) == 0:
        score += 0.05

    return round(min(score, 1.0), 4)
