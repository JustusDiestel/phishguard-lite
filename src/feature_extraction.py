"""Handcrafted URL feature extraction for phishing detection."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urlparse

import pandas as pd

try:
    import tldextract
except Exception:  # pragma: no cover - optional dependency fallback
    tldextract = None

_TLD_EXTRACTOR = (
    tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None) if tldextract is not None else None
)


SUSPICIOUS_TLDS = {
    "zip",
    "mov",
    "top",
    "xyz",
    "gq",
    "tk",
    "ml",
    "cf",
    "work",
    "click",
    "country",
    "stream",
}

URL_SHORTENERS = {
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "goo.gl",
    "ow.ly",
    "is.gd",
    "buff.ly",
    "cutt.ly",
    "rebrand.ly",
    "shorturl.at",
}

KEYWORDS = [
    "login",
    "verify",
    "secure",
    "account",
    "update",
    "bank",
    "paypal",
    "google",
    "microsoft",
    "apple",
    "amazon",
]

FEATURE_NAMES = [
    "url_length",
    "domain_length",
    "path_length",
    "query_length",
    "num_dots",
    "num_hyphens",
    "num_digits",
    "num_special_chars",
    "num_slashes",
    "subdomain_count",
    "path_depth",
    "has_https",
    "has_ip_address",
    "has_at_symbol",
    "has_double_slash_redirect",
    "has_suspicious_tld",
    "has_url_shortener",
    "contains_login",
    "contains_verify",
    "contains_secure",
    "contains_account",
    "contains_update",
    "contains_bank",
    "contains_paypal",
    "contains_google",
    "contains_microsoft",
    "contains_apple",
    "contains_amazon",
    "entropy_score",
]


@dataclass(frozen=True)
class ParsedURL:
    """Normalized URL parse result used by the feature extractor."""

    raw_url: str
    normalized_url: str
    scheme: str
    netloc: str
    domain: str
    suffix: str
    registered_domain: str
    subdomain: str
    path: str
    query: str


def normalize_url(url: str) -> str:
    """Return a parseable URL, adding an HTTP scheme when missing."""

    cleaned = str(url or "").strip()
    if not cleaned:
        return ""
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", cleaned):
        return f"http://{cleaned}"
    return cleaned


def parse_url(url: str) -> ParsedURL:
    """Parse a URL robustly without raising on malformed input."""

    normalized = normalize_url(url)
    try:
        parsed = urlparse(normalized)
    except Exception:
        parsed = urlparse("")

    netloc = parsed.netloc.lower()
    if "@" in netloc:
        netloc = netloc.split("@")[-1]
    netloc = netloc.split(":")[0].strip("[]")

    if _TLD_EXTRACTOR is not None:
        extracted = _TLD_EXTRACTOR(normalized)
        domain = extracted.domain.lower()
        suffix = extracted.suffix.lower()
        subdomain = extracted.subdomain.lower()
        registered_domain = ".".join(part for part in [domain, suffix] if part)
    else:
        parts = [part for part in netloc.split(".") if part]
        suffix = parts[-1] if len(parts) >= 2 else ""
        domain = parts[-2] if len(parts) >= 2 else (parts[0] if parts else "")
        subdomain = ".".join(parts[:-2]) if len(parts) > 2 else ""
        registered_domain = ".".join(parts[-2:]) if len(parts) >= 2 else netloc

    return ParsedURL(
        raw_url=str(url or ""),
        normalized_url=normalized,
        scheme=(parsed.scheme or "").lower(),
        netloc=netloc,
        domain=domain,
        suffix=suffix,
        registered_domain=registered_domain,
        subdomain=subdomain,
        path=parsed.path or "",
        query=parsed.query or "",
    )


def shannon_entropy(text: str) -> float:
    """Compute Shannon entropy for a string."""

    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    return -sum((count / length) * math.log2(count / length) for count in counts.values())


def has_ip_address(host: str) -> int:
    """Return 1 when the host looks like an IPv4 address."""

    ipv4_pattern = r"^(?:\d{1,3}\.){3}\d{1,3}$"
    if not re.match(ipv4_pattern, host):
        return 0
    parts = host.split(".")
    return int(all(0 <= int(part) <= 255 for part in parts))


def count_special_chars(url: str) -> int:
    """Count non-alphanumeric URL characters."""

    return sum(1 for char in url if not char.isalnum())


def extract_url_features(url: str) -> dict[str, float]:
    """Extract handcrafted numeric features from a URL."""

    parsed = parse_url(url)
    raw_lower = parsed.raw_url.lower()
    normalized_lower = parsed.normalized_url.lower()
    domain_text = parsed.netloc
    path_segments = [segment for segment in parsed.path.split("/") if segment]
    subdomain_parts = [part for part in parsed.subdomain.split(".") if part]

    features: dict[str, float] = {
        "url_length": len(parsed.raw_url),
        "domain_length": len(domain_text),
        "path_length": len(parsed.path),
        "query_length": len(parsed.query),
        "num_dots": parsed.raw_url.count("."),
        "num_hyphens": parsed.raw_url.count("-"),
        "num_digits": sum(char.isdigit() for char in parsed.raw_url),
        "num_special_chars": count_special_chars(parsed.raw_url),
        "num_slashes": parsed.raw_url.count("/"),
        "subdomain_count": len(subdomain_parts),
        "path_depth": len(path_segments),
        "has_https": int(parsed.scheme == "https"),
        "has_ip_address": has_ip_address(domain_text),
        "has_at_symbol": int("@" in parsed.raw_url),
        "has_double_slash_redirect": int("//" in parsed.path or "//" in parsed.query),
        "has_suspicious_tld": int(parsed.suffix in SUSPICIOUS_TLDS),
        "has_url_shortener": int(parsed.registered_domain in URL_SHORTENERS),
        "entropy_score": round(shannon_entropy(parsed.raw_url), 4),
    }

    for keyword in KEYWORDS:
        features[f"contains_{keyword}"] = int(keyword in normalized_lower or keyword in raw_lower)

    return {feature_name: features.get(feature_name, 0) for feature_name in FEATURE_NAMES}


def extract_features_dataframe(urls: Iterable[str]) -> pd.DataFrame:
    """Extract features for many URLs and return a dataframe."""

    rows = [extract_url_features(url) for url in urls]
    return pd.DataFrame(rows, columns=FEATURE_NAMES)


def extract_dataset_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append handcrafted features to a dataframe containing URL and label columns."""

    features = extract_features_dataframe(df["url"].astype(str).tolist())
    output = pd.concat([df[["url", "label"]].reset_index(drop=True), features], axis=1)
    return output
