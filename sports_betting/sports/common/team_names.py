"""Shared team-name normalization across injury, odds, and merge pipelines."""

from __future__ import annotations

import unicodedata


def normalize_team_name(name: str) -> str:
    if not isinstance(name, str):
        return ""

    normalized = unicodedata.normalize("NFKD", name)
    name_ascii = normalized.encode("ascii", "ignore").decode("utf-8")
    name_ascii = name_ascii.lower().strip()
    cleaned = " ".join("".join(ch if ch.isalnum() else " " for ch in name_ascii).split())

    replacements = {
        "trail blazers": "blazers",
        "la clippers": "los angeles clippers",
        "la lakers": "los angeles lakers",
        "ny knicks": "new york knicks",
        "okc thunder": "oklahoma city thunder",
        "utah mammoth": "utah hockey club",
        "montreal canadiens": "canadiens",
    }
    for source, target in replacements.items():
        cleaned = cleaned.replace(source, target)

    return cleaned
