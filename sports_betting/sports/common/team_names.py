"""Shared team-name normalization across injury, odds, and merge pipelines."""

from __future__ import annotations

import unicodedata


def normalize_team_name(name: object) -> object:
    if not isinstance(name, str):
        return name

    normalized = unicodedata.normalize("NFKD", name)
    name_ascii = normalized.encode("ascii", "ignore").decode("utf-8")
    cleaned = " ".join("".join(ch if ch.isalnum() else " " for ch in name_ascii).split()).lower().strip()

    replacements = {
        "portland blazers": "portland trail blazers",
        "st louis blues": "st. louis blues",
        "st louis cardinals": "st. louis cardinals",
        "canadiens": "montreal canadiens",
        "utah mammoth": "utah hockey club",
        "la clippers": "los angeles clippers",
        "ny knicks": "new york knicks",
        "ny rangers": "new york rangers",
        "la lakers": "los angeles lakers",
        "okc thunder": "oklahoma city thunder",
    }
    return replacements.get(cleaned, cleaned)
