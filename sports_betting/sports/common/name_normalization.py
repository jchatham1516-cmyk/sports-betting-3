"""Shared name normalization helpers for source matching."""

from __future__ import annotations

import re
import unicodedata

_SUFFIX_RE = re.compile(r"\b(?:jr|sr|ii|iii|iv|v)\b", re.IGNORECASE)
_PUNCT_RE = re.compile(r"[^a-z0-9\s]")
_WS_RE = re.compile(r"\s+")


def normalize_person_name(name: object) -> str:
    """Normalize person names consistently for cross-source lookups.

    The normalization intentionally removes accents, punctuation, common suffixes,
    and duplicate whitespace so scraped names and stat-source names share keys.
    """

    text = str(name or "").strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = _PUNCT_RE.sub(" ", text)
    text = _SUFFIX_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()
