"""MLB pitcher ERA lookups."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import requests

from sports_betting.sports.common.name_normalization import normalize_person_name

MLB_STATS_API_URL = "https://statsapi.mlb.com/api/v1/stats"

# Known static values are a last-resort supplement only. Runtime code still
# treats the neutral 4.20 fallback as non-real and excludes it from coverage.
LAST_PITCHER_ERA_SOURCE_MAP: dict[str, str] = {}

STATIC_PITCHER_ERA: dict[str, float] = {
    "shane baz": 3.45,
    "bailey ober": 3.85,
    "jesus luzardo": 3.70,
    "mackenzie gore": 4.10,
    "eric lauer": 4.25,
    "max meyer": 3.90,
    "jose quintana": 4.05,
    "shota imanaga": 2.95,
    "jake irvin": 4.30,
    "emerson hancock": 4.50,
}


def _coerce_era(value: object) -> float | None:
    try:
        era = float(value)
    except (TypeError, ValueError):
        return None
    if era <= 0 or era > 15:
        return None
    return era


def _fetch_mlb_stats_api_pitcher_eras(season: int) -> dict[str, dict[str, Any]]:
    params = {
        "stats": "season",
        "group": "pitching",
        "playerPool": "ALL",
        "season": season,
        "sportIds": 1,
        "limit": 5000,
        "fields": "stats,splits,player,id,fullName,stat,era",
    }
    try:
        response = requests.get(MLB_STATS_API_URL, params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        print(f"[MLB PITCHER STATS WARNING] MLB Stats API ERA request failed: {exc}")
        return {}

    records: dict[str, dict[str, Any]] = {}
    for stat_block in payload.get("stats", []):
        for split in stat_block.get("splits", []):
            player = split.get("player", {}) if isinstance(split, dict) else {}
            raw_name = player.get("fullName")
            norm_name = normalize_person_name(raw_name)
            if not norm_name:
                continue
            era = _coerce_era((split.get("stat") or {}).get("era"))
            if era is None:
                continue
            records[norm_name] = {
                "name": raw_name,
                "normalized_name": norm_name,
                "era": era,
                "source": f"mlb_stats_api_{season}",
            }
    return records


def build_pitcher_era_records(
    pitchers_dict: dict[str, str] | None = None,
    *,
    season: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Build normalized pitcher-name -> ERA metadata records.

    Names from all sources are normalized with the same helper used by the ESPN
    probable-pitcher scrape, improving cross-source matching without counting
    neutral defaults as real data.
    """

    if season is None:
        season = datetime.now(UTC).year

    records = _fetch_mlb_stats_api_pitcher_eras(season)

    # If a very early season endpoint is sparse, last season's final stats are a
    # better real-data fallback than neutral defaults for known probable names.
    if pitchers_dict and len(records) < len({normalize_person_name(v) for v in pitchers_dict.values() if normalize_person_name(v)}):
        previous_records = _fetch_mlb_stats_api_pitcher_eras(season - 1)
        for key, value in previous_records.items():
            records.setdefault(key, value)

    for name, era in STATIC_PITCHER_ERA.items():
        norm_name = normalize_person_name(name)
        if norm_name and norm_name not in records:
            records[norm_name] = {
                "name": name,
                "normalized_name": norm_name,
                "era": era,
                "source": "static_supplement",
            }
    return records


def get_pitcher_era_source_map() -> dict[str, str]:
    """Return source labels from the most recent ERA map build."""

    return dict(LAST_PITCHER_ERA_SOURCE_MAP)


def build_pitcher_era_map(pitchers_dict: dict[str, str] | None = None) -> dict[str, float]:
    """Build normalized pitcher-name -> ERA mapping for compatibility."""

    global LAST_PITCHER_ERA_SOURCE_MAP
    records = build_pitcher_era_records(pitchers_dict)
    LAST_PITCHER_ERA_SOURCE_MAP = {name: str(record.get("source", "unknown")) for name, record in records.items()}
    return {name: float(record["era"]) for name, record in records.items()}
