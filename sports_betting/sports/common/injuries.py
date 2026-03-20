"""Injury ingestion and sport-aware impact scoring."""

from __future__ import annotations

import os

import json
import logging
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

LOGGER = logging.getLogger(__name__)

STATUS_MAP = {
    "out": "out",
    "doubtful": "doubtful",
    "questionable": "questionable",
    "probable": "probable",
    "inactive": "inactive",
    "ir": "ir",
    "day-to-day": "day-to-day",
    "day to day": "day-to-day",
}

STATUS_WEIGHT = {
    "out": 1.0,
    "ir": 1.0,
    "inactive": 1.0,
    "doubtful": 0.75,
    "questionable": 0.45,
    "day-to-day": 0.35,
    "probable": 0.15,
}

ROLE_WEIGHTS = {
    "superstar": 4.0,
    "all-star": 3.0,
    "elite qb": 3.0,
    "starting goalie": 3.0,
    "starter": 2.0,
    "rotation": 1.0,
    "bench": 0.5,
    "depth": 0.5,
}

NFL_OFF_POSITIONS = {"qb", "rb", "wr", "te", "ol", "lt", "rt", "c", "g"}
NFL_DEF_POSITIONS = {"de", "dt", "lb", "cb", "s", "edge", "dl"}


@dataclass
class InjurySource:
    sport: str
    source_name: str
    url: str


def normalize_status(status: str) -> str:
    return STATUS_MAP.get(str(status or "").strip().lower(), "questionable")


def normalize_team_name(name: object) -> str:
    normalized = unicodedata.normalize("NFKD", str(name or ""))
    ascii_name = normalized.encode("ascii", "ignore").decode("utf-8")
    return ascii_name.lower().strip()


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _timestamp_to_iso(value: object) -> str:
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
    return datetime.now(tz=timezone.utc).isoformat()


def _role_weight(role: str, expected_minutes_or_role: object) -> float:
    role_str = str(role or "").strip().lower()
    if role_str in ROLE_WEIGHTS:
        return ROLE_WEIGHTS[role_str]
    minutes = _to_float(expected_minutes_or_role, default=0.0)
    if minutes >= 35:
        return 4.0
    if minutes >= 30:
        return 3.0
    if minutes >= 24:
        return 2.0
    if minutes >= 14:
        return 1.0
    return 0.5


def _sport_multiplier(sport: str, position: str, role: str) -> tuple[float, str]:
    pos = str(position or "").strip().lower()
    role_s = str(role or "").strip().lower()
    unit = "offense"
    multiplier = 1.0

    if sport == "nba":
        if pos in {"pg", "sg", "sf", "pf", "c"}:
            unit = "offense"
        multiplier = 1.15 if role_s in {"superstar", "all-star", "starter"} else 1.0
    elif sport == "nfl":
        if pos in NFL_DEF_POSITIONS:
            unit = "defense"
        if pos == "qb":
            multiplier = 1.35
        elif pos in {"ol", "lt", "rt", "edge", "cb", "de"}:
            multiplier = 1.2
    elif sport == "nhl":
        if "g" in pos or "goalie" in role_s:
            multiplier = 1.35
            unit = "defense"
        elif pos in {"d", "ld", "rd"}:
            unit = "defense"
        if role_s in {"superstar", "all-star", "starting goalie"}:
            multiplier = max(multiplier, 1.2)
    return multiplier, unit


def _fetch_json(url: str) -> list[dict]:
    with urlopen(url, timeout=20) as response:  # nosec B310
        payload = json.loads(response.read().decode("utf-8"))
    if isinstance(payload, dict):
        data = payload.get("data") or payload.get("injuries") or payload.get("players") or []
    else:
        data = payload
    return data if isinstance(data, list) else []


def _fetch_primary_source(sport: str, data_root: Path) -> pd.DataFrame:
    key = ("INJURY_API_KEY", "SPORTSDATA_API_KEY")
    token = next((os.getenv(k, "").strip() for k in key if os.getenv(k, "").strip()), "")
    if not token:
        return pd.DataFrame()

    source = InjurySource(
        sport=sport,
        source_name="sportsdata-io",
        url=f"https://api.sportsdata.io/v3/{sport}/scores/json/Injuries?{urlencode({'key': token})}",
    )
    try:
        records = _fetch_json(source.url)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        LOGGER.warning("[%s] Injury primary source failed: %s", sport.upper(), exc)
        return pd.DataFrame()

    normalized = []
    for rec in records:
        normalized.append(
            {
                "player_name": rec.get("Name") or rec.get("player_name") or rec.get("Player") or "",
                "team": rec.get("Team") or rec.get("team") or "",
                "status": normalize_status(rec.get("Status") or rec.get("status")),
                "position": rec.get("Position") or rec.get("position") or "",
                "expected_minutes_or_role": rec.get("ExpectedMinutes") or rec.get("Role") or rec.get("expected_minutes_or_role") or "",
                "source_timestamp": _timestamp_to_iso(rec.get("Updated") or rec.get("LastUpdated") or rec.get("source_timestamp")),
                "source": source.source_name,
            }
        )
    return pd.DataFrame(normalized)


def _normalize_team_name(team: object) -> str:
    return normalize_team_name(team)


def _load_json_injury_fallback(data_root: Path) -> pd.DataFrame:
    candidate_paths = [
        data_root / "injuries" / "injuries.json",
        Path("sports_betting/data/injuries/injuries.json"),
        data_root / "inputs" / "injuries.json",
        Path("data/inputs/injuries.json"),
    ]
    path = next((p for p in candidate_paths if p.exists()), None)
    if path is None:
        return pd.DataFrame()

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    rows: list[dict[str, str]] = []
    if isinstance(payload, dict):
        for team, players in payload.items():
            if isinstance(players, dict):
                for player, status in players.items():
                    rows.append(
                        {
                            "team": _normalize_team_name(team),
                            "player_name": str(player or "").strip(),
                            "status": normalize_status(status),
                            "position": "",
                            "expected_minutes_or_role": "",
                            "source_timestamp": datetime.now(tz=timezone.utc).isoformat(),
                            "source": "espn-json",
                        }
                    )
            elif isinstance(players, list):
                for rec in players:
                    if not isinstance(rec, dict):
                        continue
                    rows.append(
                        {
                            "team": _normalize_team_name(rec.get("team", team)),
                            "player_name": str(rec.get("player") or rec.get("player_name") or rec.get("name") or "").strip(),
                            "status": normalize_status(rec.get("status")),
                            "position": str(rec.get("position", "")).strip(),
                            "expected_minutes_or_role": str(rec.get("expected_minutes_or_role") or rec.get("role") or "").strip(),
                            "source_timestamp": _timestamp_to_iso(rec.get("source_timestamp")),
                            "source": "espn-json",
                        }
                    )
    elif isinstance(payload, list):
        for rec in payload:
            if not isinstance(rec, dict):
                continue
            rows.append(
                {
                    "team": _normalize_team_name(rec.get("team")),
                    "player_name": str(rec.get("player") or rec.get("player_name") or rec.get("name") or "").strip(),
                    "status": normalize_status(rec.get("status")),
                    "position": str(rec.get("position", "")).strip(),
                    "expected_minutes_or_role": str(rec.get("expected_minutes_or_role") or rec.get("role") or "").strip(),
                    "source_timestamp": _timestamp_to_iso(rec.get("source_timestamp")),
                    "source": "espn-json",
                }
            )
    return pd.DataFrame(rows)


def load_injury_frame(sport: str, data_root: Path) -> pd.DataFrame:
    """Get normalized injuries with API->CSV fallback and zero-crash behavior."""
    external = data_root / "external"
    external.mkdir(parents=True, exist_ok=True)
    csv_path = external / f"{sport}_injuries.csv"

    frame = _fetch_primary_source(sport, data_root)
    if frame.empty and csv_path.exists():
        frame = pd.read_csv(csv_path)
    if frame.empty:
        frame = _load_json_injury_fallback(data_root)

    if frame.empty:
        return pd.DataFrame(columns=["player_name", "team", "status", "position", "expected_minutes_or_role", "source_timestamp", "source"])

    frame.columns = [c.strip().lower() for c in frame.columns]
    rename_map = {"player": "player_name", "name": "player_name"}
    frame = frame.rename(columns=rename_map)
    for col in ["player_name", "team", "status", "position", "expected_minutes_or_role", "source_timestamp"]:
        if col not in frame.columns:
            frame[col] = ""
    for legacy_col in ["qb", "starting_goalie", "is_starter", "is_star", "impact_rating", "unit"]:
        if legacy_col not in frame.columns:
            frame[legacy_col] = 0 if legacy_col != "unit" else ""
    frame["team"] = frame["team"].map(_normalize_team_name)
    frame["status"] = frame["status"].map(normalize_status)
    frame["source_timestamp"] = frame["source_timestamp"].map(_timestamp_to_iso)
    frame["source"] = frame.get("source", "fallback")

    # Persist normalized copy for auditability.
    frame.to_csv(csv_path, index=False)
    return frame


def summarize_team_injuries(frame: pd.DataFrame, sport: str) -> pd.DataFrame:
    if frame.empty:
        cols = [
            "team",
            "injury_impact",
            "starter_out_count",
            "star_player_out",
            "qb_out_flag",
            "starting_goalie_out_flag",
            "offensive_injury_weight",
            "defensive_injury_weight",
            "injury_confidence_score",
            "injury_data_stale",
        ]
        return pd.DataFrame(columns=cols)

    work = frame.copy()
    work["status"] = work["status"].map(normalize_status)
    work["status_weight"] = work["status"].map(STATUS_WEIGHT).fillna(0.4)
    work["impact_rating"] = pd.to_numeric(work.get("impact_rating", 0.0), errors="coerce").fillna(0.0)
    work["role_weight"] = work.apply(
        lambda r: max(_role_weight(str(r.get("expected_minutes_or_role", "")), r.get("expected_minutes_or_role", 0.0)), float(r.get("impact_rating", 0.0))), axis=1
    )

    multipliers_units = work.apply(
        lambda r: _sport_multiplier(sport=sport, position=str(r.get("position", "")), role=str(r.get("expected_minutes_or_role", ""))),
        axis=1,
    )
    work["sport_multiplier"] = [m for m, _ in multipliers_units]
    work["unit"] = [u for _, u in multipliers_units]
    legacy_unit = work.get("unit", "")
    work["unit"] = pd.Series(legacy_unit, index=work.index).astype(str).str.lower().where(pd.Series(legacy_unit, index=work.index).astype(str).str.len() > 0, work["unit"])
    work["weighted_impact"] = work["role_weight"] * work["status_weight"] * work["sport_multiplier"]

    work["status_out_like"] = work["status"].isin({"out", "inactive", "ir", "doubtful"}).astype(int)
    work["star_role"] = ((work["role_weight"] >= 3.0) | (pd.to_numeric(work.get("is_star", 0), errors="coerce").fillna(0) > 0)).astype(int)
    work["starter_role"] = ((work["role_weight"] >= 2.0) | (pd.to_numeric(work.get("is_starter", 0), errors="coerce").fillna(0) > 0)).astype(int)
    work["qb_flag"] = (((work["position"].astype(str).str.lower() == "qb") | (pd.to_numeric(work.get("qb", 0), errors="coerce").fillna(0) > 0)) & (work["status_out_like"] == 1)).astype(int)
    work["goalie_flag"] = (
        (work["position"].astype(str).str.lower().isin({"g", "goalie"}) | work["expected_minutes_or_role"].astype(str).str.lower().eq("starting goalie") | (pd.to_numeric(work.get("starting_goalie", 0), errors="coerce").fillna(0) > 0))
        & (work["status_out_like"] == 1)
    ).astype(int)

    now = datetime.now(tz=timezone.utc)
    age_hours = pd.to_datetime(work["source_timestamp"], errors="coerce", utc=True)
    stale_flag = ((now - age_hours) > timedelta(hours=30)).fillna(True)
    work["stale_flag"] = stale_flag.astype(int)

    summary = (
        work.groupby("team", dropna=False)
        .agg(
            injury_impact=("weighted_impact", "sum"),
            starter_out_count=("starter_role", lambda s: int((s & work.loc[s.index, "status_out_like"]).sum())),
            star_player_out=("star_role", lambda s: int(((s == 1) & (work.loc[s.index, "status_out_like"] == 1)).any())),
            qb_out_flag=("qb_flag", "max"),
            starting_goalie_out_flag=("goalie_flag", "max"),
            offensive_injury_weight=("weighted_impact", lambda s: float(s[work.loc[s.index, "unit"] == "offense"].sum())),
            defensive_injury_weight=("weighted_impact", lambda s: float(s[work.loc[s.index, "unit"] == "defense"].sum())),
            injury_confidence_score=("stale_flag", lambda s: float(max(0.0, 1.0 - s.mean()))),
            injury_data_stale=("stale_flag", "max"),
        )
        .reset_index()
    )

    if sport != "nfl":
        summary["qb_out_flag"] = 0
    if sport != "nhl":
        summary["starting_goalie_out_flag"] = 0
    return summary
