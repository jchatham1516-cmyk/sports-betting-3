from __future__ import annotations

from datetime import datetime, timezone

from sports_betting.sports.common import game_filters


class _MorningUTC(datetime):
    @classmethod
    def utcnow(cls) -> datetime:
        return cls(2025, 1, 15, 5, 0, 0)


class _EveningUTC(datetime):
    @classmethod
    def utcnow(cls) -> datetime:
        return cls(2025, 1, 15, 18, 0, 0)


def test_filter_games_for_today_keeps_previous_noon_to_current_noon_when_before_noon(monkeypatch):
    monkeypatch.setattr(game_filters, "datetime", _MorningUTC)

    raw_games = [
        {"id": "in_start", "commence_time": "2025-01-14T12:00:00Z"},
        {"id": "in_middle", "commence_time": "2025-01-15T04:59:00Z"},
        {"id": "in_end", "commence_time": "2025-01-15T12:00:00Z"},
        {"id": "out_before", "commence_time": "2025-01-14T11:59:00Z"},
        {"id": "out_after", "commence_time": "2025-01-15T12:01:00Z"},
    ]

    filtered = game_filters.filter_games_for_today(raw_games)

    assert [g["id"] for g in filtered] == ["in_start", "in_middle", "in_end"]


def test_filter_games_for_today_keeps_current_noon_to_next_noon_when_after_noon(monkeypatch):
    monkeypatch.setattr(game_filters, "datetime", _EveningUTC)

    raw_games = [
        {"id": "in_start", "commence_time": "2025-01-15T12:00:00Z"},
        {"id": "in_middle", "commence_time": "2025-01-16T03:00:00Z"},
        {"id": "in_end", "commence_time": "2025-01-16T12:00:00Z"},
        {"id": "out_before", "commence_time": "2025-01-15T11:59:00Z"},
        {"id": "out_after", "commence_time": "2025-01-16T12:01:00Z"},
    ]

    filtered = game_filters.filter_games_for_today(raw_games)

    assert [g["id"] for g in filtered] == ["in_start", "in_middle", "in_end"]


def test_current_sports_day_window_returns_utc_bounds(monkeypatch):
    monkeypatch.setattr(game_filters, "datetime", _EveningUTC)

    start, end = game_filters.current_sports_day_window()

    assert start == datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    assert end == datetime(2025, 1, 16, 12, 0, 0, tzinfo=timezone.utc)
