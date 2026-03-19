from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sports_betting.sports.common import game_filters


class _FixedUTC(datetime):
    @classmethod
    def utcnow(cls) -> datetime:
        return cls(2025, 1, 15, 18, 0, 0)


def test_filter_games_window_keeps_only_games_between_now_and_next_18_hours(monkeypatch):
    monkeypatch.setattr(game_filters, "datetime", _FixedUTC)

    raw_games = [
        {"id": "in_start", "commence_time": "2025-01-15T18:00:00Z"},
        {"id": "in_middle", "commence_time": "2025-01-16T06:00:00Z"},
        {"id": "in_end", "commence_time": "2025-01-16T12:00:00Z"},
        {"id": "out_before", "commence_time": "2025-01-15T17:59:00Z"},
        {"id": "out_after", "commence_time": "2025-01-16T12:01:00Z"},
    ]

    filtered = game_filters.filter_games_window(raw_games)

    assert [g["id"] for g in filtered] == ["in_start", "in_middle", "in_end"]


def test_filter_games_for_today_aliases_filter_games_window(monkeypatch):
    monkeypatch.setattr(game_filters, "datetime", _FixedUTC)

    raw_games = [
        {"id": "in_window", "commence_time": "2025-01-16T03:00:00Z"},
        {"id": "out_window", "commence_time": "2025-01-16T13:00:00Z"},
    ]

    assert game_filters.filter_games_for_today(raw_games) == game_filters.filter_games_window(raw_games)


def test_current_sports_day_window_returns_now_to_now_plus_18h(monkeypatch):
    monkeypatch.setattr(game_filters, "datetime", _FixedUTC)

    start, end = game_filters.current_sports_day_window()

    expected_start = datetime(2025, 1, 15, 18, 0, 0, tzinfo=timezone.utc)
    assert start == expected_start
    assert end == expected_start + timedelta(hours=18)
