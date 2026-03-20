"""Injury pipeline script wrappers for daily runs."""

from __future__ import annotations

from sports_betting.data.injuries.fetch_espn_injuries import run_injury_pipeline

__all__ = ["run_injury_pipeline"]
