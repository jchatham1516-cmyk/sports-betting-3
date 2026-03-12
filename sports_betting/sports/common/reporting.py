"""Output and reporting helpers."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from sports_betting.models.entities import BetRecommendation


def save_dataframe(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".csv":
        df.to_csv(output_path, index=False)
    elif output_path.suffix.lower() == ".json":
        df.to_json(output_path, orient="records", indent=2)
    else:
        raise ValueError("Unsupported output extension.")


def save_recommendations_json(recommendations: list[BetRecommendation], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in recommendations], f, indent=2, default=str)


def render_daily_card(date_str: str, recs_by_sport: dict[str, list[BetRecommendation]], passes: list[str]) -> str:
    lines = ["=== DAILY BETTING CARD ===", f"Date: {date_str}", ""]
    for sport, recs in recs_by_sport.items():
        lines.append(f"[{sport.upper()}]")
        for idx, rec in enumerate(recs, 1):
            line_txt = "" if rec.line is None else f" {rec.line}"
            lines.extend(
                [
                    f"{idx}. {rec.side}{line_txt} ({rec.odds:+d}) - {rec.game}",
                    f"   Model probability: {rec.model_probability:.1%}",
                    f"   Edge: {rec.edge:.1%}",
                    f"   EV: {rec.expected_value:.3f} units",
                    f"   Confidence: {rec.confidence_tier} ({rec.confidence_score:.2f})",
                    f"   Stake: {rec.recommended_units:.2f} units",
                    f"   Reason: {rec.reason_summary}",
                ]
            )
        if not recs:
            lines.append("No qualified bets.")
        lines.append("")

    lines.append("[PASSES]")
    lines.extend([f"- {p}" for p in passes] if passes else ["- None"])
    return "\n".join(lines)
