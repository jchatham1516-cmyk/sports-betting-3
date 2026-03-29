"""MLB pitcher ERA lookups."""

from __future__ import annotations


def build_pitcher_era_map(pitchers_dict: dict[str, str] | None = None) -> dict[str, float]:
    """Build a simple pitcher-name -> ERA mapping.

    The map intentionally stays static for now to keep the runtime pipeline
    reliable. Unknown pitchers are handled by downstream fallback logic.
    """

    return {
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
