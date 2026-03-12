# Sports Betting Engine (NBA / NFL / NHL)

Production-oriented, modular Python framework for disciplined, data-driven daily betting recommendations.

## What this system does
- Ingests live daily schedules + odds from The Odds API (no implicit mock fallback in live mode).
- Trains interpretable probability models per sport and market (moneyline/spread/totals).
- Calibrates probabilities with isotonic calibration.
- Computes no-vig market probability, edge, and expected value.
- Selects bets only when configurable quality thresholds are met.
- Sizes stakes with flat betting or fractional Kelly + caps + uncertainty discount.
- Ranks bets by EV/edge/confidence/model-quality composite score.
- Exports daily predictions and recommendation cards (CSV/JSON/TXT).
- Generates backtest summary outputs (starter includes proxy grading, replace with realized grading feed).

## Project structure

```text
sports_betting/
  config/
    default.yaml
    settings.py
  data/
    raw/
    processed/
    historical/
    outputs/
  sports/
    common/
      base.py
      odds.py
      risk.py
      calibration.py
      selection.py
      validation.py
      reporting.py
      logging_utils.py
    nba/model.py
    nfl/model.py
    nhl/model.py
  models/entities.py
  backtesting/engine.py
  scripts/data_io.py
  tests/test_common.py
main.py
requirements.txt
README.md
```

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run daily predictions
```bash
python main.py
```
Outputs:
- `sports_betting/data/outputs/daily_predictions.csv`
- `sports_betting/data/outputs/daily_recommendations.csv`
- `sports_betting/data/outputs/daily_recommendations.json`
- `sports_betting/data/outputs/daily_card.txt`
- `sports_betting/data/outputs/backtest_summary.csv`

### Production model behavior
- Live daily runs now **prefer loading saved model artifacts** at:
  - `sports_betting/data/models/nba_model.pkl`
  - `sports_betting/data/models/nfl_model.pkl`
  - `sports_betting/data/models/nhl_model.pkl`
- If an artifact for a sport is missing, the app trains from that sport's historical CSV and saves a new artifact.
- Preflight remains strict and will fail when both historical CSV and model artifact are missing.

## Train and save model artifacts
```bash
python -m sports_betting.scripts.train_models --sports nba nfl nhl
```

Per-sport examples:
```bash
python -m sports_betting.scripts.train_models --sports nba
python -m sports_betting.scripts.train_models --sports nfl
python -m sports_betting.scripts.train_models --sports nhl
```

Validate production inputs before daily runs:
```bash
python -m sports_betting.scripts.preflight --sports nba nfl nhl
```

## Run tests
```bash
pytest sports_betting/tests -q
```

## Data contract (daily/historical)
Place files in:
- `sports_betting/data/historical/{sport}_historical.csv`
- `sports_betting/data/raw/{sport}_daily.csv`

Required base fields:
- identifiers: `game_id, event_date, away_team, home_team`
- odds: `away_odds, home_odds, away_spread_odds, home_spread_odds, over_odds, under_odds`
- lines: `spread_line, total_line`
- labels for training historical: `home_win, home_cover, over_hit`
- feature columns: defined per sport model in each `model.py`

If files are missing in live mode, the run fails fast. Synthetic data is only allowed when `TEST_MODE=true`.

## Configuration and tuning
Edit `sports_betting/config/default.yaml`:
- bankroll and stake mode
- min edge/EV/confidence by sport+market
- output top N
- date range and API placeholders

### Tune for profitability first
1. Increase `min_edge` and `min_ev` until pass rate is healthy.
2. Set tighter `min_confidence` for markets with poor calibration.
3. Reduce `kelly_fraction` (e.g., 0.15–0.25) to control variance.
4. Inspect performance by market, confidence tier, and sport.

## Extending features
- Add new columns to historical + daily feeds.
- Include them in sport-specific `WIN_FEATURES`, `SPREAD_FEATURES`, `TOTAL_FEATURES`.
- Retrain, validate calibration, and retune thresholds.

## Known limitations / honesty notes
- No model guarantees profits.
- Current backtest summary uses EV-proxy result grading unless realized outcomes are provided.
- Injury/lineup/goaltender certainty filters are scaffolded via features and flags; production requires robust live data integrations.
- Closing line value tracking requires line snapshots over time (to add in next iteration).


## Live Odds API requirements
- Required GitHub secret: `ODDS_API_KEY`
- Optional local env: `TEST_MODE=true` to explicitly enable synthetic demo fallback for development only
- Live mode (`TEST_MODE=false`, default) will:
  - fetch daily odds from The Odds API per sport (`nba`, `nfl`, `nhl`)
  - fail if `ODDS_API_KEY` is missing
  - fail if no usable live events/markets are returned
