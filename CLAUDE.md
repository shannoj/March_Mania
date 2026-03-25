# March Machine Learning Mania 2026

NCAA tournament prediction project for the Kaggle March Machine Learning Mania 2026 competition. Predicts win probabilities for men's and women's tournament matchups.

## Project Structure

```
March_Mania/
├── features.py            # Feature engineering (stats, ELO, embeddings, etc.)
├── main.py                # Model training and evaluation
├── predict.py             # Men's predictions + bracket simulation
├── predict_w.py           # Women's predictions + bracket simulation
├── submit.py              # Kaggle submission generator
├── tune.py                # Optuna hyperparameter tuning
├── app.py                 # Streamlit web dashboard (4 pages)
├── generate_bracket.py    # Generates standalone bracket.html
├── bracket.html           # Yahoo-style interactive bracket (open via HTTP server)
├── lookup.py              # CLI team prediction lookup
├── predictions.csv        # Men's matchup probabilities
├── predictions_w.csv      # Women's matchup probabilities
├── bracket_sim.csv        # Men's championship odds (Monte Carlo)
├── bracket_sim_w.csv      # Women's championship odds
└── submission.csv         # Final Kaggle submission
```

## Data

Kaggle dataset lives in `march-machine-learning-mania-2026/`. Files use `M` prefix for men's and `W` prefix for women's:
- `MRegularSeasonDetailedResults.csv` — box scores for efficiency stats
- `MNCAATourneyCompactResults.csv` / `MNCAATourneyDetailedResults.csv` — tournament history
- `MNCAATourneySeeds.csv` — bracket seeds
- `MMasseyOrdinals.csv` — third-party consensus rankings
- `MTeams.csv` — team name/ID mapping

## Models

**Men's:** XGBoost (tuned via Optuna — 365 estimators, max_depth=5, lr=0.0131)
**Women's:** Logistic Regression with StandardScaler (better CV log loss than XGBoost)

## Features (features.py)

Key functions used in `return_X_y()` to build the ~28-feature training matrix:
- `calculate_team_stats()` — win rate, scoring, point differential
- `calculate_efficiency_stats()` — offensive/defensive efficiency per 100 possessions, pace
- `calculate_late_season_form()` — last-N-games win rate and scoring
- `calculate_sos()` — two-level strength of schedule
- `calculate_elo()` — Elo ratings with seasonal carryover
- `calculate_massey_rank()` — consensus external rankings
- `train_team_embeddings()` — 16-dim PyTorch embeddings from regular season outcomes
- `calculate_conf_tourney_performance()` — conference tournament stats

## Workflow

1. **Train:** `python main.py` — trains and cross-validates models
2. **Tune:** `python tune.py` — Optuna hyperparameter search, auto-updates submit.py
3. **Predict:** `python predict.py` / `python predict_w.py` — regenerates prediction CSVs
4. **Submit:** `python submit.py` — builds Kaggle submission.csv
5. **Explore:** `streamlit run app.py` — interactive dashboard (Head-to-Head, Team Matchups, Championship Odds, Bracket View)
6. **Bracket:** `python generate_bracket.py` — generates bracket.html, then serve with `python -m http.server 8080` and open http://localhost:8080/bracket.html
7. **Lookup:** `python lookup.py <team1> <team2>` — quick CLI prediction

## Tech Stack

- **ML:** scikit-learn, XGBoost, PyTorch (embeddings), Optuna
- **Data:** pandas, numpy
- **Web:** Streamlit, standalone HTML/CSS/JS (bracket.html)
- **Python:** 3.14 (.venv)

## Bracket View (app.py)

The Streamlit app's "Bracket View" page and `bracket.html` both show:
- Actual results for completed games (Rounds 1 & 2 as of 2026-03-25) with model win probabilities
- Model predictions + probabilities for upcoming games (Sweet 16 onward)
- Green border = correct model pick, Red border = wrong pick, Amber = upset
- Men's/Women's toggle

`ACTUAL_RESULTS` dict in both `app.py` and `generate_bracket.py` must be updated as games are played.
Slot codes: **W=East, X=South, Y=Midwest, Z=West** (men's); W/X/Y/Z=Regionals 1/4/3/2 (women's).

## Notes

- CSV data files are gitignored
- `bracket.html` must be served via HTTP (`python -m http.server 8080`), not opened as file://
- Reproducibility seeded with `np.random.seed(42)` and PyTorch manual seed
- `UPSET_THRESHOLD` and `SEED_PROXY_SEASON` constants control bracket simulation behavior
