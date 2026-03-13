# pip install optuna
import optuna
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from features import (
    calculate_efficiency_stats, calculate_late_season_form,
    calculate_team_stats, return_X_y, calculate_sos,
    calculate_conf_tourney_performance, calculate_elo, calculate_massey_rank,
)

np.random.seed(42)
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA = 'march-machine-learning-mania-2026'


def build_dataset(prefix, massey_df=None):
    conf_tourney            = pd.read_csv(f'{DATA}/{prefix}ConferenceTourneyGames.csv')
    regular_season_detailed = pd.read_csv(f'{DATA}/{prefix}RegularSeasonDetailedResults.csv')
    regular_season          = pd.read_csv(f'{DATA}/{prefix}RegularSeasonCompactResults.csv')
    tourney_detailed        = pd.read_csv(f'{DATA}/{prefix}NCAATourneyDetailedResults.csv')
    seeds                   = pd.read_csv(f'{DATA}/{prefix}NCAATourneySeeds.csv')

    seeds['SeedNum'] = seeds['Seed'].str[1:3].astype(int)

    team_stats   = calculate_team_stats(regular_season)
    eff_stats    = calculate_efficiency_stats(regular_season_detailed)
    sos          = calculate_sos(regular_season)
    late_stats   = calculate_late_season_form(regular_season, n_games=10)
    conf_stats   = calculate_conf_tourney_performance(conf_tourney, regular_season)
    elo_ratings  = calculate_elo(regular_season)
    massey_ranks = calculate_massey_rank(massey_df) if massey_df is not None else None

    _, X, y = return_X_y(seeds, tourney_detailed, team_stats, eff_stats,
                          late_stats, sos, conf_stats, elo_ratings, massey_ranks)
    return X, y


def tune(prefix, label, massey_df=None, n_trials=200):
    print(f"\n{'='*60}")
    print(f"  Tuning {label}'s model ({n_trials} trials)")
    print(f"{'='*60}")

    X, y = build_dataset(prefix, massey_df)

    def objective(trial):
        params = {
            'n_estimators'    : trial.suggest_int('n_estimators', 50, 500),
            'learning_rate'   : trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'max_depth'       : trial.suggest_int('max_depth', 2, 6),
            'subsample'       : trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha'       : trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
            'reg_lambda'      : trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
            'eval_metric'     : 'logloss',
            'random_state'    : 42,
            'verbosity'       : 0,
        }
        scores = cross_val_score(XGBClassifier(**params), X, y, cv=5, scoring='neg_log_loss')
        return -scores.mean()

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"Best log loss: {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    return study.best_params


massey_df = pd.read_csv(f'{DATA}/MMasseyOrdinals.csv')

mens_params   = tune('M', "Men",   massey_df=massey_df)
womens_params = tune('W', "Women", massey_df=None)

print("\n\n" + "="*60)
print("PASTE THESE INTO submit.py build_xgb():")
print("="*60)
print("\nMen's params:")
for k, v in mens_params.items():
    print(f"  {k}={repr(v)},")
print("\nWomen's params:")
for k, v in womens_params.items():
    print(f"  {k}={repr(v)},")
