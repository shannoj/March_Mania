# pip install optuna
import optuna
import numpy as np
import pandas as pd

# Import torch before xgboost to avoid segfault on Python 3.14
import torch
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from features import (
    calculate_efficiency_stats, calculate_late_season_form,
    calculate_team_stats, return_X_y, calculate_sos,
    calculate_conf_tourney_performance, calculate_elo, calculate_massey_rank,
    train_team_embeddings,
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

    print(f"  Training {prefix} embeddings...")
    embeddings = train_team_embeddings(regular_season, embed_dim=16, epochs=30)

    _, X, y = return_X_y(seeds, tourney_detailed, team_stats, eff_stats,
                          late_stats, sos, conf_stats, elo_ratings, massey_ranks,
                          embeddings=embeddings)
    return X, y


def tune(prefix, label, massey_df=None, n_trials=200):
    print(f"\n{'='*60}")
    print(f"  Tuning {label}'s XGBoost ({n_trials} trials, {28} features)")
    print(f"{'='*60}")

    X, y = build_dataset(prefix, massey_df)
    print(f"  Dataset: {X.shape[0]} rows x {X.shape[1]} features")

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

    return study.best_params, study.best_value


massey_df = pd.read_csv(f'{DATA}/MMasseyOrdinals.csv')

# Only tune men's — women's uses Logistic Regression (better CV log loss)
mens_params, mens_best = tune('M', "Men", massey_df=massey_df)

# ── Auto-update MENS_PARAMS in submit.py ─────────────────────────────────────
import re

params_str = 'MENS_PARAMS = dict(\n'
for k, v in mens_params.items():
    params_str += f'    {k}={repr(v)},\n'
params_str += ')'

with open('submit.py', 'r') as f:
    content = f.read()

content = re.sub(
    r'MENS_PARAMS = dict\(.*?\)',
    params_str,
    content,
    flags=re.DOTALL,
)

with open('submit.py', 'w') as f:
    f.write(content)

print(f"\nUpdated MENS_PARAMS in submit.py (best log loss: {mens_best:.4f})")
print("Run `python submit.py` to regenerate submission.csv")
