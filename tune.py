# pip install optuna
import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from features import (calculate_efficiency_stats, calculate_late_season_form,
                      calculate_team_stats, return_X_y, calculate_sos,
                      calculate_conf_tourney_performance)

# --- Load data and build features (same as main.py) ---
np.random.seed(42)

regular_season          = pd.read_csv('march-machine-learning-mania-2026/MRegularSeasonCompactResults.csv')
regular_season_detailed = pd.read_csv('march-machine-learning-mania-2026/MRegularSeasonDetailedResults.csv')
tourney                 = pd.read_csv('march-machine-learning-mania-2026/MNCAATourneyDetailedResults.csv')
seeds                   = pd.read_csv('march-machine-learning-mania-2026/MNCAATourneySeeds.csv')
conf_tourney            = pd.read_csv('march-machine-learning-mania-2026/MConferenceTourneyGames.csv')

seeds['SeedNum'] = seeds['Seed'].str[1:3].astype(int)
team_stats  = calculate_team_stats(regular_season)
eff_stats   = calculate_efficiency_stats(regular_season_detailed)
sos         = calculate_sos(regular_season)
late_stats  = calculate_late_season_form(regular_season, n_games=10)
conf_stats  = calculate_conf_tourney_performance(conf_tourney, regular_season)

df, X, y = return_X_y(seeds, tourney, team_stats, eff_stats, late_stats, sos, conf_stats)

optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    params = {
        'n_estimators'    : trial.suggest_int('n_estimators', 50, 500),
        'learning_rate'   : trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth'       : trial.suggest_int('max_depth', 2, 6),
        'subsample'       : trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha'       : trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda'      : trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 5, 50),
        'eval_metric'     : 'logloss',
        'random_state'    : 42,
        'verbosity'       : 0,
    }
    model = XGBClassifier(**params)
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_log_loss')
    return -scores.mean()

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=200, show_progress_bar=True)

print(f"Best log loss: {study.best_value:.4f}")
print("Best params:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")

# Train final model with best params
best_xgb = XGBClassifier(**study.best_params, eval_metric='logloss', 
                          random_state=42, verbosity=0)
cv_final = cross_val_score(best_xgb, X, y, cv=5, scoring='neg_log_loss')
print(f"XGBoost tuned log loss: {-cv_final.mean():.4f}")