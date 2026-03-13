import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from features import calculate_efficiency_stats, calculate_late_season_form, calculate_team_stats, return_X_y, calculate_sos, calculate_conf_tourney_performance

conf_tourney = pd.read_csv('march-machine-learning-mania-2026/MConferenceTourneyGames.csv')

regular_season_detailed = pd.read_csv('march-machine-learning-mania-2026/MRegularSeasonDetailedResults.csv')

regular_season = pd.read_csv('march-machine-learning-mania-2026/MRegularSeasonCompactResults.csv')

tournament_results = pd.read_csv('march-machine-learning-mania-2026/MNCAATourneyCompactResults.csv')

tourney = pd.read_csv('march-machine-learning-mania-2026/MNCAATourneyDetailedResults.csv')

seeds = pd.read_csv('march-machine-learning-mania-2026/MNCAATourneySeeds.csv')

np.random.seed(42)


# --- Run it ---

seeds['SeedNum'] = seeds['Seed'].str[1:3].astype(int)
team_stats  = calculate_team_stats(regular_season)
eff_stats   = calculate_efficiency_stats(regular_season_detailed)
sos = calculate_sos(regular_season)
late_stats = calculate_late_season_form(regular_season, n_games=10)
conf_stats = calculate_conf_tourney_performance(conf_tourney, regular_season)

df, X, y = return_X_y(seeds, tourney, team_stats, eff_stats, late_stats, sos, conf_stats)

# Logistic regression baseline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  LogisticRegression())
])

cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_log_loss')
print(f"Logistic Regression Log Loss: {-cv_scores.mean():.4f}")

# Gradient boosting — no scaling needed, trees are scale-invariant
from sklearn.ensemble import GradientBoostingClassifier
gb_pipeline = Pipeline([
    ('model', GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.02,   # slower learning
    max_depth=2,
    subsample=0.6,
    min_samples_leaf=25,
    max_features=0.8,     # like colsample_bytree — use 80% of features per split
    random_state=42
))
])
cv_scores_gb = cross_val_score(gb_pipeline, X, y, cv=5, scoring='neg_log_loss')
print(f"Gradient Boosting Log Loss:   {-cv_scores_gb.mean():.4f}")

# Fit final GB model
gb_pipeline.fit(X, y)
gb_model = gb_pipeline.named_steps['model']

xgb_pipeline = Pipeline([
    ('model', XGBClassifier(
        n_estimators=365,
        learning_rate=0.0131,
        max_depth=5,
        subsample=0.9708,
        colsample_bytree=0.9451,
        reg_alpha=0.000359,
        reg_lambda=0.000637,
        min_child_weight=23,
        eval_metric='logloss',
        random_state=42,
        verbosity=0,
    ))
])

cv_scores_xgb = cross_val_score(xgb_pipeline, X, y, cv=5, scoring='neg_log_loss')

print(f"XGBoost Log Loss: {-cv_scores_xgb.mean():.4f}")

# Feature importance (different from coefficients — this is reduction in loss)
for name, imp in sorted(zip(X.columns, gb_model.feature_importances_), 
                         key=lambda x: -x[1]):
    print(f"{name}: {imp:.4f}")


