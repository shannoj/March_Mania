"""
submit.py
---------
Generate the Kaggle submission CSV for March Machine Learning Mania 2026.
Covers all men's AND women's matchup pairs listed in SampleSubmissionStage2.csv.

Usage:
    python submit.py

Output:
    submission.csv  – Kaggle-format file with ID,Pred columns
"""

import itertools
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from features import (
    calculate_efficiency_stats,
    calculate_late_season_form,
    calculate_team_stats,
    return_X_y,
    calculate_sos,
    calculate_conf_tourney_performance,
    calculate_elo,
    calculate_massey_rank,
)

np.random.seed(42)

DATA = 'march-machine-learning-mania-2026'
PREDICT_SEASON = 2026
# Fall back to 2025 seeds since 2026 tournament brackets aren't announced yet
SEED_PROXY_SEASON = 2025
DEFAULT_SEED = 16  # weakest seed for teams not in the seed proxy season

FEATURE_COLS = [
    'SeedDiff', 'WinRateDiff', 'PtDiffDiff', 'NetEffDiff', 'TempoDiff',
    'WinRateMomentum', 'PtDiffMomentum', 'SOSDiff', 'AdjWinRateDiff', 'SOS2Diff',
    'EloDiff', 'MasseyRankDiff',
]


MENS_PARAMS = dict(
    n_estimators=234,
    learning_rate=0.016056625129573475,
    max_depth=3,
    subsample=0.7272325021079066,
    colsample_bytree=0.8376124755483081,
    reg_alpha=0.0013874206365291576,
    reg_lambda=0.024733780434062328,
    min_child_weight=7,
)

WOMENS_PARAMS = dict(
    n_estimators=386,
    learning_rate=0.011978621118783415,
    max_depth=2,
    subsample=0.5450461171243898,
    colsample_bytree=0.5211229272878785,
    reg_alpha=0.00014016937971181176,
    reg_lambda=0.00010478510031154102,
    min_child_weight=2,
)


def build_xgb(params):
    return Pipeline([
        ('model', XGBClassifier(**params, eval_metric='logloss', random_state=42, verbosity=0))
    ])


def train_and_predict(prefix, sample_ids, xgb_params, massey_df=None):
    """
    Train a model on historical tournament data for the given gender prefix
    ('M' or 'W'), then predict win probabilities for all matchup pairs in
    sample_ids.

    Returns a dict: {(lower_id, higher_id): prob_lower_wins}
    """
    print(f"\n{'='*60}")
    print(f"  Processing {'Men' if prefix == 'M' else 'Women'}'s tournament")
    print(f"{'='*60}")

    # ── Load raw data ──────────────────────────────────────────────
    conf_tourney            = pd.read_csv(f'{DATA}/{prefix}ConferenceTourneyGames.csv')
    regular_season_detailed = pd.read_csv(f'{DATA}/{prefix}RegularSeasonDetailedResults.csv')
    regular_season          = pd.read_csv(f'{DATA}/{prefix}RegularSeasonCompactResults.csv')
    tourney_detailed        = pd.read_csv(f'{DATA}/{prefix}NCAATourneyDetailedResults.csv')
    seeds_all               = pd.read_csv(f'{DATA}/{prefix}NCAATourneySeeds.csv')
    teams                   = pd.read_csv(f'{DATA}/{prefix}Teams.csv')

    seeds_all['SeedNum'] = seeds_all['Seed'].str[1:3].astype(int)

    # ── Build features on all historical data ──────────────────────
    team_stats  = calculate_team_stats(regular_season)
    eff_stats   = calculate_efficiency_stats(regular_season_detailed)
    sos         = calculate_sos(regular_season)
    late_stats  = calculate_late_season_form(regular_season, n_games=10)
    conf_stats  = calculate_conf_tourney_performance(conf_tourney, regular_season)
    elo_ratings  = calculate_elo(regular_season)
    massey_ranks = calculate_massey_rank(massey_df) if massey_df is not None else None

    df, X, y = return_X_y(seeds_all, tourney_detailed, team_stats, eff_stats,
                           late_stats, sos, conf_stats, elo_ratings, massey_ranks)

    # ── Train model ────────────────────────────────────────────────
    model = build_xgb(xgb_params)
    model.fit(X, y)
    print(f"  Model trained on {len(X)} historical matchups.")

    # ── Build 2026 stat lookup tables ──────────────────────────────
    ts = team_stats.set_index(['Season', 'TeamID'])
    es = eff_stats.set_index(['Season', 'TeamID'])
    ls = late_stats.set_index(['Season', 'TeamID'])
    so = sos.set_index(['Season', 'TeamID'])
    cs = conf_stats.set_index(['Season', 'TeamID'])
    er = elo_ratings.set_index(['Season', 'TeamID'])['Elo']
    mr = massey_ranks.set_index(['Season', 'TeamID'])['MasseyRank'] if massey_ranks is not None else None

    # Seed proxy: use SEED_PROXY_SEASON seeds; default to DEFAULT_SEED if absent
    seeds_proxy = (
        seeds_all[seeds_all['Season'] == SEED_PROXY_SEASON]
        .set_index('TeamID')['SeedNum']
        .to_dict()
    )

    name_map = teams.set_index('TeamID')['TeamName'].to_dict()

    def get_seed(team_id):
        return seeds_proxy.get(team_id, DEFAULT_SEED)

    def get_conf_stat(team_id, col):
        try:
            return cs.loc[(PREDICT_SEASON, team_id)][col]
        except KeyError:
            return 0.0

    def build_features(t1, t2):
        try:
            s1 = ts.loc[(PREDICT_SEASON, t1)]
            s2 = ts.loc[(PREDICT_SEASON, t2)]
            e1 = es.loc[(PREDICT_SEASON, t1)]
            e2 = es.loc[(PREDICT_SEASON, t2)]
            l1 = ls.loc[(PREDICT_SEASON, t1)]
            l2 = ls.loc[(PREDICT_SEASON, t2)]
            o1 = so.loc[(PREDICT_SEASON, t1)]
            o2 = so.loc[(PREDICT_SEASON, t2)]
        except KeyError:
            return None

        try:
            elo_diff = er.loc[(PREDICT_SEASON, t1)] - er.loc[(PREDICT_SEASON, t2)]
        except KeyError:
            elo_diff = 0.0

        try:
            massey_diff = mr.loc[(PREDICT_SEASON, t1)] - mr.loc[(PREDICT_SEASON, t2)] if mr is not None else 0.0
        except KeyError:
            massey_diff = 0.0

        row = {
            'SeedDiff'        : get_seed(t1) - get_seed(t2),
            'WinRateDiff'     : s1['WinRate']      - s2['WinRate'],
            'PtDiffDiff'      : s1['AvgPointDiff'] - s2['AvgPointDiff'],
            'NetEffDiff'      : e1['NetEff']        - e2['NetEff'],
            'TempoDiff'       : e1['Tempo']         - e2['Tempo'],
            'WinRateMomentum' : (l1['LateWinRate'] - s1['WinRate']) - (l2['LateWinRate'] - s2['WinRate']),
            'PtDiffMomentum'  : (l1['LatePtDiff']  - s1['AvgPointDiff']) - (l2['LatePtDiff'] - s2['AvgPointDiff']),
            'SOSDiff'         : o1['SOS']  - o2['SOS'],
            'SOS2Diff'        : o1['SOS2'] - o2['SOS2'],
            'AdjWinRateDiff'  : (s1['WinRate'] * o1['SOS']) - (s2['WinRate'] * o2['SOS']),
            'EloDiff'         : elo_diff,
            'MasseyRankDiff'  : massey_diff,
        }
        return pd.DataFrame([row])[FEATURE_COLS]

    # ── Predict all required pairs ─────────────────────────────────
    records = []
    skipped = 0
    for lower_id, higher_id in sample_ids:
        feats = build_features(lower_id, higher_id)
        if feats is None:
            # Fall back to 0.5 if stats are missing for either team
            records.append({'ID': f'{PREDICT_SEASON}_{lower_id}_{higher_id}', 'Pred': 0.5})
            skipped += 1
            continue

        prob = model.predict_proba(feats)[0][1]
        records.append({'ID': f'{PREDICT_SEASON}_{lower_id}_{higher_id}', 'Pred': round(prob, 6)})

    print(f"  Generated {len(records)} predictions ({skipped} fell back to 0.5).")
    return records


# ── Load sample submission to get the exact required matchup IDs ──────────────
print("Loading sample submission matchup pairs...")
sample = pd.read_csv(f'{DATA}/SampleSubmissionStage2.csv')

mens_pairs   = []
womens_pairs = []

for id_str in sample['ID']:
    _, t1, t2 = id_str.split('_')
    t1, t2 = int(t1), int(t2)
    assert t1 < t2, f"Expected lower ID first: {id_str}"
    if t1 < 3000:   # Men's teams: 1xxx
        mens_pairs.append((t1, t2))
    else:            # Women's teams: 3xxx
        womens_pairs.append((t1, t2))

print(f"  Men's pairs:   {len(mens_pairs):,}")
print(f"  Women's pairs: {len(womens_pairs):,}")

# ── Generate predictions for each gender ─────────────────────────────────────
massey_df = pd.read_csv(f'{DATA}/MMasseyOrdinals.csv')

mens_records   = train_and_predict('M', mens_pairs, MENS_PARAMS,   massey_df=massey_df)
womens_records = train_and_predict('W', womens_pairs, WOMENS_PARAMS)

# ── Combine and save ──────────────────────────────────────────────────────────
all_records = mens_records + womens_records
submission = pd.DataFrame(all_records)[['ID', 'Pred']]

# Verify it matches the sample submission order
submission = submission.set_index('ID').reindex(sample['ID']).reset_index()
submission['Pred'] = submission['Pred'].fillna(0.5)

submission.to_csv('submission.csv', index=False)
print(f"\nSaved {len(submission):,} predictions → submission.csv")
print(submission.head())
