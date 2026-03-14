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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from features import (
    calculate_efficiency_stats,
    calculate_late_season_form,
    calculate_team_stats,
    return_X_y,
    calculate_sos,
    calculate_conf_tourney_performance,
    calculate_elo,
    calculate_massey_rank,
    train_team_embeddings,
)

np.random.seed(42)

DATA = 'march-machine-learning-mania-2026'
PREDICT_SEASON = 2026
# Fall back to 2025 seeds since 2026 tournament brackets aren't announced yet
SEED_PROXY_SEASON = 2025
DEFAULT_SEED = 16  # weakest seed for teams not in the seed proxy season

MENS_PARAMS = dict(
    n_estimators=231,
    learning_rate=0.020348000114701417,
    max_depth=2,
    subsample=0.5437066884244697,
    colsample_bytree=0.5211671523937955,
    reg_alpha=0.00015816307747082452,
    reg_lambda=0.00019858901258540043,
    min_child_weight=8,
)


def train_and_predict(prefix, sample_ids, xgb_params=None, massey_df=None):
    """
    Train a model on historical tournament data for the given gender prefix
    ('M' or 'W'), then predict win probabilities for all matchup pairs in
    sample_ids.

    Men's: XGBoost with tuned hyperparameters + embeddings
    Women's: Logistic Regression + embeddings (better CV log loss for women)

    Returns a list of {'ID': ..., 'Pred': ...} dicts.
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

    # ── Build hand-crafted features ────────────────────────────────
    team_stats   = calculate_team_stats(regular_season)
    eff_stats    = calculate_efficiency_stats(regular_season_detailed)
    sos          = calculate_sos(regular_season)
    late_stats   = calculate_late_season_form(regular_season, n_games=10)
    conf_stats   = calculate_conf_tourney_performance(conf_tourney, regular_season)
    elo_ratings  = calculate_elo(regular_season)
    massey_ranks = calculate_massey_rank(massey_df) if massey_df is not None else None

    # ── Train team embeddings on regular season data ───────────────
    print(f"  Training team embeddings...")
    embeddings = train_team_embeddings(regular_season, embed_dim=16, epochs=30)
    emb_cols = [c for c in embeddings.columns if c.startswith('Emb')]
    eb = embeddings.set_index(['Season', 'TeamID'])

    # ── Import XGBoost after torch finishes (avoids segfault on Py 3.14) ──
    from xgboost import XGBClassifier

    df, X, y = return_X_y(seeds_all, tourney_detailed, team_stats, eff_stats,
                           late_stats, sos, conf_stats, elo_ratings, massey_ranks,
                           embeddings=embeddings)

    feature_cols = list(X.columns)

    # ── Train model ────────────────────────────────────────────────
    if prefix == 'W':
        # LR outperforms XGBoost on women's data (less upset-prone, more linear)
        model = Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression())])
    else:
        model = Pipeline([
            ('model', XGBClassifier(**xgb_params, eval_metric='logloss', random_state=42, verbosity=0))
        ])
    model.fit(X, y)
    print(f"  Model trained on {len(X)} historical matchups ({len(feature_cols)} features).")

    # ── Build 2026 stat lookup tables ──────────────────────────────
    ts = team_stats.set_index(['Season', 'TeamID'])
    es = eff_stats.set_index(['Season', 'TeamID'])
    ls = late_stats.set_index(['Season', 'TeamID'])
    so = sos.set_index(['Season', 'TeamID'])
    cs = conf_stats.set_index(['Season', 'TeamID'])
    er = elo_ratings.set_index(['Season', 'TeamID'])['Elo']
    mr = massey_ranks.set_index(['Season', 'TeamID'])['MasseyRank'] if massey_ranks is not None else None

    seeds_proxy = (
        seeds_all[seeds_all['Season'] == SEED_PROXY_SEASON]
        .set_index('TeamID')['SeedNum']
        .to_dict()
    )

    def get_seed(team_id):
        return seeds_proxy.get(team_id, DEFAULT_SEED)

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

        try:
            e1v = eb.loc[(PREDICT_SEASON, t1)][emb_cols].values
            e2v = eb.loc[(PREDICT_SEASON, t2)][emb_cols].values
            emb_diff = {f'EmbDiff{d}': float(e1v[d] - e2v[d]) for d in range(len(emb_cols))}
        except KeyError:
            emb_diff = {f'EmbDiff{d}': 0.0 for d in range(len(emb_cols))}

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
            **emb_diff,
        }
        return pd.DataFrame([row])[feature_cols]

    # ── Predict all required pairs ─────────────────────────────────
    records = []
    skipped = 0
    for lower_id, higher_id in sample_ids:
        feats = build_features(lower_id, higher_id)
        if feats is None:
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

mens_records   = train_and_predict('M', mens_pairs, xgb_params=MENS_PARAMS, massey_df=massey_df)
womens_records = train_and_predict('W', womens_pairs)

# ── Combine and save ──────────────────────────────────────────────────────────
all_records = mens_records + womens_records
submission = pd.DataFrame(all_records)[['ID', 'Pred']]

submission = submission.set_index('ID').reindex(sample['ID']).reset_index()
submission['Pred'] = submission['Pred'].fillna(0.5)

submission.to_csv('submission.csv', index=False)
print(f"\nSaved {len(submission):,} predictions → submission.csv")
print(submission.head())
