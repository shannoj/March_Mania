"""
predict.py
----------
Generate win-probability predictions for all possible 2026 NCAA Tournament
matchups, then simulate the bracket using those probabilities.

Usage:
    python predict.py

Outputs:
    - predictions.csv   : P(TeamA beats TeamB) for every pair of 2026 teams
    - bracket_sim.csv   : Most likely winner at each round via Monte Carlo
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
)

UPSET_THRESHOLD = 0.54

# ── 0. Reproducibility ────────────────────────────────────────────────────────
np.random.seed(42)

# ── 1. Load data ──────────────────────────────────────────────────────────────
conf_tourney            = pd.read_csv('march-machine-learning-mania-2026/MConferenceTourneyGames.csv')
regular_season_detailed = pd.read_csv('march-machine-learning-mania-2026/MRegularSeasonDetailedResults.csv')
regular_season          = pd.read_csv('march-machine-learning-mania-2026/MRegularSeasonCompactResults.csv')
tourney                 = pd.read_csv('march-machine-learning-mania-2026/MNCAATourneyDetailedResults.csv')
seeds                   = pd.read_csv('march-machine-learning-mania-2026/MNCAATourneySeeds.csv')
teams                   = pd.read_csv('march-machine-learning-mania-2026/MTeams.csv')  # for readable names

# ── 2. Build features (same pipeline as main.py) ──────────────────────────────
seeds['SeedNum'] = seeds['Seed'].str[1:3].astype(int)

team_stats  = calculate_team_stats(regular_season)
eff_stats   = calculate_efficiency_stats(regular_season_detailed)
sos         = calculate_sos(regular_season)
late_stats  = calculate_late_season_form(regular_season, n_games=10)
conf_stats  = calculate_conf_tourney_performance(conf_tourney, regular_season)

df, X, y = return_X_y(seeds, tourney, team_stats, eff_stats, late_stats, sos, conf_stats)

FEATURE_COLS = [
    'SeedDiff', 'WinRateDiff', 'PtDiffDiff', 'NetEffDiff', 'TempoDiff',
    'WinRateMomentum', 'PtDiffMomentum', 'SOSDiff', 'AdjWinRateDiff', 'SOS2Diff',
]

# ── 3. Train final model on all historical data ───────────────────────────────
model = Pipeline([
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
model.fit(X, y)
print("Model trained on full historical dataset.")

# ── 4. Build 2026 feature lookup ──────────────────────────────────────────────
PREDICT_SEASON = seeds['Season'].max()  # should be 2026
print(f"Predicting for season: {PREDICT_SEASON}")

# Index all stat tables for fast lookup
ts = team_stats.set_index(['Season', 'TeamID'])
es = eff_stats.set_index(['Season', 'TeamID'])
ls = late_stats.set_index(['Season', 'TeamID'])
so = sos.set_index(['Season', 'TeamID'])
cs = conf_stats.set_index(['Season', 'TeamID'])

seeds_2026 = seeds[seeds['Season'] == PREDICT_SEASON].set_index('TeamID')
team_ids_2026 = seeds_2026.index.tolist()

name_map = teams.set_index('TeamID')['TeamName'].to_dict()

def get_conf(team, col):
    try:
        return cs.loc[(PREDICT_SEASON, team)][col]
    except KeyError:
        return 0.0

def build_features_for_pair(t1, t2):
    """
    Returns a single-row DataFrame of features for team1 vs team2.
    Returns None if stat data is missing for either team.
    """
    try:
        s1 = ts.loc[(PREDICT_SEASON, t1)]
        s2 = ts.loc[(PREDICT_SEASON, t2)]
        e1 = es.loc[(PREDICT_SEASON, t1)]
        e2 = es.loc[(PREDICT_SEASON, t2)]
        l1 = ls.loc[(PREDICT_SEASON, t1)]
        l2 = ls.loc[(PREDICT_SEASON, t2)]
        o1 = so.loc[(PREDICT_SEASON, t1)]
        o2 = so.loc[(PREDICT_SEASON, t2)]
        seed1 = seeds_2026.loc[t1, 'SeedNum']
        seed2 = seeds_2026.loc[t2, 'SeedNum']
    except KeyError:
        return None

    c1_wr = get_conf(t1, 'ConfWinRate')
    c2_wr = get_conf(t2, 'ConfWinRate')
    c1_pd = get_conf(t1, 'ConfPtDiff')
    c2_pd = get_conf(t2, 'ConfPtDiff')

    row = {
        'SeedDiff'        : seed1 - seed2,
        'WinRateDiff'     : s1['WinRate']      - s2['WinRate'],
        'PtDiffDiff'      : s1['AvgPointDiff'] - s2['AvgPointDiff'],
        'NetEffDiff'      : e1['NetEff']        - e2['NetEff'],
        'TempoDiff'       : e1['Tempo']         - e2['Tempo'],
        'LateWinRateDiff' : l1['LateWinRate']   - l2['LateWinRate'],
        'LatePtDiffDiff'  : l1['LatePtDiff']    - l2['LatePtDiff'],
        'WinRateMomentum' : (l1['LateWinRate'] - s1['WinRate']) - (l2['LateWinRate'] - s2['WinRate']),
        'PtDiffMomentum'  : (l1['LatePtDiff']  - s1['AvgPointDiff']) - (l2['LatePtDiff'] - s2['AvgPointDiff']),
        'SOSDiff'         : o1['SOS']           - o2['SOS'],
        'SOS2Diff'        : o1['SOS2']          - o2['SOS2'],
        'AdjWinRateDiff'  : (s1['WinRate'] * o1['SOS']) - (s2['WinRate'] * o2['SOS']),
        'ConfWinRateDiff' : c1_wr - c2_wr,
        'ConfPtDiffDiff'  : c1_pd - c2_pd,
    }
    return pd.DataFrame([row])[FEATURE_COLS]

# ── 5. Predict all matchup probabilities ─────────────────────────────────────
print("Generating predictions for all matchup pairs...")

records = []
for t1, t2 in itertools.combinations(team_ids_2026, 2):
    feats = build_features_for_pair(t1, t2)
    if feats is None:
        continue

    prob_t1_wins = model.predict_proba(feats)[0][1]  # P(label=1) = P(t1 wins)

    records.append({
        'TeamID_A'   : t1,
        'TeamID_B'   : t2,
        'TeamName_A' : name_map.get(t1, t1),
        'TeamName_B' : name_map.get(t2, t2),
        'Seed_A'     : seeds_2026.loc[t1, 'SeedNum'],
        'Seed_B'     : seeds_2026.loc[t2, 'SeedNum'],
        'P_A_wins'   : round(prob_t1_wins, 4),
        'P_B_wins'   : round(1 - prob_t1_wins, 4),
    })

predictions = pd.DataFrame(records).sort_values('P_A_wins', ascending=False)
predictions.to_csv('predictions.csv', index=False)
print(f"Saved {len(predictions)} matchup predictions → predictions.csv")

# ── 6. Monte Carlo bracket simulation ────────────────────────────────────────
# Build a probability lookup: prob_lookup[(t1, t2)] = P(t1 beats t2)
prob_lookup = {}
for _, row in predictions.iterrows():
    a, b = row['TeamID_A'], row['TeamID_B']
    prob_lookup[(a, b)] = row['P_A_wins']
    prob_lookup[(b, a)] = row['P_B_wins']

def win_prob(t1, t2):
    """P(t1 beats t2). Falls back to seed-based estimate if missing."""
    if (t1, t2) in prob_lookup:
        return prob_lookup[(t1, t2)]
    # Fallback: seed comparison
    s1 = seeds_2026.loc[t1, 'SeedNum'] if t1 in seeds_2026.index else 8
    s2 = seeds_2026.loc[t2, 'SeedNum'] if t2 in seeds_2026.index else 8
    return 1 / (1 + np.exp(s1 - s2))

def simulate_bracket(team_list, n_simulations=10_000):
    """
    Runs n_simulations single-elimination tournaments.
    Returns a dict of {team_id: fraction of sims they won the championship}.
    """
    win_counts = {t: 0 for t in team_list}

    for _ in range(n_simulations):
        remaining = list(team_list)
        np.random.shuffle(remaining)

        while len(remaining) > 1:
            next_round = []
            # Pair up teams sequentially (shuffle re-randomizes bracket each sim)
            for i in range(0, len(remaining) - 1, 2):
                t1, t2 = remaining[i], remaining[i + 1]
                p = win_prob(t1, t2)
                winner = t1 if np.random.rand() < p else t2
                next_round.append(winner)

            # If odd number (bye), last team advances automatically
            if len(remaining) % 2 == 1:
                next_round.append(remaining[-1])

            remaining = next_round

        win_counts[remaining[0]] += 1

    return {t: c / n_simulations for t, c in win_counts.items()}

print("Running Monte Carlo bracket simulation (10,000 trials)...")
champ_probs = simulate_bracket(team_ids_2026, n_simulations=10_000)

bracket_results = pd.DataFrame([
    {
        'TeamID'    : t,
        'TeamName'  : name_map.get(t, t),
        'Seed'      : seeds_2026.loc[t, 'SeedNum'],
        'ChampProb' : round(p, 4),
    }
    for t, p in champ_probs.items()
]).sort_values('ChampProb', ascending=False).reset_index(drop=True)

bracket_results.to_csv('bracket_sim.csv', index=False)
print("Top 10 championship contenders:")
print(bracket_results.head(10).to_string(index=False))
print("\nSaved full bracket simulation → bracket_sim.csv")

# ── 7. Region-aware deterministic bracket simulation ─────────────────────────
slots = pd.read_csv('march-machine-learning-mania-2026/MNCAATourneySlots.csv')
slots_2026 = slots[slots['Season'] == PREDICT_SEASON]

# Build seed → TeamID mapping for 2026
seed_to_team = seeds[seeds['Season'] == PREDICT_SEASON].set_index('Seed')['TeamID'].to_dict()

def simulate_bracket_from_slots(slots_df, deterministic=True):
    """
    Simulate the tournament using the actual bracket slot structure.
    
    deterministic=True  → always pick the higher-probability team
    deterministic=False → sample by probability (Monte Carlo single run)
    """
    # slot_winner maps each slot name to the TeamID that won it
    slot_winner = {}

    # First, populate all first-round slots with actual seeded teams
    # Slots that reference seeds directly (e.g. StrongSeed='W01') get filled first
    for _, row in slots_df.iterrows():
        slot      = row['Slot']
        strong    = row['StrongSeed']
        weak      = row['WeakSeed']

        # Resolve strong seed to a team if it's a direct seed reference
        if strong in seed_to_team:
            t_strong = seed_to_team[strong]
        else:
            t_strong = slot_winner.get(strong)  # winner of a prior slot

        if weak in seed_to_team:
            t_weak = seed_to_team[weak]
        else:
            t_weak = slot_winner.get(weak)

        if t_strong is None or t_weak is None:
            continue  # prior round not yet resolved — will hit this slot again

        p = win_prob(t_strong, t_weak)
        if deterministic:
            winner = t_strong if p >= 0.5 else t_weak
        else:
            winner = t_strong if np.random.rand() < p else t_weak

        slot_winner[slot] = winner

    return slot_winner

ROUND_LABELS = {
    'R1': 'Round of 64',
    'R2': 'Round of 32', 
    'R3': 'Sweet 16',
    'R4': 'Elite 8',
    'R5': 'Final Four',
    'R6': 'Championship',
}

def resolve_bracket(slots_df):
    slot_winner = {}
    # Collect all games first, then print grouped by round
    games_by_round = {}
    max_passes = 10

    for _ in range(max_passes):
        prev_len = len(slot_winner)

        for _, row in slots_df.iterrows():
            slot   = row['Slot']
            strong = row['StrongSeed']
            weak   = row['WeakSeed']

            if slot in slot_winner:
                continue

            t_strong = seed_to_team.get(strong) or slot_winner.get(strong)
            t_weak   = seed_to_team.get(weak)   or slot_winner.get(weak)

            if t_strong is None or t_weak is None:
                continue

            p = win_prob(t_strong, t_weak)
            if max(p, 1 - p) < UPSET_THRESHOLD:
                # Too close to call — flip a coin
                winner = t_strong if np.random.rand() > 0.5 else t_weak
            else:
                winner = t_strong if p >= 0.5 else t_weak
            loser = t_weak if winner == t_strong else t_strong

            slot_winner[slot] = winner

            round_key = slot[:2]  # e.g. 'R1', 'R2', etc.
            if round_key not in games_by_round:
                games_by_round[round_key] = []

            seed_w = seeds_2026.loc[winner, 'SeedNum'] if winner in seeds_2026.index else '?'
            seed_l = seeds_2026.loc[loser,  'SeedNum'] if loser  in seeds_2026.index else '?'

            too_close = max(p, 1 - p) < UPSET_THRESHOLD
            label_str = "  [TOSS-UP]" if too_close else ""

            games_by_round[round_key].append(
                f"  ({seed_w:>2}) {name_map.get(winner, winner):25s} "
                f"def. ({seed_l:>2}) {name_map.get(loser, loser):25s}  "
                f"[{max(p, 1-p):.1%}]{label_str}"
            )

        if len(slot_winner) == prev_len:
            break

    # Print everything grouped by round
    for round_key in sorted(games_by_round.keys()):
        label = ROUND_LABELS.get(round_key, round_key)
        print(f"\n{'─'*60}")
        print(f"  {label}")
        print(f"{'─'*60}")
        for line in games_by_round[round_key]:
            print(line)

    return slot_winner


print("\n" + "="*70)
print("DETERMINISTIC BRACKET SIMULATION (actual bracket structure)")
print("="*70)
results = resolve_bracket(slots_2026)

# The championship slot is typically named 'R6CH' or similar — find it
champ_slot = slots_2026['Slot'].max()  # highest slot alphabetically is usually the final
champion_id = results.get(champ_slot)
if champion_id:
    print(f"\n🏆 Champion: {name_map.get(champion_id, champion_id)} "
          f"(Seed {seeds_2026.loc[champion_id, 'SeedNum']})")