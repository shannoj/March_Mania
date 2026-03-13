import pandas as pd
import numpy as np

def calculate_efficiency_stats(detailed_df):
    """
    Compute offensive/defensive efficiency from detailed box score data.
    Efficiency = points scored per 100 possessions (KenPom-style).
    
    Possessions estimate (Dean Oliver formula):
      Poss = FGA - OReb + TO + 0.44 * FTA
    """

    # --- Winner perspective ---
    winners = detailed_df[[
        'Season', 'WTeamID', 'LTeamID',
        'WScore', 'LScore',
        'WFGA', 'WFTA', 'WOR', 'WTO',   # offensive stats
        'LFGA', 'LFTA', 'LOR', 'LTO',   # opponent stats (for def possession estimate)
    ]].copy()
    winners.columns = [
        'Season', 'TeamID', 'OppID',
        'PtsFor', 'PtsAgainst',
        'FGA', 'FTA', 'OReb', 'TO',
        'OppFGA', 'OppFTA', 'OppOReb', 'OppTO',
    ]

    # --- Loser perspective ---
    losers = detailed_df[[
        'Season', 'LTeamID', 'WTeamID',
        'LScore', 'WScore',
        'LFGA', 'LFTA', 'LOR', 'LTO',
        'WFGA', 'WFTA', 'WOR', 'WTO',
    ]].copy()
    losers.columns = winners.columns

    all_games = pd.concat([winners, losers], ignore_index=True)

    # Possession estimates for each side
    all_games['Poss'] = (
        all_games['FGA']
        - all_games['OReb']
        + all_games['TO']
        + 0.44 * all_games['FTA']
    )
    all_games['OppPoss'] = (
        all_games['OppFGA']
        - all_games['OppOReb']
        + all_games['OppTO']
        + 0.44 * all_games['OppFTA']
    )

    # Use average of both sides' possession estimate (more stable)
    all_games['PossAvg'] = (all_games['Poss'] + all_games['OppPoss']) / 2

    all_games['OffEff'] = all_games['PtsFor']    / all_games['PossAvg'] * 100
    all_games['DefEff'] = all_games['PtsAgainst'] / all_games['PossAvg'] * 100
    all_games['NetEff'] = all_games['OffEff'] - all_games['DefEff']
    all_games['Tempo']  = all_games['PossAvg']  # possessions per game = pace proxy

    eff_stats = all_games.groupby(['Season', 'TeamID']).agg(
        OffEff = ('OffEff', 'mean'),
        DefEff = ('DefEff', 'mean'),
        NetEff = ('NetEff', 'mean'),
        Tempo  = ('Tempo',  'mean'),
    ).reset_index()

    return eff_stats


def calculate_team_stats(df):
    winners = df[['Season', 'WTeamID', 'WScore', 'LScore']].copy()
    winners.columns = ['Season', 'TeamID', 'ScoreFor', 'ScoreAgainst']
    winners['Win'] = 1

    losers = df[['Season', 'LTeamID', 'LScore', 'WScore']].copy()
    losers.columns = ['Season', 'TeamID', 'ScoreFor', 'ScoreAgainst']
    losers['Win'] = 0

    all_games = pd.concat([winners, losers], ignore_index=True)

    team_stats = all_games.groupby(['Season', 'TeamID']).agg(
        WinRate    = ('Win', 'mean'),
        AvgScore   = ('ScoreFor', 'mean'),
        AvgAllowed = ('ScoreAgainst', 'mean'),
    ).reset_index()

    team_stats['AvgPointDiff'] = team_stats['AvgScore'] - team_stats['AvgAllowed']

    return team_stats


def calculate_late_season_form(df, n_games=10):
    """
    For each team/season, compute stats over their last n_games.
    Returns WinRate, AvgPointDiff, and NetEff for the late season window.
    """
    winners = df[['Season', 'DayNum', 'WTeamID', 'WScore', 'LScore']].copy()
    winners.columns = ['Season', 'DayNum', 'TeamID', 'ScoreFor', 'ScoreAgainst']
    winners['Win'] = 1

    losers = df[['Season', 'DayNum', 'LTeamID', 'LScore', 'WScore']].copy()
    losers.columns = ['Season', 'DayNum', 'TeamID', 'ScoreFor', 'ScoreAgainst']
    losers['Win'] = 0

    all_games = pd.concat([winners, losers], ignore_index=True)
    all_games = all_games.sort_values(['Season', 'TeamID', 'DayNum'])

    # Take last n_games per team per season
    late = (
        all_games
        .groupby(['Season', 'TeamID'])
        .tail(n_games)
    )

    late_stats = late.groupby(['Season', 'TeamID']).agg(
        LateWinRate    = ('Win', 'mean'),
        LateAvgPtDiff  = ('ScoreFor', 'mean'),  # will subtract ScoreAgainst below
        LateAvgAllowed = ('ScoreAgainst', 'mean'),
    ).reset_index()

    late_stats['LatePtDiff'] = late_stats['LateAvgPtDiff'] - late_stats['LateAvgAllowed']
    late_stats = late_stats.drop(columns=['LateAvgPtDiff', 'LateAvgAllowed'])

    return late_stats


def calculate_elo(regular_season, k=32, carry_over=0.6, base=1500):
    """
    Compute end-of-regular-season Elo ratings for every team.

    Ratings are updated game-by-game in chronological order within each season.
    At the start of each new season, ratings are regressed toward the mean:
        new_rating = base + carry_over * (prev_rating - base)

    Returns a DataFrame with columns: Season, TeamID, Elo
    """
    games = regular_season[['Season', 'DayNum', 'WTeamID', 'LTeamID']].copy()
    games = games.sort_values(['Season', 'DayNum']).reset_index(drop=True)

    ratings = {}   # TeamID -> current Elo
    records = []

    prev_season = None
    for _, row in games.iterrows():
        season = row['Season']
        w, l   = row['WTeamID'], row['LTeamID']

        # Season rollover: regress all ratings toward the mean
        if season != prev_season:
            if prev_season is not None:
                # Save end-of-season ratings before regression
                for tid, elo in ratings.items():
                    records.append({'Season': prev_season, 'TeamID': tid, 'Elo': elo})
                # Regress toward mean for new season
                ratings = {
                    tid: base + carry_over * (elo - base)
                    for tid, elo in ratings.items()
                }
            prev_season = season

        r_w = ratings.get(w, base)
        r_l = ratings.get(l, base)

        expected_w = 1 / (1 + 10 ** ((r_l - r_w) / 400))
        expected_l = 1 - expected_w

        ratings[w] = r_w + k * (1 - expected_w)
        ratings[l] = r_l + k * (0 - expected_l)

    # Save final season
    if prev_season is not None:
        for tid, elo in ratings.items():
            records.append({'Season': prev_season, 'TeamID': tid, 'Elo': elo})

    return pd.DataFrame(records)


def calculate_massey_rank(massey_df, pre_tourney_day=133):
    """
    For each team/season, compute a consensus ranking by averaging ordinal
    ranks across all rating systems at the latest snapshot on or before
    pre_tourney_day (default 133 ~ Selection Sunday).

    Lower MasseyRank = better team.
    Returns a DataFrame with columns: Season, TeamID, MasseyRank
    """
    df = massey_df[massey_df['RankingDayNum'] <= pre_tourney_day].copy()

    # Keep only the latest snapshot available per season
    latest_day = df.groupby('Season')['RankingDayNum'].max().reset_index()
    latest_day.columns = ['Season', 'LatestDay']
    df = df.merge(latest_day, on='Season')
    df = df[df['RankingDayNum'] == df['LatestDay']]

    consensus = df.groupby(['Season', 'TeamID'])['OrdinalRank'].mean().reset_index()
    consensus.columns = ['Season', 'TeamID', 'MasseyRank']
    return consensus


def calculate_sos(df):
    """
    Two-level strength of schedule.
    Level 1: average opponent win rate (who did you play?)
    Level 2: average of opponents' SOS (who did your opponents play?)
    """
    winners = df[['Season', 'WTeamID', 'LTeamID']].copy()
    winners.columns = ['Season', 'TeamID', 'OppID']

    losers = df[['Season', 'LTeamID', 'WTeamID']].copy()
    losers.columns = ['Season', 'TeamID', 'OppID']

    all_games = pd.concat([winners, losers], ignore_index=True)

    # --- Level 0: raw win rates ---
    win_flags = pd.concat([
        df[['Season', 'WTeamID']].assign(Win=1).rename(columns={'WTeamID': 'TeamID'}),
        df[['Season', 'LTeamID']].assign(Win=0).rename(columns={'LTeamID': 'TeamID'}),
    ])
    win_rates = win_flags.groupby(['Season', 'TeamID'])['Win'].mean().reset_index()
    win_rates.columns = ['Season', 'OppID', 'OppWinRate']

    # --- Level 1: SOS = mean opponent win rate ---
    l1 = all_games.merge(win_rates, on=['Season', 'OppID'], how='left')
    sos_l1 = l1.groupby(['Season', 'TeamID'])['OppWinRate'].mean().reset_index()
    sos_l1.columns = ['Season', 'OppID', 'OppSOS']  # rename for level 2 join

    # --- Level 2: SOS2 = mean of opponents' SOS ---
    l2 = all_games.merge(sos_l1, on=['Season', 'OppID'], how='left')
    sos_l2 = l2.groupby(['Season', 'TeamID'])['OppSOS'].mean().reset_index()
    sos_l2.columns = ['Season', 'TeamID', 'SOS2']

    # Rename sos_l1 back and join both levels together
    sos_l1.columns = ['Season', 'TeamID', 'SOS']
    sos_final = sos_l1.merge(sos_l2, on=['Season', 'TeamID'])

    return sos_final

def calculate_conf_tourney_performance(conf_tourney, regular_season):
    """
    Use actual conference tourney results instead of day number proxy.
    Merge with regular season detailed results to get scores.
    """
    # conf_tourney has Season, ConfAbbrev, DayNum, WTeamID, LTeamID
    # We need scores — join with regular season compact results on Season+DayNum+WTeamID+LTeamID
    conf_with_scores = conf_tourney.merge(
        regular_season[['Season', 'DayNum', 'WTeamID', 'LTeamID', 'WScore', 'LScore']],
        on=['Season', 'DayNum', 'WTeamID', 'LTeamID'],
        how='left'
    )

    winners = conf_with_scores[['Season', 'WTeamID', 'WScore', 'LScore']].copy()
    winners.columns = ['Season', 'TeamID', 'ScoreFor', 'ScoreAgainst']
    winners['Win'] = 1

    losers = conf_with_scores[['Season', 'LTeamID', 'LScore', 'WScore']].copy()
    losers.columns = ['Season', 'TeamID', 'ScoreFor', 'ScoreAgainst']
    losers['Win'] = 0

    all_games = pd.concat([winners, losers], ignore_index=True)

    conf_stats = all_games.groupby(['Season', 'TeamID']).agg(
        ConfWinRate = ('Win', 'mean'),
        ConfPtDiff  = ('ScoreFor', 'mean'),
        ConfAllowed = ('ScoreAgainst', 'mean'),
        ConfGames   = ('Win', 'count'),
    ).reset_index()

    conf_stats['ConfPtDiff'] = conf_stats['ConfPtDiff'] - conf_stats['ConfAllowed']
    conf_stats = conf_stats.drop(columns=['ConfAllowed'])

    return conf_stats


def return_X_y(seeds, tourney, team_stats, eff_stats, late_stats, sos, conf_stats, elo_ratings=None, massey_ranks=None):

    ts = team_stats.set_index(['Season', 'TeamID'])
    es = eff_stats.set_index(['Season', 'TeamID'])
    ls = late_stats.set_index(['Season', 'TeamID'])
    so = sos.set_index(['Season', 'TeamID'])
    cs = conf_stats.set_index(['Season', 'TeamID'])
    sd = seeds.set_index(['Season', 'TeamID'])['SeedNum']
    er = elo_ratings.set_index(['Season', 'TeamID'])['Elo'] if elo_ratings is not None else None
    mr = massey_ranks.set_index(['Season', 'TeamID'])['MasseyRank'] if massey_ranks is not None else None

    rows = []
    for _, game in tourney.iterrows():
        season = game['Season']
        winner, loser = game['WTeamID'], game['LTeamID']

        if np.random.rand() > 0.5:
            t1, t2, label = winner, loser, 1
        else:
            t1, t2, label = loser, winner, 0

        try:
            s1, s2 = ts.loc[(season, t1)], ts.loc[(season, t2)]
            e1, e2 = es.loc[(season, t1)], es.loc[(season, t2)]
            l1, l2 = ls.loc[(season, t1)], ls.loc[(season, t2)]
            o1, o2 = so.loc[(season, t1)], so.loc[(season, t2)]
            seed1, seed2 = sd.loc[(season, t1)], sd.loc[(season, t2)]
        except KeyError:
            continue

        # Conf stats may be missing for some teams — use 0 as neutral fallback
        def get_conf(cx, team, season, col):
            try:
                return cx.loc[(season, team)][col]
            except KeyError:
                return 0.0

        c1_wr = get_conf(cs, t1, season, 'ConfWinRate')
        c2_wr = get_conf(cs, t2, season, 'ConfWinRate')
        c1_pd = get_conf(cs, t1, season, 'ConfPtDiff')
        c2_pd = get_conf(cs, t2, season, 'ConfPtDiff')

        elo_diff = 0.0
        if er is not None:
            try:
                elo_diff = er.loc[(season, t1)] - er.loc[(season, t2)]
            except KeyError:
                pass

        massey_diff = 0.0
        if mr is not None:
            try:
                massey_diff = mr.loc[(season, t1)] - mr.loc[(season, t2)]
            except KeyError:
                pass

        rows.append({
            'SeedDiff'        : seed1 - seed2,
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
            'label'           : label,
        })

    df = pd.DataFrame(rows)
    feature_cols = [
        'SeedDiff', 'WinRateDiff', 'PtDiffDiff', 'NetEffDiff', 'TempoDiff',
        'WinRateMomentum', 'PtDiffMomentum', 'SOSDiff', 'AdjWinRateDiff', 'SOS2Diff',
        'EloDiff', 'MasseyRankDiff',
    ]
    X = df[feature_cols]
    y = df['label']

    return df, X, y
