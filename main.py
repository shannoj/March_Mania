import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score


regular_season = pd.read_csv('march-machine-learning-mania-2026/MRegularSeasonCompactResults.csv')

tournament_results = pd.read_csv('march-machine-learning-mania-2026/MNCAATourneyCompactResults.csv')

tourney = pd.read_csv('march-machine-learning-mania-2026/MNCAATourneyDetailedResults.csv')

seeds = pd.read_csv('march-machine-learning-mania-2026/MNCAATourneySeeds.csv')

np.random.seed(42)


def calculate_team_stats(df):

    winners = df[['Season', 'WTeamID', 'WScore', 'LScore']].copy()
    winners.columns = ['Season', 'TeamID', 'ScoreFor', 'ScoreAgainst']
    winners['Win'] = 1

    losers = regular_season[['Season', 'LTeamID', 'LScore', 'WScore']].copy()
    losers.columns = ['Season', 'TeamID', 'ScoreFor', 'ScoreAgainst']
    losers['Win'] = 0

    all_games = pd.concat([winners, losers], ignore_index=True)

    team_stats = all_games.groupby(['Season', 'TeamID']).agg(
        WinRate   = ('Win', 'mean'),
        AvgScore  = ('ScoreFor', 'mean'),
        AvgAllowed = ('ScoreAgainst', 'mean'),
    ).reset_index()

    team_stats['AvgPointDiff'] = team_stats['AvgScore'] - team_stats['AvgAllowed']

    return team_stats

# For each tourney game, we need both teams' stats
# Randomly assign winner/loser to team1/team2 so model doesn't learn position bias
def return_X_y(seeds, tourney, team_stats):
    
    rows = []
    for _, game in tourney.iterrows():
        season = game['Season']
        winner, loser = game['WTeamID'], game['LTeamID']
        
        # Randomly decide which team is "team1"
        if np.random.rand() > 0.5:
            t1, t2, label = winner, loser, 1  # team1 won
        else:
            t1, t2, label = loser, winner, 0  # team1 lost

        # Look up stats for each team that season
        s1 = team_stats[(team_stats['Season'] == season) & (team_stats['TeamID'] == t1)]
        s2 = team_stats[(team_stats['Season'] == season) & (team_stats['TeamID'] == t2)]
        
        seed1 = seeds[(seeds['Season'] == season) & (seeds['TeamID'] == t1)]['SeedNum'].values[0]
        seed2 = seeds[(seeds['Season'] == season) & (seeds['TeamID'] == t2)]['SeedNum'].values[0]

        rows.append({
            'SeedDiff'    : seed1 - seed2,
            'WinRateDiff' : s1['WinRate'].values[0]      - s2['WinRate'].values[0],
            'PtDiffDiff'  : s1['AvgPointDiff'].values[0] - s2['AvgPointDiff'].values[0],
            'label'       : label
        })

    df = pd.DataFrame(rows)
    X = df[['SeedDiff', 'WinRateDiff', 'PtDiffDiff']]
    y = df['label']

    return df, X, y

# Extract seed number from e.g. 'W01' -> 1
seeds['SeedNum'] = seeds['Seed'].str[1:3].astype(int)

team_stats = calculate_team_stats(regular_season)

df, X, y = return_X_y(seeds, tourney, team_stats)

print(X)

print(y)

# Scale features - important for logistic regression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_scaled, y)

# Check how good it is with cross-validation
cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='neg_log_loss')
print(f"Log Loss: {-cv_scores.mean():.4f}")

# See which features matter most
for name, coef in zip(X.columns, model.coef_[0]):
    print(f"{name}: {coef:.4f}")