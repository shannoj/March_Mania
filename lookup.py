#!/usr/bin/env python3
"""Look up predictions by team name.

Usage:
  python lookup.py <team_name>               # show all games involving this team
  python lookup.py <team1> vs <team2>        # show prediction for specific matchup
  python lookup.py <team1> <team2>           # same as above
"""

import sys
import pandas as pd

DATA = "march-machine-learning-mania-2026"
SUBMISSION = "submission.csv"

teams = pd.read_csv(f"{DATA}/MTeams.csv")[["TeamID", "TeamName"]]
sub = pd.read_csv(SUBMISSION)

# Parse ID into season, team1, team2
sub[["Season", "T1", "T2"]] = sub["ID"].str.split("_", expand=True).astype(int)

# Join team names
sub = sub.merge(teams.rename(columns={"TeamID": "T1", "TeamName": "Team1"}), on="T1")
sub = sub.merge(teams.rename(columns={"TeamID": "T2", "TeamName": "Team2"}), on="T2")

def find_team(name):
    # Try exact match first, then word-start match, then substring
    lower = name.lower()
    exact = teams[teams["TeamName"].str.lower() == lower]
    if len(exact) == 1:
        return exact.iloc[0]["TeamName"]
    word = teams[teams["TeamName"].str.lower().str.startswith(lower)]
    if len(word) == 1:
        return word.iloc[0]["TeamName"]
    matches = word if not word.empty else teams[teams["TeamName"].str.contains(name, case=False)]
    if matches.empty:
        print(f"No team found matching '{name}'")
        sys.exit(1)
    if len(matches) > 1:
        print(f"Multiple teams match '{name}':")
        print(matches[["TeamID", "TeamName"]].to_string(index=False))
        sys.exit(1)
    return matches.iloc[0]["TeamName"]

args = [a for a in sys.argv[1:] if a.lower() != "vs"]

if len(args) == 0:
    print(__doc__)
    sys.exit(0)

elif len(args) == 1:
    team = find_team(args[0])
    mask = (sub["Team1"] == team) | (sub["Team2"] == team)
    results = sub[mask].copy()
    # Show prob that the matched team wins
    results["WinProb"] = results.apply(
        lambda r: r["Pred"] if r["Team1"] == team else 1 - r["Pred"], axis=1
    )
    results["Opponent"] = results.apply(
        lambda r: r["Team2"] if r["Team1"] == team else r["Team1"], axis=1
    )
    results = results[["Opponent", "WinProb"]].sort_values("WinProb", ascending=False)
    print(f"\n{team} win probability vs each opponent:\n")
    print(results.to_string(index=False))

elif len(args) == 2:
    team1 = find_team(args[0])
    team2 = find_team(args[1])
    row = sub[((sub["Team1"] == team1) & (sub["Team2"] == team2)) |
              ((sub["Team1"] == team2) & (sub["Team2"] == team1))]
    if row.empty:
        print(f"No matchup found between {team1} and {team2}")
        sys.exit(1)
    row = row.iloc[0]
    if row["Team1"] == team1:
        p1, p2 = row["Pred"], 1 - row["Pred"]
    else:
        p1, p2 = 1 - row["Pred"], row["Pred"]
    print(f"\n{team1}: {p1:.1%}  vs  {team2}: {p2:.1%}")

else:
    print("Too many arguments. See usage above.")
    sys.exit(1)
