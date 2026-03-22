import streamlit as st
import pandas as pd

st.set_page_config(page_title="March Mania 2026", page_icon="🏀", layout="wide")

@st.cache_data
def load_data():
    preds = pd.read_csv("predictions.csv")
    champ = pd.read_csv("bracket_sim.csv")
    return preds, champ

preds, champ = load_data()

# Build a unified lookup keyed by (min_id, max_id) -> P(min_id team wins)
lookup = {}
for _, row in preds.iterrows():
    a, b = row["TeamID_A"], row["TeamID_B"]
    p_a = row["P_A_wins"]
    lo, hi = (a, b) if a < b else (b, a)
    p_lo = p_a if a < b else 1 - p_a
    lookup[(lo, hi)] = p_lo

all_teams = sorted(preds["TeamName_A"].unique().tolist() +
                   [n for n in preds["TeamName_B"].unique() if n not in preds["TeamName_A"].values])
all_teams = sorted(set(all_teams))

name_to_id = {}
for _, row in preds.iterrows():
    name_to_id[row["TeamName_A"]] = row["TeamID_A"]
    name_to_id[row["TeamName_B"]] = row["TeamID_B"]

seed_map = {}
for _, row in preds.iterrows():
    seed_map[row["TeamName_A"]] = row["Seed_A"]
    seed_map[row["TeamName_B"]] = row["Seed_B"]

def get_prob(team1_name, team2_name):
    t1, t2 = name_to_id[team1_name], name_to_id[team2_name]
    lo, hi = min(t1, t2), max(t1, t2)
    p_lo = lookup.get((lo, hi))
    if p_lo is None:
        return None, None
    return (p_lo, 1 - p_lo) if t1 == lo else (1 - p_lo, p_lo)

# ── Sidebar nav ──────────────────────────────────────────────────────────────
st.sidebar.title("🏀 March Mania 2026")
page = st.sidebar.radio("View", ["Head-to-Head", "Team Matchups", "Championship Odds"])

# ── Page: Head-to-Head ───────────────────────────────────────────────────────
if page == "Head-to-Head":
    st.title("Head-to-Head Matchup")
    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox("Team 1", all_teams, index=all_teams.index("Duke") if "Duke" in all_teams else 0)
    with col2:
        team2 = st.selectbox("Team 2", [t for t in all_teams if t != team1],
                             index=0)

    if team1 and team2:
        p1, p2 = get_prob(team1, team2)
        if p1 is not None:
            s1, s2 = seed_map.get(team1, "?"), seed_map.get(team2, "?")

            st.divider()
            c1, c2 = st.columns(2)
            with c1:
                st.metric(f"#{s1} {team1}", f"{p1:.1%}")
                st.progress(float(p1))
            with c2:
                st.metric(f"#{s2} {team2}", f"{p2:.1%}")
                st.progress(float(p2))

            favorite = team1 if p1 > p2 else team2
            fav_prob = max(p1, p2)
            st.info(f"Model favors **{favorite}** with **{fav_prob:.1%}** win probability.")
        else:
            st.warning("Matchup not found in predictions.")

# ── Page: Team Matchups ──────────────────────────────────────────────────────
elif page == "Team Matchups":
    st.title("All Matchups for a Team")
    team = st.selectbox("Select team", all_teams, index=all_teams.index("Duke") if "Duke" in all_teams else 0)

    if team:
        rows = []
        for other in all_teams:
            if other == team:
                continue
            p_win, _ = get_prob(team, other)
            if p_win is not None:
                rows.append({
                    "Opponent": other,
                    "Opp Seed": seed_map.get(other, "?"),
                    f"{team} Win%": f"{p_win:.1%}",
                    "_sort": p_win,
                })

        df = pd.DataFrame(rows).sort_values("_sort", ascending=False).drop(columns="_sort").reset_index(drop=True)
        df.index += 1

        st.write(f"**{team}** (Seed #{seed_map.get(team, '?')}) — {len(df)} matchups")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Most Likely Wins")
            st.dataframe(df.head(20), use_container_width=True)
        with col2:
            st.subheader("Toughest Matchups")
            st.dataframe(df.tail(20).iloc[::-1].reset_index(drop=True), use_container_width=True)

# ── Page: Championship Odds ──────────────────────────────────────────────────
elif page == "Championship Odds":
    st.title("Championship Odds (Monte Carlo Simulation)")
    st.caption("Based on 10,000 simulated tournaments using model win probabilities.")

    display = champ[["TeamName", "Seed", "ChampProb"]].copy()
    display["ChampProb"] = display["ChampProb"].apply(lambda x: f"{x:.1%}")
    display.index = range(1, len(display) + 1)
    display.columns = ["Team", "Seed", "Championship Prob"]

    st.dataframe(display, use_container_width=True, height=600)
