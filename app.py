import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="March Mania 2026", page_icon="🏀", layout="wide")

@st.cache_data
def load_data():
    preds_m = pd.read_csv("predictions.csv")
    champ_m = pd.read_csv("bracket_sim.csv")
    preds_w = pd.read_csv("predictions_w.csv")
    champ_w = pd.read_csv("bracket_sim_w.csv")
    seeds_m = pd.read_csv("march-machine-learning-mania-2026_2/MNCAATourneySeeds.csv")
    slots_m = pd.read_csv("march-machine-learning-mania-2026_2/MNCAATourneySlots.csv")
    seeds_w = pd.read_csv("march-machine-learning-mania-2026_2/WNCAATourneySeeds.csv")
    slots_w = pd.read_csv("march-machine-learning-mania-2026_2/WNCAATourneySlots.csv")
    return preds_m, champ_m, preds_w, champ_w, seeds_m, slots_m, seeds_w, slots_w

preds_m, champ_m, preds_w, champ_w, seeds_m, slots_m, seeds_w, slots_w = load_data()

# ── Existing lookups (used by pages 1-3, men's only) ─────────────────────────
preds = preds_m
champ = champ_m

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

# ── Actual tournament results (completed games) ───────────────────────────────
# Slot codes: W=East, X=South, Y=Midwest, Z=West (men's)
#             W=Regional1/UConn, X=Regional4/SCarolina, Y=Regional3/Texas, Z=Regional2/UCLA (women's)
ACTUAL_RESULTS = {
    "M": {
        # Play-ins
        "X16": "Prairie View", "Y11": "Miami OH", "Y16": "Howard", "Z11": "Texas",
        # East (W) — R1
        "R1W1": "Duke",        "R1W2": "Connecticut",  "R1W3": "Michigan St", "R1W4": "Kansas",
        "R1W5": "St John's",   "R1W6": "Louisville",   "R1W7": "UCLA",        "R1W8": "TCU",
        # East (W) — R2
        "R2W1": "Duke",        "R2W2": "Connecticut",  "R2W3": "Michigan St", "R2W4": "St John's",
        # South (X) — R1
        "R1X1": "Florida",     "R1X2": "Houston",      "R1X3": "Illinois",    "R1X4": "Nebraska",
        "R1X5": "Vanderbilt",  "R1X6": "VCU",          "R1X7": "Texas A&M",   "R1X8": "Iowa",
        # South (X) — R2
        "R2X1": "Iowa",        "R2X2": "Houston",      "R2X3": "Illinois",    "R2X4": "Nebraska",
        # Midwest (Y) — R1
        "R1Y1": "Michigan",    "R1Y2": "Iowa St",      "R1Y3": "Virginia",    "R1Y4": "Alabama",
        "R1Y5": "Texas Tech",  "R1Y6": "Tennessee",    "R1Y7": "Kentucky",    "R1Y8": "St Louis",
        # Midwest (Y) — R2
        "R2Y1": "Michigan",    "R2Y2": "Iowa St",      "R2Y3": "Tennessee",   "R2Y4": "Alabama",
        # West (Z) — R1
        "R1Z1": "Arizona",     "R1Z2": "Purdue",       "R1Z3": "Gonzaga",     "R1Z4": "Arkansas",
        "R1Z5": "High Point",  "R1Z6": "Texas",        "R1Z7": "Miami FL",    "R1Z8": "Utah St",
        # West (Z) — R2
        "R2Z1": "Arizona",     "R2Z2": "Purdue",       "R2Z3": "Texas",       "R2Z4": "Arkansas",
    },
    "W": {
        # Play-ins
        "X10": "Virginia", "X16": "Southern Univ", "Y16": "Missouri St", "Z11": "Nebraska",
        # Regional 1 / Connecticut (W) — R1
        "R1W1": "Connecticut",    "R1W2": "Vanderbilt",      "R1W3": "Ohio St",      "R1W4": "North Carolina",
        "R1W5": "Maryland",       "R1W6": "Notre Dame",      "R1W7": "Illinois",     "R1W8": "Syracuse",
        # R2
        "R2W1": "Connecticut",    "R2W2": "Vanderbilt",      "R2W3": "Notre Dame",   "R2W4": "North Carolina",
        # Regional 4 / South Carolina (X) — R1
        "R1X1": "South Carolina", "R1X2": "Iowa",            "R1X3": "TCU",          "R1X4": "Oklahoma",
        "R1X5": "Michigan St",    "R1X6": "Washington",      "R1X7": "Virginia",     "R1X8": "USC",
        # R2
        "R2X1": "South Carolina", "R2X2": "Virginia",        "R2X3": "TCU",          "R2X4": "Oklahoma",
        # Regional 3 / Texas (Y) — R1
        "R1Y1": "Texas",          "R1Y2": "Michigan",        "R1Y3": "Louisville",   "R1Y4": "West Virginia",
        "R1Y5": "Kentucky",       "R1Y6": "Alabama",         "R1Y7": "NC State",     "R1Y8": "Oregon",
        # R2
        "R2Y1": "Texas",          "R2Y2": "Michigan",        "R2Y3": "Louisville",   "R2Y4": "Kentucky",
        # Regional 2 / UCLA (Z) — R1
        "R1Z1": "UCLA",           "R1Z2": "LSU",             "R1Z3": "Duke",         "R1Z4": "Minnesota",
        "R1Z5": "Mississippi",    "R1Z6": "Baylor",          "R1Z7": "Texas Tech",   "R1Z8": "Oklahoma St",
        # R2
        "R2Z1": "UCLA",           "R2Z2": "LSU",             "R2Z3": "Duke",         "R2Z4": "Minnesota",
    },
}

# ── Bracket data (gender-parameterized) ──────────────────────────────────────
def build_gender_data(preds_df, champ_df, seeds_df, slots_df):
    lo = preds_df[["TeamID_A", "TeamID_B"]].min(axis=1)
    hi = preds_df[["TeamID_A", "TeamID_B"]].max(axis=1)
    p_lo = np.where(preds_df["TeamID_A"] < preds_df["TeamID_B"],
                    preds_df["P_A_wins"], 1 - preds_df["P_A_wins"])
    blookup = dict(zip(zip(lo, hi), p_lo))

    id_to_name = dict(zip(preds_df["TeamID_A"], preds_df["TeamName_A"]))
    id_to_name.update(dict(zip(preds_df["TeamID_B"], preds_df["TeamName_B"])))
    bname_to_id = {v: k for k, v in id_to_name.items()}

    champ_prob_map = dict(zip(champ_df["TeamID"], champ_df["ChampProb"]))

    s2026 = seeds_df[seeds_df["Season"] == 2026].copy()
    s2026["SeedNum"] = s2026["Seed"].str.extract(r"(\d+)")[0].astype(int)
    seed_to_team = s2026.set_index("Seed")["TeamID"].to_dict()
    team_to_seednum = s2026.set_index("TeamID")["SeedNum"].to_dict()
    sl2026 = slots_df[slots_df["Season"] == 2026].copy()

    return blookup, id_to_name, bname_to_id, champ_prob_map, seed_to_team, team_to_seednum, sl2026

gender_data = {
    "M": build_gender_data(preds_m, champ_m, seeds_m, slots_m),
    "W": build_gender_data(preds_w, champ_w, seeds_w, slots_w),
}

def resolve_bracket_for_display(gender):
    blookup, id_to_name, bname_to_id, champ_prob_map, seed_to_team, team_to_seednum, sl2026 = gender_data[gender]
    actual = ACTUAL_RESULTS[gender]
    slot_winner = {}
    matchup_info = {}

    for _ in range(10):
        progress = False
        for _, row in sl2026.iterrows():
            slot = row["Slot"]
            if slot in slot_winner:
                continue
            t1 = seed_to_team.get(row["StrongSeed"]) or slot_winner.get(row["StrongSeed"])
            t2 = seed_to_team.get(row["WeakSeed"])   or slot_winner.get(row["WeakSeed"])
            if t1 is None or t2 is None:
                continue

            s1 = team_to_seednum.get(t1, 99)
            s2 = team_to_seednum.get(t2, 99)
            lo_id, hi_id = min(t1, t2), max(t1, t2)
            p_lo = blookup.get((lo_id, hi_id), 0.5)
            prob_t1 = p_lo if t1 == lo_id else 1 - p_lo

            if slot in actual:
                winner = bname_to_id.get(actual[slot], t1 if prob_t1 >= 0.5 else t2)
                is_final = True
            else:
                winner = t1 if prob_t1 >= 0.5 else t2
                is_final = False

            loser = t2 if winner == t1 else t1
            winner_seed = s1 if winner == t1 else s2
            loser_seed  = s2 if winner == t1 else s1
            winner_prob = prob_t1 if winner == t1 else 1 - prob_t1

            slot_winner[slot] = winner
            matchup_info[slot] = {
                "winner_name":       id_to_name.get(winner, str(winner)),
                "winner_seed":       winner_seed,
                "loser_name":        id_to_name.get(loser, str(loser)),
                "loser_seed":        loser_seed,
                "winner_prob":       winner_prob,
                "is_upset":          winner_seed > loser_seed,
                "is_final":          is_final,
                "champ_prob_winner": champ_prob_map.get(winner, 0),
            }
            progress = True
        if not progress:
            break

    return slot_winner, matchup_info


def render_bracket_html(matchup_info):
    css = """
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, -apple-system, sans-serif; font-size: 11px; background: #f0f2f5; }
.bracket-outer { display: flex; flex-direction: column; min-width: 1280px; padding: 6px; gap: 4px; }
.bracket-row { display: flex; align-items: stretch; gap: 3px; }
.region { display: flex; flex-direction: column; flex: 4; min-width: 0; }
.region-label { text-align: center; font-weight: 700; font-size: 12px; padding: 3px 6px;
                background: #1a237e; color: #fff; border-radius: 3px; margin-bottom: 3px; }
.rounds-container { display: flex; flex: 1; gap: 2px; }
.round { display: flex; flex-direction: column; justify-content: space-around; flex: 1; gap: 1px; min-width: 0; }
.matchup-card { border: 1px solid #ccc; border-radius: 3px; overflow: hidden; background: #fff; }
.matchup-card.final { border-color: #4caf50; }
.team-row { display: flex; align-items: center; padding: 2px 3px; gap: 3px; min-height: 19px; }
.team-row.winner { background: #fff; font-weight: 600; }
.team-row.loser  { background: #f7f7f7; color: #aaa; }
.team-row.upset  { background: #fff8e1 !important; color: #5d4037 !important; }
.seed { min-width: 14px; text-align: right; font-size: 10px; color: #888; flex-shrink: 0; }
.winner .seed, .upset .seed { color: #333; }
.tname { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.pct  { min-width: 28px; text-align: right; font-size: 9px; color: #777; flex-shrink: 0; }
.winner .pct, .upset .pct { color: #333; }
.cpct { min-width: 28px; text-align: right; font-size: 9px; color: #1565c0; flex-shrink: 0; }
.check { min-width: 14px; text-align: right; font-size: 10px; color: #4caf50; flex-shrink: 0; font-weight: 700; }
.div  { height: 1px; background: #e8e8e8; }
.center-col { display: flex; flex-direction: column; justify-content: center; align-items: center;
              flex: 1; min-width: 140px; gap: 4px; }
.ff-label { font-size: 11px; font-weight: 700; color: #444; text-align: center; margin-bottom: 2px;
            text-transform: uppercase; letter-spacing: 0.5px; }
.champ-row { display: flex; justify-content: center; padding: 5px; background: #fff;
             border: 1px solid #e0c060; border-radius: 4px; }
.champ-title { font-size: 12px; font-weight: 700; color: #b8860b; text-align: center;
               margin-bottom: 3px; text-transform: uppercase; letter-spacing: 0.5px; }
.champ-box { background: #fffde7; border: 2px solid #f9a825; border-radius: 4px; padding: 5px 12px;
             font-weight: 700; font-size: 13px; display: inline-block; }
</style>
"""

    def make_card(slot):
        info = matchup_info.get(slot)
        if info is None:
            return ('<div class="matchup-card">'
                    '<div class="team-row"><span class="tname" style="color:#ccc;font-size:10px">TBD</span></div>'
                    '<div class="div"></div>'
                    '<div class="team-row"><span class="tname" style="color:#ccc;font-size:10px">TBD</span></div>'
                    '</div>')

        wn, ws = info["winner_name"], info["winner_seed"]
        ln, ls = info["loser_name"],  info["loser_seed"]
        wp     = info["winner_prob"]
        upset  = " upset" if info["is_upset"] else ""

        if info["is_final"]:
            w_extra = '<span class="check">✓</span>'
            l_extra = ""
            card_cls = " final"
        else:
            cp = f'{info["champ_prob_winner"]:.1%}'
            w_extra = f'<span class="pct">{wp:.0%}</span><span class="cpct">{cp}</span>'
            l_extra = f'<span class="pct">{1-wp:.0%}</span>'
            card_cls = ""

        return (f'<div class="matchup-card{card_cls}">'
                f'<div class="team-row winner{upset}">'
                f'<span class="seed">{ws}</span>'
                f'<span class="tname">{wn}</span>'
                f'{w_extra}'
                f'</div>'
                f'<div class="div"></div>'
                f'<div class="team-row loser">'
                f'<span class="seed">{ls}</span>'
                f'<span class="tname">{ln}</span>'
                f'{l_extra}'
                f'</div>'
                f'</div>')

    counts = {"R1": 8, "R2": 4, "R3": 2, "R4": 1}

    def make_round_col(region, rnd):
        n = counts[rnd]
        cards = "".join(make_card(f"{rnd}{region}{i}") for i in range(1, n + 1))
        return f'<div class="round">{cards}</div>'

    def make_region(code, label, reverse=False):
        rnd_order = ["R4", "R3", "R2", "R1"] if reverse else ["R1", "R2", "R3", "R4"]
        rounds_html = "".join(make_round_col(code, r) for r in rnd_order)
        return (f'<div class="region">'
                f'<div class="region-label">{label}</div>'
                f'<div class="rounds-container">{rounds_html}</div>'
                f'</div>')

    def make_ff(slot):
        return (f'<div style="text-align:center;width:140px">'
                f'<div class="ff-label">Final Four</div>'
                f'{make_card(slot)}'
                f'</div>')

    champ_info = matchup_info.get("R6CH")
    if champ_info:
        cn, cs = champ_info["winner_name"], champ_info["winner_seed"]
        if champ_info["is_final"]:
            champ_box = f'<div class="champ-box">🏆 #{cs} {cn}</div>'
        else:
            cp_pct = f'{champ_info["champ_prob_winner"]:.1%}'
            champ_box = f'<div class="champ-box">🏆 #{cs} {cn} <span style="font-size:11px;color:#888">({cp_pct})</span></div>'
    else:
        champ_box = '<div class="champ-box">TBD</div>'

    return (css
            + '<div class="bracket-outer">'

            + '<div class="bracket-row">'
            + make_region("W", "East" if "W01" in str(gender_data) else "Regional 1")
            + f'<div class="center-col">{make_ff("R5WX")}</div>'
            + make_region("X", "South" if "W01" in str(gender_data) else "Regional 4", reverse=True)
            + "</div>"

            + '<div class="champ-row"><div style="text-align:center">'
            + '<div class="champ-title">Championship</div>'
            + champ_box
            + "</div></div>"

            + '<div class="bracket-row">'
            + make_region("Y", "Midwest" if "W01" in str(gender_data) else "Regional 3")
            + f'<div class="center-col">{make_ff("R5YZ")}</div>'
            + make_region("Z", "West" if "W01" in str(gender_data) else "Regional 2", reverse=True)
            + "</div>"

            + "</div>")


# ── Sidebar nav ──────────────────────────────────────────────────────────────
st.sidebar.title("🏀 March Mania 2026")
page = st.sidebar.radio("View", ["Head-to-Head", "Team Matchups", "Championship Odds", "Bracket View"])

# ── Page: Head-to-Head ───────────────────────────────────────────────────────
if page == "Head-to-Head":
    st.title("Head-to-Head Matchup")
    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox("Team 1", all_teams, index=all_teams.index("Duke") if "Duke" in all_teams else 0)
    with col2:
        team2 = st.selectbox("Team 2", [t for t in all_teams if t != team1], index=0)

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

# ── Page: Bracket View ───────────────────────────────────────────────────────
elif page == "Bracket View":
    st.title("2026 Tournament Bracket")
    gender = st.radio("Tournament", ["Men's", "Women's"], horizontal=True)
    g = "M" if gender == "Men's" else "W"
    st.caption("✓ = actual result  |  % = model win probability  |  Blue % = championship odds  |  Amber = upset")

    _, matchup_info = resolve_bracket_for_display(g)

    # Pass gender label to HTML renderer via a simple wrapper
    def render(mi, g):
        region_names = (
            {"W": "East", "X": "South", "Y": "Midwest", "Z": "West"}
            if g == "M" else
            {"W": "Regional 1", "X": "Regional 4", "Y": "Regional 3", "Z": "Regional 2"}
        )

        css = """
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, -apple-system, sans-serif; font-size: 11px; background: #f0f2f5; }
.bracket-outer { display: flex; flex-direction: column; min-width: 1280px; padding: 6px; gap: 4px; }
.bracket-row { display: flex; align-items: stretch; gap: 3px; }
.region { display: flex; flex-direction: column; flex: 4; min-width: 0; }
.region-label { text-align: center; font-weight: 700; font-size: 12px; padding: 3px 6px;
                background: #1a237e; color: #fff; border-radius: 3px; margin-bottom: 3px; }
.rounds-container { display: flex; flex: 1; gap: 2px; }
.round { display: flex; flex-direction: column; justify-content: space-around; flex: 1; gap: 1px; min-width: 0; }
.matchup-card { border: 1px solid #ccc; border-radius: 3px; overflow: hidden; background: #fff; }
.matchup-card.final { border-color: #4caf50; }
.team-row { display: flex; align-items: center; padding: 2px 3px; gap: 3px; min-height: 19px; }
.team-row.winner { background: #fff; font-weight: 600; }
.team-row.loser  { background: #f7f7f7; color: #aaa; }
.team-row.upset  { background: #fff8e1 !important; color: #5d4037 !important; }
.seed  { min-width: 14px; text-align: right; font-size: 10px; color: #888; flex-shrink: 0; }
.winner .seed, .upset .seed { color: #333; }
.tname { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.pct   { min-width: 28px; text-align: right; font-size: 9px; color: #777; flex-shrink: 0; }
.winner .pct, .upset .pct { color: #333; }
.cpct  { min-width: 28px; text-align: right; font-size: 9px; color: #1565c0; flex-shrink: 0; }
.check { min-width: 14px; text-align: right; font-size: 10px; color: #4caf50; flex-shrink: 0; font-weight: 700; }
.matchup-card.wrong { border: 2px solid #f44336; }
.div   { height: 1px; background: #e8e8e8; }
.center-col { display: flex; flex-direction: column; justify-content: center; align-items: center;
              flex: 1; min-width: 140px; gap: 4px; }
.ff-label { font-size: 11px; font-weight: 700; color: #444; text-align: center; margin-bottom: 2px;
            text-transform: uppercase; letter-spacing: 0.5px; }
.champ-row { display: flex; justify-content: center; padding: 5px; background: #fff;
             border: 1px solid #e0c060; border-radius: 4px; }
.champ-title { font-size: 12px; font-weight: 700; color: #b8860b; text-align: center;
               margin-bottom: 3px; text-transform: uppercase; letter-spacing: 0.5px; }
.champ-box { background: #fffde7; border: 2px solid #f9a825; border-radius: 4px;
             padding: 5px 12px; font-weight: 700; font-size: 13px; display: inline-block; }
</style>
"""

        def make_card(slot):
            info = mi.get(slot)
            if info is None:
                return ('<div class="matchup-card">'
                        '<div class="team-row"><span class="tname" style="color:#ccc;font-size:10px">TBD</span></div>'
                        '<div class="div"></div>'
                        '<div class="team-row"><span class="tname" style="color:#ccc;font-size:10px">TBD</span></div>'
                        '</div>')
            wn, ws = info["winner_name"], info["winner_seed"]
            ln, ls = info["loser_name"],  info["loser_seed"]
            wp     = info["winner_prob"]
            upset  = " upset" if info["is_upset"] else ""
            if info["is_final"] and wp >= 0.5:
                # Model was correct — show probabilities too
                w_extra  = f'<span class="pct">{wp:.0%}</span><span class="check">✓</span>'
                l_extra  = f'<span class="pct">{1-wp:.0%}</span>'
                card_cls = " final"
            elif info["is_final"] and wp < 0.5:
                # Model was wrong — red border, show probabilities
                w_extra  = f'<span class="pct">{wp:.0%}</span>'
                l_extra  = f'<span class="pct">{1-wp:.0%}</span>'
                card_cls = " wrong"
            else:
                cp = f'{info["champ_prob_winner"]:.1%}'
                w_extra  = f'<span class="pct">{wp:.0%}</span><span class="cpct">{cp}</span>'
                l_extra  = f'<span class="pct">{1-wp:.0%}</span>'
                card_cls = ""
            return (f'<div class="matchup-card{card_cls}">'
                    f'<div class="team-row winner{upset}">'
                    f'<span class="seed">{ws}</span><span class="tname">{wn}</span>{w_extra}'
                    f'</div><div class="div"></div>'
                    f'<div class="team-row loser">'
                    f'<span class="seed">{ls}</span><span class="tname">{ln}</span>{l_extra}'
                    f'</div></div>')

        # Bracket display order (top→bottom): mirrors actual NCAA bracket structure
        SLOT_ORDER = {
            "R1": [1, 8, 5, 4, 6, 3, 7, 2],
            "R2": [1, 4, 3, 2],
            "R3": [1, 2],
            "R4": [1],
        }

        def make_round_col(region, rnd):
            cards = "".join(make_card(f"{rnd}{region}{i}") for i in SLOT_ORDER[rnd])
            return f'<div class="round">{cards}</div>'

        def make_region(code, reverse=False):
            label = region_names[code]
            order = ["R4", "R3", "R2", "R1"] if reverse else ["R1", "R2", "R3", "R4"]
            rounds_html = "".join(make_round_col(code, r) for r in order)
            return (f'<div class="region"><div class="region-label">{label}</div>'
                    f'<div class="rounds-container">{rounds_html}</div></div>')

        def make_ff(slot):
            return (f'<div style="text-align:center;width:140px">'
                    f'<div class="ff-label">Final Four</div>{make_card(slot)}</div>')

        champ_info = mi.get("R6CH")
        if champ_info:
            cn, cs = champ_info["winner_name"], champ_info["winner_seed"]
            if champ_info["is_final"]:
                champ_box = f'<div class="champ-box">🏆 #{cs} {cn}</div>'
            else:
                cp_pct = f'{champ_info["champ_prob_winner"]:.1%}'
                champ_box = (f'<div class="champ-box">🏆 #{cs} {cn} '
                             f'<span style="font-size:11px;color:#888">({cp_pct})</span></div>')
        else:
            champ_box = '<div class="champ-box">TBD</div>'

        return (css + '<div class="bracket-outer">'
                + '<div class="bracket-row">'
                + make_region("W") + f'<div class="center-col">{make_ff("R5WX")}</div>' + make_region("X", reverse=True)
                + "</div>"
                + '<div class="champ-row"><div style="text-align:center">'
                + '<div class="champ-title">Championship</div>' + champ_box + "</div></div>"
                + '<div class="bracket-row">'
                + make_region("Y") + f'<div class="center-col">{make_ff("R5YZ")}</div>' + make_region("Z", reverse=True)
                + "</div></div>")

    st.components.v1.html(render(matchup_info, g), height=960, scrolling=True)
