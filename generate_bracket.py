#!/usr/bin/env python3
"""Generate a standalone bracket.html. Run: python generate_bracket.py"""

import json, webbrowser
import numpy as np
import pandas as pd

# ── Load data ─────────────────────────────────────────────────────────────────
preds_m = pd.read_csv("predictions.csv")
champ_m = pd.read_csv("bracket_sim.csv")
preds_w = pd.read_csv("predictions_w.csv")
champ_w = pd.read_csv("bracket_sim_w.csv")
seeds_m = pd.read_csv("march-machine-learning-mania-2026_2/MNCAATourneySeeds.csv")
slots_m = pd.read_csv("march-machine-learning-mania-2026_2/MNCAATourneySlots.csv")
seeds_w = pd.read_csv("march-machine-learning-mania-2026_2/WNCAATourneySeeds.csv")
slots_w = pd.read_csv("march-machine-learning-mania-2026_2/WNCAATourneySlots.csv")

# ── Actual results ─────────────────────────────────────────────────────────────
ACTUAL_RESULTS = {
    "M": {
        "X16": "Prairie View", "Y11": "Miami OH", "Y16": "Howard", "Z11": "Texas",
        "R1W1": "Duke",        "R1W2": "Connecticut",  "R1W3": "Michigan St", "R1W4": "Kansas",
        "R1W5": "St John's",   "R1W6": "Louisville",   "R1W7": "UCLA",        "R1W8": "TCU",
        "R2W1": "Duke",        "R2W2": "Connecticut",  "R2W3": "Michigan St", "R2W4": "St John's",
        "R1X1": "Florida",     "R1X2": "Houston",      "R1X3": "Illinois",    "R1X4": "Nebraska",
        "R1X5": "Vanderbilt",  "R1X6": "VCU",          "R1X7": "Texas A&M",   "R1X8": "Iowa",
        "R2X1": "Iowa",        "R2X2": "Houston",      "R2X3": "Illinois",    "R2X4": "Nebraska",
        "R1Y1": "Michigan",    "R1Y2": "Iowa St",      "R1Y3": "Virginia",    "R1Y4": "Alabama",
        "R1Y5": "Texas Tech",  "R1Y6": "Tennessee",    "R1Y7": "Kentucky",    "R1Y8": "St Louis",
        "R2Y1": "Michigan",    "R2Y2": "Iowa St",      "R2Y3": "Tennessee",   "R2Y4": "Alabama",
        "R1Z1": "Arizona",     "R1Z2": "Purdue",       "R1Z3": "Gonzaga",     "R1Z4": "Arkansas",
        "R1Z5": "High Point",  "R1Z6": "Texas",        "R1Z7": "Miami FL",    "R1Z8": "Utah St",
        "R2Z1": "Arizona",     "R2Z2": "Purdue",       "R2Z3": "Texas",       "R2Z4": "Arkansas",
    },
    "W": {
        "X10": "Virginia", "X16": "Southern Univ", "Y16": "Missouri St", "Z11": "Nebraska",
        "R1W1": "Connecticut",    "R1W2": "Vanderbilt",      "R1W3": "Ohio St",      "R1W4": "North Carolina",
        "R1W5": "Maryland",       "R1W6": "Notre Dame",      "R1W7": "Illinois",     "R1W8": "Syracuse",
        "R2W1": "Connecticut",    "R2W2": "Vanderbilt",      "R2W3": "Notre Dame",   "R2W4": "North Carolina",
        "R1X1": "South Carolina", "R1X2": "Iowa",            "R1X3": "TCU",          "R1X4": "Oklahoma",
        "R1X5": "Michigan St",    "R1X6": "Washington",      "R1X7": "Virginia",     "R1X8": "USC",
        "R2X1": "South Carolina", "R2X2": "Virginia",        "R2X3": "TCU",          "R2X4": "Oklahoma",
        "R1Y1": "Texas",          "R1Y2": "Michigan",        "R1Y3": "Louisville",   "R1Y4": "West Virginia",
        "R1Y5": "Kentucky",       "R1Y6": "Alabama",         "R1Y7": "NC State",     "R1Y8": "Oregon",
        "R2Y1": "Texas",          "R2Y2": "Michigan",        "R2Y3": "Louisville",   "R2Y4": "Kentucky",
        "R1Z1": "UCLA",           "R1Z2": "LSU",             "R1Z3": "Duke",         "R1Z4": "Minnesota",
        "R1Z5": "Mississippi",    "R1Z6": "Baylor",          "R1Z7": "Texas Tech",   "R1Z8": "Oklahoma St",
        "R2Z1": "UCLA",           "R2Z2": "LSU",             "R2Z3": "Duke",         "R2Z4": "Minnesota",
    },
}

# ── Build gender data ──────────────────────────────────────────────────────────
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

def resolve_bracket(gender):
    blookup, id_to_name, bname_to_id, champ_prob_map, seed_to_team, team_to_seednum, sl2026 = gender_data[gender]
    actual = ACTUAL_RESULTS[gender]
    slot_winner, matchup_info = {}, {}
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
                "winner":        id_to_name.get(winner, str(winner)),
                "winner_seed":   int(winner_seed),
                "loser":         id_to_name.get(loser, str(loser)),
                "loser_seed":    int(loser_seed),
                "winner_prob":   round(float(winner_prob), 3),
                "is_upset":      int(winner_seed > loser_seed),
                "is_final":      int(is_final),
                "model_correct": int(is_final and winner_prob >= 0.5),
                "champ_prob":    round(float(champ_prob_map.get(winner, 0)), 4),
            }
            progress = True
        if not progress:
            break
    return matchup_info

bracket_data = {g: resolve_bracket(g) for g in ["M", "W"]}

# ── Generate HTML ──────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>March Mania 2026 Bracket</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #f3f4f6; min-height: 100vh; }

  /* ── Header ── */
  #header { background: #1a237e; color: white; padding: 12px 20px;
             display: flex; align-items: center; gap: 20px; position: sticky;
             top: 0; z-index: 100; box-shadow: 0 2px 8px rgba(0,0,0,.3); }
  #header h1 { font-size: 18px; font-weight: 700; white-space: nowrap; }
  .toggle-btn { padding: 5px 14px; border-radius: 20px; border: 2px solid rgba(255,255,255,.5);
                background: transparent; color: white; cursor: pointer; font-size: 13px;
                font-weight: 600; transition: all .15s; }
  .toggle-btn.active { background: white; color: #1a237e; border-color: white; }
  #legend { display: flex; gap: 14px; margin-left: auto; font-size: 11px; flex-wrap: wrap; }
  .leg { display: flex; align-items: center; gap: 4px; }
  .leg-dot { width: 10px; height: 10px; border-radius: 2px; flex-shrink: 0; }

  /* ── Scroll container ── */
  #bracket-scroll { overflow-x: auto; overflow-y: auto; padding: 16px;
                    max-height: calc(100vh - 56px); }

  /* ── Bracket layout ── */
  #bracket-wrap { display: inline-flex; flex-direction: column; gap: 0;
                  position: relative; }
  .bracket-half { display: flex; align-items: stretch; }
  .half-inner { display: flex; align-items: stretch; }
  .region { display: flex; flex-direction: column; }
  .region-name { font-size: 22px; font-weight: 900; text-align: center;
                 padding: 4px 8px; letter-spacing: 1px; color: #1a237e;
                 writing-mode: horizontal-tb; }
  .rounds-row { display: flex; align-items: stretch; gap: 0; }
  .round { display: flex; flex-direction: column; justify-content: space-around;
           width: 168px; padding: 0 3px; }

  /* ── Cards ── */
  .game-card { background: white; border-radius: 8px;
               box-shadow: 0 1px 3px rgba(0,0,0,.1);
               border: 1.5px solid #e5e7eb; margin: 2px 0;
               padding: 5px 7px; cursor: default; min-width: 0; }
  .game-card.correct  { border-color: #22c55e; border-width: 1.5px; }
  .game-card.wrong    { border-color: #ef4444; border-width: 2px; }
  .game-card.upcoming { border-color: #e5e7eb; }
  .game-card.tbd-card { border-color: #e5e7eb; opacity: .5; }

  .team-row { display: flex; align-items: center; gap: 4px;
              padding: 2px 0; font-size: 12px; line-height: 1.3; }
  .team-row.winner-row { font-weight: 700; color: #111; }
  .team-row.loser-row  { color: #9ca3af; }
  .team-row.upset-row  { color: #92400e; background: #fef3c7;
                         border-radius: 3px; padding: 2px 3px; margin: 0 -3px; }
  .seed  { font-size: 10px; color: #6b7280; min-width: 16px;
           text-align: right; flex-shrink: 0; }
  .winner-row .seed { color: #374151; }
  .tname { flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .loser-row .tname s { text-decoration-color: #9ca3af; }
  .prob  { font-size: 10px; margin-left: 4px; flex-shrink: 0; color: #374151; }
  .loser-row .prob { color: #9ca3af; }
  .champ-p { font-size: 9px; color: #2563eb; margin-left: 2px; flex-shrink: 0; }
  .check   { font-size: 10px; color: #22c55e; margin-left: 2px; flex-shrink: 0; }
  .cross   { font-size: 10px; color: #ef4444; margin-left: 2px; flex-shrink: 0; }
  .divider { height: 1px; background: #f3f4f6; margin: 1px 0; }
  .card-footer { font-size: 9px; color: #9ca3af; text-align: right; margin-top: 1px; }

  /* ── Connector column (gap between rounds) ── */
  .connector-col { width: 20px; flex-shrink: 0; }

  /* ── Final Four column ── */
  .ff-col { display: flex; flex-direction: column; justify-content: center;
            align-items: center; width: 188px; padding: 0 6px; gap: 6px; }
  .ff-label { font-size: 10px; font-weight: 800; color: #6b7280;
              letter-spacing: 1px; text-transform: uppercase; text-align: center; }

  /* ── Championship ── */
  #championship { display: flex; justify-content: center; align-items: center;
                  padding: 8px 0; background: #fffbeb;
                  border-top: 1px solid #fde68a; border-bottom: 1px solid #fde68a; }
  .champ-inner { text-align: center; }
  .champ-label { font-size: 10px; font-weight: 800; color: #92400e;
                 letter-spacing: 1px; text-transform: uppercase; margin-bottom: 4px; }
  .champ-box { background: white; border: 2px solid #f59e0b; border-radius: 10px;
               padding: 6px 16px; font-weight: 700; font-size: 14px;
               display: inline-flex; align-items: center; gap: 6px; }
  .champ-sub { font-size: 11px; color: #6b7280; font-weight: 400; }

  /* ── SVG overlay ── */
  #connector-svg { position: absolute; top: 0; left: 0;
                   pointer-events: none; z-index: 0; overflow: visible; }
  .bracket-half, .half-inner, .region, .rounds-row, .round,
  .game-card, .ff-col { position: relative; z-index: 1; }
</style>
</head>
<body>

<div id="header">
  <h1>🏀 March Mania 2026</h1>
  <button class="toggle-btn active" id="btn-M" onclick="setGender('M')">Men's</button>
  <button class="toggle-btn"        id="btn-W" onclick="setGender('W')">Women's</button>
  <div id="legend">
    <div class="leg"><div class="leg-dot" style="border:1.5px solid #22c55e;background:#f0fdf4"></div> Correct pick</div>
    <div class="leg"><div class="leg-dot" style="border:2px solid #ef4444;background:#fef2f2"></div> Wrong pick</div>
    <div class="leg"><div class="leg-dot" style="background:#fef3c7;border:1px solid #f59e0b"></div> Upset</div>
    <div class="leg" style="color:rgba(255,255,255,.8)">% = model probability &nbsp;|&nbsp; Blue = champ odds</div>
  </div>
</div>

<div id="bracket-scroll">
  <div id="bracket-wrap">
    <svg id="connector-svg"></svg>
    <div class="bracket-half" id="top-half"></div>
    <div id="championship"></div>
    <div class="bracket-half" id="bot-half"></div>
  </div>
</div>

<script>
const DATA = BRACKET_DATA_PLACEHOLDER;

const REGION_NAMES = {
  M: { W:'East', X:'South', Y:'Midwest', Z:'West' },
  W: { W:'Regional 1', X:'Regional 4', Y:'Regional 3', Z:'Regional 2' }
};

const SLOT_ORDER = {
  R1:[1,8,5,4,6,3,7,2],
  R2:[1,4,3,2],
  R3:[1,2],
  R4:[1]
};

let currentGender = 'M';

function pct(p){ return Math.round(p*100)+'%'; }

function makeCard(slot, data) {
  const info = data[slot];
  if (!info) return `<div class="game-card tbd-card" id="card-${slot}"><div class="team-row" style="color:#ccc;font-size:11px;justify-content:center">TBD</div></div>`;

  const {winner, winner_seed, loser, loser_seed, winner_prob, is_upset, is_final, model_correct, champ_prob} = info;

  let cardCls = 'game-card ';
  let wExtra = '', lExtra = '', footer = '';

  if (is_final && model_correct) {
    cardCls += 'correct';
    wExtra = `<span class="prob">${pct(winner_prob)}</span><span class="check">✓</span>`;
    lExtra = `<span class="prob">${pct(1-winner_prob)}</span>`;
    footer = `<div class="card-footer">Final</div>`;
  } else if (is_final && !model_correct) {
    cardCls += 'wrong';
    wExtra = `<span class="prob">${pct(winner_prob)}</span><span class="cross">✗</span>`;
    lExtra = `<span class="prob">${pct(1-winner_prob)}</span>`;
    footer = `<div class="card-footer">Final</div>`;
  } else {
    cardCls += 'upcoming';
    wExtra = `<span class="prob">${pct(winner_prob)}</span><span class="champ-p">${pct(champ_prob)}</span>`;
    lExtra = `<span class="prob">${pct(1-winner_prob)}</span>`;
  }

  const wRowCls = 'team-row winner-row' + (is_upset ? ' upset-row' : '');
  const loserName = is_final ? `<s>${loser}</s>` : loser;

  return `<div class="${cardCls}" id="card-${slot}">
    <div class="${wRowCls}">
      <span class="seed">${winner_seed}</span>
      <span class="tname">${winner}</span>
      ${wExtra}
    </div>
    <div class="divider"></div>
    <div class="team-row loser-row">
      <span class="seed">${loser_seed}</span>
      <span class="tname">${loserName}</span>
      ${lExtra}
    </div>
    ${footer}
  </div>`;
}

function makeRound(region, rnd, data) {
  const cards = SLOT_ORDER[rnd].map(i => makeCard(`${rnd}${region}${i}`, data)).join('');
  return `<div class="round" id="round-${rnd}${region}">${cards}</div><div class="connector-col"></div>`;
}

function makeRegion(code, reversed, data, name) {
  const rnds = reversed ? ['R4','R3','R2','R1'] : ['R1','R2','R3','R4'];
  // For reversed, the connector-col goes BEFORE each round (left side)
  let roundsHtml;
  if (reversed) {
    roundsHtml = rnds.map(r => {
      const cards = SLOT_ORDER[r].map(i => makeCard(`${r}${code}${i}`, data)).join('');
      return `<div class="connector-col"></div><div class="round" id="round-${r}${code}">${cards}</div>`;
    }).join('');
  } else {
    roundsHtml = rnds.map(r => {
      const cards = SLOT_ORDER[r].map(i => makeCard(`${r}${code}${i}`, data)).join('');
      return `<div class="round" id="round-${r}${code}">${cards}</div><div class="connector-col"></div>`;
    }).join('');
  }
  return `<div class="region" id="region-${code}">
    <div class="region-name">${name}</div>
    <div class="rounds-row">${roundsHtml}</div>
  </div>`;
}

function renderBracket(gender) {
  const data = DATA[gender];
  const names = REGION_NAMES[gender];

  // Top half: W (left) + FF WX + X (right, reversed)
  document.getElementById('top-half').innerHTML = `
    <div class="half-inner">
      ${makeRegion('W', false, data, names.W)}
      <div class="ff-col" id="ff-WX">
        <div class="ff-label">Final Four</div>
        ${makeCard('R5WX', data)}
      </div>
      ${makeRegion('X', true, data, names.X)}
    </div>`;

  // Championship
  const champ = data['R6CH'];
  let champHtml = '<div class="champ-box">TBD</div>';
  if (champ) {
    const sub = champ.is_final ? '' : `<span class="champ-sub">${pct(champ.champ_prob)}</span>`;
    const mark = champ.is_final ? (champ.model_correct ? ' ✓' : ' ✗') : '';
    champHtml = `<div class="champ-box">🏆 #${champ.winner_seed} ${champ.winner}${mark} ${sub}</div>`;
  }
  document.getElementById('championship').innerHTML = `
    <div class="champ-inner">
      <div class="champ-label">Championship</div>
      ${champHtml}
    </div>`;

  // Bottom half: Y (left) + FF YZ + Z (right, reversed)
  document.getElementById('bot-half').innerHTML = `
    <div class="half-inner">
      ${makeRegion('Y', false, data, names.Y)}
      <div class="ff-col" id="ff-YZ">
        <div class="ff-label">Final Four</div>
        ${makeCard('R5YZ', data)}
      </div>
      ${makeRegion('Z', true, data, names.Z)}
    </div>`;

  setTimeout(() => drawConnectors(gender), 60);
}

function drawConnectors(gender) {
  const wrap = document.getElementById('bracket-wrap');
  const svg  = document.getElementById('connector-svg');
  const wRect = wrap.getBoundingClientRect();

  svg.setAttribute('width',  wrap.scrollWidth);
  svg.setAttribute('height', wrap.scrollHeight);
  svg.innerHTML = '';

  const data = DATA[gender];
  const STROKE = '#d1d5db';
  const SW = 1.5;

  function line(x1,y1,x2,y2) {
    const el = document.createElementNS('http://www.w3.org/2000/svg','line');
    el.setAttribute('x1',x1); el.setAttribute('y1',y1);
    el.setAttribute('x2',x2); el.setAttribute('y2',y2);
    el.setAttribute('stroke', STROKE); el.setAttribute('stroke-width', SW);
    svg.appendChild(el);
  }

  function cardEdges(id) {
    const el = document.getElementById('card-'+id);
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return {
      left:  r.left  - wRect.left,
      right: r.right - wRect.left,
      cy:    (r.top + r.bottom)/2 - wRect.top,
      top:   r.top  - wRect.top,
      bot:   r.bottom - wRect.top,
    };
  }

  const ROUNDS = ['R1','R2','R3','R4'];
  const LEFT_REGIONS  = ['W','Y'];
  const RIGHT_REGIONS = ['X','Z'];

  // Draw within-region connectors
  ['W','X','Y','Z'].forEach(region => {
    const isRight = RIGHT_REGIONS.includes(region);

    ROUNDS.forEach((rnd, ri) => {
      if (ri === ROUNDS.length-1) return;
      const nextRnd = ROUNDS[ri+1];
      const curOrd  = SLOT_ORDER[rnd];
      const nxtOrd  = SLOT_ORDER[nextRnd];

      nxtOrd.forEach((nIdx, ni) => {
        const c1 = cardEdges(`${rnd}${region}${curOrd[2*ni]}`);
        const c2 = cardEdges(`${rnd}${region}${curOrd[2*ni+1]}`);
        const cn = cardEdges(`${nextRnd}${region}${nIdx}`);
        if (!c1||!c2||!cn) return;

        const midY = (c1.cy + c2.cy)/2;

        if (!isRight) {
          const xM = (c1.right + cn.left)/2;
          line(c1.right, c1.cy, xM, c1.cy);
          line(xM, c1.cy, xM, c2.cy);
          line(c2.right, c2.cy, xM, c2.cy);
          line(xM, midY, cn.left, cn.cy);
        } else {
          const xM = (c1.left + cn.right)/2;
          line(c1.left, c1.cy, xM, c1.cy);
          line(xM, c1.cy, xM, c2.cy);
          line(c2.left, c2.cy, xM, c2.cy);
          line(xM, midY, cn.right, cn.cy);
        }
      });
    });

    // R4 → Final Four connectors
    const r4 = cardEdges(`R4${region}1`);
    const ffSlot = (region==='W'||region==='X') ? 'R5WX' : 'R5YZ';
    const ff = cardEdges(ffSlot);
    if (r4 && ff) {
      if (!isRight) {
        line(r4.right, r4.cy, ff.left, ff.cy);
      } else {
        line(r4.left, r4.cy, ff.right, ff.cy);
      }
    }
  });

  // Final Four → Championship
  const ffWX = cardEdges('R5WX');
  const ffYZ = cardEdges('R5YZ');
  const champ = cardEdges('R6CH');
  if (ffWX && champ) {
    const xM = (ffWX.right + champ.left)/2;
    if (Math.abs(xM - ffWX.right) < 5) {
      line(ffWX.right, ffWX.bot, ffWX.right, champ.top);
    } else {
      line(ffWX.right, ffWX.cy, xM, ffWX.cy);
      line(xM, ffWX.cy, xM, champ.cy);
      line(xM, champ.cy, champ.left, champ.cy);
    }
  }
  if (ffYZ && champ) {
    const xM = (ffYZ.right + champ.left)/2;
    if (Math.abs(xM - ffYZ.right) < 5) {
      line(ffYZ.right, ffYZ.top, ffYZ.right, champ.bot);
    } else {
      line(ffYZ.right, ffYZ.cy, xM, ffYZ.cy);
      line(xM, ffYZ.cy, xM, champ.cy);
      line(xM, champ.cy, champ.left, champ.cy);
    }
  }
}

function setGender(g) {
  currentGender = g;
  document.getElementById('btn-M').classList.toggle('active', g==='M');
  document.getElementById('btn-W').classList.toggle('active', g==='W');
  renderBracket(g);
}

renderBracket('M');
window.addEventListener('resize', () => drawConnectors(currentGender));
</script>
</body>
</html>
"""

html = HTML.replace("BRACKET_DATA_PLACEHOLDER", json.dumps(bracket_data))

with open("bracket.html", "w") as f:
    f.write(html)

print("✓ bracket.html generated")
webbrowser.open("bracket.html")
