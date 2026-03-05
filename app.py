# -*- coding: utf-8 -*-
"""
FPL Transfer Optimizer v5.2 — Rolling 6-GW Transfer Planner (Streamlit)
========================================================================
Corrected scoring per official 2025/26 FPL rules:
  - CS = 4pts for DEF/GKP (was incorrectly 6 in v5.1)
  - CS = 1pt for MID
  - DC = 2pts (probability × 2, not doubled again)
  - Captain projection uses cs × 4.0 for DEF/GKP (fixed from cs × 6.0)

Data sources (all free, no auth):
  - FPL API    → squad, prices, form, ep_next, GW history, DC stats
  - Understat  → xG, xA, npxG, shots, key passes per player
  - FBref      → xG, xA, npxG, shots, key passes (StatsBomb/Opta model)
                  Used as cross-validation against Understat's independent model.

Deploy: streamlit run app.py
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import unicodedata
import re
import json
import time
import warnings
from io import StringIO
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────
MAX_TRANSFER_BANK = 5
BASE = "https://fantasy.premierleague.com/api"
DC_THREADS = 25

TEAM_MAP = {
    "Manchester City":         "Man City",
    "Manchester United":       "Man Utd",
    "Tottenham":               "Spurs",
    "Nottingham Forest":       "Nott'm Forest",
    "Newcastle United":        "Newcastle",
    "Wolverhampton Wanderers": "Wolves",
    "Leicester":               "Leicester",
    "Ipswich":                 "Ipswich",
}


# ══════════════════════════════════════════════════════════════════════
#  1.  FPL API HELPERS
# ══════════════════════════════════════════════════════════════════════

def fpl_get(endpoint):
    r = requests.get(f"{BASE}/{endpoint}", timeout=20)
    r.raise_for_status()
    return r.json()

def get_bootstrap():
    return fpl_get("bootstrap-static/")

def get_my_team(manager_id, boot):
    now    = datetime.now(timezone.utc)
    events = boot["events"]

    next_gw = None
    for event in events:
        deadline = datetime.fromisoformat(
            event["deadline_time"].replace("Z", "+00:00")
        )
        if deadline > now:
            next_gw = event["id"]
            break

    if next_gw is None:
        active_gw = events[-1]["id"]
        next_gw   = active_gw
    elif next_gw == 1:
        active_gw = 1
    else:
        active_gw = next_gw - 1

    picks = fpl_get(f"entry/{manager_id}/event/{active_gw}/picks/")
    return picks, active_gw, next_gw


def get_free_transfers(manager_id, active_gw):
    try:
        history    = fpl_get(f"entry/{manager_id}/history/")
        gw_history = history.get("current", [])
        transfers_made = {h["event"]: h["event_transfers"] for h in gw_history}

        bank = 1
        for gw in range(1, active_gw + 1):
            made = transfers_made.get(gw, 0)
            bank = min(bank + 1, MAX_TRANSFER_BANK)
            bank = max(bank - made, 0)

        free_transfers = max(1, min(bank, MAX_TRANSFER_BANK))

        chips      = history.get("chips", [])
        chips_used = [c["name"] for c in chips]
        chips_available = []
        if "wildcard" not in chips_used:
            chips_available.append("Wildcard")
        if "freehit" not in chips_used:
            chips_available.append("Free Hit")
        if "bboost" not in chips_used:
            chips_available.append("Bench Boost")
        if "3xc" not in chips_used:
            chips_available.append("Triple Captain")

        return free_transfers, chips_available
    except Exception:
        return 1, []


def compute_sell_prices(squad_ids, players_df, manager_id):
    """
    Calculate actual FPL sell prices for each player in the squad.

    FPL sell price formula:
        profit = now_cost - purchase_price
        sell_price = purchase_price + floor(profit / 2)

    Purchase price is determined by:
      - Transfer history: the element_in_cost of the most recent transfer IN
      - GW1 picks: now_cost - cost_change_start (i.e. start-of-season price)

    Returns dict: {fpl_id: sell_price_in_millions}
    """
    # Get transfer history (public endpoint, no auth needed)
    try:
        transfers = fpl_get(f"entry/{manager_id}/transfers/")
    except Exception:
        transfers = []

    # Build map: player_id -> most recent purchase price (in tenths, raw API)
    # Transfers are returned newest-first by the API
    purchase_map = {}  # fpl_id -> element_in_cost (tenths of £)
    for t in transfers:
        pid = int(t.get("element_in", 0))
        if pid in squad_ids and pid not in purchase_map:
            purchase_map[pid] = int(t.get("element_in_cost", 0))

    # For players not in transfer history (owned since GW1), calculate from
    # cost_change_start: start_price = now_cost - cost_change_start
    sell_prices = {}
    for pid in squad_ids:
        row = players_df[players_df["fpl_id"] == pid]
        if row.empty:
            continue
        row = row.iloc[0]
        now_cost_raw = int(row.get("now_cost", 0))

        if pid in purchase_map:
            purchase_price = purchase_map[pid]
        else:
            # Owned since start of season
            ccs = int(row.get("cost_change_start", 0))
            purchase_price = now_cost_raw - ccs

        # FPL formula: you keep half profit, rounded down
        profit = now_cost_raw - purchase_price
        if profit > 0:
            sell_raw = purchase_price + (profit // 2)
        else:
            # Price dropped: you lose the full drop
            sell_raw = now_cost_raw

        sell_prices[pid] = sell_raw / 10  # convert to £m

    return sell_prices


def get_fixtures():
    return fpl_get("fixtures/")

def build_fpl_table(boot):
    players = pd.DataFrame(boot["elements"]).copy()
    teams   = pd.DataFrame(boot["teams"])[["id","name","short_name"]].copy()
    teams   = teams.rename(columns={
        "id": "team_id", "name": "team_name", "short_name": "team_short"
    })

    players = players.rename(columns={"team": "team_ref"})
    players = players.merge(teams, left_on="team_ref", right_on="team_id", how="left")

    pos_map = {1:"GKP", 2:"DEF", 3:"MID", 4:"FWD"}
    players["position"] = players["element_type"].map(pos_map)
    players["price"]    = players["now_cost"] / 10
    players["full_name"]= players["first_name"] + " " + players["second_name"]
    players["fpl_id"]   = players["id"]

    num_cols = [
        "total_points","form","minutes","goals_scored","assists",
        "clean_sheets","bonus","ep_next","ep_this","bps",
        "selected_by_percent","saves","penalties_saved",
        "yellow_cards","red_cards","goals_conceded",
        "expected_goals","expected_assists",
        "expected_goal_involvements","expected_goals_conceded",
        "threat","creativity","influence","ict_index",
        "transfers_in_event","transfers_out_event",
        "chance_of_playing_next_round","chance_of_playing_this_round",
    ]
    for c in num_cols:
        if c in players.columns:
            players[c] = pd.to_numeric(players[c], errors="coerce").fillna(0)

    keep = ["fpl_id","full_name","web_name","team_name","team_short","team_id",
            "position","price","now_cost","cost_change_start","status"] + \
           [c for c in num_cols if c in players.columns]

    players = players.loc[:, ~players.columns.duplicated()]
    keep    = [c for c in keep if c in players.columns]
    return players[keep].reset_index(drop=True).copy()


# ══════════════════════════════════════════════════════════════════════
#  2.  DEFENSIVE CONTRIBUTION STATS (2025/26 rules)
# ══════════════════════════════════════════════════════════════════════
#  DEF:     2 pts if CBIT >= 10 in a match
#  MID/FWD: 2 pts if CBIRT >= 12 in a match (CBIT + ball recoveries)
#  GKP:     DC does not apply.
#  Cap:     Hard cap of 2 pts per match.
# ══════════════════════════════════════════════════════════════════════

def _fetch_one_dc(pid, gw_start, position_type):
    if position_type == 1:
        return {"fpl_id": pid}
    try:
        data    = fpl_get(f"element-summary/{pid}/")
        history = data.get("history", [])
        recent  = [
            h for h in history
            if h.get("round", 0) >= gw_start and h.get("minutes", 0) > 0
        ]
        if not recent:
            return {"fpl_id": pid}

        dc_pts_earned = 0.0
        total_mins    = 0.0
        cbit_total    = 0.0
        gw_count      = 0

        for gw in recent:
            total_mins    += float(gw.get("minutes", 0) or 0)
            tackles        = float(gw.get("tackles",       gw.get("attempted_tackles", 0)) or 0)
            interceptions  = float(gw.get("interceptions", 0) or 0)
            blocks         = float(gw.get("blocked_shots", gw.get("blocked", 0)) or 0)
            clearances     = float(gw.get("clearances",    gw.get("clearances_off_line", 0)) or 0)
            recoveries     = float(gw.get("recoveries",    0) or 0)

            cbit = tackles + interceptions + blocks + clearances

            if position_type == 2:        # DEF: threshold 10 CBIT
                threshold     = 10
                total_actions = cbit
            else:                          # MID/FWD: threshold 12 CBIRT
                threshold     = 12
                total_actions = cbit + recoveries

            dc_pts_earned += 2.0 if total_actions >= threshold else 0.0
            cbit_total    += cbit
            gw_count      += 1

        nineties = total_mins / 90 if total_mins > 0 else None
        return {
            "fpl_id":      pid,
            "dc_pts_p90":  dc_pts_earned / nineties if nineties else 0.0,
            "dc_hit_rate": (dc_pts_earned / 2.0) / gw_count if gw_count else 0.0,
            "cbit_p90":    cbit_total / nineties if nineties else 0.0,
        }
    except Exception:
        return {"fpl_id": pid}


def fetch_dc_stats(boot, current_gw, lookback=8, progress_bar=None):
    elements = pd.DataFrame(boot["elements"])
    targets  = elements[
        elements["element_type"].isin([2, 3, 4])
    ][["id", "element_type"]].values.tolist()

    gw_start  = max(1, current_gw - lookback + 1)
    total     = len(targets)

    rows, completed = [], 0
    with ThreadPoolExecutor(max_workers=DC_THREADS) as executor:
        futures = {
            executor.submit(_fetch_one_dc, int(pid), gw_start, int(pos)): pid
            for pid, pos in targets
        }
        for future in as_completed(futures):
            rows.append(future.result())
            completed += 1
            if progress_bar and (completed % 25 == 0 or completed == total):
                progress_bar.progress(completed / total,
                                      text=f"Fetching DC stats: {completed}/{total}")

    dc_df = pd.DataFrame(rows).fillna(0)
    for col in ["dc_pts_p90", "dc_hit_rate", "cbit_p90"]:
        if col not in dc_df.columns:
            dc_df[col] = 0.0
    return dc_df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════
#  3.  FIXTURE ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def compute_fixture_difficulty(boot, fixtures, from_gw, horizon=3):
    future = [
        f for f in fixtures
        if f["event"]
        and from_gw < f["event"] <= from_gw + horizon
        and not f["finished_provisional"]
    ]
    team_diffs  = {t["id"]: [] for t in boot["teams"]}
    team_ngames = {t["id"]: 0  for t in boot["teams"]}

    for f in future:
        h, a = f["team_h"], f["team_a"]
        team_diffs[h].append(f["team_h_difficulty"])
        team_diffs[a].append(f["team_a_difficulty"])
        team_ngames[h] += 1
        team_ngames[a] += 1

    teams = pd.DataFrame(boot["teams"])[["id","name","short_name"]]
    rows  = []
    for tid in team_diffs:
        diffs = team_diffs[tid]
        rows.append({
            "team_id":        tid,
            "avg_difficulty": float(np.mean(diffs)) if diffs else 4.5,
            "num_fixtures":   team_ngames[tid],
            "has_dgw":        int(team_ngames[tid] > horizon),
        })
    return pd.DataFrame(rows).merge(teams, left_on="team_id", right_on="id", how="left")


# ══════════════════════════════════════════════════════════════════════
#  4.  CS PROBABILITY
# ══════════════════════════════════════════════════════════════════════

def get_team_defensive_stats(boot):
    players = pd.DataFrame(boot["elements"]).copy()
    teams   = pd.DataFrame(boot["teams"])[["id","name"]].copy()

    for c in ["expected_goals_conceded","minutes","clean_sheets"]:
        players[c] = pd.to_numeric(players.get(c, 0), errors="coerce").fillna(0)

    gks = players[players["element_type"] == 1].copy()
    gk_agg = gks.groupby("team").agg(
        team_xgc     = ("expected_goals_conceded", "sum"),
        team_cs      = ("clean_sheets", "sum"),
        team_minutes = ("minutes", "sum"),
    ).reset_index().rename(columns={"team": "team_id"})

    gk_agg["games_played"] = (gk_agg["team_minutes"] / 90).replace(0, np.nan)
    gk_agg["xgc_p90"]      = gk_agg["team_xgc"] / gk_agg["games_played"]
    gk_agg["cs_rate"]      = gk_agg["team_cs"]  / gk_agg["games_played"]
    gk_agg["cs_prob"]      = np.exp(-gk_agg["xgc_p90"].fillna(1.2))

    gk_agg = gk_agg.merge(teams, left_on="team_id", right_on="id", how="left")
    return gk_agg[["team_id","name","xgc_p90","cs_rate","cs_prob"]].reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════
#  5.  UNDERSTAT
# ══════════════════════════════════════════════════════════════════════

def get_understat_stats():
    url = "https://understat.com/league/EPL/2024"
    headers = {"User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )}
    try:
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
    except Exception:
        return pd.DataFrame()

    match = re.search(r"var\s+playersData\s*=\s*JSON\.parse\('(.+?)'\)", r.text)
    if not match:
        return pd.DataFrame()

    try:
        raw  = match.group(1).encode("utf-8").decode("unicode_escape")
        data = json.loads(raw)
    except Exception:
        return pd.DataFrame()

    rows = []
    for p in data:
        mins = float(p.get("time", 0) or 0)
        n90  = mins / 90 if mins >= 90 else np.nan
        rows.append({
            "player_name": p.get("player_name", ""),
            "team_title":  p.get("team_title",  ""),
            "minutes_us":  mins,
            "xg_p90":    float(p.get("xG",         0) or 0) / n90 if n90 else 0,
            "xa_p90":    float(p.get("xA",         0) or 0) / n90 if n90 else 0,
            "npxg_p90":  float(p.get("npxG",       0) or 0) / n90 if n90 else 0,
            "shots_p90": float(p.get("shots",      0) or 0) / n90 if n90 else 0,
            "kp_p90":    float(p.get("key_passes", 0) or 0) / n90 if n90 else 0,
        })

    df = pd.DataFrame(rows)
    df = df[df["minutes_us"] >= 90].reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════════════
#  5b. FBREF (StatsBomb/Opta xG — independent second source)
# ══════════════════════════════════════════════════════════════════════

FBREF_TEAM_MAP = {
    "Manchester Utd":  "Man Utd",
    "Manchester City": "Man City",
    "Tottenham":       "Spurs",
    "Nott'ham Forest": "Nott'm Forest",
    "Newcastle Utd":   "Newcastle",
    "Wolverhampton":   "Wolves",
    "West Ham":        "West Ham",
}

def get_fbref_stats():
    """
    Scrape FBref's Premier League player standard stats table.
    Uses StatsBomb/Opta xG model — independent from Understat.
    Returns per-90 stats with fbref_ prefix for cross-validation.

    FBref rate-limits to 1 request per 3 seconds. We scrape two pages
    (standard stats + shooting) with appropriate delays.
    """
    headers = {"User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )}

    # ── Page 1: Standard stats (xG, xA, npxG, goals, assists, minutes) ──
    url_std = "https://fbref.com/en/comps/9/stats/Premier-League-Stats"
    try:
        r = requests.get(url_std, headers=headers, timeout=30)
        r.raise_for_status()
    except Exception:
        return pd.DataFrame()

    # FBref hides some tables inside HTML comments — strip them
    html = r.text.replace('<!--', '').replace('-->', '')

    try:
        tables = pd.read_html(StringIO(html))
    except Exception:
        return pd.DataFrame()

    # Find the main player stats table — it has 'Player' and 'xG' columns
    std_df = None
    for tbl in tables:
        # Handle multi-level columns
        if isinstance(tbl.columns, pd.MultiIndex):
            tbl.columns = [
                c[-1] if isinstance(c, tuple) else c
                for c in tbl.columns
            ]
        cols_lower = [str(c).lower() for c in tbl.columns]
        if 'player' in cols_lower and 'xg' in cols_lower and len(tbl) > 50:
            std_df = tbl.copy()
            break

    if std_df is None:
        return pd.DataFrame()

    # Flatten multi-index columns if needed
    if isinstance(std_df.columns, pd.MultiIndex):
        std_df.columns = [c[-1] for c in std_df.columns]

    # Remove header rows that appear inline (FBref quirk)
    std_df = std_df[std_df["Player"] != "Player"].copy()

    # Normalise column names
    col_map = {}
    for c in std_df.columns:
        cl = str(c).strip()
        col_map[c] = cl
    std_df = std_df.rename(columns=col_map)

    # Extract the columns we need
    needed = {
        "Player": "player_name",
        "Squad": "team_name",
        "Min": "minutes_fb",
        "90s": "nineties",
        "Gls": "goals",
        "Ast": "assists",
        "xG": "xg_total",
        "xAG": "xa_total",
        "npxG": "npxg_total",
        "Sh": "shots_total",
    }

    # Some columns might use slightly different names across seasons
    alt_names = {
        "xAG": ["xAG", "xA"],
        "Min": ["Min", "Playing Time Min"],
        "90s": ["90s", "Playing Time 90s"],
        "Sh": ["Sh", "Shooting Sh"],
    }

    available = {}
    for target, aliases in alt_names.items():
        for alias in aliases:
            if alias in std_df.columns:
                available[target] = alias
                break

    # Build rename dict with available columns
    rename = {}
    for orig, dest in needed.items():
        col = available.get(orig, orig)
        if col in std_df.columns:
            rename[col] = dest

    std_df = std_df.rename(columns=rename)

    # Keep only rows with data
    keep = [v for v in rename.values() if v in std_df.columns]
    if "player_name" not in keep:
        return pd.DataFrame()
    std_df = std_df[keep].copy()

    # Convert numeric columns
    for c in std_df.columns:
        if c not in ("player_name", "team_name"):
            std_df[c] = pd.to_numeric(std_df[c], errors="coerce").fillna(0)

    # Filter minimum minutes
    if "minutes_fb" in std_df.columns:
        std_df = std_df[std_df["minutes_fb"] >= 90].copy()
    elif "nineties" in std_df.columns:
        std_df = std_df[std_df["nineties"] >= 1.0].copy()

    # Compute per-90 stats
    if "nineties" in std_df.columns:
        n90 = std_df["nineties"].replace(0, np.nan)
    elif "minutes_fb" in std_df.columns:
        n90 = (std_df["minutes_fb"] / 90).replace(0, np.nan)
    else:
        return pd.DataFrame()

    std_df["fbref_xg_p90"]    = std_df.get("xg_total", 0) / n90
    std_df["fbref_xa_p90"]    = std_df.get("xa_total", 0) / n90
    std_df["fbref_npxg_p90"]  = std_df.get("npxg_total", 0) / n90
    std_df["fbref_shots_p90"] = std_df.get("shots_total", 0) / n90

    # Clean team names
    if "team_name" in std_df.columns:
        std_df["team_name"] = std_df["team_name"].apply(
            lambda t: FBREF_TEAM_MAP.get(str(t).strip(), str(t).strip())
        )

    result = std_df[["player_name", "team_name",
                      "fbref_xg_p90", "fbref_xa_p90",
                      "fbref_npxg_p90", "fbref_shots_p90"]].copy()
    result = result.dropna(subset=["player_name"]).reset_index(drop=True)

    return result


def match_fbref(fpl_df, fb_df, threshold=0.45):
    """Match FBref players to FPL players using fuzzy name + team matching."""
    fbref_cols = ["fbref_xg_p90", "fbref_xa_p90", "fbref_npxg_p90", "fbref_shots_p90"]
    for c in fbref_cols:
        fpl_df[c] = 0.0

    if fb_df.empty:
        return fpl_df.reset_index(drop=True)

    fb_names_norm = [_norm(n) for n in fb_df["player_name"].tolist()]
    fb_teams_norm = [_norm(FBREF_TEAM_MAP.get(t, t)) for t in fb_df["team_name"].tolist()]

    matched = 0
    for idx in fpl_df.index:
        fpl_tokens  = set(_norm(fpl_df.at[idx, "full_name"]).split())
        team_tokens = set(_norm(fpl_df.at[idx, "team_name"]).split())

        best_score, best_i = 0.0, -1
        for i, (fn, ft) in enumerate(zip(fb_names_norm, fb_teams_norm)):
            s = _token_match(fpl_tokens, fn, ft, team_tokens)
            if s > best_score:
                best_score, best_i = s, i

        if best_score >= threshold and best_i >= 0:
            row = fb_df.iloc[best_i]
            for c in fbref_cols:
                fpl_df.at[idx, c] = float(row.get(c, 0) or 0)
            matched += 1

    return fpl_df.reset_index(drop=True)


def merge_dual_source_xg(df):
    """
    Blend Understat and FBref xG/xA into a single consensus estimate.

    Strategy:
      - If both sources have data: average them (reduces model-specific noise)
      - If only one has data: use that source alone
      - Stores the blended values AND keeps originals for comparison

    Also computes discrepancy metrics for the UI.
    """
    for metric in ["xg_p90", "xa_p90", "npxg_p90", "shots_p90"]:
        us_col = metric                    # Understat column
        fb_col = f"fbref_{metric}"         # FBref column

        us = df[us_col].fillna(0) if us_col in df.columns else 0
        fb = df[fb_col].fillna(0) if fb_col in df.columns else 0

        # Store originals for comparison display
        df[f"us_{metric}"]    = us
        df[f"fb_{metric}"]    = fb

        # Blend: average when both have data, otherwise use whichever has it
        has_us = us > 0
        has_fb = fb > 0
        both   = has_us & has_fb

        blended = pd.Series(0.0, index=df.index)
        blended[both]          = (us[both] + fb[both]) / 2
        blended[has_us & ~both] = us[has_us & ~both]
        blended[has_fb & ~both] = fb[has_fb & ~both]

        df[metric] = blended  # overwrite with blended value

    # Discrepancy: absolute difference between sources (for flagging)
    df["xg_discrepancy"] = abs(df.get("us_xg_p90", 0) - df.get("fb_xg_p90", 0))
    df["xa_discrepancy"] = abs(df.get("us_xa_p90", 0) - df.get("fb_xa_p90", 0))

    return df


# ══════════════════════════════════════════════════════════════════════
#  6.  NAME MATCHING
# ══════════════════════════════════════════════════════════════════════

def _norm(name: str) -> str:
    name = unicodedata.normalize("NFD", str(name))
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z ]", "", name.lower().strip())

def _token_match(a_tokens, b_norm, b_team_norm, ref_team_tokens, squad_bonus=0.08):
    b_tokens = set(b_norm.split())
    if not b_tokens:
        return 0.0
    overlap = len(a_tokens & b_tokens) / max(len(a_tokens | b_tokens), 1)
    bonus   = squad_bonus if (ref_team_tokens & set(b_team_norm.split())) else 0
    return overlap + bonus

def match_understat(fpl_df, us_df, threshold=0.45):
    ext_cols = ["xg_p90","xa_p90","npxg_p90","shots_p90","kp_p90"]
    for c in ext_cols:
        fpl_df[c] = 0.0

    if us_df.empty:
        return fpl_df.reset_index(drop=True)

    us_names_norm = [_norm(n) for n in us_df["player_name"].tolist()]
    us_teams_norm = [_norm(TEAM_MAP.get(t, t)) for t in us_df["team_title"].tolist()]

    for idx in fpl_df.index:
        fpl_tokens  = set(_norm(fpl_df.at[idx, "full_name"]).split())
        team_tokens = set(_norm(fpl_df.at[idx, "team_name"]).split())

        best_score, best_i = 0.0, -1
        for i, (un, ut) in enumerate(zip(us_names_norm, us_teams_norm)):
            s = _token_match(fpl_tokens, un, ut, team_tokens)
            if s > best_score:
                best_score, best_i = s, i

        if best_score >= threshold and best_i >= 0:
            row = us_df.iloc[best_i]
            for c in ext_cols:
                fpl_df.at[idx, c] = float(row.get(c, 0) or 0)

    return fpl_df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════
#  7.  RISK FLAGS
# ══════════════════════════════════════════════════════════════════════

def add_risk_flags(df):
    df    = df.reset_index(drop=True)
    mins  = df["minutes"].values.astype(float)
    pts   = df["total_points"].values.astype(float)
    games = np.where(pts > 0, np.maximum(pts / 3.0, 1.0), 1.0)
    df["minutes_per_game"] = mins / games

    risk = []
    for s, c, m in zip(df["status"].values,
                        df["chance_of_playing_next_round"].values.astype(float),
                        mins):
        if   s == "i":  risk.append("Injured")
        elif s == "s":  risk.append("Suspended")
        elif c < 50:    risk.append("Doubt")
        elif m < 500:   risk.append("Low mins")
        else:           risk.append("Ok")
    df["rotation_risk"] = risk
    return df


# ══════════════════════════════════════════════════════════════════════
#  8.  EXPECTED FPL POINTS MODEL (v5.3 — replaces z-score weights)
# ══════════════════════════════════════════════════════════════════════
#
#  Converts per-90 stats into actual expected FPL points per match,
#  using the official 2025/26 scoring table. This allows meaningful
#  cross-position comparison (e.g. Salah vs Tarkowski).
#
#  Expected points per match =
#    appearance_pts (2 if >60min expected)
#  + position_goal_pts × xG/90
#  + assist_pts × xA/90
#  + cs_prob × cs_pts_for_position
#  + dc_hit_rate × 2  (DC points)
#  + saves_p90 / 3    (GKP only, 1pt per 3 saves)
#  + bonus_p90         (estimated from historical bonus)
#  - yellow_rate × 1   (yellow card deduction)
#  - goals_conceded_penalty (GKP/DEF: -0.5 per goal conceded expected)
#
#  Then adjusted for fixture difficulty.
# ══════════════════════════════════════════════════════════════════════

def _safe(df, col):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(0)
    return pd.Series(np.zeros(len(df)), index=df.index)


def compute_expected_pts(df):
    """
    Compute expected FPL points per match for every player.
    Uses official 2025/26 scoring and per-90 underlying stats.
    Returns the dataframe with 'xPts' and 'fpl_value_score' columns.
    """
    df = df.copy().reset_index(drop=True)

    mins     = _safe(df, "minutes").replace(0, np.nan)
    total_gw = mins / 90  # approximate games played

    # Per-90 stats
    xg       = _safe(df, "xg_p90")
    xa       = _safe(df, "xa_p90")
    cs_prob  = _safe(df, "cs_prob")
    dc_rate  = _safe(df, "dc_hit_rate")
    fix_s    = _safe(df, "fix_score")  # fixture difficulty score
    saves    = _safe(df, "saves") / mins * 90  # saves per 90
    saves    = saves.fillna(0)
    bonus_p90= _safe(df, "bonus") / mins * 90
    bonus_p90= bonus_p90.fillna(0)
    yc_rate  = _safe(df, "yellow_cards") / mins * 90
    yc_rate  = yc_rate.fillna(0)
    ep_next  = _safe(df, "ep_next")  # FPL's own short-term estimate
    xgc_rate = _safe(df, "goals_conceded") / mins * 90
    xgc_rate = xgc_rate.fillna(1.0)

    # Goal and CS points by position
    goal_pts = df["position"].map({"GKP": 6, "DEF": 6, "MID": 5, "FWD": 4}).fillna(4)
    cs_pts   = df["position"].map({"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0}).fillna(0)
    gc_pen   = df["position"].map({"GKP": -0.5, "DEF": -0.5, "MID": 0, "FWD": 0}).fillna(0)

    # ── Expected FPL points per match ──
    xPts = (
        2.0                         # appearance points (assume >60 mins)
        + goal_pts * xg             # expected goal points
        + 3.0 * xa                  # expected assist points (3pts all positions)
        + cs_pts * cs_prob          # expected clean sheet points
        + 2.0 * dc_rate             # expected DC points
        + saves / 3.0               # save points (GKP: 1pt per 3 saves)
        + bonus_p90                 # expected bonus points
        - 1.0 * yc_rate             # yellow card deduction
        + gc_pen * xgc_rate         # goals conceded penalty (GKP/DEF)
    )

    # Fixture difficulty adjustment: scale by fixture_score
    # fix_score is higher for easier fixtures; normalise around 1.0
    fix_mean = fix_s.mean()
    if fix_mean > 0:
        fix_mult = 0.7 + 0.3 * (fix_s / fix_mean)  # range ~0.7 to ~1.3
    else:
        fix_mult = 1.0

    xPts_adjusted = xPts * fix_mult

    # Blend with FPL's own ep_next (short-term prediction) — gives
    # weight to injury/rotation info that our model can't see
    # 70% our model, 30% FPL ep_next
    ep_has_data = ep_next > 0
    blended = xPts_adjusted.copy()
    blended[ep_has_data] = (
        0.70 * xPts_adjusted[ep_has_data] +
        0.30 * ep_next[ep_has_data]
    )

    df["xPts"]           = xPts.round(2)
    df["xPts_adjusted"]  = blended.round(2)
    df["fpl_value_score"] = blended  # this is what the transfer engine ranks by

    return df


def apply_position_scores(df):
    """Apply the expected points model to all players."""
    return compute_expected_pts(df)

def apply_risk_appetite(df, appetite):
    own = pd.to_numeric(df["selected_by_percent"], errors="coerce").fillna(0)
    df["ownership"] = own
    # Normalise ownership to 0-1 range for adjustment
    own_max = own.max()
    own_norm = (own / own_max) if own_max > 0 else own
    if   appetite == "safe":         df["fpl_value_score"] += 0.5 * own_norm
    elif appetite == "differential": df["fpl_value_score"] -= 0.5 * own_norm
    return df


# ══════════════════════════════════════════════════════════════════════
#  9.  SINGLE-GW PLAYER RESCORING
# ══════════════════════════════════════════════════════════════════════

def rescore_for_gw(players_base, boot, fixtures, target_gw, cs_map, risk_appetite):
    df = players_base.copy()

    fix_df  = compute_fixture_difficulty(boot, fixtures, target_gw - 1, horizon=3)
    fix_map = fix_df.set_index("team_id")[
        ["avg_difficulty","num_fixtures","has_dgw"]
    ].to_dict("index")

    df["avg_fix_diff"] = df["team_id"].map(
        lambda t: fix_map.get(t, {}).get("avg_difficulty", 3.0)
    )
    df["num_fixtures"] = df["team_id"].map(
        lambda t: fix_map.get(t, {}).get("num_fixtures", 3)
    )
    df["has_dgw"]   = df["team_id"].map(
        lambda t: fix_map.get(t, {}).get("has_dgw", 0)
    )
    df["fix_score"] = (6 - df["avg_fix_diff"]) * (df["num_fixtures"] / 3)
    df["cs_prob"]   = df["team_id"].map(cs_map).fillna(0.25)

    df = apply_position_scores(df)
    df = apply_risk_appetite(df, risk_appetite)
    return df


# ══════════════════════════════════════════════════════════════════════
#  10.  GW-SPECIFIC CAPTAIN PROJECTION (v5.2 — CS×4 corrected)
# ══════════════════════════════════════════════════════════════════════

def compute_gw_projected_pts(squad_df, target_gw, fixtures):
    """
    Project points per player for a specific GW.

    v5.2 corrected scoring:
      GKP: cs_prob × 4 + saves_p90/3 + 2   (CS=4pts, NOT 6)
      DEF: cs_prob × 4 + xg×6 + xa×3 + dc×2 + 2   (CS=4pts, NOT 6)
      MID: cs_prob × 1 + xg×5 + xa×3 + dc×2 + 2
      FWD: xg×4 + xa×3 + 2

    Fixture difficulty multiplier: diff 3 = 1.0×, ±0.10 per step.
    """
    df = squad_df.copy().reset_index(drop=True)

    gw_fixtures = [
        f for f in fixtures
        if f["event"] == target_gw and not f["finished_provisional"]
    ]
    team_gw_diff  = {}
    teams_playing = set()
    for f in gw_fixtures:
        team_gw_diff[f["team_h"]] = f["team_h_difficulty"]
        team_gw_diff[f["team_a"]] = f["team_a_difficulty"]
        teams_playing.add(f["team_h"])
        teams_playing.add(f["team_a"])

    df["gw_diff"]        = df["team_id"].map(team_gw_diff).fillna(3.0)
    df["fix_multiplier"] = 1.0 + (3.0 - df["gw_diff"]) * 0.10
    df["has_fixture"]    = df["team_id"].isin(teams_playing)

    max_mins = df["minutes"].max()
    if max_mins > 0:
        df["mins_factor"] = (df["minutes"] / max_mins).clip(0.5, 1.0)
    else:
        df["mins_factor"] = 1.0

    proj_pts = []
    for _, p in df.iterrows():
        if not p["has_fixture"]:
            proj_pts.append(0.0)
            continue

        pos = p.get("position", "")
        fm  = float(p["fix_multiplier"])
        mf  = float(p["mins_factor"])
        cs  = float(p.get("cs_prob",     0.25))
        xg  = float(p.get("xg_p90",     0.0))
        xa  = float(p.get("xa_p90",     0.0))
        dc  = float(p.get("dc_hit_rate", 0.0))
        sv  = float(p.get("saves",       0.0))
        sm  = max(float(p.get("minutes", 90.0)), 1.0)

        saves_p90 = sv / sm * 90.0

        # ──────────────────────────────────────────────────────
        # v5.2 CORRECTED: CS = 4pts for GKP/DEF (was 6 in v5.1)
        # ──────────────────────────────────────────────────────
        if pos == "GKP":
            base = (cs * 4.0) + (saves_p90 / 3.0) + 2.0
        elif pos == "DEF":
            base = (cs * 4.0) + (xg * 6.0) + (xa * 3.0) + (dc * 2.0) + 2.0
        elif pos == "MID":
            base = (cs * 1.0) + (xg * 5.0) + (xa * 3.0) + (dc * 2.0) + 2.0
        else:   # FWD
            base = (xg * 4.0) + (xa * 3.0) + 2.0

        proj_pts.append(base * fm * mf)

    df["gw_proj_pts"] = proj_pts
    return df


# ══════════════════════════════════════════════════════════════════════
#  11.  TRANSFER ENGINE
# ══════════════════════════════════════════════════════════════════════

def suggest_transfers_for_gw(squad_ids, scored_players, my_team_df, itb,
                              num_transfers=2, budget_padding=0.0):
    available = scored_players[
        (~scored_players["fpl_id"].isin(squad_ids)) &
        (scored_players["status"].isin(["a","d"]))
    ].copy()

    team_counts = my_team_df["team_id"].value_counts().to_dict()

    results = []
    for _, out_p in my_team_df.iterrows():
        # Use actual sell price (from picks endpoint), not buy price (now_cost).
        # FPL sell price = purchase_price + floor((now_cost - purchase_price) / 2)
        # The picks endpoint provides this as 'selling_price'.
        sell     = float(out_p.get("sell_price", out_p["price"]))
        buy_price = float(out_p["price"])  # current market price (for display)
        bdgt     = sell + itb + budget_padding
        pos      = out_p["position"]
        s_out    = float(out_p.get("fpl_value_score", 0))
        out_team = int(out_p["team_id"])

        counts_after_out = dict(team_counts)
        counts_after_out[out_team] = counts_after_out.get(out_team, 0) - 1

        def within_team_limit(cand_team_id, _counts=counts_after_out):
            return _counts.get(int(cand_team_id), 0) < 3

        cands = available[
            (available["position"] == pos) &
            (available["price"] <= bdgt) &
            (available["team_id"].apply(within_team_limit))
        ].sort_values("fpl_value_score", ascending=False).head(5)

        for _, cand in cands.iterrows():
            gain = float(cand["fpl_value_score"]) - s_out
            if gain > 0.05:
                results.append({
                    "OUT":             out_p["web_name"],
                    "OUT_pos":         pos,
                    "OUT_sell_price":  sell,       # what you actually get back
                    "OUT_buy_price":   buy_price,  # current market price
                    "OUT_score":       round(s_out, 3),
                    "OUT_ep_next":     float(out_p.get("ep_next", 0)),
                    "OUT_dc_hit_rate": round(float(out_p.get("dc_hit_rate", 0)), 2),
                    "OUT_fix_diff":    round(float(out_p.get("avg_fix_diff", 0)), 2),
                    "OUT_risk":        out_p.get("rotation_risk", "?"),
                    "IN":              cand["web_name"],
                    "IN_team":         cand["team_short"],
                    "IN_team_id":      int(cand["team_id"]),
                    "IN_pos":          pos,
                    "IN_price":        float(cand["price"]),  # cost to buy
                    "IN_score":        round(float(cand["fpl_value_score"]), 3),
                    "IN_ep_next":      float(cand.get("ep_next", 0)),
                    "IN_xPts":         round(float(cand.get("xPts", 0)), 2),
                    "IN_xg_p90":       round(float(cand.get("xg_p90",      0)), 3),
                    "IN_xa_p90":       round(float(cand.get("xa_p90",      0)), 3),
                    "IN_dc_hit_rate":  round(float(cand.get("dc_hit_rate", 0)), 2),
                    "IN_dc_pts_p90":   round(float(cand.get("dc_pts_p90",  0)), 2),
                    "IN_cbit_p90":     round(float(cand.get("cbit_p90",    0)), 2),
                    "IN_fix_diff":     round(float(cand.get("avg_fix_diff",0)), 2),
                    "IN_cs_prob":      round(float(cand.get("cs_prob",     0)), 2),
                    "IN_own_pct":      float(cand.get("ownership",         0)),
                    "IN_risk":         cand.get("rotation_risk", "?"),
                    "score_gain":      round(gain, 3),
                    "price_diff":      round(float(cand["price"]) - sell, 1),
                    "_out_fpl_id":     int(out_p["fpl_id"]),
                    "_out_team_id":    out_team,
                })

    if not results:
        return [], my_team_df.copy(), squad_ids.copy()

    results_df = pd.DataFrame(results).sort_values(
        "score_gain", ascending=False
    ).reset_index(drop=True)

    selected_transfers  = []
    updated_squad       = my_team_df.copy()
    updated_squad_ids   = set(squad_ids)
    current_team_counts = dict(team_counts)

    for _ in range(num_transfers):
        best = None
        for _, row in results_df.iterrows():
            if row["OUT"] not in updated_squad["web_name"].values:
                continue
            if any(row["IN"] == prev["IN"] for prev in selected_transfers):
                continue

            sim_counts = dict(current_team_counts)
            for prev in selected_transfers:
                sim_counts[prev["_out_team_id"]] = (
                    sim_counts.get(prev["_out_team_id"], 0) - 1
                )
                sim_counts[prev["IN_team_id"]] = (
                    sim_counts.get(prev["IN_team_id"], 0) + 1
                )
            sim_counts[int(row["_out_team_id"])] = (
                sim_counts.get(int(row["_out_team_id"]), 0) - 1
            )
            if sim_counts.get(int(row["IN_team_id"]), 0) + 1 > 3:
                continue

            best = row
            break

        if best is None:
            break

        selected_transfers.append(best.to_dict())

        out_fpl_id = int(
            updated_squad.loc[
                updated_squad["web_name"] == best["OUT"], "fpl_id"
            ].values[0]
        )
        in_row = scored_players[scored_players["web_name"] == best["IN"]]
        if not in_row.empty:
            in_player = in_row.iloc[[0]].copy()
            # A newly bought player's sell price = their buy price (no profit yet)
            in_player["sell_price"] = in_player["price"]
            updated_squad = updated_squad[updated_squad["fpl_id"] != out_fpl_id]
            updated_squad = pd.concat(
                [updated_squad, in_player], ignore_index=True
            )
            updated_squad_ids.discard(out_fpl_id)
            updated_squad_ids.add(int(in_player.iloc[0]["fpl_id"]))

        results_df = results_df[
            results_df["OUT"] != best["OUT"]
        ].reset_index(drop=True)

    return selected_transfers, updated_squad, updated_squad_ids


# ══════════════════════════════════════════════════════════════════════
#  12.  ROLLING 6-GW PLANNER
# ══════════════════════════════════════════════════════════════════════

def run_rolling_plan(players_base, boot, fixtures, cs_map, my_team_df,
                     squad_ids, itb, free_transfers, next_gw,
                     horizon_gws, risk_appetite, budget_padding):
    current_squad    = my_team_df.copy()
    current_ids      = set(squad_ids)
    current_itb      = itb
    banked_transfers = free_transfers
    plan_summary     = []

    for gw_offset in range(horizon_gws):
        target_gw     = next_gw + gw_offset
        num_transfers = min(banked_transfers, MAX_TRANSFER_BANK)

        scored = rescore_for_gw(players_base, boot, fixtures, target_gw, cs_map, risk_appetite)
        scored_squad = scored[scored["fpl_id"].isin(current_ids)].copy()

        # Carry sell_price from current_squad into the rescored squad.
        # rescore_for_gw rebuilds from players_base which has no sell_price,
        # so we need to map it back from current_squad.
        if "sell_price" in current_squad.columns:
            sp_map = current_squad.set_index("fpl_id")["sell_price"].to_dict()
            scored_squad["sell_price"] = scored_squad["fpl_id"].map(sp_map)
            scored_squad["sell_price"] = scored_squad["sell_price"].fillna(scored_squad["price"])

        transfers, updated_squad, updated_ids = suggest_transfers_for_gw(
            current_ids, scored, scored_squad, current_itb,
            num_transfers=min(num_transfers, 2),
            budget_padding=budget_padding,
        )

        transfers_made = len(transfers)

        # Captain projection
        cap_options = []
        cap_pool = scored[scored["fpl_id"].isin(updated_ids)].copy()
        if not cap_pool.empty:
            cap_pool = compute_gw_projected_pts(cap_pool, target_gw, fixtures)
            cap_pool = cap_pool[cap_pool["has_fixture"]]
            if not cap_pool.empty:
                top3 = cap_pool.sort_values("gw_proj_pts", ascending=False).head(3)
                for _, p in top3.iterrows():
                    cap_options.append({
                        "name":      p["web_name"],
                        "team":      p.get("team_short", "?"),
                        "proj_pts":  round(float(p["gw_proj_pts"]), 2),
                        "fix_diff":  round(float(p.get("gw_diff", 3.0)), 1),
                        "xg_p90":    round(float(p.get("xg_p90", 0)), 3),
                        "xa_p90":    round(float(p.get("xa_p90", 0)), 3),
                        "cs_prob":   round(float(p.get("cs_prob", 0)), 2),
                        "risk":      p.get("rotation_risk", "?"),
                    })

        plan_summary.append({
            "gw":              target_gw,
            "free_transfers":  num_transfers,
            "itb":             round(current_itb, 1),
            "transfers":       transfers,
            "transfers_made":  transfers_made,
            "captain_options": cap_options,
        })

        current_squad = updated_squad
        current_ids   = updated_ids
        banked_transfers = min(
            banked_transfers - transfers_made + 1,
            MAX_TRANSFER_BANK
        )
        banked_transfers = max(banked_transfers, 0)

        for t in transfers:
            current_itb = current_itb + t["OUT_sell_price"] - t["IN_price"]

    return plan_summary


# ══════════════════════════════════════════════════════════════════════
#  13.  STREAMLIT APP
# ══════════════════════════════════════════════════════════════════════


def main():
    st.set_page_config(
        page_title="Datumly — FPL Transfer Optimizer",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # ── Embedded logo (base64) ─────────────────────────────
    LOGO_B64 = "/9j/4AAQSkZJRgABAQAAAQABAAD/4gHYSUNDX1BST0ZJTEUAAQEAAAHIAAAAAAQwAABtbnRyUkdCIFhZWiAH4AABAAEAAAAAAABhY3NwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAA9tYAAQAAAADTLQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAlkZXNjAAAA8AAAACRyWFlaAAABFAAAABRnWFlaAAABKAAAABRiWFlaAAABPAAAABR3dHB0AAABUAAAABRyVFJDAAABZAAAAChnVFJDAAABZAAAAChiVFJDAAABZAAAAChjcHJ0AAABjAAAADxtbHVjAAAAAAAAAAEAAAAMZW5VUwAAAAgAAAAcAHMAUgBHAEJYWVogAAAAAAAAb6IAADj1AAADkFhZWiAAAAAAAABimQAAt4UAABjaWFlaIAAAAAAAACSgAAAPhAAAts9YWVogAAAAAAAA9tYAAQAAAADTLXBhcmEAAAAAAAQAAAACZmYAAPKnAAANWQAAE9AAAApbAAAAAAAAAABtbHVjAAAAAAAAAAEAAAAMZW5VUwAAACAAAAAcAEcAbwBvAGcAbABlACAASQBuAGMALgAgADIAMAAxADb/2wBDAAUDBAQEAwUEBAQFBQUGBwwIBwcHBw8LCwkMEQ8SEhEPERETFhwXExQaFRERGCEYGh0dHx8fExciJCIeJBweHx7/2wBDAQUFBQcGBw4ICA4eFBEUHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh7/wAARCAK8BwgDASIAAhEBAxEB/8QAHQABAAIDAQEBAQAAAAAAAAAAAAYIBQcJBAMCAf/EAGEQAQABAwMBAwUJBw0NBQYHAQABAgMEBQYRBxIhMQgTFUFRFDdWYXGBlrTVCSIyQnaRsxYXNjhSdYKDhJWhsdIYI0ZXYnKFkqKlssTTJHSkweEzNUNVY5NERVNzlNHx8P/EABwBAQEBAAMBAQEAAAAAAAAAAAABAgMGBwUIBP/EADoRAQABAgMEBQsCBQUAAAAAAAABAhEDBDEFBhIhQXFygbEHExQiNDVRYZGhskLBFTJS0fAXI1OC4f/aAAwDAQACEQMRAD8AuKAyyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA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AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAH/2Q=="

    # ── Embedded CSS ───────────────────────────────────────
    CSS = """@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700&display=swap');

:root {
    --pink: #e8366d;
    --pink-lt: #f2648a;
    --bg: #111318;
    --card: #1a1d25;
    --card2: #1e222c;
    --border: #2a2e3a;
    --text: #e8eaf0;
    --dim: #8b90a0;
    --green: #3ddc84;
    --red: #ff5c5c;
    --gold: #fbbf24;
}

/* ── Force dark background ── */
.stApp,
.stApp > header,
.main,
.main .block-container,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Hide default header chrome */
header[data-testid="stHeader"] {
    background: transparent !important;
}
#MainMenu, footer, [data-testid="stToolbar"] {
    visibility: hidden !important;
}

/* ── All text ── */
.stMarkdown, .stMarkdown p, .stMarkdown li,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
.stMarkdown h4, .stMarkdown h5, .stMarkdown h6,
.stText, [data-testid="stText"],
.stCaption, [data-testid="stCaptionContainer"] {
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stCaption, [data-testid="stCaptionContainer"] {
    color: var(--dim) !important;
}

/* ── Branded header ── */
.datumly-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
.datumly-header img {
    height: 72px;
    margin-bottom: 0.5rem;
}
.datumly-subtitle {
    color: var(--dim);
    font-size: 0.85rem;
    letter-spacing: 0.5px;
    margin-top: 0.25rem;
}
.datumly-metrics-row {
    font-size: 0.75rem;
    color: var(--dim);
    letter-spacing: 2px;
    margin-top: 0.4rem;
}

/* ── App title ── */
.app-title {
    text-align: center;
    margin: 0.5rem 0 1.5rem;
}
.app-title h1 {
    font-family: 'DM Sans', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--text);
    margin: 0;
}
.app-title h1 span {
    color: var(--pink);
}
.app-title p {
    color: var(--dim);
    font-size: 0.9rem;
    margin: 0.25rem 0 0;
}

/* ── Stat cards ── */
.stat-cards {
    display: flex;
    gap: 1rem;
    margin: 1.5rem 0;
}
.stat-card {
    flex: 1;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    text-align: left;
}
.stat-card .stat-icon {
    font-size: 1.1rem;
    margin-right: 0.4rem;
    opacity: 0.7;
}
.stat-card .stat-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--text);
    line-height: 1.1;
    margin: 0.3rem 0 0.15rem;
}
.stat-card .stat-value.pink {
    color: var(--pink);
}
.stat-card .stat-label {
    font-size: 0.78rem;
    color: var(--dim);
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.stat-card .stat-sub {
    font-size: 0.72rem;
    color: var(--dim);
    margin-top: 0.15rem;
}

/* ── Section headers ── */
.section-hdr {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--text);
    margin: 2rem 0 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
}
.section-hdr .ico {
    margin-right: 0.5rem;
}

/* ── Transfer colours ── */
.transfer-out { color: var(--red); font-weight: 600; }
.transfer-in  { color: var(--green); font-weight: 600; }
.captain-pick { color: var(--gold); font-weight: 700; }

/* ── Instructions panel ── */
.instructions {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
}
.instructions h4 {
    margin: 0 0 0.75rem;
    color: var(--text);
    font-size: 1rem;
}
.instructions ol {
    color: var(--dim);
    font-size: 0.85rem;
    padding-left: 1.2rem;
    margin: 0;
}
.instructions li { margin-bottom: 0.3rem; }
.instructions .warn {
    color: var(--gold);
    font-weight: 600;
    font-size: 0.82rem;
    margin-top: 0.75rem;
}

/* ── Footer ── */
.datumly-footer {
    text-align: center;
    padding: 2rem 0 1.5rem;
    color: var(--dim);
    font-size: 0.8rem;
    border-top: 1px solid var(--border);
    margin-top: 3rem;
}

/* ══════════════════════════════════════════════════════════
   STREAMLIT COMPONENT OVERRIDES
   ══════════════════════════════════════════════════════════ */

/* ── Metric cards ── */
div[data-testid="stMetric"],
[data-testid="stMetricValue"] {
    background-color: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 14px 18px !important;
}
[data-testid="stMetricLabel"] {
    color: var(--dim) !important;
    font-size: 0.78rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}
[data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-weight: 700 !important;
    border: none !important;
    padding: 0 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-highlight"] {
    background-color: transparent !important;
}
.stTabs [data-baseweb="tab-border"] {
    display: none !important;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 0px !important;
    background: var(--card) !important;
    border-radius: 8px !important;
    padding: 4px !important;
    border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 6px !important;
    color: var(--dim) !important;
    font-weight: 500 !important;
    padding: 8px 20px !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    background: var(--pink) !important;
    color: white !important;
    border-radius: 6px !important;
}

/* ── Buttons ── */
.stButton > button,
button[kind="primary"],
[data-testid="stBaseButton-primary"] {
    background-color: var(--pink) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 1.5rem !important;
}
.stButton > button:hover,
button[kind="primary"]:hover,
[data-testid="stBaseButton-primary"]:hover {
    background-color: var(--pink-lt) !important;
    border: none !important;
}

/* ── Number input ── */
.stNumberInput input,
[data-testid="stNumberInput"] input {
    background-color: var(--card2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
}
.stNumberInput label,
[data-testid="stNumberInput"] label {
    color: var(--dim) !important;
}

/* ── Radio buttons ── */
.stRadio > div[role="radiogroup"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 4px 8px !important;
}
.stRadio label,
div[data-baseweb="radio"] label {
    color: var(--dim) !important;
}
div[data-baseweb="radio"] > div:first-child > div {
    border-color: var(--pink) !important;
}
div[data-baseweb="radio"] > div:first-child > div > div {
    background-color: var(--pink) !important;
}

/* ── Selectbox ── */
.stSelectbox label { color: var(--dim) !important; }
.stSelectbox [data-baseweb="select"],
.stSelectbox [data-baseweb="select"] > div {
    background-color: var(--card2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}

/* ── Slider ── */
.stSlider label { color: var(--dim) !important; }

/* ── Expanders ── */
details[data-testid="stExpander"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    margin-bottom: 0.75rem !important;
}
details[data-testid="stExpander"] summary {
    color: var(--text) !important;
    font-weight: 600 !important;
}
details[data-testid="stExpander"] summary:hover {
    color: var(--pink) !important;
}
details[data-testid="stExpander"] > div[data-testid="stExpanderDetails"] {
    border-top: 1px solid var(--border) !important;
}

/* ── Progress bar ── */
.stProgress > div > div > div {
    background-color: var(--pink) !important;
}
.stProgress > div > div {
    background-color: var(--card) !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* ── Divider ── */
hr {
    border-color: var(--border) !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"],
[data-testid="stSidebarContent"] {
    background-color: var(--card) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * {
    font-family: 'DM Sans', sans-serif !important;
}
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] .stMarkdown p {
    color: var(--text) !important;
}
section[data-testid="stSidebar"] hr {
    border-color: var(--border) !important;
}
"""

    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

    # ── Branded header ──────────────────────────────────────
    st.markdown(f"""
    <div class="datumly-header">
        <img src="data:image/png;base64,{LOGO_B64}" alt="Datumly" />
        <div class="datumly-subtitle">Data-driven FPL intelligence</div>
        <div class="datumly-metrics-row">xG &nbsp;·&nbsp; xA &nbsp;·&nbsp; CS% &nbsp;·&nbsp; xPts &nbsp;·&nbsp; FDR</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="app-title">
        <h1>⚽ FPL Transfer <span>Optimizer</span> 📊</h1>
        <p>Rolling 6-GW Strategy Planner for Fantasy Premier League</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Input row: Manager ID + Run button ──────────────────
    input_col, spacer, btn_col = st.columns([2, 5, 2])
    with input_col:
        fpl_id = st.number_input(
            "FPL Manager ID:",
            min_value=1, value=210697, step=1,
            help="Find your ID at fantasy.premierleague.com/entry/YOUR_ID/history",
        )
    with btn_col:
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("🔍  Run Analysis", type="primary", use_container_width=True)

    # ── Sidebar (advanced settings) ─────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:1rem 0 0.5rem;">
            <span style="font-size:1.3rem;font-weight:700;color:#e8eaf0;">⚙️ Settings</span>
        </div>
        """, unsafe_allow_html=True)
        horizon_gws = st.slider("Gameweeks to plan ahead", 1, 10, 6)
        risk_appetite = st.selectbox(
            "Risk appetite",
            ["safe", "balanced", "differential"],
            index=1,
        )
        dc_lookback = st.slider("DC stats lookback (GWs)", 3, 15, 8)
        budget_padding = st.number_input(
            "Budget padding (£m)", min_value=0.0, max_value=5.0,
            value=0.0, step=0.1,
        )

    # ── Main area: waiting state ────────────────────────────
    if not run_btn:
        st.markdown("""
        <div class="stat-cards">
            <div class="stat-card">
                <span class="stat-icon" style="color:#e8366d;">↑</span>
                <div class="stat-value pink">—</div>
                <div class="stat-label">Free Transfers</div>
            </div>
            <div class="stat-card">
                <span class="stat-icon">🛡️</span>
                <div class="stat-value">—</div>
                <div class="stat-label">Chips Available</div>
                <div class="stat-sub">Run analysis to see</div>
            </div>
            <div class="stat-card">
                <span class="stat-icon">💰</span>
                <div class="stat-value">£ —</div>
                <div class="stat-label">ITB (£m)</div>
            </div>
            <div class="stat-card">
                <span class="stat-icon">📅</span>
                <div class="stat-value">GW —</div>
                <div class="stat-label">Next GW</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="instructions">
            <h4>👥 Instructions</h4>
            <ol>
                <li>Enter your FPL Manager ID above.</li>
                <li>Adjust settings in the sidebar and click <strong>"Run Analysis"</strong>.</li>
            </ol>
            <div class="warn">⚠️ Note: <strong>Ensure your ID is correct!</strong></div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Data loading ────────────────────────────────────────
    progress = st.progress(0, text="Loading FPL data …")

    try:
        boot    = get_bootstrap()
        players = build_fpl_table(boot)
    except Exception as e:
        st.error(f"Failed to load FPL data: {e}")
        return
    progress.progress(0.10, text=f"{len(players)} players loaded.")

    try:
        picks_data, active_gw, next_gw = get_my_team(fpl_id, boot)
    except Exception as e:
        st.error(f"Could not fetch team for manager {fpl_id}: {e}")
        return
    picks = pd.DataFrame(picks_data["picks"])
    itb   = picks_data["entry_history"]["bank"]  / 10
    tv    = picks_data["entry_history"]["value"] / 10
    progress.progress(0.15, text=f"Team loaded. GW{active_gw} → planning from GW{next_gw}")

    free_transfers, chips_available = get_free_transfers(fpl_id, active_gw)
    progress.progress(0.20, text=f"{free_transfers} free transfers banked.")

    fixtures = get_fixtures()
    fix_df   = compute_fixture_difficulty(boot, fixtures, active_gw, horizon_gws)
    progress.progress(0.25, text="Fixtures analysed.")

    team_def = get_team_defensive_stats(boot)
    cs_map   = team_def.set_index("team_id")["cs_prob"].to_dict()
    progress.progress(0.30, text="Clean sheet probabilities computed.")

    progress.progress(0.32, text="Fetching DC stats (this takes ~30s) …")
    dc_bar = st.progress(0, text="Fetching DC stats …")
    dc_df  = fetch_dc_stats(boot, active_gw, lookback=dc_lookback, progress_bar=dc_bar)
    dc_bar.empty()
    progress.progress(0.70, text="DC stats complete.")

    progress.progress(0.72, text="Fetching Understat xG/xA data …")
    us_df = get_understat_stats()
    progress.progress(0.76, text="Understat loaded.")

    progress.progress(0.77, text="Fetching FBref xG/xA data (StatsBomb model) …")
    fb_df = get_fbref_stats()
    progress.progress(0.80, text=f"FBref: {len(fb_df)} players loaded." if not fb_df.empty else "FBref: unavailable.")

    progress.progress(0.82, text="Building player pool …")
    fix_map = fix_df.set_index("team_id")[["avg_difficulty","num_fixtures","has_dgw"]].to_dict("index")
    players["avg_fix_diff"] = players["team_id"].map(lambda t: fix_map.get(t, {}).get("avg_difficulty", 3.0))
    players["num_fixtures"] = players["team_id"].map(lambda t: fix_map.get(t, {}).get("num_fixtures", horizon_gws))
    players["has_dgw"]   = players["team_id"].map(lambda t: fix_map.get(t, {}).get("has_dgw", 0))
    players["fix_score"] = (6 - players["avg_fix_diff"]) * (players["num_fixtures"] / horizon_gws)
    players["cs_prob"]   = players["team_id"].map(cs_map).fillna(0.25)

    players = players.merge(dc_df, on="fpl_id", how="left")
    for col in ["dc_pts_p90", "dc_hit_rate", "cbit_p90"]:
        players[col] = players[col].fillna(0) if col in players.columns else 0.0

    players = match_understat(players, us_df)
    players = match_fbref(players, fb_df)
    players = merge_dual_source_xg(players)
    players = add_risk_flags(players)
    players = apply_position_scores(players)
    players = apply_risk_appetite(players, risk_appetite)

    my_ids     = set(int(x) for x in picks["element"].tolist())
    my_team_df = players[players["fpl_id"].isin(my_ids)].copy().reset_index(drop=True)

    sell_price_map = compute_sell_prices(my_ids, players, fpl_id)
    if sell_price_map:
        my_team_df["sell_price"] = my_team_df["fpl_id"].map(sell_price_map)
        my_team_df["sell_price"] = my_team_df["sell_price"].fillna(my_team_df["price"])
    else:
        my_team_df["sell_price"] = my_team_df["price"]

    progress.progress(0.90, text="Running rolling plan …")

    plan = run_rolling_plan(
        players_base=players, boot=boot, fixtures=fixtures, cs_map=cs_map,
        my_team_df=my_team_df, squad_ids=my_ids, itb=itb,
        free_transfers=free_transfers, next_gw=next_gw,
        horizon_gws=horizon_gws, risk_appetite=risk_appetite,
        budget_padding=budget_padding,
    )

    progress.progress(1.0, text="Analysis complete!")
    progress.empty()

    # ════════════════════════════════════════════════════════
    #  DISPLAY RESULTS
    # ════════════════════════════════════════════════════════

    chips_str = ", ".join(chips_available) if chips_available else "None"
    st.markdown(f"""
    <div class="stat-cards">
        <div class="stat-card">
            <span class="stat-icon" style="color:#e8366d;">↑</span>
            <div class="stat-value pink">{free_transfers}</div>
            <div class="stat-label">Free Transfers</div>
        </div>
        <div class="stat-card">
            <span class="stat-icon">🛡️</span>
            <div class="stat-value" style="font-size:1.4rem;">{len(chips_available)}</div>
            <div class="stat-label">Chips Available</div>
            <div class="stat-sub">{chips_str}</div>
        </div>
        <div class="stat-card">
            <span class="stat-icon">💰</span>
            <div class="stat-value">£{itb:.1f}</div>
            <div class="stat-label">ITB (£m)</div>
            <div class="stat-sub">+£{tv:.1f}m TV</div>
        </div>
        <div class="stat-card">
            <span class="stat-icon">📅</span>
            <div class="stat-value">GW{next_gw}</div>
            <div class="stat-label">Next GW</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabbed content ──────────────────────────────────────
    tab_squad, tab_fixtures, tab_insights = st.tabs(["👥 Your Squad ⚽", "Fixture Difficulty 📅", "Insights 📊"])

    with tab_squad:
        show_cols = ["web_name","team_short","price","sell_price","form","xPts","cs_prob",
                     "xg_p90","xa_p90","dc_hit_rate","ownership","fpl_value_score","rotation_risk"]
        show_cols = [c for c in show_cols if c in my_team_df.columns]

        pos_filter = st.radio("Position filter", ["ALL","GKP","DEF","MID","FWD"],
                              horizontal=True, label_visibility="collapsed")

        filtered = my_team_df if pos_filter == "ALL" else my_team_df[my_team_df["position"] == pos_filter]

        if not filtered.empty:
            display = filtered[show_cols].copy()
            nice = {"web_name":"Player","team_short":"Team","price":"Price","sell_price":"Sell",
                    "form":"Form","xPts":"xPts","cs_prob":"CS%","xg_p90":"xG/90","xa_p90":"xA/90",
                    "dc_hit_rate":"DC%","ownership":"Own%","fpl_value_score":"Score","rotation_risk":"Risk"}
            display = display.rename(columns={c: nice.get(c,c) for c in display.columns})
            st.dataframe(display, use_container_width=True, hide_index=True)

    with tab_fixtures:
        st.markdown(f"#### 📅 Fixture Difficulty — GW{active_gw+1} to GW{active_gw+horizon_gws}")
        fs = fix_df.sort_values("avg_difficulty")
        col_easy, col_hard = st.columns(2)
        with col_easy:
            st.markdown("**🟢 Easiest Runs**")
            easy = fs[["name","avg_difficulty","num_fixtures","has_dgw"]].head(8).copy()
            easy.columns = ["Team","Avg Diff","Fixtures","DGW"]
            st.dataframe(easy, use_container_width=True, hide_index=True)
        with col_hard:
            st.markdown("**🔴 Hardest Runs**")
            hard = fs[["name","avg_difficulty","num_fixtures","has_dgw"]].tail(8).copy()
            hard.columns = ["Team","Avg Diff","Fixtures","DGW"]
            st.dataframe(hard, use_container_width=True, hide_index=True)

    with tab_insights:
        st.markdown(f"#### 🛡️ Top DC Earners (last {dc_lookback} GWs)")
        dc_col1, dc_col2 = st.columns(2)
        for col_target, (pos, label) in zip([dc_col1, dc_col2], [("DEF","10 CBIT"), ("MID","12 CBIRT")]):
            sub = players[(players["position"]==pos) & (players["status"].isin(["a","d","n"])) & (players["minutes"]>=180)].sort_values("dc_hit_rate", ascending=False).head(8)
            if sub.empty: continue
            dc_show = [c for c in ["web_name","team_short","price","dc_hit_rate","dc_pts_p90","cbit_p90","cs_prob","rotation_risk"] if c in sub.columns]
            with col_target:
                st.markdown(f"**{pos}** — threshold: {label}/match")
                st.dataframe(sub[dc_show].copy().rename(columns={c: c.replace("_"," ").title() for c in dc_show}), use_container_width=True, hide_index=True)

    # ── Rolling transfer plan ───────────────────────────────
    st.markdown(f"#### 📋 Rolling {horizon_gws}-GW Transfer Plan (GW{next_gw} → GW{next_gw + horizon_gws - 1})")

    for gw_plan in plan:
        gw = gw_plan["gw"]
        with st.expander(
            f"**GW{gw}** — {gw_plan['transfers_made']} transfer{'s' if gw_plan['transfers_made'] != 1 else ''} | FT: {gw_plan['free_transfers']} | ITB: £{gw_plan['itb']:.1f}m",
            expanded=(gw_plan["transfers_made"] > 0),
        ):
            if not gw_plan["transfers"]:
                st.success(f"No beneficial transfers — hold for GW{gw}.")
            else:
                for i, t in enumerate(gw_plan["transfers"], 1):
                    c1, c2, c3 = st.columns([5, 1, 5])
                    with c1:
                        st.markdown(f'''<span class="transfer-out">⬇ OUT: {t["OUT"]}</span> [{t["OUT_pos"]}] sell £{t["OUT_sell_price"]:.1f}m (mkt £{t["OUT_buy_price"]:.1f}m)<br>ep: {t["OUT_ep_next"]:.2f} · DC: {t["OUT_dc_hit_rate"]:.0%} · fix: {t["OUT_fix_diff"]:.2f} · {t["OUT_risk"]}''', unsafe_allow_html=True)
                    with c2:
                        st.markdown("### →")
                    with c3:
                        st.markdown(f'''<span class="transfer-in">⬆ IN: {t["IN"]}</span> [{t["IN_pos"]}] ({t["IN_team"]}) £{t["IN_price"]:.1f}m<br>ep: {t["IN_ep_next"]:.2f} · **xPts: {t["IN_xPts"]:.2f}** · xG: {t["IN_xg_p90"]:.3f} · xA: {t["IN_xa_p90"]:.3f} · DC: {t["IN_dc_hit_rate"]:.0%} · CS: {t["IN_cs_prob"]:.0%}<br>**Score gain: {t["score_gain"]:+.3f}** · Price diff: {t["price_diff"]:+.1f}m''', unsafe_allow_html=True)
                    if i < len(gw_plan["transfers"]):
                        st.divider()

            if gw_plan["captain_options"]:
                st.markdown(f"**🎖️ Captain options for GW{gw}:**")
                cap_df = pd.DataFrame(gw_plan["captain_options"])
                cap_df.columns = ["Player","Team","Proj Pts","Fix Diff","xG/90","xA/90","CS Prob","Risk"]
                st.dataframe(cap_df, use_container_width=True)

    st.markdown("""
    <div class="datumly-footer">
        Built with ⚽ and ❤️ by <strong>Datumly</strong>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
