# ⚽ FPL Transfer Optimizer v5.2

A rolling 6-gameweek Fantasy Premier League transfer planner built with Streamlit.

## What's new in v5.2

**Scoring corrections** verified against the [official FPL points table](https://www.premierleague.com/en/news/2174909):

- **Captain projection fixed**: GKP/DEF clean sheet multiplier corrected from `cs × 6.0` to `cs × 4.0` — a clean sheet is worth 4 points, not 6
- **DEF scoring rebalanced**: CS weight raised to 0.30 (biggest single DEF earner at 4pts), DC weight reduced to 0.15 (DC=2pts, half a CS), attacking returns raised to 0.18
- **MID scoring rebalanced**: CS weight dropped to 0.02 (only 1pt for MID — near irrelevant), xG weight raised to 0.22 (goals=5pts, primary MID earner), xA raised to 0.16
- **Yellow card penalty** increased to -0.06 across all positions

## Features

* **Rolling GW-by-GW plan** — suggests the best 1–2 transfers for each of the next 6 gameweeks, updating the squad after each recommendation so suggestions are always coherent
* **GW-specific captain projections** — projects points using actual fixture difficulty for each specific GW rather than a static `ep_next` value, so the captain recommendation changes each week
* **Corrected DC model** — defensive contributions modelled per official 2025/26 rules: DEF needs ≥10 CBIT, MID needs ≥12 CBIRT, capped at 2pts/match; GKPs excluded
* **Auto-detects free transfers** — reads your full transfer history to compute exactly how many free transfers you have banked (cap: 5 per 2025/26 rules)
* **Deadline-aware** — automatically plans for the next GW once the current deadline has passed
* **3-per-team rule enforced** — all transfer suggestions respect the maximum 3 players per club rule

## Data Sources

* [FPL API](https://fantasy.premierleague.com/api) — squad, prices, form, expected points, GW history, DC stats
* [Understat](https://understat.com) — xG, xA, npxG, shots, key passes per player

## How to Use

1. Go to the app URL
2. Enter your **FPL Manager ID** in the sidebar (find it in the URL on the FPL website: `fantasy.premierleague.com/entry/YOUR_ID/history`)
3. Adjust settings (horizon, risk appetite, DC lookback)
4. Click **▶ RUN ANALYSIS**

## Deploy Your Own Copy

### 1. Fork this repo

Click **Fork** at the top right of this page.

### 2. Deploy on Streamlit Community Cloud (free)

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **New app**
4. Select your forked repo, set the main file as `app.py`
5. Click **Deploy**

Your app will be live at `https://your-username-fpl-optimizer.streamlit.app` within a few minutes.

### 3. Updates

Any `git push` to your repo automatically redeploys the app.

## Run Locally

```
pip install -r requirements.txt
streamlit run app.py
```

## Scoring Reference (2025/26)

### Points Table

| Action | GKP | DEF | MID | FWD |
|--------|-----|-----|-----|-----|
| Clean Sheet | 4pts | 4pts | 1pt | — |
| Goal | — | 6pts | 5pts | 4pts |
| Assist | 3pts | 3pts | 3pts | 3pts |
| DC | — | 2pts | 2pts | 2pts |
| Yellow | -1pt | -1pt | -1pt | -1pt |

### DC Scoring Rules

| Position | Actions counted | Threshold | Points |
|----------|----------------|-----------|--------|
| GKP | N/A | N/A | N/A |
| DEF | Clearances + Blocks + Interceptions + Tackles (CBIT) | ≥ 10 | 2 pts |
| MID/FWD | CBIT + Ball Recoveries (CBIRT) | ≥ 12 | 2 pts |

Points are capped at 2 per match regardless of total actions.

### Captain Projection Model (v5.2 corrected)

Points are projected per-GW using:

* **GKP**: `cs_prob × 4 + saves_p90 / 3 + 2`
* **DEF**: `cs_prob × 4 + xg_p90 × 6 + xa_p90 × 3 + dc_hit_rate × 2 + 2`
* **MID**: `cs_prob × 1 + xg_p90 × 5 + xa_p90 × 3 + dc_hit_rate × 2 + 2`
* **FWD**: `xg_p90 × 4 + xa_p90 × 3 + 2`

Each is multiplied by a fixture difficulty factor: difficulty 3 = 1.0×, easier = up to 1.2×, harder = down to 0.8×.

> **v5.1 bug fixed**: GKP and DEF previously used `cs_prob × 6` — a clean sheet is worth 4 points, not 6.

## License

MIT
