# NHL Unders Board

Scores every NHL team total on the slate: one team's offense against the opposing
skaters and starting goalie. Green = good spot to score, red = under spot. The board
logs its own pregame scores, grades them, checks its accuracy every day, re-runs a
multi-season backtest every week, and adjusts its weights when the data says so.

| Page | What it shows |
|---|---|
| `index.html` — the board | Today's slate, sortable, with goalie picker and weight sliders |
| `monitor.html` — accuracy monitor | How this season's pregame scores are doing vs the backtest, data health, model history |
| `backtest.html` — backtest | Five-plus seasons replayed game by game: which inputs predict goals, and how well the score ranks games |

## What runs on its own

**Hourly, 9am–11pm ET** (`.github/workflows/daily.yml`)
1. `build.py` pulls the latest data and rebuilds the board.
2. It saves each of today's games to `monitor/log_<season>.csv`. A row keeps updating
   until puck drop (so it picks up confirmed goalies), then never changes.
3. `monitor/grade.py` fills in final goals for finished games (includes OT, empty-net
   and the shootout winner, like most books).
4. `monitor/accuracy.py` measures the season so far and runs health checks.
5. `monitor/alerts.py` opens or closes GitHub issues.

**Every Monday ~7am ET** (`.github/workflows/weekly.yml`)
1. `backtest/backtest.py` replays every game from 2021-22 through yesterday using
   only data available before each game.
2. `backtest/learn.py` refits the weights and changes `model.json` only if the
   guardrails below pass.
3. `backtest/report.py` rebuilds `backtest.html`, and the board is rebuilt with the
   current model.

## Where the data comes from
| Data | Source | Updated |
|---|---|---|
| Schedule, scores, standings, team stats, rosters | NHL API (`api-web.nhle.com`, `api.nhle.com`) | Live |
| Expected goals, shot quality, goalie GSAx | MoneyPuck (`moneypuck.com`, `peter-tanner.com/moneypuck`) | Daily, overnight |
| Starting goalies (Confirmed / Likely / Projected) | Daily Faceoff | Through the day |

All free and public. Nothing needs a laptop.

## How the model learns (guardrails)
The learner refits the weights weekly on the last four seasons plus this one
(this season counts double), then tests on the last 6 weeks of games it didn't fit on.
It changes the board only when **all** of these hold:
- 500+ team-games played this season (about mid-November)
- 28+ days since the last change
- the new weights rank the recent games better (by 0.3+ points of ranking accuracy)
- and do no worse on last season
- each input moves at most 10 points per change, in 5% steps

Every change is a commit to `model.json` with the reason in its `history`, and opens
an issue. To undo one, revert that commit or edit the weights back.

## Alerts
Issues labeled `nhl-alert` (GitHub emails the repo owner; you're @mentioned):

| Alert | When |
|---|---|
| Board didn't refresh / games not logged / games not graded | Data pipeline broke |
| Starting goalies aren't coming through | >50% of today's games have no starter report by 5pm |
| Backtest data stale | MoneyPuck's file is 4+ days behind in season |
| Score ranking worse than backtest | 500+ graded games and the 95% range sits below the worst backtest season |
| An input stopped predicting | 800+ games and a weighted input points the wrong way |
| Projections running high/low | Last 300 games off by 0.2+ goals, beyond noise |
| Home/rest effects drifted, or a large weight shift | From the weekly learner — suggestions for you to review |
| Model updated | The learner changed the weights (stays open until you close it) |

Data alerts close themselves when the check passes again. A failed workflow run also emails you.

## Run by hand
```
python build.py                 # board + today's log
python monitor/grade.py         # grade finished games
python monitor/accuracy.py      # monitor page + alerts (status.json)
python backtest/backtest.py     # full backtest (downloads ~150 MB the first time)
python backtest/learn.py        # learner (may edit model.json)
python backtest/report.py       # backtest page
```
Or use **Actions → Run workflow** on GitHub.

## Files
- `model.json` — the live weights, projection settings, and change history
- `engine.js` — the scoring engine, shared by the board and the daily log
- `template.html` — board page; `build.py` fills it in
- `monitor/` — log, grading, accuracy scan, alerts
- `backtest/` — backtest, learner, report
