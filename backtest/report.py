"""Turn results.json + team_games_scored.csv into report.html (run after backtest.py)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from backtest import auc, BOARD_TO_BT

HERE = Path(__file__).parent
r = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
model = json.loads((HERE.parent / "model.json").read_text(encoding="utf-8"))
hist = model.get("history", [])
r["model"] = model
r["prev_weights"] = hist[-2]["weights"] if len(hist) > 1 else {"goalie": .30, "def": .20, "off": .30, "st": .10, "l5": .10, "fin": 0}
learn = HERE / "learn_status.json"
r["learn"] = json.loads(learn.read_text(encoding="utf-8")) if learn.exists() else None
f = pd.read_csv(HERE / "team_games_scored.csv")

# situational averages
sit = []
for label, col in [("Home", "home"), ("Team on back-to-back", "b2b"), ("Opponent on back-to-back", "opp_b2b")]:
    a, b = f[f[col] == 1], f[f[col] == 0]
    sit.append(dict(label=label, yes=round(a.gf.mean(), 2), no=round(b.gf.mean(), 2),
                    yes_over=round(a.over.mean(), 3), no_over=round(b.over.mean(), 3), n=int(len(a))))

# projection with the updated settings, and its calibration
base = f.proj / np.where(f.home == 1, 1.025, .975) / np.where(f.b2b == 1, .96, 1) / np.where(f.opp_b2b == 1, 1.04, 1)
raw = base * np.where(f.home == 1, 1.035, .965) * np.where(f.b2b == 1, .93, 1) * np.where(f.opp_b2b == 1, 1.06, 1)
lam = raw.mean() + 0.85 * (raw - raw.mean())
p = 1 - np.exp(-lam) * (1 + lam + lam * lam / 2)
bins = pd.cut(p, [0, .45, .5, .55, .6, .65, .7, 1])
cal = f.assign(p=p).groupby(bins, observed=False).agg(pred=("p", "mean"), act=("over", "mean"), n=("gf", "size"))
r["calibration_new"] = [dict(pred=round(x.pred, 3), actual=round(x.act, 3), n=int(x.n)) for _, x in cal.iterrows() if x.n > 50]
r["proj_new_auc"] = round(auc(lam, f.over), 4)
r["situational"] = sit

# suggested score on every season (for the by-season table)
W = {BOARD_TO_BT[k]: v for k, v in model["weights"].items() if v > 0}
f["score_new"] = sum(f["s_" + k].fillna(0) * w for k, w in W.items()) / sum(f["s_" + k].notna() * w for k, w in W.items())
for row in r["by_season"]:
    y = int(row["season"][:4])
    row["new"] = round(auc(f[f.season == y].score_new, f[f.season == y].over), 4)
q = pd.qcut(f.score_new.rank(method="first"), 5, labels=False)
r["new_quintiles"] = [dict(q=int(i) + 1, goals=round(x.goals, 3), over=round(x.over, 4)) for i, x in
                      f.assign(q=q).groupby("q").agg(goals=("gf", "mean"), over=("over", "mean")).iterrows()]
bands = pd.cut(f.score_new, [-1, 30, 40, 50, 60, 70, 101], labels=["Under 30", "30–40", "40–50", "50–60", "60–70", "70+"])
r["bands_new"] = [dict(band=str(i), goals=round(x.goals, 2), over=round(x.over, 3), n=int(x.n)) for i, x in
                  f.groupby(bands, observed=False).agg(goals=("gf", "mean"), over=("over", "mean"), n=("gf", "size")).iterrows()]
r["auc_new_all"] = round(auc(f.score_new, f.over), 4)
Wp = {BOARD_TO_BT[k]: v for k, v in r["prev_weights"].items() if v > 0}
f["score_prev"] = sum(f["s_" + k].fillna(0) * w for k, w in Wp.items()) / sum(f["s_" + k].notna() * w for k, w in Wp.items())
r["auc_prev_all"] = round(auc(f.score_prev, f.over), 4)

tpl = (HERE / "report_template.html").read_text(encoding="utf-8")
head = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"></head><body>')
html = tpl.replace("/*__DATA__*/null", json.dumps(r, default=float))
(HERE / "report.html").write_text(html, encoding="utf-8")
(HERE.parent / "backtest.html").write_text(head + html + "</body></html>", encoding="utf-8")
print("wrote report.html", r["auc_new_all"], r["bands_new"])
