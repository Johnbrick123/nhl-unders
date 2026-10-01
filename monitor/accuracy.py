"""Daily accuracy scanner.

Reads the graded log, measures how the board's pregame scores are doing this season,
compares that with the backtest, runs health checks on today's data, and writes:
  monitor/status.json   metrics + any alerts (alerts.py turns alerts into GitHub issues)
  monitor.html          the Accuracy Monitor page
Run:  python monitor/accuracy.py
"""
import csv, json, math
from datetime import date, datetime, timedelta
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
MODEL = json.loads((ROOT / "model.json").read_text(encoding="utf-8"))
EXP = MODEL["expect"]
SUBS = [("off", "Offense", "Offense rating"), ("def", "Opp defense", "Opp xGA/G"), ("goalie", "Opp goalie", "Opp goalie factor (shrunk)"),
        ("st", "PP vs PK", "PP vs PK matchup"), ("l5", "Goals L5", "Goals L5"), ("fin", "Finishing", "Luck (GF ÷ xGF)")]
MIN_RANK_N = 500      # graded team-games before ranking alerts can fire (~mid-November)
MIN_INPUT_N = 800


# ---------------------------------------------------------------- stats
def auc(scores, ys):
    pairs = sorted(zip(scores, ys))
    ranks, i = [0.0] * len(pairs), 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + j) / 2 + 1
        i = j + 1
    npos = sum(y for _, y in pairs)
    nneg = len(pairs) - npos
    if not npos or not nneg:
        return None, None, None
    a = (sum(r for r, (_, y) in zip(ranks, pairs) if y) - npos * (npos + 1) / 2) / (npos * nneg)
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)                                   # Hanley-McNeil
    se = math.sqrt((a * (1 - a) + (npos - 1) * (q1 - a * a) + (nneg - 1) * (q2 - a * a)) / (npos * nneg))
    return a, a - 1.96 * se, a + 1.96 * se


def spearman(x, y):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2
            i = j + 1
        return r
    if len(x) < 30:
        return None, None
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx); syy = sum((b - my) ** 2 for b in ry)
    rho = sxy / math.sqrt(sxx * syy) if sxx and syy else 0
    return rho, 1.96 / math.sqrt(len(x) - 3)


def mean(v):
    return sum(v) / len(v) if v else None


# ---------------------------------------------------------------- main
def main():
    now = datetime.now()
    today = date.today().isoformat()
    logs = sorted(HERE.glob("log_*.csv"))
    rows = []
    if logs:
        with open(logs[-1], newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    season = logs[-1].stem[4:] if logs else "—"
    num = lambda v: float(v) if v not in ("", None) else None
    g = [r for r in rows if r["actual"] != ""]
    for r in g:
        r["y"] = 1 if int(r["actual"]) >= 3 else 0
    alerts = []

    def alert(key, level, title, body):
        alerts.append(dict(key=key, level=level, title=title, body=body))

    # -- season metrics
    st = dict(season=season, logged=len(rows), graded=len(g), today=today, built=now.strftime("%Y-%m-%d %H:%M"),
              model_version=MODEL["version"], weights=MODEL["weights"], expect=EXP)
    if g:
        st["over_rate"] = mean([r["y"] for r in g])
        st["avg_goals"] = mean([int(r["actual"]) for r in g])
        st["avg_proj"] = mean([float(r["proj"]) for r in g])
        a, lo, hi = auc([float(r["score"]) for r in g], [r["y"] for r in g])
        st["auc"], st["auc_lo"], st["auc_hi"] = a, lo, hi
        st["auc_proj"] = auc([float(r["proj"]) for r in g], [r["y"] for r in g])[0]
        bands = [("Under 30", 0, 30), ("30–40", 30, 40), ("40–50", 40, 50), ("50–60", 50, 60), ("60–70", 60, 70), ("70+", 70, 101)]
        st["bands"] = []
        for lab, a0, a1 in bands:
            b = [r for r in g if a0 <= float(r["score"]) < a1]
            st["bands"].append(dict(band=lab, n=len(b), over=mean([r["y"] for r in b]),
                                    goals=mean([int(r["actual"]) for r in b])))
        cal = []
        for a0, a1 in [(0, .45), (.45, .5), (.5, .55), (.55, .6), (.6, .65), (.65, .7), (.7, 1)]:
            b = [r for r in g if a0 <= float(r["p_over"]) < a1]
            if b:
                cal.append(dict(lo=a0, hi=a1, n=len(b), pred=mean([float(r["p_over"]) for r in b]), actual=mean([r["y"] for r in b])))
        st["calibration"] = cal
        bt = {}
        res = ROOT / "backtest" / "results.json"
        if res.exists():
            bt = {i["label"]: i["rho"] for i in json.loads(res.read_text(encoding="utf-8")).get("inputs", [])}
        st["inputs"] = []
        for k, lab, btlab in SUBS:
            pts = [(float(r["s_" + k]), int(r["actual"])) for r in g if r["s_" + k] != ""]
            rho, ci = spearman([p[0] for p in pts], [p[1] for p in pts])
            st["inputs"].append(dict(key=k, label=lab, n=len(pts), rho=rho, ci=ci, backtest=bt.get(btlab), weight=MODEL["weights"].get(k, 0)))
            if rho is not None and len(pts) >= MIN_INPUT_N and MODEL["weights"].get(k, 0) > 0 and rho + ci < 0:
                alert(f"input-{k}", "action", f"{lab} has stopped predicting goals",
                      f"Season to date, {lab} is pointing the wrong way (correlation {rho:+.3f}, n={len(pts)}) "
                      f"while it carries {MODEL['weights'][k]:.0%} of the score. The backtest had it at {bt.get(btlab, 0):+.3f}. "
                      f"The weekly learner will lower it if the drop holds on fresh games; check the data source too.")
        # red/green spots
        red = [r for r in g if float(r["score"]) < 35]
        green = [r for r in g if float(r["score"]) >= 65]
        st["red"] = dict(n=len(red), under=mean([1 - r["y"] for r in red]))
        st["green"] = dict(n=len(green), over=mean([r["y"] for r in green]))
        # ranking alert
        if len(g) >= MIN_RANK_N and hi is not None and hi < EXP["auc_low"]:
            alert("ranking-drop", "action", "The score is ranking games worse than the backtest",
                  f"Season-to-date ranking accuracy is {a:.1%} (95% range {lo:.1%}–{hi:.1%}, n={len(g)}). "
                  f"The backtest's worst season was {EXP['auc_low']:.1%}. Something changed: league scoring, a broken input, "
                  f"or weights that no longer fit. See the Inputs table on the monitor page.")
        # calibration drift on the most recent 300 graded rows
        recent = sorted(g, key=lambda r: r["date"])[-300:]
        if len(recent) >= 300:
            act = [int(r["actual"]) for r in recent]; pr = [float(r["proj"]) for r in recent]
            diff = mean(act) - mean(pr)
            sd = math.sqrt(sum((x - mean(act)) ** 2 for x in act) / (len(act) - 1))
            z = diff / (sd / math.sqrt(len(act)))
            st["recent_bias"] = dict(n=len(recent), diff=diff, z=z)
            if abs(diff) > 0.2 and abs(z) > 3:
                alert("projection-bias", "watch", f"Projections are running {'low' if diff > 0 else 'high'} by {abs(diff):.2f} goals",
                      f"Over the last {len(recent)} graded team-games, teams averaged {mean(act):.2f} goals against {mean(pr):.2f} projected. "
                      f"League scoring may have shifted. The board's league average updates as games are played, so this often fixes itself; "
                      f"if it persists two weeks, the projection shrink in model.json should be refit.")
    # -- daily table, last 14 days
    days = sorted({r["date"] for r in rows})[-14:]
    st["days"] = []
    for d in reversed(days):
        dr = [r for r in rows if r["date"] == d]; dg = [r for r in dr if r["actual"] != ""]
        red = [r for r in dg if float(r["score"]) < 35]
        st["days"].append(dict(date=d, rows=len(dr), graded=len(dg),
                               proj=mean([float(r["proj"]) for r in dr]), goals=mean([int(r["actual"]) for r in dg]),
                               red_n=len(red), red_under=sum(1 for r in red if int(r["actual"]) <= 2),
                               default_goalies=sum(1 for r in dr if r["goalie_status"] == "Default")))

    # -- health checks on today's data
    dj = ROOT / "data.json"
    if dj.exists():
        D = json.loads(dj.read_text(encoding="utf-8"))
        games_today = len(D.get("schedule", {}).get(today, []))
        st["games_today"] = games_today
        if games_today and D.get("today") != today and now.hour >= 12:
            alert("stale-board", "action", "The board didn't refresh today",
                  f"data.json is from {D.get('today')} but there are {games_today} games today. Check the latest run under the Actions tab.")
        todays = [r for r in rows if r["date"] == today]
        if games_today and not todays and now.hour >= 15:
            alert("log-missing", "action", "Today's games weren't logged",
                  "The board built but no pregame rows were saved, so today can't be graded. Check the 'Build board' step log.")
        if todays and now.hour >= 17:
            share = sum(1 for r in todays if r["goalie_status"] == "Default") / len(todays)
            if share > 0.5:
                alert("starters-missing", "watch", "Starting goalies aren't coming through",
                      f"{share:.0%} of today's rows have no starter report, so the board is guessing the most-used goalie. "
                      f"Daily Faceoff may have changed its page. Until it's fixed, the opposing-goalie part of those scores may be using the wrong goalie.")
    stale = [r for r in rows if r["actual"] == "" and r["date"] <= (date.today() - timedelta(days=2)).isoformat()]
    if stale:
        alert("grading-stuck", "action", f"{len(stale)} finished games haven't been graded",
              f"Rows from {min(r['date'] for r in stale)} on still have no final score. The NHL score API may have changed; check the 'Grade' step.")

    st["alerts"] = alerts
    old = HERE / "status.json"
    if old.exists():                        # keep the old timestamp if nothing else changed
        o = json.loads(old.read_text(encoding="utf-8"))
        if {k: v for k, v in o.items() if k != "built"} == {k: v for k, v in json.loads(json.dumps(st, default=str)).items() if k != "built"}:
            st["built"] = o["built"]
    (HERE / "status.json").write_text(json.dumps(st, indent=1, default=str), encoding="utf-8")
    tpl = (HERE / "monitor_template.html").read_text(encoding="utf-8")
    learn = ROOT / "backtest" / "learn_status.json"
    st["learn"] = json.loads(learn.read_text(encoding="utf-8")) if learn.exists() else None
    st["history"] = MODEL.get("history", [])
    (ROOT / "monitor.html").write_text(tpl.replace("/*__DATA__*/null", json.dumps(st, default=str)), encoding="utf-8")
    print(f"{season}: {len(rows)} logged, {len(g)} graded, auc={st.get('auc')}, alerts={[a['key'] for a in alerts]}")


if __name__ == "__main__":
    main()
