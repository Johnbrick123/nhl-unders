"""Build the NHL Unders Board and log today's pregame rows.

Pulls team, goalie, expected-goals, schedule and projected-starter data, blends last
season with the current one, and writes:
  index.html                 the board (template.html + engine.js + data)
  data.json                  the same data, for the monitor
  monitor/log_<season>.csv   today's rows, frozen once each game starts

Run:  python build.py [YYYY-MM-DD]   (the GitHub Action runs it hourly on game days)
"""
import csv, io, json, re, subprocess, sys, urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

HERE = Path(__file__).parent
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/128 Safari/537.36"}
MODEL = json.loads((HERE / "model.json").read_text(encoding="utf-8"))
PRIOR_GAMES = MODEL.get("priorGames", 20)  # last season counts like this many current-season games


def get(url, as_json=True, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                body = r.read().decode("utf-8")
            return json.loads(body) if as_json else body
        except Exception:
            if i == tries - 1:
                raise


def season_ids(today):
    start = today.year if today.month >= 9 else today.year - 1
    return start - 1, start  # (prior, current) MoneyPuck-style start years


def nhl_stats(report, start_year, kind="team"):
    sid = f"{start_year}{start_year + 1}"
    url = (f"https://api.nhle.com/stats/rest/en/{kind}/{report}"
           f"?cayenneExp=seasonId={sid}%20and%20gameTypeId=2&limit=-1")
    try:
        return get(url)["data"]
    except Exception:
        return []


def moneypuck(kind, start_year):
    url = f"https://moneypuck.com/moneypuck/playerData/seasonSummary/{start_year}/regular/{kind}.csv"
    try:
        return list(csv.DictReader(io.StringIO(get(url, as_json=False))))
    except Exception:
        return []


def team_ids():
    rows = get("https://api.nhle.com/stats/rest/en/team")["data"]
    return {r["id"]: r["triCode"] for r in rows}


def season_block(start_year, ids):
    """Per-team season totals keyed by tri-code."""
    out = {}
    for r in nhl_stats("summary", start_year):
        ab = ids.get(r["teamId"])
        gp = r["gamesPlayed"] or 0
        out[ab] = dict(name=r["teamFullName"], gp=gp, gf=r["goalsFor"], ga=r["goalsAgainst"],
                       sf=r["shotsForPerGame"] * gp, sa=r["shotsAgainstPerGame"] * gp,
                       ppPct=r["powerPlayPct"] or 0, pkPct=r["penaltyKillPct"] or 0)
    for r in nhl_stats("powerplay", start_year):
        ab = ids.get(r["teamId"])
        if ab in out:
            out[ab]["ppo"] = r.get("ppOpportunities") or 0
            out[ab]["ppg"] = r.get("powerPlayGoalsFor") or 0
    for r in nhl_stats("penaltykill", start_year):
        ab = ids.get(r["teamId"])
        if ab in out:
            out[ab]["tsh"] = r.get("timesShorthanded") or 0
            out[ab]["ppga"] = r.get("ppGoalsAgainst") or 0
    for r in moneypuck("teams", start_year):
        ab = r["team"]
        if ab not in out:
            continue
        f = lambda k: float(r[k] or 0)
        if r["situation"] == "all":
            out[ab].update(xgf=f("xGoalsFor"), xga=f("xGoalsAgainst"),
                           hdf=f("highDangerShotsFor"), hda=f("highDangerShotsAgainst"))
        elif r["situation"] == "5on5":
            out[ab].update(xgf5=f("xGoalsFor"), xga5=f("xGoalsAgainst"),
                           cf5=f("corsiPercentage"))
    return out


def blend_teams(prior, cur):
    keys = ["gf", "ga", "sf", "sa", "ppo", "ppg", "tsh", "ppga", "xgf", "xga", "xgf5", "xga5", "hdf", "hda"]
    teams = {}
    for ab in sorted(set(prior) | set(cur)):
        p, c = prior.get(ab, {}), cur.get(ab, {})
        pgp, cgp = p.get("gp", 0), c.get("gp", 0)
        wp = PRIOR_GAMES / pgp if pgp else 0  # scale last season down to PRIOR_GAMES worth
        has_mp_cur = "xgf" in c
        t = {"ab": ab, "name": c.get("name") or p.get("name"), "gpCur": cgp, "gpPrior": pgp}
        g = wp * pgp + cgp
        for k in keys:
            pv = p.get(k, 0) * wp
            if k in ("xgf", "xga", "xgf5", "xga5", "hdf", "hda") and not has_mp_cur:
                t[k] = p.get(k, 0) / pgp if pgp else 0
            else:
                t[k] = (pv + c.get(k, 0)) / g if g else 0
        t["ppPct"] = t["ppg"] / t["ppo"] if t["ppo"] else p.get("ppPct", 0)
        t["pkPct"] = 1 - t["ppga"] / t["tsh"] if t["tsh"] else p.get("pkPct", 0)
        t["cf5"] = c.get("cf5") if has_mp_cur and cgp >= 10 else p.get("cf5", 0.5)
        t["cur"] = {"gp": cgp, "gf": c.get("gf", 0), "ga": c.get("ga", 0)}
        t["prior"] = {"gp": pgp, "gfpg": p.get("gf", 0) / pgp if pgp else 0,
                      "gapg": p.get("ga", 0) / pgp if pgp else 0,
                      "sfpg": p.get("sf", 0) / pgp if pgp else 0,
                      "ppPct": p.get("ppPct", 0)}
        teams[ab] = t
    return teams


def goalie_rows(start_year):
    out = {}
    for r in moneypuck("goalies", start_year):
        if r["situation"] != "all":
            continue
        out[r["playerId"]] = dict(name=r["name"], team=r["team"], gp=int(float(r["games_played"])),
                                  xga=float(r["xGoals"]), ga=float(r["goals"]), sog=float(r["ongoal"]))
    return out


def rosters(abbrevs):
    out = {}
    for ab in abbrevs:
        try:
            d = get(f"https://api-web.nhle.com/v1/roster/{ab}/current")
        except Exception:
            continue
        for g in d.get("goalies", []):
            out[str(g["id"])] = dict(team=ab, name=f'{g["firstName"]["default"]} {g["lastName"]["default"]}',
                                     last=g["lastName"]["default"])
    return out


def build_goalies(prior_y, cur_y, abbrevs):
    pr, cu, ro = goalie_rows(prior_y), goalie_rows(cur_y), rosters(abbrevs)
    goalies = []
    for pid, info in ro.items():
        p, c = pr.get(pid, {}), cu.get(pid, {})
        g = {"id": pid, "name": info["name"], "last": info["last"], "team": info["team"],
             "gpPrior": p.get("gp", 0), "gpCur": c.get("gp", 0)}
        for k in ("xga", "ga", "sog"):
            g[k] = p.get(k, 0) + c.get(k, 0)
        g["svPct"] = 1 - g["ga"] / g["sog"] if g["sog"] else None
        g["gsax"] = g["xga"] - g["ga"]
        goalies.append(g)
    return goalies


def schedule(today):
    days = {}
    start = today - timedelta(days=1)
    for anchor in (start, start + timedelta(days=7)):
        try:
            d = get(f"https://api-web.nhle.com/v1/schedule/{anchor.isoformat()}")
        except Exception:
            continue
        for wk in d.get("gameWeek", []):
            days[wk["date"]] = [dict(id=g["id"], type=g["gameType"], start=g["startTimeUTC"],
                                     away=g["awayTeam"]["abbrev"], home=g["homeTeam"]["abbrev"],
                                     awayScore=g["awayTeam"].get("score"), homeScore=g["homeTeam"].get("score"),
                                     state=g.get("gameState"))
                                for g in wk["games"] if g["gameType"] in (2, 3)]
    return days


def recent(abbrevs, n=5):
    """Last n completed regular-season games per team: [gf, ga, opp, date]."""
    out = {}
    for ab in abbrevs:
        try:
            games = get(f"https://api-web.nhle.com/v1/club-schedule-season/{ab}/now")["games"]
        except Exception:
            continue
        done = [g for g in games if g["gameType"] == 2 and g.get("gameState") in ("FINAL", "OFF")]
        rows = []
        for g in done[-n:]:
            home = g["homeTeam"]["abbrev"] == ab
            me, them = (g["homeTeam"], g["awayTeam"]) if home else (g["awayTeam"], g["homeTeam"])
            rows.append([me.get("score"), them.get("score"), them["abbrev"], g["gameDate"]])
        out[ab] = rows
    return out


def starters(dates):
    """Projected/confirmed starters from Daily Faceoff, keyed by date then team full name."""
    out = {}
    for ds in dates:
        try:
            h = get(f"https://www.dailyfaceoff.com/starting-goalies/{ds}", as_json=False)
            m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', h, re.S)
            rows = json.loads(m.group(1))["props"]["pageProps"]["data"] or []
        except Exception:
            continue
        day = {}
        for r in rows:
            for side in ("home", "away"):
                if r.get(f"{side}GoalieName"):
                    day[r[f"{side}TeamName"]] = dict(name=r[f"{side}GoalieName"],
                                                     status=r.get(f"{side}NewsStrengthName") or "Projected")
        out[ds] = day
    return out


def main():
    today = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date.today()
    prior_y, cur_y = season_ids(today)
    ids = team_ids()
    prior, cur = season_block(prior_y, ids), season_block(cur_y, ids)
    teams = blend_teams(prior, cur)
    goalies = build_goalies(prior_y, cur_y, list(teams))
    sched = schedule(today)
    upcoming = [d for d in sorted(sched) if d >= today.isoformat()][:4]
    data = dict(built=datetime.now().strftime("%Y-%m-%d %H:%M"), today=today.isoformat(),
                seasons=dict(prior=f"{prior_y}-{str(prior_y + 1)[2:]}", cur=f"{cur_y}-{str(cur_y + 1)[2:]}"),
                priorGames=PRIOR_GAMES, teams=teams, goalies=goalies, schedule=sched,
                recent=recent(list(teams)),
                starters=starters(upcoming))
    data["model"] = MODEL
    prev = HERE / "data.json"
    if prev.exists():                       # keep the old timestamp if nothing else changed (no empty commits)
        old = json.loads(prev.read_text(encoding="utf-8"))
        if {k: v for k, v in old.items() if k != "built"} == {k: v for k, v in json.loads(json.dumps(data)).items() if k != "built"}:
            data["built"] = old["built"]
    if len(teams) < 32 or not goalies:
        sys.exit(f"Refusing to publish: {len(teams)} teams, {len(goalies)} goalies")
    tpl = (HERE / "template.html").read_text(encoding="utf-8")
    engine = (HERE / "engine.js").read_text(encoding="utf-8")
    payload = json.dumps(data, separators=(",", ":"))
    html = tpl.replace("/*__ENGINE__*/", engine).replace("/*__DATA__*/null", payload)
    head = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"></head><body>')
    (HERE / "index.html").write_text(head + html + "</body></html>", encoding="utf-8")
    (HERE / "data.json").write_text(payload, encoding="utf-8")
    n_games = sum(len(sched.get(d, [])) for d in upcoming)
    print(f"Built index.html: {len(teams)} teams, {len(goalies)} goalies, "
          f"{n_games} games over {upcoming}, starters for {[d for d in data['starters'] if data['starters'][d]]}")
    log_today(data)


LOG_FIELDS = ["date", "gameId", "start", "team", "opp", "ha", "goalie_id", "goalie", "goalie_status",
              "s_goalie", "s_def", "s_off", "s_st", "s_l5", "s_fin", "score", "proj", "p_over",
              "b2b", "opp_b2b", "model_version", "logged_at", "actual", "last_period", "graded_at"]


def log_path(data):
    return HERE / "monitor" / f"log_{data['seasons']['cur']}.csv"


def log_today(data):
    """Upsert today's rows. A game's row keeps updating until puck drop, then never changes."""
    try:
        out = subprocess.run(["node", str(HERE / "monitor" / "snapshot.js"), str(HERE / "data.json")],
                             capture_output=True, text=True, check=True).stdout
    except Exception as e:
        print(f"Snapshot skipped: {e}")
        return
    rows = json.loads(out)
    path = log_path(data)
    existing = []
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
    by_key = {(r["gameId"], r["team"]): r for r in existing}
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    changed = 0
    for r in rows:
        key = (str(r["gameId"]), r["team"])
        if r.pop("state") not in ("FUT", "PRE"):
            continue                                   # started or finished: frozen (or never logged)
        r.update(gameId=str(r["gameId"]), logged_at=stamp, actual="", last_period="", graded_at="")
        by_key[key] = {k: r.get(k, "") for k in LOG_FIELDS}
        changed += 1
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        w.writeheader()
        for r in sorted(by_key.values(), key=lambda r: (r["date"], r["gameId"], r["team"])):
            w.writerow({k: r.get(k, "") for k in LOG_FIELDS})
    print(f"Logged {changed} pregame rows to {path.name} ({len(by_key)} total)")


if __name__ == "__main__":
    main()
