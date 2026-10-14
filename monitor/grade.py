"""Fill in final goals for every logged row whose game has finished.

Goals come from the NHL score API and include overtime, empty-net and the shootout
winner (last_period = SO), which is how most books settle team totals.
Run:  python monitor/grade.py
"""
import csv, json, sys, urllib.request
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
FIELDS = None


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    for path in sorted(HERE.glob("log_*.csv")):
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fields, rows = list(reader.fieldnames), list(reader)
        if "opp_starter_id" not in fields:
            fields.append("opp_starter_id")
            for r in rows:
                r["opp_starter_id"] = ""
        todo = sorted({r["date"] for r in rows if r["actual"] == ""})
        graded = 0
        for d in todo:
            try:
                games = {str(g["id"]): g for g in get(f"https://api-web.nhle.com/v1/score/{d}").get("games", [])}
            except Exception as e:
                print(f"{d}: score API failed ({e}); will retry next run")
                continue
            for r in rows:
                if r["date"] != d or r["actual"] != "":
                    continue
                g = games.get(r["gameId"])
                if not g or g.get("gameState") not in ("FINAL", "OFF"):
                    continue
                side = "homeTeam" if r["ha"] == "Home" else "awayTeam"
                r["actual"] = g[side].get("score", "")
                r["last_period"] = (g.get("gameOutcome") or {}).get("lastPeriodType", "")
                r["graded_at"] = stamp
                graded += 1
        # who actually started in goal against each logged team (scores Daily Faceoff's projections)
        need = sorted({r["gameId"] for r in rows if r["actual"] != "" and not r.get("opp_starter_id")})
        for gid in need:
            try:
                bs = get(f"https://api-web.nhle.com/v1/gamecenter/{gid}/boxscore")
            except Exception:
                continue
            started = {}
            for side in ("awayTeam", "homeTeam"):
                for gk in bs.get("playerByGameStats", {}).get(side, {}).get("goalies", []):
                    if gk.get("starter"):
                        started[bs[side]["abbrev"]] = str(gk["playerId"])
            for r in rows:
                if r["gameId"] == gid and r["opp"] in started:
                    r["opp_starter_id"] = started[r["opp"]]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        left = sum(1 for r in rows if r["actual"] == "")
        print(f"{path.name}: graded {graded}, {left} waiting on results")


if __name__ == "__main__":
    sys.exit(main())
