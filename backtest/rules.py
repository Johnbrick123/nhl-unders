"""Score-threshold rules: what happened when the board's score was very low or very high.

Rebuilds every team-game's score exactly the way the live board computes it (each input
scaled 0-100 between the best and worst team in the league as of that day), so a "30"
here means the same thing as a 30 on the board. Then, for team scores and combined game
scores (the two teams added, 0-200), reports average goals and hit rates on the usual
lines, with the shootout winner counted the way books settle totals.

Run after backtest.py:  python backtest/rules.py   -> adds "rules" to results.json
"""
import json, math
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
MODEL = json.loads((HERE.parent / "model.json").read_text(encoding="utf-8"))
TEST_SEASON = None  # set in main(): last complete season


def board_scores(f):
    """Live-board scaling: min-max across all 32 teams' current values on each game date."""
    f = f.sort_values("date").copy()
    out = []
    for season, fs in f.groupby("season"):
        teams = fs[["date", "team", "offense", "pre_xga"]].sort_values("date")
        dates = pd.DataFrame({"date": sorted(fs.date.unique())})
        snaps = []
        for t, tg in teams.groupby("team"):
            # a team's value on day d = the pregame value of its next game on/after d,
            # which only uses games before d (no look-ahead)
            m = pd.merge_asof(dates, tg.drop(columns="team"), on="date", direction="forward")
            m["team"] = t
            snaps.append(m)
        snap = pd.concat(snaps)
        rng = snap.groupby("date").agg(off_lo=("offense", "min"), off_hi=("offense", "max"),
                                       xga_lo=("pre_xga", "min"), xga_hi=("pre_xga", "max")).reset_index()
        fs = fs.merge(rng, on="date", how="left")
        gq = fs.loc[fs.cxga >= 40, "goalie_mult"]
        g_lo, g_hi = (np.percentile(gq, 1), np.percentile(gq, 99)) if len(gq) > 50 else (0.9, 1.1)
        sc = lambda v, lo, hi: np.clip((v - lo) / (hi - lo), 0, 1) * 100
        fs["b_off"] = sc(fs.offense, fs.off_lo, fs.off_hi)
        fs["b_def"] = sc(fs.opp_xga, fs.xga_lo, fs.xga_hi)
        fs["b_goalie"] = sc(fs.goalie_mult, g_lo, g_hi)
        out.append(fs)
    f = pd.concat(out)
    W = MODEL["weights"]
    f["board_score"] = (W["off"] * f.b_off + W["def"] * f.b_def + W["goalie"] * f.b_goalie) / (W["off"] + W["def"] + W["goalie"])
    return f


def wilson_lo(p, n, z=1.96):
    if n == 0:
        return None
    d = 1 + z * z / n
    return (p + z * z / (2 * n) - z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / d


def team_rows(f):
    t = f[["gameId", "season", "board_score", "gf", "ga"]].copy()
    t["tie"] = (t.gf == t.ga).astype(float)               # MoneyPuck excludes shootout goals: tie = shootout
    return t


def game_rows(f):
    g = f.groupby("gameId").agg(season=("season", "first"), combined=("board_score", "sum"),
                                goals=("gf", "sum"), n=("gf", "size"), tie=("tie_flag", "max")).reset_index()
    g = g[g.n == 2].copy()
    g["book"] = g.goals + g.tie                            # shootout winner counts as one goal
    return g


def summarize(d, value_col, lines, rule, label, kind):
    out = dict(label=label, kind=kind, rule=rule, n=int(len(d)))
    if not len(d):
        return out
    if kind == "team":
        out["avg_goals"] = round(float((d.gf + 0.5 * d.tie).mean()), 2)
        hits = {}
        for L in lines:
            # book goals = gf, plus 1 if the team won the shootout (50/50 -> expected half)
            under = ((d.gf <= L) & (d.tie == 0)) | ((d.gf <= L - 1) & (d.tie == 1))
            half = (d.gf == L) & (d.tie == 1)
            p_under = (under.sum() + 0.5 * half.sum()) / len(d)
            hits[f"u{L}.5"] = round(float(p_under), 4)
    else:
        out["avg_goals"] = round(float(d.book.mean()), 2)
        hits = {f"u{L}.5": round(float((d.book <= L).mean()), 4) for L in lines}
    out["hits"] = hits
    return out


def main():
    f = pd.read_csv(HERE / "team_games_scored.csv", parse_dates=["date"])
    seasons = sorted(f.season.unique())
    current = max(seasons)
    complete = [s for s in seasons if s < current] if (f.season == current).sum() < 2000 else seasons
    f = f[f.season.isin(complete)]
    test = max(complete)
    f = board_scores(f)
    f["tie_flag"] = (f.gf == f.ga).astype(int)
    T = team_rows(f)
    G = game_rows(f)

    team_lines, game_lines = [2, 3], [5, 6]
    rules = []
    specs = [
        ("All team totals", "team", lambda d: d.board_score >= -1, "all"),
        ("Team score under 20", "team", lambda d: d.board_score < 20, "under"),
        ("Team score under 25", "team", lambda d: d.board_score < 25, "under"),
        ("Team score under 30", "team", lambda d: d.board_score < 30, "under"),
        ("Team score 70 or higher", "team", lambda d: d.board_score >= 70, "over"),
        ("Team score 75 or higher", "team", lambda d: d.board_score >= 75, "over"),
        ("Team score 80 or higher", "team", lambda d: d.board_score >= 80, "over"),
        ("All games", "game", lambda d: d.combined >= -1, "all"),
        ("Combined under 50", "game", lambda d: d.combined < 50, "under"),
        ("Combined under 60", "game", lambda d: d.combined < 60, "under"),
        ("Combined under 70", "game", lambda d: d.combined < 70, "under"),
        ("Combined under 80", "game", lambda d: d.combined < 80, "under"),
        ("Combined 100 or higher", "game", lambda d: d.combined >= 100, "over"),
        ("Combined 110 or higher", "game", lambda d: d.combined >= 110, "over"),
        ("Combined 120 or higher", "game", lambda d: d.combined >= 120, "over"),
    ]
    for label, kind, mask, side in specs:
        D = T if kind == "team" else G
        lines = team_lines if kind == "team" else game_lines
        r = summarize(D[mask(D)], None, lines, side, label, kind)
        r["side"] = side
        r["share"] = round(len(D[mask(D)]) / len(D), 4)
        tr = summarize(D[mask(D) & (D.season < test)], None, lines, side, label, kind)
        te = summarize(D[mask(D) & (D.season == test)], None, lines, side, label, kind)
        r["train"], r["test"] = tr.get("hits"), te.get("hits")
        r["test_n"] = te["n"]
        # 95% lower bound on the hit rate of the side you'd bet (under for low scores, over for high)
        r["floor"] = {k: round(wilson_lo(v if side != "over" else 1 - v, r["n"]), 4)
                      for k, v in r.get("hits", {}).items()} if r["n"] else {}
        rules.append(r)

    res_path = HERE / "results.json"
    res = json.loads(res_path.read_text(encoding="utf-8"))
    res["rules"] = dict(seasons=[f"{s}-{str(s + 1)[2:]}" for s in complete], test=f"{test}-{str(test + 1)[2:]}",
                        team_n=int(len(T)), game_n=int(len(G)), rows=rules,
                        dist=dict(team=[round(float(x), 1) for x in np.percentile(T.board_score, [10, 25, 50, 75, 90])],
                                  game=[round(float(x), 1) for x in np.percentile(G.combined, [10, 25, 50, 75, 90])]))
    res_path.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    pct = lambda v: "—" if v is None else f"{v * 100:.1f}%"
    print(f"team-games {len(T)}, games {len(G)}; team score pctiles {res['rules']['dist']['team']}, combined {res['rules']['dist']['game']}")
    for r in rules:
        h = r.get("hits", {})
        lines = "  ".join(f"{k.upper()} {pct(v)} (O {pct(1 - v)})" for k, v in h.items())
        tt = "  ".join(f"{k} {pct(v)}" for k, v in (r.get("test") or {}).items())
        print(f"{r['label']:<26} n={r['n']:>5} ({r['share'] * 100:4.1f}%)  avg {r.get('avg_goals')}  {lines}  | {res['rules']['test']}: n={r['test_n']} {tt}")


if __name__ == "__main__":
    main()
