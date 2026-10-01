"""Walk-forward backtest of the NHL Unders Board scoring.

For every regular-season team-game from 2021-22 through 2025-26, rebuild each input
using only games played before that date (plus last season, counted as 20 games,
exactly like the live board), then check which inputs predict that team's goals.

Run:  python backtest/backtest.py    (downloads what it needs into backtest/data/)
Writes results.json and team_games_scored.csv next to this file. The weekly GitHub
Action runs it, then learn.py and report.py.
"""
import json, shutil, urllib.request, zipfile
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data"
FIRST_SEASON = 2021
_t = date.today()
CURRENT = _t.year if _t.month >= 9 else _t.year - 1     # MoneyPuck start year of the season in progress
SEASONS = list(range(FIRST_SEASON, CURRENT + 1))         # trimmed to seasons with games in load_team_games()
MODEL = json.loads((HERE.parent / "model.json").read_text(encoding="utf-8"))
PRIOR_GAMES = MODEL.get("priorGames", 20)
BOARD_TO_BT = {"goalie": "goalie", "def": "defense", "off": "offense", "st": "pp_pk", "l5": "l5", "fin": "luck"}
DEFAULT_W = {BOARD_TO_BT[k]: v for k, v in MODEL["weights"].items()}
UA = {"User-Agent": "Mozilla/5.0"}


def download():
    """all_teams.csv and the current season's shots are refreshed every run; past seasons are cached."""
    DATA.mkdir(exist_ok=True)
    jobs = [("https://moneypuck.com/moneypuck/playerData/careers/gameByGame/all_teams.csv", DATA / "all_teams.csv", True)]
    for y in range(FIRST_SEASON - 1, CURRENT + 1):
        jobs.append((f"https://peter-tanner.com/moneypuck/downloads/shots_{y}.zip", DATA / f"shots_{y}.zip", y == CURRENT))
    for url, path, fresh in jobs:
        if path.exists() and not fresh:
            continue
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=600) as r, open(path, "wb") as o:
                shutil.copyfileobj(r, o)
            print(f"downloaded {path.name} ({path.stat().st_size // 1_000_000} MB)")
        except Exception as e:
            if not path.exists():
                raise
            print(f"kept cached {path.name}: {e}")
RELOCATED = {"UTA": "ARI"}                         # Utah's first-season prior = Arizona


# ---------------------------------------------------------------- load
def load_team_games():
    df = pd.read_csv(DATA / "all_teams.csv", low_memory=False)
    df = df[df["season"].between(SEASONS[0] - 1, SEASONS[-1])]
    df = df[df["gameId"].astype(str).str[4:6] == "02"]           # regular season only
    keep = ["team", "season", "gameId", "opposingTeam", "home_or_away", "gameDate"]
    allsit = df[df.situation == "all"][keep + ["goalsFor", "goalsAgainst", "xGoalsFor", "xGoalsAgainst",
                                               "shotsOnGoalFor", "shotsOnGoalAgainst"]]
    pp = df[df.situation == "5on4"][["team", "gameId", "goalsFor"]].rename(columns={"goalsFor": "ppgf"})
    pk = df[df.situation == "4on5"][["team", "gameId", "goalsAgainst"]].rename(columns={"goalsAgainst": "pkga"})
    g = allsit.merge(pp, on=["team", "gameId"], how="left").merge(pk, on=["team", "gameId"], how="left")
    g = g.rename(columns={"goalsFor": "gf", "goalsAgainst": "ga", "xGoalsFor": "xgf", "xGoalsAgainst": "xga",
                          "shotsOnGoalFor": "sf", "shotsOnGoalAgainst": "sa", "opposingTeam": "opp"})
    g[["ppgf", "pkga"]] = g[["ppgf", "pkga"]].fillna(0)
    g["date"] = pd.to_datetime(g["gameDate"].astype(str), format="%Y%m%d")
    g["home"] = (g["home_or_away"] == "HOME").astype(int)
    SEASONS[:] = [y for y in SEASONS if (g.season == y).sum() > 0]
    return g.sort_values(["date", "gameId", "team"]).reset_index(drop=True)


def load_goalie_games():
    """Per goalie per game: xGA and GA faced (unblocked, non-empty-net), plus each team's starter."""
    rows, starters = [], []
    for y in range(SEASONS[0] - 1, SEASONS[-1] + 1):
        if not (DATA / f"shots_{y}.zip").exists():
            continue
        with zipfile.ZipFile(DATA / f"shots_{y}.zip") as z:
            name = [n for n in z.namelist() if n.endswith(".csv")][0]
            s = pd.read_csv(z.open(name), usecols=["season", "game_id", "isPlayoffGame", "period", "time",
                                                   "isHomeTeam", "homeTeamCode", "awayTeamCode", "goal", "xGoal",
                                                   "goalieIdForShot", "shotOnEmptyNet"], low_memory=False)
        s = s[(s.isPlayoffGame == 0) & (s.shotOnEmptyNet == 0) & (s.goalieIdForShot > 0)]
        s["gameId"] = s["season"] * 1_000_000 + s["game_id"]
        s["defTeam"] = np.where(s.isHomeTeam == 1, s.awayTeamCode, s.homeTeamCode)
        agg = s.groupby(["gameId", "defTeam", "goalieIdForShot"]).agg(xga=("xGoal", "sum"), ga=("goal", "sum")).reset_index()
        rows.append(agg)
        first = s.sort_values(["period", "time"]).groupby(["gameId", "defTeam"]).head(1)
        starters.append(first[["gameId", "defTeam", "goalieIdForShot"]])
    gg = pd.concat(rows).rename(columns={"goalieIdForShot": "goalie"})
    st = pd.concat(starters).rename(columns={"goalieIdForShot": "starter", "defTeam": "team"})
    return gg, st


# ---------------------------------------------------------------- walk-forward features
def build_features(g, gg, st):
    g = g.merge(st, on=["gameId", "team"], how="left")
    gdate = g[["gameId", "date", "season"]].drop_duplicates("gameId")
    gg = gg.merge(gdate, on="gameId", how="left").dropna(subset=["date"])

    stat_cols = ["gf", "ga", "xgf", "xga", "sf", "sa", "ppgf", "pkga"]
    season_tot = g.groupby(["team", "season"])[stat_cols].sum()
    season_gp = g.groupby(["team", "season"]).size()
    league = {y: (g[g.season == y][stat_cols].sum() / len(g[g.season == y])).to_dict()
              for y in g.season.unique()}

    out = []
    for (team, season), tg in g[g.season.isin(SEASONS)].groupby(["team", "season"]):
        tg = tg.sort_values("date")
        pteam = team if (team, season - 1) in season_gp.index else RELOCATED.get(team, team)
        if (pteam, season - 1) in season_gp.index:
            prior = season_tot.loc[(pteam, season - 1)] / season_gp.loc[(pteam, season - 1)]
        else:
            prior = pd.Series(league[season - 1])                  # expansion team: league average
        cum = tg[stat_cols].cumsum().shift(1).fillna(0)
        n = np.arange(len(tg))
        for c in stat_cols:
            tg["pre_" + c] = (prior[c] * PRIOR_GAMES + cum[c].values) / (PRIOR_GAMES + n)
        gfs = tg["gf"].shift(1)
        tg["pre_l5"] = gfs.rolling(5, min_periods=3).mean()
        tg["pre_l10"] = gfs.rolling(10, min_periods=5).mean()
        prev = tg["date"].shift(1)
        tg["b2b"] = ((tg["date"] - prev).dt.days == 1).astype(int)
        tg["n_cur"] = n
        out.append(tg)
    f = pd.concat(out)

    # goalie pregame GSAx: previous season + current season before this date
    gg = gg.sort_values("date")
    gk = []
    for gid, gx in gg.groupby("goalie"):
        gx = gx.sort_values("date").copy()
        gx["cxga"] = gx["xga"].cumsum().shift(1).fillna(0)
        gx["cga"] = gx["ga"].cumsum().shift(1).fillna(0)
        # restrict history to previous + current season (subtract anything older)
        for y in gx.season.unique():
            old = gx[gx.season < y - 1]
            m = gx.season == y
            gx.loc[m, "cxga"] -= old["xga"].sum()
            gx.loc[m, "cga"] -= old["ga"].sum()
        gk.append(gx[["gameId", "goalie", "cxga", "cga"]])
    gk = pd.concat(gk)

    # attach opponent context
    opp = f[["gameId", "team", "pre_xga", "pre_ga", "pre_sa", "pre_pkga", "b2b", "starter"]].rename(
        columns={"team": "opp", "pre_xga": "opp_xga", "pre_ga": "opp_ga", "pre_sa": "opp_sa",
                 "pre_pkga": "opp_pkga", "b2b": "opp_b2b", "starter": "opp_starter"})
    f = f.merge(opp, on=["gameId", "opp"], how="inner")
    f = f.merge(gk.rename(columns={"goalie": "opp_starter"}), on=["gameId", "opp_starter"], how="left")
    f[["cxga", "cga"]] = f[["cxga", "cga"]].fillna(0)
    return f, league


def add_model(f, league):
    lg = pd.DataFrame({y: league[y - 1] for y in SEASONS}).T    # last season's league averages (known pregame)
    L = lg.loc[f.season].reset_index(drop=True)
    f = f.reset_index(drop=True)
    rate = np.where(f.cxga > 0, (f.cxga - f.cga) / f.cxga.where(f.cxga > 0, 1), 0)
    w = f.cxga / (f.cxga + 60)
    f["goalie_mult"] = np.clip(1 - w * rate, 0.85, 1.15)
    f["opp_gsax"] = f.cxga - f.cga
    f["opp_svx"] = np.where(f.cxga > 0, f.cga / f.cxga.where(f.cxga > 0, 1), 1)   # goals allowed per expected
    f["offense"] = 0.65 * f.pre_xgf / L.xgf + 0.35 * f.pre_gf / L.gf
    f["defense"] = f.opp_xga / L.xgf
    f["pp_pk"] = (f.pre_ppgf / L.ppgf) * (f.opp_pkga / L.ppgf)
    f["luck"] = f.pre_gf / f.pre_xgf
    f["sh_pct"] = f.pre_gf / f.pre_sf
    venue = np.where(f.home == 1, 1.025, 0.975)
    f["proj"] = L.gf * f.offense * f.defense * f.goalie_mult * venue * np.where(f.b2b == 1, .96, 1) * np.where(f.opp_b2b == 1, 1.04, 1)
    f["over"] = (f.gf >= 3).astype(int)
    return f


# ---------------------------------------------------------------- scoring like the board
SUB = dict(goalie=("goalie_mult", 1), defense=("defense", 1), offense=("offense", 1),
           pp_pk=("pp_pk", 1), l5=("pre_l5", 1), luck=("luck", 1))


def subscores(f):
    for k, (col, sign) in SUB.items():
        out = pd.Series(np.nan, index=f.index)
        for y, idx in f.groupby("season").groups.items():
            v = f.loc[idx, col]
            lo, hi = np.nanpercentile(v, 5), np.nanpercentile(v, 95)
            s = np.clip((v - lo) / (hi - lo), 0, 1) * 100
            out.loc[idx] = s if sign > 0 else 100 - s
        f["s_" + k] = out
    return f


def board_score(f, W):
    num = sum(f["s_" + k].fillna(0) * w for k, w in W.items())
    den = sum(f["s_" + k].notna() * w for k, w in W.items())
    return num / den.replace(0, np.nan)


# ---------------------------------------------------------------- stats helpers
def auc(score, y):
    m = score.notna()
    s, y = score[m].values, y[m].values
    r = pd.Series(s).rank().values
    pos = y == 1
    return (r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())


def spearman(a, b):
    m = a.notna() & b.notna()
    return float(np.corrcoef(a[m].rank(), b[m].rank())[0, 1])


def quintiles(score, f):
    m = score.notna()
    q = pd.qcut(score[m].rank(method="first"), 5, labels=False)
    d = f[m].assign(q=q.values)
    t = d.groupby("q").agg(goals=("gf", "mean"), over=("over", "mean"), n=("gf", "size"))
    return [dict(q=int(i) + 1, goals=round(r.goals, 3), over=round(r.over, 4), n=int(r.n)) for i, r in t.iterrows()]


def deciles(score, f):
    m = score.notna()
    q = pd.qcut(score[m].rank(method="first"), 10, labels=False)
    d = f[m].assign(q=q.values)
    t = d.groupby("q").agg(goals=("gf", "mean"), over=("over", "mean"), lo=("_s", "min"), hi=("_s", "max")) if "_s" in d else None
    return t


def logistic(X, y, l2=1e-3, iters=50):
    X = np.column_stack([np.ones(len(X)), X])
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        g = X.T @ (y - p) - l2 * np.r_[0, b[1:]]
        H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.diag(np.r_[0, np.ones(len(b) - 1)])
        b += np.linalg.solve(H, g)
    return b


def logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# ---------------------------------------------------------------- main
def main():
    download()
    g = load_team_games()
    gg, st = load_goalie_games()
    f, league = build_features(g, gg, st)
    f = add_model(f, league)
    f = subscores(f)
    f["score_default"] = board_score(f, DEFAULT_W)

    # your original sheet: team-only GF/G, shots/G, Sh%, PP — averaged percentiles, no opponent
    sheet = sum(f.groupby("season")[c].rank(pct=True) for c in ["pre_gf", "pre_sf", "sh_pct", "pre_ppgf"]) / 4
    f["score_sheet"] = sheet * 100
    season_label = lambda y: f"{y}-{str(y + 1)[2:]}" + (" (to date)" if y == CURRENT else "")
    res = {"n": int(len(f)), "seasons": [season_label(y) for y in SEASONS], "current": season_label(CURRENT),
           "current_n": int((f.season == CURRENT).sum()), "data_through": str(f.date.max().date()),
           "base_over": round(float(f.over.mean()), 4), "base_goals": round(float(f.gf.mean()), 3)}

    # 1. single inputs
    inputs = [
        ("Team GF/G", "pre_gf", 1, "team"), ("Team xGF/G", "pre_xgf", 1, "team"), ("Team shots/G", "pre_sf", 1, "team"),
        ("Team Sh%", "sh_pct", 1, "team"), ("Team PP goals/G", "pre_ppgf", 1, "team"), ("Goals L5", "pre_l5", 1, "team"),
        ("Goals L10", "pre_l10", 1, "team"), ("Luck (GF ÷ xGF)", "luck", 1, "team"),
        ("Offense rating", "offense", 1, "team"),
        ("Opp xGA/G", "opp_xga", 1, "opp"), ("Opp GA/G", "opp_ga", 1, "opp"), ("Opp shots against/G", "opp_sa", 1, "opp"),
        ("Opp PK goals allowed/G", "opp_pkga", 1, "opp"), ("Opp goalie GSAx", "opp_gsax", -1, "opp"),
        ("Opp goalie factor (shrunk)", "goalie_mult", 1, "opp"), ("PP vs PK matchup", "pp_pk", 1, "matchup"),
        ("Home ice", "home", 1, "spot"), ("Team on back-to-back", "b2b", -1, "spot"), ("Opp on back-to-back", "opp_b2b", 1, "spot"),
    ]
    rows = []
    for label, col, sign, grp in inputs:
        v = f[col] * sign
        r = dict(label=label, group=grp, rho=round(spearman(v, f.gf), 4), auc=round(auc(v, f.over), 4))
        if f[col].nunique() <= 2:
            a, b = f[f[col] * sign == f[col].mul(sign).max()], f[f[col] * sign != f[col].mul(sign).max()]
            r.update(binary=True, yes_goals=round(a.gf.mean(), 3), no_goals=round(b.gf.mean(), 3),
                     yes_over=round(a.over.mean(), 4), no_over=round(b.over.mean(), 4), yes_n=int(len(a)))
        else:
            qs = quintiles(v, f)
            r.update(binary=False, low_over=qs[0]["over"], high_over=qs[-1]["over"],
                     low_goals=qs[0]["goals"], high_goals=qs[-1]["goals"])
        rows.append(r)
    res["inputs"] = rows

    # 2. composite scores
    comps = [("Board score (current weights)", "score_default"), ("Projected goals", "proj"), ("Your original sheet", "score_sheet")]
    res["composites"] = []
    for label, col in comps:
        res["composites"].append(dict(label=label, auc=round(auc(f[col], f.over), 4), rho=round(spearman(f[col], f.gf), 4),
                                      quintiles=quintiles(f[col], f)))
    # deciles of the board score
    q = pd.qcut(f.score_default.rank(method="first"), 10, labels=False)
    d = f.assign(q=q).groupby("q").agg(goals=("gf", "mean"), over=("over", "mean"), lo=("score_default", "min"), hi=("score_default", "max"))
    res["deciles"] = [dict(d=int(i) + 1, goals=round(r.goals, 3), over=round(r.over, 4), lo=round(r.lo, 1), hi=round(r.hi, 1)) for i, r in d.iterrows()]
    # score bands as shown on the board (0-100)
    bands = pd.cut(f.score_default, [-1, 30, 40, 50, 60, 70, 101], labels=["<30", "30-40", "40-50", "50-60", "60-70", "70+"])
    b = f.groupby(bands, observed=False).agg(goals=("gf", "mean"), over=("over", "mean"), n=("gf", "size"))
    res["bands"] = [dict(band=str(i), goals=round(r.goals, 3), over=round(r.over, 4), n=int(r.n)) for i, r in b.iterrows()]

    # 3. calibration of projected goals (Poisson chance of 3+)
    from math import exp, factorial
    p3 = 1 - f.proj.apply(lambda l: sum(exp(-l) * l ** k / factorial(k) for k in range(3)))
    f["p_over"] = p3
    cb = pd.cut(p3, [0, .45, .5, .55, .6, .65, .7, 1])
    c = f.groupby(cb, observed=False).agg(pred=("p_over", "mean"), actual=("over", "mean"), n=("gf", "size"),
                                          proj=("proj", "mean"), goals=("gf", "mean"))
    res["calibration"] = [dict(pred=round(r.pred, 4), actual=round(r.actual, 4), n=int(r.n), proj=round(r.proj, 3), goals=round(r.goals, 3))
                          for _, r in c.iterrows() if r.n > 50]
    res["proj_mean"] = round(float(f.proj.mean()), 3)

    # 4. fitted weights: train 2021-22..2024-25, test 2025-26
    keys = list(DEFAULT_W)
    holdout = max(y for y in SEASONS if y < CURRENT)      # last complete season
    train, test = f[f.season < holdout], f[f.season == holdout]

    def fit(df):
        X = df[["s_" + k for k in keys]].copy()
        X = X.fillna(X.mean()) / 100
        b = logistic(X.values, df.over.values)
        coef = dict(zip(keys, b[1:]))
        pos = {k: max(v, 0) for k, v in coef.items()}
        tot = sum(pos.values()) or 1
        return coef, {k: v / tot for k, v in pos.items()}

    coef, fitted = fit(train)
    res["coef"] = {k: round(v, 3) for k, v in coef.items()}
    res["fitted_w"] = {k: round(v, 3) for k, v in fitted.items()}
    res["per_season_w"] = {season_label(y): {k: round(v, 3) for k, v in fit(f[f.season == y])[1].items()}
                           for y in SEASONS if (f.season == y).sum() >= 1000}
    rounded = {k: round(v * 20) / 20 for k, v in fitted.items()}
    res["suggested_w"] = rounded
    test = test.copy()
    test["score_fitted"] = board_score(test, rounded)
    res["test"] = dict(season=season_label(holdout), n=int(len(test)),
                       default_auc=round(auc(test.score_default, test.over), 4),
                       fitted_auc=round(auc(test.score_fitted, test.over), 4),
                       sheet_auc=round(auc(test.score_sheet, test.over), 4),
                       proj_auc=round(auc(test.proj, test.over), 4),
                       default_q=quintiles(test.score_default, test), fitted_q=quintiles(test.score_fitted, test),
                       sheet_q=quintiles(test.score_sheet, test))
    # per-season AUC stability for the board vs sheet
    res["by_season"] = [dict(season=season_label(y), n=int(len(s)), board=round(auc(s.score_default, s.over), 4),
                             sheet=round(auc(s.score_sheet, s.over), 4), proj=round(auc(s.proj, s.over), 4),
                             over=round(s.over.mean(), 4))
                        for y, s in f.groupby("season") if len(s) >= 50]
    (HERE / "results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    f.to_csv(HERE / "team_games_scored.csv", index=False)
    print(json.dumps({k: res[k] for k in ["n", "base_over", "base_goals", "proj_mean", "fitted_w", "test", "by_season"]}, indent=1, default=float))
    for r in rows:
        print(f'{r["label"]:<28} rho {r["rho"]:+.3f}  auc {r["auc"]:.3f}')


if __name__ == "__main__":
    main()
