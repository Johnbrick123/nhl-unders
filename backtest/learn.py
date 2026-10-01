"""Weekly learner: refit the score weights on the latest games and update model.json
only when the guardrails pass.

Guardrails (all must hold for a change):
  1. At least MIN_CURRENT team-games played this season (about mid-November).
  2. At least MIN_DAYS since the last change.
  3. Fit on everything before the last VALIDATE_DAYS, then test on those recent games.
     New weights must beat the current ones there by MIN_GAIN in ranking accuracy...
  4. ...and must not do worse on the last complete season by more than MAX_LOSS.
  5. Each input moves at most MAX_STEP per change (weights are rounded to 5%).
Writes learn_status.json (read by the monitor page and alerts.py).
Run after backtest.py:  python backtest/learn.py
"""
import json
from datetime import date, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from backtest import auc, logistic, board_score, CURRENT, BOARD_TO_BT

HERE = Path(__file__).parent
ROOT = HERE.parent
MIN_CURRENT, MIN_DAYS, VALIDATE_DAYS = 500, 28, 42
MIN_GAIN, MAX_LOSS, MAX_STEP = 0.003, 0.002, 0.10
LOOKBACK = 4          # completed seasons used in the fit, plus the current one
CURRENT_WEIGHT = 2.0  # this season's games count double


def weighted_logistic(X, y, w, l2=1e-3, iters=50):
    X = np.column_stack([np.ones(len(X)), X])
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        g = X.T @ (w * (y - p)) - l2 * np.r_[0, b[1:]]
        H = (X * (w * p * (1 - p))[:, None]).T @ X + l2 * np.diag(np.r_[0, np.ones(len(b) - 1)])
        b += np.linalg.solve(H, g)
    return b


def normalize(w):
    w = {k: max(0.0, v) for k, v in w.items()}
    t = sum(w.values()) or 1
    w = {k: round(v / t * 20) / 20 for k, v in w.items()}          # 5% steps
    drift = round(1 - sum(w.values()), 2)
    if drift:
        k = max(w, key=w.get); w[k] = round(w[k] + drift, 2)
    return w


def main():
    model = json.loads((ROOT / "model.json").read_text(encoding="utf-8"))
    f = pd.read_csv(HERE / "team_games_scored.csv", parse_dates=["date"])
    keys = list(BOARD_TO_BT.values())
    to_board = {v: k for k, v in BOARD_TO_BT.items()}
    cur = {BOARD_TO_BT[k]: v for k, v in model["weights"].items()}
    today = date.today()
    through = f.date.max().date()
    n_cur = int((f.season == CURRENT).sum())
    alerts = []
    status = dict(date=today.isoformat(), data_through=through.isoformat(), current_n=n_cur, model_version=model["version"])

    if today.month in (10, 11, 12, 1, 2, 3, 4) and (today - through).days > 4:
        alerts.append(dict(key="moneypuck-stale", level="action", title="Backtest data is more than 4 days old",
                           body=f"MoneyPuck's game file ends {through}. The weekly learner and backtest are working from stale data; "
                                f"check that moneypuck.com still publishes all_teams.csv at the same address."))

    cutoff = pd.Timestamp(through - timedelta(days=VALIDATE_DAYS))
    fit_set = f[(f.season >= CURRENT - LOOKBACK) & (f.date < cutoff)]
    val = f[(f.season == CURRENT) & (f.date >= cutoff)]
    prev = f[f.season == CURRENT - 1]
    X = fit_set[["s_" + k for k in keys]].fillna(50).values / 100
    w = np.where(fit_set.season == CURRENT, CURRENT_WEIGHT, 1.0)
    b = weighted_logistic(X, fit_set.over.values, w)
    target = normalize(dict(zip(keys, b[1:])))
    stepped = normalize({k: cur[k] + float(np.clip(target[k] - cur[k], -MAX_STEP, MAX_STEP)) for k in keys})

    def score_auc(d, W):
        return auc(board_score(d, W), d.over) if len(d) else None
    m = dict(val_n=int(len(val)), val_cur=score_auc(val, cur), val_new=score_auc(val, stepped), val_target=score_auc(val, target),
             prev_cur=score_auc(prev, cur), prev_new=score_auc(prev, stepped))
    status.update(current={to_board[k]: v for k, v in cur.items()}, target={to_board[k]: v for k, v in target.items()},
                  candidate={to_board[k]: v for k, v in stepped.items()},
                  metrics={k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()})

    # home / rest effects, measured on the lookback window (shown; changed by hand if they drift)
    recent = f[f.season >= CURRENT - 2]
    base = recent.gf.mean()
    status["measured"] = dict(home=round(recent[recent.home == 1].gf.mean() / base, 3),
                              away=round(recent[recent.home == 0].gf.mean() / base, 3),
                              b2b=round(recent[recent.b2b == 1].gf.mean() / recent[recent.b2b == 0].gf.mean(), 3),
                              oppB2b=round(recent[recent.opp_b2b == 1].gf.mean() / recent[recent.opp_b2b == 0].gf.mean(), 3))
    pj = model["projection"]
    off = {k: v for k, v in status["measured"].items() if abs(v - pj[k]) > 0.025}
    if off and n_cur >= MIN_CURRENT:
        alerts.append(dict(key="projection-effects", level="watch", title="Home ice or rest effects have drifted",
                           body="Measured over the last three seasons: " + ", ".join(f"{k} ×{v} (board uses ×{pj[k]})" for k, v in off.items())
                                + ". Update model.json → projection if this holds for a few weeks."))

    last_change = date.fromisoformat(model["history"][-1]["date"])
    days_since = (today - last_change).days
    gain = (m["val_new"] or 0) - (m["val_cur"] or 0)
    loss = (m["prev_cur"] or 0) - (m["prev_new"] or 0)
    if stepped == cur:
        decision, reason = "kept", "The refit agrees with the current weights."
    elif n_cur < MIN_CURRENT:
        decision, reason = "waiting", f"Only {n_cur} team-games played this season; changes need {MIN_CURRENT}. The fit is shown for reference."
    elif days_since < MIN_DAYS:
        decision, reason = "waiting", f"Last change was {days_since} days ago; changes need {MIN_DAYS} days apart."
    elif m["val_n"] < 300:
        decision, reason = "waiting", f"Only {m['val_n']} recent games to test on; need 300."
    elif gain < MIN_GAIN:
        decision, reason = "kept", f"New weights didn't beat the current ones on the last {VALIDATE_DAYS} days ({m['val_new']:.3f} vs {m['val_cur']:.3f})."
    elif loss > MAX_LOSS:
        decision, reason = "kept", f"New weights helped recently but did worse on last season ({m['prev_new']:.3f} vs {m['prev_cur']:.3f})."
    else:
        decision = "changed"
        reason = (f"Recent {m['val_n']} games: ranking accuracy {m['val_cur']:.3f} → {m['val_new']:.3f}; "
                  f"last season {m['prev_cur']:.3f} → {m['prev_new']:.3f}.")
        nw = {to_board[k]: v for k, v in stepped.items()}
        model["version"] += 1
        model["updated"] = today.isoformat()
        model["weights"] = nw
        model["history"].append(dict(version=model["version"], date=today.isoformat(), weights=nw, why="Weekly learner. " + reason))
        (ROOT / "model.json").write_text(json.dumps(model, indent=2), encoding="utf-8")
        fmt = lambda W: ", ".join(f"{k} {round(W[k] * 100)}%" for k in ["off", "def", "goalie", "st", "l5", "fin"] if W.get(k))
        alerts.append(dict(key=f"model-v{model['version']}", level="info", sticky=True,
                           title=f"Model updated to v{model['version']}",
                           body=f"Weights changed from {fmt(status['current'])} to {fmt(nw)}. {reason} "
                                f"To undo, revert the commit that changed model.json, or edit its weights back."))
    status.update(decision=decision, reason=reason, alerts=alerts)
    big = {to_board[k]: (cur[k], target[k]) for k in keys if abs(target[k] - cur[k]) >= 0.20}
    if big and n_cur >= MIN_CURRENT and (m["val_target"] or 0) > (m["val_cur"] or 0) + 0.005:
        alerts.append(dict(key="big-shift", level="watch", title="The data wants a large weight change",
                           body="The full refit differs from the board by 20+ points on: "
                                + ", ".join(f"{k} {round(a * 100)}% → {round(b * 100)}%" for k, (a, b) in big.items())
                                + f". It ranks recent games at {m['val_target']:.3f} vs {m['val_cur']:.3f} now. The learner moves 10 points at a time; "
                                  f"you can jump straight there by editing model.json."))
    (HERE / "learn_status.json").write_text(json.dumps(status, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: status[k] for k in ["decision", "reason", "current", "target", "candidate", "metrics"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
