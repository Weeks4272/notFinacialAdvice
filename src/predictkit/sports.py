import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class EloConfig:
    k_factor: float = 20.0
    home_field_adv: float = 65.0

def _expected(r_a, r_b):
    return 1.0 / (1 + 10 ** ((r_b - r_a)/400.0))

def _update(r_a, r_b, score_a, k):
    exp_a = _expected(r_a, r_b)
    new_a = r_a + k * (score_a - exp_a)
    new_b = r_b + k * ((1 - score_a) - (1 - exp_a))
    return new_a, new_b

def run_elo_backtest(cfg: dict, history_csv):
    k = cfg["sports"].get("k_factor", 20)
    hfa = cfg["sports"].get("home_field_adv", 65)
    df = pd.read_csv(history_csv, parse_dates=["date"])
    teams = pd.unique(pd.concat([df["home_team"], df["away_team"]], ignore_index=True))
    ratings = {t: 1500.0 for t in teams}
    preds, results = [], []
    for _, row in df.sort_values("date").iterrows():
        h, a = row["home_team"], row["away_team"]
        r_h = ratings[h] + hfa
        r_a = ratings[a]
        p_home = _expected(r_h, r_a)
        preds.append(p_home)
        results.append(1.0 if row["home_score"] > row["away_score"] else 0.0)
        new_h_raw, new_a_raw = _update(ratings[h] + hfa, ratings[a], results[-1], k)
        ratings[h] = new_h_raw - hfa
        ratings[a] = new_a_raw
    eps = 1e-12
    preds = np.clip(np.array(preds), eps, 1-eps)
    y = np.array(results)
    logloss = -float((y*np.log(preds) + (1-y)*np.log(1-preds)).mean())
    return {"ratings": ratings, "logloss": logloss}

def predict_upcoming_games(cfg: dict, ratings: dict, upcoming_csv):
    hfa = cfg["sports"].get("home_field_adv", 65)
    rows = []
    df = pd.read_csv(upcoming_csv, parse_dates=["date"])
    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        r_h = ratings.get(h, 1500.0) + hfa
        r_a = ratings.get(a, 1500.0)
        p_home = 1.0 / (1 + 10 ** ((r_a - r_h)/400.0))
        rows.append({"date": row["date"].date(), "home_team": h, "away_team": a, "p_home_win": float(p_home)})
    return pd.DataFrame(rows).sort_values(["date","home_team"])
