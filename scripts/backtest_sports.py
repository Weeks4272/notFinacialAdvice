from pathlib import Path
import json
from predictkit.config import load_config
from predictkit.sports import run_elo_backtest

def main():
    cfg = load_config()
    history = Path("data/sports/history_sample.csv")
    if not history.exists():
        history.write_text(
            "date,home_team,away_team,home_score,away_score\n"
            "2024-10-25,Bulls,Bucks,102,98\n"
            "2024-10-27,Bucks,Bulls,110,105\n"
            "2024-11-01,Bulls,Celtics,95,103\n"
            "2024-11-03,Celtics,Bulls,90,92\n"
        )

    res = run_elo_backtest(cfg, history)
    Path("data/sports").mkdir(parents=True, exist_ok=True)
    (Path("data/sports/elo_ratings.json")).write_text(json.dumps(res["ratings"], indent=2))
    (Path("data/sports/elo_metrics.json")).write_text(json.dumps({"logloss": res["logloss"]}, indent=2))
    print("Sports Elo backtest complete. Ratings & metrics saved.")

if __name__ == "__main__":
    main()
