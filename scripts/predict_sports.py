from pathlib import Path
import json
import pandas as pd
from predictkit.config import load_config
from predictkit.sports import predict_upcoming_games

def main():
    cfg = load_config()
    ratings_path = Path("data/sports/elo_ratings.json")
    if not ratings_path.exists():
        raise SystemExit("Run scripts/backtest_sports.py first to produce elo_ratings.json")

    ratings = json.loads(ratings_path.read_text())
    upcoming = Path("data/sports/upcoming_sample.csv")
    if not upcoming.exists():
        upcoming.write_text("date,home_team,away_team\n2024-11-05,Bulls,Celtics\n")

    df = predict_upcoming_games(cfg, ratings, upcoming)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
