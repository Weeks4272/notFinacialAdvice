from pathlib import Path
from predictkit.config import load_config
from predictkit.weather import fetch_forecasts, verify_and_score_weather

def main():
    cfg = load_config()
    df_fore = fetch_forecasts(cfg)
    out_dir = Path("data/weather")
    out_dir.mkdir(parents=True, exist_ok=True)
    df_fore.to_csv(out_dir / "forecasts.csv", index=False)
    print("Saved forecasts.csv")

    df_scores = verify_and_score_weather(cfg, df_fore)
    if not df_scores.empty:
        df_scores.to_csv(out_dir / "scores.csv", index=False)
        print("Saved scores.csv")

if __name__ == "__main__":
    main()
