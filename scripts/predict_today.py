from predictkit.config import load_config
from predictkit.weather import fetch_forecasts
from predictkit.markets import predict_markets_today

def main():
    cfg = load_config()

    print("=== Weather (next 48h consensus) ===")
    df_fore = fetch_forecasts(cfg)
    print(df_fore.head(2).to_string(index=False))

    print("\n=== Markets: Prob. Up Next Day ===")
    df = predict_markets_today(cfg)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
