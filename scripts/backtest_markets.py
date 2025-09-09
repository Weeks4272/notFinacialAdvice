from pathlib import Path
from predictkit.config import load_config
from predictkit.markets import backtest_markets
import pandas as pd

def main():
    cfg = load_config()
    outdir = Path("data/markets/models")
    df = backtest_markets(cfg, outdir)
    out = Path("data/markets/backtest.csv")
    df.to_csv(out, index=False)
    print(f"Market backtest written to {out}")

if __name__ == "__main__":
    main()
