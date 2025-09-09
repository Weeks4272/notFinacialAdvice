import pandas as pd, numpy as np
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from pathlib import Path

def _features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    df_feat = pd.DataFrame(index=df.index)
    df_feat["ret1"] = close.pct_change()
    df_feat["ma5"] = close.rolling(5).mean() / close - 1
    df_feat["ma10"] = close.rolling(10).mean() / close - 1
    df_feat["ma20"] = close.rolling(20).mean() / close - 1
    df_feat["vol20"] = close.pct_change().rolling(20).std()
    delta = close.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    rs = up / (down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    df_feat["rsi14"] = rsi/100.0
    df_feat["label"] = (close.shift(-1) > close).astype(int)
    return df_feat.dropna()

def backtest_markets(cfg: dict, outdir: Path) -> pd.DataFrame:
    tickers = cfg["markets"]["tickers"]
    start = cfg["markets"]["start_date"]
    train_ratio = cfg["markets"].get("train_ratio", 0.8)
    out = []
    for t in tickers:
        data = yf.download(t, start=start, progress=False, auto_adjust=True)
        if len(data) < 300:
            continue
        feat = _features(data)
        n = len(feat)
        n_train = int(n * train_ratio)
        X = feat.drop(columns=["label"])
        y = feat["label"]
        model = LogisticRegression(max_iter=1000)
        model.fit(X.iloc[:n_train], y.iloc[:n_train])
        proba = model.predict_proba(X.iloc[n_train:])[:,1]
        pred = (proba >= 0.5).astype(int)
        acc = accuracy_score(y.iloc[n_train:], pred)
        try:
            auc = roc_auc_score(y.iloc[n_train:], proba)
        except ValueError:
            auc = np.nan
        out.append({"ticker": t, "n_test": len(pred), "acc": float(acc), "auc": float(auc)})
    return pd.DataFrame(out)

def predict_markets_today(cfg: dict) -> pd.DataFrame:
    tickers = cfg["markets"]["tickers"]
    rows = []
    for t in tickers:
        data = yf.download(t, period="6mo", interval="1d", progress=False, auto_adjust=True)
        feat = _features(data)
        X = feat.drop(columns=["label"])
        y = feat["label"]
        if len(X) < 60:
            continue
        model = LogisticRegression(max_iter=1000)
        model.fit(X.iloc[:-1], y.iloc[:-1])
        proba = model.predict_proba(X.iloc[[-1]])[0,1]
        rows.append({"ticker": t, "prob_up_next_day": float(proba)})
    return pd.DataFrame(rows).sort_values("prob_up_next_day", ascending=False)
