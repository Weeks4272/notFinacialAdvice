from __future__ import annotations
import requests
import pandas as pd
from meteostat import Daily, Point

def _open_meteo_daily(lat: float, lon: float) -> dict:
    url = ("https://api.open-meteo.com/v1/forecast"
           f"?latitude={lat}&longitude={lon}"
           "&daily=temperature_2m_max,precipitation_probability_max"
           "&forecast_days=2&timezone=auto")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

def _nws_point(lat: float, lon: float) -> dict:
    headers = {"User-Agent": "PredictKit (educational)"}
    r = requests.get(f"https://api.weather.gov/points/{lat},{lon}", headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()

def _nws_forecast(url: str) -> dict:
    headers = {"User-Agent": "PredictKit (educational)"}
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()

def fetch_forecasts(cfg: dict) -> pd.DataFrame:
    lat = cfg["location"]["latitude"]
    lon = cfg["location"]["longitude"]
    out = {}

    if cfg["weather"].get("use_open_meteo", True):
        om = _open_meteo_daily(lat, lon)
        df = pd.DataFrame({
            "date": om["daily"]["time"],
            "om_temp_max": om["daily"]["temperature_2m_max"],
            "om_pop_max": om["daily"]["precipitation_probability_max"],
        })
        out["open_meteo"] = df

    if cfg["weather"].get("use_nws", True):
        pt = _nws_point(lat, lon)
        forecast_url = pt["properties"]["forecast"]
        fc = _nws_forecast(forecast_url)
        recs = []
        for p in fc["properties"]["periods"]:
            if p.get("isDaytime", True):
                dt = pd.to_datetime(p["startTime"]).date()
                temp = p.get("temperature")
                pop = p.get("probabilityOfPrecipitation", {}).get("value")
                recs.append({"date": dt, "nws_temp_max": temp, "nws_pop": pop})
        if recs:
            df = pd.DataFrame(recs).groupby("date").max(numeric_only=True).reset_index()
            out["nws"] = df

    df_all = None
    for _, df in out.items():
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df_all = df if df_all is None else pd.merge(df_all, df, on="date", how="outer")

    if df_all is None:
        raise RuntimeError("No forecasts fetched.")

    df_all["cons_temp_max"] = df_all[["om_temp_max","nws_temp_max"]].mean(axis=1, skipna=True)
    if "nws_pop" in df_all:
        df_all["cons_pop"] = (df_all[["om_pop_max","nws_pop"]].mean(axis=1, skipna=True)) / 100.0
    else:
        df_all["cons_pop"] = df_all["om_pop_max"] / 100.0

    df_all["date"] = pd.to_datetime(df_all["date"]).dt.date
    return df_all.sort_values("date").reset_index(drop=True)

def brier_score(p: float, o: int) -> float:
    return (p - o) ** 2

def verify_and_score_weather(cfg: dict, df_fore: pd.DataFrame) -> pd.DataFrame:
    lat = cfg["location"]["latitude"]
    lon = cfg["location"]["longitude"]
    recs = []
    for _, row in df_fore.iterrows():
        pt = Point(lat, lon)
        start = pd.Timestamp(row["date"])
        end = start + pd.Timedelta(days=1)
        df = Daily(pt, start, end).fetch()
        if df.empty:
            continue
        tmax_c = df["tmax"].iloc[0] if "tmax" in df.columns else None
        prcp = df["prcp"].iloc[0] if "prcp" in df.columns else 0.0
        tmax_f = tmax_c * 9/5 + 32 if tmax_c is not None else None
        precip_occurred = 1 if (prcp is not None and prcp > 0) else 0
        mae = abs(row["cons_temp_max"] - tmax_f) if (row.get("cons_temp_max") is not None and tmax_f is not None) else None
        bs = brier_score(row["cons_pop"], precip_occurred) if row.get("cons_pop") is not None else None
        recs.append({
            "date": row["date"],
            "cons_temp_max": row.get("cons_temp_max"),
            "cons_pop": row.get("cons_pop"),
            "obs_temp_max_f": tmax_f,
            "obs_precip_occurred": precip_occurred,
            "temp_mae": mae,
            "pop_brier": bs
        })
    return pd.DataFrame(recs)
