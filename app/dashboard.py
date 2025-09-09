import streamlit as st
import pandas as pd
from pathlib import Path
import json

from predictkit.config import load_config
from predictkit.weather import fetch_forecasts, verify_and_score_weather
from predictkit.markets import predict_markets_today
from predictkit.sports import predict_upcoming_games

st.set_page_config(page_title="PredictKit Dashboard", layout="wide")
cfg = load_config()

st.title("PredictKit — Weather • Markets • Sports")

# --- Weather Tab ---
tab1, tab2, tab3 = st.tabs(["🌤 Weather", "💹 Stock Picks", "🏀 Sports Elo"])

with tab1:
    st.subheader("Weather Forecast (next 48h)")
    try:
        df_fore = fetch_forecasts(cfg)
        st.dataframe(df_fore, use_container_width=True)
        st.caption("cons_temp_max = consensus max temperature (°F), cons_pop = consensus probability of precipitation (0–1)")

        # Try verifying yesterday’s forecast with Meteostat
        df_scores = verify_and_score_weather(cfg, df_fore)
        if not df_scores.empty:
            st.subheader("Verification (Observed vs Forecast)")
            st.dataframe(df_scores, use_container_width=True)
    except Exception as e:
        st.error(f"Weather error: {e}")

# --- Stock Picks Tab ---
with tab2:
    st.subheader("Stock Predictions (Logistic Regression)")
    try:
        df = predict_markets_today(cfg)
        if df.empty:
            st.info("No predictions available. Check your tickers in config.yaml.")
        else:
            st.dataframe(df, use_container_width=True)
            st.caption("prob_up_next_day = probability model predicts stock will close higher tomorrow")
    except Exception as e:
        st.error(f"Markets error: {e}")

# --- Sports Tab ---
with tab3:
    st.subheader("Sports Elo Ratings & Game Predictions")
    ratings_path = Path("data/sports/elo_ratings.json")
    upcoming = Path("data/sports/upcoming_sample.csv")

    if ratings_path.exists():
        ratings = json.loads(ratings_path.read_text())
        st.write(f"Loaded Elo ratings for {len(ratings)} teams.")
        top = sorted(ratings.items(), key=lambda x: x[1], reverse=True)[:10]
        st.write("Top 10 Elo teams:")
        st.dataframe(pd.DataFrame(top, columns=["team","elo"]).round(1), use_container_width=True)

        if upcoming.exists():
            try:
                dfp = predict_upcoming_games(cfg, ratings, upcoming)
                st.subheader("Upcoming Predictions")
                st.dataframe(dfp, use_container_width=True)
            except Exception as e:
                st.error(f"Error running upcoming predictions: {e}")
        else:
            st.info("No upcoming_sample.csv found in data/sports/")
    else:
        st.info("Run `python scripts/backtest_sports.py` first to generate Elo ratings.")
