import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from datetime import date, timedelta
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="BTC Candlestick Forecast", layout="centered")
st.title("📈 Prévision BTC/USD avec bougies chandelier")

# -------------------------
# Téléchargement BTC
# -------------------------
@st.cache_data
def load_data():
    end_date = date.today()
    start_date = end_date - timedelta(days=365)
    data = yf.download("BTC-USD", start=start_date, end=end_date, progress=False)
    data.reset_index(inplace=True)
    return data

data = load_data()
st.success("Données BTC chargées")

# -------------------------
# Slider prévision
# -------------------------
forecast_days = st.slider("Nombre de jours à prévoir", 1, 14, 7)

# -------------------------
# Bouton prévision
# -------------------------
if st.button("Lancer la prévision"):

    df_train = data.set_index("Date")
    model = ARIMA(df_train["Close"], order=(5,1,0))
    model_fit = model.fit()
    forecast_close = model_fit.forecast(steps=forecast_days)

    last_date = pd.to_datetime(data["Date"].iloc[-1])
    last_price = float(data["Close"].iloc[-1])

    future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days+1)]
    dates_formattees = [d.strftime("%d/%m/%Y") for d in future_dates]

    # -------------------------
    # Génération OHLC pour bougies
    # -------------------------
    ohlc_forecast = []
    prev_close = last_price
    range_moyen = (data["High"] - data["Low"]).mean()

    for i in range(forecast_days):
        predicted_close = float(forecast_close.iloc[i])
        predicted_open = prev_close
        predicted_high = predicted_open + range_moyen*0.75
        predicted_low = predicted_open - range_moyen*0.55

        ohlc_forecast.append({
            "Date": dates_formattees[i],
            "Open": float(predicted_open),
            "High": float(predicted_high),
            "Low": float(predicted_low),
            "Close": float(predicted_close)
        })

        prev_close = predicted_close

    df_ohlc = pd.DataFrame(ohlc_forecast)

    # -------------------------
    # Table OHLC
    # -------------------------
    st.subheader("Tableau OHLC")
    st.dataframe(df_ohlc)

    # -------------------------
    # Graphique chandeliers
    # -------------------------
    df_ohlc["Date"] = pd.to_datetime(df_ohlc["Date"], dayfirst=True)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df_ohlc["Date"],
        open=df_ohlc["Open"],
        high=df_ohlc["High"],
        low=df_ohlc["Low"],
        close=df_ohlc["Close"],
        increasing_line_color="green",
        decreasing_line_color="red"
    ))

    fig.update_layout(
        title="Prévision BTC/USD",
        xaxis_title="Date",
        yaxis_title="Prix USD",
        xaxis_rangeslider_visible=False,
        yaxis=dict(autorange=True)
    )

    st.plotly_chart(fig, use_container_width=True)
