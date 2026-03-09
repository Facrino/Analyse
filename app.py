import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from datetime import date, timedelta
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="BTC Candlestick Forecast", layout="centered")
st.title("📈 Historique + Prévision BTC/USD en bougies vertes/rouges")

# -------------------------
# Télécharger données BTC
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

    # -------------------------
    # Génération OHLC pour prévision
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
            "Date": future_dates[i],
            "Open": float(predicted_open),
            "High": float(predicted_high),
            "Low": float(predicted_low),
            "Close": float(predicted_close)
        })

        prev_close = predicted_close

    df_forecast = pd.DataFrame(ohlc_forecast)

    # -------------------------
    # Graphique chandeliers vert/rouge
    # -------------------------
    st.subheader("🕯️ Historique + Prévision BTC/USD")

    # Assurer le format datetime
    data["Date"] = pd.to_datetime(data["Date"])
    df_forecast["Date"] = pd.to_datetime(df_forecast["Date"])

    fig = go.Figure()

    # Historique 1 an avec vert/rouge
    fig.add_trace(go.Candlestick(
        x=data["Date"],
        open=data["Open"],
        high=data["High"],
        low=data["Low"],
        close=data["Close"],
        name="Historique",
        increasing_line_color="green",
        decreasing_line_color="red"
    ))

    # Prévision ARIMA avec vert/rouge
    fig.add_trace(go.Candlestick(
        x=df_forecast["Date"],
        open=df_forecast["Open"],
        high=df_forecast["High"],
        low=df_forecast["Low"],
        close=df_forecast["Close"],
        name="Prévision",
        increasing_line_color="green",
        decreasing_line_color="red"
    ))

    fig.update_layout(
        title="BTC/USD Historique + Prévision",
        xaxis_title="Date",
        yaxis_title="Prix USD",
        xaxis_rangeslider_visible=True,
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # Tableau prévision
    # -------------------------
    st.subheader("Tableau prévision OHLC")
    st.dataframe(df_forecast)
