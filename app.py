import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from datetime import date, timedelta
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="BTC Candlestick Forecast", layout="centered")
st.title("📈 Historique + Prévision BTC/USD (bougies)")

# -------------------------
# Télécharger données BTC
# -------------------------
def load_data():
    end_date = date.today()
    start_date = end_date - timedelta(days=365)

    data = yf.download("BTC-USD", start=start_date, end=end_date, progress=False)
    if data.empty:
        st.error("⚠️ Aucune donnée récupérée")
        return pd.DataFrame()
    
    data.reset_index(inplace=True)
    for col in ["Open","High","Low","Close"]:
        if col not in data.columns:
            st.error(f"⚠️ Colonne manquante : {col}")
            return pd.DataFrame()
    return data

data = load_data()
if data.empty:
    st.stop()
else:
    st.success("✅ Données BTC chargées")

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
    last_close = float(data["Close"].iloc[-1])
    future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days+1)]

    # Générer bougies prévision simples
    range_moyen = float((data["High"] - data["Low"]).mean())
    ohlc_forecast = []
    prev_close = last_close
    for i in range(forecast_days):
        close = float(forecast_close.iloc[i])
        open_ = prev_close
        high = open_ + max(range_moyen*0.75, 0.01)
        low  = open_ - max(range_moyen*0.55, 0.01)
        ohlc_forecast.append({"Date": future_dates[i], "Open": open_, "High": high, "Low": low, "Close": close})
        prev_close = close
    df_forecast = pd.DataFrame(ohlc_forecast)

    # -------------------------
    # Graphique chandeliers
    # -------------------------
    st.subheader("🕯️ Historique + Prévision BTC/USD")
    fig = go.Figure()

    # Historique vert/rouge
    fig.add_trace(go.Candlestick(
        x=data["Date"], open=data["Open"], high=data["High"],
        low=data["Low"], close=data["Close"],
        name="Historique", increasing_line_color="green",
        decreasing_line_color="red"
    ))

    # Prévision bleu/orange
    fig.add_trace(go.Candlestick(
        x=df_forecast["Date"], open=df_forecast["Open"], high=df_forecast["High"],
        low=df_forecast["Low"], close=df_forecast["Close"],
        name="Prévision", increasing_line_color="blue",
        decreasing_line_color="orange"
    ))

    y_min = float(min(data["Low"].min(), df_forecast["Low"].min())) * 0.995
    y_max = float(max(data["High"].max(), df_forecast["High"].max())) * 1.005
    fig.update_layout(xaxis_rangeslider_visible=True, yaxis=dict(range=[y_min, y_max]), template="plotly_dark")

    st.plotly_chart(fig, use_container_width=True)
