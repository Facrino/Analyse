import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from datetime import date, timedelta
import warnings

warnings.filterwarnings('ignore')

# Configuration
st.set_page_config(page_title="BTC Analyse", layout="centered")
st.title("📈 Prévision BTC/USD (ARIMA)")

# Chargement des données
@st.cache_data
def load_data():
    end_date = date.today()
    start_date = end_date - timedelta(days=365)
    data = yf.download('BTC-USD', start=start_date, end=end_date, progress=False)
    data.reset_index(inplace=True)
    return data

data = load_data()
st.success("✅ Données actuelles téléchargées avec succès !")

# Choix des jours de prévision
forecast_days = st.slider(
    "🗓️ Combien de jours voulez-vous prédire ?",
    min_value=1,
    max_value=14,
    value=7
)

if st.button(f"Lancer les prévisions pour {forecast_days} jours"):

    with st.spinner("L'IA calcule les prévisions mathématiques..."):

        df_train = data.set_index('Date')

        model = ARIMA(df_train['Close'], order=(5,1,0))
        model_fit = model.fit()

        forecast = model_fit.forecast(steps=forecast_days)

        last_date = pd.to_datetime(data['Date'].iloc[-1])
        last_price = float(data['Close'].iloc[-1])

        future_dates = [
            last_date + timedelta(days=i)
            for i in range(1, forecast_days+1)
        ]

        st.divider()

        # Graphique chandelier
        st.subheader("🕯️ Graphique Chandelier BTC/USD")

        df_candles = data.tail(60).copy()

        if isinstance(df_candles.columns, pd.MultiIndex):
            df_candles.columns = df_candles.columns.get_level_values(0)

        fig = go.Figure()

        # ───────── Historique ─────────
        fig.add_trace(go.Candlestick(
            x=df_candles['Date'],
            open=df_candles['Open'],
            high=df_candles['High'],
            low=df_candles['Low'],
            close=df_candles['Close'],
            name='Historique (60j)',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350',
            increasing_fillcolor='#26a69a',
            decreasing_fillcolor='#ef5350'
        ))

        # ───────── Prévision OHLC ─────────
        pred_close = list(forecast)
        pred_open = [last_price] + pred_close[:-1]

        pred_high = [c * 1.01 for c in pred_close]
        pred_low = [c * 0.99 for c in pred_close]

        # ───────── Bougies prévision jaunes ─────────
        fig.add_trace(go.Candlestick(
            x=future_dates,
            open=pred_open,
            high=pred_high,
            low=pred_low,
            close=pred_close,

            name='Prévision',

            increasing_line_color='#FFD700',
            decreasing_line_color='#FFD700',

            increasing_fillcolor='rgba(255,215,0,0.6)',
            decreasing_fillcolor='rgba(255,215,0,0.6)',

            hovertemplate=
            "<b>Date:</b> %{x}<br>"
            "Open: $%{open:.2f}<br>"
            "High: $%{high:.2f}<br>"
            "Low: $%{low:.2f}<br>"
            "Close: $%{close:.2f}"
            "<extra>Prévision</extra>"
        ))

        # ───────── Zone de prévision ─────────
        fig.add_vrect(
            x0=last_date,
            x1=future_dates[-1],
            fillcolor="rgba(255,215,0,0.08)",
            layer="below",
            line_width=0,
            annotation_text="Zone prévision",
            annotation_position="top left",
            annotation_font_color="#FFD700"
        )

        # ───────── Mise en forme ─────────
        fig.update_layout(

            title=f'BTC/USD — Chandelier 60j + Prévision {forecast_days} jours',

            xaxis_title='Date',
            yaxis_title='Prix USD',

            template='plotly_dark',

            xaxis_rangeslider_visible=False,

            height=550,

            plot_bgcolor='#1e1e2f',
            paper_bgcolor='#1e1e2f',

            bargap=0.25
        )

        # Espacer les bougies
        fig.update_xaxes(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.08)',
            range=[
                df_candles['Date'].iloc[0],
                future_dates[-1] + timedelta(days=3)
            ]
        )

        fig.update_yaxes(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.08)',
            tickprefix='$'
        )

        st.plotly_chart(fig, use_container_width=True)
