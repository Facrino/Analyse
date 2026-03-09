import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from datetime import date, timedelta, datetime
import warnings
import requests
from bs4 import BeautifulSoup

warnings.filterwarnings('ignore')

st.set_page_config(page_title="BTC Analyse", layout="centered")
st.title("📈 BTC/USD Prévisions dynamiques selon sources économiques")

# -----------------------------
# Charger données BTC
# -----------------------------
@st.cache_data
def load_data():
    end_date = date.today()
    start_date = end_date - timedelta(days=365)
    data = yf.download('BTC-USD', start=start_date, end=end_date, progress=False)
    data.reset_index(inplace=True)
    return data

data = load_data()
df_train = data.set_index('Date')
last_price = float(data['Close'].iloc[-1])

# -----------------------------
# Fonction récupérer annonces économiques
# -----------------------------
def get_economic_announcements():
    sites = {
        "CoinDesk": "https://www.coindesk.com",
        "Investing": "https://www.investing.com/news/cryptocurrency-news",
        "ForexFactory": "https://www.forexfactory.com/calendar",
        "TradingEconomics": "https://tradingeconomics.com/calendar",
        "MarketWatch": "https://www.marketwatch.com/investing/cryptocurrency"
    }

    keywords_positive = ["bull","adoption","ETF","growth","approval","institution"]
    keywords_negative = ["ban","hack","crash","lawsuit","regulation","collapse"]

    results = []

    for site_name, url in sites.items():
        try:
            r = requests.get(url, timeout=10)
            soup = BeautifulSoup(r.text, "html.parser")
            titles = soup.find_all("h2")
            for t in titles[:5]:
                text = t.text.strip()
                text_lower = text.lower()
                impact = 0
                for p in keywords_positive:
                    if p in text_lower:
                        impact += 1
                for n in keywords_negative:
                    if n in text_lower:
                        impact -= 1

                time_event = datetime.now() + timedelta(minutes=5*len(results))

                results.append({
                    "Site": site_name,
                    "Date/Heure": time_event.strftime("%Y-%m-%d %H:%M"),
                    "Annonce": text,
                    "Impact": impact,
                    "URL": url
                })
        except:
            results.append({
                "Site": site_name,
                "Date/Heure": "-",
                "Annonce": "Impossible de récupérer",
                "Impact": 0,
                "URL": url
            })
    return pd.DataFrame(results)

# -----------------------------
# Option Prévision BTC
# -----------------------------
forecast_days = st.slider("🗓️ Combien de jours voulez-vous prédire ?", 1, 14, 7)
historique_jours = st.slider("📉 Nombre de jours historiques à afficher", 10, 60, 30)

# Choix sources
sources_disponibles = ["CoinDesk","Investing","ForexFactory","TradingEconomics","MarketWatch"]
sources_selectionnees = st.multiselect(
    "Sélectionnez les sources à utiliser pour calcul OHLC",
    options=sources_disponibles,
    default=sources_disponibles
)

if st.button(f"Lancer les prévisions pour {forecast_days} jours"):

    with st.spinner("Calcul ARIMA + OHLC dynamique selon sources..."):

        # Prévision ARIMA
        model = ARIMA(df_train['Close'], order=(5,1,0))
        model_fit = model.fit()
        forecast_base = model_fit.forecast(steps=forecast_days)

        # Récupérer annonces filtrées selon sources sélectionnées
        df_annonces = get_economic_announcements()
        df_filtered = df_annonces[df_annonces['Site'].isin(sources_selectionnees)]

        # Calcul impact combiné pour chaque jour
        # Ici on suppose que l'impact total est réparti uniformément sur tous les jours prévisionnels
        impact_total = df_filtered['Impact'].sum() * 0.002

        # Construire OHLC prévisionnel
        last_date = pd.to_datetime(data['Date'].iloc[-1])
        future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days+1)]

        pred_close = list(forecast_base * (1 + impact_total))
        pred_open = [last_price] + pred_close[:-1]
        pred_high = [c*1.01 for c in pred_close]
        pred_low = [c*0.99 for c in pred_close]

        # ---------------------
        # Tableau prévision avec liens cliquables
        # ---------------------
        st.subheader("📋 OHLC prévisionnel dynamique selon sources sélectionnées")
        for i in range(forecast_days):
            sources_html = " | ".join([f'<a href="{url}" target="_blank">{site}</a>' for site, url in zip(df_filtered['Site'], df_filtered['URL'])])
            heures = ", ".join(df_filtered['Date/Heure'])
            st.markdown(
                f"**{future_dates[i].strftime('%d/%m/%Y')}** | O: {pred_open[i]:.2f} | H: {pred_high[i]:.2f} | L: {pred_low[i]:.2f} | C: {pred_close[i]:.2f} | Heures: {heures}",
                unsafe_allow_html=True
            )
            st.markdown(sources_html, unsafe_allow_html=True)
            st.markdown("---")

        # ---------------------
        # Graphique
        # ---------------------
        df_candles = data.tail(historique_jours).copy()
        if isinstance(df_candles.columns, pd.MultiIndex):
            df_candles.columns = df_candles.columns.get_level_values(0)

        fig = go.Figure()

        # Historique
        fig.add_trace(go.Candlestick(
            x=df_candles['Date'],
            open=df_candles['Open'],
            high=df_candles['High'],
            low=df_candles['Low'],
            close=df_candles['Close'],
            name='Historique',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350',
            increasing_fillcolor='#26a69a',
            decreasing_fillcolor='#ef5350'
        ))

        # Prévision bougies jaunes
        fig.add_trace(go.Candlestick(
            x=future_dates,
            open=pred_open,
            high=pred_high,
            low=pred_low,
            close=pred_close,
            name='Prévision BTC',
            increasing_line_color='#FFD700',
            decreasing_line_color='#FFD700',
            increasing_fillcolor='rgba(255,215,0,0.6)',
            decreasing_fillcolor='rgba(255,215,0,0.6)'
        ))

        # Zone prévision
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

        fig.update_layout(
            title=f'BTC/USD — Historique {historique_jours}j + Prévision {forecast_days} jours',
            xaxis_title='Date',
            yaxis_title='Prix USD',
            template='plotly_dark',
            xaxis_rangeslider_visible=False,
            height=550,
            plot_bgcolor='#1e1e2f',
            paper_bgcolor='#1e1e2f'
        )

        fig.update_xaxes(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.08)',
            range=[df_candles['Date'].iloc[0] - timedelta(days=2), future_dates[-1] + timedelta(days=5)]
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.08)',
            tickprefix='$'
        )

        st.plotly_chart(fig, use_container_width=True)
