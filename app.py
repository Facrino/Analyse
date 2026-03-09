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

# Configuration
st.set_page_config(page_title="BTC Analyse", layout="centered")
st.title("📈 BTC/USD Analyse & Prévisions")

# Sidebar menu
menu = st.sidebar.selectbox(
    "Choisir une option",
    ["Prévision BTC", "Analyse annonces économiques"]
)

# -------------------------
# Charger données BTC
# -------------------------
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

# -------------------------
# Fonction analyse news BTC
# -------------------------
def analyse_news_btc():
    sites = [
        "https://www.coindesk.com",
        "https://cointelegraph.com",
        "https://cryptoslate.com",
        "https://bitcoinmagazine.com",
        "https://decrypt.co"
    ]

    keywords_positive = ["adoption","bull","institution","ETF","approval","growth"]
    keywords_negative = ["ban","hack","crash","lawsuit","regulation","collapse"]

    score = 0
    headlines = []

    for url in sites:
        try:
            r = requests.get(url, timeout=10)
            soup = BeautifulSoup(r.text,"html.parser")
            titles = soup.find_all("h2")
            for t in titles[:5]:
                text = t.text.strip()
                text_lower = text.lower()
                headlines.append(text)
                for p in keywords_positive:
                    if p in text_lower:
                        score += 1
                for n in keywords_negative:
                    if n in text_lower:
                        score -= 1
        except:
            pass

    return score, headlines

# -------------------------
# Fonction pour récupérer annonces économiques scalping
# -------------------------
def get_economic_announcements():
    sites = {
        "Investing": "https://www.investing.com/news/cryptocurrency-news",
        "ForexFactory": "https://www.forexfactory.com/calendar",
        "TradingEconomics": "https://tradingeconomics.com/calendar",
        "MarketWatch": "https://www.marketwatch.com/investing/cryptocurrency",
        "CoinDesk": "https://www.coindesk.com"
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

                # Heures simulées pour scalping
                time_event = datetime.now() + timedelta(minutes=5*len(results))

                open_price = last_price
                close_price = last_price * (1 + 0.001*impact)
                high_price = max(open_price, close_price) * 1.001
                low_price = min(open_price, close_price) * 0.999

                results.append({
                    "Site": site_name,
                    "Date/Heure": time_event.strftime("%Y-%m-%d %H:%M"),
                    "Annonce": text,
                    "O": round(open_price,2),
                    "H": round(high_price,2),
                    "L": round(low_price,2),
                    "C": round(close_price,2),
                    "Impact": impact
                })
        except:
            results.append({
                "Site": site_name,
                "Date/Heure": "-",
                "Annonce": "Impossible de récupérer les données",
                "O": "-",
                "H": "-",
                "L": "-",
                "C": "-",
                "Impact": 0
            })
    return pd.DataFrame(results)

# -------------------------
# Option 1 : Prévision BTC
# -------------------------
if menu == "Prévision BTC":

    forecast_days = st.slider("🗓️ Combien de jours voulez-vous prédire ?", 1, 14, 7)

    if st.button(f"Lancer les prévisions pour {forecast_days} jours"):

        with st.spinner("Calcul des prévisions ARIMA..."):

            model = ARIMA(df_train['Close'], order=(5,1,0))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=forecast_days)

            # Analyse news BTC
            news_score, headlines = analyse_news_btc()
            impact = news_score * 0.002
            forecast = forecast * (1 + impact)

            last_date = pd.to_datetime(data['Date'].iloc[-1])
            future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days+1)]

            # Construire OHLC prévision
            pred_close = list(forecast)
            pred_open = [last_price] + pred_close[:-1]
            pred_high = [c*1.01 for c in pred_close]
            pred_low = [c*0.99 for c in pred_close]

            st.subheader("📰 Headlines BTC analysées")
            for h in headlines[:10]:
                st.write("•", h)

            # Tableau prévision
            st.subheader("📋 Tableau prévision OHLC")
            df_forecast = pd.DataFrame({
                "Date": [d.strftime("%d/%m/%Y") for d in future_dates],
                "Open": pred_open,
                "High": pred_high,
                "Low": pred_low,
                "Close": pred_close
            })
            st.dataframe(df_forecast, use_container_width=True)

            # Graphique chandelier
            df_candles = data.tail(60).copy()
            fig = go.Figure()

            # Historique
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
                decreasing_fillcolor='rgba(255,215,0,0.6)',
                hovertemplate="<b>Date:</b> %{x}<br>Open: $%{open:.2f}<br>High: $%{high:.2f}<br>Low: $%{low:.2f}<br>Close: $%{close:.2f}<extra>Prévision</extra>"
            ))

            # Zone jaune
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

            fig.update_xaxes(
                showgrid=True,
                gridcolor='rgba(255,255,255,0.08)',
                range=[df_candles['Date'].iloc[0], future_dates[-1] + timedelta(days=3)]
            )

            fig.update_yaxes(
                showgrid=True,
                gridcolor='rgba(255,255,255,0.08)',
                tickprefix='$'
            )

            st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Option 2 : Annonces économiques
# -------------------------
if menu == "Analyse annonces économiques":

    st.subheader("📋 Annonces économiques et impact OHLC pour scalping")

    df_annonces = get_economic_announcements()
    st.dataframe(df_annonces, use_container_width=True)
