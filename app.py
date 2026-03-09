import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from datetime import date, timedelta, datetime
import warnings
import random

warnings.filterwarnings('ignore')
st.set_page_config(page_title="BTC Analyse par Source", layout="centered")
st.title("📈 BTC/USD Prévisions OHLC par Source Individuelle")

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
# Générer OHLC unique par source
# -----------------------------
def generate_ohlc_per_source(source_name, base_price, n=3):
    """
    Génère n annonces OHLC simulées uniques pour une source
    """
    annonces = []
    for i in range(n):
        impact = random.randint(-3, 3)
        open_price = round(base_price * (1 + 0.001*impact), 2)
        high_price = round(open_price * (1 + random.uniform(0.0005,0.002)), 2)
        low_price = round(open_price * (1 - random.uniform(0.0005,0.002)), 2)
        close_price = round(open_price * (1 + 0.0005*impact), 2)
        time_event = datetime.now() + timedelta(minutes=5*i)
        annonces.append({
            "Site": source_name,
            "Date/Heure": time_event.strftime("%Y-%m-%d %H:%M"),
            "Annonce": f"Annonce simulée {i+1} ({source_name})",
            "O": open_price,
            "H": high_price,
            "L": low_price,
            "C": close_price,
            "URL": f"https://www.{source_name.lower()}.com"
        })
    return pd.DataFrame(annonces)

# -----------------------------
# Paramètres utilisateur
# -----------------------------
forecast_days = st.slider("🗓️ Combien de jours prédire ?", 1, 14, 7)
historique_jours = st.slider("📉 Nombre de jours historiques", 10, 60, 30)

sources_disponibles = ["CoinDesk","Investing","ForexFactory","TradingEconomics","MarketWatch"]
sources_selectionnees = st.multiselect(
    "Sélectionner sources pour OHLC",
    options=sources_disponibles,
    default=sources_disponibles
)

if st.button(f"Lancer prévision {forecast_days} jours"):

    with st.spinner("Prévision OHLC par source individuelle..."):

        # Prévision ARIMA base
        model = ARIMA(df_train['Close'], order=(5,1,0))
        model_fit = model.fit()
        forecast_base = model_fit.forecast(steps=forecast_days)

        # Générer OHLC par source
        df_all_sources = pd.DataFrame()
        for source in sources_selectionnees:
            df_source = generate_ohlc_per_source(source, last_price, n=forecast_days)
            df_all_sources = pd.concat([df_all_sources, df_source], ignore_index=True)

        # ---------------------
        # Tableau OHLC par source
        # ---------------------
        st.subheader("📋 OHLC Prévisionnel par Source")
        for site in sources_selectionnees:
            df_site = df_all_sources[df_all_sources['Site']==site]
            st.markdown(f"### 🔹 Source : {site}")
            for i, row in df_site.iterrows():
                st.markdown(
                    f"**{row['Date/Heure']}** | O: {row['O']} | H: {row['H']} | L: {row['L']} | C: {row['C']} | Annonce: {row['Annonce']}",
                    unsafe_allow_html=True
                )
                st.markdown(f'<a href="{row["URL"]}" target="_blank">Voir l’annonce complète</a>', unsafe_allow_html=True)
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

        # Bougies par source
        colors = ['#FFD700','#FFA500','#FFFF00','#FFDAB9','#FFFACD']
        for i, site in enumerate(sources_selectionnees):
            df_site = df_all_sources[df_all_sources['Site']==site]
            fig.add_trace(go.Candlestick(
                x=pd.to_datetime(df_site['Date/Heure']),
                open=df_site['O'],
                high=df_site['H'],
                low=df_site['L'],
                close=df_site['C'],
                name=f'Prévision {site}',
                increasing_line_color=colors[i%len(colors)],
                decreasing_line_color=colors[i%len(colors)],
                increasing_fillcolor='rgba(255,215,0,0.6)',
                decreasing_fillcolor='rgba(255,215,0,0.6)'
            ))

            # Zone prévision pour chaque source
            if len(df_site) > 0:
                start_zone = pd.to_datetime(df_site['Date/Heure']).min()
                end_zone = pd.to_datetime(df_site['Date/Heure']).max()
                fig.add_vrect(
                    x0=start_zone,
                    x1=end_zone,
                    fillcolor='rgba(255,215,0,0.08)',
                    layer='below',
                    line_width=0,
                    annotation_text=f"Zone {site}",
                    annotation_position="top left",
                    annotation_font_color="#FFD700"
                )

        fig.update_layout(
            title=f'BTC/USD — Historique {historique_jours}j + Prévisions par source',
            xaxis_title='Date',
            yaxis_title='Prix USD',
            template='plotly_dark',
            xaxis_rangeslider_visible=False,
            height=600,
            plot_bgcolor='#1e1e2f',
            paper_bgcolor='#1e1e2f'
        )

        st.plotly_chart(fig, use_container_width=True)
