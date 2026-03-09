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
st.title("📈 BTC/USD Prévision OHLC (1 Jour) par Source")

# -----------------------------
# Charger données BTC
# -----------------------------
@st.cache_data
def load_data():
    end_date = date.today()
    start_date = end_date - timedelta(days=365)

    data = yf.download(
        'BTC-USD',
        start=start_date,
        end=end_date,
        progress=False
    )

    data.reset_index(inplace=True)
    return data

data = load_data()
df_train = data.set_index('Date')

last_price = float(data['Close'].iloc[-1])

# -----------------------------
# Générer OHLC unique
# -----------------------------
def generate_ohlc_per_source(source_name, base_price):

    impact = random.randint(-3, 3)

    open_price = round(base_price * (1 + 0.001*impact), 2)
    high_price = round(open_price * (1 + random.uniform(0.0005,0.002)), 2)
    low_price = round(open_price * (1 - random.uniform(0.0005,0.002)), 2)
    close_price = round(open_price * (1 + 0.0005*impact), 2)

    time_event = datetime.now() + timedelta(days=1)

    return pd.DataFrame([{
        "Site": source_name,
        "Date/Heure": time_event.strftime("%Y-%m-%d"),
        "Annonce": f"Prévision BTC 24h ({source_name})",
        "O": open_price,
        "H": high_price,
        "L": low_price,
        "C": close_price,
        "URL": f"https://www.{source_name.lower()}.com"
    }])

# -----------------------------
# Paramètres utilisateur
# -----------------------------
historique_jours = st.slider(
    "📉 Nombre de jours historiques",
    10,
    60,
    30
)

sources_disponibles = [
"CoinDesk",
"Investing",
"ForexFactory",
"TradingEconomics",
"MarketWatch"
]

sources_selectionnees = st.multiselect(
    "Sélectionner sources pour OHLC",
    options=sources_disponibles,
    default=sources_disponibles
)

# -----------------------------
# Lancer prévision
# -----------------------------
if st.button("Lancer prévision 24h"):

    with st.spinner("Prévision OHLC..."):

        # Prévision ARIMA
        model = ARIMA(df_train['Close'], order=(5,1,0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)

        # Générer OHLC
        df_all_sources = pd.DataFrame()

        for source in sources_selectionnees:

            df_source = generate_ohlc_per_source(
                source,
                last_price
            )

            df_all_sources = pd.concat(
                [df_all_sources, df_source],
                ignore_index=True
            )

        # ---------------------
        # Tableau OHLC
        # ---------------------
        st.subheader("📋 Prévision OHLC (24h)")

        for site in sources_selectionnees:

            df_site = df_all_sources[
                df_all_sources['Site']==site
            ]

            st.markdown(f"### 🔹 Source : {site}")

            for i,row in df_site.iterrows():

                st.markdown(
                    f"**{row['Date/Heure']}** | "
                    f"O: {row['O']} | "
                    f"H: {row['H']} | "
                    f"L: {row['L']} | "
                    f"C: {row['C']} | "
                    f"Annonce: {row['Annonce']}"
                )

                st.markdown(
                    f"[Voir annonce]({row['URL']})"
                )

            st.markdown("---")

        # ---------------------
        # Graphique
        # ---------------------
        df_candles = data.tail(historique_jours).copy()

        fig = go.Figure()

        # Historique
        fig.add_trace(
            go.Candlestick(
                x=df_candles['Date'],
                open=df_candles['Open'],
                high=df_candles['High'],
                low=df_candles['Low'],
                close=df_candles['Close'],
                name='Historique',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350'
            )
        )

        # Prévision
        for i,site in enumerate(sources_selectionnees):

            df_site = df_all_sources[
                df_all_sources['Site']==site
            ]

            fig.add_trace(
                go.Candlestick(
                    x=pd.to_datetime(df_site['Date/Heure']),
                    open=df_site['O'],
                    high=df_site['H'],
                    low=df_site['L'],
                    close=df_site['C'],
                    name=f'Prévision {site}'
                )
            )

        fig.update_layout(
            title=f'BTC/USD Historique {historique_jours} jours + Prévision 24h',
            xaxis_title='Date',
            yaxis_title='Prix USD',
            template='plotly_dark',
            xaxis_rangeslider_visible=False,
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)
