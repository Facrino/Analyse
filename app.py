import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import date, timedelta
import requests
import warnings

warnings.filterwarnings('ignore')

# Configuration
st.set_page_config(page_title="BTC Analyse", layout="centered")
st.title("📈 Prévision BTC/USD (ARIMA + Analyse News)")

# =========================
# TELECHARGEMENT DES DONNEES
# =========================

@st.cache_data
def load_data():
    end_date = date.today()
    start_date = end_date - timedelta(days=365)
    data = yf.download('BTC-USD', start=start_date, end=end_date, progress=False)
    data.reset_index(inplace=True)
    return data

data = load_data()

st.success("✅ Données BTC chargées")

# =========================
# CALENDRIER ECONOMIQUE
# =========================

@st.cache_data(ttl=1800)
def get_multi_source_calendar():

    events = []

    sources = [
        "https://www.jblanked.com/news/api/forex-factory/calendar/today/",
        "https://www.jblanked.com/news/api/fxstreet/calendar/today/",
        "https://www.jblanked.com/news/api/mql5/calendar/today/"
    ]

    for url in sources:
        try:
            r = requests.get(url, timeout=8)

            if r.status_code == 200:
                events.extend(r.json())

        except:
            pass

    if len(events) > 0:
        return pd.DataFrame(events)
    else:
        return pd.DataFrame()

calendar_df = get_multi_source_calendar()

if not calendar_df.empty:
    st.info(f"📅 Calendrier chargé depuis 3 sites → {len(calendar_df)} annonces")

# =========================
# PARAMETRES UTILISATEUR
# =========================

forecast_days = st.slider(
    "🗓️ Combien de jours voulez-vous prédire ?",
    1,
    14,
    7
)

# =========================
# BOUTON PREVISION
# =========================

if st.button("Lancer les prévisions"):

    with st.spinner("Calcul de l'IA..."):

        # =========================
        # MODELE ARIMA
        # =========================

        df_train = data.set_index('Date')

        model = ARIMA(df_train['Close'], order=(5,1,0))
        model_fit = model.fit()

        forecast_close = model_fit.forecast(steps=forecast_days)

        last_date = pd.to_datetime(data['Date'].iloc[-1])
        last_price = float(data['Close'].iloc[-1])

        future_dates = [
            last_date + timedelta(days=i)
            for i in range(1, forecast_days + 1)
        ]

        dates_formattees = [
            d.strftime('%d/%m/%Y')
            for d in future_dates
        ]

        # =========================
        # STRATEGIE SCALPING
        # =========================

        ohlc_forecast = []

        prev_close = last_price

        range_moyen = (data['High'] - data['Low']).mean()

        for i in range(forecast_days):

            day_str = future_dates[i].strftime('%Y-%m-%d')

            predicted_close = float(forecast_close.iloc[i])

            # Vérifier les news
            high_impact = 0

            if not calendar_df.empty:

                if 'date' in calendar_df.columns:

                    day_events = calendar_df[
                        calendar_df['date'].astype(str).str.contains(day_str, na=False)
                    ]

                    if 'impact' in calendar_df.columns:

                        high_impact = len(
                            day_events[
                                day_events['impact'].astype(str).str.contains('High|3', case=False, na=False)
                            ]
                        )

            multiplier = 2.8 if high_impact > 0 else 1.0

            predicted_open = prev_close

            predicted_high = predicted_open + (range_moyen * multiplier * 0.75)

            predicted_low = predicted_open - (range_moyen * multiplier * 0.55)

            ohlc_forecast.append({

                "Date": dates_formattees[i],

                "Open": round(predicted_open,2),

                "High": round(predicted_high,2),

                "Low": round(predicted_low,2),

                "Close": round(predicted_close,2),

                "News": f"{high_impact} haute impact" if high_impact > 0 else "Aucune"

            })

            prev_close = predicted_close

        df_ohlc = pd.DataFrame(ohlc_forecast)

        # =========================
        # RESULTATS
        # =========================

        st.subheader("📊 Prévision OHLC (Stratégie Scalping)")

        st.dataframe(df_ohlc, use_container_width=True)

        st.caption("Analyse automatique ForexFactory + FxStreet + MQL5")

        # =========================
        # GRAPHIQUE
        # =========================

        st.subheader("🎨 Graphique")

        fig, ax = plt.subplots(figsize=(10,5))

        ax.plot(
            data['Date'].tail(60),
            data['Close'].tail(60),
            label="Historique",
            color="orange"
        )

        ax.plot(
            future_dates,
            forecast_close,
            label="Prévision",
            color="green",
            marker='o'
        )

        ax.legend()

        ax.grid(True)

        plt.xticks(rotation=45)

        st.pyplot(fig)
