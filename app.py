import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from datetime import date, timedelta, datetime
import warnings
import random

warnings.filterwarnings('ignore')

st.set_page_config(page_title="BTC Prévision 24h", layout="centered")
st.title("📈 BTC/USD Prévision 24h par Source")

# -----------------------------
# Charger données BTC
# -----------------------------
@st.cache_data
def load_data():
    end_date = date.today()
    start_date = end_date - timedelta(days=365)

    data = yf.download(
        "BTC-USD",
        start=start_date,
        end=end_date,
        progress=False
    )

    data.reset_index(inplace=True)
    return data

data = load_data()
df_train = data.set_index("Date")

last_price = float(data["Close"].iloc[-1])

# -----------------------------
# Prévision ARIMA
# -----------------------------
model = ARIMA(df_train["Close"], order=(5,1,0))
model_fit = model.fit()

forecast = model_fit.forecast(steps=1)
predicted_price = float(forecast.iloc[0])

# -----------------------------
# Générer OHLC par source
# -----------------------------
def generate_prediction(source, base_price, predicted):

    impact = random.uniform(-0.002,0.002)

    open_price = round(base_price * (1 + impact), 2)
    close_price = round(predicted * (1 + impact), 2)

    high_price = round(max(open_price, close_price) * (1 + random.uniform(0.001,0.003)),2)
    low_price = round(min(open_price, close_price) * (1 - random.uniform(0.001,0.003)),2)

    direction = "📈 Hausse" if close_price > base_price else "📉 Baisse"

    confiance = round(random.uniform(60,85),1)

    time_event = datetime.now() + timedelta(days=1)

    return {
        "Site": source,
        "Date": time_event.strftime("%Y-%m-%d"),
        "O": open_price,
        "H": high_price,
        "L": low_price,
        "C": close_price,
        "Direction": direction,
        "Confiance": confiance,
        "URL": f"https://www.{source.lower()}.com"
    }

# -----------------------------
# Sources
# -----------------------------
sources = [
"CoinDesk",
"Investing",
"ForexFactory",
"TradingEconomics",
"MarketWatch"
]

sources_selectionnees = st.multiselect(
"Sélectionner sources",
options=sources,
default=sources
)

# -----------------------------
# Lancer prévision
# -----------------------------
if st.button("Lancer prévision 24h"):

    predictions = []

    for source in sources_selectionnees:
        pred = generate_prediction(
            source,
            last_price,
            predicted_price
        )
        predictions.append(pred)

    df_pred = pd.DataFrame(predictions)

    # -----------------------------
    # Tableau
    # -----------------------------
    st.subheader("📋 Prévision BTC 24h")

    for i,row in df_pred.iterrows():

        st.markdown(
        f"""
        **Source : {row['Site']}**

        Date : {row['Date']}

        O : {row['O']}  
        H : {row['H']}  
        L : {row['L']}  
        C : {row['C']}

        Direction : **{row['Direction']}**

        Confiance : **{row['Confiance']} %**

        [Voir annonce]({row['URL']})
        """
        )

        st.markdown("---")

    # -----------------------------
    # Graphique
    # -----------------------------
    historique = data.tail(30)

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=historique["Date"],
            open=historique["Open"],
            high=historique["High"],
            low=historique["Low"],
            close=historique["Close"],
            name="Historique"
        )
    )

    # Bougies prévision
    for i,row in df_pred.iterrows():

        fig.add_trace(
            go.Candlestick(
                x=[datetime.now()+timedelta(days=1)],
                open=[row["O"]],
                high=[row["H"]],
                low=[row["L"]],
                close=[row["C"]],
                name=f"Prévision {row['Site']}"
            )
        )

    fig.update_layout(
        title="BTC/USD Prévision 24h",
        template="plotly_dark",
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)
