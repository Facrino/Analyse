import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from datetime import date, timedelta
import warnings

warnings.filterwarnings('ignore')

# Configuration de l'écran pour le téléphone
st.set_page_config(page_title="BTC Analyse", layout="centered")
st.title("📈 Prévision BTC/USD")

# Téléchargement des données
@st.cache_data
def load_data():
    end = date.today()
    start = end - timedelta(days=365)
    data = yf.download('BTC-USD', start=start, end=end)
    data.reset_index(inplace=True)
    return data

with st.spinner('Chargement des données en direct...'):
    data = load_data()

# Graphique de l'historique
st.subheader('Historique du Bitcoin')
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Prix', line=dict(color='orange')))
fig.update_layout(xaxis_title='Date', yaxis_title='Prix (USD)', margin=dict(l=0, r=0, t=30, b=0))
st.plotly_chart(fig, use_container_width=True)

# Prévision IA (ARIMA)
st.subheader('🔮 Prévision IA (7 prochains jours)')
if st.button("Lancer l'IA (ARIMA)"):
    with st.spinner("Calculs mathématiques en cours..."):
        df_train = data.set_index('Date')
        model = ARIMA(df_train['Close'], order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=7)
        
        last_date = pd.to_datetime(data['Date'].iloc[-1])
        future_dates = [last_date + timedelta(days=i) for i in range(1, 8)]
        
        # Affichage du résultat
        for d, p in zip(future_dates, forecast):
            st.success(f"Le {d.strftime('%d/%m/%Y')} : {p:.2f} $")
            
st.caption("Application créée via Sketchware & Python")
