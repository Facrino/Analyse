import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import date, timedelta
import warnings

warnings.filterwarnings('ignore')

st.title("📈 Prévision BTC/USD (ARIMA)")

# Téléchargement des données
@st.cache_data
def load_data():
    end_date = date.today()
    start_date = end_date - timedelta(days=365)
    data = yf.download('BTC-USD', start=start_date, end=end_date, progress=False)
    data.reset_index(inplace=True)
    return data

data = load_data()
data['MA_50'] = data['Close'].rolling(window=50).mean()

st.write("✅ Données téléchargées avec succès !")

# Lancement de l'IA (ARIMA)
forecast_days = 7
if st.button("Lancer les prévisions (7 jours)"):
    with st.spinner("Calcul en cours, veuillez patienter..."):
        df_train = data.set_index('Date')
        model = ARIMA(df_train['Close'], order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=forecast_days)
        
        last_date = pd.to_datetime(data['Date'].iloc[-1])
        future_dates =[last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]
        
        st.subheader("📊 Résultats :")
        for d, p in zip(future_dates, forecast):
            st.write(f"**{d.strftime('%Y-%m-%d')}** : {p:.2f} USD")
            
        # Affichage du graphique avec Streamlit
        st.subheader("🎨 Graphique d'analyse")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(data['Date'], data['Close'], label='Historique', color='orange')
        ax.plot(data['Date'], data['MA_50'], label='Moyenne 50j', color='blue', linestyle='--')
        ax.plot(future_dates, forecast, label='Prévision', color='green', marker='o')
        
        ax.set_title('Analyse BTC/USD')
        ax.set_xlabel('Date')
        ax.set_ylabel('Prix USD')
        ax.legend()
        ax.grid(True)
        
        # C'est CETTE ligne qui remplace plt.show() pour le web :
        st.pyplot(fig)
