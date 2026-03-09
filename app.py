import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import date, timedelta
import warnings

warnings.filterwarnings('ignore')

Configuration de la page

st.set_page_config(page_title="BTC Analyse", layout="centered")
st.title("📈 Prévision BTC/USD (ARIMA)")

1. Téléchargement des données

@st.cache_data
def load_data():
end_date = date.today()
start_date = end_date - timedelta(days=365)
data = yf.download('BTC-USD', start=start_date, end=end_date, progress=False)
data.reset_index(inplace=True)
return data

data = load_data()
st.success("✅ Données actuelles téléchargées avec succès !")

2. Choix du nombre de jours par l'utilisateur

forecast_days = st.slider("🗓️ Combien de jours voulez-vous prédire ?", min_value=1, max_value=14, value=7)

if st.button(f"Lancer les prévisions pour {forecast_days} jours"):
with st.spinner("L'IA calcule les prévisions mathématiques..."):

# 3. Entraînement du modèle ARIMA  
    df_train = data.set_index('Date')  
    model = ARIMA(df_train['Close'], order=(5, 1, 0))  
    model_fit = model.fit()  
    forecast = model_fit.forecast(steps=forecast_days)  
      
    # Récupérer la dernière date et le dernier prix connu  
    last_date = pd.to_datetime(data['Date'].iloc[-1])  
    last_price = float(data['Close'].iloc[-1])  
      
    # Créer la liste des dates futures  
    future_dates =[last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]  
    dates_formattees =[d.strftime('%d/%m/%Y') for d in future_dates]  
      
    # Calculer la différence de prix jour par jour  
    prix_liste = [last_price] + list(forecast)  
    evolution = [prix_liste[i] - prix_liste[i-1] for i in range(1, len(prix_liste))]  
      
    st.divider()  
      
    # --- NOUVEAUTÉ 1 : CARTES VISUELLES POUR LES 3 PREMIERS JOURS ---  
    st.subheader("🔥 Focus sur les 3 prochains jours")  
    cols = st.columns(min(3, forecast_days))  
    for i in range(min(3, forecast_days)):  
        cols[i].metric(  
            label=dates_formattees[i],  
            value=f"{forecast.iloc[i]:.0f} $",  
            delta=f"{evolution[i]:.2f} $" # Flèche verte si positif, rouge si négatif  
        )  
          
    st.divider()  

    # --- NOUVEAUTÉ 2 : TABLEAU DÉTAILLÉ JOUR PAR JOUR ---  
    st.subheader("📋 Tableau détaillé (Jour par jour)")  
      
    # Création d'un tableau propre avec Pandas  
    df_result = pd.DataFrame({  
        "Date": dates_formattees,  
        "Prix Prévu (USD)":[round(p, 2) for p in forecast],  
        "Évolution journalière":[round(e, 2) for e in evolution]  
    })  
      
    # Affichage du tableau dans Streamlit  
    st.dataframe(df_result, use_container_width=True)  
      
    # --- NOUVEAUTÉ 3 : GRAPHIQUE AMÉLIORÉ (Zoom sur la fin) ---  
    st.subheader("🎨 Graphique d'évolution")  
    fig, ax = plt.subplots(figsize=(10, 5))  
      
    # On n'affiche que les 60 derniers jours pour mieux voir la prévision  
    ax.plot(data['Date'].tail(60), data['Close'].tail(60), label='Historique récent (60j)', color='orange', linewidth=2)  
    ax.plot(future_dates, forecast, label='Prévision IA', color='green', marker='o', linestyle='dashed', linewidth=2)  
      
    ax.set_title(f'Prévision BTC/USD sur {forecast_days} jours')  
    ax.set_ylabel('Prix en USD')  
    ax.legend()  
    ax.grid(True, linestyle=':', alpha=0.7)  
    plt.xticks(rotation=45)  
      
    st.pyplot(fig)
