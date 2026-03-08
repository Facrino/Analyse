import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import date, timedelta
import warnings

# Ignorer les avertissements
warnings.filterwarnings('ignore')

print("📊 Téléchargement des données BTC-USD en cours...")

# Définir les dates (1 an en arrière)
end_date = date.today()
start_date = end_date - timedelta(days=365)

# Télécharger les données depuis Yahoo Finance
data = yf.download('BTC-USD', start=start_date, end=end_date, progress=False)
data.reset_index(inplace=True)
print("✅ Données téléchargées avec succès !")

# Calcul de la moyenne mobile sur 50 jours
data['MA_50'] = data['Close'].rolling(window=50).mean()

# --- MODÈLE ARIMA (PRÉVISION) ---
forecast_days = 7
print(f"🔮 Calcul des prévisions pour les {forecast_days} prochains jours (Veuillez patienter...)")

# Préparation des données pour l'entraînement
df_train = data.set_index('Date')
model = ARIMA(df_train['Close'], order=(5, 1, 0))
model_fit = model.fit()

# Générer la prévision
forecast = model_fit.forecast(steps=forecast_days)

# Créer les dates futures
last_date = pd.to_datetime(data['Date'].iloc[-1])
future_dates =[last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]

# Afficher les prix prédits dans la console
print("\n📈 PRÉVISIONS BTC/USD :")
for d, p in zip(future_dates, forecast):
    print(f"{d.strftime('%Y-%m-%d')} : {p:.2f} USD")

print("\n🎨 Génération du graphique...")

# --- AFFICHAGE DU GRAPHIQUE ---
plt.figure(figsize=(10, 6))

# Tracer l'historique
plt.plot(data['Date'], data['Close'], label='Historique (Prix)', color='orange')

# Tracer la moyenne mobile
plt.plot(data['Date'], data['MA_50'], label='Moyenne Mobile (50j)', color='blue', linestyle='--')

# Tracer la prévision
plt.plot(future_dates, forecast, label='Prévision (ARIMA)', color='green', marker='o')

# Mise en forme du graphique
plt.title('Analyse et Prévisions du Bitcoin (BTC/USD)')
plt.xlabel('Date')
plt.ylabel('Prix en USD')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# Afficher le graphique sur l'écran
plt.show()
