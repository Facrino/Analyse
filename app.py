import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from datetime import date, timedelta
import warnings

warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(page_title="BTC Analyse", layout="centered")
st.title("📈 Prévision BTC/USD (ARIMA)")

# 1. Téléchargement des données
@st.cache_data
def load_data():
    end_date = date.today()
    start_date = end_date - timedelta(days=365)
    data = yf.download('BTC-USD', start=start_date, end=end_date, progress=False)
    data.reset_index(inplace=True)
    return data

data = load_data()
st.success("✅ Données actuelles téléchargées avec succès !")

# 2. Choix du nombre de jours par l'utilisateur
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
        future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]
        dates_formattees = [d.strftime('%d/%m/%Y') for d in future_dates]

        # Calculer la différence de prix jour par jour
        prix_liste = [last_price] + list(forecast)
        evolution = [prix_liste[i] - prix_liste[i-1] for i in range(1, len(prix_liste))]

        st.divider()

        # --- CARTES VISUELLES POUR LES 3 PREMIERS JOURS ---
        st.subheader("🔥 Focus sur les 3 prochains jours")
        cols = st.columns(min(3, forecast_days))
        for i in range(min(3, forecast_days)):
            cols[i].metric(
                label=dates_formattees[i],
                value=f"{forecast.iloc[i]:.0f} $",
                delta=f"{evolution[i]:.2f} $"
            )

        st.divider()

        # --- TABLEAU DÉTAILLÉ JOUR PAR JOUR ---
        st.subheader("📋 Tableau détaillé (Jour par jour)")

        df_result = pd.DataFrame({
            "Date": dates_formattees,
            "Prix Prévu (USD)": [round(p, 2) for p in forecast],
            "Évolution journalière": [round(e, 2) for e in evolution]
        })

        st.dataframe(df_result, use_container_width=True)

        st.divider()

        # -------------------------------------------------------
        # --- GRAPHIQUE CHANDELIER (CANDLESTICK) AVEC PLOTLY ---
        # -------------------------------------------------------
        st.subheader("🕯️ Graphique Chandelier BTC/USD")

        # On prend les 60 derniers jours pour le chandelier
        df_candles = data.tail(60).copy()

        # Aplatir les colonnes MultiIndex si nécessaire (bug courant avec yfinance)
        if isinstance(df_candles.columns, pd.MultiIndex):
            df_candles.columns = df_candles.columns.get_level_values(0)

        fig = go.Figure()

        # ── 1. Bougies chandelier historiques ──
        fig.add_trace(go.Candlestick(
            x=df_candles['Date'],
            open=df_candles['Open'],
            high=df_candles['High'],
            low=df_candles['Low'],
            close=df_candles['Close'],
            name='Historique (60j)',
            increasing_line_color='#26a69a',   # Vert bougie haussière
            decreasing_line_color='#ef5350',   # Rouge bougie baissière
            increasing_fillcolor='#26a69a',
            decreasing_fillcolor='#ef5350',
        ))

        # ── 2. Ligne de prévision ARIMA ──
        # On ajoute le dernier point connu pour raccorder la courbe
        forecast_x = [last_date] + future_dates
        forecast_y = [last_price] + list(forecast)

        fig.add_trace(go.Scatter(
            x=forecast_x,
            y=forecast_y,
            mode='lines+markers',
            name='Prévision ARIMA',
            line=dict(color='#FFD700', width=2, dash='dash'),
            marker=dict(size=7, color='#FFD700', symbol='circle'),
        ))

        # ── 3. Zone ombrée pour la période de prévision ──
        fig.add_vrect(
            x0=last_date,
            x1=future_dates[-1],
            fillcolor="rgba(255, 215, 0, 0.07)",
            layer="below",
            line_width=0,
            annotation_text="Zone prévision",
            annotation_position="top left",
            annotation_font_color="#FFD700",
        )

        # ── 4. Mise en forme du graphique ──
        fig.update_layout(
            title=dict(
                text=f'BTC/USD — Chandelier 60j + Prévision {forecast_days}j (ARIMA)',
                font=dict(size=16)
            ),
            xaxis_title='Date',
            yaxis_title='Prix en USD',
            xaxis_rangeslider_visible=False,
            template='plotly_dark',
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            ),
            height=550,
            plot_bgcolor='#1e1e2f',
            paper_bgcolor='#1e1e2f',
        )

        # ✅ CORRIGÉ : update_xaxes et update_yaxes (avec "s")
        fig.update_xaxes(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.08)',
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.08)',
            tickprefix='$',
        )

        st.plotly_chart(fig, use_container_width=True)
        
