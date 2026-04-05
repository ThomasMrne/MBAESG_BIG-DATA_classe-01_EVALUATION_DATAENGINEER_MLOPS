import streamlit as st
from snowflake.snowpark.context import get_active_session
import pandas as pd
import numpy as np

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")
st.title("🏠 Estimateur de Prix Immobilier")
st.markdown("*Propulsé par la régression entraînée sur Snowflake*")
st.divider()

session = get_active_session()

@st.cache_resource
def build_model():
    df = session.table("HOUSE_PRICE_DB.ML.HOUSE_PRICES_RAW").to_pandas()

    # Nettoyage et encodage
    binary_cols = ['MAINROAD','GUESTROOM','BASEMENT','HOTWATERHEATING',
                   'AIRCONDITIONING','PREFAREA']
    for col in binary_cols + ['FURNISHINGSTATUS']:
        df[col] = df[col].astype(str).str.lower().str.strip()
    for col in binary_cols:
        df[col] = df[col].map({'yes': 1, 'no': 0})
    df['FURNISHINGSTATUS'] = df['FURNISHINGSTATUS'].map(
        {'furnished': 2, 'semi-furnished': 1, 'unfurnished': 0}
    )

    cols     = list(df.drop(columns=['PRICE']).columns)
    X        = df.drop(columns=['PRICE']).values.astype(float)
    y        = df['PRICE'].values.astype(float)

    # Normalisation manuelle (sans sklearn)
    num_idx  = [cols.index(c) for c in ['AREA','BEDROOMS','BATHROOMS','STORIES','PARKING']]
    means    = X[:, num_idx].mean(axis=0)
    stds     = X[:, num_idx].std(axis=0) + 1e-8
    X_scaled = X.copy()
    X_scaled[:, num_idx] = (X[:, num_idx] - means) / stds

    # Régression linéaire — moindres carrés numpy uniquement
    X_b   = np.c_[np.ones(len(X_scaled)), X_scaled]
    theta = np.linalg.lstsq(X_b, y, rcond=None)[0]

    return theta, means, stds, num_idx, cols

try:
    theta, means, stds, num_idx, cols = build_model()
    st.success("✅ Modèle chargé et prêt")
except Exception as e:
    st.error(f"❌ Erreur chargement : {e}")
    st.stop()

# ── Formulaire ─────────────────────────────────────────────
st.subheader("Caractéristiques de la maison")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**📐 Surface & Structure**")
    area      = st.number_input("Surface (m²)", min_value=100, max_value=20000, value=3000, step=100)
    bedrooms  = st.slider("Chambres",        min_value=1, max_value=6, value=3)
    bathrooms = st.slider("Salles de bain",  min_value=1, max_value=4, value=1)
    stories   = st.slider("Étages",          min_value=1, max_value=4, value=2)
    parking   = st.slider("Parking",         min_value=0, max_value=3, value=1)

with col2:
    st.markdown("**🏡 Équipements**")
    mainroad        = st.checkbox("Route principale",     value=True)
    guestroom       = st.checkbox("Chambre d'amis",       value=False)
    basement        = st.checkbox("Sous-sol",             value=False)
    hotwaterheating = st.checkbox("Chauffage eau chaude", value=False)
    airconditioning = st.checkbox("Climatisation",        value=False)
    prefarea        = st.checkbox("Zone privilégiée",     value=False)

with col3:
    st.markdown("**🛋️ Ameublement**")
    furnishing_label = st.radio(
        "État", ["Meublée", "Semi-meublée", "Non meublée"], index=1
    )
    furnishing = {"Meublée": 2, "Semi-meublée": 1, "Non meublée": 0}[furnishing_label]
    st.markdown("---")
    st.markdown(f"- **{area} m²** · {bedrooms} ch. · {bathrooms} sdb")
    st.markdown(f"- {stories} étage(s) · {parking} parking(s)")

# ── Prédiction ─────────────────────────────────────────────
st.divider()
col_btn, col_res = st.columns([1, 2])

with col_btn:
    predict = st.button("💰 Estimer le prix", type="primary", use_container_width=True)

if predict:
    # Construction du vecteur d'entrée 1D
    inp = np.array([
        area, bedrooms, bathrooms, stories,
        int(mainroad), int(guestroom), int(basement),
        int(hotwaterheating), int(airconditioning),
        parking, int(prefarea), furnishing
    ], dtype=float)

    # Normalisation
    inp[num_idx] = (inp[num_idx] - means) / stds

    # Prédiction avec np.dot — garantit un scalaire
    inp_b = np.concatenate([[1.0], inp])
    prix  = float(np.dot(inp_b, theta))

    with col_res:
        st.success("✅ Estimation réalisée !")
        m1, m2, m3 = st.columns(3)
        m1.metric("💵 Prix estimé", f"{max(prix, 0):,.0f} €")
        m2.metric("📏 Prix au m²",  f"{max(prix, 0)/area:,.0f} €/m²")
        m3.metric("🎯 Modèle",      "Régression Linéaire")

# ── Historique ─────────────────────────────────────────────
st.divider()
with st.expander("📊 Historique des prédictions enregistrées"):
    try:
        hist = session.sql("""
            SELECT PRIX_REEL, PRIX_PREDIT, ECART_PCT,
                CASE WHEN ECART_PCT < 10 THEN '✅ Excellent'
                     WHEN ECART_PCT < 20 THEN '🟡 Bon'
                     ELSE '🔴 À améliorer' END AS QUALITE
            FROM HOUSE_PRICE_DB.ML.HOUSE_PRICES_PREDICTIONS
            ORDER BY ECART_PCT ASC LIMIT 20
        """).to_pandas()
        st.dataframe(hist, use_container_width=True)
    except Exception as e:
        st.warning(f"Historique indisponible : {e}")

# ── Footer ──────────────────────────────────────────────────
st.divider()
st.caption("Pipeline ML complet — Snowflake · Snowpark · Régression Linéaire · Streamlit")
