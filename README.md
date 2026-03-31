# 🏠 House Price Prediction — Snowflake ML Pipeline

## MBAESG_[BIG DATA & IA]_[01]_EVALUATION_DATAENGINEER_MLOPS

---

## Auteur
- Thomas MARIE-ANNE
---

## Contexte
Pipeline complet de Data Engineering et Machine Learning déployé 
intégralement sur Snowflake. Les données immobilières sont ingérées 
depuis S3, transformées, utilisées pour entraîner des modèles ML, 
puis exposées via une application Streamlit interactive.

---

## 🏗️ Architecture du pipeline
```
S3 (JSON) → Snowflake Stage → COPY INTO → Table RAW
→ Snowpark EDA → Feature Engineering → Train/Test Split
→ Model Training (5 modèles) → Evaluation → Hyperparameter Tuning
→ Model Registry → Inference → Table PREDICTIONS → Streamlit App
```

---

## 📂 Dataset
- **Source** : s3://logbrain-datalake/datasets/house_price/
- **Fichier** : Housing_Price_Data.json
- **Lignes** : 545
- **Features** : 13
- **Variable cible** : PRICE (prix de vente en €)

---

## 🔧 Stack technique
| Composant | Technologie |
|-----------|-------------|
| Plateforme | Snowflake |
| Manipulation données | Snowpark Python |
| Notebooks | Snowflake Notebooks |
| ML | scikit-learn, XGBoost |
| Optimisation | GridSearchCV, RandomizedSearchCV |
| Registry | Snowflake Model Registry |
| Application | Streamlit in Snowflake |

---

## 📊 Résultats des modèles

| Modèle | RMSE | MAE | R² |
|--------|------|-----|----|
| Linear Regression | 53,980 | 40,236 | 0.6732 |
| Ridge | 53,980 | 40,236 | 0.6732 |
| Lasso | 53,980 | 40,236 | 0.6732 |
| Random Forest | 32,391 | 19,538 | 0.8823 |
| **XGBoost** ⭐ | **29,314** | **18,497** | **0.9036** |
| XGBoost Optimisé (Grid) | 29,903 | 17,777 | 0.8997 |
| Random Forest Optimisé | 33,096 | 23,081 | 0.8772 |

---

## 🏆 Meilleur modèle : XGBoost (base)

| Métrique | Valeur | Interprétation |
|----------|--------|----------------|
| **R²** | **0.9036** | Le modèle explique 90% de la variance des prix |
| **RMSE** | **29,314** | Erreur quadratique moyenne de ±29,314 |
| **MAE** | **18,497** | Écart absolu moyen de 18,497 |
| **MAPE** | ~12% | Erreur relative moyenne de 12% |
| Prédictions dans ±10% | 71.1% | 71% des estimations sont très précises |
| Prédictions dans ±20% | 89.9% | 90% des estimations sont acceptables |

### Hyperparamètres retenus
```
n_estimators  = 100
learning_rate = 0.1
random_state  = 42
```

---

## 📝 Note méthodologique — Métriques
Le sujet mentionne Accuracy, Precision et Recall. Ces métriques 
s'appliquent à la classification. Le problème est une 
**régression (prédiction d'un prix continu). J'ai donc
utilisé les métriques adaptées : RMSE, MAE et R².

---

## 🗂️ Livrables
| Fichier | Description |
|---------|-------------|
| `notebook_house_price.ipynb` | Pipeline ML complet Snowflake |
| `README.md` | Documentation et analyse des performances |

---

## ⚠️ Note technique
Le compte Snowflake Trial ne supporte pas l'External Access 
Integration (EAI). L'application Streamlit utilise donc une 
régression linéaire numpy native plutôt que le modèle XGBoost 
du Model Registry, qui reste accessible depuis le Notebook.
```
