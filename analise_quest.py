import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import shap

# 1. Carregamento do Modelo e dos Dados
rf_model = joblib.load('modelo_rf_ansiedade.pkl')
X_train, X_test, y_train, y_test = joblib.load('dados_holdout.pkl')

# --- Questão 1: Interação Sono x Estresse (PDP) ---
fig, ax = plt.subplots(figsize=(8, 6))
# Verifique o nome exato da coluna na sua base. Caso possua sufixo, altere abaixo.
features = [('Sleep Hours', 'Stress Level (1-10)')]
PartialDependenceDisplay.from_estimator(rf_model, X_train, features, ax=ax)
plt.title("PDP 2D: Interação entre Horas de Sono e Nível de Estresse")
plt.tight_layout()
plt.show()

# --- Questão 2 e Geral: Direção do Impacto (SHAP) ---
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, show=False)
plt.title("Impacto e Direção das Variáveis (SHAP)")
plt.tight_layout()
plt.show()

# --- Questão Geral: Peso dos Pilares ---
importancias = rf_model.feature_importances_
cols = X_train.columns

pilares = {
    "Demográfico/Ocupacional": [c for c in cols if any(x in c for x in ["Age", "Gender", "Occupation"])],
    "Estilo de Vida": [c for c in cols if any(x in c for x in ["Sleep", "Physical Activity", "Diet", "Caffeine", "Alcohol", "Smoking"])],
    "Biométrico/Saúde": [c for c in cols if any(x in c for x in ["Heart Rate", "Breathing", "Sweating", "Dizziness", "Stress"])],
    "Clínico/Histórico": [c for c in cols if any(x in c for x in ["Family History", "Medication", "Recent", "Therapy"])]
}

pesos_pilares = {}
for pilar, colunas in pilares.items():
    peso = sum([importancias[list(cols).index(col)] for col in colunas])
    pesos_pilares[pilar] = peso

plt.figure(figsize=(8, 5))
plt.bar(pesos_pilares.keys(), pesos_pilares.values(),
        color=['#4C72B0', '#55A868', '#C44E52', '#8172B2'])
plt.title("Peso Preditivo de Cada Pilar")
plt.ylabel("Soma da Importância")
plt.tight_layout()
plt.show()

# --- Questão 3: Modelo Restrito (Apenas Físicos e Clínicos) ---
colunas_restritas = pilares["Biométrico/Saúde"] + pilares["Clínico/Histórico"]
X_train_restrito = X_train[colunas_restritas]
X_test_restrito = X_test[colunas_restritas]

rf_restrito = RandomForestRegressor(
    n_estimators=100, random_state=42, n_jobs=-1)
rf_restrito.fit(X_train_restrito, y_train)
prev_restrito = rf_restrito.predict(X_test_restrito)

print("\n--- Desempenho do Modelo Completo vs Restrito ---")
print("Modelo Completo:")
print(f"  MAE: {mean_absolute_error(y_test, rf_model.predict(X_test)):.4f}")
print(f"  MSE: {mean_squared_error(y_test, rf_model.predict(X_test)):.4f}")

print("\nModelo Restrito (Sem Estilo de Vida e Demografia):")
print(f"  MAE: {mean_absolute_error(y_test, prev_restrito):.4f}")
print(f"  MSE: {mean_squared_error(y_test, prev_restrito):.4f}")
