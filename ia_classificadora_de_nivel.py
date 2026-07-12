import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Pré-processamento
df = pd.read_csv('enhanced_anxiety_dataset.csv')

binary_mapping = {'Yes': 1, 'No': 0}
binary_columns = ['Smoking', 'Family History of Anxiety',
                  'Dizziness', 'Medication', 'Recent Major Life Event']
for col in binary_columns:
    if col in df.columns:
        df[col] = df[col].map(binary_mapping)

df = pd.get_dummies(df, columns=['Gender', 'Occupation'], drop_first=True)

X = df.drop('Anxiety Level (1-10)', axis=1)
y = df['Anxiety Level (1-10)']

# 2. Divisão Holdout
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)

# 3. Treinamento
modelos = {
    "Regressão Linear Múltipla": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
}

kf = KFold(n_splits=10, shuffle=True, random_state=42)

print("--- Validação Cruzada K-Fold (Apenas Treino 80%) ---")
for nome, modelo in modelos.items():
    # Avaliação de robustez interna no treino
    mse_scores = cross_val_score(
        modelo, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
    mae_scores = cross_val_score(
        modelo, X_train, y_train, cv=kf, scoring='neg_mean_absolute_error')

    print(f"{nome}:")
    print(f"  MSE Médio Interno: {-mse_scores.mean():.4f}")
    print(f"  MAE Médio Interno: {-mae_scores.mean():.4f}\n")

# Avaliação Final: Cálculo exato do erro na escala 1-10 usando os 20% intocados
print("--- Avaliação Final (Dados de Teste 20% Intocados) ---")
for nome, modelo in modelos.items():
    # Treinamento definitivo com a totalidade dos 80% de treino
    modelo.fit(X_train, y_train)

    # Previsão sobre os dados isolados para evitar data leakage
    previsoes = modelo.predict(X_test)

    # Métricas de precisão numérica
    mae = mean_absolute_error(y_test, previsoes)
    mse = mean_squared_error(y_test, previsoes)

    print(f"{nome}:")
    print(f"  MAE Final: {mae:.4f}")
    print(f"  MSE Final: {mse:.4f}\n")

# 4. Exportação do Modelo Principal e dos Dados
rf_model = modelos["Random Forest Regressor"]
joblib.dump(rf_model, 'modelo_rf_ansiedade.pkl')
joblib.dump((X_train, X_test, y_train, y_test), 'dados_holdout.pkl')

print("\nArquivos 'modelo_rf_ansiedade.pkl' e 'dados_holdout.pkl' salvos com sucesso.")
