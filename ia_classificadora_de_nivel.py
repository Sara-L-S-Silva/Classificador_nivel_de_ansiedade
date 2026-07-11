import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Pré-processamento de Dados
# Carregamento da base de dados
df = pd.read_csv('enhanced_anxiety_dataset.csv')

# Mapeamento de variáveis binárias
binary_mapping = {'Yes': 1, 'No': 0}
binary_columns = ['Smoking', 'Family History of Anxiety',
                  'Dizziness', 'Medication', 'Recent Major Life Event']
for col in binary_columns:
    df[col] = df[col].map(binary_mapping)

# One-Hot Encoding para variáveis nominais
df = pd.get_dummies(df, columns=['Gender', 'Occupation'], drop_first=True)

# Definição dos recursos (X) e do alvo (y)
X = df.drop('Anxiety Level', axis=1)
y = df['Anxiety Level']

# 2. Divisão dos Dados (Holdout 80/20)
# A semente (random_state=42) garante a reprodutibilidade do experimento
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)

# 3. Definição dos Modelos e Validação Cruzada (K-Fold)
# O parâmetro n_jobs=-1 acelera o processamento utilizando todos os núcleos do processador
modelos = {
    "Regressão Linear Múltipla": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("--- Validação Cruzada (Dados de Treino) ---")
for nome, modelo in modelos.items():
    # Calculando MSE negativo e convertendo para positivo
    scores = cross_val_score(modelo, X_train, y_train,
                             cv=kf, scoring='neg_mean_squared_error')
    mse_cv = -scores.mean()
    print(f"{nome} - MSE Médio no K-Fold: {mse_cv:.4f}")

# 4. Treinamento Final e Avaliação na Base de Teste (20% intocados)
print("\n--- Avaliação Final (Dados de Teste) ---")
modelos_treinados = {}

for nome, modelo in modelos.items():
    # Treinando o modelo com todos os dados de treino
    modelo.fit(X_train, y_train)
    modelos_treinados[nome] = modelo

    # Prevendo nos dados de teste
    previsoes = modelo.predict(X_test)

    # Calculando métricas
    mae = mean_absolute_error(y_test, previsoes)
    mse = mean_squared_error(y_test, previsoes)

    print(f"{nome}:")
    print(f"  MAE: {mae:.4f}")
    print(f"  MSE: {mse:.4f}\n")

# 5. Geração de Gráfico para o Relatório (Feature Importance)
# Extraindo a importância do modelo Random Forest
rf_model = modelos_treinados["Random Forest Regressor"]
importancias = rf_model.feature_importances_
indices = np.argsort(importancias)[-10:]  # Top 10 variáveis

plt.figure(figsize=(10, 6))
plt.title('Importância das Variáveis (Random Forest)')
plt.barh(range(len(indices)), importancias[indices], color='b', align='center')
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.xlabel('Importância Relativa')
plt.tight_layout()
plt.show()
