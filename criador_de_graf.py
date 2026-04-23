import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# =================================================================
# PASSO 1: CARREGAMENTO E CURADORIA (SEMANA 1)
# =================================================================
# Carregamos o dataset completo
df = pd.read_csv('enhanced_anxiety_dataset.csv')

# Verificamos e removemos valores nulos do DataFrame inteiro
nulos_encontrados = df.isnull().sum().sum()
df_limpo = df.dropna().copy()

# =================================================================
# PASSO 2: CODIFICAÇÃO DE DADOS (SEMANA 2)
# =================================================================
# 1. Transformar variáveis binárias (Yes/No) em 1 e 0 para toda a base
cols_binarias = ['Smoking', 'Family History of Anxiety', 'Dizziness',
                 'Medication', 'Recent Major Life Event']

for col in cols_binarias:
    df_limpo[col] = df_limpo[col].map({'Yes': 1, 'No': 0})

# 2. Transformar variáveis nominais em numéricas (One-Hot Encoding)
# Aplicamos em todo o dataframe limpo antes da amostragem
df_organizado = pd.get_dummies(df_limpo, columns=['Gender', 'Occupation'])

# =================================================================
# PASSO 3: EXPLORAÇÃO E ESTATÍSTICAS (SEMANA 3)
# =================================================================
# Agora que o DF está organizado, extraímos a subamostra de 10%
subamostra = df_organizado.sample(frac=0.1, random_state=42)

# Gerar estatísticas descritivas da subamostra
estatisticas = subamostra.describe()

# =================================================================
# PASSO 4: VISUALIZAÇÃO DE DADOS (SEMANA 4)
# =================================================================

# Configuramos o estilo global para todos os gráficos
sns.set_theme(style="whitegrid")

# --- GRÁFICO 1: DISTRIBUIÇÃO DA ANSIEDADE (VARIÁVEL ALVO) ---
plt.figure(figsize=(10, 5))
sns.histplot(data=subamostra, x='Anxiety Level (1-10)',
             discrete=True, shrink=0.8, kde=True, color='teal')
plt.title('Distribuição dos Níveis de Ansiedade na Subamostra')
plt.xlabel('Nível de Ansiedade (1-10)')
plt.ylabel('Contagem de Pessoas')
plt.xticks(range(1, 11))  # Garante que todos os números apareçam no eixo X
plt.show()

# --- GRÁFICO 2: QUALIDADE DA DIETA VS ANSIEDADE ---
plt.figure(figsize=(10, 5))
sns.boxplot(data=subamostra, x='Diet Quality (1-10)',
            y='Anxiety Level (1-10)', palette='viridis')
plt.title('Relação entre Qualidade da Dieta e Nível de Ansiedade')
plt.xlabel('Qualidade da Dieta (1-10)')
plt.ylabel('Nível de Ansiedade')
plt.show()

# --- GRÁFICO 3: HORAS DE SONO VS ANSIEDADE (DISPERSÃO) ---
plt.figure(figsize=(10, 5))
sns.scatterplot(data=subamostra, x='Sleep Hours',
                y='Anxiety Level (1-10)', alpha=0.6)
plt.title('Tendência: Horas de Sono vs Nível de Ansiedade')
plt.xlabel('Horas de Sono')
plt.ylabel('Nível de Ansiedade')
plt.show()

# --- GRÁFICO 4: IMPACTO DO HISTÓRICO FAMILIAR ---
plt.figure(figsize=(8, 5))
# Nota: Usamos a variável original de texto para o rótulo ficar bonito
sns.boxplot(data=subamostra, x='Family History of Anxiety',
            y='Anxiety Level (1-10)')
plt.title('Diferença de Ansiedade por Histórico Familiar')
plt.xlabel('Tem Histórico Familiar de Ansiedade?')
plt.ylabel('Nível de Ansiedade')
plt.show()

# --- RESULTADOS PARA CONFERÊNCIA ---
print(f"--- Relatório de Execução ---")
print(f"Valores nulos removidos: {nulos_encontrados}")
print(f"Registros na base organizada: {len(df_organizado)}")
print(f"Registros na subamostra (10%): {len(subamostra)}")
print("\n--- Estatísticas (Variáveis Principais) ---")
print(estatisticas[['Age', 'Sleep Hours',
      'Anxiety Level (1-10)']].loc[['mean', 'std', 'max']])
