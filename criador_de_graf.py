import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import math
from matplotlib.ticker import MaxNLocator

# =================================================================
# PASSO 0: CONFIGURAÇÃO DA ESTRUTURA DE PASTAS
# =================================================================
pasta_mae = "resultados_completos_ep2"
pasta_comparacoes = os.path.join(pasta_mae, "1_comparacoes_total_vs_amostra")
pasta_relacoes = os.path.join(pasta_mae, "2_relacoes_e_dispersao_boxplot")
pasta_estatisticas = os.path.join(pasta_mae, "3_estatisticas")

for pasta in [pasta_comparacoes, pasta_relacoes, pasta_estatisticas]:
    os.makedirs(pasta, exist_ok=True)

# Carregamento dos dados brutos
df_total = pd.read_csv('enhanced_anxiety_dataset.csv')
# Extração isolada da subamostra de 10% (Garantia contra vazamento de dados)
subamostra = df_total.sample(frac=0.1, random_state=42).copy()

caminho_subamostra = os.path.join(pasta_mae, "subamostra_10_porcento_ep2.csv")
subamostra.to_csv(caminho_subamostra, index=False, encoding='utf-8')
print(
    f"[OK] Subamostra de {len(subamostra)} registros salva em: {caminho_subamostra}")

print("Iniciando o processamento estatístico e visual do EP2...")
sns.set_theme(style="whitegrid")

# Dicionário de apoio para mapear os tipos conforme o planejamento do EP1
mapeamento_tipos = {
    'Age': 'Quantitativa', 'Sleep Hours': 'Quantitativa', 'Physical Activity (hrs/week)': 'Quantitativa',
    'Caffeine Intake (mg/day)': 'Quantitativa', 'Alcohol Consumption (drinks/week)': 'Quantitativa',
    'Heart Rate (bpm)': 'Quantitativa', 'Breathing Rate (breaths/min)': 'Quantitativa',
    'Therapy Sessions (per month)': 'Quantitativa', 'Anxiety Level (1-10)': 'Quantitativa',
    'Stress Level (1-10)': 'Qualitativa Ordinal', 'Sweating Level (1-5)': 'Qualitativa Ordinal',
    'Diet Quality (1-10)': 'Qualitativa Ordinal', 'Gender': 'Qualitativa Nominal',
    'Occupation': 'Qualitativa Nominal', 'Smoking': 'Qualitativa Nominal',
    'Family History of Anxiety': 'Qualitativa Nominal', 'Dizziness': 'Qualitativa Nominal',
    'Medication': 'Qualitativa Nominal', 'Recent Major Life Event': 'Qualitativa Nominal'
}

# =================================================================
# PASSO 1: GERAÇÃO DOS GRÁFICOS COMPARATIVOS (ALINHADOS E CUSTOMIZADOS)
# =================================================================
sns.set_theme(style="whitegrid")

for col in df_total.columns:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- VARIÁVEIS QUANTITATIVAS (HISTOGRAMAS) ---
    if pd.api.types.is_numeric_dtype(df_total[col]) and df_total[col].nunique() > 10:

        # Pega o mínimo real da variável
        min_global = math.floor(df_total[col].min())
        max_absoluto = df_total[col].max()  # Pegamos o máximo real

        largura_salto = 1
        if col == 'Caffeine Intake (mg/day)':
            largura_salto = 25
        elif col == 'Heart Rate (bpm)':
            largura_salto = 5
        elif col == 'Age':
            largura_salto = 5

        # A MÁGICA DA CORREÇÃO:
        # Calcula qual deve ser o fim do gráfico para que a última barra caiba inteira.
        # Ele vê a distância total (ex: 64 - 18 = 46), divide pelo salto de 5 (9.2),
        # arredonda pra cima (10 blocos de 5) e soma no mínimo. Resultado: 68!
        quantidade_blocos = math.ceil(
            (max_absoluto - min_global) / largura_salto)
        max_global = min_global + (quantidade_blocos * largura_salto)

        # O resto do código continua igual, mas usando o novo max_global perfeito
        sns.histplot(data=df_total, x=col, kde=True, color='royalblue', ax=axes[0],
                     binwidth=largura_salto, binrange=(min_global, max_global))
        axes[0].set_xlim(min_global, max_global)

        if col in ['Age', 'Heart Rate (bpm)', 'Caffeine Intake (mg/day)']:
            marcacoes = list(range(min_global, max_global +
                             largura_salto, largura_salto))
            axes[0].set_xticks(marcacoes)
            if col == 'Caffeine Intake (mg/day)':
                axes[0].tick_params(axis='x', rotation=45)
            else:
                axes[0].tick_params(axis='x', rotation=0)

        axes[0].set_title(f'TOTAL (100%): Distribuição de {col}')
        axes[0].set_ylabel('Frequência Absoluta')

        sns.histplot(data=subamostra, x=col, kde=True, color='darkorange', ax=axes[1],
                     binwidth=largura_salto, binrange=(min_global, max_global))
        axes[1].set_xlim(min_global, max_global)

        if col in ['Age', 'Heart Rate (bpm)', 'Caffeine Intake (mg/day)']:
            axes[1].set_xticks(marcacoes)
            if col == 'Caffeine Intake (mg/day)':
                axes[1].tick_params(axis='x', rotation=45)
            else:
                axes[1].tick_params(axis='x', rotation=0)

        axes[1].set_title(f'SUBAMOSTRA (10%): Distribuição de {col}')
        axes[1].set_ylabel('Frequência Absoluta')

        if col == 'Alcohol Consumption (drinks/week)':
            axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
            axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))

    # --- VARIÁVEIS QUALITATIVAS/ORDINAIS (BARRAS) ---
    else:
        ordem_categorias = sorted(df_total[col].dropna().unique())

        sns.countplot(data=df_total, x=col, color='royalblue',
                      ax=axes[0], order=ordem_categorias)
        axes[0].set_title(f'TOTAL (100%): Contagem de {col}')

        rotacao_qualitativa = 45 if col == 'Occupation' else 0

        axes[0].tick_params(axis='x', rotation=rotacao_qualitativa)
        axes[0].set_ylabel('Quantidade')

        sns.countplot(data=subamostra, x=col, color='darkorange',
                      ax=axes[1], order=ordem_categorias)
        axes[1].set_title(f'SUBAMOSTRA (10%): Contagem de {col}')
        axes[1].tick_params(axis='x', rotation=rotacao_qualitativa)
        axes[1].set_ylabel('Quantidade')

    plt.tight_layout()

    nome_arquivo = f"comparacao_{col.replace('/', '_')}.png"
    plt.savefig(os.path.join(pasta_comparacoes, nome_arquivo),
                bbox_inches='tight')
    plt.close()

# =================================================================
# PASSO 2: GERAÇÃO DE GRÁFICOS DE RELAÇÃO E DISPERSÃO (BOXPLOTS)
# =================================================================
print("-> Gerando Boxplots de relação e dispersão com a Ansiedade...")
for col in subamostra.columns:
    if col == 'Anxiety Level (1-10)':
        continue

    plt.figure(figsize=(10, 6))
    tipo = mapeamento_tipos.get(col, 'Quantitativa')

    if tipo == 'Quantitativa':
        # Eixo X fixo com os níveis discretos da Ansiedade para mapear a dispersão do indicador clínico
        sns.boxplot(data=subamostra, x='Anxiety Level (1-10)',
                    y=col, palette='plasma')
        plt.title(f'Dispersão de {col} através dos Níveis de Ansiedade')
        plt.xlabel('Nível de Ansiedade (1-10)')
        plt.ylabel(col)
    else:
        # Para Qualitativos, avaliamos a dispersão da ansiedade dentro de cada grupo categórico/ordinal
        ordem = sorted(subamostra[col].dropna().unique(
        )) if tipo == 'Qualitativa Ordinal' else None
        sns.boxplot(data=subamostra, x=col, y='Anxiety Level (1-10)',
                    order=ordem, palette='viridis')
        plt.title(f'Impacto de {col} na Intensidade de Ansiedade')

        rotacao_qualitativa = 45 if col == 'Occupation' else 0
        plt.xlabel(col)
        plt.ylabel('Nível de Ansiedade (1-10)')
        plt.xticks(rotation=rotacao_qualitativa)

    nome_arquv_rel = f"relacao_ansiedade_{col.replace('/', '_')}.png"
    plt.savefig(os.path.join(pasta_relacoes, nome_arquv_rel),
                bbox_inches='tight')
    plt.close()

# =================================================================
# PASSO 3: RELATÓRIO METODOLÓGICO DE TENDÊNCIA CENTRAL (MÉDIA, MEDIANA, MODA)
# =================================================================
print("-> Calculando métricas de tendência central da subamostra...")
registro_estatisticas = []

for col in subamostra.columns:
    tipo = mapeamento_tipos.get(col, 'Quantitativa')

    # Cálculo da Moda (Aplicável a absolutamente todos os tipos)
    moda_calc = subamostra[col].mode()
    moda_val = moda_calc.iloc[0] if not moda_calc.empty else "N/A"

    if tipo == 'Quantitativa':
        media_val = round(subamostra[col].mean(), 2)
        mediana_val = round(subamostra[col].median(), 2)
    elif tipo == 'Qualitativa Ordinal':
        # Alinhado com a restrição teórica
        media_val = "N/A (Inadequado para Tipo Ordinal)"
        mediana_val = int(subamostra[col].median())
    else:  # Qualitativa Nominal
        media_val = "N/A (Inadequado para Tipo Nominal)"
        mediana_val = "N/A (Inadequado para Tipo Nominal)"

    registro_estatisticas.append({
        'Variável': col, 'Classificação': tipo,
        'Média': media_val, 'Mediana': mediana_val, 'Moda': moda_val
    })

df_estatisticas_final = pd.DataFrame(registro_estatisticas)
df_estatisticas_final.to_csv(os.path.join(
    pasta_estatisticas, "tendencia_central_ep2.csv"), index=False, encoding='utf-8')

# =================================================================
# PASSO 4: CODIFICAÇÃO E HEATMAP DE CORRELAÇÃO
# =================================================================
print("-> Transformando dados e gerando Heatmaps de Correlação...")
pasta_heatmap = os.path.join(pasta_mae, "4_mapas_de_correlacao")
os.makedirs(pasta_heatmap, exist_ok=True)

# 1. Fazemos uma cópia da subamostra para não alterar a original usada nas estatísticas
subamostra_cod = subamostra.copy()

# 2. Transformação Binária (Yes/No -> 1/0)
cols_binarias = ['Smoking', 'Family History of Anxiety',
                 'Dizziness', 'Medication', 'Recent Major Life Event']
for col in cols_binarias:
    subamostra_cod[col] = subamostra_cod[col].map({'Yes': 1, 'No': 0})

# 3. Filtrando apenas as colunas numéricas para a matriz de correlação
# (Deixamos Gênero e Ocupação de fora deste mapa geral para ele não ficar gigante e ilegível)
colunas_numericas = subamostra_cod.select_dtypes(
    include=['float64', 'int64']).columns
matriz_correlacao = subamostra_cod[colunas_numericas].corr()

# --- HEATMAP 1: O MAPA GERAL DOS FATORES FÍSICOS E DE HÁBITOS ---
plt.figure(figsize=(14, 10))
sns.heatmap(matriz_correlacao, annot=True, fmt=".2f", cmap='coolwarm',
            vmin=-1, vmax=1, linewidths=0.5)
plt.title('Heatmap de Correlação Geral (Fatores Numéricos e Binários)')
plt.xticks(rotation=45, ha='right')
plt.savefig(os.path.join(pasta_heatmap,
            "heatmap_geral_completo.png"), bbox_inches='tight')
plt.close()

# --- HEATMAP 2: FOCADO APENAS NA ANSIEDADE ---
# Aqui isolamos a correlação de todas as variáveis apenas contra a variável alvo
corr_ansiedade = matriz_correlacao[[
    'Anxiety Level (1-10)']].sort_values(by='Anxiety Level (1-10)', ascending=False)

plt.figure(figsize=(6, 8))
# Usamos annot=True para ver o número exato do peso de cada variável
sns.heatmap(corr_ansiedade, annot=True, fmt=".3f",
            cmap='coolwarm', vmin=-1, vmax=1, cbar=False)
plt.title('Força de Correlação com o Nível de Ansiedade')
plt.ylabel('Variáveis Preditivas')
plt.savefig(os.path.join(pasta_heatmap,
            "heatmap_foco_ansiedade.png"), bbox_inches='tight')
plt.close()

print("-> Gerando Heatmap específico para variáveis categóricas (One-Hot Encoding)...")

# Aplica o One-Hot Encoding prometido no EP1 apenas em Gender e Occupation
subamostra_categoricas = pd.get_dummies(subamostra[['Gender', 'Occupation', 'Anxiety Level (1-10)']],
                                        columns=['Gender', 'Occupation'])

# Calcula a correlação dessas novas subcategorias apenas contra a Ansiedade
corr_categoricas = subamostra_categoricas.corr(
)[['Anxiety Level (1-10)']].sort_values(by='Anxiety Level (1-10)', ascending=False)

# Remove a linha da própria ansiedade (já que a correlação dela com ela mesma é 1.0)
corr_categoricas = corr_categoricas.drop('Anxiety Level (1-10)')

# Geração do Gráfico
plt.figure(figsize=(6, 10))
sns.heatmap(corr_categoricas, annot=True, fmt=".3f",
            cmap='coolwarm', vmin=-1, vmax=1, cbar=False)
plt.title('Impacto do Gênero e Ocupação na Ansiedade')
plt.ylabel('Variáveis Nominais (One-Hot Encoded)')

# Salva na mesma pasta dos heatmaps
pasta_heatmap = os.path.join(
    "resultados_completos_ep2", "4_mapas_de_correlacao")
os.makedirs(pasta_heatmap, exist_ok=True)  # Garante que a pasta existe
plt.savefig(os.path.join(pasta_heatmap,
            "heatmap_foco_categoricas.png"), bbox_inches='tight')
plt.close()

print(
    f"\n[SUCESSO] Processamento concluído! Navegue até a pasta '{pasta_mae}' para extrair os arquivos.")
