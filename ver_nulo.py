import pandas as pd

# Carregue o seu dataset (lembre-se de ajustar o caminho/nome do arquivo se necessário)
df = pd.read_csv('enhanced_anxiety_dataset.csv')

# Conta a quantidade de valores nulos em cada coluna
nulos_por_coluna = df.isnull().sum()

print("--- Quantidade de valores nulos por coluna ---")
print(nulos_por_coluna)

# Faz uma checagem geral no dataset inteiro
tem_nulo = df.isnull().values.any()

print("\n--- Checagem Geral ---")
if tem_nulo:
    print("Atenção: Existem valores nulos perdidos pelo dataset!")
else:
    print("Tudo limpo! Não há NENHUM valor nulo no dataset.")
