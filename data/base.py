import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Carregar os dados
df = pd.read_csv('docs/data/fitness_dataset.csv')


print("Primeiras 5 linhas do dataset:")
print(df.head())

print("\nInformações do dataset:")
print(df.info())

print("\nEstatísticas descritivas:")
print(df.describe(include='all'))

print("\nValores ausentes por coluna:")
print(df.isnull().sum())

import matplotlib.pyplot as plt

# Estatísticas descritivas básicas
desc_stats = df.describe(include="all")

# Valores ausentes
missing_values = df.isnull().sum()

# Distribuição da variável alvo (is_fit)
fit_distribution = df["is_fit"].value_counts(normalize=True)

# Distribuição por gênero e fitness
gender_fit_distribution = pd.crosstab(df["gender"], df["is_fit"], normalize="index")

# Visualizações
plt.figure(figsize=(12,5))

# Distribuição de is_fit
plt.subplot(1,2,1)
df["is_fit"].value_counts().plot(kind="bar", color=["skyblue", "salmon"])
plt.xticks([0,1], ["Not Fit (0)", "Fit (1)"], rotation=0)
plt.title("Distribuição de Fitness (is_fit)")
plt.ylabel("Contagem")

# Distribuição de gênero vs fitness
plt.subplot(1,2,2)
gender_fit_distribution.plot(kind="bar", stacked=True, ax=plt.gca(), colormap="coolwarm")
plt.title("Proporção de Fitness por Gênero")
plt.ylabel("Proporção")

plt.tight_layout()
plt.show()

desc_stats, missing_values, fit_distribution