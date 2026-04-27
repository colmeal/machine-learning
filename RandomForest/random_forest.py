# ============================================================
# PROJETO: Classificação com Random Forest
# Dataset: fitness_dataset.csv
# Autor: (Seu Nome)
# ============================================================

# ------------------------------------------------------------
# 0. IMPORTS E CARREGAMENTO DO DATASET
# ------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

plt.style.use("seaborn-v0_8")

print("\n===== CARREGANDO O DATASET =====")

df = pd.read_csv(
    r"C:\Users\mateus.colmeal\Documents\GitHub\machine-learning\docs\data\fitness_dataset.csv",
    sep=","
)

df.columns = df.columns.str.replace("\ufeff", "").str.strip()

print(df.head())
print(df.info())


# ------------------------------------------------------------
# 1. EXPLORAÇÃO DOS DADOS (EDA)
# ------------------------------------------------------------
print("\n===== EXPLORAÇÃO DOS DADOS =====")

print("\nValores ausentes:")
print(df.isnull().sum())

print("\nDistribuição da variável is_fit:")
print(df["is_fit"].value_counts(normalize=True))

plt.figure(figsize=(10,4))
df["is_fit"].value_counts().plot(kind="bar", color=["skyblue", "salmon"])
plt.title("Distribuição de is_fit")
plt.xticks([0,1], ["Not Fit (0)", "Fit (1)"])
plt.ylabel("Contagem")
plt.show()


# ------------------------------------------------------------
# 2. PRÉ-PROCESSAMENTO
# ------------------------------------------------------------
print("\n===== PRÉ-PROCESSAMENTO =====")

# 2.1 Imputação de valores ausentes
df["sleep_hours"].fillna(df["sleep_hours"].median(), inplace=True)

# 2.2 Codificação de variáveis categóricas
df["smokes"] = df["smokes"].astype(str).map({
    "yes": 1, "no": 0, "1": 1, "0": 0
})

df["gender"] = df["gender"].map({"F": 0, "M": 1})

print("\nAmostra após pré-processamento:")
print(df.head())


# ------------------------------------------------------------
# 3. DIVISÃO TREINO/TESTE
# ------------------------------------------------------------
print("\n===== DIVIDINDO TREINO E TESTE =====")

X = df.drop("is_fit", axis=1)
y = df["is_fit"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print("Tamanho treino:", X_train.shape)
print("Tamanho teste :", X_test.shape)


# ------------------------------------------------------------
# 4. TREINAMENTO DO MODELO RANDOM FOREST
# ------------------------------------------------------------
print("\n===== TREINANDO MODELO RANDOM FOREST =====")

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)


# ------------------------------------------------------------
# 5. AVALIAÇÃO DO MODELO
# ------------------------------------------------------------
print("\n===== AVALIAÇÃO DO MODELO =====")

y_pred = rf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\nAcurácia: {acc:.4f}\n")

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("\nMatriz de Confusão:")
print(cm)

plt.figure(figsize=(5,4))
plt.imshow(cm, cmap="Blues")
plt.title("Matriz de Confusão")
plt.colorbar()
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.show()


# ------------------------------------------------------------
# 6. IMPORTÂNCIA DAS VARIÁVEIS
# ------------------------------------------------------------
print("\n===== IMPORTÂNCIA DAS VARIÁVEIS =====")

importances = rf.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]

print("\nImportância das variáveis (ordem decrescente):")
for i in indices:
    print(f"{feature_names[i]}: {importances[i]:.4f}")

plt.figure(figsize=(8,5))
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), feature_names[indices], rotation=45, ha="right")
plt.title("Importância das Features - Random Forest")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# FIM DO SCRIPT
# ------------------------------------------------------------
print("\n===== FINALIZADO COM SUCESSO =====")
