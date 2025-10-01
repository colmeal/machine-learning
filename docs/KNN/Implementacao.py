# 1. Importar bibliotecas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2. Carregar dados
file_path = "docs/data/fitness_dataset.csv"
df = pd.read_csv(file_path, sep=None, engine="python")

# Converter coluna smokes para binário (se existir)
df["smokes"] = df["smokes"].astype(str).str.lower().str.strip()
df["smokes_bin"] = df["smokes"].replace({"yes": 1, "1": 1, "no": 0, "0": 0})
print(df.head(10))


# 3. Selecionar variáveis
features = ["age", "weight_kg", "activity_index"]  # você pode mudar
X = df[features].applymap(lambda v: str(v).replace(",", ".")).astype(float)
y = df["is_fit"]

# Para visualização em 2D, vamos usar só 2 features
X_plot = X[["age", "weight_kg"]]

# 4. Dividir treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_plot, y, test_size=0.2, random_state=42)

# Normalizar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Testar vários valores de k
k_values = range(1, 21)
acc_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    preds = knn.predict(X_test_scaled)
    acc_scores.append(accuracy_score(y_test, preds))

# Gráfico de acurácia vs k
plt.figure(figsize=(8,5))
plt.plot(k_values, acc_scores, marker='o')
plt.xlabel("Número de vizinhos (k)")
plt.ylabel("Acurácia no teste")
plt.title("Curva de Acurácia vs k")
plt.show()

# 6. Escolher melhor k
best_k = k_values[np.argmax(acc_scores)]
print(f"Melhor k encontrado: {best_k} com acurácia de {max(acc_scores):.2f}")

# 7. Avaliação final com o melhor k
knn_final = KNeighborsClassifier(n_neighbors=best_k)
knn_final.fit(X_train_scaled, y_train)
y_pred = knn_final.predict(X_test_scaled)

print("\nMatriz de confusão:")
print(confusion_matrix(y_test, y_pred))

print("\nRelatório de classificação:")
print(classification_report(y_test, y_pred))

# 8. Fronteira de decisão (apenas para 2D)
h = 0.02
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = knn_final.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10,8))
plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.3)
sns.scatterplot(x=X_train_scaled[:, 0], y=X_train_scaled[:, 1], hue=y_train, style=y_train, s=80, palette="deep")
plt.xlabel("Idade (padronizada)")
plt.ylabel("Peso (padronizado)")
plt.title(f"KNN - Fronteira de decisão (k={best_k})")
plt.show()
