

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix)
# === Configurações de diretórios/arquivos ===
DATA_DIR = "data"
IMG_DIR  = "docs/img"
os.makedirs(IMG_DIR, exist_ok=True)

X_TRAIN_PATH = os.path.join(DATA_DIR, "dataset-x-train.csv")
X_TEST_PATH  = os.path.join(DATA_DIR, "dataset-x-test.csv")
Y_TRAIN_PATH = os.path.join(DATA_DIR, "dataset-y-train.csv")
Y_TEST_PATH  = os.path.join(DATA_DIR, "dataset-y-test.csv")

# === 1) Carregar dados ===
x_train = pd.read_csv(X_TRAIN_PATH)
x_test  = pd.read_csv(X_TEST_PATH)

y_train = pd.read_csv(Y_TRAIN_PATH)["is_fit"]
y_test  = pd.read_csv(Y_TEST_PATH)["is_fit"]

# === 2) Seu pré-processamento: one-hot + align (sem alterar sua lógica) ===
categorical_cols = x_train.select_dtypes(include=['object', 'category']).columns
if len(categorical_cols) > 0:
    x_train = pd.get_dummies(x_train, columns=categorical_cols, drop_first=True)
    x_test  = pd.get_dummies(x_test,  columns=categorical_cols, drop_first=True)
    # Alinha colunas (preenche colunas faltantes com 0)
    x_train, x_test = x_train.align(x_test, join='left', axis=1, fill_value=0)

# Garante tipo numérico (caso sobre alguma coluna problem)
x_train = x_train.apply(pd.to_numeric, errors="coerce")
x_test  = x_test.apply(pd.to_numeric, errors="coerce")

# Se houver NaN residuais, substitui pela média DA COLUNA DO TREINO (conservador)
if x_train.isnull().any().any():
    x_train = x_train.fillna(x_train.mean(numeric_only=True))
if x_test.isnull().any().any():
    # usa médias do treino para não vazar info de teste
    x_test = x_test.fillna(x_train.mean(numeric_only=True))

# === 3) Padronização (essencial para KNN) ===
scaler = StandardScaler()
Xtr = scaler.fit_transform(x_train)
Xte = scaler.transform(x_test)

# === 4) GridSearchCV para KNN (em cima do X escalado) ===
param_grid = {
    "n_neighbors": list(range(1, 21)),
    "weights": ["uniform", "distance"],
    "p": [1, 2]  # 1 = Manhattan, 2 = Euclidiana
}
knn = KNeighborsClassifier()
gs = GridSearchCV(knn, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
gs.fit(Xtr, y_train)

best_knn = gs.best_estimator_
best_params = gs.best_params_
print("Melhores hiperparâmetros:", best_params)

# === 5) Avaliação no conjunto de teste ===
y_pred = best_knn.predict(Xte)
acc = accuracy_score(y_test, y_pred)
print(f"Acurácia (teste): {acc:.4f}")
print("\n== Classification Report (KNN) ==")
print(classification_report(y_test, y_pred, target_names=["Not Fit (0)","Fit (1)"], zero_division=0))

# === 6) GRÁFICO 1 – Curva de acurácia vs k (fixando weights e p ótimos) ===
fixed_weights = best_params["weights"]          
fixed_p       = best_params["p"]
k_values = list(range(1, 21))
cv_means = []

for k in k_values:
    knn_k = KNeighborsClassifier(n_neighbors=k, weights=fixed_weights, p=fixed_p)
    scores = cross_val_score(knn_k, Xtr, y_train, cv=5, scoring="accuracy", n_jobs=-1)
    cv_means.append(scores.mean())

plt.figure(figsize=(8,5))
plt.plot(k_values, cv_means, marker="o")
plt.xlabel("Número de vizinhos (k)")
plt.ylabel("Acurácia média (CV)")
plt.title("Curva de Acurácia vs k (KNN)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "Acuracia.png"), dpi=180)
plt.close()

# === 7) GRÁFICO 2 – Matriz de confusão (teste) ===
cm = confusion_matrix(y_test, y_pred, labels=[0,1])
plt.figure(figsize=(5,4))
plt.imshow(cm, interpolation='nearest')
plt.title("Matriz de Confusão (Teste)")
plt.colorbar()
plt.xticks([0,1], ["Not Fit (0)","Fit (1)"])
plt.yticks([0,1], ["Not Fit (0)","Fit (1)"])
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
plt.ylabel("Verdadeiro")
plt.xlabel("Predito")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "Matriz Fit or not.png"), dpi=180)
plt.close()

# === 8) GRÁFICO 3 – Fronteira de decisão em PCA 2D (visual) ===
# Projeção do espaço (já escalado) para 2 componentes
pca = PCA(n_components=2, random_state=42)
Xtr_pca = pca.fit_transform(Xtr)
Xte_pca = pca.transform(Xte)

# Treina KNN (melhores hiperparâmetros) no espaço PCA APENAS para plot
knn_pca = KNeighborsClassifier(
    n_neighbors=best_params["n_neighbors"],
    weights=best_params["weights"],
    p=best_params["p"]
)
knn_pca.fit(Xtr_pca, y_train)

# Gera a malha para a fronteira
h = 0.03
x_min, x_max = Xtr_pca[:, 0].min() - 0.5, Xtr_pca[:, 0].max() + 0.5
y_min, y_max = Xtr_pca[:, 1].min() - 0.5, Xtr_pca[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = knn_pca.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(9,7))
plt.contourf(xx, yy, Z, alpha=0.30)                      # superfície de decisão
plt.scatter(Xtr_pca[:,0], Xtr_pca[:,1], c=y_train, s=28, # pontos do treino (PCA)
            edgecolors="k", alpha=0.9)
plt.title(f"KNN – Fronteira (PCA 2D) | k={best_params['n_neighbors']}, "
          f"weights={best_params['weights']}, p={best_params['p']}")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "KNN.png"), dpi=180)
plt.close()

print("\nImagens salvas em:", IMG_DIR)
print(" - Acuracia.png")
print(" - Matriz Fit or not.png")
print(" - KNN.png")