import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# === 1. Carregar dados ===
file_path = "docs/data/fitness_dataset.csv"
df = pd.read_csv(file_path, sep=None, engine="python", decimal=",")

# Converter smokes em binário
df["smokes"] = df["smokes"].astype(str).str.lower().str.strip()
df["smokes_bin"] = df["smokes"].replace({"yes":1,"1":1,"sim":1,"true":1,"no":0,"0":0,"não":0,"nao":0,"false":0})

# Features e target
features = ["age", "weight_kg", "activity_index", "smokes_bin"]
X = df[features].apply(pd.to_numeric, errors="coerce")
y = df["is_fit"].astype(int)

# Split estratificado
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Pipeline KNN
pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                 ("scaler", StandardScaler()),
                 ("knn", KNeighborsClassifier())])

# GridSearch
param_grid = {"knn__n_neighbors": range(1,16),
              "knn__weights": ["uniform","distance"],
              "knn__p": [1,2]}

gs = GridSearchCV(pipe, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
gs.fit(X_train, y_train)
best_model = gs.best_estimator_

# Avaliação
y_pred = best_model.predict(X_test)
print("Melhores parâmetros:", gs.best_params_)
print("Acurácia no teste:", accuracy_score(y_test, y_pred))
print("\nRelatório de classificação:\n", classification_report(y_test, y_pred))

# === Gráfico 1: Acurácia vs k ===
best_weights = gs.best_params_["knn__weights"]
best_p = gs.best_params_["knn__p"]

k_values = range(1,16)
cv_scores = []
for k in k_values:
    pipe_k = Pipeline([("imputer", SimpleImputer(strategy="median")),
                       ("scaler", StandardScaler()),
                       ("knn", KNeighborsClassifier(n_neighbors=k, weights=best_weights, p=best_p))])
    scores = cross_val_score(pipe_k, X_train, y_train, cv=3, scoring="accuracy")
    cv_scores.append(scores.mean())

plt.figure(figsize=(8,5))
plt.plot(k_values, cv_scores, marker="o")
plt.xlabel("Número de vizinhos (k)")
plt.ylabel("Acurácia média (CV)")
plt.title("Curva de Acurácia vs k")
plt.show()

# === Gráfico 2: Matriz de confusão ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
plt.imshow(cm, cmap="Blues")
plt.title("Matriz de Confusão")
plt.colorbar()
plt.xticks([0,1], ["Não Fit", "Fit"])
plt.yticks([0,1], ["Não Fit", "Fit"])
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j,i,cm[i,j],ha="center",va="center",color="white" if cm[i,j]>cm.max()/2 else "black")
plt.xlabel("Predito")
plt.ylabel("Verdadeiro")
plt.show()

# === Gráfico 3: Fronteira PCA 2D ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

knn_pca = KNeighborsClassifier(n_neighbors=gs.best_params_["knn__n_neighbors"],
                               weights=best_weights, p=best_p)
knn_pca.fit(X_pca, y)

h=0.05
x_min, x_max = X_pca[:,0].min()-1, X_pca[:,0].max()+1
y_min, y_max = X_pca[:,1].min()-1, X_pca[:,1].max()+1
xx,yy = np.meshgrid(np.arange(x_min,x_max,h), np.arange(y_min,y_max,h))
Z = knn_pca.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(8,6))
plt.contourf(xx,yy,Z,alpha=0.3)
plt.scatter(X_pca[:,0],X_pca[:,1],c=y,edgecolor="k")
plt.title("Fronteira de Decisão (PCA 2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# === Gráfico 3: Fronteira PCA 2D (com amostragem estratificada para visualização) ===
from collections import defaultdict

# 1) Pré-processa e projeta para PCA (treino/teste como antes)
imp = SimpleImputer(strategy="median")
scaler = StandardScaler()

X_train_imp = imp.fit_transform(X_train)
X_test_imp  = imp.transform(X_test)

X_train_s = scaler.fit_transform(X_train_imp)
X_test_s  = scaler.transform(X_test_imp)

pca = PCA(n_components=2, random_state=42)
X_train_pca = pca.fit_transform(X_train_s)
X_test_pca  = pca.transform(X_test_s)

# 2) Treina o KNN na projeção PCA usando os melhores hiperparâmetros do GridSearch
best_k = gs.best_params_["knn__n_neighbors"]
best_w = gs.best_params_["knn__weights"]
best_p = gs.best_params_["knn__p"]

knn_pca = KNeighborsClassifier(n_neighbors=best_k, weights=best_w, p=best_p)
knn_pca.fit(X_train_pca, y_train)

# 3) Amostragem estratificada dos pontos só para PLOT (modelo continua treinado com 100%)
def stratified_downsample(X2d, y_vec, max_por_classe=150, seed=42):
    rng = np.random.RandomState(seed)
    X_out, y_out = [], []
    idx_por_classe = defaultdict(list)
    # índices por classe
    for i, cls in enumerate(np.asarray(y_vec)):
        idx_por_classe[cls].append(i)
    # sorteia até max_por_classe por classe (ou todos, se tiver menos)
    for cls, idxs in idx_por_classe.items():
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        take = min(len(idxs), max_por_classe)
        sel = idxs[:take]
        X_out.append(X2d[sel])
        y_out.append(np.asarray(y_vec)[sel])
    return np.vstack(X_out), np.concatenate(y_out)

# ajuste aqui o tamanho da amostra por classe para “limpar” o gráfico
X_plot, y_plot = stratified_downsample(X_train_pca, y_train, max_por_classe=120)

# 4) Gera a malha para a fronteira de decisão (continua densa para ficar suave)
h = 0.03
x_min, x_max = X_train_pca[:, 0].min() - 0.5, X_train_pca[:, 0].max() + 0.5
y_min, y_max = X_train_pca[:, 1].min() - 0.5, X_train_pca[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = knn_pca.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# 5) Plot enxuto
plt.figure(figsize=(9,7))
plt.contourf(xx, yy, Z, alpha=0.30)          # fundo: fronteira
plt.scatter(X_plot[:,0], X_plot[:,1],        # pontos: amostra estratificada
            c=y_plot, s=28, edgecolors="k", alpha=0.9)
plt.title(f"KNN – Fronteira PCA 2D (k={best_k}, weights={best_w}, p={best_p})")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.tight_layout()
plt.show()