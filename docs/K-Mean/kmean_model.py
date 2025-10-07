# kmeans_execucao.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

IMG_DIR = "docs/img"
DATA_DIR = "data"
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# 1) Carregar dados pré-processados
X_scaled = np.load(os.path.join(DATA_DIR, "X_scaled_kmeans.npy"))
X_original = pd.read_csv(os.path.join(DATA_DIR, "X_original_kmeans.csv"))
y = pd.read_csv(os.path.join(DATA_DIR, "y_is_fit.csv"))["is_fit"]

# 2) Rodar K-Means (K=3 conforme seu escopo)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

# 3) Projeção PCA 2D apenas para visualização
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_scaled)

plt.figure(figsize=(7,6))
plt.scatter(coords[:,0], coords[:,1], c=cluster_labels, s=18, cmap="viridis")
plt.title("K-Means (K=3) - PCA 2D")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.tight_layout()
pca_path = os.path.join(IMG_DIR, "kmeans_pca_2d.png")
plt.savefig(pca_path, dpi=120)
plt.close()

# 4) Perfis por cluster (médias nas escalas originais, fácil de interpretar)
out = X_original.copy()
out["cluster"] = cluster_labels
out["is_fit"] = y

numeric_cols_for_profile = [
    "age","height_cm","weight_kg","bmi",
    "heart_rate","blood_pressure","sleep_hours",
    "nutrition_quality","activity_index","smokes_bin"
]

cluster_profile = out.groupby("cluster")[numeric_cols_for_profile].mean().round(2)
fit_dist = pd.crosstab(out["cluster"], out["is_fit"], normalize="index").round(3)

# 5) Salvar resultados
cluster_profile_path = os.path.join(DATA_DIR, "kmeans_cluster_profile.csv")
fit_dist_path = os.path.join(DATA_DIR, "kmeans_cluster_fit_distribution.csv")
assignments_path = os.path.join(DATA_DIR, "kmeans_assignments.csv")

cluster_profile.to_csv(cluster_profile_path)
fit_dist.to_csv(fit_dist_path)
out.to_csv(assignments_path, index=False)

# 6) Prints de resumo
print("\n== Perfis médios por cluster ==")
print(cluster_profile)
print("\n== Proporção de is_fit por cluster ==")
print(fit_dist)

print("\nArquivos gerados:")
print(f" - {pca_path}")
print(f" - {cluster_profile_path}")
print(f" - {fit_dist_path}")
print(f" - {assignments_path}")
