# run_projeto_classificacao.py
# Projeto completo: EDA -> KNN -> K-Means -> Matrizes -> Métricas -> Comparação -> Relatório .md
# Pastas de saída compatíveis com MkDocs

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, adjusted_rand_score, homogeneity_score,
    completeness_score, v_measure_score
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# -----------------------------
# Configurações e caminhos
# -----------------------------
DATASET_PATH = "docs/data/fitness_dataset.csv"
IMG_DIR = "docs/img"
DATA_DIR = "data"
REPORT_PATH = "docs/relatorio_final_classificacao.md"
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# -----------------------------
# 1) Exploração dos Dados (EDA)
# -----------------------------
df = pd.read_csv(DATASET_PATH)
df.replace("?", np.nan, inplace=True)

# Ajustes básicos de tipos/valores
df["smokes"] = df["smokes"].astype(str).str.lower().str.strip()
df["smokes"] = df["smokes"].replace({"yes": 1, "1": 1, "no": 0, "0": 0, "nan": np.nan})
df["gender"] = df["gender"].astype(str).str.upper().str.strip()

# Cria BMI
df["bmi"] = (df["weight_kg"] / ((df["height_cm"]/100.0) ** 2)).replace([np.inf, -np.inf], np.nan)

# Estatísticas descritivas (numéricas)
desc = df.select_dtypes(include=["number"]).describe().T.round(2)
desc_path = os.path.join(DATA_DIR, "eda_descritivas.csv")
desc.to_csv(desc_path)

# Gráfico 1: distribuição de is_fit
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
(df["is_fit"].value_counts()
   .reindex([0,1]).fillna(0)
   .rename(index={0:"Not Fit (0)",1:"Fit (1)"})
   .plot(kind="bar", color=["skyblue","salmon"]))
plt.title("Distribuição de Fitness (is_fit)")
plt.ylabel("Contagem"); plt.xticks(rotation=0)

# Gráfico 2: proporção de fitness por gênero
plt.subplot(1,2,2)
gender_fit = pd.crosstab(df["gender"], df["is_fit"], normalize="index")
gender_fit.plot(kind="bar", stacked=True, ax=plt.gca())
plt.title("Proporção de is_fit por Gênero")
plt.ylabel("Proporção"); plt.legend(title="is_fit", labels=["0","1"])
plt.tight_layout()
fig_eda_path = os.path.join(IMG_DIR, "eda_alvo_genero.png")
plt.savefig(fig_eda_path, dpi=160); plt.close()

# Gráfico 3: boxplots rápidos de features-chave
features_box = ["bmi", "sleep_hours", "nutrition_quality", "activity_index", "heart_rate", "blood_pressure"]
plt.figure(figsize=(11,5))
df[features_box].boxplot(rot=45)
plt.title("Boxplots de Variáveis-Chave")
plt.tight_layout()
fig_box_path = os.path.join(IMG_DIR, "eda_boxplots.png")
plt.savefig(fig_box_path, dpi=160); plt.close()

# -----------------------------
# 2) Pré-processamento comum
# -----------------------------
y = df["is_fit"].astype(int)
X = df[[
    "age","height_cm","weight_kg","bmi",
    "heart_rate","blood_pressure","sleep_hours",
    "nutrition_quality","activity_index","smokes","gender"
]]

numeric_features = [
    "age","height_cm","weight_kg","bmi",
    "heart_rate","blood_pressure","sleep_hours",
    "nutrition_quality","activity_index","smokes"
]
categorical_features = ["gender"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(drop="if_binary", handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ],
    remainder="drop"
)

# -----------------------------
# Split para KNN (supervisionado)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Pipeline completo KNN (pré-processa + classifica)
knn_clf = Pipeline(steps=[
    ("prep", preprocess),
    ("knn", KNeighborsClassifier())
])

param_grid = {
    "knn__n_neighbors": list(range(1, 21)),
    "knn__weights": ["uniform", "distance"],
    "knn__p": [1, 2],  # Manhattan, Euclidiana
}

gs = GridSearchCV(knn_clf, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
gs.fit(X_train, y_train)
best_model = gs.best_estimator_
best_params = gs.best_params_

# Treino final e predições
y_pred = best_model.predict(X_test)
y_proba = (best_model.predict_proba(X_test)[:,1]
           if hasattr(best_model.named_steps["knn"], "predict_proba") else None)

# -----------------------------
# 3) Matrizes de Confusão (KNN e K-Means)
# -----------------------------
# KNN Confusion Matrix
cm_knn = confusion_matrix(y_test, y_pred, labels=[0,1])
plt.figure(figsize=(7,6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=["Not Fit (0)","Fit (1)"])
disp.plot(cmap="viridis", values_format='d')
plt.title("Matriz de Confusão – KNN (Teste)")
plt.tight_layout()
fig_cm_knn_path = os.path.join(IMG_DIR, "cm_knn.png")
plt.savefig(fig_cm_knn_path, dpi=160); plt.close()

# K-Means: treina sem rótulos no conjunto inteiro (X completo pré-processado)
X_all_pp = preprocess.fit_transform(X)  # reusa mesmos steps (fit completo para K-Means)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_all_pp)

# Matriz "confusão" K-Means vs is_fit (contingência)
cm_km = pd.crosstab(y, clusters)  # linhas = classe real, colunas = cluster
plt.figure(figsize=(7,6))
plt.imshow(cm_km.values, cmap="Purples")
plt.title("K-Means × Classe real (is_fit)")
plt.xlabel("Cluster"); plt.ylabel("Classe real")
for i in range(cm_km.shape[0]):
    for j in range(cm_km.shape[1]):
        plt.text(j, i, int(cm_km.values[i, j]), ha="center",
                 va="center", color="white" if cm_km.values[i,j] > cm_km.values.max()/2 else "black")
plt.colorbar()
plt.tight_layout()
fig_cm_km_path = os.path.join(IMG_DIR, "cm_kmeans.png")
plt.savefig(fig_cm_km_path, dpi=160); plt.close()

# -----------------------------
# 4) Avaliação dos Modelos (métricas)
# -----------------------------
# KNN: accuracy, precision, recall, f1 (classe positiva=1)
acc = accuracy_score(y_test, y_pred)
prec1 = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
rec1 = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
f1_1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
prec0 = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
rec0 = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
f1_0 = f1_score(y_test, y_pred, pos_label=0, zero_division=0)
f1_macro = (f1_0 + f1_1)/2
bal_acc = (rec0 + rec1)/2

# KNN: curva ROC/AUC
if y_proba is not None:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(9,7))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0,1],[0,1],"r--",label="Random Classifier")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.legend()
    plt.tight_layout()
    fig_roc_path = os.path.join(IMG_DIR, "roc_knn.png")
    plt.savefig(fig_roc_path, dpi=160); plt.close()
else:
    roc_auc = np.nan
    fig_roc_path = None

# KNN: curva de validação (acurácia x k) com weights/p ótimos
fixed_weights = best_params["knn__weights"]
fixed_p = best_params["knn__p"]
k_vals = list(range(1, 21))
cv_scores = []
for k in k_vals:
    model_k = Pipeline(steps=[
        ("prep", preprocess),
        ("knn", KNeighborsClassifier(n_neighbors=k, weights=fixed_weights, p=fixed_p))
    ])
    scores = cross_val_score(model_k, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)
    cv_scores.append(scores.mean())
plt.figure(figsize=(12,5))
plt.plot(k_vals, cv_scores, marker="o")
plt.title("Curva de Acurácia vs k (KNN)")
plt.xlabel("Número de vizinhos (k)")
plt.ylabel("Acurácia média (CV)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
fig_acc_k_path = os.path.join(IMG_DIR, "acc_vs_k.png")
plt.savefig(fig_acc_k_path, dpi=160); plt.close()

# K-Means: métricas de coerência (externas)
ari = adjusted_rand_score(y, clusters)
hom = homogeneity_score(y, clusters)
comp = completeness_score(y, clusters)
vms = v_measure_score(y, clusters)

# Salvar métricas em CSV
knn_metrics = pd.DataFrame({
    "metric": ["accuracy","precision_fit(1)","recall_fit(1)","f1_fit(1)",
               "precision_notfit(0)","recall_notfit(0)","f1_notfit(0)",
               "f1_macro","balanced_accuracy","roc_auc"],
    "value": [acc,prec1,rec1,f1_1,prec0,rec0,f1_0,f1_macro,bal_acc,roc_auc]
})
knn_metrics_path = os.path.join(DATA_DIR, "knn_test_metrics.csv")
knn_metrics.to_csv(knn_metrics_path, index=False)

km_metrics = pd.DataFrame({
    "metric": ["ARI","Homogeneity","Completeness","V-Measure"],
    "value": [ari, hom, comp, vms]
})
km_metrics_path = os.path.join(DATA_DIR, "kmeans_metrics.csv")
km_metrics.to_csv(km_metrics_path, index=False)

# Guardar cm_km (contingência) e perfis de cluster
cm_km_path = os.path.join(DATA_DIR, "kmeans_crosstab_cluster_isfit.csv")
cm_km.to_csv(cm_km_path)

# -----------------------------
# 5) Comparação/tabulação final
# -----------------------------
comparacao = pd.DataFrame({
    "Algoritmo": ["KNN (sup.)","K-Means (não sup.)"],
    "Métrica principal": ["Accuracy", "ARI"],
    "Valor": [round(acc,4), round(ari,4)],
    "Observação": [
        "Bom preditor direto de is_fit; AUC {:.2f}".format(roc_auc) if not np.isnan(roc_auc) else "Bom preditor direto de is_fit",
        "Agrupa perfis; coerência moderada (V-Measure {:.2f})".format(vms)
    ]
})
comparacao_path = os.path.join(DATA_DIR, "comparacao_knn_kmeans.csv")
comparacao.to_csv(comparacao_path, index=False)

# -----------------------------
# 6) Gerar Relatório Markdown
# -----------------------------
def md_img(path, alt):
    rel = os.path.relpath(path, "docs")
    return f"![{alt}]({rel.replace(os.sep,'/')})"

md = []
md.append("# Projeto — Classificação com KNN e K-Means\n")
md.append(f"**Base:** `{DATASET_PATH}`  \n**Alvo:** `is_fit` (0 = Not Fit, 1 = Fit)  \n**Bibliotecas:** pandas · numpy · matplotlib · scikit-learn\n")
md.append("\n---\n")
md.append("## 1) Exploração dos Dados (EDA)\n")
md.append(f"- Estatísticas descritivas salvas em: `data/{os.path.basename(desc_path)}`  \n")
md.append(md_img(fig_eda_path, "Distribuição do alvo e proporção por gênero"))
md.append("\n\n" + md_img(fig_box_path, "Boxplots de variáveis-chave") + "\n")
md.append("\n---\n")
md.append("## 2) Aplicação das Técnicas\n")
md.append(f"**KNN** (GridSearchCV) → **melhor**: `{json.dumps(best_params)}`  \n")
md.append("**K-Means** ajustado com **K = 3** (hipótese de três perfis).\n")
md.append(md_img(fig_acc_k_path, "Curva de Acurácia vs k (KNN)") + "\n")
md.append("\n---\n")
md.append("## 3) Matrizes de Confusão\n")
md.append("**KNN (teste):**\n\n" + md_img(fig_cm_knn_path, "Matriz de confusão KNN") + "\n\n")
md.append("**K-Means vs Classe real:**\n\n" + md_img(fig_cm_km_path, "K-Means x is_fit") + "\n")
md.append("\n---\n")
md.append("## 4) Avaliação dos Modelos\n")
md.append("**KNN (classe positiva = Fit = 1):**\n")
md.append(f"- Accuracy: **{acc:.4f}**  \n- Precision (1): **{prec1:.4f}**  \n- Recall (1): **{rec1:.4f}**  \n- F1 (1): **{f1_1:.4f}**  \n- F1 Macro: **{f1_macro:.4f}**  \n- Balanced Accuracy: **{bal_acc:.4f}**  \n")
if fig_roc_path:
    md.append(md_img(fig_roc_path, f"Curva ROC (AUC = {roc_auc:.2f})") + "\n")
md.append("\n**K-Means (coerência externa):**\n")
md.append(f"- ARI: **{ari:.4f}**  \n- Homogeneity: **{hom:.4f}**  \n- Completeness: **{comp:.4f}**  \n- V-Measure: **{vms:.4f}**  \n")
md.append("\n---\n")
md.append("## 5) Comparação dos Resultados\n")
md.append(comparacao.to_markdown(index=False) + "\n")
md.append("\n---\n")
md.append("## 6) Documentação\n")
md.append("- Código: `run_projeto_classificacao.py`  \n")
md.append("- Figuras geradas em `docs/img/` e tabelas em `data/`  \n")
md.append("- Este relatório foi gerado automaticamente.\n")

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(md))

# -----------------------------
# Prints finais
# -----------------------------
print("\n== CONCLUÍDO ==")
print("Melhores hiperparâmetros KNN:", best_params)
print(f"Acurácia (teste): {acc:.4f} | Precision(1): {prec1:.4f} | Recall(1): {rec1:.4f} | F1(1): {f1_1:.4f} | AUC: {roc_auc:.4f}")
print(f"K-Means: ARI={ari:.4f} | V-Measure={vms:.4f}")
print("\nArquivos gerados:")
print("  Relatório:", REPORT_PATH)
print("  EDA:", desc_path, fig_eda_path, fig_box_path)
print("  KNN/K-Means figuras:", fig_acc_k_path, fig_cm_knn_path, fig_cm_km_path, (fig_roc_path or "sem ROC"))
print("  Métricas:", knn_metrics_path, km_metrics_path, cm_km_path, comparacao_path)
