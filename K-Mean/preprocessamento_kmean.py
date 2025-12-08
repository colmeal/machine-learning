# preprocessamento_kmeans.py
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# pastas do projeto
IMG_DIR = "docs/img"
DATA_DIR = "data"
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# 1) Carregar dados
df = pd.read_csv("docs/data/fitness_dataset.csv")

# 2) Limpeza básica
df.replace("?", np.nan, inplace=True)

# padroniza 'smokes' e cria binário
df["smokes"] = df["smokes"].astype(str).str.lower().str.strip()
df["smokes_bin"] = df["smokes"].replace({"yes": 1, "1": 1, "no": 0, "0": 0, "nan": np.nan})

# 3) Feature engineering: BMI
df["bmi"] = (df["weight_kg"] / (df["height_cm"] / 100.0) ** 2).replace([np.inf, -np.inf], np.nan)

# 4) Define X e y (y só para avaliação posterior; NÃO entra no K-Means)
y = df["is_fit"].copy()
X = df[[
    "age","height_cm","weight_kg","bmi",
    "heart_rate","blood_pressure","sleep_hours",
    "nutrition_quality","activity_index","smokes_bin","gender"
]].copy()

# 5) Pipeline de pré-processamento
numeric_features = [
    "age","height_cm","weight_kg","bmi",
    "heart_rate","blood_pressure","sleep_hours",
    "nutrition_quality","activity_index","smokes_bin"
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

# 6) Ajusta o pré-processamento e transforma
X_scaled = preprocess.fit_transform(X)

# 7) Salva artefatos para a próxima etapa
# Para reuso simples, salvaremos X já transformado e também X original com colunas válidas
np.save(os.path.join(DATA_DIR, "X_scaled_kmeans.npy"), X_scaled)
X.to_csv(os.path.join(DATA_DIR, "X_original_kmeans.csv"), index=False)
y.to_frame(name="is_fit").to_csv(os.path.join(DATA_DIR, "y_is_fit.csv"), index=False)


try:
    feature_names_num = numeric_features
    feature_names_cat = list(preprocess.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(categorical_features))
    feature_names_all = feature_names_num + feature_names_cat
    pd.Series(feature_names_all).to_csv(os.path.join(DATA_DIR, "kmeans_feature_names.csv"), index=False, header=["feature"])
except Exception as e:
    print("Aviso: não foi possível exportar feature names (ok prosseguir).", e)

print("Pré-processamento concluído.")
print("Arquivos gerados em 'data/':")
print(" - X_scaled_kmeans.npy")
print(" - X_original_kmeans.csv")
print(" - y_is_fit.csv")
print(" - kmeans_feature_names.csv (opcional)")