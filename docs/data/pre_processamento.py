import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder


# criando o diretorio de imagens para o mkdocs
IMG_DIR = "docs/img"
os.makedirs(IMG_DIR, exist_ok=True)


# Carregar os dados
df = pd.read_csv('docs/data/fitness_dataset.csv')
df.head(10)


# tratamento de nulos
df.replace('?', np.nan, inplace=True)
print("\nValores ausentes por coluna:\n", df.isnull().sum())

col = df.columns
for c in col:
    df[c].fillna(df[c].mode()[0], inplace=True)
print("\nValores ausentes após preenchimento:\n", df.isnull().sum())


# Distribuição de is_fit
plt.subplot(1,2,1)
df["is_fit"].value_counts().plot(kind="bar", color=["skyblue", "salmon"])
plt.xticks([0,1], ["Not Fit (0)", "Fit (1)"], rotation=0)
plt.title("Distribuição de Fitness (is_fit)")
plt.ylabel("Contagem")


# Distribuição de gênero vs fitness
gender_fit_distribution = pd.crosstab(df["gender"], df["is_fit"], normalize="index")

plt.subplot(1,2,2)
gender_fit_distribution.plot(kind="bar", stacked=True, ax=plt.gca(), colormap="coolwarm")
plt.title("Proporção de Fitness por Gênero")
plt.ylabel("Proporção")
plt.tight_layout()
plt.show()

# Divisão dos dados
x = df.drop(columns=['is_fit']) # variaveis features
y = df['is_fit'] # variavel target


# Dividir os dados em conjuntos de treino e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
x_train.to_csv("data/dataset-X-train.csv", index=False)     
x_test.to_csv("data/dataset-X-test.csv", index=False)
y_train.to_frame(name="is_fit").to_csv("data/dataset-y-train.csv", index=False)
y_test.to_frame(name="is_fit").to_csv("data/dataset-y-test.csv", index=False)
print(df.head(10))


encode = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
x_encoded_array = encode.fit_transform(x)
encoded_feature_names = encode.get_feature_names_out(x.columns)
#final df encoded
x = pd.DataFrame(x_encoded_array, columns=encoded_feature_names)
print(x.head(10))
print(x.columns)


# Converter coluna smokes para binário (se existir)
df["smokes"] = df["smokes"].astype(str).str.lower().str.strip()
df["smokes_bin"] = df["smokes"].replace({"yes": 1, "1": 1, "no": 0, "0": 0})
print(df.head(10))
