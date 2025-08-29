import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# Carregar os dados
df = pd.read_csv('docs/data/fitness_dataset.csv')

df.replace('?', np.nan, inplace=True)
print("\nValores ausentes por coluna:\n", df.isnull().sum())

x = df.drop(columns=['is_fit'])
y = df['is_fit']

# Substituir valores ausentes pela moda em todas as colunas
for col in x.columns:
    x[col].fillna(x[col].mode()[0], inplace=True)

print("\nValores ausentes após preenchimento:\n", x.isnull().sum())


x = pd.get_dummies(x, columns=['sleep_hours'], drop_first=True)
print(x.head(10))

# Certifique-se de que a pasta 'data' existe
os.makedirs("data", exist_ok=True)

# Dividir os dados em conjuntos de treino e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
x_train.to_csv("data/dataset-X-train.csv", index=False)     
x_test.to_csv("data/dataset-X-test.csv", index=False)
y_train.to_frame(name="is_fit").to_csv("data/dataset-y-train.csv", index=False)
y_test.to_frame(name="is_fit").to_csv("data/dataset-y-test.csv", index=False)
print(df.head(10))
