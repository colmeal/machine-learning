import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Carregar os dados
df = pd.read_csv('docs/data/restaurant_orders.csv')

print("Primeiras 5 linhas do dataset:")
print(df.head())

print("\nInformações do dataset:")
print(df.info())

print("\nEstatísticas descritivas:")
print(df.describe(include='all'))

print("\nValores ausentes por coluna:")
print(df.isnull().sum())

# Visualização das variáveis categóricas
plt.figure(figsize=(15, 10))


plt.subplot(2, 2, 2)
df['Customer Name'].value_counts().head(10).plot(kind='bar')
plt.title('Top 10 Clientes')
plt.xticks(rotation=45)

plt.subplot(2, 2, 3)
df['Food Item'].value_counts().plot(kind='bar')
plt.title('Distribuição de Itens do Menu')


plt.subplot(2, 2, 3)
df['Category'].value_counts().plot(kind='bar')
plt.title('Distribuição de Categorias')

plt.subplot(2, 2, 4)
df['Payment Method'].value_counts().plot(kind='bar')
plt.title('Distribuição de Métodos de Pagamento')

plt.tight_layout()
plt.savefig('./docs/data/restaurant_orders.png')
plt.show()



from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import numpy as np

# Copiar o dataframe original para preservar
df_processed = df.copy()

# Remover "Order ID" (não é útil para ML)
df_processed.drop(columns=["Order ID"], inplace=True)

# Converter "Order Time" para datetime
df_processed["Order Time"] = pd.to_datetime(df_processed["Order Time"])

# Criar novas features a partir de "Order Time"
df_processed["order_hour"] = df_processed["Order Time"].dt.hour
df_processed["order_dayofweek"] = df_processed["Order Time"].dt.dayofweek
df_processed["order_month"] = df_processed["Order Time"].dt.month

# Remover coluna original de data/hora
df_processed.drop(columns=["Order Time"], inplace=True)

# Label Encoding para colunas com muitos valores únicos
le_customer = LabelEncoder()
df_processed["Customer Name"] = le_customer.fit_transform(df_processed["Customer Name"])

le_food = LabelEncoder()
df_processed["Food Item"] = le_food.fit_transform(df_processed["Food Item"])

# One-Hot Encoding para colunas com poucas categorias
df_processed = pd.get_dummies(df_processed, columns=["Category", "Payment Method"], drop_first=True)

# Normalização das variáveis numéricas
scaler = StandardScaler()
df_processed[["Quantity", "Price"]] = scaler.fit_transform(df_processed[["Quantity", "Price"]])

# Exibir dataset processado
import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

df_processed.head()

from sklearn.model_selection import train_test_split

# Supondo que 'Price' seja a variável alvo (modifique se necessário)
X = df_processed.drop(columns=['Price'])
y = df_processed['Price']

# Divisão dos dados: 80% treino, 20% teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Tamanho do conjunto de treino: {X_train.shape}")
print(f"Tamanho do conjunto de teste: {X_test.shape}")

# Hipóteses para análise dos dados
print("\nHipóteses para análise dos dados:")
print("1. O preço do pedido depende do item do menu e da categoria.")
print("2. Certos clientes tendem a gastar mais do que outros.")
print("3. O método de pagamento influencia o valor do pedido.")
print("4. O horário do pedido (hora/dia da semana/mês) afeta o valor gasto.")
print("5. A quantidade comprada está relacionada ao tipo de item ou categoria.")
print("6. Existem padrões sazonais nas vendas (por mês ou dia da semana).")
print("7. Itens de determinadas categorias são mais populares em horários específicos.")

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Treinamento do modelo de árvore de decisão
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Previsão no conjunto de teste
y_pred = model.predict(X_test)

# Avaliação do modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nAvaliação do modelo Decision Tree:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R² Score: {r2:.4f}")

