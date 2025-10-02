# Relatório do Projeto de Machine Learning – KNN

## 1. Exploração dos Dados

Nesta etapa, foi realizada uma análise inicial do conjunto de dados `fitness_dataset.csv`, contendo informações sobre hábitos de vida e condicionamento físico dos participantes, como idade, peso, nível de atividade física e hábito de fumar.  
A variável alvo `is_fit` indica se a pessoa está em boa condição física (1) ou não (0).

A análise revelou que há mais pessoas classificadas como "Not Fit" (0) do que "Fit" (1), evidenciando um leve desbalanceamento das classes.  
Esse desequilíbrio impacta diretamente o desempenho de modelos de classificação, pois eles tendem a ter maior acerto na classe majoritária.

Essas informações iniciais ajudam a entender o perfil dos dados e a importância de avaliar o modelo com métricas além da acurácia, como precisão, recall e F1-score.

---

## 2. Pré-processamento

As seguintes etapas foram aplicadas ao dataset:

- Conversão da coluna `smokes` para valores binários (`smokes_bin`), onde 0 = não fuma e 1 = fuma.
- Seleção apenas de variáveis numéricas para o modelo: `age`, `weight_kg`, `activity_index`, `smokes_bin`.
- Tratamento de valores ausentes com `SimpleImputer` (estratégia: mediana).
- Padronização dos dados com `StandardScaler`, para garantir que todas as variáveis estejam na mesma escala (média = 0, desvio = 1).

```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
```

---

## 3. Divisão dos Dados

O conjunto de dados foi dividido em 80% treino e 20% teste, de forma estratificada, preservando a proporção entre as classes `is_fit`.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
```

Essa divisão permite avaliar o desempenho do modelo em dados que ele nunca viu, evitando overfitting.

---

## 4. Treinamento do Modelo KNN

O algoritmo escolhido foi o K-Nearest Neighbors (KNN), que classifica um indivíduo de acordo com as classes dos seus vizinhos mais próximos no espaço de atributos.

`GridSearchCV` foi utilizado para testar diferentes valores de k, pesos (`uniform` e `distance`) e métricas de distância (`p=1` Manhattan, `p=2` Euclidiana).

O melhor modelo foi encontrado com `k = 14`, `weights = uniform` e `p = 1` (Manhattan).

**Exemplo de código de treinamento:**

```python
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])

param_grid = {
    "knn__n_neighbors": range(1,16),
    "knn__weights": ["uniform","distance"],
    "knn__p": [1,2]
}

gs = GridSearchCV(pipe, param_grid, cv=3, scoring="accuracy")
gs.fit(X_train, y_train)
best_model = gs.best_estimator_
```

---

## 5. Avaliação do Modelo

O desempenho foi avaliado no conjunto de teste utilizando acurácia, matriz de confusão, precisão, recall e F1-score.

![Matriz de Confusão](../img/MatrizFitorNot.png)

- **Acurácia:** ~72%
- **Matriz de Confusão:**  
  O modelo acerta 197 "Não Fit" corretamente, mas ainda classifica 76 "Fit" como "Não Fit".  
  Isso mostra que ele tem maior recall para a classe "Não Fit", mas menor capacidade de identificar corretamente os "Fit".

- **Curva de Acurácia vs k:**  
  Observa-se que a acurácia média cresce até valores próximos de k=14, que foi o melhor hiperparâmetro encontrado.
  ![Curva de Acurácia](Acuracia.png)

- **Fronteira de Decisão em PCA 2D:**  
  A projeção PCA em 2 dimensões mostra que as classes se sobrepõem bastante, justificando a dificuldade do KNN em separar completamente "Fit" de "Não Fit".
![Fronteira KNN](KNN.png)
---

## 6. Relatório Final

O modelo KNN alcançou 72% de acurácia no conjunto de teste, mostrando-se razoável para o problema, mas ainda limitado pela sobreposição entre classes e o leve desbalanceamento do dataset.

**Pontos fortes:**
- Fácil implementação e interpretação.
- Bom desempenho geral na classe majoritária ("Não Fit").

**Pontos fracos:**
- Tendência a classificar erroneamente indivíduos "Fit" como "Não Fit".
- Sensível ao desbalanceamento das classes e à presença de ruído.

**Possíveis Melhorias:**
- Balancear as classes (SMOTE ou undersampling).
- Feature engineering com variáveis adicionais (altura, IMC, qualidade do sono, nutrição).
- Comparar com outros modelos (Random Forest, Logistic Regression, SVM).
- Ajustar métrica de distância (Minkowski, Mahalanobis).
- Validação cruzada mais robusta (k-folds maiores).

Em resumo, o KNN demonstrou potencial para identificar padrões de condicionamento físico, mas ainda há espaço para otimizações para aumentar a capacidade de identificar corretamente os indivíduos classificados como Fit.

---