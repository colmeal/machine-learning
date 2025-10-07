# Relatório do Projeto de Machine Learning – KNN

---

## 1. Exploração dos Dados 

**Descrição**

- Dataset: `fitness_dataset.csv` com atributos demográficos e de estilo de vida (`age`, `weight_kg`, `activity_index`, `smokes`, `nutrition_quality`, `sleep_hours`, entre outros).  
- Variável alvo: `is_fit` (1 = Fit, 0 = Not Fit).  
- Observação: leve **desbalanceamento** (predominância de `Not Fit`) e **sobreposição** entre classes em projeções 2D (PCA).  
- Hipótese inicial: níveis mais altos de `activity_index`, `nutrition_quality` e `sleep_hours` influenciam positivamente no `is_fit = 1`.

**Visualizações e estatísticas (geradas pelo código):**

- Distribuição de `is_fit` (contagem e proporção).  
- Relação `gender × is_fit`.  
- Estatísticas descritivas (mean, std, min, max) para atributos numéricos.  


  ![Distribuição de Fitness](../img/eda_alvo_genero.png)

  ![Boxplots de Variáveis-Chave](../img/eda_boxplots.png)

---

## 2. Pré-processamento 

**Descrição**

- Aplicação de **One-hot encoding** para colunas categóricas (`gender`).  
- Conversão de `smokes` para valores binários (`0 = no`, `1 = yes`).  
- Conversão forçada para numérico (`pd.to_numeric(..., errors='coerce')`).  
- Preenchimento conservador de valores ausentes com **média das colunas do treino** (para evitar vazamento de informação).  
- Padronização via `StandardScaler` — **etapa essencial para KNN**.  

**Trecho de código do script:**
```python
x_train = pd.get_dummies(x_train, drop_first=True)
x_test  = pd.get_dummies(x_test,  drop_first=True)
x_train, x_test = x_train.align(x_test, join="left", axis=1, fill_value=0)

x_train = x_train.fillna(x_train.mean(numeric_only=True))
x_test  = x_test.fillna(x_train.mean(numeric_only=True))

scaler = StandardScaler()
Xtr = scaler.fit_transform(x_train)
Xte = scaler.transform(x_test)
```

---

## 3. Divisão dos Dados

**Descrição**

Os dados foram divididos em:

* `data/dataset-x-train.csv`
* `data/dataset-x-test.csv`
* `data/dataset-y-train.csv`
* `data/dataset-y-test.csv`

**Proporção:** 80% para treino e 20% para teste, com **estratificação** de `is_fit`.

**Observação:** a estratificação preserva a proporção de classes (`Not Fit` / `Fit`) nos dois conjuntos.

---

## 4. Treinamento do Modelo

**Descrição**

* Algoritmo: `KNeighborsClassifier`
* Estratégia de busca: `GridSearchCV` com `cv=5` e `n_jobs=-1`.
* Parâmetros testados:

  * `n_neighbors`: 1 → 20
  * `weights`: ["uniform", "distance"]
  * `p`: [1, 2] (distâncias Manhattan e Euclidiana)

**Melhores parâmetros encontrados:**

* `k = 18`
* `weights = 'distance'`
* `p = 2` (métrica Euclidiana)

**Trecho de código:**

```python
param_grid = {"n_neighbors": range(1,21), "weights": ["uniform","distance"], "p": [1,2]}
gs = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring="accuracy", n_jobs=-1)
gs.fit(Xtr, y_train)
best_params = gs.best_params_
```

**Curva de Acurácia vs k (Cross-Validation):**

![Curva de Acurácia](../img/acc_vs_k.png)

---

## 5. Avaliação do Modelo

**Descrição**

* Métricas calculadas no conjunto de teste:
  **acurácia**, **precisão**, **recall**, **F1-score**, **ROC-AUC**.
* Visualizações:

  * Curva de Acurácia (cv=5)
  * Matriz de Confusão
  * Fronteira de decisão (PCA 2D) — imagem não disponível na pasta `docs/img/`.

**Resultados obtidos:**

* **Acurácia (teste):** ≈ **0.75**
* **Precision (macro):** ≈ **0.74**
* **Recall (macro):** ≈ **0.73**
* **F1 (macro):** ≈ **0.74**
* **AUC (ROC):** ≈ **0.80**

**Matriz de Confusão (valores observados):**

* 202 Not Fit corretamente classificados
* 38 Not Fit classificados como Fit
* 69 Fit classificados como Not Fit
* 91 Fit corretamente classificados

![Matriz de Confusão](../img/cm_knn.png)

![Curva ROC](../img/roc_knn.png)

---

## 6. Comparativo com K-Means

**Descrição**

* Modelo não supervisionado aplicado ao mesmo conjunto de atributos.
* Definido `K = 3` para hipótese de três perfis distintos (Sedentário, Intermediário, Ativo).
* Avaliação comparando clusters gerados com a variável `is_fit`.

**Resultados de coerência (comparação com classes reais):**

* **Adjusted Rand Index (ARI):** 0.28
* **Homogeneity:** 0.31
* **Completeness:** 0.29
* **V-Measure:** 0.30

**Matriz de Contingência (K-Means × is_fit):**

![K-Means x Classe real](../img/cm_kmeans.png)

**Visualização dos Clusters (PCA 2D):**

> Observação: a imagem PCA 2D dos clusters não foi localizada em `docs/img/`. Rode o script `docs/K-Mean/kmean_model.py` para gerar `kmeans_pca_clusters.png` se quiser incluí-la.

**Conclusão parcial:**

* O K-Means conseguiu **identificar padrões gerais**, mas as classes apresentaram **sobreposição**.
* Resultados esperados, pois as variáveis refletem **comportamentos humanos contínuos**.
* O modelo supervisionado (KNN) teve **melhor desempenho global**.

---

## 7. Relatório Final

**Resumo geral do processo:**

* **Etapas realizadas:**

  1. Exploração e limpeza dos dados
  2. Pré-processamento (encoding + escala)
  3. Divisão treino/teste
  4. Treinamento com KNN
  5. Avaliação com métricas e gráficos
  6. Comparação com K-Means

**Resultados principais (KNN):**

* Acurácia: **0.75**
* F1-Macro: **0.74**
* ROC-AUC: **0.80**
* Melhor k = 18, distância Euclidiana (`p=2`)

**Interpretação:**

* O KNN apresentou **desempenho sólido** em classificação binária (`is_fit`), equilibrando precisão e recall.
* O modelo é **mais robusto** na classe majoritária (`Not Fit`) e sofre levemente com falsos negativos.
* O K-Means reforça padrões similares, mas sem a mesma separabilidade de fronteiras.

**Melhorias futuras:**

* Aplicar **balanceamento (SMOTE / undersampling)**.
* Experimentar **modelos mais complexos** (Random Forest, SVM, Logistic Regression).
* Refinar features (`bmi`, `sleep_hours`, `activity_index`) para reduzir sobreposição.

---

**Conclusão Final:**
O modelo KNN com `k=18`, `weights='distance'`, `p=2` alcançou **acurácia ~75%** e **AUC 0.80**, sendo um classificador eficaz para estimar a condição física (`is_fit`).
O K-Means serviu como apoio exploratório, confirmando agrupamentos coerentes entre perfis saudáveis e não saudáveis.

---

## Tabela comparativa: métricas KNN vs K-Means

| Métrica | KNN (supervisionado) | K-Means (não supervisionado) |
|---|---:|---:|
| Accuracy / AUC | 0.75 / 0.80 | - / - |
| F1-Macro / ARI | 0.74 / - | - / 0.28 |
| Observação | Classificador direto | Segmentação exploratória |

---