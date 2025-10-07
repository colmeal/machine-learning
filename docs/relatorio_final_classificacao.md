# Projeto — Classificação com KNN e K-Means

**Base:** `docs/data/fitness_dataset.csv`  
**Alvo:** `is_fit` (0 = Not Fit, 1 = Fit)  
**Bibliotecas:** pandas · numpy · matplotlib · scikit-learn


---

## 1) Exploração dos Dados (EDA)

- Estatísticas descritivas salvas em: `data/eda_descritivas.csv`  

![Distribuição do alvo e proporção por gênero](img/eda_alvo_genero.png)


![Boxplots de variáveis-chave](img/eda_boxplots.png)


---

## 2) Aplicação das Técnicas

**KNN** (GridSearchCV) → **melhor**: `{"knn__n_neighbors": 17, "knn__p": 2, "knn__weights": "distance"}`  

**K-Means** ajustado com **K = 3** (hipótese de três perfis).

![Curva de Acurácia vs k (KNN)](img/acc_vs_k.png)


---

## 3) Matrizes de Confusão

**KNN (teste):**

![Matriz de confusão KNN](img/cm_knn.png)


**K-Means vs Classe real:**

![K-Means x is_fit](img/cm_kmeans.png)


---

## 4) Avaliação dos Modelos

**KNN (classe positiva = Fit = 1):**

- Accuracy: **0.7325**  
- Precision (1): **0.7054**  
- Recall (1): **0.5687**  
- F1 (1): **0.6298**  
- F1 Macro: **0.7102**  
- Balanced Accuracy: **0.7052**  

![Curva ROC (AUC = 0.80)](img/roc_knn.png)


**K-Means (coerência externa):**

- ARI: **0.0557**  
- Homogeneity: **0.0692**  
- Completeness: **0.0432**  
- V-Measure: **0.0532**  


---

## 5) Comparação dos Resultados

| Algoritmo          | Métrica principal   |   Valor | Observação                                         |
|:-------------------|:--------------------|--------:|:---------------------------------------------------|
| KNN (sup.)         | Accuracy            |  0.7325 | Bom preditor direto de is_fit; AUC 0.80            |
| K-Means (não sup.) | ARI                 |  0.0557 | Agrupa perfis; coerência moderada (V-Measure 0.05) |


---

## 6) Documentação

- Código: `run_projeto_classificacao.py`  

- Figuras geradas em `docs/img/` e tabelas em `data/`  

- Este relatório foi gerado automaticamente.
